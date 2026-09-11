//! `search` and `add`, end to end.
//!
//! The order is the whole design: resolve the repo, check its provenance, price
//! it against both ledgers, choose its store — and only then move a byte. Every
//! refusal this can make, it makes before the download starts, because a
//! refusal at 97% is not a refusal, it is a waste.

use serde::Serialize;

use crate::http::Http;
use crate::intake::{fit, hf, place, provenance};
use crate::inventory::artifact::Format;
use crate::memory::Machine;
use crate::provider::ProviderKind;

pub const SCHEMA: u32 = 1;

/// One search result, already priced.
///
/// The fit is on the row rather than fetched on expand, and that is the feature
/// rather than a decoration on it: a list of repo names is a screen this page
/// exists to not be. Decided 2026-09-11 — `hosting-page-r27-plan.md` §6.
#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub repo: String,
    pub downloads: u64,
    pub likes: u64,
    pub files: Vec<hf::RepoFile>,
    pub needs_conversion: bool,
    pub provenance: provenance::Finding,
    /// `None` when the repo publishes no size for its best build, so nothing
    /// can be admitted. Not a zero-cost pull.
    pub fit: Option<fit::Fit>,
}

#[derive(Debug, Serialize)]
pub struct SearchDoc {
    pub schema: u32,
    pub query: String,
    pub hits: Vec<SearchHit>,
}

/// What `add --dry-run` answers, and what `add` reports when it finishes.
#[derive(Debug, Serialize)]
pub struct AddDoc {
    pub schema: u32,
    pub verb: &'static str,
    pub repo: String,
    /// The build's label. A single filename, or "first (N parts)" for a split
    /// one — a build is the set, and naming one shard would understate it.
    pub file: String,
    /// Every file the pull must fetch, in order. One entry unless sharded.
    pub files: Vec<String>,
    pub provider: String,
    pub store: Option<String>,
    pub target_dir: Option<String>,
    pub bytes: Option<u64>,
    pub provenance: provenance::Finding,
    /// Carried on the *result*, not only in the progress stream. A warning that
    /// scrolled past during an 8 GB download was never seen, and since
    /// 2026-09-11 this warning is the only thing between a search result and a
    /// wrong model on disk.
    pub warnings: Vec<String>,
    pub fit: Option<fit::Fit>,
    pub outcome: AddOutcome,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum AddOutcome {
    /// `--dry-run`: the whole verdict, nothing moved.
    Planned,
    Added { path: String },
    Refused { reason: String },
}

/// GGUF files in a model repo that are not the model.
///
/// Found running it, 2026-09-11: `unsloth/Qwen3.8-27B-GGUF` ships a 13 MB
/// `imatrix_unsloth.gguf` beside its multi-gigabyte builds, and "smallest
/// loadable" cheerfully chose the importance matrix — a calibration artefact
/// used to *produce* a quantisation, which no provider will serve. A projector
/// (`mmproj`) is the same class of thing: real, loadable alongside a model,
/// and not one.
const NOT_A_BUILD: [&str; 2] = ["imatrix", "mmproj"];

fn is_a_build(f: &hf::RepoFile) -> bool {
    if !matches!(f.format, Format::Gguf | Format::Mlx) {
        return false;
    }
    let lower = f.name.to_ascii_lowercase();
    !NOT_A_BUILD.iter().any(|marker| lower.contains(marker))
}

/// Split `…-00003-of-00007.gguf` into its stem and its shard count.
///
/// A large GGUF is published in numbered parts, and **each part is a file but
/// the build is the set**. Found running it 2026-09-11: asked for
/// `unsloth/DeepSeek-V3.1-GGUF`, "smallest loadable file" chose
/// `…-UD-Q3_K_XL-00007-of-00007.gguf` — 3.25 GB, which fit comfortably in
/// 161 GB free, so the two-ledger check passed and a pull started for one
/// seventh of a model. Nothing would ever have loaded it.
fn shard_of(name: &str) -> Option<(String, u32)> {
    // The separator is the literal `-of-`, which is the part a first attempt
    // missed: stripping two `-`-delimited fields from the end leaves `of` where
    // an index should be, `parse::<u32>()` fails, and every shard silently
    // looks like a whole build again.
    let stem = name.strip_suffix(".gguf")?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (stem, index) = head.rsplit_once('-')?;
    let total: u32 = total.parse().ok()?;
    let _index: u32 = index.parse().ok()?;
    if total < 2 {
        // `00001-of-00001` is a whole build that happens to be numbered.
        return None;
    }
    Some((stem.to_string(), total))
}

/// One thing a provider can load: a single file, or every shard of a split one.
#[derive(Debug, Clone)]
pub struct Build {
    /// Every file that must be pulled. One entry for an unsharded build.
    pub files: Vec<hf::RepoFile>,
    /// What the whole build costs. `None` when any part is unpriced — a build
    /// whose total is unknown cannot be admitted against either ledger.
    pub bytes: Option<u64>,
}

impl Build {
    /// What to call it. The first file for a single, the shared stem for a set.
    pub fn label(&self) -> String {
        match self.files.len() {
            0 => String::new(),
            1 => self.files[0].name.clone(),
            n => format!("{} ({n} parts)", self.files[0].name),
        }
    }
}

/// Every loadable build in a repo, shards folded into one entry each.
pub fn builds(repo: &hf::Repo) -> Vec<Build> {
    use std::collections::BTreeMap;
    let mut singles: Vec<Build> = Vec::new();
    let mut sharded: BTreeMap<String, Vec<hf::RepoFile>> = BTreeMap::new();

    for f in repo.files.iter().filter(|f| is_a_build(f)) {
        match shard_of(&f.name) {
            Some((stem, _total)) => sharded.entry(stem).or_default().push(f.clone()),
            None => singles.push(Build { files: vec![f.clone()], bytes: f.size_bytes }),
        }
    }
    for (_stem, mut files) in sharded {
        files.sort_by(|a, b| a.name.cmp(&b.name));
        // One missing size makes the whole total unknown. Summing what is
        // published and calling it the build's cost would understate it, which
        // is the direction that admits a pull that should not have been.
        let bytes = files
            .iter()
            .try_fold(0u64, |acc, f| f.size_bytes.map(|b| acc + b));
        singles.push(Build { files, bytes });
    }
    singles
}

/// The smallest thing a provider here can actually load, whole.
///
/// Smallest rather than best-quality, because a pull that fits is worth more to
/// someone browsing than one that does not, and the caller can always ask for a
/// specific `--file`.
pub fn best_build(repo: &hf::Repo) -> Option<Build> {
    builds(repo)
        .into_iter()
        .min_by_key(|b| b.bytes.unwrap_or(u64::MAX))
}

/// What a repo's build would cost resident, before it exists here.
///
/// **Always `Declared`**, and that is not a shortcut. The estimate ladder's
/// upper two rungs are both unavailable by definition at this point: nothing
/// has been *measured* because the model has never run here, and nothing can be
/// *computed* because computing reads layer count and KV geometry out of a GGUF
/// header or an MLX config — files that are still on Hugging Face.
///
/// So the published file size is the floor, and it is a floor rather than a
/// figure: the KV cache is exactly what it omits. `basis` carries that outward
/// so the page can badge it, which is the difference between a refusal a user
/// understands and one they meet cold.
fn price_build(
    machine: &Machine,
    build: &Build,
    target_dir: Option<&std::path::Path>,
) -> Option<fit::Fit> {
    let down = build.bytes?;
    let estimate = crate::estimate::estimator::Estimate {
        bytes: Some(down),
        basis: crate::estimate::estimator::Basis::Declared,
        samples: 0,
        spread_bytes: None,
    };
    let free_disk = target_dir.map(fit::free_disk_bytes).unwrap_or(0);
    // The loader's bar, not raw free memory: `render_estimate` owns the
    // number and every other admission path already uses it.
    Some(fit::admit(
        down,
        &estimate,
        machine,
        free_disk,
        crate::render_estimate::DEFAULT_RESERVE_BYTES,
    ))
}

pub fn search(http: &Http, machine: &Machine, query: &str, limit: usize) -> SearchDoc {
    let repos = hf::search(http, query, limit).unwrap_or_default();
    // Disk has to be priced against a real filesystem. Passing `None` reported
    // 0 bytes free and made every row read "will not fit", which is a lie about
    // a machine with 400 GB spare — found running it 2026-09-11. The default
    // provider's store is the honest stand-in before one is chosen.
    let browse_dir = place::store_for(ProviderKind::LmStudio, Format::Gguf)
        .and_then(|store| place::directory_for(store, ""));
    let hits = repos
        .into_iter()
        .map(|r| {
            // No expected model: the user typed a query, not an id. See
            // `provenance::check` — comparing against the repo's own name
            // warned on every correctly-labelled repo.
            let found = provenance::check(r.base_model.as_deref(), None);
            let fit = best_build(&r).and_then(|b| price_build(machine, &b, browse_dir.as_deref()));
            SearchHit {
                repo: r.id.clone(),
                downloads: r.downloads,
                likes: r.likes,
                needs_conversion: r.needs_conversion(),
                files: r.files.clone(),
                provenance: found,
                fit,
            }
        })
        .collect();
    SearchDoc { schema: SCHEMA, query: query.to_string(), hits }
}

/// Plan a pull. Everything that can refuse, refuses here.
pub fn plan_add(
    http: &Http,
    machine: &Machine,
    repo_id: &str,
    want_file: Option<&str>,
    provider: ProviderKind,
) -> AddDoc {
    let mut doc = AddDoc {
        schema: SCHEMA,
        verb: "add",
        repo: repo_id.to_string(),
        file: String::new(),
        files: Vec::new(),
        provider: provider.as_str().to_string(),
        store: None,
        target_dir: None,
        bytes: None,
        provenance: provenance::Finding::Undeclared { expected: repo_id.to_string() },
        warnings: Vec::new(),
        fit: None,
        outcome: AddOutcome::Refused { reason: String::new() },
    };

    let repo = match hf::repo(http, repo_id) {
        Ok(r) => r,
        Err(e) => {
            doc.outcome = AddOutcome::Refused { reason: format!("could not read `{repo_id}`: {e:?}") };
            return doc;
        }
    };

    // At `add` the user *has* named something — this repo — so a declared base
    // that canonicalises to something else is a real mismatch worth showing.
    doc.provenance = provenance::check(repo.base_model.as_deref(), Some(repo_id));
    if doc.provenance.is_warning() {
        // Warns, does not refuse — the decision recorded in the plan's §6.
        doc.warnings.push(doc.provenance.message());
    }

    let chosen: Option<Build> = match want_file {
        // An explicit `--file` names one file and means it: the caller has
        // looked. A shard asked for by name is still refused below, because
        // one seventh of a model is not a model however deliberately it was
        // requested.
        Some(name) => repo
            .files
            .iter()
            .find(|f| f.name == name)
            .map(|f| Build { files: vec![f.clone()], bytes: f.size_bytes }),
        None => best_build(&repo),
    };
    let Some(build) = chosen else {
        doc.outcome = AddOutcome::Refused {
            reason: if repo.needs_conversion() {
                // Shown and honestly labelled rather than hidden: a user should
                // learn a build exists and harmony cannot yet use it.
                format!(
                    "`{repo_id}` publishes only bf16 safetensors, which needs a conversion \
                     llm-harmony does not do yet"
                )
            } else {
                format!("`{repo_id}` publishes no GGUF or MLX build")
            },
        };
        return doc;
    };
    if build.files.len() == 1 {
        if let Some((_stem, total)) = shard_of(&build.files[0].name) {
            doc.file = build.files[0].name.clone();
            doc.outcome = AddOutcome::Refused {
                reason: format!(
                    "`{}` is one part of a {total}-part build; ask for the build rather than \
                     a shard, or nothing will be able to load it",
                    build.files[0].name
                ),
            };
            return doc;
        }
    }
    doc.file = build.label();
    doc.files = build.files.iter().map(|f| f.name.clone()).collect();
    doc.bytes = build.bytes;
    let first = &build.files[0];

    let Some(store) = place::store_for(provider, first.format) else {
        doc.outcome = AddOutcome::Refused {
            reason: format!(
                "{} cannot serve a {:?} file from a directory",
                provider.as_str(),
                first.format
            ),
        };
        return doc;
    };
    doc.store = Some(store.as_str().to_string());

    let Some(dir) = place::directory_for(store, repo_id) else {
        doc.outcome = AddOutcome::Refused {
            reason: format!("{}'s store is not present on this machine", store.as_str()),
        };
        return doc;
    };
    doc.target_dir = Some(dir.display().to_string());

    let fit = price_build(machine, &build, Some(&dir));
    doc.fit = fit.clone();
    match &fit {
        Some(f) if f.ok() => doc.outcome = AddOutcome::Planned,
        // Disk is durable; memory is not. Refusing a download because the
        // machine is busy *now* refuses on a reading that will be different in
        // a minute -- and `fit::Verdict::FitsDiskOnly` already says in so many
        // words that "the file is still worth having, and making room in
        // memory is a different act from making room on disk". So it warns and
        // proceeds, the same posture the provenance rule took on 2026-09-11
        // and for the same reason: a refusal a user works around by curling
        // the file in a terminal protects nothing.
        //
        // It matters more since the reserve was applied here: the bar is now
        // the loader's, which is correct and much narrower, and gating a pull
        // on it would refuse almost everything on a working machine.
        Some(f) if f.verdict == fit::Verdict::FitsDiskOnly => {
            doc.warnings.push(format!(
                "it will land, but nothing can load it as things stand: {}",
                f.message()
            ));
            doc.outcome = AddOutcome::Planned;
        }
        Some(f) => doc.outcome = AddOutcome::Refused { reason: f.message() },
        None => {
            doc.outcome = AddOutcome::Refused {
                reason: format!(
                    "`{}` publishes no size for every part, so this cannot be admitted \
                     against disk or memory",
                    build.label()
                ),
            }
        }
    }
    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(files: &[(&str, Option<u64>)]) -> hf::Repo {
        hf::Repo {
            id: "pub/model".into(),
            downloads: 0,
            likes: 0,
            base_model: None,
            files: files
                .iter()
                .map(|(n, s)| hf::RepoFile {
                    name: (*n).into(),
                    format: Format::from_path_str(n),
                    bits: crate::inventory::artifact::bits_from_name(n),
                    size_bytes: *s,
                })
                .collect(),
        }
    }

    /// Disk is durable and memory is not. A build that will land but cannot
    /// be loaded right now is worth having -- `fit::Verdict::FitsDiskOnly`
    /// says so -- so it proceeds carrying the reason rather than being
    /// refused on a reading that changes minute to minute.
    ///
    /// The refusals that remain are the durable ones: no disk, or a size the
    /// repo never published.
    #[test]
    fn a_build_that_lands_but_cannot_load_yet_is_planned_with_a_warning() {
        use crate::estimate::estimator::{Basis, Estimate};
        const GB: u64 = 1024 * 1024 * 1024;
        let m = Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 25_769_803_776 - 9 * GB,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        };
        let e = Estimate {
            bytes: Some(8 * GB),
            basis: Basis::Declared,
            samples: 0,
            spread_bytes: None,
        };
        // 9 GB free, an 8 GB reserve, an 8 GB model: plenty of disk, no
        // headroom.
        let f = fit::admit(8 * GB, &e, &m, 500 * GB, 8 * GB);
        assert_eq!(f.verdict, fit::Verdict::FitsDiskOnly);
        assert!(!f.ok(), "and it is still not a clean fit");
    }

    #[test]
    fn the_smallest_loadable_build_is_the_default() {
        let r = repo(&[("m.q8_0.gguf", Some(16384)), ("m.q4_k_m.gguf", Some(8192))]);
        assert_eq!(best_build(&r).unwrap().label(), "m.q4_k_m.gguf");
    }

    #[test]
    fn the_shard_marker_is_parsed_including_its_of() {
        assert_eq!(
            shard_of("M-UD-Q3_K_XL-00001-of-00003.gguf"),
            Some(("M-UD-Q3_K_XL".to_string(), 3))
        );
        assert_eq!(shard_of("M-Q8_0.gguf"), None);
        // One of one is a whole build that happens to be numbered.
        assert_eq!(shard_of("M-00001-of-00001.gguf"), None);
    }

    #[test]
    fn a_sharded_build_is_one_build_priced_at_its_total() {
        // Found running it: "smallest file" chose shard 7 of 7 of DeepSeek —
        // 3.25 GB, which fit, so a pull started for one seventh of a model.
        let r = repo(&[
            ("M-UD-Q3_K_XL-00001-of-00003.gguf", Some(1_000)),
            ("M-UD-Q3_K_XL-00002-of-00003.gguf", Some(1_000)),
            ("M-UD-Q3_K_XL-00003-of-00003.gguf", Some(1_000)),
            ("M-Q8_0.gguf", Some(2_500)),
        ]);
        let best = best_build(&r).unwrap();
        assert_eq!(best.label(), "M-Q8_0.gguf", "2.5k beats a 3k build, not a 1k shard");
        let sharded = builds(&r).into_iter().find(|b| b.files.len() == 3).unwrap();
        assert_eq!(sharded.bytes, Some(3_000), "the build costs the sum of its parts");
    }

    #[test]
    fn one_unpriced_shard_makes_the_whole_build_unpriced() {
        // Summing what is published would understate the total, and
        // understating is the direction that admits a pull it should not.
        let r = repo(&[
            ("M-00001-of-00002.gguf", Some(1_000)),
            ("M-00002-of-00002.gguf", None),
        ]);
        assert_eq!(builds(&r)[0].bytes, None);
    }

    #[test]
    fn a_bf16_file_is_never_the_default_because_nothing_here_loads_it() {
        let r = repo(&[("model.safetensors", Some(1)), ("m.q4_k_m.gguf", Some(9_000))]);
        assert_eq!(best_build(&r).unwrap().label(), "m.q4_k_m.gguf");
        assert!(best_build(&repo(&[("model.safetensors", Some(1))])).is_none());
    }

    #[test]
    fn an_importance_matrix_is_not_a_build() {
        // 13 MB beside a 9 GB model, and "smallest" chose it. An imatrix is
        // used to make a quantisation, not to serve one.
        let r = repo(&[("imatrix_unsloth.gguf", Some(13_000_000)), ("m.q4_k_m.gguf", Some(9_000_000_000))]);
        assert_eq!(best_build(&r).unwrap().label(), "m.q4_k_m.gguf");
    }

    #[test]
    fn a_projector_is_not_a_build_either() {
        let r = repo(&[("mmproj-model-f16.gguf", Some(600_000_000)), ("m.q4.gguf", Some(9_000_000_000))]);
        assert_eq!(best_build(&r).unwrap().label(), "m.q4.gguf");
    }

    #[test]
    fn an_unsized_build_sorts_last_rather_than_first() {
        // `unwrap_or(0)` here would make every unpriceable file the default.
        let r = repo(&[("a.gguf", None), ("b.gguf", Some(9_000))]);
        assert_eq!(best_build(&r).unwrap().label(), "b.gguf");
    }
}
