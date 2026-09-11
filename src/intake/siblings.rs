//! Another build of the same model, in a format this machine can serve.
//!
//! When a repo cannot be served as published, there are two ways out and they
//! are not close in cost. Converting means downloading the source, dequantising
//! it to full precision, requantising, and writing the result: for a 27B FP8
//! repo that is 15 GB down, ~55 GB of scratch, an hour, and two lossy steps —
//! to arrive at an artifact somebody has usually already built and published.
//! Thirty-four builds of the repo that prompted this module already exist.
//!
//! So the cheap route is offered first and the conversion second. `inventory.md`
//! §6 already records why the conversion is the expensive one: quantisation is
//! one-way, and there is no MLX ↔ GGUF path.
//!
//! **Every candidate is confirmed by reading its repo.** A name is not a
//! format: llmfit's catalog labels an FP8 repo `gguf`, and labels repos merely
//! *named* `MLX-…` gguf too. Offering a sibling on a name would reproduce the
//! bug this whole slice exists to fix, one layer out. A caller may hand in
//! candidates from anywhere — a catalog, a search, a guess — and this module
//! believes none of it without the repo's own declaration.

use serde::Serialize;

use crate::intake::hf;
use crate::inventory::artifact::Format;

/// A build that would serve, where the one asked for would not.
#[derive(Debug, Clone, Serialize)]
pub struct Sibling {
    pub repo: String,
    /// The build's own file, so a pull names what it priced.
    pub file: String,
    pub format: Format,
    pub bytes: u64,
    /// The `base_model` both repos declare. Carried so a row can show the
    /// ground on which these were called the same model, rather than asking
    /// anyone to take it on trust.
    pub base_model: String,
}

/// Siblings for a repo that will not run here, or nothing.
///
/// A source that already loads here has nothing to replace, and returning
/// alternatives for it would be answering a question nobody asked.
///
/// **"Loads here" is two questions, and this guard used to ask only one.**
/// Until 2026-09-11 it returned nothing whenever the source published a
/// servable *format*, which sent an MLX build of a 27B away empty-handed on a
/// machine 1.6 GiB too small to hold it -- the exact case where a smaller
/// build is the entire answer. A format this machine serves and a size this
/// machine can hold are different facts, so `too_big` is passed in by the
/// caller that priced it.
pub fn for_repo(
    source: &hf::Repo,
    candidates: &[hf::Repo],
    servable: &[Format],
    too_big: bool,
) -> Vec<Sibling> {
    let servable_here = source.files.iter().any(|f| servable.contains(&f.format));
    if servable_here && !too_big {
        return Vec::new();
    }
    from_candidates(source, candidates, servable)
}

/// The filter itself, over candidates the caller found however it liked.
pub fn from_candidates(
    source: &hf::Repo,
    candidates: &[hf::Repo],
    servable: &[Format],
) -> Vec<Sibling> {
    let mut out: Vec<Sibling> = Vec::new();
    for candidate in candidates {
        if candidate.id == source.id {
            continue;
        }
        let Some(base_model) = same_model(source, candidate) else {
            continue;
        };
        // Confirmed by what the repo publishes, never by what it is called.
        //
        // `run::builds` and not a pass over `files`: **a build is a set, not a
        // file.** `is_a_build` came down onto `RepoFile` so this module could
        // share it, and that still left this module reimplementing the other
        // half — the shard fold — which it got wrong in the same direction.
        // `edgefloor/Qwen-Qwen3.8-27B-MTPLX` publishes its model in three
        // parts, and "smallest servable file" offered one of them, 810 MB
        // standing in for 25 GB. Found 2026-09-11, the third time that shape
        // of bug surfaced in a day.
        let Some((build, bytes)) = crate::intake::run::builds(candidate)
            .into_iter()
            .filter(|b| b.files.iter().all(|f| servable.contains(&f.format)))
            // A size the repo never published cannot be admitted against
            // either ledger, so it is not the cheap way out of anything.
            .filter_map(|b| b.bytes.map(|n| (b, n)))
            .min_by_key(|(_, n)| *n)
        else {
            continue;
        };
        out.push(Sibling {
            repo: candidate.id.clone(),
            file: build.label(),
            format: build.files[0].format,
            bytes,
            base_model,
        });
    }
    // Smallest first: the reason anyone is reading this list is that the thing
    // they asked for would not fit.
    out.sort_by_key(|s| s.bytes);
    out
}

/// Are these two repos builds of one model? If so, on what ground.
///
/// `base_model` is a **one-step parent pointer, and repos disagree about which
/// step to point at.** Found 2026-09-11: `mconcat/…-FP8-Dynamic` declares its
/// immediate parent `Jackrong/Qwen3.5-27B-Claude-…-Distilled`, while a GGUF
/// build of that same distilled model declares the *root*,
/// `Qwen/Qwen3.5-27B`, skipping a generation. Comparing declarations to each
/// other found nothing, on a pair that plainly belong together.
///
/// So the comparison is against harmony's own model identity —
/// `canonical_name`, the function the disk ledger already uses to group
/// artifacts of one model across five stores. Three grounds, and all of them
/// anchor on a repo *id*, which is a thing that exists rather than a claim:
///
/// 1. the two ids canonicalise alike — two builds published under one name;
/// 2. the candidate IS the model the source derives from;
/// 3. the reverse.
///
/// **Deliberately not a fourth:** "both declare the same base model". Every
/// fine-tune of Qwen3.5-27B declares `Qwen/Qwen3.5-27B`, so that rule would
/// call two unrelated fine-tunes siblings — `inventory.md` §5's trap, arriving
/// through the declaration door instead of the name one.
fn same_model(source: &hf::Repo, candidate: &hf::Repo) -> Option<String> {
    use crate::inventory::identity::canonical_name;
    let source_id = canonical_name(&source.id);
    let candidate_id = canonical_name(&candidate.id);

    if source_id == candidate_id {
        return Some(source.base_model.clone().unwrap_or_else(|| source.id.clone()));
    }
    if let Some(base) = source.base_model.as_deref() {
        if canonical_name(base) == candidate_id {
            return Some(base.to_string());
        }
    }
    if let Some(base) = candidate.base_model.as_deref() {
        if canonical_name(base) == source_id {
            return Some(base.to_string());
        }
    }
    None
}
