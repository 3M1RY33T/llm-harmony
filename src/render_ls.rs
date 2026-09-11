use crate::inventory::plan::RemovalPlan;
use crate::inventory::safety::{classify, Safety};
use crate::inventory::Inventory;
use crate::render::human_bytes;

/// Is this file one of a model's *builds*, or one of the tokenizers, configs
/// and READMEs sitting beside them? Counting every file reported 63 builds for
/// a model with five.
fn is_build(a: &crate::inventory::artifact::Artifact) -> bool {
    matches!(
        a.format,
        crate::inventory::artifact::Format::Gguf
            | crate::inventory::artifact::Format::Mlx
            | crate::inventory::artifact::Format::SafetensorsBf16
    )
}

/// How many builds a model really has.
///
/// Counting build-format *paths* over-counts, for the same reason summing their
/// bytes does: llama.cpp's pool symlinks into LM Studio's store, so one Q4
/// appears twice and a two-build model reports three. `FileKey` is what says
/// two paths are one allocation, so it is what this counts. A keyless artifact
/// counts on its own — not knowing whether it is shared is not a reason to drop
/// it from the tally.
fn distinct_builds(arts: &[crate::inventory::artifact::Artifact]) -> usize {
    let mut seen = std::collections::HashSet::new();
    arts.iter()
        .filter(|a| is_build(a))
        .filter(|a| match a.key {
            Some(k) => seen.insert(k),
            None => true,
        })
        .count()
}

/// One model's row, as both the table and the JSON document need it.
///
/// Factored out rather than duplicated: `total_unique_bytes` and `classify`
/// are the two figures a consumer of the JSON cannot re-derive from paths —
/// bytes because a pool symlink and its target are one allocation, safety
/// because it is a judgement about peers — so the document has to carry them,
/// and carrying a *second* implementation of them is how a table and a
/// document start disagreeing about the same disk.
pub struct ModelRow<'a> {
    pub identity: &'a crate::inventory::identity::ModelIdentity,
    pub artifacts: Vec<crate::inventory::artifact::Artifact>,
    pub builds: usize,
    pub bytes: u64,
    pub safety: &'static str,
}

/// Every model with at least one build, largest first — the reason anyone runs
/// this is to find space.
pub fn model_rows(inv: &Inventory) -> Vec<ModelRow<'_>> {
    let mut rows: Vec<ModelRow> = Vec::new();
    for id in &inv.identities {
        let arts: Vec<_> = inv
            .artifacts
            .iter()
            .filter(|a| id.artifacts.contains(&a.id))
            .cloned()
            .collect();
        if arts.is_empty() {
            continue;
        }
        let builds = distinct_builds(&arts);
        if builds == 0 {
            continue;
        }
        let bytes = crate::inventory::scan::total_unique_bytes(&arts);
        let worst = arts.iter().map(|a| classify(a, &arts)).max_by_key(|s| match s {
            Safety::Redundant { .. } => 0,
            Safety::Reproducible { .. } => 1,
            Safety::Irreplaceable { .. } => 2,
        });
        let safety = match worst {
            Some(Safety::Redundant { .. }) => "redundant",
            Some(Safety::Reproducible { .. }) => "reproducible",
            Some(Safety::Irreplaceable { .. }) => "IRREPLACEABLE",
            None => "-",
        };
        rows.push(ModelRow { identity: id, artifacts: arts, builds, bytes, safety });
    }
    rows.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    rows
}

/// `ls --json`: the columns the table computes, plus the artifacts behind them.
///
/// A hosting screen groups and filters by format and store, because those are
/// what decide which providers could serve a model; and it must know which
/// paths are links, because removing one reclaims nothing and a reclaimable
/// figure that counts them is wrong in the user's favour.
pub fn document(inv: &Inventory) -> serde_json::Value {
    let models: Vec<_> = model_rows(inv)
        .iter()
        .map(|r| {
            serde_json::json!({
                "canonical": r.identity.canonical,
                "builds": r.builds,
                "bytes": r.bytes,
                "safety": r.safety,
                "provenance": r.identity.provenance,
                "artifacts": r.artifacts.iter().map(|a| serde_json::json!({
                    "id": a.id,
                    "path": a.path.display().to_string(),
                    "store": a.store.as_str(),
                    "format": a.format,
                    "bits": a.bits,
                    "bytes": a.bytes,
                    "is_link": a.is_link,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::json!({
        "schema": crate::ledger::SCHEMA,
        "total_bytes": inv.total_bytes(),
        "artifact_count": inv.artifacts.len(),
        "models": models,
    })
}

pub fn render_ls(inv: &Inventory) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<46} {:>7} {:>9}  {}\n",
        "model", "builds", "on disk", "safety"
    ));

    let rows: Vec<(String, usize, u64, &'static str)> = model_rows(inv)
        .iter()
        .map(|r| (r.identity.canonical.clone(), r.builds, r.bytes, r.safety))
        .collect();

    for (name, builds, bytes, label) in &rows {
        let short: String = name.chars().take(46).collect();
        out.push_str(&format!(
            "{:<46} {:>7} {:>9}  {}\n",
            short,
            builds,
            human_bytes(*bytes),
            label
        ));
    }

    out.push_str(&"\u{2500}".repeat(80));
    out.push('\n');
    out.push_str(&format!(
        "{:<46} {:>7} {:>9}\n",
        "total",
        inv.artifacts.len(),
        human_bytes(inv.total_bytes())
    ));
    out.push_str("\nlineage is inferred from filenames; no conversion was recorded by llm-harmony\n");
    out
}

pub fn render_plan(plan: &RemovalPlan) -> String {
    let mut out = String::new();
    if !plan.refusals.is_empty() {
        // Group by reason: a repo has hundreds of files and one cause, and
        // printing each file buries the answer.
        use std::collections::BTreeMap;
        let mut by_reason: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for r in &plan.refusals {
            by_reason.entry(&r.reason).or_default().push(&r.id);
        }
        out.push_str("refusing:\n");
        for (reason, ids) in &by_reason {
            out.push_str(&format!("  {} \u{2014} {} artifact(s)\n", reason, ids.len()));
            for id in ids.iter().take(3) {
                out.push_str(&format!("      {id}\n"));
            }
            if ids.len() > 3 {
                out.push_str(&format!("      \u{2026} and {} more\n", ids.len() - 3));
            }
        }
        out.push_str("\nnothing planned.\n");
        return out;
    }
    out.push_str("would remove, in order:\n");
    for s in &plan.steps {
        let note = if s.is_link { "  (alias \u{2014} frees nothing)" } else { "" };
        out.push_str(&format!(
            "  {:>9}  {}{}\n",
            human_bytes(s.frees_bytes),
            s.path,
            note
        ));
    }
    out.push_str(&format!("\nreclaims {}\n", human_bytes(plan.reclaims_bytes)));
    out.push_str("dry run \u{2014} llm-harmony cannot delete\n");
    out
}

pub fn render_clean(plan: &crate::inventory::clean::CleanPlan) -> String {
    use std::collections::BTreeMap;

    let mut out = String::new();

    if plan.models.is_empty() && plan.skipped.is_empty() {
        out.push_str("nothing is redundant: every build is the best of its format\n");
        return out;
    }

    // `superseded` is the whole point of the report: a step that cannot name
    // what survives it is a step nobody should run.
    let mut keeps: BTreeMap<&str, &str> = BTreeMap::new();
    for m in plan.models.iter().chain(plan.skipped.iter()) {
        for (gone, better) in &m.superseded {
            keeps.insert(gone.as_str(), better.as_str());
        }
    }

    if !plan.models.is_empty() {
        out.push_str("would remove, in order:\n\n");
        for m in &plan.models {
            let short: String = m.model.chars().take(44).collect();
            out.push_str(&format!(
                "  {:<44} reclaims {:>9}\n",
                short,
                human_bytes(m.plan.reclaims_bytes)
            ));
            for s in &m.plan.steps {
                let note = if s.is_link {
                    "  (alias \u{2014} frees nothing)"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "      {:>9}  {}{}\n",
                    human_bytes(s.frees_bytes),
                    s.path,
                    note
                ));
                if let Some(better) = keeps.get(s.id.as_str()) {
                    out.push_str(&format!("                 superseded by {better}\n"));
                }
            }
            out.push('\n');
        }
    }

    if !plan.skipped.is_empty() {
        out.push_str("skipped \u{2014} a requirer poisons that model's whole plan:\n");
        for m in &plan.skipped {
            let short: String = m.model.chars().take(44).collect();
            out.push_str(&format!("  {short}\n"));
            let mut by_reason: BTreeMap<&str, usize> = BTreeMap::new();
            for r in &m.plan.refusals {
                *by_reason.entry(&r.reason).or_default() += 1;
            }
            for (reason, count) in by_reason.iter().take(3) {
                out.push_str(&format!("      {reason} ({count} artifact(s))\n"));
            }
            if by_reason.len() > 3 {
                out.push_str(&format!(
                    "      \u{2026} and {} more reason(s)\n",
                    by_reason.len() - 3
                ));
            }
        }
        out.push('\n');
    }

    if plan.models.is_empty() {
        out.push_str("nothing planned.\n");
        return out;
    }

    out.push_str(&"\u{2500}".repeat(80));
    out.push('\n');
    out.push_str(&format!(
        "reclaims {} across {} model(s), {} artifact(s)\n",
        human_bytes(plan.reclaims_bytes),
        plan.models.len(),
        plan.artifact_count()
    ));
    out.push_str("dry run \u{2014} llm-harmony cannot delete\n");
    out
}

pub fn render_duplicates(groups: &[crate::inventory::dupes::DuplicateGroup], min_bytes: u64) -> String {
    use crate::inventory::dupes::reclaimable_bytes;

    let mut out = String::new();
    if groups.is_empty() {
        out.push_str(&format!(
            "no duplicate allocations above {}: every copy on this disk is its own bytes\n",
            human_bytes(min_bytes)
        ));
        return out;
    }

    out.push_str("duplicate allocations:\n\n");
    for g in groups {
        out.push_str(&format!(
            "  {:>9}  x{}  reclaimable {}\n",
            human_bytes(g.bytes),
            g.members.len(),
            human_bytes(g.reclaimable_bytes)
        ));
        for m in &g.members {
            out.push_str(&format!("      {:<10} {}\n", m.store.as_str(), m.path));
        }
        out.push_str(&format!("      matched on {}\n\n", g.basis));
    }

    out.push_str(&"\u{2500}".repeat(80));
    out.push('\n');
    out.push_str(&format!(
        "reclaimable {} across {} group(s)\n",
        human_bytes(reclaimable_bytes(groups)),
        groups.len()
    ));
    out.push_str(&format!(
        "report only \u{2014} copies below {} are not compared, and replacing one with a link is not something this slice does\n",
        human_bytes(min_bytes)
    ));
    out
}
