use crate::inventory::plan::RemovalPlan;
use crate::inventory::safety::{classify, Safety};
use crate::inventory::Inventory;
use crate::render::human_bytes;

pub fn render_ls(inv: &Inventory) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<46} {:>7} {:>9}  {}\n",
        "model", "builds", "on disk", "safety"
    ));

    let mut rows: Vec<(String, usize, u64, &'static str)> = Vec::new();
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
        // A repo holds tokenizers, configs and READMEs too. Only weight-
        // bearing files are "builds"; counting every file reported 63 builds
        // for a model with five.
        let builds = arts
            .iter()
            .filter(|a| {
                matches!(
                    a.format,
                    crate::inventory::artifact::Format::Gguf
                        | crate::inventory::artifact::Format::Mlx
                        | crate::inventory::artifact::Format::SafetensorsBf16
                )
            })
            .count();
        if builds == 0 {
            continue;
        }
        let bytes = crate::inventory::scan::total_unique_bytes(&arts);
        let worst = arts.iter().map(|a| classify(a, &arts)).max_by_key(|s| match s {
            Safety::Redundant { .. } => 0,
            Safety::Reproducible { .. } => 1,
            Safety::Irreplaceable { .. } => 2,
        });
        let label = match worst {
            Some(Safety::Redundant { .. }) => "redundant",
            Some(Safety::Reproducible { .. }) => "reproducible",
            Some(Safety::Irreplaceable { .. }) => "IRREPLACEABLE",
            None => "-",
        };
        rows.push((id.canonical.clone(), builds, bytes, label));
    }
    // Largest first: the reason anyone runs this is to find space.
    rows.sort_by(|a, b| b.2.cmp(&a.2));

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
