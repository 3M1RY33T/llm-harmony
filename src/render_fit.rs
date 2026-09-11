//! `fit`: the assignment, what it is charged, and what was rejected.
//!
//! Three things the output has to make impossible to misread:
//!
//! - **what was charged and what was not.** A resident model and a running
//!   provider are already paid for, and a reader who cannot see that will
//!   think the arithmetic is wrong.
//! - **which figures rest on thin evidence.** vLLM-MLX's baseline has two
//!   readings behind it.
//! - **why not the other one.** The search is exhaustive precisely so the
//!   runner-up can be named with its price.

use crate::estimate::estimator::Basis;
use crate::fit::{Plan, Verdict};
use crate::render::human_bytes;

fn basis_label(b: Basis) -> &'static str {
    match b {
        Basis::Measured => "measured",
        Basis::Computed => "computed",
        Basis::Declared => "declared",
        Basis::Unknown => "unknown",
    }
}

pub fn render(p: &Plan) -> String {
    let mut out = String::new();

    match &p.verdict {
        Verdict::Unservable { model } => {
            return format!("  no provider serves `{model}`\n");
        }
        Verdict::Unpriced { model } => {
            out.push_str(&format!(
                "  `{model}` has no admissible figure, so this set has no price.\n"
            ));
            out.push_str("  a declared size is a weights-only floor and a set will not be\n");
            out.push_str("  admitted on one. load it once and the corpus will price it.\n");
            return out;
        }
        Verdict::CeilingRefusal { provider, allowed, wanted } => {
            out.push_str(&format!(
                "  {provider} allows {allowed} model{} at once; this set wants {wanted} there.\n",
                if *allowed == 1 { "" } else { "s" }
            ));
            out.push_str("  refused on the ceiling, not on the bytes \u{2014} no arithmetic over\n");
            out.push_str("  memory changes how many a provider will hold.\n");
            return out;
        }
        _ => {}
    }

    let width = p.assigned.iter().map(|a| a.model.chars().count()).max().unwrap_or(8).max(8);
    for a in &p.assigned {
        out.push_str(&format!(
            "  {:<width$}  {:<10} {:<38} {:>8}  {}\n",
            a.model,
            a.candidate.provider.as_str(),
            a.candidate.provider_model_id,
            if a.resident {
                "\u{2014}".to_string()
            } else {
                a.estimate.bytes.map(human_bytes).unwrap_or_else(|| "?".into())
            },
            if a.resident {
                "already resident, charged nothing".to_string()
            } else if a.estimate.basis == Basis::Measured {
                // A sample count means something only for a measurement.
                // Printing "computed (0)" reads as zero evidence when the
                // truth is that a computation takes none.
                format!("measured ({})", a.estimate.samples)
            } else {
                basis_label(a.estimate.basis).to_string()
            },
        ));
    }

    if !p.overheads.is_empty() {
        let parts: Vec<String> = p
            .overheads
            .iter()
            .map(|o| {
                format!(
                    "{} {}{}{}",
                    o.provider.as_str(),
                    human_bytes(o.bytes),
                    if o.thin { " (thin)" } else { "" },
                    if o.running { " (running, not charged)" } else { "" }
                )
            })
            .collect();
        out.push_str(&format!("  {:<width$}  {}\n", "baselines", parts.join(" \u{b7} ")));
    }

    out.push_str(&format!("  {:<width$}  {}\n", "charged", human_bytes(p.charged_bytes)));
    out.push_str(&format!("  {:<width$}  {}\n", "headroom", human_bytes(p.headroom_bytes)));

    match &p.verdict {
        Verdict::Fits => out.push_str("  fits\n"),
        Verdict::DoesNotFit { short_by_bytes, smallest_drop } => {
            out.push_str(&format!("  DOES NOT FIT by {}\n", human_bytes(*short_by_bytes)));
            if smallest_drop.is_empty() {
                out.push_str("  nothing in this set can be dropped to make it fit\n");
            } else {
                out.push_str(&format!("  smallest drop: {}\n", smallest_drop.join(", ")));
            }
        }
        _ => {}
    }

    if let Some((assignment, cost)) = &p.runner_up {
        let differs: Vec<String> = assignment
            .iter()
            .zip(&p.assigned)
            .filter(|(r, c)| r.candidate.provider != c.candidate.provider)
            .map(|(r, _)| format!("{} on {}", r.model, r.candidate.provider.as_str()))
            .collect();
        if !differs.is_empty() {
            let delta = cost.saturating_sub(p.charged_bytes);
            out.push_str(&format!(
                "  not chosen: {} \u{2014} {} dearer\n",
                differs.join(", "),
                human_bytes(delta)
            ));
        }
    }

    for p in &p.unchecked_ceilings {
        out.push_str(&format!(
            "  note: {}'s own limit could not be read, so it was not enforced\n",
            p.as_str()
        ));
    }
    if p.overheads.iter().any(|o| o.thin) {
        out.push_str("  note: a baseline above rests on too few idle readings to trust\n");
    }
    out
}

/// Its own schema, separate from the ledger's.
pub fn render_json(models: &[String], p: &Plan) -> String {
    let rows: Vec<serde_json::Value> = p
        .assigned
        .iter()
        .map(|a| {
            serde_json::json!({
                "model": a.model,
                "provider": a.candidate.provider.as_str(),
                "provider_model_id": a.candidate.provider_model_id,
                "resident": a.resident,
                "price_bytes": a.estimate.bytes,
                "basis": basis_label(a.estimate.basis),
                "charged_bytes": a.charged_bytes(),
            })
        })
        .collect();
    let overheads: Vec<serde_json::Value> = p
        .overheads
        .iter()
        .map(|o| {
            serde_json::json!({
                "provider": o.provider.as_str(),
                "baseline_bytes": o.bytes,
                "charged_bytes": o.charged_bytes(),
                "running": o.running,
                "thin": o.thin,
            })
        })
        .collect();
    let verdict = match &p.verdict {
        Verdict::Fits => serde_json::json!({"verdict": "fits"}),
        Verdict::Unpriced { model } => serde_json::json!({"verdict": "unpriced", "model": model}),
        Verdict::Unservable { model } => {
            serde_json::json!({"verdict": "unservable", "model": model})
        }
        Verdict::DoesNotFit { short_by_bytes, smallest_drop } => serde_json::json!({
            "verdict": "does-not-fit",
            "short_by_bytes": short_by_bytes,
            "smallest_drop": smallest_drop,
        }),
        Verdict::CeilingRefusal { provider, allowed, wanted } => serde_json::json!({
            "verdict": "ceiling-refusal",
            "provider": provider.as_str(),
            "allowed": allowed,
            "wanted": wanted,
        }),
    };
    serde_json::json!({
        "schema": 1,
        "models": models,
        "assigned": rows,
        "overheads": overheads,
        "charged_bytes": p.charged_bytes,
        "headroom_bytes": p.headroom_bytes,
        "decision": verdict,
        "unchecked_ceilings": p.unchecked_ceilings.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
    })
    .to_string()
}
