//! Human-readable output for `load`, `unload` and `switch`.

use crate::actuate::run::{Report, Status};
use crate::render::human_bytes;

pub fn render(report: &Report) -> String {
    let mut out = String::new();

    for u in &report.unloaded {
        out.push_str(&format!("  unloaded   {} on {}\n", u.model, u.provider));
    }

    match &report.outcome {
        Status::Ready { base_url, provider, provider_model_id, took_s, .. } => {
            out.push_str(&format!("  ready      {provider_model_id} on {provider}\n"));
            if !base_url.is_empty() {
                out.push_str(&format!("  endpoint   {base_url}\n"));
            }
            if *took_s > 0 {
                out.push_str(&format!("  took       {took_s}s\n"));
            }
        }
        Status::Refused { reason, pinned_blockers, alternatives } => {
            out.push_str(&format!("  refused    {reason}\n"));
            if !pinned_blockers.is_empty() {
                out.push_str(&format!("  pinned     {}\n", pinned_blockers.join(", ")));
            }
            for a in alternatives {
                out.push_str(&format!("  also at    {a}\n"));
            }
        }
        Status::Failed { reason, restore } => {
            out.push_str(&format!("  failed     {reason}\n"));
            // There is no rollback by design. This is how the operator puts
            // back what the failed switch already freed.
            if let Some(cmd) = restore {
                out.push_str(&format!("  restore    {cmd}\n"));
            }
        }
        Status::Aborted { reason } => {
            out.push_str(&format!("  aborted    {reason}\n"));
            out.push_str("             the load was undone; nothing new is resident\n");
        }
    }

    if let Some(bytes) = report.estimate.bytes {
        out.push_str(&format!(
            "  estimate   {:>9}   ({:?})\n",
            human_bytes(bytes),
            report.estimate.basis
        ));
    }
    out.push_str(&format!(
        "  machine    {} free \u{b7} swap {} used\n",
        human_bytes(report.free_bytes),
        human_bytes(report.swap_used_bytes)
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuate::plan::UnloadReason;
    use crate::actuate::run::{UnloadRecord, ACTUATE_SCHEMA};
    use crate::estimate::estimator::Estimate;
    use crate::provider::ProviderKind;

    fn report(outcome: Status, unloaded: Vec<UnloadRecord>) -> Report {
        Report {
            schema: ACTUATE_SCHEMA,
            verb: "switch",
            outcome,
            unloaded,
            estimate: Estimate::unknown(),
            free_bytes: 8_000_000_000,
            swap_used_bytes: 0,
        }
    }

    /// A refusal a pin caused must name the pin, or the operator reads "no
    /// room" on a half-empty machine with no way to find out why.
    #[test]
    fn a_refusal_names_the_pins_that_caused_it() {
        let out = render(&report(
            Status::Refused {
                reason: "needs 9.0G but only 2.0G can be made available".into(),
                pinned_blockers: vec!["qwen3-14b on lmstudio".into()],
                alternatives: Vec::new(),
            },
            Vec::new(),
        ));
        assert!(out.contains("pinned"), "{out}");
        assert!(out.contains("qwen3-14b on lmstudio"), "{out}");
    }

    /// No rollback by design, so the failure has to carry the way back.
    #[test]
    fn a_failure_prints_the_command_that_restores_what_was_freed() {
        let out = render(&report(
            Status::Failed {
                reason: "never became resident".into(),
                restore: Some("llm-harmony load qwen3-14b --provider lmstudio".into()),
            },
            vec![UnloadRecord {
                provider: ProviderKind::LmStudio,
                model: "qwen3-14b".into(),
                reason: UnloadReason::Named,
            }],
        ));
        assert!(out.contains("unloaded   qwen3-14b on lmstudio"), "{out}");
        assert!(out.contains("restore    llm-harmony load qwen3-14b"), "{out}");
    }

    /// An abort says the machine was left as it was found, because the whole
    /// point of aborting is that nothing was kept.
    #[test]
    fn an_abort_says_nothing_new_is_resident() {
        let out = render(&report(
            Status::Aborted { reason: "free memory fell to 1.0G during the load".into() },
            Vec::new(),
        ));
        assert!(out.contains("aborted"), "{out}");
        assert!(out.contains("nothing new is resident"), "{out}");
    }
}
