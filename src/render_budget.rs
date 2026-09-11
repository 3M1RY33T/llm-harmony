//! `status --budgets`: what each provider is allowed to take.
//!
//! **There is no total row, and that is the feature.** The four ceilings are
//! in three units -- one in bytes, two counts, one of those counts unreadable,
//! and a fourth provider with no ceiling in either unit -- so a sum would be a
//! confident wrong number. What can be said instead is the thing that actually
//! decides whether this machine survives: how much of it is bounded in bytes
//! at all.

use crate::budget::{Budget, Ceiling};
use crate::memory::Machine;
use crate::provider::ProviderKind;
use crate::render::human_bytes;

/// The unit a provider expresses its ceiling in, whether or not the value
/// could be read.
///
/// Static provider knowledge rather than a reading: llama.cpp and Ollama cap a
/// *count* however many bytes those models are, vLLM-MLX caps bytes, and LM
/// Studio caps nothing. Printing the unit beside a `?` is what makes the `?`
/// legible -- "a count harmony could not read" is a different problem from "no
/// limit exists".
fn declared_unit(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::LlamaCpp | ProviderKind::Ollama => "count",
        ProviderKind::Vllm => "bytes",
        ProviderKind::LmStudio => "\u{2014}",
    }
}

fn source_cell(b: &Budget) -> String {
    match &b.ceiling {
        Ceiling::Unknown { why } => why.clone(),
        _ => b.source.clone(),
    }
}

pub fn render(budgets: &[Budget], machine: &Machine) -> String {
    let mut out = String::new();
    out.push_str(&format!("{:<11} {:<9} {:<7} {}\n", "provider", "allowed", "unit", "source"));
    for b in budgets {
        out.push_str(&format!(
            "{:<11} {:<9} {:<7} {}\n",
            b.provider.as_str(),
            b.ceiling.cell(),
            declared_unit(b.provider),
            source_cell(b),
        ));
    }
    out.push_str(&"\u{2500}".repeat(50));
    out.push('\n');
    out.push_str(&format!("{:<11} {} total\n", "machine", human_bytes(machine.total_bytes)));
    out.push_str(&verdict(budgets, machine));
    out
}

/// The one line worth reading.
///
/// Bytes do add, so where more than one provider is bounded in bytes their sum
/// is a real figure and worth stating. What is never stated is a total across
/// units.
fn verdict(budgets: &[Budget], machine: &Machine) -> String {
    let bounded: Vec<(&Budget, u64)> = budgets
        .iter()
        .filter_map(|b| match b.ceiling {
            Ceiling::Bytes(n) => Some((b, n)),
            _ => None,
        })
        .collect();
    let unbounded: Vec<&str> = budgets
        .iter()
        .filter(|b| !matches!(b.ceiling, Ceiling::Bytes(_)))
        .map(|b| b.provider.as_str())
        .collect();

    let share = |bytes: u64| -> String {
        if machine.total_bytes == 0 {
            return String::new();
        }
        format!(" \u{2014} {}% of the machine", bytes * 100 / machine.total_bytes)
    };

    let mut out = String::new();
    match bounded.len() {
        0 => out.push_str("nothing here is bounded in bytes at all.\n"),
        1 => {
            let (b, n) = bounded[0];
            out.push_str(&format!(
                "the only byte ceiling here is {}'s {}{}.\n",
                b.provider.as_str(),
                human_bytes(n),
                share(n)
            ));
        }
        _ => {
            let sum: u64 = bounded.iter().map(|(_, n)| n).sum();
            out.push_str(&format!(
                "{} byte ceilings sum to {}{}.\n",
                bounded.len(),
                human_bytes(sum),
                share(sum)
            ));
        }
    }
    if !unbounded.is_empty() {
        out.push_str(&format!("nothing bounds {} in bytes.\n", unbounded.join(", ")));
    }
    out
}

/// Its own schema, separate from the ledger's.
pub fn render_json(budgets: &[Budget], machine: &Machine) -> String {
    serde_json::json!({
        "schema": 1,
        "machine_total_bytes": machine.total_bytes,
        // Deliberately absent: any field that would total these. See the
        // module docs.
        "budgets": budgets,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind::{LlamaCpp, LmStudio, Ollama, Vllm};

    fn machine() -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 5 << 30,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        }
    }

    fn b(provider: ProviderKind, ceiling: Ceiling) -> Budget {
        Budget { provider, ceiling, source: "from the argv".into() }
    }

    /// What was measured on this machine 2026-09-11, rendered.
    fn live() -> Vec<Budget> {
        vec![
            b(LmStudio, Ceiling::Unbounded),
            b(LlamaCpp, Ceiling::Count(1)),
            b(Ollama, Ceiling::Unknown { why: "OLLAMA_MAX_LOADED_MODELS is unset".into() }),
            b(Vllm, Ceiling::Bytes(14 * 1024 * 1024 * 1024)),
        ]
    }

    /// The point of the command. A sum across bytes, counts and nothing would
    /// be a confident wrong number.
    #[test]
    fn there_is_no_total_row() {
        let out = render(&live(), &machine());
        assert!(!out.contains("total ceiling"), "{out}");
        // `machine 24.0G total` is the capacity, not a sum of ceilings.
        assert_eq!(out.matches("total").count(), 1, "{out}");
    }

    /// The fact that decides whether the machine survives.
    #[test]
    fn the_verdict_names_the_only_byte_ceiling_and_its_share() {
        let out = render(&live(), &machine());
        assert!(out.contains("the only byte ceiling here is vllm's 14.0G"), "{out}");
        assert!(out.contains("58% of the machine"), "{out}");
    }

    /// The other three are the README's opening screenshot: nothing stopped
    /// LM Studio from holding 8.9 G.
    #[test]
    fn the_verdict_names_what_is_not_bounded_in_bytes() {
        let out = render(&live(), &machine());
        let line = out.lines().find(|l| l.starts_with("nothing bounds")).expect("{out}");
        for p in ["lmstudio", "llamacpp", "ollama"] {
            assert!(line.contains(p), "{line}");
        }
    }

    /// `?` for what cannot be read -- the convention `status` already uses for
    /// `weights` and `gap` -- with the reason beside it.
    #[test]
    fn an_unreadable_ceiling_prints_a_question_mark_and_says_why() {
        let out = render(&live(), &machine());
        let line = out.lines().find(|l| l.starts_with("ollama")).expect("{out}");
        assert!(line.contains('?'), "{line}");
        assert!(line.contains("OLLAMA_MAX_LOADED_MODELS is unset"), "{line}");
    }

    /// The unit is printed even when the value is not, because "a count I
    /// could not read" and "no limit at all" are different facts.
    #[test]
    fn the_unit_is_printed_even_when_the_value_is_unknown() {
        let out = render(&live(), &machine());
        let ollama = out.lines().find(|l| l.starts_with("ollama")).unwrap();
        let lmstudio = out.lines().find(|l| l.starts_with("lmstudio")).unwrap();
        assert!(ollama.contains("count"), "{ollama}");
        assert!(lmstudio.contains("none"), "{lmstudio}");
    }

    /// Bytes do add. A total across units never does.
    #[test]
    fn two_byte_ceilings_are_summed_because_bytes_add() {
        let out = render(
            &[b(Vllm, Ceiling::Bytes(10 << 30)), b(LlamaCpp, Ceiling::Bytes(6 << 30))],
            &machine(),
        );
        assert!(out.contains("2 byte ceilings sum to 16.0G"), "{out}");
    }

    #[test]
    fn a_machine_with_no_byte_ceiling_anywhere_says_exactly_that() {
        let out = render(&[b(LmStudio, Ceiling::Unbounded), b(LlamaCpp, Ceiling::Count(1))], &machine());
        assert!(out.contains("nothing here is bounded in bytes at all"), "{out}");
    }

    #[test]
    fn the_json_carries_its_own_schema_and_no_total() {
        let v: serde_json::Value = serde_json::from_str(&render_json(&live(), &machine())).unwrap();
        assert_eq!(v["schema"], 1);
        assert_eq!(v["budgets"].as_array().unwrap().len(), 4);
        assert!(v.get("total_ceiling_bytes").is_none(), "there is no total");
    }
}
