use crate::ledger::{Ledger, Outcome};

/// Binary units. `hw.memsize` is 25,769,803,776, which is 24.0 GiB and 25.8 GB
/// -- and every document in this project, plus the machine's own spec sheet,
/// calls it a 24 GB machine. Rendering 25.8G would contradict all of them.
pub fn human_bytes(b: u64) -> String {
    const UNITS: [(u64, &str); 4] = [
        (1024 * 1024 * 1024 * 1024, "T"),
        (1024 * 1024 * 1024, "G"),
        (1024 * 1024, "M"),
        (1024, "K"),
    ];
    for (scale, suffix) in UNITS {
        if b >= scale {
            return format!("{:.1}{suffix}", b as f64 / scale as f64);
        }
    }
    format!("{b}B")
}

fn opt(v: Option<u64>) -> String {
    v.map(human_bytes).unwrap_or_else(|| "?".to_string())
}

pub fn render_table(ledger: &Ledger) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<10} {:>7} {:>11} {:>9} {:>8}\n",
        "provider", "loaded", "footprint", "weights", "gap"
    ));

    for row in &ledger.rows {
        match &row.outcome {
            Outcome::Failed(e) => {
                out.push_str(&format!("{:<10} {:>7} {}\n", row.kind.as_str(), "\u{2014}", e));
            }
            Outcome::Ok(_) => {
                let gap = row
                    .gap_bytes()
                    .map(|g| format!("+{}", human_bytes(g)))
                    .unwrap_or_else(|| "?".to_string());
                out.push_str(&format!(
                    "{:<10} {:>7} {:>11} {:>9} {:>8}\n",
                    row.kind.as_str(),
                    row.loaded().len(),
                    opt(row.footprint_bytes),
                    opt(row.weights_bytes()),
                    gap,
                ));
            }
        }
    }

    out.push_str(&"\u{2500}".repeat(50));
    out.push('\n');
    out.push_str(&format!(
        "{:<10} {:>7} {:>11}\n",
        "providers",
        "",
        opt(ledger.total_footprint_bytes())
    ));

    let m = &ledger.machine;
    out.push_str(&format!(
        "{:<10} {} total \u{b7} {:.0}% free \u{b7} swap {} used\n",
        "machine",
        human_bytes(m.total_bytes),
        m.free_fraction() * 100.0,
        human_bytes(m.swap_used_bytes),
    ));

    out
}

pub fn render_json(ledger: &Ledger) -> String {
    serde_json::to_string_pretty(ledger).expect("ledger is serialisable")
}

/// What each provider can actually do, for `llm-harmony verify`.
///
/// Exists because the answer changed twice under this project's feet: two
/// evict paths that were documented turned out not to exist. Printing what
/// was probed is cheaper than remembering which table is current.
pub fn render_verify(
    rows: &[(crate::provider::ProviderKind, String, bool, crate::provider::Actuation)],
) -> String {
    use crate::provider::Actuation;

    let mut out = String::new();
    for (kind, url, reachable, actuation) in rows {
        let reach = if *reachable { "reachable" } else { "not running" };
        let what = match actuation {
            Actuation::ModelLevel => "model-level".to_string(),
            Actuation::SelfManaged { ceiling } => format!("self-managed ({ceiling})"),
        };
        out.push_str(&format!("{:<10} {:<24} {:<12} {}\n", kind.as_str(), url, reach, what));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Ledger, Outcome, ProviderRow};
    use crate::memory::Machine;
    use crate::provider::{LoadedModel, ProbeError, ProviderKind, State};

    fn machine() -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 17_343_315_968,
            swap_total_bytes: 6_442_450_944,
            swap_used_bytes: 4_749_852_672,
        }
    }

    fn loaded(id: &str, weights: Option<u64>) -> LoadedModel {
        LoadedModel {
            id: id.to_string(),
            state: State::Loaded,
            context_tokens: Some(8192),
            weights_bytes: weights,
            artifact_path: None,
        }
    }

    fn ledger() -> Ledger {
        Ledger {
            schema_version: crate::ledger::SCHEMA,
            machine: machine(),
            rows: vec![
                ProviderRow {
                    kind: ProviderKind::LmStudio,
                    url: "http://127.0.0.1:1234".into(),
                    outcome: Outcome::Ok(vec![loaded("qwen3-14b", None)]),
                    pids: vec![647],
                    footprint_bytes: Some(9_800_000_000),
                    phys_footprint_bytes: Some(9_800_000_000),
                    rss_bytes: Some(9_800_000_000),
                    processes: Vec::new(),
                },
                ProviderRow {
                    kind: ProviderKind::Ollama,
                    url: "http://127.0.0.1:11434".into(),
                    outcome: Outcome::Ok(vec![loaded("nomic-embed-text:latest", Some(274_302_450))]),
                    pids: vec![4140],
                    footprint_bytes: Some(400_000_000),
                    phys_footprint_bytes: Some(400_000_000),
                    rss_bytes: Some(400_000_000),
                    processes: Vec::new(),
                },
                ProviderRow {
                    kind: ProviderKind::LlamaCpp,
                    url: "http://127.0.0.1:8080".into(),
                    outcome: Outcome::Failed(ProbeError::NotListening),
                    pids: vec![],
                    footprint_bytes: None,
                    phys_footprint_bytes: None,
                    rss_bytes: None,
                    processes: Vec::new(),
                },
            ],
        }
    }

    /// Binary units, so `hw.memsize` reads as the 24G everyone calls it.
    #[test]
    fn bytes_render_in_readable_units() {
        assert_eq!(human_bytes(0), "0B");
        assert_eq!(human_bytes(274_302_450), "261.6M");
        assert_eq!(human_bytes(9_800_000_000), "9.1G");
        assert_eq!(human_bytes(25_769_803_776), "24.0G");
    }

    #[test]
    fn a_down_provider_says_not_running_and_shows_no_numbers() {
        let out = render_table(&ledger());
        let line = out.lines().find(|l| l.contains("llamacpp")).unwrap();
        assert!(line.contains("not running"), "got: {line}");
        assert!(!line.contains('G'), "a down provider must not show a size: {line}");
    }

    #[test]
    fn a_provider_without_published_weights_shows_a_question_mark() {
        let out = render_table(&ledger());
        let line = out.lines().find(|l| l.contains("lmstudio")).unwrap();
        assert!(line.contains("9.1G"), "footprint is always known: {line}");
        assert!(line.contains('?'), "weights are unknown for LM Studio: {line}");
    }

    #[test]
    fn a_provider_with_weights_shows_the_gap() {
        let out = render_table(&ledger());
        let line = out.lines().find(|l| l.contains("ollama")).unwrap();
        assert!(line.contains("261.6M"), "weights: {line}");
        assert!(line.contains("+119.9M"), "gap is footprint minus weights, signed: {line}");
    }

    #[test]
    fn the_machine_line_names_swap() {
        let out = render_table(&ledger());
        assert!(out.contains("24.0G total"), "{out}");
        assert!(out.contains("swap"), "{out}");
    }

    #[test]
    fn json_is_valid_and_carries_errors() {
        let json = render_json(&ledger());
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["rows"].as_array().unwrap().len(), 3);
        assert_eq!(v["schema"], 1, "consumers must be able to detect a shape change");
        assert_eq!(v["rows"][2]["outcome"]["status"], "failed");
        assert_eq!(v["rows"][2]["outcome"]["detail"]["error"], "not-listening");
        assert_eq!(v["rows"][0]["outcome"]["status"], "ok");
        assert_eq!(v["machine"]["total_bytes"], 25_769_803_776u64);
    }

    #[test]
    fn verify_names_the_ceiling_of_a_self_managed_provider() {
        use crate::provider::{Actuation, ProviderKind};
        let out = render_verify(&[
            (ProviderKind::LmStudio, "http://127.0.0.1:1234".into(), true, Actuation::ModelLevel),
            (
                ProviderKind::LlamaCpp,
                "http://127.0.0.1:8080".into(),
                true,
                Actuation::SelfManaged { ceiling: "max_instances" },
            ),
        ]);
        assert!(out.contains("lmstudio"), "{out}");
        assert!(out.contains("model-level"), "{out}");
        assert!(out.contains("self-managed (max_instances)"), "a bare 'self-managed' tells the operator nothing actionable: {out}");
    }
}
