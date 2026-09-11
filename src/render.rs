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

pub fn render_table(ledger: &Ledger, pins: &crate::pins::Pins) -> String {
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

    // Only when something is protected. A pin is the reason an admission can
    // be refused on a machine that looks half empty, so it has to be visible
    // here rather than only in the refusal that mentions it.
    if !pins.all().is_empty() {
        // A lease has to be distinguishable from a pin here, or the line
        // reports permanent protection for something that expires in an hour
        // -- the same invisibility `pins.rs` warns about, inverted.
        let now = crate::record::now_unix();
        let held: Vec<String> = pins.all().iter().map(|p| p.blame(now)).collect();
        let label = if pins.all().iter().any(|p| p.expires_at.is_some()) { "held" } else { "pinned" };
        out.push_str(&format!("{:<10} {}\n", label, held.join(" \u{b7} ")));
    }

    out
}

pub fn render_json(ledger: &Ledger, pins: &crate::pins::Pins) -> String {
    // `pinned` is added per model rather than as a separate list, so a
    // consumer reading a row never has to join two collections to know whether
    // it may be evicted. Additive: the document stays schema 1, which is what
    // Delroy's harmony.py checks before it will read anything at all.
    let mut doc = serde_json::to_value(ledger).expect("ledger is serialisable");
    if let Some(rows) = doc.get_mut("rows").and_then(|r| r.as_array_mut()) {
        for row in rows.iter_mut() {
            let kind = row.get("kind").and_then(|k| k.as_str()).unwrap_or("").to_string();
            let Ok(kind) = kind.parse::<crate::provider::ProviderKind>() else { continue };
            let Some(models) = row
                .get_mut("outcome")
                .and_then(|o| o.get_mut("detail"))
                .and_then(|d| d.as_array_mut())
            else {
                continue;
            };
            for model in models.iter_mut() {
                let id = model.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                if let Some(obj) = model.as_object_mut() {
                    obj.insert("pinned".into(), pins.is_pinned(kind, &id).into());
                }
            }
        }
    }
    // Beside the per-row flag, not instead of it: a row that can answer for
    // itself should, and a consumer reading one model never has to join two
    // collections. But a pin is deliberately sticky -- it survives an unload so
    // the next load is protected -- which makes "protected, not currently
    // loaded" a real state with no row to carry it. `render_table` could say
    // that and the document could not.
    if let Some(obj) = doc.as_object_mut() {
        obj.insert("pins".into(), serde_json::to_value(pins.all()).unwrap_or_default());
    }
    serde_json::to_string_pretty(&doc).expect("a document built from one")
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
            expires_at_unix: None,
            model_type: None,
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
        let out = render_table(&ledger(), &crate::pins::Pins::empty());
        let line = out.lines().find(|l| l.contains("llamacpp")).unwrap();
        assert!(line.contains("not running"), "got: {line}");
        assert!(!line.contains('G'), "a down provider must not show a size: {line}");
    }

    #[test]
    fn a_provider_without_published_weights_shows_a_question_mark() {
        let out = render_table(&ledger(), &crate::pins::Pins::empty());
        let line = out.lines().find(|l| l.contains("lmstudio")).unwrap();
        assert!(line.contains("9.1G"), "footprint is always known: {line}");
        assert!(line.contains('?'), "weights are unknown for LM Studio: {line}");
    }

    #[test]
    fn a_provider_with_weights_shows_the_gap() {
        let out = render_table(&ledger(), &crate::pins::Pins::empty());
        let line = out.lines().find(|l| l.contains("ollama")).unwrap();
        assert!(line.contains("261.6M"), "weights: {line}");
        assert!(line.contains("+119.9M"), "gap is footprint minus weights, signed: {line}");
    }

    #[test]
    fn the_machine_line_names_swap() {
        let out = render_table(&ledger(), &crate::pins::Pins::empty());
        assert!(out.contains("24.0G total"), "{out}");
        assert!(out.contains("swap"), "{out}");
    }

    #[test]
    fn json_is_valid_and_carries_errors() {
        let json = render_json(&ledger(), &crate::pins::Pins::empty());
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

    fn pinned(model: &str) -> crate::pins::Pins {
        let mut p = crate::pins::Pins::empty();
        p.add(crate::pins::Pin {
            provider: ProviderKind::LmStudio,
            model: model.into(),
            at: 0,
            note: None,
            owner: None,
            expires_at: None,
        });
        p
    }

    /// A pin is why an admission can be refused on a machine that looks half
    /// empty, so `status` has to show it without being asked.
    #[test]
    fn the_table_names_what_is_pinned() {
        let out = render_table(&ledger(), &pinned("qwen3-14b"));
        let line = out.lines().find(|l| l.starts_with("pinned")).expect("a pins line: {out}");
        assert!(line.contains("qwen3-14b on lmstudio"), "{line}");
    }

    /// And says nothing at all when nothing is protected -- a permanent empty
    /// row would train the eye to skip it.
    #[test]
    fn the_table_is_silent_when_nothing_is_pinned() {
        let out = render_table(&ledger(), &crate::pins::Pins::empty());
        assert!(!out.contains("pinned"), "{out}");
    }

    /// Per model, not as a separate list: a consumer reading a row must not
    /// have to join two collections to know whether it may be evicted.
    #[test]
    fn json_marks_the_pinned_model_and_only_that_one() {
        let json = render_json(&ledger(), &pinned("qwen3-14b"));
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["schema"], 1, "pins are additive; the schema must not move");
        assert_eq!(v["rows"][0]["outcome"]["detail"][0]["id"], "qwen3-14b");
        assert_eq!(v["rows"][0]["outcome"]["detail"][0]["pinned"], true);
        assert_eq!(
            v["rows"][1]["outcome"]["detail"][0]["pinned"], false,
            "the same pin must not protect another provider's model"
        );
    }

    /// A failed row has no models to mark, and must survive the pass anyway.
    #[test]
    fn json_leaves_a_failed_row_untouched() {
        let json = render_json(&ledger(), &pinned("qwen3-14b"));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["rows"][2]["outcome"]["status"], "failed");
    }
}
