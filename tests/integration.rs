mod support;

use std::time::Duration;

use llm_harmony::config::{Config, ProviderConfig};
use llm_harmony::http::Http;
use llm_harmony::ledger::{Ledger, Outcome};
use llm_harmony::memory::{Machine, ProcessTree};
use llm_harmony::provider::{ProbeError, ProviderKind, State};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{name}")).expect("fixture exists")
}

/// A provider entry with no launch configuration -- the shape every test here
/// wants, since none of them start anything.
fn provider(kind: ProviderKind, url: String) -> ProviderConfig {
    ProviderConfig { kind, url, start: None, launchd_label: None }
}

fn tree(pids: Vec<u32>, bytes: Option<u64>) -> ProcessTree {
    ProcessTree {
        pids,
        footprint_bytes: bytes,
        phys_footprint_bytes: bytes,
        rss_bytes: bytes,
        processes: Vec::new(),
    }
}

fn machine() -> Machine {
    Machine {
        total_bytes: 25_769_803_776,
        used_bytes: 17_343_315_968,
        swap_total_bytes: 6_442_450_944,
        swap_used_bytes: 4_749_852_672,
    }
}

#[test]
fn a_provider_that_is_down_is_a_row_not_an_error() {
    let cfg = Config {
        providers: vec![provider(ProviderKind::LlamaCpp, "http://127.0.0.1:1".to_string())],
    };
    let ledger =
        Ledger::assemble_with(&cfg, &Http::new(Duration::from_millis(500)), machine(), |_| None);

    assert_eq!(ledger.rows.len(), 1);
    assert!(matches!(
        ledger.rows[0].outcome,
        Outcome::Failed(ProbeError::NotListening)
    ));
}

#[test]
fn two_live_providers_are_both_assembled() {
    let lms_body = fixture("lmstudio/models-one-loaded.json");
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");

    let lms = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &lms_body,
    )]));
    let oll = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));

    let cfg = Config {
        providers: vec![
            provider(ProviderKind::LmStudio, lms.base_url()),
            provider(ProviderKind::Ollama, oll.base_url()),
        ],
    };
    let ledger = Ledger::assemble_with(
        &cfg,
        &Http::new(Duration::from_millis(1500)),
        machine(),
        |_port| Some(tree(vec![647], Some(9_800_000_000))),
    );

    assert_eq!(ledger.rows.len(), 2);
    assert_eq!(ledger.rows[0].kind, ProviderKind::LmStudio, "reporting order is stable");

    let loaded_total: usize = ledger
        .rows
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Ok(models) => Some(models.iter().filter(|m| m.state == State::Loaded).count()),
            Outcome::Failed(_) => None,
        })
        .sum();
    assert_eq!(loaded_total, 2, "one loaded in each provider");
}

#[test]
fn provider_footprints_sum_into_a_total() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let oll = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));

    let cfg = Config {
        providers: vec![provider(ProviderKind::Ollama, oll.base_url())],
    };
    let ledger = Ledger::assemble_with(
        &cfg,
        &Http::new(Duration::from_millis(1500)),
        machine(),
        |_port| Some(tree(vec![4140], Some(400_000_000))),
    );

    assert_eq!(ledger.total_footprint_bytes(), Some(400_000_000));
}

/// A provider whose pids cannot be read still lists its models.
#[test]
fn an_unreadable_footprint_does_not_hide_the_models() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let oll = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));

    let cfg = Config {
        providers: vec![provider(ProviderKind::Ollama, oll.base_url())],
    };
    let ledger = Ledger::assemble_with(
        &cfg,
        &Http::new(Duration::from_millis(1500)),
        machine(),
        |_port| Some(tree(vec![4140], None)),
    );

    assert_eq!(ledger.rows[0].footprint_bytes, None);
    assert!(matches!(&ledger.rows[0].outcome, Outcome::Ok(m) if !m.is_empty()));
}
