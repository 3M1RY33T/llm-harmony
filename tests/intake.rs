//! Task 8: pulling a built artifact from Hugging Face.
//!
//! The ordering is what these assert. Everything that can refuse a pull refuses
//! *before* a byte moves — provenance is read, both ledgers are checked, the
//! store is chosen — because a refusal at 97% is not a refusal, it is a waste.
//!
//! One test here is the inverse of what the plan first specified. The plan's
//! `a_repo_that_does_not_declare_its_base_model_is_refused_by_default` came
//! from `inventory.md` §5 and `architecture.md` §7, which both state provenance
//! as gating. That was **overridden on 2026-09-11** (`hosting-page-r27-plan.md`
//! §6): it warns and the pull proceeds, on the reasoning that a refusal is
//! worked around by downloading in a terminal where there is no check at all.
//! The test below therefore asserts the warning survives, which is now the
//! whole safety story.

mod support;

use llm_harmony::intake::download::{fetch, transfer_agent, Progress};
use llm_harmony::intake::provenance::{check, Finding};
use llm_harmony::intake::{hf, place};
use llm_harmony::inventory::artifact::{Format, Store};
use llm_harmony::provider::ProviderKind;
use support::tree::Tree;

fn json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap()
}

#[test]
fn a_repo_that_does_not_declare_its_base_model_is_pulled_with_a_warning() {
    // Was "refused by default" until the rule was changed to inform rather
    // than gate. What must not regress is the warning itself.
    let f = check(None, Some("qwen3-14b"));
    assert!(f.is_warning(), "the absence is surfaced");
    assert!(!matches!(f, Finding::Declared { .. }));
    assert!(f.message().contains("qwen3-14b"), "it names what was asked for");
}

#[test]
fn a_repo_declaring_a_different_model_names_both_so_a_person_can_judge() {
    let f = check(Some("Qwen/Qwen2.5-14B"), Some("qwen3-14b"));
    let m = f.message();
    assert!(m.contains("Qwen2.5-14B") && m.contains("qwen3-14b"), "{m}");
}

#[test]
fn a_repo_needing_conversion_is_reported_as_such_rather_than_refused_silently() {
    // bf16 is visible and honestly labelled: a user searching for a model
    // should see that a build exists and that harmony cannot yet use it.
    let repo = hf::repo_from_json(&json(
        r#"{"id":"x/y","siblings":[{"rfilename":"model.safetensors","size":10}]}"#,
    ))
    .unwrap();
    assert!(repo.needs_conversion());
    assert!(!repo.has_loadable_build());
}

#[test]
fn a_file_is_placed_in_the_store_for_the_provider_that_will_serve_it() {
    // Placement is the disk ledger's answer, not the user's guess: the same
    // GGUF belongs in different stores for different providers.
    assert_eq!(place::store_for(ProviderKind::LmStudio, Format::Gguf), Some(Store::LmStudio));
    assert_eq!(place::store_for(ProviderKind::LlamaCpp, Format::Gguf), Some(Store::LlamaCpp));
    let dir = place::directory_for(Store::LmStudio, "TheBloke/Qwen3-14B-GGUF").unwrap();
    assert!(dir.ends_with("TheBloke/Qwen3-14B-GGUF"), "{dir:?}");
}

#[test]
fn add_emits_one_json_document_per_progress_event() {
    // The caller is a UI polling a job. A percentage printed over itself is
    // not readable by one; a document per line is.
    let events = [
        Progress::Started { file: "m.gguf".into(), bytes_total: Some(10), target: "/s".into() },
        Progress::Advanced { bytes_done: 5, bytes_total: Some(10) },
        Progress::Finished { file: "m.gguf".into(), bytes_done: 10, path: "/s/m.gguf".into() },
    ];
    let stream: Vec<String> = events.iter().map(|e| serde_json::to_string(e).unwrap()).collect();
    for line in &stream {
        assert!(!line.contains('\n'), "one document per LINE: {line}");
        serde_json::from_str::<serde_json::Value>(line).expect("each line parses alone");
    }
    let kinds: Vec<String> = stream
        .iter()
        .map(|l| json(l)["event"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(kinds, ["started", "advanced", "finished"]);
}

#[test]
fn an_aborted_download_leaves_nothing_a_provider_could_load() {
    // A server that promises ten bytes and delivers three. Without the
    // length check the rename would publish a truncated GGUF as a finished
    // one, and LM Studio would try to load it.
    let t = Tree::new("intake-abort");
    let dir = t.root.join("lmstudio/pub/repo");
    let s = support::StubServer::start_truncating("/m.gguf", "abc", 10);

    let mut seen = Vec::new();
    let err = fetch(
        &transfer_agent(),
        &format!("{}/m.gguf", s.base_url()),
        &dir,
        "m.gguf",
        &mut |p| seen.push(p),
    )
    .expect_err("a short read is a failed download, not a small model");

    // Which layer catches it is not the contract — ureq notices the peer
    // disconnect before the length check does, and on a clean short close the
    // length check is what fires. The contract is what is left on disk.
    assert!(!err.is_empty(), "a failure states a reason");
    assert!(!dir.join("m.gguf").exists(), "nothing a scanner would index");
    assert!(!dir.join("m.gguf.part").exists(), "and no leftover either");
    assert!(
        !seen.iter().any(|p| matches!(p, Progress::Finished { .. })),
        "and it never claimed to have finished"
    );
}

#[test]
fn a_completed_download_is_renamed_into_place_atomically() {
    let t = Tree::new("intake-complete");
    let dir = t.root.join("lmstudio/pub/repo");
    let s = support::StubServer::start_bytes("/m.gguf", "hello");

    let mut seen = Vec::new();
    let path = fetch(
        &transfer_agent(),
        &format!("{}/m.gguf", s.base_url()),
        &dir,
        "m.gguf",
        &mut |p| seen.push(p),
    )
    .expect("it completes");

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    assert!(!dir.join("m.gguf.part").exists(), "the .part is gone, not left beside it");
    assert!(matches!(seen.first(), Some(Progress::Started { .. })));
    assert!(matches!(seen.last(), Some(Progress::Finished { .. })));
}
