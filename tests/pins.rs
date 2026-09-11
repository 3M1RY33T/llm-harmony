//! A pin revokes harmony's permission to evict one model.
//!
//! The store is deliberately runtime state rather than config: the whole point
//! is to protect a model that is *already loaded*, which a file the user
//! hand-edits before starting anything cannot serve.

mod support;

use llm_harmony::pins::{Pin, Pins};
use llm_harmony::provider::ProviderKind;
use support::tree::Tree;

fn pin(model: &str) -> Pin {
    Pin {
        provider: ProviderKind::LmStudio,
        model: model.to_string(),
        at: 1_788_900_000,
        note: None,
        owner: None,
        expires_at: None,
    }
}

#[test]
fn a_pin_round_trips_through_the_file() {
    let t = Tree::new("pins-roundtrip");
    let path = t.root.join("pins.json");

    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    p.save_to(&path).unwrap();

    let back = Pins::load_from(&path);
    assert!(back.is_pinned(ProviderKind::LmStudio, "qwen3-14b"));
}

/// A pin is per (provider, model): the same name on another provider is a
/// different model -- a different artifact, a different quantisation, a
/// different footprint -- and pinning one must not protect the other.
#[test]
fn a_pin_does_not_leak_across_providers() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(!p.is_pinned(ProviderKind::Ollama, "qwen3-14b"));
}

/// Sticky: a pin outlives the residency it was created for, so an unload --
/// deliberate or otherwise -- does not silently unprotect the next load.
#[test]
fn a_pin_survives_being_unloaded_and_is_cleared_only_by_unpin() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(p.is_pinned(ProviderKind::LmStudio, "qwen3-14b"));
    assert!(p.remove(ProviderKind::LmStudio, "qwen3-14b"));
    assert!(!p.is_pinned(ProviderKind::LmStudio, "qwen3-14b"));
}

#[test]
fn adding_the_same_pin_twice_is_idempotent() {
    let mut p = Pins::empty();
    assert!(p.add(pin("m")));
    assert!(!p.add(pin("m")), "already pinned");
    assert_eq!(p.all().len(), 1);
}

#[test]
fn removing_a_pin_that_was_never_set_reports_it_rather_than_pretending() {
    let mut p = Pins::empty();
    assert!(!p.remove(ProviderKind::LmStudio, "never-pinned"));
}

/// Fail open, per the slice's global constraints: a corrupt pin file must not
/// stop harmony reading the ledger. It degrades to "nothing is pinned", which
/// is visible in `status` rather than silent.
#[test]
fn a_corrupt_pin_file_reads_as_no_pins_rather_than_an_error() {
    let t = Tree::new("pins-corrupt");
    let path = t.write("pins.json", "{ not json");
    assert!(Pins::load_from(&path).all().is_empty());
}

#[test]
fn a_missing_pin_file_is_no_pins_not_an_error() {
    assert!(Pins::load_from(std::path::Path::new("/nonexistent/pins.json")).all().is_empty());
}

/// The saved document carries its own schema, for the same reason every other
/// document here does: a consumer must be able to detect a shape change.
#[test]
fn the_saved_document_is_schema_1_and_keeps_the_note() {
    let t = Tree::new("pins-schema");
    let path = t.root.join("pins.json");

    let mut p = Pins::empty();
    p.add(Pin {
        provider: ProviderKind::Ollama,
        model: "qwen3:14b".into(),
        at: 42,
        note: Some("in use for the review".into()),
        owner: None,
        expires_at: None,
    });
    p.save_to(&path).unwrap();

    let raw = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["schema"], 1);
    assert_eq!(v["pins"][0]["provider"], "ollama");
    assert_eq!(v["pins"][0]["note"], "in use for the review");
}

/// An interrupted save must not leave a half-written store: write a sibling
/// temp file and rename, so the file is either the old one or the new one.
#[test]
fn saving_creates_the_directory_and_leaves_no_temp_file_behind() {
    let t = Tree::new("pins-atomic");
    let path = t.root.join("nested/deeper/pins.json");

    let mut p = Pins::empty();
    p.add(pin("m"));
    p.save_to(&path).unwrap();

    assert!(path.exists());
    let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp file left behind: {leftovers:?}");
}

// --- r27 Task 3: a pin on a model that is not resident ----------------------

/// A pin survives an unload -- that is the point of it being sticky, so the
/// next load is protected -- which makes "protected, not currently loaded" a
/// real state. `render_json` stamped `pinned` onto resident rows only, so it
/// was the one state the document could not express, while `render_table`
/// printed the held pins separately and could.
#[test]
fn status_json_lists_every_pin_including_ones_on_models_that_are_not_resident() {
    use llm_harmony::ledger::Ledger;

    let mut pins = Pins::empty();
    pins.add(pin("a-model-nothing-has-loaded"));

    // Nothing is resident, so there is no row for `pinned` to be stamped onto.
    let ledger = Ledger {
        schema_version: llm_harmony::ledger::SCHEMA,
        machine: llm_harmony::memory::Machine::read().expect("this machine reports its memory"),
        rows: Vec::new(),
    };
    let doc: serde_json::Value =
        serde_json::from_str(&llm_harmony::render::render_json(&ledger, &pins))
            .expect("a JSON document");

    let listed = doc["pins"].as_array().expect("a top-level pins list");
    assert_eq!(listed.len(), 1, "the pin is held whether or not it is loaded: {doc}");
    assert_eq!(listed[0]["model"], "a-model-nothing-has-loaded");
    assert_eq!(listed[0]["provider"], "lmstudio");
    assert!(listed[0]["at"].as_u64().is_some(), "when it was set: {doc}");
}
