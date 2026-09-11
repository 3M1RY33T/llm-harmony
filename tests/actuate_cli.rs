//! The actuating verbs, end to end through the binary.
//!
//! Pointed at ports nothing is listening on, so these exercise the paths that
//! matter most in practice: the ones where something is not there.

mod support;

use std::process::Command;
use support::tree::Tree;

const BIN: &str = env!("CARGO_BIN_EXE_llm-harmony");

/// A config naming one provider on a port nothing is listening on.
fn dead_config(t: &Tree) -> String {
    t.write(
        "config.toml",
        r#"
        [[provider]]
        kind = "ollama"
        url = "http://127.0.0.1:1"
        "#,
    )
    .display()
    .to_string()
}

#[test]
fn load_with_no_provider_running_refuses_cleanly() {
    let t = Tree::new("cli-load-refuse");
    let out = Command::new(BIN)
        .args(["load", "some-model", "--config", &dead_config(&t)])
        .output()
        .expect("binary runs");

    assert_ne!(out.status.code(), Some(0), "a refusal is not a success");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("no provider serves"), "{text}");
}

/// The document carries its own schema, so `status` can stay at 1 and Delroy's
/// harmony.py keeps working untouched.
#[test]
fn the_actuate_document_carries_its_own_schema() {
    let t = Tree::new("cli-load-json");
    let out = Command::new(BIN)
        .args(["load", "some-model", "--json", "--config", &dead_config(&t)])
        .output()
        .expect("binary runs");

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout is a JSON document");
    assert_eq!(v["schema"], 1);
    assert_eq!(v["verb"], "load");
    assert_eq!(v["outcome"]["status"], "refused");
    assert!(v["unloaded"].as_array().unwrap().is_empty());
}

/// Unloading something that is not there is a clean failure, not a crash and
/// not a silent success.
#[test]
fn unloading_a_model_that_is_not_resident_says_so() {
    let t = Tree::new("cli-unload-absent");
    let out = Command::new(BIN)
        .args(["unload", "ghost", "--json", "--config", &dead_config(&t)])
        .output()
        .expect("binary runs");

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["verb"], "unload");
    assert_eq!(v["outcome"]["status"], "failed");
    assert!(
        v["outcome"]["reason"].as_str().unwrap().contains("not resident"),
        "{v}"
    );
}

/// `switch` reports the verb it was asked to perform, so a log of these
/// documents reads as a history of intentions rather than of side effects.
#[test]
fn switch_reports_itself_as_switch() {
    let t = Tree::new("cli-switch");
    let out = Command::new(BIN)
        .args(["switch", "a", "b", "--json", "--config", &dead_config(&t)])
        .output()
        .expect("binary runs");

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["verb"], "switch");
}

/// An unknown provider name is rejected before anything is touched.
#[test]
fn an_unknown_provider_is_refused_before_any_work() {
    let out = Command::new(BIN)
        .args(["load", "m", "--provider", "not-a-provider"])
        .output()
        .expect("binary runs");
    assert_ne!(out.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unknown provider kind"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// --- r27 Task 1: every verb answers in JSON ---------------------------------
//
// `pin`, `unpin`, `install`, `start` and `stop` printed prose and exited, and
// `estimate` emitted a bare document with no `schema`. A caller cannot compose
// a wrapper around a sentence, and it cannot tell "pinned it" from "it was
// already pinned" by matching English.

/// A HOME of its own, so a test never reads or writes the developer's pins.
fn isolated_home(t: &Tree) -> String {
    t.write("home/.keep", "").parent().unwrap().display().to_string()
}

/// A pin is idempotent, and the caller must be able to tell which happened --
/// "pinned it" and "it was already pinned" are the same outcome and different
/// events, and only one of them is worth a toast.
#[test]
fn pin_reports_each_target_and_whether_it_changed_anything() {
    let t = Tree::new("cli-pin-json");
    let home = isolated_home(&t);
    let run = || {
        Command::new(BIN)
            .args(["pin", "qwen3:14b", "--provider", "ollama", "--json"])
            .env("HOME", &home)
            .output()
            .expect("binary runs")
    };

    let first: serde_json::Value = serde_json::from_slice(&run().stdout).expect("a JSON document");
    assert_eq!(first["schema"], 1);
    assert_eq!(first["verb"], "pin");
    assert_eq!(first["changed"], true, "the first pin changed something");
    assert_eq!(first["targets"][0]["model"], "qwen3:14b");
    assert_eq!(first["targets"][0]["provider"], "ollama");
    assert_eq!(first["targets"][0]["changed"], true);

    let again: serde_json::Value = serde_json::from_slice(&run().stdout).expect("a JSON document");
    assert_eq!(again["changed"], false, "pinning twice changes nothing the second time");
    assert_eq!(again["targets"][0]["changed"], false);
}

/// Unpinning something that was never pinned is the same shape, not an error:
/// a UI asking for a state it is already in deserves an answer, not a failure.
#[test]
fn unpin_reports_nothing_changed_rather_than_failing() {
    let t = Tree::new("cli-unpin-json");
    let out = Command::new(BIN)
        .args(["unpin", "never-pinned", "--provider", "ollama", "--json"])
        .env("HOME", isolated_home(&t))
        .output()
        .expect("binary runs");

    assert_eq!(out.status.code(), Some(0), "not being pinned is not a failure");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("a JSON document");
    assert_eq!(v["verb"], "unpin");
    assert_eq!(v["changed"], false);
}

/// `start` is the verb behind a button on a dead provider. Its answer has to
/// carry the readiness wait, because "started" and "answering" are not the
/// same claim and the UI must not make the second one.
#[test]
fn start_reports_whether_the_provider_actually_answered() {
    let t = Tree::new("cli-start-json");
    // No launchd agent exists under this HOME, so this exercises the path the
    // button hits most: asked to start something that was never installed.
    let out = Command::new(BIN)
        .args(["start", "ollama", "--timeout-s", "1", "--json"])
        .env("HOME", isolated_home(&t))
        .output()
        .expect("binary runs");

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("a JSON document");
    assert_eq!(v["schema"], 1);
    assert_eq!(v["verb"], "start");
    assert_eq!(v["provider"], "ollama");
    assert_ne!(v["outcome"]["status"], "ready", "nothing was installed to start");
    assert!(
        v["outcome"]["reason"].as_str().is_some_and(|r| !r.is_empty()),
        "a failure states its reason: {v}"
    );
    assert_eq!(v["changed"], false);
}

/// `stop` passes harmony's own refusal through rather than flattening it, so
/// the page can say *why* rather than "failed".
#[test]
fn stop_passes_through_the_refusal_for_something_harmony_did_not_install() {
    let t = Tree::new("cli-stop-json");
    let out = Command::new(BIN)
        .args(["stop", "ollama", "--json"])
        .env("HOME", isolated_home(&t))
        .output()
        .expect("binary runs");

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("a JSON document");
    assert_eq!(v["verb"], "stop");
    assert_ne!(v["outcome"]["status"], "stopped");
    assert!(v["outcome"]["reason"].as_str().is_some_and(|r| !r.is_empty()), "{v}");
}

/// An estimate crossing a process boundary needs a version like every other
/// document. It is the one that already changed shape once.
#[test]
fn estimate_json_carries_a_schema() {
    let out = Command::new(BIN)
        .args(["estimate", "some-model-that-is-not-here", "--json"])
        .output()
        .expect("binary runs");

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout is a JSON document");
    assert_eq!(v["schema"], 1, "the estimate document is versioned: {v}");
}
