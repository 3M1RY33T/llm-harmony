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
