//! The one machine-readable document for the verbs that change state but are
//! not loads: `pin`, `unpin`, `install`, `start`, `stop`.
//!
//! These printed prose and exited. A caller cannot compose a wrapper around a
//! sentence, and it could not tell *pinned it* from *it was already pinned* —
//! the same outcome, different events, and only one of them worth a toast.
//!
//! `changed` is the field that distinction lives in, and it is deliberately
//! separate from success. *Already pinned*, *already running* and *already
//! installed* are all successes that did nothing, and a UI that reports them
//! as actions is a UI that lies about what the user just did.
//!
//! Versioned separately from the status and actuating documents, for the
//! reason `render_actuate` gives: these three change shape on their own
//! schedules, and a consumer pinning one must not be broken by another moving.

use serde::Serialize;

/// Bumped when a field is removed or changes meaning. Adding one is not a bump.
pub const SCHEMA: u32 = 1;

/// One model on one provider. `pin` without `--provider` fans out across every
/// provider that could serve the model, so the answer is a list even when the
/// user named one thing.
#[derive(Serialize)]
pub struct Target {
    pub provider: String,
    pub model: String,
    pub changed: bool,
}

/// What happened. `status` is the discriminant a caller matches on; every
/// failure carries a `reason` rather than leaving the caller to infer one from
/// an exit code it cannot see.
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    Pinned,
    Unpinned,
    /// Asked to unpin something that was not pinned. A success: the caller
    /// wanted a state and the state holds.
    NotPinned,
    /// Started *and answering*. Those are not the same claim, and the wait is
    /// reported so a UI can show what it cost rather than implying the second
    /// from the first.
    Ready { url: String, took_s: u64 },
    Stopped,
    Installed { path: String },
    /// `install --dry-run`: the agent that would have been written.
    Planned { path: String, plist: String },
    Failed {
        reason: String,
        /// What the launchd agent's own start command exited with, when there
        /// is one. The reason a start fails is usually here and not in ours.
        #[serde(skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
        /// Where to read more. Naming the log beats describing it.
        #[serde(skip_serializing_if = "Option::is_none")]
        log: Option<String>,
    },
}

#[derive(Serialize)]
pub struct Lifecycle {
    pub schema: u32,
    pub verb: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<Target>,
    pub outcome: Outcome,
    /// Whether anything on this machine is different for having run this.
    pub changed: bool,
}

impl Lifecycle {
    pub fn provider(verb: &'static str, provider: &str, outcome: Outcome, changed: bool) -> Self {
        Lifecycle {
            schema: SCHEMA,
            verb,
            provider: Some(provider.to_string()),
            targets: Vec::new(),
            outcome,
            changed,
        }
    }

    pub fn pins(verb: &'static str, targets: Vec<Target>, outcome: Outcome) -> Self {
        let changed = targets.iter().any(|t| t.changed);
        Lifecycle { schema: SCHEMA, verb, provider: None, targets, outcome, changed }
    }

    pub fn print(&self) {
        println!("{}", serde_json::to_string_pretty(self).unwrap());
    }

    /// A failure is still a document. The exit code says it failed; the
    /// document says why, and a caller reading stdout must not get nothing.
    pub fn failed(verb: &'static str, provider: &str, reason: impl Into<String>) -> Self {
        Lifecycle::provider(
            verb,
            provider,
            Outcome::Failed { reason: reason.into(), exit_code: None, log: None },
            false,
        )
    }
}
