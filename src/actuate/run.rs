//! Carrying out a plan: lock, decide, act, verify, report.
//!
//! The order is the whole safety argument. The lock is held across admission
//! *and* the load it authorises, so no grant outlives the process that was
//! given it. Every action is confirmed by re-reading the provider rather than
//! by its exit code -- `lms unload` exits 0 while leaving a second instance
//! resident (`docs/field-notes.md`), and an actuation that reports success and
//! changes nothing is the worst failure this code could have.

use std::time::{Duration, Instant};

use crate::actuate::lock::ActuationLock;
use crate::actuate::plan::{plan, Action, Plan, Request, Resident, UnloadReason};
use crate::actuate::watchdog::{self, Limits};
use crate::adapters::adapter_for;
use crate::config::Config;
use crate::estimate::estimator::Estimate;
use crate::http::Http;
use crate::ledger::{Ledger, Outcome as RowOutcome};
use crate::memory::Machine;
use crate::pins::{Pin, Pins};
use crate::provider::{Actuation, Adapter, LoadRequest, ProviderKind, State};

/// The sentinel a warm-up carries.
///
/// Self-managed providers have no control-plane load verb -- residency happens
/// on first inference request (`field-notes.md`). So `ready` can only mean
/// *resident* if harmony triggers that itself.
///
/// This is the one place harmony touches a completion endpoint, and
/// `architecture.md` is amended to match: the claim that carries weight is
/// that **no user prompt passes through harmony**, not that no byte does. One
/// fixed token, never content, only when residency was asked for.
const WARM_UP_SENTINEL: &str = "llm-harmony";

#[derive(Debug, Clone)]
pub struct Options {
    pub reserve_bytes: u64,
    /// Overrides the watchdog's abort threshold. Half the reserve by default.
    pub floor_bytes: Option<u64>,
    pub pin_after: bool,
    pub lock_timeout: Duration,
    pub verify_timeout: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            reserve_bytes: crate::render_estimate::DEFAULT_RESERVE_BYTES,
            floor_bytes: None,
            pin_after: false,
            // A load is slow; failing fast here would mean two callers could
            // not queue behind each other at all.
            lock_timeout: Duration::from_secs(120),
            verify_timeout: Duration::from_secs(20),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UnloadRecord {
    pub provider: ProviderKind,
    pub model: String,
    pub reason: UnloadReason,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum Status {
    Ready {
        base_url: String,
        provider: ProviderKind,
        provider_model_id: String,
        /// Confirmed by re-reading the provider, never assumed.
        resident: bool,
        took_s: u64,
    },
    Refused {
        reason: String,
        pinned_blockers: Vec<String>,
        alternatives: Vec<String>,
    },
    Failed {
        reason: String,
        /// What to type to put back what was unloaded. There is no rollback by
        /// design; this is legibility, not recovery.
        restore: Option<String>,
    },
    Aborted {
        reason: String,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Report {
    /// This document's own version. `status` stays at 1 independently -- see
    /// `docs/plans/2026-09-10-actuation.md`.
    pub schema: u32,
    pub verb: &'static str,
    pub outcome: Status,
    pub unloaded: Vec<UnloadRecord>,
    pub estimate: Estimate,
    pub free_bytes: u64,
    pub swap_used_bytes: u64,
}

pub const ACTUATE_SCHEMA: u32 = 1;

impl Report {
    pub fn is_success(&self) -> bool {
        matches!(self.outcome, Status::Ready { .. })
    }
}

/// Every resident model, as the planner needs to see it.
///
/// `last_used` is zero for all of them: no provider publishes a last-use time,
/// so LRU degrades to the ledger's own order. Recorded here rather than hidden,
/// because it means "least recently used" is currently "first seen".
pub fn residents(ledger: &Ledger, http: &Http) -> Vec<Resident> {
    let mut out = Vec::new();
    for row in &ledger.rows {
        let RowOutcome::Ok(models) = &row.outcome else { continue };
        let adapter = adapter_for(row.kind);
        let loaded: Vec<_> = models.iter().filter(|m| m.state == State::Loaded).collect();
        // Attribution, same rule the estimator uses: a provider serving
        // exactly one model can have its footprint assigned to that model.
        let attributable = loaded.len() == 1;

        for m in loaded {
            let estimated_bytes = if attributable {
                row.footprint_bytes.or(m.weights_bytes).unwrap_or(0)
            } else {
                m.weights_bytes.unwrap_or(0)
            };
            out.push(Resident {
                provider: row.kind,
                model: m.id.clone(),
                estimated_bytes,
                last_used: 0,
                // An Err means the provider could not say. See the module docs
                // on `actuate::plan` for why that is not treated as busy.
                busy: adapter.busy(http, &row.url, &m.id).unwrap_or(false),
                actuation: adapter.actuation(),
            });
        }
    }
    out
}

/// `load` and `switch`. The difference is entirely `request.free_first`.
#[allow(clippy::too_many_arguments)]
pub fn load_or_switch(
    verb: &'static str,
    config: &Config,
    http: &Http,
    machine: Machine,
    request: &Request,
    estimate_for: &dyn Fn(&Ledger, &[crate::resolve::identity::Candidate]) -> Estimate,
    options: &Options,
) -> Report {
    let _lock = match ActuationLock::acquire(verb, options.lock_timeout) {
        Ok(l) => l,
        Err(e) => return failed(verb, e.to_string(), machine, Estimate::unknown()),
    };

    let ledger = Ledger::assemble(config, http, machine);
    let pins = Pins::load();
    let inventory = crate::inventory::Inventory::scan_offline(None);
    let candidates = crate::resolve::identity::candidates(&ledger, &inventory, &request.model);
    let estimate = estimate_for(&ledger, &candidates);
    let resident = residents(&ledger, http);

    let decided = plan(
        &candidates,
        &resident,
        &pins,
        &estimate,
        &machine,
        options.reserve_bytes,
        request,
    );

    let actions = match decided {
        Plan::AlreadyResident { provider, model, base_url } => {
            if options.pin_after {
                pin(provider, &model);
            }
            return Report {
                schema: ACTUATE_SCHEMA,
                verb,
                outcome: Status::Ready {
                    base_url,
                    provider,
                    provider_model_id: model,
                    resident: true,
                    took_s: 0,
                },
                unloaded: Vec::new(),
                estimate,
                free_bytes: machine.total_bytes.saturating_sub(machine.used_bytes),
                swap_used_bytes: machine.swap_used_bytes,
            };
        }
        Plan::Refused { reason, pinned_blockers, alternatives } => {
            return Report {
                schema: ACTUATE_SCHEMA,
                verb,
                outcome: Status::Refused { reason, pinned_blockers, alternatives },
                unloaded: Vec::new(),
                estimate,
                free_bytes: machine.total_bytes.saturating_sub(machine.used_bytes),
                swap_used_bytes: machine.swap_used_bytes,
            };
        }
        Plan::Actions(a) => a,
    };

    execute(verb, config, http, actions, estimate, options)
}

fn execute(
    verb: &'static str,
    config: &Config,
    http: &Http,
    actions: Vec<Action>,
    estimate: Estimate,
    options: &Options,
) -> Report {
    let started = Instant::now();
    let mut unloaded: Vec<UnloadRecord> = Vec::new();

    for action in actions {
        match action {
            Action::Unload { provider, model, reason } => {
                let Some(url) = url_for(config, provider) else { continue };
                let adapter = adapter_for(provider);
                if let Err(e) = adapter.unload(http, &url, &model) {
                    return finish(
                        verb,
                        Status::Failed { reason: e.to_string(), restore: None },
                        unloaded,
                        estimate,
                    );
                }
                // Never trust the exit code: `lms unload` reports success while
                // leaving a duplicate instance resident.
                if !settles(options.verify_timeout, || {
                    !is_resident(http, &*adapter, &url, &model)
                }) {
                    return finish(
                        verb,
                        Status::Failed {
                            reason: format!(
                                "`{model}` on {provider} is still resident after the unload \
                                 reported success"
                            ),
                            restore: None,
                        },
                        unloaded,
                        estimate,
                    );
                }
                unloaded.push(UnloadRecord { provider, model, reason });
            }

            Action::Load { provider, model, context_tokens } => {
                let Some(url) = url_for(config, provider) else { continue };
                let adapter = adapter_for(provider);
                let req = LoadRequest { model: model.clone(), context_tokens };

                let start = match adapter.actuation() {
                    Actuation::ModelLevel => adapter.load(http, &url, &req),
                    // No load verb: residency happens on first request, so
                    // harmony sends the one it is allowed to send.
                    Actuation::SelfManaged { .. } => warm_up(http, &url, &model, context_tokens),
                };
                if let Err(e) = start {
                    return finish(
                        verb,
                        Status::Failed { reason: e.to_string(), restore: restore_hint(&unloaded) },
                        unloaded,
                        estimate,
                    );
                }

                let limits = Limits {
                    floor_bytes: options
                        .floor_bytes
                        .unwrap_or(options.reserve_bytes / 2),
                    ..Limits::from_reserve(options.reserve_bytes)
                };
                let outcome = watchdog::supervise(
                    &limits,
                    || Machine::read().unwrap_or(Machine::zero()),
                    || is_resident(http, &*adapter, &url, &model),
                );

                match outcome {
                    watchdog::Outcome::Loaded { took_s } => {
                        if options.pin_after {
                            pin(provider, &model);
                        }
                        record_observation(config, http);
                        return finish(
                            verb,
                            Status::Ready {
                                base_url: url,
                                provider,
                                provider_model_id: model,
                                resident: true,
                                took_s: took_s.max(started.elapsed().as_secs()),
                            },
                            unloaded,
                            estimate,
                        );
                    }
                    watchdog::Outcome::Aborted { reason } => {
                        // Undo what we started, where undoing is possible.
                        let _ = adapter.unload(http, &url, &model);
                        return finish(verb, Status::Aborted { reason }, unloaded, estimate);
                    }
                    watchdog::Outcome::TimedOut => {
                        return finish(
                            verb,
                            Status::Failed {
                                reason: format!(
                                    "`{model}` never became resident on {provider}"
                                ),
                                restore: restore_hint(&unloaded),
                            },
                            unloaded,
                            estimate,
                        )
                    }
                }
            }
        }
    }

    finish(
        verb,
        Status::Failed { reason: "the plan contained nothing to do".into(), restore: None },
        unloaded,
        estimate,
    )
}

/// `unload`, which is its own path: there is nothing to admit and nothing to
/// plan, only consent that was given by naming the model.
pub fn unload(
    config: &Config,
    http: &Http,
    machine: Machine,
    model: &str,
    provider: Option<ProviderKind>,
    options: &Options,
) -> Report {
    let _lock = match ActuationLock::acquire("unload", options.lock_timeout) {
        Ok(l) => l,
        Err(e) => return failed("unload", e.to_string(), machine, Estimate::unknown()),
    };

    let ledger = Ledger::assemble(config, http, machine);
    let here = residents(&ledger, http)
        .into_iter()
        .find(|r| r.model == model && provider.map(|p| p == r.provider).unwrap_or(true));

    let Some(target) = here else {
        return failed("unload", format!("`{model}` is not resident"), machine, Estimate::unknown());
    };
    if !target.actuation.can_unload() {
        return failed(
            "unload",
            format!(
                "`{model}` is on {}, which has no model-level unload; stop the provider with \
                 `llm-harmony stop {}` if you mean to free it",
                target.provider, target.provider
            ),
            machine,
            Estimate::unknown(),
        );
    }
    if target.busy {
        return failed("unload", format!("`{model}` has a request in flight"), machine, Estimate::unknown());
    }

    // A pin does not stop an unload the caller named -- that is consent -- and
    // the pin survives, so the next load is protected again.
    let actions = vec![Action::Unload {
        provider: target.provider,
        model: target.model.clone(),
        reason: UnloadReason::Named,
    }];

    let mut report = execute("unload", config, http, actions, Estimate::unknown(), options);
    // `execute` ends by looking for a load; for `unload` the unload IS the
    // whole job, so a plan that ran out of actions is success.
    if let Status::Failed { reason, .. } = &report.outcome {
        if reason.contains("nothing to do") && !report.unloaded.is_empty() {
            report.outcome = Status::Ready {
                base_url: String::new(),
                provider: target.provider,
                provider_model_id: target.model,
                resident: false,
                took_s: 0,
            };
        }
    }
    report
}

// --- helpers ---------------------------------------------------------------

fn url_for(config: &Config, kind: ProviderKind) -> Option<String> {
    config.providers.iter().find(|p| p.kind == kind).map(|p| p.url.clone())
}

fn is_resident(http: &Http, adapter: &dyn Adapter, url: &str, model: &str) -> bool {
    adapter
        .list(http, url)
        .map(|ms| ms.iter().any(|m| m.id == model && m.state == State::Loaded))
        .unwrap_or(false)
}

/// Poll until a condition holds, or give up.
///
/// Providers do not update their listings synchronously with their control
/// verbs, so a single re-read races them.
fn settles(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let started = Instant::now();
    loop {
        if done() {
            return true;
        }
        if started.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// One fixed token to a provider that has no control-plane load verb.
fn warm_up(
    http: &Http,
    url: &str,
    model: &str,
    context_tokens: Option<u32>,
) -> Result<(), crate::provider::ActuateError> {
    let _ = context_tokens; // the window is a server-side flag on these two
    let body = serde_json::json!({
        "model": model,
        "prompt": WARM_UP_SENTINEL,
        "max_tokens": 1,
        "temperature": 0,
    });
    http.post_json(&format!("{url}/v1/completions"), &body)
        .map(|_| ())
        .map_err(|e| crate::provider::ActuateError::Failed { reason: e.to_string() })
}

fn pin(provider: ProviderKind, model: &str) {
    let mut pins = Pins::load();
    if pins.add(Pin {
        provider,
        model: model.to_string(),
        at: crate::record::now_unix(),
        note: None,
    }) {
        let _ = pins.save();
    }
}

/// Growing the corpus where admission reads from it. Best effort: a recording
/// failure may never fail an actuation that worked.
fn record_observation(config: &Config, http: &Http) {
    let Ok(machine) = Machine::read() else { return };
    let Some(path) = crate::record::default_path() else { return };
    let ledger = Ledger::assemble(config, http, machine);
    let obs = crate::record::observations(&ledger, crate::record::now_unix());
    let _ = crate::record::append(&path, &obs);
}

fn restore_hint(unloaded: &[UnloadRecord]) -> Option<String> {
    let first = unloaded.first()?;
    Some(format!("llm-harmony load {} --provider {}", first.model, first.provider))
}

fn finish(
    verb: &'static str,
    outcome: Status,
    unloaded: Vec<UnloadRecord>,
    estimate: Estimate,
) -> Report {
    let machine = Machine::read().unwrap_or(Machine::zero());
    Report {
        schema: ACTUATE_SCHEMA,
        verb,
        outcome,
        unloaded,
        estimate,
        free_bytes: machine.total_bytes.saturating_sub(machine.used_bytes),
        swap_used_bytes: machine.swap_used_bytes,
    }
}

fn failed(verb: &'static str, reason: String, machine: Machine, estimate: Estimate) -> Report {
    Report {
        schema: ACTUATE_SCHEMA,
        verb,
        outcome: Status::Failed { reason, restore: None },
        unloaded: Vec::new(),
        estimate,
        free_bytes: machine.total_bytes.saturating_sub(machine.used_bytes),
        swap_used_bytes: machine.swap_used_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settling_gives_up_rather_than_waiting_forever() {
        assert!(!settles(Duration::from_millis(30), || false));
    }

    #[test]
    fn settling_returns_as_soon_as_the_condition_holds() {
        let started = Instant::now();
        assert!(settles(Duration::from_secs(10), || true));
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    /// The restore line is what replaces a rollback. Without it a failed
    /// switch leaves the operator knowing something is gone and not what.
    #[test]
    fn a_failed_switch_says_how_to_put_back_what_it_unloaded() {
        let unloaded = vec![UnloadRecord {
            provider: ProviderKind::LmStudio,
            model: "qwen3-14b".into(),
            reason: UnloadReason::Named,
        }];
        let hint = restore_hint(&unloaded).unwrap();
        assert!(hint.contains("llm-harmony load qwen3-14b"), "{hint}");
        assert!(hint.contains("--provider lmstudio"), "{hint}");
    }

    #[test]
    fn nothing_unloaded_means_nothing_to_restore() {
        assert!(restore_hint(&[]).is_none());
    }
}
