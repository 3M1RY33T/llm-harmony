//! What to unload, and what to load, decided before anything is touched.
//!
//! A pure function over the ledger, the pins and an estimate. Everything
//! interesting about this slice is decided here and none of it needs a provider
//! to be running, which is why it is a function and not a method on a client.
//!
//! ## On "never evict a busy model"
//!
//! The plan for this slice carried that as a hard constraint, with an unknown
//! busy state counting as busy. Task 6 established that **none of the four
//! providers publishes per-model request state** -- `busy()` returns `Err`
//! for all of them. Taken together those two rules make every model
//! permanently ineligible, which would leave the feature unable to free a
//! single byte.
//!
//! So the constraint is narrowed to what can actually be known: a model
//! *observed* to be busy is never evicted, and where busy state is unpublished
//! the protection is the **pin** -- which is the mechanism chosen for this
//! slice precisely because the operator knows things the providers will not
//! say. This is a deliberate weakening of a stated constraint, recorded here
//! rather than discovered later.

use crate::estimate::estimator::{Basis, Estimate};
use crate::memory::Machine;
use crate::pins::Pins;
use crate::provider::{Actuation, ProviderKind};
use crate::resolve::identity::Candidate;

/// One resident model, as the planner needs to see it.
#[derive(Debug, Clone)]
pub struct Resident {
    pub provider: ProviderKind,
    pub model: String,
    /// What it is costing, by the same ladder admission uses.
    pub estimated_bytes: u64,
    /// Eviction order: **lower goes first**.
    ///
    /// Ollama supplies its `expires_at`, which it moves forward on every use,
    /// so among Ollama's own models this really is least-recently-used. The
    /// other three publish nothing comparable and get 0, which leaves their
    /// order the ledger's own.
    ///
    /// Deliberately not called `last_used`: an expiry is not a last-use time,
    /// and two models loaded with different `keep_alive` values are not
    /// comparable by it. It ranks; it does not measure.
    pub evict_rank: u64,
    /// Observed to be serving a request. `false` also covers "could not tell" --
    /// see the module docs.
    pub busy: bool,
    pub actuation: Actuation,
}

/// What the caller asked for.
#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub context_tokens: Option<u32>,
    /// `switch`: the model to free first, whatever the arithmetic says.
    pub free_first: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UnloadReason {
    /// The caller named it. That is consent, and it overrides a pin.
    Named,
    /// Chosen by the planner to free room. Never crosses a pin.
    MakingRoom,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "action", rename_all = "kebab-case")]
pub enum Action {
    Unload { provider: ProviderKind, model: String, reason: UnloadReason },
    Load { provider: ProviderKind, model: String, context_tokens: Option<u32> },
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "plan", rename_all = "kebab-case")]
pub enum Plan {
    Actions(Vec<Action>),
    AlreadyResident { provider: ProviderKind, model: String, base_url: String },
    Refused {
        reason: String,
        /// Every pinned model that would otherwise have been a candidate.
        /// Without this an operator reads "no room" on a half-empty machine
        /// and has no way to discover why.
        pinned_blockers: Vec<String>,
        alternatives: Vec<String>,
    },
}

/// Decide what to do, touching nothing.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    candidates: &[Candidate],
    resident: &[Resident],
    pins: &Pins,
    estimate: &Estimate,
    machine: &Machine,
    reserve_bytes: u64,
    request: &Request,
) -> Plan {
    let Some(target) = choose_candidate(candidates) else {
        return Plan::Refused {
            reason: format!("no provider serves `{}`", request.model),
            pinned_blockers: Vec::new(),
            alternatives: Vec::new(),
        };
    };

    // Residency short-circuits everything, exactly as `resolve::decide` does:
    // it costs what it costs whether or not we answer. And on LM Studio a
    // redundant load would start a second instance -- see field-notes.
    let already = resident
        .iter()
        .any(|r| r.provider == target.provider && r.model == target.provider_model_id);
    if already && request.free_first.is_none() {
        return Plan::AlreadyResident {
            provider: target.provider,
            model: target.provider_model_id.clone(),
            base_url: target.url.clone(),
        };
    }

    let mut actions: Vec<Action> = Vec::new();
    let mut freed: u64 = 0;

    // `switch` frees the named model first, whatever the arithmetic says.
    // The verb means "I am done with this one"; `load` is how two models end
    // up resident together.
    if let Some(name) = &request.free_first {
        match resident.iter().find(|r| &r.model == name) {
            Some(r) if !r.actuation.can_unload() => {
                return Plan::Refused {
                    reason: format!(
                        "`{}` is on {}, which has no model-level unload; it self-manages under {} \
                         -- stop the provider with `llm-harmony stop {}` if you mean to free it",
                        name,
                        r.provider,
                        ceiling_of(r.actuation),
                        r.provider
                    ),
                    pinned_blockers: Vec::new(),
                    alternatives: alternatives(candidates, Some(target)),
                };
            }
            Some(r) if r.busy => {
                return Plan::Refused {
                    reason: format!("`{name}` has a request in flight"),
                    pinned_blockers: Vec::new(),
                    alternatives: Vec::new(),
                };
            }
            Some(r) => {
                freed = freed.saturating_add(r.estimated_bytes);
                actions.push(Action::Unload {
                    provider: r.provider,
                    model: r.model.clone(),
                    reason: UnloadReason::Named,
                });
            }
            // Not resident: switching away from something already gone is a
            // load, not an error.
            None => {}
        }
    }

    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve_bytes).saturating_add(freed);

    let wanted = match (estimate.bytes, estimate.basis) {
        (Some(b), Basis::Measured | Basis::Computed) => b,
        (Some(_), Basis::Declared) => {
            return Plan::Refused {
                reason: "only a declared figure is available, which covers weights and not the \
                         cache that scales with the window"
                    .to_string(),
                pinned_blockers: Vec::new(),
                alternatives: alternatives(candidates, Some(target)),
            }
        }
        _ => {
            return Plan::Refused {
                reason: "this shape could not be measured or computed, so it cannot be admitted"
                    .to_string(),
                pinned_blockers: Vec::new(),
                alternatives: alternatives(candidates, Some(target)),
            }
        }
    };

    if wanted <= headroom {
        actions.push(load_action(target, request));
        return Plan::Actions(actions);
    }

    // Not enough room. Find the smallest set that makes enough, never crossing
    // a pin and never taking a model observed to be serving.
    let need = wanted.saturating_sub(headroom);
    let (victims, blocked) = victims_for(need, resident, pins, target, request);

    match victims {
        Some(vs) => {
            for r in vs {
                actions.push(Action::Unload {
                    provider: r.provider,
                    model: r.model.clone(),
                    reason: UnloadReason::MakingRoom,
                });
            }
            actions.push(load_action(target, request));
            Plan::Actions(actions)
        }
        None => Plan::Refused {
            reason: refusal_reason(wanted, headroom, &blocked),
            pinned_blockers: blocked,
            alternatives: alternatives(candidates, Some(target)),
        },
    }
}

/// A reported artifact beats a derived one, as in `resolve::decide`.
fn choose_candidate(candidates: &[Candidate]) -> Option<&Candidate> {
    candidates.iter().min_by_key(|c| {
        if c.provenance.is_recorded() {
            0
        } else {
            1
        }
    })
}

fn load_action(target: &Candidate, request: &Request) -> Action {
    Action::Load {
        provider: target.provider,
        model: target.provider_model_id.clone(),
        context_tokens: request.context_tokens,
    }
}

fn ceiling_of(a: Actuation) -> &'static str {
    match a {
        Actuation::SelfManaged { ceiling } => ceiling,
        Actuation::ModelLevel => "nothing",
    }
}

/// The smallest sufficient set, least-recently-used first.
///
/// Returns the pins it had to skip either way, so a refusal can name them.
fn victims_for<'a>(
    need: u64,
    resident: &'a [Resident],
    pins: &Pins,
    target: &Candidate,
    request: &Request,
) -> (Option<Vec<&'a Resident>>, Vec<String>) {
    let mut blocked: Vec<String> = Vec::new();
    let mut eligible: Vec<&Resident> = Vec::new();

    for r in resident {
        // Already being unloaded by name, or is the thing we are loading.
        if request.free_first.as_deref() == Some(r.model.as_str()) {
            continue;
        }
        if r.provider == target.provider && r.model == target.provider_model_id {
            continue;
        }
        if !r.actuation.can_unload() || r.busy {
            continue;
        }
        if pins.is_pinned(r.provider, &r.model) {
            blocked.push(format!("{} on {}", r.model, r.provider));
            continue;
        }
        eligible.push(r);
    }

    // Lowest rank first: where the provider gives a real signal this is the
    // model nobody has touched, and where it does not it is ledger order.
    eligible.sort_by_key(|r| r.evict_rank);

    let mut taken: Vec<&Resident> = Vec::new();
    let mut freed: u64 = 0;
    for r in eligible {
        if freed >= need {
            break;
        }
        freed = freed.saturating_add(r.estimated_bytes);
        taken.push(r);
    }

    if freed >= need {
        (Some(taken), blocked)
    } else {
        (None, blocked)
    }
}

fn refusal_reason(wanted: u64, headroom: u64, blocked: &[String]) -> String {
    let base = format!(
        "needs {} but only {} can be made available",
        crate::render::human_bytes(wanted),
        crate::render::human_bytes(headroom)
    );
    if blocked.is_empty() {
        base
    } else {
        format!("{base}; pinned and therefore untouchable: {}", blocked.join(", "))
    }
}

fn alternatives(candidates: &[Candidate], chosen: Option<&Candidate>) -> Vec<String> {
    candidates
        .iter()
        .filter(|c| Some(*c) != chosen)
        .map(|c| format!("{} on {}", c.provider_model_id, c.provider))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::Provenance;
    use crate::pins::Pin;
    use crate::provider::State;

    const GB: u64 = 1024 * 1024 * 1024;

    fn machine(free: u64) -> Machine {
        Machine {
            total_bytes: 24 * GB,
            used_bytes: 24 * GB - free,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        }
    }

    fn measured(bytes: u64) -> Estimate {
        Estimate { bytes: Some(bytes), basis: Basis::Measured, samples: 3, spread_bytes: Some(0) }
    }

    fn candidates(model: &str) -> Vec<Candidate> {
        vec![Candidate {
            provider: ProviderKind::LmStudio,
            provider_model_id: model.to_string(),
            url: "http://127.0.0.1:1234".into(),
            state: State::NotLoaded,
            artifact: Some("a".into()),
            provenance: Provenance::Recorded,
        }]
    }

    fn res(model: &str, bytes: u64, last_used: u64) -> Resident {
        Resident {
            provider: ProviderKind::LmStudio,
            model: model.to_string(),
            estimated_bytes: bytes,
            evict_rank: last_used,
            busy: false,
            actuation: Actuation::ModelLevel,
        }
    }

    fn resident(models: &[&str]) -> Vec<Resident> {
        models.iter().enumerate().map(|(i, m)| res(m, 9 * GB, i as u64)).collect()
    }

    fn pins_with(entries: &[(ProviderKind, &str)]) -> Pins {
        let mut p = Pins::empty();
        for (provider, model) in entries {
            p.add(Pin { provider: *provider, model: model.to_string(), at: 0, note: None });
        }
        p
    }

    fn req(model: &str) -> Request {
        Request { model: model.into(), context_tokens: None, free_first: None }
    }

    fn req_switch(from: &str, to: &str) -> Request {
        Request { model: to.into(), context_tokens: None, free_first: Some(from.into()) }
    }

    /// A pin is a veto. The room exists, the model is idle, and it still does
    /// not move -- that is the whole point of the flag.
    #[test]
    fn a_pinned_model_is_never_chosen_to_make_room() {
        let p = pins_with(&[(ProviderKind::LmStudio, "big-pinned")]);
        match plan(
            &candidates("wanted"),
            &resident(&["big-pinned"]),
            &p,
            &measured(9 * GB),
            &machine(2 * GB),
            0,
            &req("wanted"),
        ) {
            Plan::Refused { pinned_blockers, .. } => {
                assert_eq!(pinned_blockers, vec!["big-pinned on lmstudio"]);
            }
            p => panic!("{p:?}"),
        }
    }

    /// And because it is a veto rather than a reservation, the refusal has to
    /// say so -- otherwise the operator reads "no room" on a machine that
    /// looks half empty and has no way to discover why.
    #[test]
    fn a_refusal_caused_by_pins_names_them_in_its_reason() {
        let p = pins_with(&[(ProviderKind::LmStudio, "big-pinned")]);
        match plan(
            &candidates("wanted"),
            &resident(&["big-pinned"]),
            &p,
            &measured(9 * GB),
            &machine(2 * GB),
            0,
            &req("wanted"),
        ) {
            Plan::Refused { reason, .. } => {
                assert!(reason.contains("pinned"), "{reason}");
                assert!(reason.contains("big-pinned"), "{reason}");
            }
            p => panic!("{p:?}"),
        }
    }

    /// An explicitly named model beats a pin: the human named it, and that is
    /// consent. The pin itself survives, so the next load is protected again.
    #[test]
    fn naming_a_pinned_model_for_switch_overrides_the_pin() {
        let p = pins_with(&[(ProviderKind::LmStudio, "pinned")]);
        match plan(
            &candidates("b"),
            &resident(&["pinned"]),
            &p,
            &measured(1 * GB),
            &machine(20 * GB),
            0,
            &req_switch("pinned", "b"),
        ) {
            Plan::Actions(a) => {
                assert!(matches!(
                    &a[0],
                    Action::Unload { model, reason: UnloadReason::Named, .. } if model == "pinned"
                ));
            }
            p => panic!("{p:?}"),
        }
    }

    /// Busy beats everything, where busy can actually be observed.
    #[test]
    fn a_model_observed_busy_is_never_evicted_to_make_room() {
        let mut busy = res("serving", 9 * GB, 0);
        busy.busy = true;
        match plan(
            &candidates("wanted"),
            &[busy],
            &Pins::empty(),
            &measured(9 * GB),
            &machine(2 * GB),
            0,
            &req("wanted"),
        ) {
            Plan::Refused { pinned_blockers, .. } => {
                assert!(pinned_blockers.is_empty(), "busy is not a pin");
            }
            p => panic!("a model with a request in flight must not be evicted: {p:?}"),
        }
    }

    /// And naming it does not override that: consent covers the eviction, not
    /// the interruption of something already in flight.
    #[test]
    fn naming_a_busy_model_is_refused_rather_than_interrupting_it() {
        let mut busy = res("serving", 9 * GB, 0);
        busy.busy = true;
        match plan(
            &candidates("b"),
            &[busy],
            &Pins::empty(),
            &measured(1 * GB),
            &machine(20 * GB),
            0,
            &req_switch("serving", "b"),
        ) {
            Plan::Refused { reason, .. } => assert!(reason.contains("in flight"), "{reason}"),
            p => panic!("{p:?}"),
        }
    }

    /// `switch` means replace: it frees the named model even when both would
    /// have fitted. `load` is how two models end up resident together.
    #[test]
    fn switch_always_unloads_the_named_model_even_when_both_would_fit() {
        match plan(
            &candidates("b"),
            &resident(&["a"]),
            &Pins::empty(),
            &measured(1 * GB),
            &machine(20 * GB),
            0,
            &req_switch("a", "b"),
        ) {
            Plan::Actions(a) => {
                assert_eq!(a.len(), 2);
                assert!(matches!(&a[0], Action::Unload { reason: UnloadReason::Named, .. }));
                assert!(matches!(&a[1], Action::Load { .. }));
            }
            p => panic!("{p:?}"),
        }
    }

    /// `load` adds. With room for both, nothing is disturbed -- this is the
    /// multi-residency case, and it must cost zero unloads.
    #[test]
    fn load_does_not_evict_anything_when_the_model_fits_alongside() {
        match plan(
            &candidates("b"),
            &resident(&["a"]),
            &Pins::empty(),
            &measured(1 * GB),
            &machine(20 * GB),
            0,
            &req("b"),
        ) {
            Plan::Actions(a) => {
                assert_eq!(a.len(), 1);
                assert!(matches!(&a[0], Action::Load { .. }));
            }
            p => panic!("{p:?}"),
        }
    }

    /// Free the least it can. Evicting two models when one suffices is a bug
    /// that looks like caution.
    #[test]
    fn making_room_evicts_the_smallest_sufficient_set_lru_first() {
        let residents = vec![res("old", 5 * GB, 1), res("recent", 5 * GB, 99)];
        match plan(
            &candidates("wanted"),
            &residents,
            &Pins::empty(),
            &measured(4 * GB),
            &machine(3 * GB),
            0,
            &req("wanted"),
        ) {
            Plan::Actions(a) => {
                let unloads: Vec<&str> = a
                    .iter()
                    .filter_map(|x| match x {
                        Action::Unload { model, .. } => Some(model.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(unloads, vec!["old"], "one unload, least-recently-used");
            }
            p => panic!("{p:?}"),
        }
    }

    /// A model resident on a self-managed provider cannot be evicted at all,
    /// so it can never appear in a plan.
    #[test]
    fn a_self_managed_residency_is_never_planned_for_eviction() {
        let mut stuck = res("on-llamacpp", 9 * GB, 0);
        stuck.provider = ProviderKind::LlamaCpp;
        stuck.actuation = Actuation::SelfManaged { ceiling: "max_instances" };

        match plan(
            &candidates("wanted"),
            &[stuck],
            &Pins::empty(),
            &measured(9 * GB),
            &machine(2 * GB),
            0,
            &req("wanted"),
        ) {
            Plan::Refused { reason, .. } => assert!(reason.contains("needs"), "{reason}"),
            p => panic!("harmony cannot unload one model from llama.cpp: {p:?}"),
        }
    }

    /// And naming one says so, with the lever that does exist.
    #[test]
    fn switching_away_from_a_self_managed_model_names_the_stop_verb() {
        let mut stuck = res("on-llamacpp", 9 * GB, 0);
        stuck.provider = ProviderKind::LlamaCpp;
        stuck.actuation = Actuation::SelfManaged { ceiling: "max_instances" };

        match plan(
            &candidates("b"),
            &[stuck],
            &Pins::empty(),
            &measured(1 * GB),
            &machine(20 * GB),
            0,
            &req_switch("on-llamacpp", "b"),
        ) {
            Plan::Refused { reason, .. } => {
                assert!(reason.contains("max_instances"), "name the ceiling: {reason}");
                assert!(reason.contains("stop"), "name the lever that exists: {reason}");
            }
            p => panic!("{p:?}"),
        }
    }

    /// Residency short-circuits everything. On LM Studio a redundant load
    /// would start a second instance, so this is a safety property and not
    /// merely an optimisation -- see docs/field-notes.md.
    #[test]
    fn a_model_already_resident_is_reported_not_reloaded() {
        match plan(
            &candidates("wanted"),
            &resident(&["wanted"]),
            &Pins::empty(),
            &measured(9 * GB),
            &machine(0),
            0,
            &req("wanted"),
        ) {
            Plan::AlreadyResident { model, .. } => assert_eq!(model, "wanted"),
            p => panic!("{p:?}"),
        }
    }

    #[test]
    fn a_model_nothing_serves_is_refused_by_name() {
        match plan(
            &[],
            &[],
            &Pins::empty(),
            &measured(1 * GB),
            &machine(20 * GB),
            0,
            &req("ghost"),
        ) {
            Plan::Refused { reason, .. } => assert!(reason.contains("ghost"), "{reason}"),
            p => panic!("{p:?}"),
        }
    }

    /// The reserve is held back before any of this: it is what the OS is owed.
    #[test]
    fn the_reserve_is_subtracted_before_the_arithmetic() {
        match plan(
            &candidates("b"),
            &[],
            &Pins::empty(),
            &measured(5 * GB),
            &machine(6 * GB),
            8 * GB,
            &req("b"),
        ) {
            Plan::Refused { .. } => {}
            p => panic!("6 GB free minus an 8 GB reserve is no room at all: {p:?}"),
        }
    }

    /// Switching to a model that is already resident still frees the named
    /// one: the caller said they were done with it.
    #[test]
    fn switch_to_an_already_resident_model_still_frees_the_named_one() {
        let residents = vec![res("a", 5 * GB, 0), res("b", 5 * GB, 1)];
        match plan(
            &candidates("b"),
            &residents,
            &Pins::empty(),
            &measured(5 * GB),
            &machine(20 * GB),
            0,
            &req_switch("a", "b"),
        ) {
            Plan::Actions(a) => {
                assert!(matches!(
                    &a[0],
                    Action::Unload { model, reason: UnloadReason::Named, .. } if model == "a"
                ));
            }
            p => panic!("{p:?}"),
        }
    }

    /// Ollama's expiry moves forward on every use, so the model expiring
    /// soonest is the one least recently touched.
    #[test]
    fn a_provider_that_publishes_an_expiry_orders_by_it() {
        // `stale` expires sooner, so it is the one that goes.
        let residents = vec![res("fresh", 5 * GB, 9_000), res("stale", 5 * GB, 1_000)];
        match plan(
            &candidates("wanted"),
            &residents,
            &Pins::empty(),
            &measured(4 * GB),
            &machine(3 * GB),
            0,
            &req("wanted"),
        ) {
            Plan::Actions(a) => {
                let unloads: Vec<&str> = a
                    .iter()
                    .filter_map(|x| match x {
                        Action::Unload { model, .. } => Some(model.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(unloads, vec!["stale"]);
            }
            p => panic!("{p:?}"),
        }
    }
}
