use crate::estimate::estimator::{Basis, Estimate};
use crate::inventory::artifact::Provenance;
use crate::memory::Machine;
use crate::provider::State;
use crate::resolve::identity::Candidate;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "decision", rename_all = "kebab-case")]
pub enum Decision {
    /// Talk to this endpoint. Harmony is now out of the way.
    Ready {
        base_url: String,
        provider_model_id: String,
        provider: String,
        /// True when the model is already resident: no admission was needed.
        resident: bool,
        inferred: bool,
    },
    /// It can be served, but something must happen first.
    Wait { reason: String, eta_s: u64 },
    /// It cannot be served right now.
    Deny {
        reason: String,
        /// Other places this model exists. Offered, never chosen -- silently
        /// substituting a quantisation is how a caller loses a capability it
        /// asked for. See docs/field-notes.md on MLX conversion dropping MTP
        /// tensors and a vision tower.
        alternatives: Vec<String>,
    },
}

fn alternatives(candidates: &[Candidate], chosen: Option<&Candidate>) -> Vec<String> {
    candidates
        .iter()
        .filter(|c| Some(*c) != chosen)
        .map(|c| format!("{} on {}", c.provider_model_id, c.provider))
        .collect()
}

/// Where to send the request, if anywhere.
///
/// Order of preference: a model already resident on a running provider, then a
/// running provider that lists it. Within a tier, a candidate whose artifact
/// was *reported* beats one that was derived.
///
/// This slice cannot evict. When the estimate does not fit the headroom the
/// answer is `Deny`, and nothing resident is disturbed.
pub fn decide(
    candidates: &[Candidate],
    estimate: &Estimate,
    machine: &Machine,
    reserve_bytes: u64,
) -> Decision {
    if candidates.is_empty() {
        return Decision::Deny {
            reason: "no provider serves this model".to_string(),
            alternatives: Vec::new(),
        };
    }

    let recorded_first = |c: &&Candidate| match c.provenance {
        Provenance::Recorded => 0,
        Provenance::Inferred { .. } => 1,
    };

    // Already resident: it is costing what it costs whether we answer or not,
    // so there is nothing to admit.
    if let Some(c) = candidates
        .iter()
        .filter(|c| c.state == State::Loaded)
        .min_by_key(recorded_first)
    {
        return ready(c, true);
    }

    let Some(c) = candidates.iter().min_by_key(recorded_first) else {
        unreachable!("checked non-empty");
    };

    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve_bytes);

    match (estimate.bytes, estimate.basis) {
        (Some(bytes), Basis::Measured | Basis::Declared) if bytes <= headroom => ready(c, false),
        (Some(bytes), Basis::Measured | Basis::Declared) => Decision::Deny {
            reason: format!(
                "needs {} but only {} is available; nothing was unloaded",
                crate::render::human_bytes(bytes),
                crate::render::human_bytes(headroom)
            ),
            alternatives: alternatives(candidates, Some(c)),
        },
        // design.md section 5: never invent a measurement. An unmeasured shape
        // cannot be admitted, and it cannot be refused on memory grounds
        // either -- so say which it is.
        _ => Decision::Deny {
            reason: "this shape has never been measured, so it cannot be admitted".to_string(),
            alternatives: alternatives(candidates, Some(c)),
        },
    }
}

fn ready(c: &Candidate, resident: bool) -> Decision {
    Decision::Ready {
        base_url: c.url.clone(),
        provider_model_id: c.provider_model_id.clone(),
        provider: c.provider.to_string(),
        resident,
        inferred: !c.provenance.is_recorded(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind;

    fn machine(free: u64) -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 25_769_803_776 - free,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        }
    }

    fn cand(kind: ProviderKind, state: State, recorded: bool) -> Candidate {
        Candidate {
            provider: kind,
            provider_model_id: format!("m-on-{kind}"),
            url: format!("http://127.0.0.1:{}", kind.default_port()),
            state,
            artifact: Some("a".into()),
            provenance: if recorded {
                Provenance::Recorded
            } else {
                Provenance::Inferred { basis: "derived".into() }
            },
        }
    }

    fn measured(bytes: u64) -> Estimate {
        Estimate { bytes: Some(bytes), basis: Basis::Measured, samples: 3, spread_bytes: Some(0) }
    }

    fn unknown() -> Estimate {
        Estimate { bytes: None, basis: Basis::Unknown, samples: 0, spread_bytes: None }
    }

    /// A resident model needs no admission: it already costs what it costs.
    #[test]
    fn an_already_resident_model_is_ready_immediately() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::Loaded, true)];
        // Deliberately no headroom at all -- residency must win anyway.
        match decide(&c, &measured(9_000_000_000), &machine(0), 0) {
            Decision::Ready { resident, .. } => assert!(resident),
            d => panic!("{d:?}"),
        }
    }

    #[test]
    fn a_model_that_fits_is_ready() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        match decide(&c, &measured(2_000_000_000), &machine(12_000_000_000), 0) {
            Decision::Ready { resident, .. } => assert!(!resident),
            d => panic!("{d:?}"),
        }
    }

    /// This slice cannot evict. Refusing is the honest answer.
    #[test]
    fn a_model_that_does_not_fit_is_denied_not_evicted_for() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        match decide(&c, &measured(9_000_000_000), &machine(3_000_000_000), 0) {
            Decision::Deny { reason, .. } => {
                assert!(reason.contains("nothing was unloaded"), "{reason}")
            }
            d => panic!("{d:?}"),
        }
    }

    #[test]
    fn an_unknown_estimate_is_denied_with_that_as_the_reason() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        match decide(&c, &unknown(), &machine(20_000_000_000), 0) {
            Decision::Deny { reason, .. } => assert!(reason.contains("never been measured"), "{reason}"),
            d => panic!("{d:?}"),
        }
    }

    /// A reported artifact beats a derived one when both could serve.
    #[test]
    fn a_recorded_candidate_is_preferred_over_an_inferred_one() {
        let c = vec![
            cand(ProviderKind::LmStudio, State::NotLoaded, false),
            cand(ProviderKind::LlamaCpp, State::NotLoaded, true),
        ];
        match decide(&c, &measured(1_000_000_000), &machine(20_000_000_000), 0) {
            Decision::Ready { inferred, provider, .. } => {
                assert!(!inferred);
                assert_eq!(provider, "llamacpp");
            }
            d => panic!("{d:?}"),
        }
    }

    #[test]
    fn an_inferred_candidate_produces_an_inferred_ready() {
        let c = vec![cand(ProviderKind::LmStudio, State::NotLoaded, false)];
        match decide(&c, &measured(1_000_000_000), &machine(20_000_000_000), 0) {
            Decision::Ready { inferred, .. } => assert!(inferred),
            d => panic!("{d:?}"),
        }
    }

    /// Denial names what else exists without picking one.
    #[test]
    fn a_denial_offers_alternatives_but_never_selects_one() {
        let c = vec![
            cand(ProviderKind::LlamaCpp, State::NotLoaded, true),
            cand(ProviderKind::Vllm, State::NotLoaded, true),
        ];
        match decide(&c, &measured(90_000_000_000), &machine(1_000_000_000), 0) {
            Decision::Deny { alternatives, .. } => assert_eq!(alternatives.len(), 1),
            d => panic!("{d:?}"),
        }
    }

    #[test]
    fn a_model_nothing_serves_is_denied_with_no_alternatives() {
        match decide(&[], &measured(1), &machine(20_000_000_000), 0) {
            Decision::Deny { alternatives, reason } => {
                assert!(alternatives.is_empty());
                assert!(reason.contains("no provider"), "{reason}");
            }
            d => panic!("{d:?}"),
        }
    }
}
