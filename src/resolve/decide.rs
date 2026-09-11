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
        (Some(bytes), Basis::Measured | Basis::Computed) if bytes <= headroom => ready(c, false),
        (Some(bytes), Basis::Measured | Basis::Computed) => Decision::Deny {
            reason: format!(
                "needs {} but only {} is available; nothing was unloaded",
                crate::render::human_bytes(bytes),
                crate::render::human_bytes(headroom)
            ),
            alternatives: alternatives(candidates, Some(c)),
        },
        // A declared figure covers weights and nothing else -- every source
        // that publishes one says so, and `lms --estimate-only` was measured
        // flat across a 5x context change on 2026-09-10. Admitting on it is
        // admitting on an under-estimate, which is the one failure that costs
        // the machine. It is reported, never acted on.
        (Some(_), Basis::Declared) => Decision::Deny {
            reason: "only a declared figure is available, which covers weights \
                     and not the cache that scales with the window"
                .to_string(),
            alternatives: alternatives(candidates, Some(c)),
        },
        // design.md section 5: never invent a measurement. A shape that could
        // not even be read cannot be priced -- say which it is.
        _ => Decision::Deny {
            reason: "this shape could not be measured or computed, so it cannot be admitted"
                .to_string(),
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

    /// Unknown is now narrower than it was in slice 4: a shape that can be
    /// read gets a computed figure, so reaching here means the artifact itself
    /// could not be priced.
    #[test]
    fn an_unknown_estimate_is_denied_with_that_as_the_reason() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        match decide(&c, &unknown(), &machine(20_000_000_000), 0) {
            Decision::Deny { reason, .. } => {
                assert!(reason.contains("could not be measured or computed"), "{reason}")
            }
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

    /// The live hazard this closes. A declared figure is weights-only by
    /// construction -- vLLM's `memory_gb`, Ollama's `size`, a file's length on
    /// disk, and `lms load --estimate-only` alike, all verified 2026-09-10.
    /// Admitting on it is admitting on an under-estimate.
    #[test]
    fn a_declared_figure_is_never_admitted_on_its_own() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        let declared = Estimate {
            bytes: Some(2_000_000_000),
            basis: Basis::Declared,
            samples: 0,
            spread_bytes: None,
        };
        match decide(&c, &declared, &machine(20_000_000_000), 0) {
            Decision::Deny { reason, .. } => {
                assert!(reason.contains("weights"), "say why it is not enough: {reason}")
            }
            d => panic!("a declared floor must not admit: {d:?}"),
        }
    }

    #[test]
    fn a_computed_estimate_that_fits_is_admitted() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        let e = Estimate {
            bytes: Some(2_000_000_000),
            basis: Basis::Computed,
            samples: 0,
            spread_bytes: None,
        };
        match decide(&c, &e, &machine(20_000_000_000), 0) {
            Decision::Ready { resident, .. } => assert!(!resident),
            d => panic!("{d:?}"),
        }
    }

    /// And a computed figure that does not fit refuses on the arithmetic,
    /// not on provenance.
    #[test]
    fn a_computed_estimate_that_does_not_fit_is_denied_on_the_numbers() {
        let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
        let e = Estimate {
            bytes: Some(9_000_000_000),
            basis: Basis::Computed,
            samples: 0,
            spread_bytes: None,
        };
        match decide(&c, &e, &machine(3_000_000_000), 0) {
            Decision::Deny { reason, .. } => assert!(reason.contains("only"), "{reason}"),
            d => panic!("{d:?}"),
        }
    }
}
