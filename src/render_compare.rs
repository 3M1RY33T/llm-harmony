//! Every provider that could serve one model, priced, with the resolver's own
//! pick marked.
//!
//! `resolve` builds this list and returns one row of it; the rest are thrown
//! away unless the answer was a refusal, where `decide::alternatives` keeps
//! their names and nothing else. Printing the whole list is the only new thing
//! here -- the candidates, the prices and the pick all come from code the
//! resolver already runs.
//!
//! Two invariants, because a menu that disagreed with the resolver would be
//! worse than no menu at all:
//!
//! - every price comes from the shared estimate ladder, called once per
//!   candidate, so `compare` and `resolve` cannot print different numbers;
//! - the marked row is whatever [`Decision`] the resolver returned. Nothing in
//!   this module re-derives a preference order, and *cheaper* is deliberately
//!   not the rule -- `decide` prefers residency first and recorded identity
//!   second, and a menu that quietly sorted by price would be advertising a
//!   choice the resolver will not make.

use crate::estimate::estimator::{Basis, Estimate};
use crate::inventory::artifact::Provenance;
use crate::memory::Machine;
use crate::provider::State;
use crate::render::human_bytes;
use crate::resolve::decide::Decision;
use crate::resolve::identity::Candidate;

/// One place this model could be served from, and what it would cost there.
#[derive(Debug, Clone)]
pub struct Priced {
    pub candidate: Candidate,
    pub estimate: Estimate,
}

fn basis_label(e: &Estimate) -> &'static str {
    match e.basis {
        Basis::Measured => "measured",
        Basis::Computed => "computed",
        Basis::Declared => "declared",
        Basis::Unknown => "unknown",
    }
}

/// What the row claims about admission.
///
/// `fits` on a `Declared` figure is not the same claim as `fits` on a
/// measurement -- a declared figure covers weights only -- so the word is
/// qualified rather than repeated. `decide` will not admit on either
/// `Declared` or `Unknown`, and the label says so instead of implying a
/// verdict the resolver would refuse.
fn fit_label(p: &Priced, headroom: u64) -> String {
    if p.candidate.state == State::Loaded {
        return "resident".to_string();
    }
    match (p.estimate.bytes, p.estimate.basis) {
        (None, _) | (_, Basis::Unknown) => "unpriced".to_string(),
        (Some(b), Basis::Declared) if b <= headroom => "fits on a floor".to_string(),
        (Some(b), _) if b <= headroom => "fits".to_string(),
        (Some(_), _) => "does not fit".to_string(),
    }
}

fn identity_label(c: &Candidate) -> &'static str {
    match c.provenance {
        Provenance::Recorded => "reported",
        Provenance::Inferred { .. } => "inferred",
    }
}

/// Whether this row is the one the resolver picked.
///
/// Matched on the pair the decision actually carries. A `Wait` or a `Deny`
/// marks nothing: there is no coordinate to mark.
fn is_pick(c: &Candidate, d: &Decision) -> bool {
    match d {
        Decision::Ready { provider, provider_model_id, .. } => {
            c.provider.to_string() == *provider && c.provider_model_id == *provider_model_id
        }
        _ => false,
    }
}

pub fn render(model: &str, priced: &[Priced], machine: &Machine, reserve: u64, d: &Decision) -> String {
    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve);

    if priced.is_empty() {
        return format!("  no provider serves `{model}`\n");
    }

    let id_width = priced
        .iter()
        .map(|p| p.candidate.provider_model_id.chars().count())
        .max()
        .unwrap_or(8)
        .max(8);

    let mut out = String::new();
    out.push_str(&format!(
        "  provider   {:<id_width$}  {:>8}  {:<10} {:<15} {}\n",
        "model id", "price", "basis", "fit", "identity"
    ));
    for p in priced {
        out.push_str(&format!(
            "{} {:<10} {:<id_width$}  {:>8}  {:<10} {:<15} {}\n",
            if is_pick(&p.candidate, d) { ">" } else { " " },
            p.candidate.provider.to_string(),
            p.candidate.provider_model_id,
            p.estimate.bytes.map(human_bytes).unwrap_or_else(|| "?".to_string()),
            basis_label(&p.estimate),
            fit_label(p, headroom),
            identity_label(&p.candidate),
        ));
    }
    out.push_str(&format!(
        "  headroom   {}   ({} free \u{2212} {} reserve)\n",
        human_bytes(headroom),
        human_bytes(free),
        human_bytes(reserve)
    ));
    out.push_str(&match d {
        Decision::Ready { .. } => "  > is where resolve would send this request\n".to_string(),
        Decision::Wait { reason, .. } => format!("  resolve would wait: {reason}\n"),
        Decision::Deny { reason, .. } => format!("  resolve would refuse right now: {reason}\n"),
    });
    out
}

/// Delroy's contract, and deliberately not the ledger's schema: a consumer
/// that knows this document must not have to track the ledger's version.
pub fn render_json(
    model: &str,
    priced: &[Priced],
    machine: &Machine,
    reserve: u64,
    d: &Decision,
) -> String {
    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve);
    let rows: Vec<serde_json::Value> = priced
        .iter()
        .map(|p| {
            serde_json::json!({
                "provider": p.candidate.provider.to_string(),
                "provider_model_id": p.candidate.provider_model_id,
                "base_url": p.candidate.url,
                "resident": p.candidate.state == State::Loaded,
                "price_bytes": p.estimate.bytes,
                "basis": basis_label(&p.estimate),
                "samples": p.estimate.samples,
                "fit": fit_label(p, headroom),
                "identity": identity_label(&p.candidate),
                "pick": is_pick(&p.candidate, d),
            })
        })
        .collect();
    serde_json::json!({
        "schema": 1,
        "model": model,
        "headroom_bytes": headroom,
        "reserve_bytes": reserve,
        "candidates": rows,
        "decision": d,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind::{LlamaCpp, LmStudio, Vllm};

    const RESERVE: u64 = 8 * 1024 * 1024 * 1024;

    fn machine(free: u64) -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 25_769_803_776 - free,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        }
    }

    fn cand(kind: crate::provider::ProviderKind, recorded: bool) -> Candidate {
        Candidate {
            provider: kind,
            provider_model_id: format!("m-on-{kind}"),
            url: format!("http://127.0.0.1:{}", kind.default_port()),
            state: State::NotLoaded,
            artifact: Some("a".into()),
            provenance: if recorded {
                Provenance::Recorded
            } else {
                Provenance::Inferred { basis: "derived from LM Studio's store layout".into() }
            },
        }
    }

    fn priced(kind: crate::provider::ProviderKind, bytes: u64, basis: Basis) -> Priced {
        Priced {
            candidate: cand(kind, true),
            estimate: Estimate { bytes: Some(bytes), basis, samples: 3, spread_bytes: Some(0) },
        }
    }

    fn inferred(kind: crate::provider::ProviderKind, bytes: u64) -> Priced {
        Priced {
            candidate: cand(kind, false),
            estimate: Estimate {
                bytes: Some(bytes),
                basis: Basis::Measured,
                samples: 3,
                spread_bytes: Some(0),
            },
        }
    }

    fn ready(kind: crate::provider::ProviderKind) -> Decision {
        Decision::Ready {
            base_url: format!("http://127.0.0.1:{}", kind.default_port()),
            provider_model_id: format!("m-on-{kind}"),
            provider: kind.to_string(),
            resident: false,
            inferred: false,
        }
    }

    fn deny() -> Decision {
        Decision::Deny { reason: "no room".into(), alternatives: Vec::new() }
    }

    /// The whole point of the command: `resolve` computes this list and throws
    /// all but one row away.
    #[test]
    fn every_candidate_is_listed_not_only_the_chosen_one() {
        let out = render(
            "m",
            &[priced(LlamaCpp, 8_400_000_000, Basis::Measured), priced(Vllm, 4_700_000_000, Basis::Declared)],
            &machine(21 << 30),
            RESERVE,
            &ready(LlamaCpp),
        );
        assert!(out.contains("llamacpp"), "{out}");
        assert!(out.contains("vllm"), "the unchosen candidate is the point: {out}");
    }

    /// A price is worthless without its rung. design.md section 5.
    #[test]
    fn each_candidate_carries_its_own_basis() {
        let out = render(
            "m",
            &[priced(LlamaCpp, 8_400_000_000, Basis::Measured), priced(Vllm, 4_700_000_000, Basis::Declared)],
            &machine(21 << 30),
            RESERVE,
            &ready(LlamaCpp),
        );
        assert!(out.contains("measured"), "{out}");
        assert!(out.contains("declared"), "{out}");
    }

    /// A candidate that cannot be admitted is still information: it tells the
    /// operator what would have to be freed.
    #[test]
    fn a_candidate_that_does_not_fit_is_marked_rather_than_dropped() {
        let out = render("m", &[priced(LlamaCpp, 20 << 30, Basis::Measured)], &machine(9 << 30), RESERVE, &deny());
        assert!(out.contains("llamacpp"), "{out}");
        assert!(out.contains("does not fit"), "{out}");
    }

    /// Slice 2's rule: an inferred edge may not silently drive a decision, so
    /// the row it produced has to say it was inferred.
    #[test]
    fn an_inferred_candidate_says_so() {
        let out = render("m", &[inferred(LmStudio, 8_900_000_000)], &machine(21 << 30), RESERVE, &ready(LmStudio));
        assert!(out.contains("inferred"), "{out}");
    }

    /// The invariant that keeps the menu honest: the marked row is whatever
    /// `decide` returned. Here the cheaper candidate is *not* the pick, and
    /// the renderer must say so rather than sorting by price.
    #[test]
    fn the_marked_candidate_is_the_one_resolve_would_pick() {
        let out = render(
            "m",
            &[priced(LlamaCpp, 4_000_000_000, Basis::Measured), priced(Vllm, 8_000_000_000, Basis::Measured)],
            &machine(21 << 30),
            RESERVE,
            &ready(Vllm),
        );
        let marked = out
            .lines()
            .find(|l| l.starts_with('>'))
            .expect("exactly one row is marked");
        assert!(marked.contains("vllm"), "cheaper is not the rule; decide is: {out}");
    }

    /// Nothing to compare is a sentence, not an empty table.
    #[test]
    fn a_model_nothing_serves_says_so_rather_than_printing_an_empty_table() {
        let out = render("ghost", &[], &machine(21 << 30), RESERVE, &deny());
        assert!(out.contains("no provider serves `ghost`"), "{out}");
        assert_eq!(out.lines().count(), 1, "{out}");
    }

    /// A `Wait` and a `Deny` have no coordinate, so they mark no row. An
    /// unmarked table is correct; a table marking a row the resolver would not
    /// return is not.
    #[test]
    fn a_refusal_marks_no_row_at_all() {
        let out = render("m", &[priced(LlamaCpp, 20 << 30, Basis::Measured)], &machine(9 << 30), RESERVE, &deny());
        assert!(!out.lines().any(|l| l.starts_with('>')), "{out}");
    }

    /// The JSON is a separate contract with its own schema -- the mistake
    /// slice 3 found when the ledger's version leaked into a consumer's.
    #[test]
    fn the_json_carries_its_own_schema_and_every_candidate() {
        let json = render_json(
            "m",
            &[priced(LlamaCpp, 8_400_000_000, Basis::Measured), priced(Vllm, 4_700_000_000, Basis::Declared)],
            &machine(21 << 30),
            RESERVE,
            &ready(LlamaCpp),
        );
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["schema"], 1);
        assert_eq!(v["candidates"].as_array().unwrap().len(), 2);
        assert_eq!(v["candidates"][0]["pick"], true);
        assert_eq!(v["candidates"][1]["pick"], false);
    }
}
