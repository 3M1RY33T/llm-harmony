//! The cheapest assignment of models to providers, found exhaustively.
//!
//! Exhaustive rather than greedy, for one reason that matters more than
//! performance: *why not the other one* has to be answerable. A greedy walk
//! can say what it chose; a full enumeration can also say what it rejected and
//! what that would have cost, which is the difference between a plan and an
//! assertion.
//!
//! The space is small by construction -- the roadmap's bound is six models
//! with four candidates each -- and [`MAX_ASSIGNMENTS`] is a backstop, not a
//! sampling strategy. If it is ever reached the plan says so, because a silent
//! truncation reads as *nothing fits* when the truth is *we stopped looking*.
//!
//! ## What the cost function counts
//!
//! One function, [`cost`], used both to rank assignments and to report the
//! winner's total, so the number printed is the number ranked on. It charges
//! nothing for a model that is already resident and nothing for a provider
//! that is already running -- both are in `machine.used_bytes` already.
//!
//! ## What the ceiling filter does not model
//!
//! A provider's count ceiling is checked against the **set's own** members. A
//! model already resident there also occupies a slot, and loading past the
//! ceiling makes the provider evict by its own policy -- llama.cpp's LRU, on
//! this machine, with `--models-max 1`. That eviction is the provider's
//! business and harmony does not model it, so `fit` answers "can these
//! coexist" and not "and will nothing else be disturbed."

use super::{Assigned, Inputs, Overhead};
use crate::estimate::estimator::Estimate;
use crate::provider::{ProviderKind, State};
use crate::resolve::identity::Candidate;

/// A backstop against a pathological input, never a sampling strategy.
///
/// Six models with four candidates each is 4096, which is the roadmap's stated
/// bound and takes microseconds. Beyond it the plan reports that the search was
/// capped.
pub const MAX_ASSIGNMENTS: usize = 4096;

pub struct Outcome {
    /// The cheapest assignment and its total charged bytes.
    pub best: Option<(Vec<Assigned>, u64)>,
    /// The next cheapest, for the *not chosen* line.
    pub runner_up: Option<(Vec<Assigned>, u64)>,
    /// Providers the winning assignment places anything on.
    pub providers_used: Vec<ProviderKind>,
    /// Set when every assignment was rejected by one provider's count ceiling,
    /// which is a different refusal from "does not fit".
    pub ceiling_refusal: Option<(ProviderKind, u32, usize)>,
    /// The search stopped early. Never silently.
    pub capped: bool,
}

/// Total bytes an assignment adds to the machine: models plus provider
/// overheads, each charged once and only when not already paid.
pub fn cost(assignment: &[Assigned], i: &Inputs) -> u64 {
    let models: u64 = assignment.iter().map(Assigned::charged_bytes).sum();
    let overheads: u64 = super::overheads_for(assignment, i)
        .iter()
        .map(Overhead::charged_bytes)
        .sum();
    models + overheads
}

fn admissible_options(
    priced: &[(Candidate, Estimate)],
) -> impl Iterator<Item = &(Candidate, Estimate)> {
    priced.iter().filter(|(c, e)| super::admissible(c, e))
}

/// Every provider's count ceiling respected, or the one that refused.
fn within_ceilings(
    assignment: &[Assigned],
    i: &Inputs,
) -> Result<(), (ProviderKind, u32, usize)> {
    for p in ProviderKind::ALL {
        let wanted = assignment.iter().filter(|a| a.candidate.provider == p).count();
        if wanted <= 1 {
            continue;
        }
        if let Some(limit) = i
            .budgets
            .iter()
            .find(|b| b.provider == p)
            .and_then(crate::budget::count_limit)
        {
            if wanted > limit as usize {
                return Err((p, limit, wanted));
            }
        }
    }
    Ok(())
}

/// Enumerate, filter, price, and keep the best two.
pub fn cheapest(options: &[(String, Vec<(Candidate, Estimate)>)], i: &Inputs) -> Outcome {
    let widths: Vec<usize> = options
        .iter()
        .map(|(_, priced)| admissible_options(priced).count())
        .collect();
    let total: usize = widths.iter().product();

    let mut out = Outcome {
        best: None,
        runner_up: None,
        providers_used: Vec::new(),
        ceiling_refusal: None,
        capped: total > MAX_ASSIGNMENTS,
    };
    if widths.iter().any(|w| *w == 0) {
        return out;
    }

    let limit = total.min(MAX_ASSIGNMENTS);
    let mut refusal: Option<(ProviderKind, u32, usize)> = None;

    for n in 0..limit {
        // Mixed-radix decode: index n picks one admissible option per model.
        let mut rest = n;
        let mut assignment: Vec<Assigned> = Vec::with_capacity(options.len());
        for (slot, (model, priced)) in options.iter().enumerate() {
            let w = widths[slot];
            let pick = rest % w;
            rest /= w;
            let (c, e) = admissible_options(priced).nth(pick).expect("width counted above");
            assignment.push(Assigned {
                model: model.clone(),
                candidate: c.clone(),
                estimate: e.clone(),
                resident: c.state == State::Loaded,
            });
        }

        if let Err(r) = within_ceilings(&assignment, i) {
            // Keep the first ceiling that refused, so a set that no assignment
            // satisfies can say which provider stood in the way rather than
            // reporting a byte shortfall it never reached.
            refusal.get_or_insert(r);
            continue;
        }

        let c = cost(&assignment, i);
        match &out.best {
            Some((_, best_cost)) if *best_cost <= c => {
                if out.runner_up.as_ref().is_none_or(|(_, ru)| c < *ru) {
                    out.runner_up = Some((assignment, c));
                }
            }
            Some(_) => {
                out.runner_up = out.best.take();
                out.best = Some((assignment, c));
            }
            None => out.best = Some((assignment, c)),
        }
    }

    if out.best.is_none() {
        out.ceiling_refusal = refusal;
    }
    if let Some((a, _)) = &out.best {
        let mut used: Vec<ProviderKind> = a.iter().map(|x| x.candidate.provider).collect();
        used.sort_by_key(|p| p.as_str());
        used.dedup();
        out.providers_used = used;
    }
    out
}
