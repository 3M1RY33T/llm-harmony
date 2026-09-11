//! Can these models be resident **at once**, and under which assignment?
//!
//! Admission generalised from one item to a set. `estimate` and `resolve` ask
//! the question of a single model, and answer it against the headroom that
//! exists at the moment they are asked. A set has two dimensions neither of
//! them has:
//!
//! 1. **It is a selection, not only a sum.** The same logical model exists as
//!    several artifacts at several prices -- `resolve::identity::candidates`
//!    already enumerates them -- so *does this set fit* is an assignment
//!    problem over the candidate lists, and the candidate lists exist only
//!    because of the identity graph.
//! 2. **Sequential admission cannot answer it.** Three `load` calls each admit
//!    against a ledger the previous one changed, so the third is refused after
//!    the first two have already moved. Answering before anything moves is the
//!    whole point.
//!
//! ## What is charged, and what is not
//!
//! ```text
//! charged  = Σ estimate(m)  for every m not already resident
//!          + Σ baseline(p)  for every provider not already running
//! headroom = total − used − reserve
//! ```
//!
//! Two subtractions matter more than they look:
//!
//! - **A resident model is charged nothing.** Its bytes are in `used` already,
//!   and `decide` has always treated residency as needing no admission ("it is
//!   costing what it costs whether we answer or not"). Charging it again would
//!   refuse sets that are already satisfied.
//! - **A provider's baseline is charged once**, however many of its models the
//!   assignment picks, and not at all if it is already running. Summing
//!   per-model footprints counts LM Studio's 0.6 GB of Electron once per model
//!   it holds.
//!
//! ## Why a count ceiling can refuse what the bytes allow
//!
//! llama.cpp runs `--models-max 1` on this machine. No arithmetic over memory
//! gets two GGUFs into a router that holds one, so the search filters
//! assignments by each provider's own count ceiling before pricing them. Where
//! that ceiling is *unknown* -- Ollama's, which is unset and therefore its own
//! undocumented default -- the set is marked and admitted: exceeding a count
//! makes a provider evict one of its own models by its own policy, which is a
//! surprise rather than a wedged machine. The asymmetry that justifies
//! refusing on memory does not apply.

pub mod search;

use crate::budget::Budget;
use crate::estimate::baseline::Baseline;
use crate::estimate::estimator::{Basis, Estimate};
use crate::memory::Machine;
use crate::pins::Pins;
use crate::provider::{ProviderKind, State};
use crate::resolve::identity::Candidate;

/// One model, placed.
#[derive(Debug, Clone)]
pub struct Assigned {
    pub model: String,
    pub candidate: Candidate,
    pub estimate: Estimate,
    /// Already loaded on this provider, so it is charged nothing.
    pub resident: bool,
}

impl Assigned {
    /// What this row adds to the machine. Zero when it is already there.
    pub fn charged_bytes(&self) -> u64 {
        if self.resident {
            0
        } else {
            self.estimate.bytes.unwrap_or(0)
        }
    }
}

/// A provider's own overhead, and whether the assignment has to pay it.
#[derive(Debug, Clone)]
pub struct Overhead {
    pub provider: ProviderKind,
    pub bytes: u64,
    /// Already up, so its baseline is in `used` and is not charged.
    pub running: bool,
    /// The figure rests on too few idle readings. Carried so a plan that
    /// depends on it says so -- vLLM-MLX has two.
    pub thin: bool,
}

impl Overhead {
    pub fn charged_bytes(&self) -> u64 {
        if self.running {
            0
        } else {
            self.bytes
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Fits,
    /// One member could not be priced, so the set cannot be. `decide` refuses
    /// to admit on `Declared` or `Unknown`; a total that looked authoritative
    /// because two of its three terms were measured would be exactly the
    /// under-estimate this project biases against.
    Unpriced { model: String },
    /// The bytes do not fit, with the smallest set of members whose removal
    /// would make them fit.
    DoesNotFit { short_by_bytes: u64, smallest_drop: Vec<String> },
    /// A provider cannot hold that many models at once, whatever the bytes say.
    CeilingRefusal { provider: ProviderKind, allowed: u32, wanted: usize },
    /// No provider serves one of the named models.
    Unservable { model: String },
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub assigned: Vec<Assigned>,
    pub overheads: Vec<Overhead>,
    pub charged_bytes: u64,
    pub headroom_bytes: u64,
    pub verdict: Verdict,
    /// The next-best assignment and what it would have cost, so *why not the
    /// other one* is answerable. `None` when there was only one.
    pub runner_up: Option<(Vec<Assigned>, u64)>,
    /// Providers whose count ceiling could not be read, and so was not
    /// enforced. Marked rather than refused.
    pub unchecked_ceilings: Vec<ProviderKind>,
}

/// What one model costs on one candidate.
///
/// Supplied by the caller because the ladder lives in `main.rs`, shared with
/// `estimate` and `resolve` so the three verbs can never disagree -- the same
/// reason `actuate::run` takes a closure rather than pricing anything itself.
pub type Price<'a> = &'a dyn Fn(&Candidate, &str) -> Estimate;

/// Everything the planner needs that it will not read for itself.
pub struct Inputs<'a> {
    pub candidates: &'a dyn Fn(&str) -> Vec<Candidate>,
    pub price: Price<'a>,
    pub baselines: &'a [Baseline],
    pub budgets: &'a [Budget],
    pub running: &'a [ProviderKind],
    pub machine: &'a Machine,
    pub reserve_bytes: u64,
    pub pins: &'a Pins,
    pub now: u64,
}

fn baseline_for(baselines: &[Baseline], p: ProviderKind) -> Option<&Baseline> {
    baselines.iter().find(|b| b.provider == p)
}

fn count_limit_for(budgets: &[Budget], p: ProviderKind) -> Option<u32> {
    budgets
        .iter()
        .find(|b| b.provider == p)
        .and_then(crate::budget::count_limit)
}

/// Plan a set. Touches nothing, and cannot: there is no actuation in this
/// module, by construction, exactly as slices 1 and 2 could not act.
pub fn plan(models: &[String], i: &Inputs) -> Plan {
    let free = i.machine.total_bytes.saturating_sub(i.machine.used_bytes);
    let headroom = free.saturating_sub(i.reserve_bytes);

    // Every model's options, priced. An empty list at this stage is fatal for
    // the set: nothing serves it, and no assignment can.
    let mut options: Vec<(String, Vec<(Candidate, Estimate)>)> = Vec::new();
    for m in models {
        let cands = (i.candidates)(m);
        if cands.is_empty() {
            return Plan {
                assigned: Vec::new(),
                overheads: Vec::new(),
                charged_bytes: 0,
                headroom_bytes: headroom,
                verdict: Verdict::Unservable { model: m.clone() },
                runner_up: None,
                unchecked_ceilings: Vec::new(),
            };
        }
        let priced: Vec<(Candidate, Estimate)> =
            cands.into_iter().map(|c| { let e = (i.price)(&c, m); (c, e) }).collect();

        // The weakest rung rule, applied before any search: if no option for
        // this model carries an admissible figure, the set is unpriced and a
        // cheapest assignment would be a fiction.
        if priced.iter().all(|(c, e)| !admissible(c, e)) {
            return Plan {
                assigned: Vec::new(),
                overheads: Vec::new(),
                charged_bytes: 0,
                headroom_bytes: headroom,
                verdict: Verdict::Unpriced { model: m.clone() },
                runner_up: None,
                unchecked_ceilings: Vec::new(),
            };
        }
        options.push((m.clone(), priced));
    }

    let outcome = search::cheapest(&options, i);

    let unchecked: Vec<ProviderKind> = outcome
        .providers_used
        .iter()
        .copied()
        .filter(|p| count_limit_for(i.budgets, *p).is_none())
        .collect();

    match outcome.best {
        None => Plan {
            assigned: Vec::new(),
            overheads: Vec::new(),
            charged_bytes: 0,
            headroom_bytes: headroom,
            verdict: outcome
                .ceiling_refusal
                .map(|(provider, allowed, wanted)| Verdict::CeilingRefusal {
                    provider,
                    allowed,
                    wanted,
                })
                .unwrap_or(Verdict::Unpriced { model: models.join(", ") }),
            runner_up: None,
            unchecked_ceilings: unchecked,
        },
        Some((assigned, charged)) => {
            // `charged` is `search::cost`, which already includes the
            // overheads below. They are recomputed only to be shown, so the
            // number printed and the number ranked on are one function.
            let overheads = overheads_for(&assigned, i);
            let total = charged;
            let verdict = if total <= headroom {
                Verdict::Fits
            } else {
                let short = total - headroom;
                Verdict::DoesNotFit {
                    short_by_bytes: short,
                    smallest_drop: smallest_drop(&assigned, i.pins, i.now, short),
                }
            };
            Plan {
                assigned,
                overheads,
                charged_bytes: total,
                headroom_bytes: headroom,
                verdict,
                runner_up: outcome.runner_up,
                unchecked_ceilings: unchecked,
            }
        }
    }
}

/// Whether a figure may be admitted on, by `design.md` §5's ladder.
///
/// A resident candidate is admissible whatever its price: it is already there.
pub fn admissible(c: &Candidate, e: &Estimate) -> bool {
    if c.state == State::Loaded {
        return true;
    }
    matches!(e.basis, Basis::Measured | Basis::Computed) && e.bytes.is_some()
}

fn overheads_for(assigned: &[Assigned], i: &Inputs) -> Vec<Overhead> {
    let mut used: Vec<ProviderKind> = assigned.iter().map(|a| a.candidate.provider).collect();
    used.sort_by_key(|p| p.as_str());
    used.dedup();
    used.into_iter()
        .filter_map(|p| {
            baseline_for(i.baselines, p).map(|b| Overhead {
                provider: p,
                bytes: b.bytes,
                running: i.running.contains(&p),
                thin: b.thin,
            })
        })
        .collect()
}

/// The fewest members whose removal frees `short`.
///
/// Largest first, because dropping one 8 GB model beats dropping four small
/// ones and the caller asked what the *smallest* change is. A member that is
/// already resident frees nothing by being dropped -- it was never charged --
/// and one held by a pin or a lease is not on the table at all.
fn smallest_drop(assigned: &[Assigned], pins: &Pins, now: u64, short: u64) -> Vec<String> {
    let mut droppable: Vec<&Assigned> = assigned
        .iter()
        .filter(|a| a.charged_bytes() > 0)
        .filter(|a| pins.holds(a.candidate.provider, &a.candidate.provider_model_id, now).is_none())
        .collect();
    droppable.sort_by_key(|a| std::cmp::Reverse(a.charged_bytes()));
    let mut freed = 0u64;
    let mut out = Vec::new();
    for a in droppable {
        if freed >= short {
            break;
        }
        freed += a.charged_bytes();
        out.push(a.model.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Ceiling;
    use crate::inventory::artifact::Provenance;
    use crate::provider::ProviderKind::{LlamaCpp, LmStudio, Ollama, Vllm};

    const RESERVE: u64 = 8 * 1024 * 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    /// One way one model could be served, as a test says it.
    #[derive(Clone)]
    struct Opt {
        model: &'static str,
        provider: ProviderKind,
        bytes: Option<u64>,
        basis: Basis,
        resident: bool,
    }

    fn opt(model: &'static str, provider: ProviderKind, bytes: u64) -> Opt {
        Opt { model, provider, bytes: Some(bytes), basis: Basis::Measured, resident: false }
    }

    struct Bench {
        opts: Vec<Opt>,
        baselines: Vec<Baseline>,
        budgets: Vec<Budget>,
        running: Vec<ProviderKind>,
        free: u64,
        pins: Pins,
    }

    impl Bench {
        fn new(opts: Vec<Opt>) -> Bench {
            Bench {
                opts,
                baselines: Vec::new(),
                budgets: Vec::new(),
                running: ProviderKind::ALL.to_vec(),
                free: 21 * GB,
                pins: Pins::empty(),
            }
        }

        fn plan(&self, models: &[&str]) -> Plan {
            let machine = Machine {
                total_bytes: 25_769_803_776,
                used_bytes: 25_769_803_776 - self.free,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
            };
            let cands = |m: &str| -> Vec<Candidate> {
                self.opts
                    .iter()
                    .filter(|o| o.model == m)
                    .map(|o| Candidate {
                        provider: o.provider,
                        provider_model_id: format!("{}-on-{}", o.model, o.provider),
                        url: format!("http://127.0.0.1:{}", o.provider.default_port()),
                        state: if o.resident { State::Loaded } else { State::NotLoaded },
                        artifact: Some(o.model.into()),
                        provenance: Provenance::Recorded,
                    })
                    .collect()
            };
            let price = |c: &Candidate, m: &str| -> Estimate {
                match self.opts.iter().find(|o| o.model == m && o.provider == c.provider) {
                    Some(o) => Estimate {
                        bytes: o.bytes,
                        basis: o.basis,
                        samples: 3,
                        spread_bytes: Some(0),
                    },
                    None => Estimate::unknown(),
                }
            };
            let i = Inputs {
                candidates: &cands,
                price: &price,
                baselines: &self.baselines,
                budgets: &self.budgets,
                running: &self.running,
                machine: &machine,
                reserve_bytes: RESERVE,
                pins: &self.pins,
                now: 1_789_000_000,
            };
            plan(&models.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &i)
        }
    }

    fn baseline(provider: ProviderKind, bytes: u64, thin: bool) -> Baseline {
        Baseline { provider, bytes, p50_bytes: bytes, samples: if thin { 2 } else { 100 }, thin }
    }

    fn count(provider: ProviderKind, n: u32) -> Budget {
        Budget { provider, ceiling: Ceiling::Count(n), source: "test".into() }
    }

    #[test]
    fn a_set_that_fits_names_the_assignment_it_fits_under() {
        let b = Bench::new(vec![opt("a", LlamaCpp, 5 * GB), opt("e", Ollama, 1 * GB)]);
        let p = b.plan(&["a", "e"]);
        assert_eq!(p.verdict, Verdict::Fits);
        assert_eq!(p.assigned.len(), 2);
        assert_eq!(p.charged_bytes, 6 * GB);
    }

    /// The dimension one model does not have: the same logical model exists at
    /// several prices, so this is a selection as well as a sum.
    #[test]
    fn the_same_model_on_two_providers_is_priced_twice_and_the_cheaper_assignment_wins() {
        let b = Bench::new(vec![opt("a", LlamaCpp, 5 * GB), opt("a", Vllm, 8 * GB)]);
        let p = b.plan(&["a"]);
        assert_eq!(p.assigned[0].candidate.provider, LlamaCpp, "{:?}", p.assigned);
        assert_eq!(p.charged_bytes, 5 * GB);
        let (_, runner) = p.runner_up.expect("the rejected assignment is reportable");
        assert_eq!(runner, 8 * GB, "why not the other one has to be answerable");
    }

    /// The arithmetic the whole feature turns on: two models on one provider
    /// pay that provider's baseline once, not twice.
    #[test]
    fn a_providers_baseline_is_charged_once_however_many_of_its_models_are_chosen() {
        let mut b = Bench::new(vec![opt("a", LmStudio, 2 * GB), opt("c", LmStudio, 3 * GB)]);
        b.baselines = vec![baseline(LmStudio, GB, false)];
        b.running = vec![]; // nothing up, so the baseline is payable
        b.budgets = vec![count(LmStudio, 4)];
        let p = b.plan(&["a", "c"]);
        assert_eq!(p.charged_bytes, 2 * GB + 3 * GB + GB, "one baseline, not two");
        assert_eq!(p.overheads.len(), 1);
    }

    /// A resident model is in `used_bytes` already, and `decide` has always
    /// treated residency as needing no admission. Charging it again would
    /// refuse sets that are already satisfied.
    #[test]
    fn a_model_already_resident_is_not_charged_again() {
        let mut o = opt("a", LlamaCpp, 20 * GB);
        o.resident = true;
        let b = Bench::new(vec![o]);
        let p = b.plan(&["a"]);
        assert_eq!(p.charged_bytes, 0, "{:?}", p.verdict);
        assert_eq!(p.verdict, Verdict::Fits);
    }

    /// Same argument one level up: a running provider's overhead is spent.
    #[test]
    fn a_running_providers_baseline_is_not_charged_again() {
        let mut b = Bench::new(vec![opt("a", LmStudio, 2 * GB)]);
        b.baselines = vec![baseline(LmStudio, GB, false)];
        b.running = vec![LmStudio];
        let p = b.plan(&["a"]);
        assert_eq!(p.charged_bytes, 2 * GB);
        assert!(!p.overheads[0].charged_bytes() > 0);
    }

    /// design.md section 5 and decide's own rule: one unpriced member makes
    /// the set unpriced. A total that looked authoritative because two of its
    /// three terms were measured is the under-estimate to avoid.
    #[test]
    fn a_set_holding_one_unpriced_model_is_an_unpriced_set() {
        let unpriced = Opt {
            model: "u",
            provider: Vllm,
            bytes: None,
            basis: Basis::Unknown,
            resident: false,
        };
        let b = Bench::new(vec![opt("a", LlamaCpp, 2 * GB), unpriced]);
        assert_eq!(b.plan(&["a", "u"]).verdict, Verdict::Unpriced { model: "u".into() });
    }

    /// A declared figure is a weights-only floor. `decide` will not admit on
    /// one, so neither will a set.
    #[test]
    fn a_declared_figure_does_not_price_a_set_either() {
        let declared = Opt {
            model: "d",
            provider: Ollama,
            bytes: Some(4 * GB),
            basis: Basis::Declared,
            resident: false,
        };
        let b = Bench::new(vec![declared]);
        assert_eq!(b.plan(&["d"]).verdict, Verdict::Unpriced { model: "d".into() });
    }

    #[test]
    fn a_set_that_does_not_fit_names_the_smallest_drop_that_would_fit() {
        let b = Bench::new(vec![
            opt("big", LlamaCpp, 10 * GB),
            opt("mid", Vllm, 4 * GB),
            opt("small", Ollama, GB),
        ]);
        let p = b.plan(&["big", "mid", "small"]);
        match p.verdict {
            Verdict::DoesNotFit { smallest_drop, .. } => {
                assert_eq!(smallest_drop, vec!["big".to_string()], "largest first, fewest members");
            }
            other => panic!("expected a byte refusal, got {other:?}"),
        }
    }

    /// llama.cpp runs `--models-max 1` here. No arithmetic over bytes gets two
    /// GGUFs into a router that holds one.
    #[test]
    fn two_models_on_a_provider_that_allows_one_is_a_ceiling_refusal_not_a_byte_refusal() {
        let mut b = Bench::new(vec![opt("a", LlamaCpp, GB), opt("c", LlamaCpp, GB)]);
        b.budgets = vec![count(LlamaCpp, 1)];
        let p = b.plan(&["a", "c"]);
        assert_eq!(
            p.verdict,
            Verdict::CeilingRefusal { provider: LlamaCpp, allowed: 1, wanted: 2 },
            "the bytes fit; the ceiling does not"
        );
    }

    /// An unknown count ceiling is not an unknown memory cost: exceeding it
    /// makes the provider evict by its own policy. Mark it, proceed.
    #[test]
    fn an_unknown_count_ceiling_is_marked_and_does_not_refuse_the_set() {
        let b = Bench::new(vec![opt("a", Ollama, GB), opt("c", Ollama, GB)]);
        let p = b.plan(&["a", "c"]);
        assert_eq!(p.verdict, Verdict::Fits);
        assert!(p.unchecked_ceilings.contains(&Ollama), "{:?}", p.unchecked_ceilings);
    }

    /// A ceiling that allows the set does not refuse it, and does not appear
    /// as unchecked either.
    #[test]
    fn a_ceiling_that_allows_the_set_is_neither_a_refusal_nor_unchecked() {
        let mut b = Bench::new(vec![opt("a", LlamaCpp, GB), opt("c", LlamaCpp, GB)]);
        b.budgets = vec![count(LlamaCpp, 2)];
        let p = b.plan(&["a", "c"]);
        assert_eq!(p.verdict, Verdict::Fits);
        assert!(p.unchecked_ceilings.is_empty());
    }

    /// A thin baseline is carried into the plan rather than smoothed over:
    /// vLLM-MLX's rests on two readings.
    #[test]
    fn a_thin_baseline_is_reported_on_the_plan_that_used_it() {
        let mut b = Bench::new(vec![opt("a", Vllm, 4 * GB)]);
        b.baselines = vec![baseline(Vllm, 332 * 1024 * 1024, true)];
        b.running = vec![];
        assert!(b.plan(&["a"]).overheads.iter().any(|o| o.thin));
    }

    #[test]
    fn a_model_no_provider_serves_makes_the_set_unservable() {
        let b = Bench::new(vec![opt("a", LlamaCpp, GB)]);
        assert_eq!(b.plan(&["a", "ghost"]).verdict, Verdict::Unservable { model: "ghost".into() });
    }

    /// A pinned or leased member is not a drop candidate: harmony may not
    /// propose freeing what it has been told to protect.
    #[test]
    fn a_held_member_is_never_proposed_as_the_drop() {
        let mut b = Bench::new(vec![opt("big", LlamaCpp, 10 * GB), opt("mid", Vllm, 9 * GB)]);
        b.pins.add(crate::pins::Pin {
            provider: LlamaCpp,
            model: "big-on-llamacpp".into(),
            at: 1,
            note: None,
            owner: None,
            expires_at: None,
        });
        let p = b.plan(&["big", "mid"]);
        match p.verdict {
            Verdict::DoesNotFit { smallest_drop, .. } => {
                assert!(!smallest_drop.contains(&"big".to_string()), "{smallest_drop:?}");
                assert_eq!(smallest_drop, vec!["mid".to_string()]);
            }
            other => panic!("expected a byte refusal, got {other:?}"),
        }
    }
}
