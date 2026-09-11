//! What a provider costs with nothing loaded.
//!
//! `estimate` never needed this: one model means one baseline, and the
//! model-bearing process dwarfs it (8.39 GB against 0.28 GB, measured
//! 2026-09-09), so per-process attribution answers the question without
//! subtracting anything. A *set* is different. Summing per-model footprints
//! counts each provider's own overhead once per model it holds, and LM Studio's
//! overhead is 0.6 GB across five Electron processes -- large enough to decide
//! a set on a 24 GB machine.
//!
//! ## Why p99 and not the maximum
//!
//! `architecture.md` §2 biases everything towards over-estimating, which reads
//! as an argument for the largest reading ever seen. Measured 2026-09-11, that
//! is the wrong statistic: LM Studio's largest idle reading moved from 637 MiB
//! to 788 MiB inside an hour of recording, on a single sample, while p95 sat
//! still at 571 MiB. The tail is a handful of isolated spikes rather than a
//! distribution, and it grows with the corpus -- so a baseline taken from the
//! maximum drifts upward forever and never converges.
//!
//! Over-estimating against a distribution buys safety. Over-estimating against
//! an unbounded tail buys an arms race with whatever Electron did once.

use crate::provider::ProviderKind;
use crate::record::Observation;

/// Below this many idle readings, the figure is reported but marked.
///
/// vLLM-MLX is the case this exists for: it has two, and that is structural
/// rather than bad luck -- with no model-level unload it is only ever idle
/// between process start and its first inference request, so the state a
/// recorder has to catch is a short window rather than a steady condition.
pub const THIN_SAMPLES: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Baseline {
    pub provider: ProviderKind,
    /// The figure to charge: p99 of the idle readings.
    pub bytes: u64,
    /// The middle of the distribution, carried so a report can show how far
    /// the charged figure sits above the usual case.
    pub p50_bytes: u64,
    pub samples: usize,
    /// Fewer than [`THIN_SAMPLES`] readings. A plan that rests on a thin
    /// baseline has to say so; it is not refused, because refusing would make
    /// a provider that cannot idle unusable rather than merely unmeasured.
    pub thin: bool,
}

/// Nearest-rank percentile, 1-indexed and clamped.
///
/// `p99` of a hundred readings is the ninety-ninth of them, which leaves the
/// single largest out of the figure -- the whole point. Deliberately not
/// interpolated: an interpolated percentile invents a value that was never
/// observed, and this module exists to stop doing that.
fn percentile(sorted: &[u64], q: f64) -> u64 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    let rank = (q * n as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(n - 1)]
}

/// The baseline for one provider, or `None` if it has never been seen idle.
///
/// An observation counts only when the provider held nothing *and* harmony
/// could attribute the reading. An unattributable reading is not evidence:
/// `record.rs` sets that flag precisely because a footprint it cannot assign
/// is a number without a subject.
pub fn from_corpus(corpus: &[Observation], provider: ProviderKind) -> Option<Baseline> {
    let mut idle: Vec<u64> = corpus
        .iter()
        .filter(|o| o.provider == provider && o.loaded == 0 && o.attributable)
        .map(|o| o.footprint_bytes)
        .collect();
    if idle.is_empty() {
        return None;
    }
    idle.sort_unstable();
    Some(Baseline {
        provider,
        bytes: percentile(&idle, 0.99),
        p50_bytes: percentile(&idle, 0.50),
        samples: idle.len(),
        thin: idle.len() < THIN_SAMPLES,
    })
}

/// Every provider that has ever been observed idle.
pub fn all(corpus: &[Observation]) -> Vec<Baseline> {
    ProviderKind::ALL.into_iter().filter_map(|k| from_corpus(corpus, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind::{LmStudio, Vllm};
    use crate::record::OBSERVATION_SCHEMA;

    fn obs(provider: ProviderKind, loaded: usize, footprint: u64) -> Observation {
        Observation {
            schema: OBSERVATION_SCHEMA,
            at: 1_789_000_000,
            provider,
            loaded,
            model: None,
            context_tokens: None,
            attributable: true,
            footprint_bytes: footprint,
            phys_footprint_bytes: Some(footprint),
            rss_bytes: Some(footprint),
            processes: Vec::new(),
            machine_total_bytes: 25_769_803_776,
            machine_used_bytes: 10 << 30,
            swap_used_bytes: 0,
        }
    }

    const MIB: u64 = 1024 * 1024;

    /// Only a reading with nothing loaded prices the provider itself.
    #[test]
    fn a_reading_with_a_model_loaded_is_not_a_baseline() {
        let c = vec![obs(LmStudio, 1, 9 << 30), obs(LmStudio, 0, 560 * MIB)];
        let b = from_corpus(&c, LmStudio).unwrap();
        assert_eq!(b.samples, 1);
        assert_eq!(b.bytes, 560 * MIB);
    }

    /// Measured 2026-09-11: LM Studio's largest idle reading moved 637 -> 788
    /// MiB inside an hour, on one sample, while p95 held at 571. The maximum
    /// is a tail that grows with the corpus; p99 is a statistic.
    #[test]
    fn the_figure_is_p99_and_not_the_maximum() {
        let mut c: Vec<Observation> = (0..99).map(|_| obs(LmStudio, 0, 560 * MIB)).collect();
        c.push(obs(LmStudio, 0, 788 * MIB));
        let b = from_corpus(&c, LmStudio).unwrap();
        assert_eq!(b.bytes, 560 * MIB, "the lone spike must not set the figure");
        assert!(b.bytes < 788 * MIB);
    }

    /// It is still an over-estimate: p99 sits above the middle of the
    /// distribution, which is the direction design.md section 8 requires.
    #[test]
    fn the_figure_is_never_below_the_median() {
        let mut c: Vec<Observation> = (0..50).map(|_| obs(LmStudio, 0, 537 * MIB)).collect();
        c.extend((0..50).map(|_| obs(LmStudio, 0, 571 * MIB)));
        let b = from_corpus(&c, LmStudio).unwrap();
        assert!(b.bytes >= b.p50_bytes, "{b:?}");
    }

    /// vLLM-MLX has two idle readings and that is structural: with no
    /// model-level unload it is idle only between start and first request. A
    /// figure resting on two samples has to say so.
    #[test]
    fn a_provider_with_too_few_idle_readings_is_marked_thin() {
        let c = vec![obs(Vllm, 0, 332 * MIB), obs(Vllm, 0, 331 * MIB)];
        let b = from_corpus(&c, Vllm).unwrap();
        assert!(b.thin);
        assert_eq!(b.samples, 2);
    }

    #[test]
    fn a_sufficiently_observed_provider_is_not_marked_thin() {
        let c: Vec<Observation> = (0..THIN_SAMPLES).map(|_| obs(LmStudio, 0, 537 * MIB)).collect();
        assert!(!from_corpus(&c, LmStudio).unwrap().thin);
    }

    /// A provider that has never idled yields nothing rather than a guess --
    /// the same posture the estimator takes for a shape it has never seen.
    #[test]
    fn a_provider_never_observed_idle_has_no_baseline() {
        assert!(from_corpus(&[obs(Vllm, 1, 5 << 30)], Vllm).is_none());
    }

    /// An observation harmony could not attribute is a number without a
    /// subject, and `record.rs` flags it for exactly this reason.
    #[test]
    fn an_unattributable_reading_is_skipped() {
        let mut o = obs(LmStudio, 0, 560 * MIB);
        o.attributable = false;
        assert!(from_corpus(&[o], LmStudio).is_none());
    }

    /// One provider's readings never price another's.
    #[test]
    fn baselines_do_not_leak_between_providers() {
        let c = vec![obs(LmStudio, 0, 560 * MIB), obs(Vllm, 0, 332 * MIB)];
        assert_eq!(from_corpus(&c, Vllm).unwrap().bytes, 332 * MIB);
        assert_eq!(all(&c).len(), 2);
    }
}
