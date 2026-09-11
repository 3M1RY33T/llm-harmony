use crate::estimate::attribute::model_bytes;
use crate::record::Observation;

/// How an estimate was reached. `design.md` section 5 ranks measured over
/// computed over declared; slice 3 implements the first and refuses the rest
/// rather than pretending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Basis {
    /// Real observations of this exact shape.
    Measured,
    /// Derived from the artifact's own geometry: weights, plus the KV cache at
    /// the requested window, plus a margin. Bias-high and never observed.
    ///
    /// Ranked below measurement and far above `Declared`, because unlike a
    /// declared figure it *moves with the context window* -- which is the term
    /// that decides whether a load fits.
    Computed,
    /// A figure the provider or the disk ledger declares -- vLLM-MLX's
    /// `memory_gb`, Ollama's `size`, or the artifact's size on disk.
    ///
    /// **A floor, never a peak.** It covers weights only: vLLM-MLX's own
    /// startup log says so, and `docs/design.md` section 5 makes the same
    /// point -- the KV cache and activations are the part that actually
    /// kills you. Admitting on a declared figure is admitting on an
    /// under-estimate, which is why it is ranked below measurement and
    /// reported as what it is.
    Declared,
    /// Never seen and nothing declared. Not a number.
    Unknown,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Estimate {
    pub bytes: Option<u64>,
    pub basis: Basis,
    pub samples: usize,
    /// Largest minus smallest measurement. A wide spread is a warning that the
    /// shape is not as fixed as it looks.
    pub spread_bytes: Option<u64>,
}

impl Estimate {
    fn unknown() -> Estimate {
        Estimate { bytes: None, basis: Basis::Unknown, samples: 0, spread_bytes: None }
    }
}

/// The estimate ladder: measured, then computed, then declared.
///
/// `design.md` §5's ranking, now that all three rungs exist. Slice 4 had only
/// the first and the last, which meant nearly every model was priced by a
/// weights-only floor -- see `Basis::Declared`.
///
/// A declared figure is still *returned*, because saying "about 9 GB, from the
/// file size" beats saying nothing. It is simply not something `decide` will
/// admit on, and the label is how the caller can tell.
pub fn ladder(measured: Estimate, computed: Estimate, declared_bytes: Option<u64>) -> Estimate {
    if !matches!(measured.basis, Basis::Unknown) {
        return measured;
    }
    if !matches!(computed.basis, Basis::Unknown) {
        return computed;
    }
    with_declared(measured, declared_bytes)
}

/// Fall back to a declared figure when nothing better exists.
///
/// Kept separate from `ladder` so the "labelled but not admissible" rule has
/// one home. A declared floor is worse than a measurement, worse than a
/// computation, and better than silence -- provided it is labelled.
pub fn with_declared(measured: Estimate, declared_bytes: Option<u64>) -> Estimate {
    if !matches!(measured.basis, Basis::Unknown) {
        return measured;
    }
    match declared_bytes {
        Some(b) => Estimate { bytes: Some(b), basis: Basis::Declared, samples: 0, spread_bytes: None },
        None => measured,
    }
}

/// What this model costs, if it has been seen.
///
/// `context_tokens: None` asks about the worst case seen for the model at any
/// window. A specific window matches only that window: KV cache scales with
/// context, and interpolating between two windows would produce a number
/// nothing measured.
/// Swap in use above which an observation stops being a measurement.
///
/// A resident-set reading taken while the machine is paging records what
/// survived eviction, not what the model wanted -- so it is an **under-count**,
/// and an under-count that outranks a bias-high computation is exactly
/// backwards for the one failure that costs the machine.
///
/// Measured 2026-09-11: every observation in this machine's corpus was taken
/// with 5.0-8.8 GB of swap in use, and each reported a 14B at a 40,960-token
/// window as costing 9.6 GB against 9.0 GB of weights -- a KV cache of
/// approximately zero, which is not physically possible. See
/// `docs/field-notes.md`, *a measurement taken while swapping is not a
/// measurement*.
///
/// 1 GiB rather than zero because macOS keeps some swap allocated in ordinary
/// operation; this is a proxy for "the machine was under real pressure", and a
/// crude one. Pageout deltas would be the honest signal and nothing records
/// them yet.
pub const TRUSTWORTHY_SWAP_CEILING_BYTES: u64 = 1024 * 1024 * 1024;

/// Was the machine calm enough for this reading to mean anything?
pub fn is_trustworthy(o: &Observation) -> bool {
    o.swap_used_bytes <= TRUSTWORTHY_SWAP_CEILING_BYTES
}

pub fn for_model(corpus: &[Observation], model: &str, context_tokens: Option<u32>) -> Estimate {
    let measurements: Vec<u64> = corpus
        .iter()
        .filter(|o| is_trustworthy(o))
        .filter(|o| o.model.as_deref() == Some(model))
        .filter(|o| match context_tokens {
            Some(want) => o.context_tokens == Some(want),
            None => true,
        })
        .filter_map(model_bytes)
        .collect();

    if measurements.is_empty() {
        return Estimate::unknown();
    }
    let max = *measurements.iter().max().expect("non-empty");
    let min = *measurements.iter().min().expect("non-empty");
    Estimate {
        // The high-water mark. design.md section 8: over-estimating wastes
        // capacity, under-estimating wedges the machine. Bias here.
        bytes: Some(max),
        basis: Basis::Measured,
        samples: measurements.len(),
        spread_bytes: Some(max - min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::fixtures;

    #[test]
    fn a_measured_shape_reports_its_measurement() {
        let corpus = vec![
            fixtures::loaded("qwen3-14b", 40960, 9_000_000_000),
            fixtures::loaded("qwen3-14b", 40960, 9_100_000_000),
        ];
        let e = for_model(&corpus, "qwen3-14b", Some(40960));
        assert!(matches!(e.basis, Basis::Measured));
        assert_eq!(e.samples, 2);
        // The high-water mark, not the mean: admitting by the average of two
        // readings admits below the larger one, and under-estimating is the
        // failure that wedges the machine.
        assert_eq!(e.bytes, Some(9_100_000_000));
        assert_eq!(e.spread_bytes, Some(100_000_000));
    }

    #[test]
    fn an_unseen_shape_is_unknown_never_interpolated() {
        let corpus = vec![fixtures::loaded("qwen3-14b", 40960, 9_000_000_000)];
        let e = for_model(&corpus, "qwen3-14b", Some(8192));
        assert!(matches!(e.basis, Basis::Unknown));
        assert_eq!(e.bytes, None, "8k is cheaper than 40k, but by how much is not known");
    }

    #[test]
    fn an_unseen_model_is_unknown() {
        let corpus = vec![fixtures::loaded("qwen3-14b", 40960, 9_000_000_000)];
        assert!(matches!(for_model(&corpus, "llama-3.2-1b", Some(8192)).basis, Basis::Unknown));
    }

    /// Asking without a context is asking about the worst case seen.
    #[test]
    fn omitting_the_context_takes_the_largest_measurement_for_that_model() {
        let corpus = vec![
            fixtures::loaded("qwen3-14b", 8192, 7_000_000_000),
            fixtures::loaded("qwen3-14b", 40960, 9_000_000_000),
        ];
        assert_eq!(for_model(&corpus, "qwen3-14b", None).bytes, Some(9_000_000_000));
    }

    #[test]
    fn unattributable_observations_are_not_measurements() {
        let corpus = vec![fixtures::two_loaded("qwen3-14b", 40960, 12_000_000_000)];
        assert!(matches!(for_model(&corpus, "qwen3-14b", Some(40960)).basis, Basis::Unknown));
    }

    fn of(basis: Basis, bytes: u64) -> Estimate {
        Estimate { bytes: Some(bytes), basis, samples: 0, spread_bytes: None }
    }

    fn nothing() -> Estimate {
        Estimate { bytes: None, basis: Basis::Unknown, samples: 0, spread_bytes: None }
    }

    /// Measurement wins even when a computation exists: it is the only source
    /// that can be right about this machine.
    #[test]
    fn the_ladder_prefers_a_measurement_to_a_computation() {
        let e = ladder(of(Basis::Measured, 1), of(Basis::Computed, 2), Some(3));
        assert_eq!((e.basis, e.bytes), (Basis::Measured, Some(1)));
    }

    #[test]
    fn the_ladder_prefers_a_computation_to_a_declared_floor() {
        let e = ladder(nothing(), of(Basis::Computed, 2), Some(3));
        assert_eq!((e.basis, e.bytes), (Basis::Computed, Some(2)));
    }

    /// The floor is still reported when it is all there is -- labelled, so
    /// `decide` can refuse to admit on it while the operator still sees a
    /// number.
    #[test]
    fn the_ladder_still_reports_a_declared_floor_when_nothing_else_exists() {
        let e = ladder(nothing(), nothing(), Some(3));
        assert_eq!((e.basis, e.bytes), (Basis::Declared, Some(3)));
    }

    #[test]
    fn the_ladder_reports_unknown_when_there_is_nothing_at_all() {
        assert_eq!(ladder(nothing(), nothing(), None).basis, Basis::Unknown);
    }

    /// The finding that reordered this ladder's trust, 2026-09-11.
    ///
    /// An observation taken under swap pressure is not a measurement. Keeping
    /// it would let a depressed figure outrank the bias-high computation that
    /// replaced it, which inverts the safety property the whole slice exists
    /// to provide.
    #[test]
    fn an_observation_taken_while_swapping_is_not_a_measurement() {
        let calm = fixtures::loaded("m", 40_960, 9_600_000_000);
        let paging = fixtures::while_swapping("m", 40_960, 9_600_000_000, 5_000_000_000);

        assert!(is_trustworthy(&calm));
        assert!(!is_trustworthy(&paging));

        assert_eq!(for_model(&[paging], "m", Some(40_960)).basis, Basis::Unknown);
        assert_eq!(for_model(&[calm], "m", Some(40_960)).basis, Basis::Measured);
    }

    /// Filtering must not silently drop the good readings alongside the bad.
    #[test]
    fn a_calm_observation_survives_beside_a_paging_one() {
        let corpus = vec![
            fixtures::while_swapping("m", 40_960, 12_000_000_000, 6_000_000_000),
            fixtures::loaded("m", 40_960, 9_600_000_000),
        ];
        let e = for_model(&corpus, "m", Some(40_960));
        assert_eq!(e.samples, 1, "only the calm reading counts");
        assert_eq!(e.bytes, Some(9_600_000_000), "and the paging one cannot inflate it either");
    }
}
