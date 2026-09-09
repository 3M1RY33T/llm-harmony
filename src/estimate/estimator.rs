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
    /// Never seen. Not a number.
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

/// What this model costs, if it has been seen.
///
/// `context_tokens: None` asks about the worst case seen for the model at any
/// window. A specific window matches only that window: KV cache scales with
/// context, and interpolating between two windows would produce a number
/// nothing measured.
pub fn for_model(corpus: &[Observation], model: &str, context_tokens: Option<u32>) -> Estimate {
    let measurements: Vec<u64> = corpus
        .iter()
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
}
