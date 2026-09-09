use crate::record::Observation;

/// What the resident model costs, from one observation.
///
/// The rule is the largest process in the provider's tree. A provider mixes a
/// shell with a backend, and the backend holding the weights dwarfs it --
/// 8.39 GB against 0.28 GB, measured 2026-09-09. The alternative the roadmap
/// originally proposed, whole-tree minus idle baseline, cannot work for a
/// provider that never sits idle, and LM Studio never has.
///
/// `None` whenever the answer would be a guess: no model resident, more than
/// one resident, or an observation with no per-process detail.
pub fn model_bytes(o: &Observation) -> Option<u64> {
    if o.loaded != 1 || !o.attributable {
        return None;
    }
    o.processes.iter().map(|p| p.footprint_bytes).max()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::fixtures;

    /// Measured 2026-09-09: llama-server held 8.39 GB while LM Studio's five
    /// Electron processes together held 0.28 GB. The model is the big one.
    #[test]
    fn the_model_is_the_largest_process_in_the_tree() {
        let o = fixtures::loaded("qwen3-14b", 40960, 9_000_000_000);
        assert_eq!(model_bytes(&o), Some(9_000_000_000));
    }

    /// Two resident models share a tree and the largest process is one of
    /// them, not both. Refusing beats guessing.
    #[test]
    fn two_resident_models_cannot_be_attributed() {
        let o = fixtures::two_loaded("qwen3-14b", 40960, 9_000_000_000);
        assert_eq!(model_bytes(&o), None);
    }

    #[test]
    fn a_provider_with_nothing_loaded_attributes_nothing() {
        assert_eq!(model_bytes(&fixtures::idle()), None);
    }

    #[test]
    fn an_observation_with_no_process_detail_attributes_nothing() {
        let mut o = fixtures::loaded("qwen3-14b", 40960, 9_000_000_000);
        o.processes.clear();
        assert_eq!(model_bytes(&o), None, "a v1 record cannot be attributed");
    }
}
