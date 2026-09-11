//! What a model will cost, computed from its shape when nothing has measured it.
//!
//! `design.md` §5 ranks estimates measured > computed > declared. Slice 3 built
//! the first and refused the rest; the consequence, counted 2026-09-10, was that
//! `Basis::Measured` covered exactly one (model, context) pair on this machine
//! and every other model fell through to a **declared** figure -- which is
//! weights only, by construction, for every source that publishes one.
//!
//! Admitting on weights alone is admitting on an under-estimate, and an
//! under-estimate is the failure that wedges the machine. This module is the
//! missing rung.

use crate::estimate::estimator::{Basis, Estimate};
use crate::estimate::shape::ModelShape;

/// Bytes per cache element. f16 -- what every one of these providers uses
/// unless told otherwise.
///
/// llama.cpp can quantise the cache with `-ctk`/`-ctv` and publishes nothing
/// about the choice, so harmony assumes the expensive case. `design.md` §8:
/// over-estimating wastes capacity, under-estimating wedges the machine.
pub const KV_DTYPE_BYTES: u64 = 2;

/// Activations, compute buffers, the allocator's own slack -- everything that
/// is neither weights nor cache.
///
/// A margin, not a measurement. Deliberately modest: the two terms it sits on
/// top of are already the whole of a load, and a large multiplier here would
/// refuse models that fit.
pub const MARGIN_NUM: u64 = 108;
pub const MARGIN_DEN: u64 = 100;

/// The KV cache at a given window.
///
/// `2` for K and V; per layer, per kv head, per head dimension, per token.
/// Saturating throughout: a caller may ask for any window, and a wrapped
/// product would be a small, confident, catastrophic number.
pub fn kv_bytes(shape: &ModelShape, context_tokens: u32, kv_dtype_bytes: u64) -> u64 {
    2u64.saturating_mul(u64::from(shape.n_layers))
        .saturating_mul(u64::from(shape.n_kv_heads))
        .saturating_mul(u64::from(shape.head_dim))
        .saturating_mul(u64::from(context_tokens))
        .saturating_mul(kv_dtype_bytes)
}

/// Weights plus cache plus margin, labelled as computed.
///
/// Never `Unknown`: a shape that parsed is a shape that can be priced. The
/// caller decides whether a computed figure is good enough to admit on --
/// `resolve::decide` says yes, and says so in the output.
pub fn computed(shape: &ModelShape, context_tokens: u32, kv_dtype_bytes: u64) -> Estimate {
    let base = shape
        .weights_bytes
        .saturating_add(kv_bytes(shape, context_tokens, kv_dtype_bytes));
    let bytes = base.saturating_mul(MARGIN_NUM) / MARGIN_DEN;

    Estimate { bytes: Some(bytes), basis: Basis::Computed, samples: 0, spread_bytes: None }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::estimator::Basis;
    use crate::estimate::shape::ModelShape;

    /// The real geometry of the 14B on this machine, read from its own header
    /// 2026-09-11.
    fn qwen3_14b() -> ModelShape {
        ModelShape {
            arch: "qwen3".into(),
            n_layers: 40,
            n_kv_heads: 8,
            head_dim: 128,
            weights_bytes: 9_000_000_000,
            trained_context: Some(40_960),
        }
    }

    /// K and V, per layer, per kv head, per token, at the cache dtype.
    /// 2 x 40 x 8 x 128 x 8192 x 2 = 1,342,177,280.
    #[test]
    fn kv_scales_linearly_with_context() {
        let s = qwen3_14b();
        assert_eq!(kv_bytes(&s, 8_192, 2), 1_342_177_280);
        assert_eq!(kv_bytes(&s, 16_384, 2), 2_684_354_560);
    }

    /// The whole point of the basis: the figure must move when the window
    /// moves. LM Studio's own estimator does not, which is exactly why it
    /// cannot carry admission.
    #[test]
    fn a_bigger_window_costs_more_than_a_smaller_one() {
        let s = qwen3_14b();
        let small = computed(&s, 8_192, 2).bytes.unwrap();
        let big = computed(&s, 40_960, 2).bytes.unwrap();
        assert!(big > small, "{big} should exceed {small}");
    }

    #[test]
    fn the_estimate_is_weights_plus_kv_plus_margin() {
        let s = qwen3_14b();
        let e = computed(&s, 8_192, 2);
        let floor = s.weights_bytes + kv_bytes(&s, 8_192, 2);
        assert!(e.bytes.expect("a computed estimate is always a number") > floor);
        assert_eq!(e.basis, Basis::Computed);
    }

    /// A quantised cache halves the term that dominates at long windows.
    /// Harmony cannot see the choice -- llama.cpp publishes nothing about
    /// `-ctk`/`-ctv` -- so it assumes the expensive case and takes an override.
    #[test]
    fn a_quantised_cache_costs_less_and_is_expressible() {
        let s = qwen3_14b();
        assert_eq!(kv_bytes(&s, 8_192, 1), kv_bytes(&s, 8_192, 2) / 2);
    }

    /// Multiplied in u64 with the biggest window a caller could ask for: the
    /// arithmetic must not wrap into a small, confident number.
    #[test]
    fn an_absurd_context_does_not_wrap() {
        let s = qwen3_14b();
        let e = computed(&s, u32::MAX, 8);
        assert!(e.bytes.unwrap() > 1 << 40, "saturating, not wrapping");
    }
}
