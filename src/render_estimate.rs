use crate::estimate::estimator::{Basis, Estimate};
use crate::memory::Machine;
use crate::render::human_bytes;

/// Memory held back for the OS and everything that is not a model.
///
/// `design.md` section 9 calls 8 GB on 24 GB "a guess to be tuned by
/// measurement, not a claim", and it remains one: ten observations is not a
/// basis for tuning it. This is a knob with a documented default, not a
/// finding.
pub const DEFAULT_RESERVE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub fn render_estimate(e: &Estimate, machine: &Machine, reserve: Option<u64>) -> String {
    let reserve = reserve.unwrap_or(DEFAULT_RESERVE_BYTES);
    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve);

    let mut out = String::new();
    match (e.bytes, e.basis) {
        (Some(bytes), Basis::Declared) => {
            out.push_str(&format!(
                "  predicted  {:>9}   (declared by the provider — weights only)\n",
                human_bytes(bytes)
            ));
            out.push_str(&format!(
                "  headroom   {:>9}   ({} free \u{2212} {} reserve)\n",
                human_bytes(headroom), human_bytes(free), human_bytes(reserve)
            ));
            out.push_str(if bytes <= headroom {
                "  verdict    fits, on a floor — the KV cache is not counted\n"
            } else {
                "  verdict    WOULD NOT FIT\n"
            });
        }
        (Some(bytes), Basis::Measured) => {
            let spread = e
                .spread_bytes
                .filter(|s| *s > 0)
                .map(|s| format!(" \u{b1}{}", human_bytes(s)))
                .unwrap_or_default();
            out.push_str(&format!(
                "  predicted  {:>9}{}  (measured, {} observation{})\n",
                human_bytes(bytes),
                spread,
                e.samples,
                if e.samples == 1 { "" } else { "s" },
            ));
            out.push_str(&format!(
                "  headroom   {:>9}   ({} free \u{2212} {} reserve)\n",
                human_bytes(headroom),
                human_bytes(free),
                human_bytes(reserve),
            ));
            out.push_str(if bytes <= headroom {
                "  verdict    fits\n"
            } else {
                "  verdict    WOULD NOT FIT\n"
            });
        }
        _ => {
            out.push_str("  predicted  unknown \u{2014} this shape has never been measured\n");
            out.push_str(&format!(
                "  headroom   {:>9}   ({} free \u{2212} {} reserve)\n",
                human_bytes(headroom),
                human_bytes(free),
                human_bytes(reserve),
            ));
            out.push_str("  verdict    none \u{2014} load it once and it will be known\n");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::estimator::{Basis, Estimate};
    use crate::memory::Machine;

    fn machine(free: u64) -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 25_769_803_776 - free,
            swap_total_bytes: 6_442_450_944,
            swap_used_bytes: 0,
        }
    }

    fn est(bytes: u64, samples: usize) -> Estimate {
        Estimate {
            bytes: Some(bytes),
            basis: Basis::Measured,
            samples,
            spread_bytes: Some(100_000_000),
        }
    }

    #[test]
    fn a_model_larger_than_headroom_would_not_fit() {
        let out = render_estimate(&est(9_000_000_000, 3), &machine(4_000_000_000), None);
        assert!(out.contains("WOULD NOT FIT"), "{out}");
        assert!(out.contains("measured"), "the basis is always stated: {out}");
    }

    #[test]
    fn a_model_inside_headroom_fits() {
        let out = render_estimate(&est(2_000_000_000, 1), &machine(12_000_000_000), None);
        assert!(out.contains("fits"), "{out}");
    }

    /// The reserve is subtracted before the verdict: headroom the OS needs is
    /// not headroom a model may take.
    #[test]
    fn the_reserve_is_held_back_from_headroom() {
        let e = est(5_000_000_000, 1);
        let generous = render_estimate(&e, &machine(6_000_000_000), Some(0));
        let reserved = render_estimate(&e, &machine(6_000_000_000), Some(4_000_000_000));
        assert!(generous.contains("fits"), "{generous}");
        assert!(reserved.contains("WOULD NOT FIT"), "{reserved}");
    }

    /// An unmeasured shape must not produce a verdict at all.
    #[test]
    fn an_unknown_estimate_refuses_to_judge() {
        let e = Estimate { bytes: None, basis: Basis::Unknown, samples: 0, spread_bytes: None };
        let out = render_estimate(&e, &machine(1_000_000), None);
        assert!(out.contains("unknown"), "{out}");
        assert!(!out.contains("FIT"), "no verdict without a measurement: {out}");
    }
}
