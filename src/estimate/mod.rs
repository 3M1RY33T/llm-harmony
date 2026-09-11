pub mod computed;
pub mod attribute;
pub mod baseline;
pub mod corpus;
pub mod estimator;

#[cfg(test)]
pub mod fixtures {
    use crate::memory::ProcessSample;
    use crate::provider::ProviderKind;
    use crate::record::{Observation, OBSERVATION_SCHEMA};

    pub fn sample(pid: u32, bytes: u64) -> ProcessSample {
        ProcessSample {
            pid,
            footprint_bytes: bytes,
            phys_footprint_bytes: Some(bytes),
            rss_bytes: Some(bytes),
        }
    }

    fn base(
        model: &str,
        ctx: u32,
        loaded: usize,
        attributable: bool,
        procs: Vec<ProcessSample>,
    ) -> Observation {
        Observation {
            schema: OBSERVATION_SCHEMA,
            at: 1,
            provider: ProviderKind::LmStudio,
            loaded,
            model: Some(model.to_string()),
            context_tokens: Some(ctx),
            attributable,
            footprint_bytes: procs.iter().map(|p| p.footprint_bytes).sum(),
            phys_footprint_bytes: None,
            rss_bytes: None,
            processes: procs,
            machine_total_bytes: 25_769_803_776,
            machine_used_bytes: 20_000_000_000,
            swap_used_bytes: 0,
        }
    }

    /// One resident model whose backing process holds `bytes`, beside a small
    /// shell process -- the real shape of an LM Studio tree.
    pub fn loaded(model: &str, ctx: u32, bytes: u64) -> Observation {
        base(model, ctx, 1, true, vec![sample(647, 300_000_000), sample(25164, bytes)])
    }

    /// Two resident models: attributable is false and nothing may be inferred.
    pub fn two_loaded(model: &str, ctx: u32, bytes: u64) -> Observation {
        base(model, ctx, 2, false, vec![sample(1, bytes / 2), sample(2, bytes / 2)])
    }

    /// The same reading, taken while the machine was paging. Not a
    /// measurement: see `estimator::TRUSTWORTHY_SWAP_CEILING_BYTES`.
    pub fn while_swapping(model: &str, ctx: u32, bytes: u64, swap: u64) -> Observation {
        let mut o = loaded(model, ctx, bytes);
        o.swap_used_bytes = swap;
        o
    }

    pub fn idle() -> Observation {
        base("unused", 0, 0, true, vec![sample(647, 600_000_000)])
    }
}
pub mod shape;
