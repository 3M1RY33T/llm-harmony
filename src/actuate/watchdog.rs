//! Watching a load happen, so the guarantee does not rest on the prediction.
//!
//! Admission decides whether a model *should* fit. This decides whether it
//! *is* fitting, from the machine rather than from the artifact, and it is the
//! only part of the slice that can be right when the estimator is wrong.
//!
//! It samples through injected closures so the failure paths are testable
//! without allocating nine gigabytes. And it only ever *decides*: the caller
//! performs the abort, because only the caller knows which adapter to call --
//! and on a self-managed provider there is no abort to perform, which is why
//! those loads have to clear a stricter margin before they are started at all.

use std::time::{Duration, Instant};

use crate::memory::Machine;

#[derive(Debug, Clone)]
pub struct Limits {
    /// Abort below this much free physical memory.
    ///
    /// Defaults to half the reserve: the reserve is what the OS is owed, and
    /// crossing half of it mid-load means the estimate was wrong by more than
    /// the margin was built to absorb.
    pub floor_bytes: u64,
    /// Abort if swap in use grows by more than this during the load.
    ///
    /// On unified memory this is the signal that matters. There is no VRAM
    /// ceiling to fail an allocation first: the machine simply starts paging,
    /// and `field-notes.md` measured 30,187 pageouts with a 15 GB model
    /// resident at an 8k window.
    pub max_swap_growth_bytes: u64,
    pub poll: Duration,
    pub timeout: Duration,
}

impl Limits {
    /// The ordinary limits for a given reserve.
    pub fn from_reserve(reserve_bytes: u64) -> Limits {
        Limits {
            floor_bytes: reserve_bytes / 2,
            max_swap_growth_bytes: 2 * 1024 * 1024 * 1024,
            poll: Duration::from_millis(250),
            timeout: Duration::from_secs(180),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "outcome", rename_all = "kebab-case")]
pub enum Outcome {
    Loaded { took_s: u64 },
    /// The load was still running and the machine was losing. The caller
    /// unloads; this only decides.
    Aborted { reason: String },
    /// Never became resident. Not necessarily a failure of the provider --
    /// but not a success either, and the caller must not report one.
    TimedOut,
}

/// Watch a load that is already under way.
///
/// `sample` reads the machine; `resident` answers whether the model has
/// arrived. Both are polled until one of the three outcomes is reached.
pub fn supervise<S, R>(limits: &Limits, mut sample: S, mut resident: R) -> Outcome
where
    S: FnMut() -> Machine,
    R: FnMut() -> bool,
{
    let started = Instant::now();
    let baseline_swap = sample().swap_used_bytes;

    loop {
        let m = sample();

        // Residency first: a load that finished is a load that finished, even
        // if it left the machine tight. Refusing to acknowledge it would leave
        // harmony's ledger disagreeing with the provider's.
        if resident() {
            return Outcome::Loaded { took_s: started.elapsed().as_secs() };
        }

        let free = m.total_bytes.saturating_sub(m.used_bytes);
        if free < limits.floor_bytes {
            return Outcome::Aborted {
                reason: format!(
                    "free memory fell to {} during the load, below the {} floor",
                    crate::render::human_bytes(free),
                    crate::render::human_bytes(limits.floor_bytes)
                ),
            };
        }

        let swap_growth = m.swap_used_bytes.saturating_sub(baseline_swap);
        if swap_growth > limits.max_swap_growth_bytes {
            return Outcome::Aborted {
                reason: format!(
                    "swap grew by {} during the load; the machine is already paging",
                    crate::render::human_bytes(swap_growth)
                ),
            };
        }

        if started.elapsed() >= limits.timeout {
            return Outcome::TimedOut;
        }
        std::thread::sleep(limits.poll);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    /// A machine sample with `n` GB free and no swap in use.
    fn free(n: u64) -> Machine {
        Machine {
            total_bytes: 24 * GB,
            used_bytes: 24 * GB - n * GB,
            swap_total_bytes: 8 * GB,
            swap_used_bytes: 0,
        }
    }

    /// Plenty free, and `n` GB of swap in use.
    fn swap(n: u64) -> Machine {
        Machine { swap_used_bytes: n * GB, ..free(12) }
    }

    fn limits() -> Limits {
        Limits {
            floor_bytes: 4 * GB,
            max_swap_growth_bytes: 2 * GB,
            poll: Duration::from_millis(0),
            timeout: Duration::from_millis(50),
        }
    }

    /// Hand back one prepared sample per poll, repeating the last forever, so
    /// a test never depends on how many times `supervise` looks.
    fn stepper(samples: Vec<Machine>) -> impl FnMut() -> Machine {
        let mut i = 0;
        move || {
            let s = samples[i.min(samples.len() - 1)];
            i += 1;
            s
        }
    }

    fn after_n_polls(n: usize) -> impl FnMut() -> bool {
        let mut i = 0;
        move || {
            i += 1;
            i >= n
        }
    }

    fn never_resident() -> impl FnMut() -> bool {
        || false
    }

    /// The ordinary path: pressure stays fine, residency appears, done.
    #[test]
    fn a_load_that_stays_within_the_floor_completes() {
        let out = supervise(&limits(), stepper(vec![free(12), free(11), free(10)]), after_n_polls(3));
        assert!(matches!(out, Outcome::Loaded { .. }), "{out:?}");
    }

    /// The case this exists for: free memory crosses the floor mid-load and
    /// the load is abandoned rather than completed.
    #[test]
    fn crossing_the_floor_aborts_the_load() {
        match supervise(&limits(), stepper(vec![free(12), free(6), free(1)]), never_resident()) {
            Outcome::Aborted { reason } => assert!(reason.contains("free memory"), "{reason}"),
            o => panic!("{o:?}"),
        }
    }

    /// Swap growth is the second signal, and on unified memory it is the one
    /// that means the machine is already losing.
    #[test]
    fn swap_growing_during_the_load_aborts_it_even_above_the_floor() {
        match supervise(&limits(), stepper(vec![swap(0), swap(1), swap(4)]), never_resident()) {
            Outcome::Aborted { reason } => assert!(reason.contains("swap"), "{reason}"),
            o => panic!("{o:?}"),
        }
    }

    /// Growth, not level: a machine that was already paging before the load
    /// started has not been made worse by it, and aborting on the standing
    /// figure would make every load fail on a busy machine.
    #[test]
    fn swap_already_in_use_before_the_load_is_not_growth() {
        let out = supervise(&limits(), stepper(vec![swap(6), swap(6), swap(6)]), after_n_polls(3));
        assert!(matches!(out, Outcome::Loaded { .. }), "{out:?}");
    }

    /// A provider that never reports residency must not hang the caller.
    #[test]
    fn a_load_that_never_becomes_resident_times_out() {
        match supervise(&limits(), stepper(vec![free(12); 4]), never_resident()) {
            Outcome::TimedOut => {}
            o => panic!("{o:?}"),
        }
    }

    /// A finished load is a finished load. Reporting otherwise would leave
    /// harmony's ledger disagreeing with the provider's about what is resident.
    #[test]
    fn residency_is_checked_before_the_pressure_limits() {
        let out = supervise(&limits(), stepper(vec![free(1)]), after_n_polls(1));
        assert!(matches!(out, Outcome::Loaded { .. }), "{out:?}");
    }
}
