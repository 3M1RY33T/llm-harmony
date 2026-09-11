//! Will this land, and will it run once it has?
//!
//! Two ledgers, and the whole project is built on their being different. The
//! bytes have to fit on disk **and** the result has to be loadable afterwards,
//! and a pull admitted against only the first is how a machine ends up holding
//! a 28 GB model it can never serve.
//!
//! Refused before a byte moves, not after. That is the difference between this
//! and every download button that reports the problem at 97%.

use serde::Serialize;

use crate::estimate::estimator::Estimate;
use crate::memory::Machine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// Fits both ledgers as things stand.
    Fits,
    /// The bytes will land, but nothing could load the result right now.
    /// Deliberately distinct from `NoDisk`: the file is still worth having, and
    /// making room in memory is a different act from making room on disk.
    FitsDiskOnly,
    NoDisk,
    /// Neither. Named separately so the message can say both at once rather
    /// than reporting the first wall it hit.
    Neither,
}

#[derive(Debug, Clone, Serialize)]
pub struct Fit {
    pub verdict: Verdict,
    pub down_bytes: u64,
    /// What it would cost resident, once loaded. `None` when nothing can price
    /// it — and an unknown cost is **not** treated as zero, because a floor of
    /// zero admits everything.
    pub resident_bytes: Option<u64>,
    pub free_disk_bytes: u64,
    /// Memory a model may actually take: free **minus the reserve**.
    ///
    /// Not raw free memory. Admitting against that would put intake's bar
    /// above the loader's, and the pull it let through would be refused by
    /// `load` afterwards -- which is the exact failure this module's second
    /// ledger exists to prevent, arriving one step later.
    ///
    /// Saturates at zero, and that is why the three fields below exist. On a
    /// machine with 7.9G free and an 8.0G reserve this is `0` -- a true
    /// number that reads as a measurement of an empty machine, when what
    /// happened is that the reserve swallowed the lot.
    pub free_memory_bytes: u64,
    /// Free memory before the reserve came off. Carried so a headroom of zero
    /// can say which of the two zeros it is.
    pub free_before_reserve_bytes: u64,
    /// What was held back.
    pub reserve_bytes: u64,
    /// Everything the machine has, ever, with every provider stopped.
    ///
    /// The one figure that separates "not right now" from "not here". Above
    /// it, no unload and no reserve setting changes the answer, and a message
    /// that invites someone to free memory is inviting them to fail.
    pub machine_bytes: u64,
    /// The basis harmony priced it on: measured, computed or declared. Shown as
    /// a badge, because a refusal on a declared figure is one users otherwise
    /// meet cold.
    pub basis: String,
}

impl Fit {
    pub fn ok(&self) -> bool {
        matches!(self.verdict, Verdict::Fits)
    }

    /// Did the *memory* ledger admit it? Disk is a separate fact, and callers
    /// that only want to know whether this machine could hold the thing
    /// should not have to care whether a target directory was priced.
    pub fn memory_ok(&self) -> bool {
        matches!(self.verdict, Verdict::Fits | Verdict::NoDisk)
    }

    /// Bigger than the whole machine, so no amount of freeing reaches it.
    ///
    /// Found 2026-09-11 on a 24 GiB Mac asked for a 25.6 GiB MLX build of a
    /// 27B: the refusal said "0B is available", which is what a busy machine
    /// looks like, so it read as "come back later". There is no later.
    pub fn beyond_machine(&self) -> bool {
        matches!(self.resident_bytes, Some(b) if b > self.machine_bytes)
    }

    /// Why memory said no, in the machine's own terms.
    fn memory_clause(&self) -> String {
        let res = self
            .resident_bytes
            .map(crate::render::human_bytes)
            .unwrap_or_else(|| "an unknown amount".into());
        if self.beyond_machine() {
            return format!(
                "loading it needs {res}, and this machine has {} in total \u{2014} \
                 nothing that can be unloaded frees enough",
                crate::render::human_bytes(self.machine_bytes),
            );
        }
        if self.free_memory_bytes == 0 && self.free_before_reserve_bytes > 0 {
            // Not an empty machine: a reserve wider than what is free.
            return format!(
                "loading it needs {res}, and the {} free is all inside the {} reserve",
                crate::render::human_bytes(self.free_before_reserve_bytes),
                crate::render::human_bytes(self.reserve_bytes),
            );
        }
        format!(
            "loading it needs {res} where {} is available",
            crate::render::human_bytes(self.free_memory_bytes),
        )
    }

    pub fn message(&self) -> String {
        let down = crate::render::human_bytes(self.down_bytes);
        let disk = crate::render::human_bytes(self.free_disk_bytes);
        match self.verdict {
            Verdict::Fits => {
                let res = self
                    .resident_bytes
                    .map(crate::render::human_bytes)
                    .unwrap_or_else(|| "an unknown amount".into());
                let mem = crate::render::human_bytes(self.free_memory_bytes);
                // Both ledgers, because "fits" on one of them is the claim
                // this module exists to stop anyone making. And the basis,
                // because a pull can only ever be priced on a declared floor
                // -- see `run::price_build` -- so "fits" here means "the
                // weights fit", never "it will serve at your window".
                let floor = if self.basis == "declared" {
                    " (a declared floor: the KV cache is not in it)"
                } else {
                    ""
                };
                format!("{down} down, {disk} free on disk; {res} resident against {mem} headroom{floor}")
            }
            Verdict::NoDisk => format!("needs {down} on disk but only {disk} is free"),
            Verdict::FitsDiskOnly | Verdict::Neither => {
                let disk_part = if matches!(self.verdict, Verdict::Neither) {
                    format!("needs {down} on disk but only {disk} is free, and ")
                } else {
                    format!("{down} would land, but ")
                };
                format!("{disk_part}{}", self.memory_clause())
            }
        }
    }
}

/// Admit a pull against both ledgers.
///
/// `free_disk_bytes` is passed in rather than read here so this stays a pure
/// function of two numbers and an estimate — the part worth testing is the
/// arithmetic, not the statfs.
pub fn admit(
    down_bytes: u64,
    estimate: &Estimate,
    machine: &Machine,
    free_disk_bytes: u64,
    reserve_bytes: u64,
) -> Fit {
    let disk_ok = free_disk_bytes >= down_bytes;
    // The same bar `resolve` admits against, for the same reason the estimate
    // ladder is shared: two admission paths over one resource must not hold
    // each other to different numbers.
    let headroom = machine.free_bytes().saturating_sub(reserve_bytes);
    // An unpriceable model is not a free one. `None` cannot admit, because a
    // floor of zero would let everything through on the one path where nobody
    // can say what it costs.
    let memory_ok = match estimate.bytes {
        Some(b) => headroom >= b,
        None => false,
    };
    let verdict = match (disk_ok, memory_ok) {
        (true, true) => Verdict::Fits,
        (true, false) => Verdict::FitsDiskOnly,
        (false, true) => Verdict::NoDisk,
        (false, false) => Verdict::Neither,
    };
    Fit {
        verdict,
        down_bytes,
        resident_bytes: estimate.bytes,
        free_disk_bytes,
        free_memory_bytes: headroom,
        free_before_reserve_bytes: machine.free_bytes(),
        reserve_bytes,
        machine_bytes: machine.total_bytes,
        basis: format!("{:?}", estimate.basis).to_lowercase(),
    }
}

/// Bytes free on the filesystem holding `path`, walking up to the first parent
/// that exists — the target directory usually does not yet.
pub fn free_disk_bytes(path: &std::path::Path) -> u64 {
    let mut probe = path;
    loop {
        if probe.exists() {
            break;
        }
        match probe.parent() {
            Some(p) => probe = p,
            None => return 0,
        }
    }
    statvfs_free(probe)
}

fn statvfs_free(path: &std::path::Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return 0 };
    // SAFETY: `c` is a valid NUL-terminated path and `stat` is fully written by
    // statvfs before it is read.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut stat) != 0 {
            return 0;
        }
        (stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::estimator::{Basis, Estimate};

    fn machine(free: u64) -> Machine {
        let total = 25_769_803_776u64;
        Machine {
            total_bytes: total,
            used_bytes: total - free,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        }
    }

    fn priced(bytes: u64) -> Estimate {
        Estimate { bytes: Some(bytes), basis: Basis::Computed, samples: 0, spread_bytes: None }
    }

    #[test]
    fn a_pull_is_admitted_against_disk_and_memory_before_a_byte_moves() {
        let f = admit(8_000, &priced(11_000), &machine(20_000), 100_000, 0);
        assert_eq!(f.verdict, Verdict::Fits);
        assert!(f.ok());
    }

    #[test]
    fn landing_on_disk_and_being_loadable_are_reported_separately() {
        // The case the two-ledger claim exists for: plenty of disk, no room to
        // run it. Worth downloading; not worth pretending it will serve.
        let f = admit(8_000, &priced(30_000), &machine(3_000), 100_000, 0);
        assert_eq!(f.verdict, Verdict::FitsDiskOnly);
        assert!(!f.ok());
        assert!(f.message().contains("would land"), "{}", f.message());
    }

    #[test]
    fn no_disk_says_so_in_disk_terms() {
        let f = admit(500_000, &priced(1), &machine(20_000), 100_000, 0);
        assert_eq!(f.verdict, Verdict::NoDisk);
        assert!(f.message().contains("on disk"), "{}", f.message());
    }

    #[test]
    fn failing_both_names_both_rather_than_the_first_wall_it_hit() {
        let f = admit(500_000, &priced(30_000), &machine(3_000), 100_000, 0);
        assert_eq!(f.verdict, Verdict::Neither);
        let m = f.message();
        assert!(m.contains("on disk") && m.contains("loading it needs"), "{m}");
    }

    #[test]
    fn an_unpriceable_model_does_not_admit_on_a_floor_of_zero() {
        let f = admit(8_000, &Estimate::unknown(), &machine(20_000), 100_000, 0);
        assert!(!f.ok(), "nobody can say what it costs, so nobody can say it fits");
        assert_eq!(f.verdict, Verdict::FitsDiskOnly);
        assert!(f.message().contains("unknown amount"), "{}", f.message());
    }

    /// The defect this signature exists for: 9 GB free, an 8 GB model, and an
    /// 8 GB reserve. Raw free memory says yes and the loader says no, so
    /// intake said yes and `load` refused afterwards -- the pull admitted
    /// against one ledger that the second ledger was meant to stop.
    #[test]
    fn the_reserve_is_held_back_so_intake_and_load_cannot_disagree() {
        const GB: u64 = 1024 * 1024 * 1024;
        let eight = 8 * GB;
        let m = machine(9 * GB);
        assert_eq!(admit(eight, &priced(eight), &m, 500 * GB, 0).verdict, Verdict::Fits);
        assert_eq!(
            admit(eight, &priced(eight), &m, 500 * GB, eight).verdict,
            Verdict::FitsDiskOnly,
            "the reserve is not headroom a model may take"
        );
    }

    /// Found on a real machine 2026-09-11: a 24 GiB Mac, 7.9G free, an 8.0G
    /// reserve, asked for a 25.6 GiB MLX build of a 27B. The refusal read
    /// "loading it needs 25.6G where 0B is available" -- two wrong impressions
    /// in one line. Nothing was going to free 25.6G on a 24 GiB machine, and
    /// the machine was not out of memory; the reserve was simply wider than
    /// what was free.
    #[test]
    fn a_model_larger_than_the_machine_says_so_rather_than_blaming_the_moment() {
        const GB: u64 = 1024 * 1024 * 1024;
        // machine() is 24 GiB total. 7.9 free, 8 held back: headroom is 0.
        let m = machine(7_900 * GB / 1000);
        let f = admit(25 * GB, &priced(25 * GB + GB / 2), &m, 500 * GB, 8 * GB);
        assert_eq!(f.verdict, Verdict::FitsDiskOnly);
        assert!(f.beyond_machine(), "25.5 GiB does not fit in 24 GiB, ever");
        let msg = f.message();
        assert!(msg.contains("in total"), "it names the machine's own ceiling: {msg}");
        assert!(
            !msg.contains("0B is available"),
            "and never reports a saturated reserve as a measurement: {msg}"
        );
    }

    /// The other zero. A model that WOULD fit on an idle machine, refused
    /// because the reserve is currently wider than free memory -- which is a
    /// thing that changes in a minute, and the message should say which of
    /// the two situations the reader is in.
    #[test]
    fn a_reserve_wider_than_free_memory_names_the_reserve_not_an_empty_machine() {
        const GB: u64 = 1024 * 1024 * 1024;
        let f = admit(4 * GB, &priced(4 * GB), &machine(7 * GB), 500 * GB, 8 * GB);
        assert_eq!(f.verdict, Verdict::FitsDiskOnly);
        assert!(!f.beyond_machine(), "4 GiB fits in 24 GiB; the machine is not the wall");
        assert_eq!(f.free_memory_bytes, 0);
        let msg = f.message();
        assert!(msg.contains("reserve"), "{msg}");
        assert!(msg.contains("7.0G free"), "the free memory survives the saturation: {msg}");
    }

    /// Memory and disk are separate ledgers, and a caller that priced no
    /// target directory is asking about exactly one of them.
    #[test]
    fn memory_ok_answers_for_memory_alone() {
        const GB: u64 = 1024 * 1024 * 1024;
        // No disk priced at all: `NoDisk`, and memory still said yes.
        let f = admit(4 * GB, &priced(4 * GB), &machine(20 * GB), 0, 0);
        assert_eq!(f.verdict, Verdict::NoDisk);
        assert!(f.memory_ok());
        let f = admit(4 * GB, &priced(30 * GB), &machine(20 * GB), 0, 0);
        assert!(!f.memory_ok());
    }

    /// A pull can only ever be priced on the published file size -- see
    /// `run::price_build` -- and `decide` will not admit a load on one at all.
    /// So "fits" here has to say which ledger it is speaking about and on what
    /// basis, or it reads as a promise the loader will not keep.
    #[test]
    fn a_fitting_pull_reports_both_ledgers_and_badges_a_declared_floor() {
        let mut e = priced(11_000);
        e.basis = Basis::Declared;
        let f = admit(8_000, &e, &machine(20_000), 100_000, 0);
        let m = f.message();
        assert!(m.contains("free on disk"), "{m}");
        assert!(m.contains("headroom"), "the memory half is stated too: {m}");
        assert!(m.contains("declared floor"), "and the basis is badged: {m}");
    }
}
