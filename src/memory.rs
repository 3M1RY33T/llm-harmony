use libproc::libproc::file_info::{pidfdinfo, ListFDs, ProcFDType};
use libproc::libproc::net_info::{SocketFDInfo, SocketInfoKind};
use libproc::libproc::pid_rusage::{pidrusage, RUsageInfoV2};
use libproc::libproc::proc_pid::listpidinfo;
use libproc::processes::{pids_by_type, ProcFilter};
use sysinfo::{MemoryRefreshKind, ProcessRefreshKind, RefreshKind, System};

/// Machine-wide totals. `total_bytes` equals `sysctl hw.memsize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Machine {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
}

impl Machine {
    pub fn read() -> Result<Machine, String> {
        let sys = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        let total = sys.total_memory();
        if total == 0 {
            return Err("cannot read machine memory total".to_string());
        }
        Ok(Machine {
            total_bytes: total,
            used_bytes: sys.used_memory(),
            swap_total_bytes: sys.total_swap(),
            swap_used_bytes: sys.used_swap(),
        })
    }

    pub fn free_fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.total_bytes.saturating_sub(self.used_bytes)) as f64 / self.total_bytes as f64
    }
}

/// One process's contribution, kept so a model's cost can be attributed to the
/// process that actually holds it rather than to the provider as a whole.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProcessSample {
    pub pid: u32,
    /// `max(phys_footprint, rss)` for this process.
    pub footprint_bytes: u64,
    pub phys_footprint_bytes: Option<u64>,
    pub rss_bytes: Option<u64>,
}

/// The processes attributable to one provider, and what they cost.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ProcessTree {
    pub pids: Vec<u32>,
    /// The admission figure: per-process `max(phys_footprint, rss)`, summed.
    /// `None` when no pid's usage could be read -- never silently zero.
    pub footprint_bytes: Option<u64>,
    /// Component, kept so the estimator can revisit this choice later.
    pub phys_footprint_bytes: Option<u64>,
    /// Component. Sees mmapped weights that `phys_footprint` does not.
    pub rss_bytes: Option<u64>,
    /// Per-process breakdown. The estimator attributes a model to the largest
    /// entry here rather than to the tree total.
    pub processes: Vec<ProcessSample>,
}

/// The larger of the two metrics, because each is blind to something the
/// other sees -- mmapped GGUF weights for `phys_footprint`, Metal buffers in
/// unified memory for RSS. See the tests for the measured evidence.
pub fn combined(phys: Option<u64>, rss: Option<u64>) -> Option<u64> {
    match (phys, rss) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

pub fn port_from_url(url: &str) -> Option<u16> {
    let after_scheme = url.split("://").nth(1)?;
    let hostport = after_scheme.split('/').next()?;
    hostport.rsplit(':').next()?.parse().ok()
}

/// The pid holding a listening TCP socket on `port`.
///
/// Uniform across all four providers, which is why this is here and not on the
/// `Adapter` trait: verified 2026-09-09, pid 647 owns :1234 (LM Studio) and pid
/// 4140 owns :11434 (Ollama), with model memory in their descendants.
pub fn pid_listening_on(port: u16) -> Option<u32> {
    let all = pids_by_type(ProcFilter::All).ok()?;
    for pid in all {
        let Ok(fds) = listpidinfo::<ListFDs>(pid as i32, 4096) else {
            continue;
        };
        for fd in fds {
            if !matches!(fd.proc_fdtype.into(), ProcFDType::Socket) {
                continue;
            }
            let Ok(sock) = pidfdinfo::<SocketFDInfo>(pid as i32, fd.proc_fd) else {
                continue;
            };
            if !matches!(sock.psi.soi_kind.into(), SocketInfoKind::Tcp) {
                continue;
            }
            // SAFETY: the union discriminant was just checked to be Tcp.
            let tcp = unsafe { sock.psi.soi_proto.pri_tcp };
            if u16::from_be(tcp.tcpsi_ini.insi_lport as u16) == port {
                return Some(pid);
            }
        }
    }
    None
}

fn descendants(sys: &System, root: u32) -> Vec<u32> {
    let mut out = vec![root];
    let mut frontier = vec![root];
    while let Some(parent) = frontier.pop() {
        for (pid, proc_) in sys.processes() {
            if proc_.parent().map(|p| p.as_u32()) == Some(parent) {
                let child = pid.as_u32();
                if !out.contains(&child) {
                    out.push(child);
                    frontier.push(child);
                }
            }
        }
    }
    out
}

/// `phys_footprint` for a pid. On Apple Silicon this includes Metal buffers in
/// unified memory, which is what makes it the right metric here -- sysinfo's
/// `Process::memory()` is resident size and undercounts by roughly 3x on the
/// GPU helper processes that actually hold model weights (verified 2026-09-09:
/// LM Studio pid 647 reported 104 MB resident against 285 MB footprint).
/// `(phys_footprint, resident_size)` for a pid, from one syscall.
fn usage(pid: u32) -> Option<(u64, u64)> {
    pidrusage::<RUsageInfoV2>(pid as i32)
        .ok()
        .map(|r| (r.ri_phys_footprint, r.ri_resident_size))
}

/// The process tree serving `port`, and its summed footprint.
///
/// Note this includes a provider's non-model overhead -- LM Studio idles at
/// ~0.62 GB across five Electron processes with nothing loaded. That is real
/// memory the machine cannot give to a model, so it belongs in the total.
pub fn footprint_for_port(port: u16) -> Option<ProcessTree> {
    let root = pid_listening_on(port)?;
    let sys = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
    );
    let pids = descendants(&sys, root);

    let (mut total, mut phys_total, mut rss_total) = (0u64, 0u64, 0u64);
    let mut any = false;
    let mut samples = Vec::new();
    for pid in &pids {
        if let Some((phys, rss)) = usage(*pid) {
            // Per process, not per tree: a provider mixes an Electron shell
            // whose cost is anonymous with a backend whose cost is mmapped,
            // and taking the max of the two sums would lose one of them.
            let combined = phys.max(rss);
            total += combined;
            phys_total += phys;
            rss_total += rss;
            any = true;
            samples.push(ProcessSample {
                pid: *pid,
                footprint_bytes: combined,
                phys_footprint_bytes: Some(phys),
                rss_bytes: Some(rss),
            });
        }
    }

    Some(ProcessTree {
        pids,
        footprint_bytes: any.then_some(total),
        phys_footprint_bytes: any.then_some(phys_total),
        rss_bytes: any.then_some(rss_total),
        processes: samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neither macOS metric sees the whole picture, and they are blind to
    /// different things. Verified 2026-09-09 on this machine:
    ///
    /// - `phys_footprint` misses memory-mapped GGUF weights, because clean
    ///   file-backed pages are not counted. `llama-server` held an 8.38 GB
    ///   model and the whole LM Studio tree reported 6.89 GB.
    /// - RSS misses Metal buffers in unified memory. LM Studio's main process
    ///   reported 104 MB resident against a 285 MB footprint.
    ///
    /// Taking the larger is crude, but it is wrong in the direction
    /// docs/design.md section 8 demands: over-estimating wastes capacity,
    /// under-estimating wedges the machine.
    #[test]
    fn the_larger_of_the_two_metrics_wins() {
        assert_eq!(combined(Some(6_890_000_000), Some(8_670_000_000)), Some(8_670_000_000));
        assert_eq!(combined(Some(285_000_000), Some(104_000_000)), Some(285_000_000));
    }

    #[test]
    fn a_readable_metric_is_used_even_when_the_other_is_not() {
        assert_eq!(combined(Some(500), None), Some(500));
        assert_eq!(combined(None, Some(700)), Some(700));
    }

    /// An unreadable process must not silently contribute zero.
    #[test]
    fn neither_metric_readable_is_unknown_not_zero() {
        assert_eq!(combined(None, None), None);
    }
}
