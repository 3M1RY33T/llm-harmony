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

/// The processes attributable to one provider, and what they cost.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ProcessTree {
    pub pids: Vec<u32>,
    /// `None` when no pid's usage could be read -- e.g. a root-owned process.
    /// Never silently zero.
    pub footprint_bytes: Option<u64>,
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
fn phys_footprint(pid: u32) -> Option<u64> {
    pidrusage::<RUsageInfoV2>(pid as i32)
        .ok()
        .map(|r| r.ri_phys_footprint)
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

    let mut total: u64 = 0;
    let mut any = false;
    for pid in &pids {
        if let Some(bytes) = phys_footprint(*pid) {
            total += bytes;
            any = true;
        }
    }

    Some(ProcessTree {
        pids,
        footprint_bytes: any.then_some(total),
    })
}
