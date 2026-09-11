//! The actuation lock: one at a time, per machine.
//!
//! `flock(2)`, deliberately, rather than a pidfile. A pidfile has to be reaped
//! when its owner dies badly, and the owner here is a process that might be
//! killed mid-load; an advisory lock is released by the kernel on close, so a
//! dead harmony cannot wedge the next one.
//!
//! Read-only verbs never take it. `status`, `ls` and `estimate` describe the
//! world and cannot change it, so making them queue behind a slow load would
//! trade the one thing this project promises -- that it is out of the way --
//! for nothing.

use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Who is holding the lock, written into the file by the holder so a waiter
/// can say something better than "busy".
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LockHolder {
    pub pid: u32,
    pub verb: String,
    pub started_at: u64,
}

#[derive(Debug)]
pub enum LockError {
    /// Someone else is mid-actuation. `holder` is absent when the record could
    /// not be read -- the lock is still held, which is the part that matters.
    Busy { holder: Option<LockHolder>, waited_s: u64 },
    Io(String),
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Busy { holder: Some(h), waited_s } => write!(
                f,
                "another actuation is in progress: {} (pid {}), waited {waited_s}s",
                h.verb, h.pid
            ),
            LockError::Busy { holder: None, waited_s } => {
                write!(f, "another actuation is in progress, waited {waited_s}s")
            }
            LockError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Held for as long as the value lives. Dropping it, or dying, releases it.
#[derive(Debug)]
pub struct ActuationLock {
    _file: File,
}

impl ActuationLock {
    pub fn path() -> Option<PathBuf> {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".local/state/llm-harmony/actuation.lock"))
    }

    pub fn acquire(verb: &str, timeout: Duration) -> Result<ActuationLock, LockError> {
        let path = ActuationLock::path()
            .ok_or_else(|| LockError::Io("no HOME; cannot locate the lock".to_string()))?;
        ActuationLock::acquire_at(&path, verb, timeout)
    }

    pub fn acquire_at(
        path: &Path,
        verb: &str,
        timeout: Duration,
    ) -> Result<ActuationLock, LockError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| LockError::Io(format!("{}: {e}", dir.display())))?;
        }
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|e| LockError::Io(format!("{}: {e}", path.display())))?;

        let started = Instant::now();
        loop {
            // SAFETY: `file` owns the descriptor for the whole call.
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                write_holder(path, verb);
                return Ok(ActuationLock { _file: file });
            }
            if started.elapsed() >= timeout {
                return Err(LockError::Busy {
                    holder: read_holder(path),
                    waited_s: started.elapsed().as_secs(),
                });
            }
            // Polling rather than blocking: a blocking flock cannot be given a
            // deadline, and a caller that waits forever is a caller that hangs.
            std::thread::sleep(Duration::from_millis(20).min(timeout));
        }
    }
}

/// Best effort: failing to record who we are must never fail an actuation that
/// is otherwise fine. The cost is a less helpful message for the next waiter.
fn write_holder(path: &Path, verb: &str) {
    let holder = LockHolder {
        pid: std::process::id(),
        verb: verb.to_string(),
        started_at: crate::record::now_unix(),
    };
    if let Ok(body) = serde_json::to_string(&holder) {
        let _ = std::fs::write(path, body);
    }
}

fn read_holder(path: &Path) -> Option<LockHolder> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}
