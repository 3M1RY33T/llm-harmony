//! Moving the bytes, and saying so while it happens.
//!
//! Two rules shape everything here.
//!
//! **A document per line, not a percentage printed over itself.** The caller is
//! a UI polling a job, and `\r`-redrawn progress is not readable by one. Each
//! event is one JSON object on its own line — NDJSON — so a reader that gets
//! half a line waits rather than misparsing.
//!
//! **Nothing half-written is ever visible to a provider.** The download lands
//! on a `.part` beside its destination and is renamed only once complete, so an
//! interrupted pull leaves something a scanner ignores rather than a truncated
//! GGUF that LM Studio will happily try to load. A crash is then recoverable
//! rather than a corrupt store, and it needs no job database to be so.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

/// One line of the progress stream.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum Progress {
    Started { file: String, bytes_total: Option<u64>, target: String },
    /// Emitted on a byte threshold rather than a timer: a stalled download
    /// should go quiet, because silence is information and a heartbeat that
    /// ticks through a stall is not.
    Advanced { bytes_done: u64, bytes_total: Option<u64> },
    Finished { file: String, bytes_done: u64, path: String },
    Failed { file: String, reason: String },
}

impl Progress {
    pub fn print(&self) {
        // Compact, not pretty: one document per LINE is the contract, and
        // pretty-printing puts a document across several.
        if let Ok(line) = serde_json::to_string(self) {
            println!("{line}");
            // A UI polling this cannot see a buffer.
            let _ = std::io::stdout().flush();
        }
    }
}

/// How often to report. Every chunk would be thousands of lines for one file.
const REPORT_EVERY_BYTES: u64 = 8 * 1024 * 1024;

/// Fetch one file into `dir`, atomically.
///
/// `on_progress` is called rather than printed to, so the same code path serves
/// the CLI's NDJSON and a test that asserts on the sequence.
pub fn fetch(
    agent: &ureq::Agent,
    url: &str,
    dir: &Path,
    file_name: &str,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let final_path = dir.join(file_name);
    // Beside the destination, not in /tmp: a rename across filesystems is a
    // copy, and a copy is not atomic.
    let part_path = dir.join(format!("{file_name}.part"));

    let mut response = agent.get(url).call().map_err(|e| e.to_string())?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    on_progress(Progress::Started {
        file: file_name.to_string(),
        bytes_total: total,
        target: dir.display().to_string(),
    });

    let mut reader = response.body_mut().as_reader();
    let mut out = std::fs::File::create(&part_path)
        .map_err(|e| format!("cannot write {}: {e}", part_path.display()))?;

    let mut buf = vec![0u8; 1024 * 1024];
    let mut done: u64 = 0;
    let mut reported: u64 = 0;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                // The partial file goes with the failure. Leaving it would put
                // a truncated model where a provider looks.
                let _ = std::fs::remove_file(&part_path);
                return Err(format!("transfer failed after {done} bytes: {e}"));
            }
        };
        if let Err(e) = out.write_all(&buf[..n]) {
            let _ = std::fs::remove_file(&part_path);
            return Err(format!("write failed after {done} bytes: {e}"));
        }
        done += n as u64;
        if done - reported >= REPORT_EVERY_BYTES {
            reported = done;
            on_progress(Progress::Advanced { bytes_done: done, bytes_total: total });
        }
    }

    if let Err(e) = out.sync_all() {
        let _ = std::fs::remove_file(&part_path);
        return Err(format!("could not flush {}: {e}", part_path.display()));
    }
    drop(out);

    // A short read is a failed download, not a small model. Without this the
    // rename would publish a truncated file as a finished one.
    if let Some(expected) = total {
        if done != expected {
            let _ = std::fs::remove_file(&part_path);
            return Err(format!("expected {expected} bytes, got {done}"));
        }
    }

    std::fs::rename(&part_path, &final_path).map_err(|e| {
        let _ = std::fs::remove_file(&part_path);
        format!("could not place {}: {e}", final_path.display())
    })?;

    on_progress(Progress::Finished {
        file: file_name.to_string(),
        bytes_done: done,
        path: final_path.display().to_string(),
    });
    Ok(final_path)
}

/// An agent for transfers rather than probes: no global timeout, because that
/// is what `Http` uses to keep `status` fast and it would abort a 28 GB pull at
/// the three-second mark.
pub fn transfer_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(20)))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_progress_event_is_one_line() {
        let line = serde_json::to_string(&Progress::Advanced {
            bytes_done: 10,
            bytes_total: Some(100),
        })
        .unwrap();
        assert!(!line.contains('\n'), "a reader splits on newlines: {line}");
        let back: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(back["event"], "advanced");
        assert_eq!(back["bytes_done"], 10);
    }

    #[test]
    fn every_event_names_its_kind_so_a_reader_can_switch_on_it() {
        for (p, want) in [
            (Progress::Started { file: "m.gguf".into(), bytes_total: None, target: "/s".into() }, "started"),
            (Progress::Finished { file: "m.gguf".into(), bytes_done: 1, path: "/s/m.gguf".into() }, "finished"),
            (Progress::Failed { file: "m.gguf".into(), reason: "boom".into() }, "failed"),
        ] {
            let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(v["event"], want);
        }
    }
}
