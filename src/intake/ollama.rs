//! Ollama's transport: the same admission, a different way of moving bytes.
//!
//! Every other provider here takes a file drop — harmony downloads into the
//! store it chose and the provider finds it there. Ollama does not. Its store
//! is a content-addressed blob database written by its own `pull`, so a file
//! written into it is a file nothing will ever load; `place::store_for` says so
//! and is right.
//!
//! What that refusal got wrong was treating a statement about the *transport*
//! as a statement about the provider. Ollama pulls a GGUF repo straight from
//! Hugging Face under an `hf.co/<repo>:<QUANT>` name, so the bridge is **exact
//! rather than a guess**: the repo the user searched for and the repo Ollama
//! fetches are the same id, and no name in Ollama's own library has to be
//! matched by shape. `inventory.md` §5's rule about names that lie is satisfied
//! by not doing any naming.
//!
//! **The gates do not move.** Provenance is read and both ledgers are checked
//! in `plan_add`, before this module is reached. A second transport that
//! skipped them would be a hole in the first one's rules.

use crate::intake::download::Progress;
use crate::intake::run::Build;
use crate::inventory::artifact::Format;

/// Ollama's CLI. Its whole intake surface, as `lms` is LM Studio's.
pub const BINARY: &str = "ollama";

/// The `hf.co/…` name that pulls this build, or why it cannot be pulled.
///
/// The quantisation tag is carried because omitting it lets Ollama choose its
/// own default — a different build from the one that was priced and admitted,
/// which would make the figure the user was shown describe a file they did not
/// get. A build whose name carries no quantisation is pulled untagged, because
/// inventing a tag would name a build that may not exist in the repo.
pub fn transport_for(repo_id: &str, build: &Build) -> Result<String, String> {
    let Some(first) = build.files.first() else {
        return Err("this build has no files".to_string());
    };
    if first.format != Format::Gguf {
        return Err(format!(
            "ollama pulls GGUF from Hugging Face; `{}` is {:?}, which reaches Ollama \
             through its own library rather than through a repo id",
            first.name, first.format
        ));
    }
    if build.files.len() > 1 {
        // `hf.co/…` resolves to one file. A split build pulled through it is
        // one part wearing the whole build's name — the same failure
        // `run::shard_of` exists to prevent on the download path.
        return Err(format!(
            "`{}` is a {}-part build, and an hf.co pull resolves to a single file; \
             nothing would be able to load one part",
            first.name,
            build.files.len()
        ));
    }
    Ok(match quantisation(&first.name) {
        Some(tag) => format!("hf.co/{repo_id}:{tag}"),
        None => format!("hf.co/{repo_id}"),
    })
}

/// The quantisation label out of a GGUF filename — `…-Q4_K_M.gguf` -> `Q4_K_M`.
///
/// Taken from the **last** quant-shaped segment, because a repo name can carry
/// something that looks like one (`Qwen3-Q8-Distill-Q4_K_M.gguf`) and the tag
/// that matters is the file's own, which is always last.
///
/// Three families, all seen in the wild on 2026-09-11:
/// `Q4_K_M` (llama.cpp's k-quants), `IQ2_XXS` (importance-matrix quants, which
/// are not `Q`-prefixed), and unsloth's `UD-` dynamic quants, where the prefix
/// is part of the published tag rather than part of the model name. Dropping
/// any of them yields an untagged pull, and an untagged pull is Ollama choosing
/// its own default — a different build from the one that was priced.
fn quantisation(file_name: &str) -> Option<String> {
    let stem = file_name.strip_suffix(".gguf")?;
    let segments: Vec<&str> = stem.split('-').collect();
    let last = segments.iter().rposition(|seg| is_quant_shaped(seg))?;
    // `UD` is a prefix on the tag, not a word in the model's name.
    let from = match last.checked_sub(1) {
        Some(prev) if segments[prev].eq_ignore_ascii_case("UD") => prev,
        _ => last,
    };
    Some(segments[from..].join("-").to_ascii_uppercase())
}

/// `Q4_K_M`, `IQ2_XXS`, `BF16`. A digit after the `Q` is what separates a
/// quantisation from a word that merely starts with one.
fn is_quant_shaped(segment: &str) -> bool {
    let upper = segment.to_ascii_uppercase();
    if matches!(upper.as_str(), "BF16" | "F16" | "F32") {
        return true;
    }
    let body = upper.strip_prefix("IQ").or_else(|| upper.strip_prefix('Q'));
    body.is_some_and(|b| b.starts_with(|c: char| c.is_ascii_digit()))
}

pub fn pull_argv(model: &str) -> Vec<String> {
    vec![BINARY.to_string(), "pull".to_string(), model.to_string()]
}

/// Run a pull, reporting it in the same documents a download reports.
///
/// A caller polling a job must never have to special-case the provider, so this
/// speaks `Progress` rather than inventing a second vocabulary. Ollama writes
/// human progress to stderr with carriage returns; the percentages it prints
/// become `Advanced` events and everything else is dropped — a progress bar
/// redrawn over itself is exactly what the NDJSON contract exists to replace.
pub fn run_pull(
    argv: &[String],
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(), String> {
    use std::io::BufReader;
    use std::process::{Command, Stdio};
    use std::sync::mpsc;

    let model = argv.last().cloned().unwrap_or_default();
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                // Named, never silent: harmony must not report a pull it did
                // not perform.
                format!("`{}` is not on PATH; Ollama has no other intake surface", argv[0])
            } else {
                e.to_string()
            }
        })?;

    on_progress(Progress::Started {
        file: model.clone(),
        bytes_total: None,
        target: format!("{BINARY} store"),
    });

    // Both streams, on their own threads. Which one Ollama writes progress to
    // depends on whether it believes it has a terminal, and reading one to
    // completion before the other can deadlock on a full pipe — so neither is
    // assumed and neither is read second.
    let (tx, rx) = mpsc::channel::<String>();
    let mut readers = Vec::new();
    for stream in [
        child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || {
            for line in split_progress(BufReader::new(stream)) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        }));
    }
    drop(tx);

    // The last lines of whichever stream spoke are the whole error message
    // when this fails, so a bounded tail is kept rather than the entire log.
    let mut tail: Vec<String> = Vec::new();
    for line in rx {
        if let Some((done, total)) = parsed_progress(&line) {
            on_progress(Progress::Advanced { bytes_done: done, bytes_total: total });
        }
        tail.push(line);
        if tail.len() > 12 {
            tail.remove(0);
        }
    }
    for r in readers {
        let _ = r.join();
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        let reason = tail
            .iter()
            .rev()
            .find(|l| !l.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| {
                format!("`{}` exited {}", argv.join(" "), status.code().unwrap_or(-1))
            });
        on_progress(Progress::Failed { file: model, reason: reason.clone() });
        return Err(reason);
    }

    on_progress(Progress::Finished { file: model.clone(), bytes_done: 0, path: model });
    Ok(())
}

/// Ollama redraws one progress line with `\r`. Split on both, so a redrawn bar
/// yields one entry per update rather than one enormous final line.
fn split_progress<R: std::io::BufRead>(reader: R) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for byte in reader.bytes().map_while(Result::ok) {
        match byte {
            b'\r' | b'\n' => {
                if !current.trim().is_empty() {
                    out.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
            b => current.push(b as char),
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

/// `pulling 1a2b3c... 50% ▕███ ▏ 1.0 GB/2.0 GB` -> `(done, total)`.
///
/// Sizes rather than the percentage, because a percentage cannot be turned back
/// into bytes and the caller's job table is denominated in them. `None` when
/// the line is not a transfer line at all, which is most of them.
fn parsed_progress(line: &str) -> Option<(u64, Option<u64>)> {
    let (left, right) = line.rsplit_once('/')?;
    let done = trailing_size(left)?;
    let total = leading_size(right);
    Some((done, total))
}

fn trailing_size(s: &str) -> Option<u64> {
    let trimmed = s.trim_end();
    let unit_end = trimmed.len();
    let unit_start = trimmed.rfind(|c: char| c.is_ascii_digit() || c == '.')? + 1;
    let unit = trimmed[unit_start..unit_end].trim();
    let rest = trimmed[..unit_start].trim_end();
    let num_start = rest.rfind(|c: char| !(c.is_ascii_digit() || c == '.')).map(|i| i + 1).unwrap_or(0);
    bytes_of(rest[num_start..].trim(), unit)
}

fn leading_size(s: &str) -> Option<u64> {
    let t = s.trim_start();
    let num_end = t.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
    let rest = t[num_end..].trim_start();
    let unit_end = rest.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(rest.len());
    bytes_of(&t[..num_end], &rest[..unit_end])
}

fn bytes_of(number: &str, unit: &str) -> Option<u64> {
    let n: f64 = number.parse().ok()?;
    let scale = match unit.to_ascii_uppercase().as_str() {
        "B" => 1.0,
        "KB" | "KIB" => 1024.0,
        "MB" | "MIB" => 1024.0 * 1024.0,
        "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((n * scale) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quantisation_is_read_off_the_end_of_the_file_name() {
        assert_eq!(quantisation("Qwen3.8-4B-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quantisation("y-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quantisation("y-BF16.gguf").as_deref(), Some("BF16"));
    }

    /// A repo name that itself contains something quant-shaped must not win
    /// over the file's own tag, which is always last.
    #[test]
    fn an_earlier_quant_shaped_word_does_not_win() {
        assert_eq!(quantisation("Qwen3-Q8-Distill-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
    }

    /// Importance-matrix quants are not `Q`-prefixed. Found running it against
    /// `unsloth/Qwen3.5-4B-GGUF`, whose smallest build is `UD-IQ2_XXS` — the
    /// first reader dropped the tag entirely and would have pulled whatever
    /// Ollama defaults to instead of the build that was priced.
    #[test]
    fn an_importance_matrix_quant_is_recognised() {
        assert_eq!(quantisation("Qwen3.5-4B-IQ2_XXS.gguf").as_deref(), Some("IQ2_XXS"));
    }

    /// And unsloth's `UD-` prefix is part of the published tag, not part of
    /// the model's name.
    #[test]
    fn an_unsloth_dynamic_quant_keeps_its_prefix() {
        assert_eq!(quantisation("Qwen3.5-4B-UD-IQ2_XXS.gguf").as_deref(), Some("UD-IQ2_XXS"));
        assert_eq!(quantisation("Qwen3.5-4B-UD-Q4_K_XL.gguf").as_deref(), Some("UD-Q4_K_XL"));
    }

    /// A word that merely starts with Q is not a quantisation.
    #[test]
    fn a_word_starting_with_q_is_not_a_quantisation() {
        assert_eq!(quantisation("Qwen3-4B.gguf"), None);
    }

    #[test]
    fn a_name_with_no_quantisation_yields_none() {
        assert_eq!(quantisation("model.gguf"), None);
        assert_eq!(quantisation("model.safetensors"), None);
    }

    #[test]
    fn a_transfer_line_becomes_bytes_rather_than_a_percentage() {
        let (done, total) = parsed_progress("pulling 1a2b3c... 50% ▕██ ▏ 1.0 GB/2.0 GB").unwrap();
        assert_eq!(done, 1024 * 1024 * 1024);
        assert_eq!(total, Some(2 * 1024 * 1024 * 1024));
    }

    #[test]
    fn a_line_that_is_not_a_transfer_is_not_progress() {
        assert!(parsed_progress("pulling manifest").is_none());
        assert!(parsed_progress("success").is_none());
    }

    /// A redrawn bar is many updates, not one long line.
    #[test]
    fn carriage_returns_split_into_separate_updates() {
        let lines = split_progress(std::io::Cursor::new("a\rb\rc\n"));
        assert_eq!(lines, vec!["a", "b", "c"]);
    }
}
