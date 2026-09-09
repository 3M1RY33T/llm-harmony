use std::io::Write;
use std::path::{Path, PathBuf};

use crate::ledger::{Ledger, Outcome};
use crate::provider::{ProviderKind, State};

/// One provider's memory cost at one instant.
///
/// The estimator in slice 3 learns from these. The valuable shapes are
/// `loaded == 0` (a provider's idle baseline) and `loaded == 1` (a footprint
/// attributable to exactly one model at a known context). Both are ordinary
/// `status` output, which is why any caller can contribute them and no daemon
/// is required to start collecting.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Observation {
    pub schema: u32,
    /// Unix seconds. Passed in rather than read, so this is testable.
    pub at: u64,
    pub provider: ProviderKind,
    pub loaded: usize,
    /// Set only when exactly one model is resident.
    pub model: Option<String>,
    /// The serving window, never a capability figure.
    pub context_tokens: Option<u32>,
    /// True when the footprint can be assigned: no models, or exactly one.
    pub attributable: bool,
    /// `max(phys_footprint, rss)` per process, summed.
    pub footprint_bytes: u64,
    pub phys_footprint_bytes: Option<u64>,
    pub rss_bytes: Option<u64>,
    pub machine_total_bytes: u64,
    pub machine_used_bytes: u64,
    pub swap_used_bytes: u64,
}

/// Observations worth keeping from one ledger.
///
/// A provider that did not answer, or whose processes could not be read,
/// yields nothing: an observation without a measurement is not an observation.
pub fn observations(ledger: &Ledger, at: u64) -> Vec<Observation> {
    ledger
        .rows
        .iter()
        .filter_map(|row| {
            let Outcome::Ok(models) = &row.outcome else {
                return None;
            };
            let footprint_bytes = row.footprint_bytes?;

            let resident: Vec<_> = models.iter().filter(|m| m.state == State::Loaded).collect();
            let loaded = resident.len();
            let attributable = loaded <= 1;
            let (model, context_tokens) = match resident.as_slice() {
                [only] => (Some(only.id.clone()), only.context_tokens),
                _ => (None, None),
            };

            Some(Observation {
                schema: crate::ledger::SCHEMA,
                at,
                provider: row.kind,
                loaded,
                model,
                context_tokens,
                attributable,
                footprint_bytes,
                phys_footprint_bytes: row.phys_footprint_bytes,
                rss_bytes: row.rss_bytes,
                machine_total_bytes: ledger.machine.total_bytes,
                machine_used_bytes: ledger.machine.used_bytes,
                swap_used_bytes: ledger.machine.swap_used_bytes,
            })
        })
        .collect()
}

/// `~/.local/state/llm-harmony/observations.jsonl`
pub fn default_path() -> Option<PathBuf> {
    Some(
        std::env::var_os("HOME")
            .map(PathBuf::from)?
            .join(".local/state/llm-harmony/observations.jsonl"),
    )
}

/// Append one line per observation. Never truncates, never rewrites.
///
/// This is the only place llm-harmony writes to disk, and it writes only to
/// its own state file -- never into a model store.
pub fn append(path: &Path, obs: &[Observation]) -> std::io::Result<()> {
    if obs.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for o in obs {
        let line = serde_json::to_string(o).map_err(std::io::Error::other)?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Ledger, Outcome, ProviderRow};
    use crate::memory::Machine;
    use crate::provider::{LoadedModel, ProbeError, ProviderKind, State};

    fn machine() -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 20_000_000_000,
            swap_total_bytes: 6_442_450_944,
            swap_used_bytes: 5_100_000_000,
        }
    }

    fn model(id: &str, state: State, ctx: Option<u32>) -> LoadedModel {
        LoadedModel { id: id.into(), state, context_tokens: ctx, weights_bytes: None }
    }

    fn row(kind: ProviderKind, outcome: Outcome, fp: Option<u64>) -> ProviderRow {
        ProviderRow {
            kind,
            url: "http://127.0.0.1:1".into(),
            outcome,
            pids: vec![1],
            footprint_bytes: fp,
            phys_footprint_bytes: fp,
            rss_bytes: fp,
        }
    }

    fn ledger(rows: Vec<ProviderRow>) -> Ledger {
        Ledger { schema_version: crate::ledger::SCHEMA, machine: machine(), rows }
    }

    /// The high-value case: one model resident means
    /// `footprint - idle_baseline` is attributable to it, with no
    /// before/after delta needed.
    #[test]
    fn one_resident_model_is_an_attributable_observation() {
        let l = ledger(vec![row(
            ProviderKind::LmStudio,
            Outcome::Ok(vec![
                model("qwen3-14b", State::Loaded, Some(40960)),
                model("other", State::NotLoaded, None),
            ]),
            Some(9_600_000_000),
        )]);
        let obs = observations(&l, 1_757_000_000);
        assert_eq!(obs.len(), 1);
        assert!(obs[0].attributable);
        assert_eq!(obs[0].model.as_deref(), Some("qwen3-14b"));
        assert_eq!(obs[0].context_tokens, Some(40960));
    }

    /// Equally valuable: a provider with nothing loaded is its idle baseline,
    /// which the estimator must subtract. LM Studio's is ~0.58 GB.
    #[test]
    fn zero_resident_models_is_a_baseline_observation() {
        let l = ledger(vec![row(
            ProviderKind::LmStudio,
            Outcome::Ok(vec![model("x", State::NotLoaded, None)]),
            Some(620_000_000),
        )]);
        let obs = observations(&l, 1);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].loaded, 0);
        assert!(obs[0].attributable, "a baseline is attributable to nothing, which is the point");
        assert_eq!(obs[0].model, None);
    }

    /// Two resident models share one footprint and cannot be split.
    #[test]
    fn two_resident_models_are_recorded_but_not_attributable() {
        let l = ledger(vec![row(
            ProviderKind::LmStudio,
            Outcome::Ok(vec![
                model("a", State::Loaded, Some(8192)),
                model("b", State::Loaded, Some(8192)),
            ]),
            Some(12_000_000_000),
        )]);
        let obs = observations(&l, 1);
        assert_eq!(obs[0].loaded, 2);
        assert!(!obs[0].attributable);
        assert_eq!(obs[0].model, None, "no single model owns this footprint");
    }

    #[test]
    fn a_provider_that_did_not_answer_records_nothing() {
        let l = ledger(vec![row(
            ProviderKind::LlamaCpp,
            Outcome::Failed(ProbeError::NotListening),
            None,
        )]);
        assert!(observations(&l, 1).is_empty());
    }

    #[test]
    fn a_provider_with_an_unreadable_footprint_records_nothing() {
        let l = ledger(vec![row(
            ProviderKind::LmStudio,
            Outcome::Ok(vec![model("a", State::Loaded, Some(8192))]),
            None,
        )]);
        assert!(observations(&l, 1).is_empty(), "an observation with no measurement is not one");
    }

    #[test]
    fn appending_preserves_what_is_already_there() {
        let dir = std::env::temp_dir().join(format!("harmony-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("observations.jsonl");

        let l = ledger(vec![row(
            ProviderKind::Ollama,
            Outcome::Ok(vec![model("nomic", State::Loaded, Some(2048))]),
            Some(300_000_000),
        )]);
        append(&path, &observations(&l, 100)).unwrap();
        append(&path, &observations(&l, 200)).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "append, never overwrite");
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).expect("each line is valid JSON");
            assert_eq!(v["schema"], crate::ledger::SCHEMA);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
