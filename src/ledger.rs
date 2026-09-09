use crate::adapters::adapter_for;
use crate::config::Config;
use crate::http::Http;
use crate::memory::{footprint_for_port, port_from_url, Machine, ProcessSample, ProcessTree};
use crate::provider::{LoadedModel, ProbeError, ProviderKind, State};

/// Adjacently tagged so a consumer can branch on `status` rather than on
/// whether `outcome` happens to be an array or an object.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", content = "detail", rename_all = "kebab-case")]
pub enum Outcome {
    Ok(Vec<LoadedModel>),
    Failed(ProbeError),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProviderRow {
    pub kind: ProviderKind,
    pub url: String,
    pub outcome: Outcome,
    pub pids: Vec<u32>,
    /// `max(phys_footprint, rss)` per process, summed. See memory::combined.
    pub footprint_bytes: Option<u64>,
    /// Components, so the estimator can revisit the choice of measure.
    pub phys_footprint_bytes: Option<u64>,
    pub rss_bytes: Option<u64>,
    /// Per-process breakdown, for attribution. Empty when unreadable.
    #[serde(default)]
    pub processes: Vec<ProcessSample>,
}

impl ProviderRow {
    pub fn loaded(&self) -> Vec<&LoadedModel> {
        match &self.outcome {
            Outcome::Ok(models) => models.iter().filter(|m| m.state == State::Loaded).collect(),
            Outcome::Failed(_) => Vec::new(),
        }
    }

    /// Summed weights of resident models -- `None` unless *every* resident model
    /// publishes a size. A partial sum would understate, and understating is
    /// the failure mode that crashes the machine.
    pub fn weights_bytes(&self) -> Option<u64> {
        let loaded = self.loaded();
        if loaded.is_empty() {
            return None;
        }
        loaded.iter().map(|m| m.weights_bytes).sum()
    }

    /// KV cache, prefix cache, and activations: what the weights figure omits.
    pub fn gap_bytes(&self) -> Option<u64> {
        let fp = self.footprint_bytes?;
        let w = self.weights_bytes()?;
        Some(fp.saturating_sub(w))
    }
}

/// Bumped whenever the serialised shape changes in a way a consumer would
/// notice. Consumers should refuse a schema they do not know.
pub const SCHEMA: u32 = 1;

fn schema() -> u32 {
    SCHEMA
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Ledger {
    #[serde(default = "schema", rename = "schema")]
    pub schema_version: u32,
    pub machine: Machine,
    pub rows: Vec<ProviderRow>,
}

impl Ledger {
    /// Poll every configured provider. Concurrent because four sequential
    /// timeouts against dead ports would otherwise dominate the runtime.
    pub fn assemble(config: &Config, http: &Http, machine: Machine) -> Ledger {
        Self::assemble_with(config, http, machine, footprint_for_port)
    }

    /// `footprint` is injected so tests never touch real processes.
    pub fn assemble_with<F>(config: &Config, http: &Http, machine: Machine, footprint: F) -> Ledger
    where
        F: Fn(u16) -> Option<ProcessTree> + Sync,
    {
        let footprint = &footprint;
        let rows = std::thread::scope(|scope| {
            let handles: Vec<_> = config
                .providers
                .iter()
                .map(|p| {
                    scope.spawn(move || {
                        let adapter = adapter_for(p.kind);
                        let outcome = match adapter.list(http, &p.url) {
                            Ok(models) => Outcome::Ok(models),
                            Err(e) => Outcome::Failed(e),
                        };
                        // Only attribute memory to a provider that answered.
                        let tree = match (&outcome, port_from_url(&p.url)) {
                            (Outcome::Ok(_), Some(port)) => footprint(port),
                            _ => None,
                        };
                        ProviderRow {
                            kind: p.kind,
                            url: p.url.clone(),
                            outcome,
                            pids: tree.as_ref().map(|t| t.pids.clone()).unwrap_or_default(),
                            footprint_bytes: tree.as_ref().and_then(|t| t.footprint_bytes),
                            phys_footprint_bytes: tree.as_ref().and_then(|t| t.phys_footprint_bytes),
                            rss_bytes: tree.as_ref().and_then(|t| t.rss_bytes),
                            processes: tree
                                .as_ref()
                                .map(|t| t.processes.clone())
                                .unwrap_or_default(),
                        }
                    })
                })
                .collect();

            handles
                .into_iter()
                .filter_map(|h| h.join().ok())
                .collect::<Vec<_>>()
        });

        Ledger {
            schema_version: SCHEMA,
            machine,
            rows,
        }
    }

    /// `None` only when no provider reported a readable footprint.
    pub fn total_footprint_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for r in &self.rows {
            if let Some(b) = r.footprint_bytes {
                total += b;
                any = true;
            }
        }
        any.then_some(total)
    }
}
