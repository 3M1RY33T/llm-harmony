pub mod artifact;
pub mod graph;
pub mod identity;
pub mod plan;
pub mod safety;
pub mod scan;

use std::path::PathBuf;

use crate::config::Config;
use crate::http::Http;
use crate::inventory::artifact::{Artifact, Store};
use crate::inventory::graph::{Graph, Requirer};
use crate::inventory::identity::{group_with_aliases, ModelIdentity};
use crate::inventory::scan::{
    hf::HfCache, llamacpp::LlamaCppPool, lmstudio::LmStudioStore, ollama::OllamaStore,
    vllm::VllmStore, StoreScanner,
};
use crate::ledger::{Ledger, Outcome};
use crate::memory::Machine;

pub struct Inventory {
    pub artifacts: Vec<Artifact>,
    pub graph: Graph,
    pub identities: Vec<ModelIdentity>,
}

fn scanners() -> Vec<Box<dyn StoreScanner>> {
    vec![
        Box::new(HfCache),
        Box::new(LmStudioStore),
        Box::new(LlamaCppPool),
        Box::new(VllmStore),
        Box::new(OllamaStore),
    ]
}

impl Inventory {
    /// Filesystem only. No provider is contacted, so nothing is `ServedLive`,
    /// and the HF cache is treated as scanned because we cannot rule it out.
    pub fn scan_offline(roots: Option<Vec<(Store, PathBuf)>>) -> Inventory {
        Self::scan_fs(roots, true)
    }

    fn scan_fs(roots: Option<Vec<(Store, PathBuf)>>, llamacpp_may_be_running: bool) -> Inventory {
        let artifacts: Vec<Artifact> = match roots {
            Some(explicit) => explicit
                .into_iter()
                .flat_map(|(store, root)| {
                    scanners()
                        .into_iter()
                        .find(|s| s.store() == store)
                        .map(|s| s.scan(&root))
                        .unwrap_or_default()
                })
                .collect(),
            None => scanners()
                .into_iter()
                .flat_map(|s| match s.root() {
                    Some(r) => s.scan(&r),
                    None => Vec::new(),
                })
                .collect(),
        };

        let mut graph = Graph::build(&artifacts);

        // llama.cpp's router scans the HF cache at runtime, so every entry in
        // it is a serving dependency *while the router is up*. Offline we
        // cannot know, so we assume the worst; live, we check, because
        // refusing the whole 69 GB cache unconditionally makes `rm` useless.
        if llamacpp_may_be_running {
            for a in artifacts.iter().filter(|a| a.store == Store::HfCache) {
                graph.add_required_by(
                    a.id.clone(),
                    Requirer::Scanned {
                        provider: crate::provider::ProviderKind::LlamaCpp,
                    },
                );
            }
        }

        // vLLM-MLX names models in models.yaml, where a stale path is a
        // startup failure -- docs/inventory.md section 2, question 3.
        if let Some(vllm_root) = VllmStore.root() {
            for entry in VllmStore.registry(&vllm_root) {
                for a in artifacts.iter().filter(|a| a.store == Store::Vllm) {
                    if a.path.starts_with(&entry.path) || a.name_hint == entry.name {
                        graph.add_required_by(
                            a.id.clone(),
                            Requirer::Registry {
                                store: Store::Vllm.as_str().to_string(),
                                name: entry.name.clone(),
                            },
                        );
                    }
                }
            }
        }

        // Alias-driven, not name-driven: a shared allocation is proof of
        // identity that unlike names cannot override.
        let identities = group_with_aliases(&artifacts, &graph);
        Inventory {
            artifacts,
            graph,
            identities,
        }
    }

    /// Filesystem plus the live memory ledger, so resident models become
    /// `ServedLive` requirers and cannot be planned for removal.
    pub fn scan(config: &Config, http: &Http, machine: Machine) -> Inventory {
        let ledger = Ledger::assemble(config, http, machine);
        // With a live view we know whether the router is actually up.
        let llamacpp_up = ledger
            .rows
            .iter()
            .any(|r| r.kind == crate::provider::ProviderKind::LlamaCpp
                && matches!(r.outcome, Outcome::Ok(_)));
        let mut inv = Inventory::scan_fs(None, llamacpp_up);

        for row in &ledger.rows {
            let Outcome::Ok(models) = &row.outcome else {
                continue;
            };
            for m in models
                .iter()
                .filter(|m| m.state == crate::provider::State::Loaded)
            {
                // Match a resident model to artifacts by canonical name. This
                // is inference, and it is why `rm` refuses rather than warns.
                let want = identity::canonical_name(&m.id);
                let direct: Vec<String> = inv
                    .artifacts
                    .iter()
                    .filter(|a| identity::canonical_name(&a.name_hint) == want)
                    .map(|a| a.id.clone())
                    .collect();
                // Serving a file makes every path that reaches it a
                // dependency -- including a symlink in another store's pool,
                // whose name may not resemble the served model's at all.
                let mut ids = direct.clone();
                for d in &direct {
                    ids.extend(inv.graph.aliases_of(d));
                }
                // And every artifact the identity graph unified with them.
                for ident in &inv.identities {
                    if ident.artifacts.iter().any(|a| direct.contains(a)) {
                        ids.extend(ident.artifacts.iter().cloned());
                    }
                }
                ids.sort();
                ids.dedup();
                for id in ids {
                    inv.graph.add_required_by(
                        id,
                        Requirer::ServedLive {
                            provider: row.kind,
                            model_id: m.id.clone(),
                        },
                    );
                }
            }
        }
        inv
    }

    pub fn total_bytes(&self) -> u64 {
        scan::total_unique_bytes(&self.artifacts)
    }
}
