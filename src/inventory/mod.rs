pub mod artifact;
pub mod clean;
pub mod dupes;
pub mod graph;
pub mod identity;
pub mod plan;
pub mod safety;
pub mod scan;

use std::path::PathBuf;

use crate::config::Config;
use crate::http::Http;
use crate::inventory::artifact::{Artifact, ArtifactId, Format, Store};
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
    /// The artifact carrying a model's weights, given whatever artifact a provider
    /// pointed at.
    ///
    /// A reported path is not necessarily the weights: LM Studio's candidate for a
    /// GGUF model resolves to the `config.json` beside it, all 1,940 bytes of it,
    /// which priced a 9 GB model at 1.3 GB. Found 2026-09-11, the first time a
    /// computed figure was not masked by a measurement.
    ///
    /// The rule is deliberately narrow -- **the largest weights-format file in the
    /// same directory** -- because the two wider rules are both wrong:
    ///
    /// * the artifact as given prices a config file;
    /// * the largest artifact in the model's *identity group* crosses builds, and
    ///   this machine's identities merge a Q4_K_M with a Q8_0. It priced LM
    ///   Studio's 9 GB Q4 as llama.cpp's 15.7 GB Q8.
    pub fn weights_artifact(&self, id: &ArtifactId) -> Option<&Artifact> {
        let given = self.artifacts.iter().find(|a| &a.id == id)?;
        let is_weights =
            |f: Format| f.is_weights();

        if is_weights(given.format) {
            return Some(given);
        }

        let dir = given.path.parent()?;
        self.artifacts
            .iter()
            .filter(|a| is_weights(a.format) && a.path.parent() == Some(dir))
            .max_by_key(|a| a.bytes)
    }

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
                .flat_map(|s| {
                    s.roots().iter().flat_map(|r| s.scan(r)).collect::<Vec<_>>()
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

#[cfg(test)]
mod weights_artifact_tests {
    use super::*;
    use crate::inventory::graph::Graph;

    fn art(path: &str, format: Format, bytes: u64) -> Artifact {
        Artifact {
            id: path.to_string(),
            path: PathBuf::from(path),
            store: Store::LmStudio,
            format,
            bits: None,
            bytes,
            key: None,
            is_link: false,
            name_hint: path.to_string(),
        }
    }

    fn inventory(artifacts: Vec<Artifact>) -> Inventory {
        let graph = Graph::build(&artifacts);
        Inventory { artifacts, graph, identities: Vec::new() }
    }

    /// The defect, 2026-09-11: LM Studio's candidate resolves to the
    /// `config.json` beside the GGUF, and pricing it reported a 9 GB model as
    /// 1.3 GB -- the KV cache alone.
    #[test]
    fn a_config_file_resolves_to_the_weights_beside_it() {
        let inv = inventory(vec![
            art("/models/TeichAI/Qwen3-GGUF/config.json", Format::Other, 1_940),
            art("/models/TeichAI/Qwen3-GGUF/qwen3-14b.q4_k_m.gguf", Format::Gguf, 9_000_000_000),
        ]);
        let a = inv.weights_artifact(&"/models/TeichAI/Qwen3-GGUF/config.json".to_string());
        assert_eq!(a.unwrap().bytes, 9_000_000_000);
    }

    /// A weights file is its own answer: llama.cpp and vLLM report the real
    /// path, and second-guessing them would be how a correct answer gets lost.
    #[test]
    fn a_weights_file_is_returned_as_itself() {
        let inv = inventory(vec![
            art("/m/a.gguf", Format::Gguf, 9_000_000_000),
            art("/m/b.gguf", Format::Gguf, 15_700_000_000),
        ]);
        let a = inv.weights_artifact(&"/m/a.gguf".to_string());
        assert_eq!(a.unwrap().bytes, 9_000_000_000, "not the larger sibling");
    }

    /// The rule stops at the directory. A Q4 and a Q8 of one model are the
    /// same *identity* and different builds; crossing that line priced a 9 GB
    /// Q4 as a 15.7 GB Q8.
    #[test]
    fn the_search_does_not_cross_into_another_directory() {
        let inv = inventory(vec![
            art("/lmstudio/m/config.json", Format::Other, 1_940),
            art("/lmstudio/m/q4.gguf", Format::Gguf, 9_000_000_000),
            art("/llamacpp/models/q8.gguf", Format::Gguf, 15_700_000_000),
        ]);
        let a = inv.weights_artifact(&"/lmstudio/m/config.json".to_string());
        assert_eq!(a.unwrap().bytes, 9_000_000_000);
    }

    #[test]
    fn a_config_beside_no_weights_at_all_is_none() {
        let inv = inventory(vec![art("/m/config.json", Format::Other, 1_940)]);
        assert!(inv.weights_artifact(&"/m/config.json".to_string()).is_none());
    }
}
