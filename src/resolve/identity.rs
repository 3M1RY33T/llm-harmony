use crate::inventory::artifact::{ArtifactId, FileKey, Provenance, Store};
use crate::inventory::Inventory;
use crate::ledger::{Ledger, Outcome};
use crate::provider::{ProviderKind, State};

/// One place a model could be served from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub provider: ProviderKind,
    pub provider_model_id: String,
    pub url: String,
    pub state: State,
    /// The artifact on disk, when it could be identified.
    pub artifact: Option<ArtifactId>,
    /// How the artifact was identified. `Recorded` when the provider told us
    /// the path; `Inferred` when it was derived.
    pub provenance: Provenance,
}

/// LM Studio's model id, derived from its store layout.
///
/// It publishes no path, but its layout is `{publisher}/{repo}` and the served
/// id is `repo` lowercased with `-gguf` removed. `-mlx` is **kept**: verified
/// live 2026-09-10 against both store directories, `LocateAnything-3B-MLX`
/// serves as `locateanything-3b-mlx` while
/// `Qwen3-14B-...-Distill-GGUF` serves as `qwen3-14b-...-distill`.
///
/// This is a derivation, not a lookup, so candidates built from it are marked
/// `Inferred` -- but it is exact where slice 2's generic `canonical_name`
/// bridged only one of three cross-provider cases.
pub fn lmstudio_id_from_store_hint(hint: &str) -> Option<String> {
    let repo = hint.split_once('/').map(|(_, r)| r).unwrap_or(hint);
    let lowered = repo.to_ascii_lowercase();
    Some(lowered.strip_suffix("-gguf").unwrap_or(&lowered).to_string())
}

/// Every provider that could serve `model_ref`, in no particular order.
///
/// Matching is by artifact allocation wherever the provider reports a path:
/// `(dev, ino)` equality, so a pool symlink and its target are one artifact.
/// Falls back to the served id itself, which is always exact for the provider
/// that reported it.
pub fn candidates(ledger: &Ledger, inventory: &Inventory, model_ref: &str) -> Vec<Candidate> {
    let want = model_ref.to_ascii_lowercase();

    // Artifact index: allocation -> id, from the disk ledger.
    let by_key: std::collections::HashMap<FileKey, ArtifactId> = inventory
        .artifacts
        .iter()
        .filter_map(|a| a.key.map(|k| (k, a.id.clone())))
        .collect();

    let mut out = Vec::new();
    for row in &ledger.rows {
        let Outcome::Ok(models) = &row.outcome else {
            continue;
        };
        for m in models {
            if m.id.to_ascii_lowercase() != want {
                continue;
            }
            let (artifact, provenance) = match m.artifact_path.as_deref() {
                Some(path) => (
                    FileKey::of(std::path::Path::new(path)).and_then(|k| by_key.get(&k).cloned()),
                    Provenance::Recorded,
                ),
                None if row.kind == ProviderKind::Ollama => (
                    ollama_artifact(inventory, &m.id),
                    Provenance::Inferred {
                        basis: "matched by Ollama's manifest name".to_string(),
                    },
                ),
                None if row.kind == ProviderKind::LmStudio => (
                    lmstudio_artifact(inventory, &m.id, m.model_type.as_deref()),
                    Provenance::Inferred {
                        basis: "derived from LM Studio's store layout".to_string(),
                    },
                ),
                None => (
                    None,
                    Provenance::Inferred {
                        basis: "provider publishes no artifact path".to_string(),
                    },
                ),
            };
            out.push(Candidate {
                provider: row.kind,
                provider_model_id: m.id.clone(),
                url: row.url.clone(),
                state: m.state,
                artifact,
                provenance,
            });
        }
    }
    out
}

/// LM Studio serves embedding models under a prefix its store layout does
/// not carry: `nomic-ai/nomic-embed-text-v1.5-GGUF` on disk becomes
/// `text-embedding-nomic-embed-text-v1.5` on the wire. Verified 2026-09-11.
///
/// Gated on the provider's own `type` rather than on the name starting with
/// `text-embedding-`, so a model merely *named* that way is never bridged to
/// an unrelated directory.
const EMBEDDING_PREFIX: &str = "text-embedding-";

fn lmstudio_artifact(
    inventory: &Inventory,
    served_id: &str,
    model_type: Option<&str>,
) -> Option<ArtifactId> {
    inventory
        .artifacts
        .iter()
        .filter(|a| a.store == Store::LmStudio)
        .find(|a| {
            let Some(derived) = lmstudio_id_from_store_hint(&a.name_hint) else {
                return false;
            };
            let want = served_id.to_ascii_lowercase();
            if derived == want {
                return true;
            }
            model_type == Some("embeddings") && want == format!("{EMBEDDING_PREFIX}{derived}")
        })
        .map(|a| a.id.clone())
}

/// The blob an Ollama model's manifest names.
///
/// Ollama publishes a `digest` on `/api/tags`, but it is the **manifest's**
/// digest, not the weights' -- verified 2026-09-11: the API reported
/// `0a109f42...` while the model layer inside that manifest is `970aa74c...`.
/// Resolving one to the other means reading the manifest, which is slice 2's
/// scanner's job and which it already does.
///
/// So the join is by name: the scanner names each artifact after the manifest
/// path it came from (`<name>:<tag>`), and that is precisely the id the API
/// serves. An inference, and marked as one -- it matches a name rather than an
/// allocation -- but a name inside Ollama's own store rather than a guess about
/// what a file contains.
fn ollama_artifact(inventory: &Inventory, served_id: &str) -> Option<ArtifactId> {
    let want = served_id.to_ascii_lowercase();
    inventory
        .artifacts
        .iter()
        .filter(|a| a.store == Store::Ollama)
        .find(|a| a.name_hint.to_ascii_lowercase() == want)
        .map(|a| a.id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::Artifact;

    /// Verified live 2026-09-10 against both LM Studio store directories.
    #[test]
    fn lmstudio_ids_derive_from_the_store_layout() {
        assert_eq!(
            lmstudio_id_from_store_hint("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF")
                .as_deref(),
            Some("qwen3-14b-claude-4.5-opus-high-reasoning-distill")
        );
    }

    /// `-MLX` is part of the served id; `-GGUF` is not. Getting this backwards
    /// misses one of the two models on this machine.
    #[test]
    fn the_mlx_suffix_is_kept_while_gguf_is_stripped() {
        assert_eq!(
            lmstudio_id_from_store_hint("andai-labs/LocateAnything-3B-MLX").as_deref(),
            Some("locateanything-3b-mlx")
        );
    }

    fn artifact(store: Store, id: &str, name_hint: &str) -> Artifact {
        Artifact {
            id: id.to_string(),
            path: std::path::PathBuf::from(format!("/store/{name_hint}")),
            store,
            format: crate::inventory::artifact::Format::Gguf,
            bits: None,
            bytes: 1,
            key: None,
            is_link: false,
            name_hint: name_hint.to_string(),
        }
    }

    fn inventory(artifacts: Vec<Artifact>) -> Inventory {
        let graph = crate::inventory::graph::Graph::build(&artifacts);
        Inventory { artifacts, graph, identities: Vec::new() }
    }

    /// Ollama's scanner already resolves a manifest to the blob holding the
    /// weights and names the artifact after the manifest -- which is exactly
    /// the id the API serves. Before this bridge existed the two never met,
    /// so every Ollama model priced as a declared floor and was refused.
    #[test]
    fn an_ollama_model_finds_the_blob_its_manifest_names() {
        let inv = inventory(vec![artifact(
            Store::Ollama,
            "ollama:nomic-embed-text:latest",
            "nomic-embed-text:latest",
        )]);
        assert_eq!(
            ollama_artifact(&inv, "nomic-embed-text:latest").as_deref(),
            Some("ollama:nomic-embed-text:latest")
        );
    }

    /// Tags matter: `qwen3:14b` and `qwen3:8b` are different models with very
    /// different footprints, and matching on the bare name would confuse them.
    #[test]
    fn an_ollama_match_is_by_name_and_tag_together() {
        let inv = inventory(vec![artifact(Store::Ollama, "ollama:qwen3:14b", "qwen3:14b")]);
        assert!(ollama_artifact(&inv, "qwen3:8b").is_none());
        assert!(ollama_artifact(&inv, "qwen3").is_none());
    }

    /// And it never reaches into another store: a same-named model in the
    /// LM Studio tree is a different artifact with a different quantisation.
    #[test]
    fn an_ollama_match_does_not_cross_stores() {
        let inv = inventory(vec![artifact(Store::LmStudio, "lms:x", "nomic-embed-text:latest")]);
        assert!(ollama_artifact(&inv, "nomic-embed-text:latest").is_none());
    }

    /// LM Studio serves this one as `text-embedding-nomic-embed-text-v1.5`
    /// while its directory is `nomic-ai/nomic-embed-text-v1.5-GGUF`. Before
    /// 2026-09-11 the artifact was in neither the scanned roots nor reachable
    /// by name, so the ledger watched LM Studio serve a model it could not
    /// price.
    #[test]
    fn an_lmstudio_embedding_model_bridges_across_its_served_prefix() {
        let inv = inventory(vec![artifact(
            Store::LmStudio,
            "lms:nomic",
            "nomic-ai/nomic-embed-text-v1.5-GGUF",
        )]);
        assert_eq!(
            lmstudio_artifact(&inv, "text-embedding-nomic-embed-text-v1.5", Some("embeddings"))
                .as_deref(),
            Some("lms:nomic")
        );
    }

    /// And only for a model the provider calls an embedding model. A chat
    /// model that merely happens to be named that way must not be bridged to
    /// an unrelated directory -- names lie, per docs/field-notes.md.
    #[test]
    fn the_embedding_prefix_is_not_stripped_for_other_model_types() {
        let inv = inventory(vec![artifact(
            Store::LmStudio,
            "lms:nomic",
            "nomic-ai/nomic-embed-text-v1.5-GGUF",
        )]);
        assert!(lmstudio_artifact(&inv, "text-embedding-nomic-embed-text-v1.5", Some("llm")).is_none());
        assert!(lmstudio_artifact(&inv, "text-embedding-nomic-embed-text-v1.5", None).is_none());
    }

    /// The ordinary case keeps working: a chat model matches its directory
    /// directly, with no prefix involved.
    #[test]
    fn a_chat_model_still_matches_its_directory_exactly() {
        let inv = inventory(vec![artifact(
            Store::LmStudio,
            "lms:qwen",
            "TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF",
        )]);
        assert_eq!(
            lmstudio_artifact(&inv, "qwen3-14b-claude-4.5-opus-high-reasoning-distill", Some("llm"))
                .as_deref(),
            Some("lms:qwen")
        );
    }
}
