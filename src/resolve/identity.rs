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
                None if row.kind == ProviderKind::LmStudio => (
                    lmstudio_artifact(inventory, &m.id),
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

fn lmstudio_artifact(inventory: &Inventory, served_id: &str) -> Option<ArtifactId> {
    inventory
        .artifacts
        .iter()
        .filter(|a| a.store == Store::LmStudio)
        .find(|a| {
            lmstudio_id_from_store_hint(&a.name_hint).as_deref()
                == Some(&served_id.to_ascii_lowercase())
        })
        .map(|a| a.id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
