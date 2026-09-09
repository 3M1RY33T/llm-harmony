use std::path::{Path, PathBuf};

use super::{home, StoreScanner};
use crate::inventory::artifact::{Artifact, FileKey, Format, Store};

pub struct OllamaStore;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub model_digest: String,
    pub bytes: u64,
}

const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Ollama uses an OCI-style manifest. Only the `image.model` layer is weights;
/// the `license` and `params` layers are metadata and must not be counted as
/// model bytes.
pub fn parse_manifest(json: &str) -> Option<Manifest> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let layers = v["layers"].as_array()?;
    let model = layers
        .iter()
        .find(|l| l["mediaType"].as_str() == Some(MODEL_MEDIA_TYPE))?;
    Some(Manifest {
        model_digest: model["digest"].as_str()?.to_string(),
        bytes: model["size"].as_u64().unwrap_or(0),
    })
}

/// `sha256:abcd` -> `<root>/blobs/sha256-abcd`
pub fn blob_path(root: &Path, digest: &str) -> PathBuf {
    root.join("blobs").join(digest.replace(':', "-"))
}

impl StoreScanner for OllamaStore {
    fn store(&self) -> Store {
        Store::Ollama
    }

    fn root(&self) -> Option<PathBuf> {
        Some(home()?.join(".ollama/models"))
    }

    /// One artifact per tag, each pointing at the blob it references. Two tags
    /// sharing a digest share a `FileKey`, so `total_unique_bytes` counts the
    /// allocation once without any Ollama-specific logic.
    fn scan(&self, root: &Path) -> Vec<Artifact> {
        let manifests = root.join("manifests");
        if !manifests.exists() {
            return Vec::new();
        }
        walkdir::WalkDir::new(&manifests)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| {
                let text = std::fs::read_to_string(e.path()).ok()?;
                let m = parse_manifest(&text)?;
                let blob = blob_path(root, &m.model_digest);

                // `registry.ollama.ai/library/nomic-embed-text/latest`
                //   -> `nomic-embed-text:latest`
                let rel = e.path().strip_prefix(&manifests).ok()?;
                let comps: Vec<String> = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                    .collect();
                let tag = comps.last()?.clone();
                let name = comps.get(comps.len().checked_sub(2)?)?.clone();
                let name_hint = format!("{name}:{tag}");

                Some(Artifact {
                    id: format!("ollama:{name_hint}"),
                    path: blob.clone(),
                    store: Store::Ollama,
                    format: Format::Gguf,
                    bits: None,
                    bytes: m.bytes,
                    key: FileKey::of(&blob),
                    is_link: false,
                    name_hint,
                })
            })
            .collect()
    }
}
