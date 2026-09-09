use std::path::{Path, PathBuf};

use super::{home, scan_dir, StoreScanner};
use crate::inventory::artifact::{bits_from_name, Artifact, Store};

pub struct VllmStore;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryEntry {
    pub name: String,
    pub path: String,
}

/// `models.yaml` names what the pool will serve. A name here is a REQUIRED_BY
/// edge: a stale path is a startup failure, per docs/inventory.md.
///
/// Note `path:` is not tilde-expanded by vllm-mlx itself -- a leading `~` is
/// read as part of a Hugging Face repo id. We record what is written.
pub fn parse_registry(yaml: &str) -> Vec<RegistryEntry> {
    let Ok(v) = serde_yaml::from_str::<serde_yaml::Value>(yaml) else {
        return Vec::new();
    };
    let Some(models) = v.get("models").and_then(|m| m.as_sequence()) else {
        return Vec::new();
    };
    models
        .iter()
        .filter_map(|m| {
            Some(RegistryEntry {
                name: m.get("name")?.as_str()?.to_string(),
                path: m.get("path")?.as_str()?.to_string(),
            })
        })
        .collect()
}

impl VllmStore {
    pub fn registry(&self, root: &Path) -> Vec<RegistryEntry> {
        std::fs::read_to_string(root.join("models.yaml"))
            .map(|s| parse_registry(&s))
            .unwrap_or_default()
    }
}

impl StoreScanner for VllmStore {
    fn store(&self) -> Store {
        Store::Vllm
    }

    fn root(&self) -> Option<PathBuf> {
        Some(home()?.join(".vllm-mlx"))
    }

    /// Scans `models/` only. The store root also contains `venv/`, `serve`,
    /// and `serve-pool`, none of which are artifacts.
    fn scan(&self, root: &Path) -> Vec<Artifact> {
        let models_root = root.join("models");
        let mut out = scan_dir(&models_root, Store::Vllm);
        for a in &mut out {
            if let Some(hint) = model_dir(&models_root, &a.path) {
                a.bits = a.bits.or_else(|| bits_from_name(&hint));
                a.name_hint = hint;
            }
        }
        out
    }
}

fn model_dir(models_root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(models_root).ok()?;
    Some(rel.components().next()?.as_os_str().to_string_lossy().to_string())
}
