use std::path::{Path, PathBuf};

use super::{home, scan_dir, StoreScanner};
use crate::inventory::artifact::{bits_from_name, Artifact, Store};

pub struct HfCache;

/// `models--BAAI--bge-small-en-v1.5` -> `BAAI/bge-small-en-v1.5`.
///
/// Only `models--` prefixed directories; `datasets--` and `CACHEDIR.TAG` are
/// not model repos.
pub fn repo_id_from_dir(dir_name: &str) -> Option<String> {
    let rest = dir_name.strip_prefix("models--")?;
    let (org, name) = rest.split_once("--")?;
    Some(format!("{org}/{name}"))
}

impl StoreScanner for HfCache {
    fn store(&self) -> Store {
        Store::HfCache
    }

    fn root(&self) -> Option<PathBuf> {
        Some(home()?.join(".cache/huggingface/hub"))
    }

    fn scan(&self, root: &Path) -> Vec<Artifact> {
        let mut out = scan_dir(root, Store::HfCache);
        // `refs/<branch>` holds a commit sha, and `CACHEDIR.TAG` marks the
        // cache root. Neither is a model artifact, and counting them makes the
        // store total disagree with itself by a few bytes per repo.
        // Keep only files that live inside a `models--org--name` repo and are
        // not HF's own bookkeeping. Found live 2026-09-09: `.locks/` and
        // `*.lock` otherwise appear as dozens of zero-byte "models".
        out.retain(|a| {
            let in_model_repo = repo_of(root, &a.path).is_some();
            let is_ref = a.path.components().any(|c| c.as_os_str() == "refs");
            let is_lock = a
                .path
                .to_string_lossy()
                .contains(".lock");
            let is_tag = a.path.file_name().map(|n| n == "CACHEDIR.TAG").unwrap_or(false);
            in_model_repo && !is_ref && !is_lock && !is_tag
        });
        // Re-hint each artifact with its repo id, so identity grouping has a
        // name to work with rather than an opaque blob hash.
        for a in &mut out {
            if let Some(repo) = repo_of(root, &a.path) {
                a.bits = a.bits.or_else(|| bits_from_name(&repo));
                a.name_hint = repo;
            }
        }
        out
    }
}

fn repo_of(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let first = rel.components().next()?.as_os_str().to_string_lossy();
    repo_id_from_dir(&first)
}
