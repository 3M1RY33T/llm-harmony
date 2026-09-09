use std::path::{Path, PathBuf};

use super::{home, scan_dir, StoreScanner};
use crate::inventory::artifact::{bits_from_name, Artifact, Store};

pub struct LmStudioStore;

impl StoreScanner for LmStudioStore {
    fn store(&self) -> Store {
        Store::LmStudio
    }

    fn root(&self) -> Option<PathBuf> {
        Some(home()?.join(".lmstudio/models"))
    }

    /// Layout is `{publisher}/{repo}/...`, so the hint is the first two
    /// components rather than the filename -- `model.q4_k_m.gguf` alone
    /// identifies nothing.
    fn scan(&self, root: &Path) -> Vec<Artifact> {
        let mut out = scan_dir(root, Store::LmStudio);
        for a in &mut out {
            if let Some(hint) = publisher_repo(root, &a.path) {
                a.bits = a.bits.or_else(|| bits_from_name(&hint));
                a.name_hint = hint;
            }
        }
        out
    }
}

fn publisher_repo(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let mut it = rel.components();
    let publisher = it.next()?.as_os_str().to_string_lossy().to_string();
    let repo = it.next()?.as_os_str().to_string_lossy().to_string();
    Some(format!("{publisher}/{repo}"))
}
