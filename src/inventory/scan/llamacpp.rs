use std::path::{Path, PathBuf};

use super::{home, scan_dir, StoreScanner};
use crate::inventory::artifact::{Artifact, Store};

pub struct LlamaCppPool;

impl StoreScanner for LlamaCppPool {
    fn store(&self) -> Store {
        Store::LlamaCpp
    }

    fn roots(&self) -> Vec<PathBuf> {
        home().map(|h| vec![h.join(".llamacpp/models")]).unwrap_or_default()
    }

    /// A flat pool of mixed real files and symlinks into other stores. The
    /// filename is the whole identity here.
    fn scan(&self, root: &Path) -> Vec<Artifact> {
        scan_dir(root, Store::LlamaCpp)
    }
}
