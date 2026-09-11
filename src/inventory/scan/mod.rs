pub mod hf;
pub mod llamacpp;
pub mod lmstudio;
pub mod ollama;
pub mod vllm;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::inventory::artifact::{Artifact, FileKey, Store};

/// Directories that are never model artifacts.
///
/// `venv` is not defensive: `~/.vllm-mlx/venv` holds 47,588 files, and walking
/// it is both wrong and slow enough to notice.
const EXCLUDED_DIRS: [&str; 7] = [
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".cache_delete",
    "site-packages",
];

pub fn is_excluded(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        EXCLUDED_DIRS.contains(&s.as_ref())
    })
}

/// Every artifact-shaped file under `root`, or nothing if `root` is absent.
pub fn scan_dir(root: &Path, store: Store) -> Vec<Artifact> {
    if !root.exists() {
        return Vec::new();
    }
    walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !is_excluded(e.path()))
        .filter_map(|e| e.ok())
        .filter(|e| !e.file_type().is_dir())
        .filter_map(|e| Artifact::from_path(e.path(), store))
        .collect()
}

/// Bytes on disk, counting each allocation once however many paths reach it.
pub fn total_unique_bytes(artifacts: &[Artifact]) -> u64 {
    let mut seen: HashSet<FileKey> = HashSet::new();
    let mut total = 0u64;
    for a in artifacts {
        match a.key {
            Some(k) => {
                if seen.insert(k) {
                    total += a.bytes;
                }
            }
            // No key means the path could not be stat'd -- a broken symlink.
            // It occupies nothing.
            None => {}
        }
    }
    total
}

pub trait StoreScanner: Send + Sync {
    fn store(&self) -> Store;
    /// Where this store keeps artifacts, under `$HOME`.
    ///
    /// A list because LM Studio has two: the models you install, and the ones
    /// it ships with under `.internal/bundled-models`. Found 2026-09-11, when
    /// a model LM Studio was serving turned out to exist nowhere the ledger
    /// looked.
    fn roots(&self) -> Vec<PathBuf>;

    /// The first root, for callers that want one. `roots()` is the truth.
    fn root(&self) -> Option<PathBuf> {
        self.roots().into_iter().next()
    }
    fn scan(&self, root: &Path) -> Vec<Artifact>;
}

pub fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
