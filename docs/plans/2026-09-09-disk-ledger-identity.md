# llm-harmony Slice 2 — Disk Ledger and Model Identity

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llm-harmony ls` and `llm-harmony rm --dry-run` — a disk ledger over all five model stores, a graph of what aliases and derives from what, and a removal plan the code is structurally incapable of executing.

**Architecture:** Five store scanners behind one trait, producing `Artifact` rows deduplicated by `(device, inode)` so bytes are counted once no matter how many symlinks point at them. Artifacts group into canonical model identities. Every derived edge carries its provenance, and inferred edges are forbidden from driving decisions. Rendering and the removal planner sit on top; neither can delete.

**Tech Stack:** Same crate as slice 1. Adds `walkdir` for tree traversal and `serde_yaml` for vLLM-MLX's registry. Reuses `ProviderKind`, `Http`, and the slice-1 adapters to answer "is this served live?"

**Spec:** [`../architecture.md`](../architecture.md) §2 (two ledgers) and §3 (model identity); [`../inventory.md`](../inventory.md) throughout.

## Global Constraints

- **Index in place. Never move, link, or write a model file.** This slice reads the filesystem and writes nothing but its own output.
- **No deletion path may exist.** `std::fs::remove_file`, `remove_dir`, `remove_dir_all`, `rename`, `File::create`, and `Command::new` must not appear anywhere in `src/inventory/`. Verified by grep in Task 10, exactly as slice 1's no-unload property was.
- **Bytes are counted once.** Two paths resolving to the same `(dev, ino)` are one allocation. Verified 2026-09-09: `~/.llamacpp/models/Qwen3-14B-Claude-4.5-Opus-Distill-Q4_K_M.gguf` is a symlink into `~/.lmstudio/models/TeichAI/...`, and every file under an HF `snapshots/<sha>/` directory is a symlink into `../../blobs/`.
- **Every `DerivesFrom` and identity grouping carries `Provenance`.** `Recorded` only when llm-harmony performed the conversion — which, in this slice, is never. Everything is `Inferred`, and the type must make it impossible to consume an inferred edge without acknowledging it.
- **`~/.vllm-mlx/venv` must be excluded.** Verified 2026-09-09: that directory holds 47,588 files. Scanning it is both wrong and slow.
- **A missing store is not an error.** Same contract as a provider that is not running.
- Target is macOS on Apple Silicon; `(dev, ino)` comes from `std::os::unix::fs::MetadataExt`.

## Real store layouts

Probed on this machine 2026-09-09. Every scanner is written against these.

| Store | Root | Layout |
|---|---|---|
| HF cache | `~/.cache/huggingface/hub` | `models--{org}--{name}/{blobs,refs,snapshots}`; `refs/main` holds a sha; `snapshots/<sha>/*` are **all symlinks** into `../../blobs/` |
| LM Studio | `~/.lmstudio/models` | `{publisher}/{repo}/` with real files (`config.json`, `*.gguf`) |
| llama.cpp | `~/.llamacpp/models` | flat pool, **mixed** real `.gguf` files and symlinks into other stores |
| vLLM-MLX | `~/.vllm-mlx` | `models.yaml` registry + `models/{name}/`; **`venv/` must be skipped** |
| Ollama | `~/.ollama/models` | `manifests/registry.ollama.ai/library/{name}/{tag}` → JSON with `layers[].digest`; `blobs/sha256-*` content-addressed |

## File Structure

| File | Responsibility |
|---|---|
| `src/inventory/mod.rs` | module wiring, `Inventory::scan()` |
| `src/inventory/artifact.rs` | `Artifact`, `ArtifactId`, `Store`, `Format`, `Provenance`, `FileKey` |
| `src/inventory/scan/mod.rs` | `trait StoreScanner`, `scan_all()`, exclusion rules |
| `src/inventory/scan/hf.rs` | HF cache scanner |
| `src/inventory/scan/lmstudio.rs` | LM Studio tree scanner |
| `src/inventory/scan/llamacpp.rs` | llama.cpp flat pool scanner |
| `src/inventory/scan/vllm.rs` | `models.yaml` + `models/` scanner |
| `src/inventory/scan/ollama.rs` | manifest → blob scanner |
| `src/inventory/graph.rs` | `Edge`, `Requirer`, `Graph`, alias detection |
| `src/inventory/identity.rs` | canonical model grouping |
| `src/inventory/safety.rs` | redundant / reproducible / irreplaceable |
| `src/inventory/plan.rs` | removal planning (no execution) |
| `src/render_ls.rs` | `ls` and `rm --dry-run` output |
| `tests/inventory/*` | fixture trees + scanner tests |

---

### Task 1: Artifact domain types and byte-counting identity

**Files:**
- Create: `src/inventory/mod.rs`, `src/inventory/artifact.rs`
- Modify: `src/lib.rs`
- Test: unit tests in `src/inventory/artifact.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Store`, `Format`, `Provenance`, `FileKey`, `ArtifactId`, `Artifact`, `Artifact::from_path()`.

- [ ] **Step 1: Write the failing test**

Create `src/inventory/artifact.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_is_read_from_the_file_shape() {
        assert_eq!(Format::from_path_str("model.q4_k_m.gguf"), Format::Gguf);
        assert_eq!(Format::from_path_str("model.safetensors"), Format::SafetensorsBf16);
        assert_eq!(Format::from_path_str("weights.npz"), Format::Mlx);
        assert_eq!(Format::from_path_str("README.md"), Format::Other);
    }

    /// Quantisation appears in filenames three different ways across stores.
    #[test]
    fn bits_are_parsed_from_the_naming_conventions_in_use() {
        assert_eq!(bits_from_name("Qwen3-14B-Q4_K_M.gguf"), Some(4));
        assert_eq!(bits_from_name("Qwen3.5-9B-Heretic-Q6_K.gguf"), Some(6));
        assert_eq!(bits_from_name("Qwen3-14B-Q8_0.gguf"), Some(8));
        assert_eq!(bits_from_name("Qwen3.5-9B-MLX-6bit"), Some(6));
        assert_eq!(bits_from_name("Llama-3.2-1B-Instruct-4bit"), Some(4));
        assert_eq!(bits_from_name("bge-small-en-v1.5"), None);
    }

    /// The rule the whole disk ledger rests on: two paths, one allocation.
    #[test]
    fn provenance_cannot_be_silently_treated_as_fact() {
        let inferred = Provenance::Inferred { basis: "name match".into() };
        assert!(!inferred.is_recorded());
        assert!(Provenance::Recorded.is_recorded());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib artifact`
Expected: FAIL — `cannot find type Format in this scope`.

- [ ] **Step 3: Write the implementation above the test module**

```rust
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Store {
    HfCache,
    LmStudio,
    LlamaCpp,
    Vllm,
    Ollama,
}

impl Store {
    pub const ALL: [Store; 5] = [
        Store::HfCache,
        Store::LmStudio,
        Store::LlamaCpp,
        Store::Vllm,
        Store::Ollama,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Store::HfCache => "hf-cache",
            Store::LmStudio => "lmstudio",
            Store::LlamaCpp => "llamacpp",
            Store::Vllm => "vllm",
            Store::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    Gguf,
    Mlx,
    SafetensorsBf16,
    Other,
}

impl Format {
    pub fn from_path_str(name: &str) -> Format {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".gguf") {
            Format::Gguf
        } else if lower.ends_with(".safetensors") {
            Format::SafetensorsBf16
        } else if lower.ends_with(".npz") {
            Format::Mlx
        } else {
            Format::Other
        }
    }
}

/// Where a claim came from.
///
/// Nothing in slice 2 can produce `Recorded` — llm-harmony has not performed a
/// conversion. The variant exists so slice 5 has somewhere to put the truth,
/// and so consumers must pattern-match rather than assume.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "provenance", rename_all = "kebab-case")]
pub enum Provenance {
    Recorded,
    Inferred { basis: String },
}

impl Provenance {
    pub fn is_recorded(&self) -> bool {
        matches!(self, Provenance::Recorded)
    }
}

/// Filesystem allocation identity. Two paths with the same `FileKey` are the
/// same bytes and must be counted once.
///
/// This matters concretely: every file under an HF `snapshots/<sha>/` is a
/// symlink into `../../blobs/`, and `~/.llamacpp/models` holds a symlink into
/// `~/.lmstudio/models`. Verified 2026-09-09.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct FileKey {
    pub dev: u64,
    pub ino: u64,
}

impl FileKey {
    pub fn of(path: &Path) -> Option<FileKey> {
        use std::os::unix::fs::MetadataExt;
        // metadata() follows symlinks, which is what we want: we are asking
        // "which allocation is this?", not "is this a link?".
        let md = std::fs::metadata(path).ok()?;
        Some(FileKey {
            dev: md.dev(),
            ino: md.ino(),
        })
    }
}

pub type ArtifactId = String;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Artifact {
    pub id: ArtifactId,
    pub path: PathBuf,
    pub store: Store,
    pub format: Format,
    pub bits: Option<u8>,
    /// Real bytes of the allocation. Identical for every path sharing a `key`.
    pub bytes: u64,
    pub key: Option<FileKey>,
    /// True when this path is a symlink. Removing it reclaims nothing.
    pub is_link: bool,
    /// The name used for identity inference.
    pub name_hint: String,
}

/// Quantisation from a filename. GGUF uses `Q4_K_M`; MLX uses `4bit`.
pub fn bits_from_name(name: &str) -> Option<u8> {
    let upper = name.to_ascii_uppercase();
    for b in [2u8, 3, 4, 5, 6, 8] {
        if upper.contains(&format!("Q{b}_")) || upper.contains(&format!("-{b}BIT")) {
            return Some(b);
        }
    }
    // Trailing `Q8_0`-style with no underscore after, and bare `4bit`.
    for b in [2u8, 3, 4, 5, 6, 8] {
        if upper.ends_with(&format!("Q{b}")) || upper.contains(&format!("{b}BIT")) {
            return Some(b);
        }
    }
    None
}

impl Artifact {
    pub fn from_path(path: &Path, store: Store) -> Option<Artifact> {
        let name = path.file_name()?.to_string_lossy().to_string();
        let is_link = std::fs::symlink_metadata(path).ok()?.file_type().is_symlink();
        let key = FileKey::of(path);
        let bytes = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
        Some(Artifact {
            id: format!("{}:{}", store.as_str(), path.display()),
            path: path.to_path_buf(),
            store,
            format: Format::from_path_str(&name),
            bits: bits_from_name(&name),
            bytes,
            key,
            is_link,
            name_hint: name,
        })
    }
}
```

Create `src/inventory/mod.rs` with `pub mod artifact;` and add `pub mod inventory;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib artifact`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/inventory src/lib.rs
git commit -m "feat: artifact domain types with allocation identity"
```

---

### Task 2: Scanner trait, exclusions, and byte-counting

**Files:**
- Create: `src/inventory/scan/mod.rs`
- Create: `tests/inventory_scan.rs`, `tests/support/tree.rs`
- Test: `tests/inventory_scan.rs`

**Interfaces:**
- Consumes: `Artifact`, `Store`, `FileKey`.
- Produces: `trait StoreScanner { fn store(&self) -> Store; fn root(&self) -> PathBuf; fn scan(&self, root: &Path) -> Vec<Artifact>; }`, `is_excluded(&Path) -> bool`, `total_unique_bytes(&[Artifact]) -> u64`.

- [ ] **Step 1: Write the tree-building test helper**

Create `tests/support/tree.rs`:

```rust
use std::path::{Path, PathBuf};

/// A throwaway directory tree for scanner tests.
pub struct Tree {
    pub root: PathBuf,
}

impl Tree {
    pub fn new(name: &str) -> Tree {
        let root = std::env::temp_dir().join(format!("llm-harmony-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create test tree");
        Tree { root }
    }

    pub fn file(&self, rel: &str, bytes: usize) -> PathBuf {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, vec![0u8; bytes]).unwrap();
        p
    }

    pub fn write(&self, rel: &str, content: &str) -> PathBuf {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, content).unwrap();
        p
    }

    pub fn link(&self, rel: &str, target: &Path) -> PathBuf {
        let p = self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(target, &p).unwrap();
        p
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/inventory_scan.rs`:

```rust
mod support;

use llm_harmony::inventory::artifact::{Artifact, Store};
use llm_harmony::inventory::scan::{is_excluded, total_unique_bytes};

use support::tree::Tree;

fn artifacts(paths: &[std::path::PathBuf], store: Store) -> Vec<Artifact> {
    paths.iter().filter_map(|p| Artifact::from_path(p, store)).collect()
}

/// The rule the disk ledger rests on. A symlink and its target are one
/// allocation, so `du`-style summing overstates by the size of every link.
#[test]
fn a_symlink_and_its_target_count_once() {
    let t = Tree::new("dedupe");
    let real = t.file("store-a/model.gguf", 4096);
    let link = t.link("store-b/model.gguf", &real);

    let all = artifacts(&[real, link], Store::LlamaCpp);
    assert_eq!(all.len(), 2, "both paths are real artifacts");
    assert_eq!(
        total_unique_bytes(&all),
        4096,
        "but they are one allocation, not two"
    );
}

#[test]
fn a_symlink_is_marked_so_removing_it_is_known_to_reclaim_nothing() {
    let t = Tree::new("islink");
    let real = t.file("a/model.gguf", 128);
    let link = t.link("b/model.gguf", &real);

    let all = artifacts(&[real, link], Store::LlamaCpp);
    assert!(!all[0].is_link);
    assert!(all[1].is_link);
    assert_eq!(all[0].key, all[1].key, "same allocation");
}

/// Verified 2026-09-09: ~/.vllm-mlx/venv holds 47,588 files.
#[test]
fn python_virtualenvs_and_caches_are_excluded() {
    assert!(is_excluded(std::path::Path::new("/x/.vllm-mlx/venv/lib/python3.12/foo.py")));
    assert!(is_excluded(std::path::Path::new("/x/venv")));
    assert!(is_excluded(std::path::Path::new("/x/__pycache__/y.pyc")));
    assert!(is_excluded(std::path::Path::new("/x/.git/config")));
    assert!(!is_excluded(std::path::Path::new("/x/models/model.gguf")));
}

#[test]
fn a_store_that_does_not_exist_yields_nothing_and_does_not_panic() {
    use llm_harmony::inventory::scan::scan_dir;
    let out = scan_dir(std::path::Path::new("/nonexistent/store"), Store::LmStudio);
    assert!(out.is_empty());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --test inventory_scan`
Expected: FAIL — `unresolved import llm_harmony::inventory::scan`.

- [ ] **Step 4: Add `walkdir` and write the implementation**

Add to `Cargo.toml` `[dependencies]`:

```toml
walkdir = "2"
```

Create `src/inventory/scan/mod.rs`:

```rust
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
            // No key means the path could not be stat'd — a broken symlink.
            // It occupies nothing.
            None => {}
        }
    }
    total
}

pub trait StoreScanner: Send + Sync {
    fn store(&self) -> Store;
    /// Default root for this store, under `$HOME`.
    fn root(&self) -> Option<PathBuf>;
    fn scan(&self, root: &Path) -> Vec<Artifact>;
}

pub fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
```

Add `pub mod scan;` to `src/inventory/mod.rs`. Add `mod tree;` to `tests/support/mod.rs`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test inventory_scan`
Expected: PASS — 4 tests.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/inventory tests/
git commit -m "feat: store scanner trait with allocation-deduplicated byte counting"
```

---

### Task 3: Hugging Face cache scanner

The store where "counted once" matters most: every snapshot entry is a symlink.

**Files:**
- Create: `src/inventory/scan/hf.rs`
- Modify: `tests/inventory_scan.rs`

**Interfaces:** Produces `scan::hf::HfCache` implementing `StoreScanner`, plus `repo_id_from_dir(&str) -> Option<String>`.

- [ ] **Step 1: Write the failing test**

Append to `tests/inventory_scan.rs`:

```rust
use llm_harmony::inventory::scan::hf::{repo_id_from_dir, HfCache};
use llm_harmony::inventory::scan::StoreScanner;

#[test]
fn hf_directory_names_decode_to_repo_ids() {
    assert_eq!(
        repo_id_from_dir("models--BAAI--bge-small-en-v1.5").as_deref(),
        Some("BAAI/bge-small-en-v1.5")
    );
    assert_eq!(
        repo_id_from_dir("models--mlx-community--Llama-3.2-1B-Instruct-4bit").as_deref(),
        Some("mlx-community/Llama-3.2-1B-Instruct-4bit")
    );
    assert_eq!(repo_id_from_dir("CACHEDIR.TAG"), None);
    assert_eq!(repo_id_from_dir("datasets--foo--bar"), None);
}

/// Real HF layout: snapshots/<sha>/x -> ../../blobs/<hash>. Counting the
/// snapshot entries as well as the blobs would double every repo.
#[test]
fn hf_snapshot_symlinks_do_not_double_count_the_blobs() {
    let t = Tree::new("hf");
    let blob = t.file("models--BAAI--bge/blobs/3c9f31665447", 8192);
    t.link("models--BAAI--bge/snapshots/5c38ec/model.safetensors", &blob);
    t.write("models--BAAI--bge/refs/main", "5c38ec");

    let found = HfCache.scan(&t.root);
    assert_eq!(
        total_unique_bytes(&found),
        8192,
        "blob and its snapshot link are one allocation"
    );
    assert!(found.len() >= 2, "both paths are still listed: {found:?}");
}

#[test]
fn hf_artifacts_carry_their_repo_id_as_the_name_hint() {
    let t = Tree::new("hf-hint");
    let blob = t.file("models--mlx-community--Llama-3.2-1B-Instruct-4bit/blobs/aaaa", 64);
    t.link(
        "models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/s1/model.safetensors",
        &blob,
    );

    let found = HfCache.scan(&t.root);
    assert!(
        found.iter().any(|a| a.name_hint.contains("Llama-3.2-1B-Instruct-4bit")),
        "repo id must reach the artifact for identity grouping: {found:?}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test inventory_scan hf`
Expected: FAIL — `unresolved import ...scan::hf`.

- [ ] **Step 3: Write the implementation**

Create `src/inventory/scan/hf.rs`:

```rust
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

/// The `models--org--name` component of a path under the hub root.
fn repo_of(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let first = rel.components().next()?.as_os_str().to_string_lossy();
    repo_id_from_dir(&first)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inventory_scan`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/inventory/scan/hf.rs tests/inventory_scan.rs
git commit -m "feat: HF cache scanner resolving snapshots to blobs"
```

---

### Task 4: LM Studio and llama.cpp scanners

Two stores, one task: they differ only in tree shape, and the llama.cpp pool is where the live cross-store alias lives.

**Files:**
- Create: `src/inventory/scan/lmstudio.rs`, `src/inventory/scan/llamacpp.rs`
- Modify: `tests/inventory_scan.rs`

**Interfaces:** Produces `scan::lmstudio::LmStudioStore`, `scan::llamacpp::LlamaCppPool`.

- [ ] **Step 1: Write the failing test**

Append to `tests/inventory_scan.rs`:

```rust
use llm_harmony::inventory::scan::llamacpp::LlamaCppPool;
use llm_harmony::inventory::scan::lmstudio::LmStudioStore;

#[test]
fn lmstudio_hints_with_publisher_and_repo() {
    let t = Tree::new("lms");
    t.file("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF/model.q4_k_m.gguf", 512);
    t.write("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF/config.json", "{}");

    let found = LmStudioStore.scan(&t.root);
    let gguf = found.iter().find(|a| a.name_hint.contains("Qwen3-14B")).expect("found");
    assert!(gguf.name_hint.contains("TeichAI"), "publisher must survive: {}", gguf.name_hint);
    assert_eq!(gguf.bits, Some(4));
}

/// The live case on this machine, 2026-09-09: the llama.cpp pool holds a
/// symlink into ~/.lmstudio/models. Deleting the target removes the model from
/// two tools at once; deleting the link frees nothing.
#[test]
fn llamacpp_pool_sees_symlinks_into_other_stores_as_zero_reclaim() {
    let t = Tree::new("pool");
    let target = t.file("lmstudio/TeichAI/repo/model.q4_k_m.gguf", 4096);
    t.link("pool/Qwen3-14B-Claude-4.5-Opus-Distill-Q4_K_M.gguf", &target);
    t.file("pool/Qwen3-14B-Claude-4.5-Opus-Distill-Q8_0.gguf", 8192);

    let found = LlamaCppPool.scan(&t.root.join("pool"));
    assert_eq!(found.len(), 2);

    let link = found.iter().find(|a| a.is_link).expect("the symlink");
    let real = found.iter().find(|a| !a.is_link).expect("the real file");
    assert_eq!(link.bits, Some(4));
    assert_eq!(real.bits, Some(8));
    assert_eq!(total_unique_bytes(&found), 4096 + 8192, "distinct allocations");
}

#[test]
fn llamacpp_pool_tolerates_a_broken_symlink() {
    let t = Tree::new("broken");
    t.link("pool/gone.gguf", std::path::Path::new("/nonexistent/model.gguf"));
    let found = LlamaCppPool.scan(&t.root.join("pool"));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].key, None, "unstattable");
    assert_eq!(total_unique_bytes(&found), 0, "a dangling link occupies nothing");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test inventory_scan lmstudio`
Expected: FAIL — unresolved imports.

- [ ] **Step 3: Write both implementations**

`src/inventory/scan/lmstudio.rs`:

```rust
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
    /// components rather than the filename — `model.q4_k_m.gguf` alone
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
```

`src/inventory/scan/llamacpp.rs`:

```rust
use std::path::{Path, PathBuf};

use super::{home, scan_dir, StoreScanner};
use crate::inventory::artifact::{Artifact, Store};

pub struct LlamaCppPool;

impl StoreScanner for LlamaCppPool {
    fn store(&self) -> Store {
        Store::LlamaCpp
    }

    fn root(&self) -> Option<PathBuf> {
        Some(home()?.join(".llamacpp/models"))
    }

    /// A flat pool of mixed real files and symlinks into other stores. The
    /// filename is the whole identity here, which is why `Artifact::from_path`
    /// already does the right thing.
    fn scan(&self, root: &Path) -> Vec<Artifact> {
        scan_dir(root, Store::LlamaCpp)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inventory_scan`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add src/inventory/scan/lmstudio.rs src/inventory/scan/llamacpp.rs tests/inventory_scan.rs
git commit -m "feat: LM Studio and llama.cpp pool scanners"
```

---

### Task 5: vLLM-MLX scanner with registry

**Files:**
- Create: `src/inventory/scan/vllm.rs`
- Modify: `Cargo.toml`, `tests/inventory_scan.rs`

**Interfaces:** Produces `scan::vllm::VllmStore`, `RegistryEntry { name: String, path: String }`, `parse_registry(&str) -> Vec<RegistryEntry>`.

- [ ] **Step 1: Write the failing test**

Append to `tests/inventory_scan.rs`:

```rust
use llm_harmony::inventory::scan::vllm::{parse_registry, VllmStore};

/// Registry shape captured from ~/.vllm-mlx/models.yaml, 2026-09-09.
#[test]
fn vllm_registry_names_are_parsed() {
    let yaml = r#"
manager:
  memory_budget_gb: 14
  contention_policy:
    strategy: wait_then_fail
models:
  - name: Qwen3.5-9B-ultra-uncensored-heretic-MLX
    path: /Users/x/.vllm-mlx/models/Qwen3.5-9B-ultra-uncensored-heretic-MLX
  - name: Qwen3.5-9B-Defiant-Heretic-NEO-IMATRIX-MAX-MTP-MLX-6bit
    path: /Users/x/.vllm-mlx/models/Qwen3.5-9B-Defiant-Heretic-NEO-IMATRIX-MAX-MTP-MLX-6bit
"#;
    let entries = parse_registry(yaml);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].name, "Qwen3.5-9B-ultra-uncensored-heretic-MLX");
    assert!(entries[1].path.ends_with("MLX-6bit"));
}

#[test]
fn vllm_registry_survives_a_file_with_no_models_key() {
    assert!(parse_registry("manager:\n  memory_budget_gb: 14\n").is_empty());
    assert!(parse_registry("not: valid: yaml: [").is_empty());
}

/// Verified 2026-09-09: ~/.vllm-mlx/venv holds 47,588 files. Walking it makes
/// the scan both wrong and slow.
#[test]
fn vllm_scan_skips_the_virtualenv_entirely() {
    let t = Tree::new("vllm");
    t.file("models/Qwen3.5-9B-MLX-4bit/weights.npz", 2048);
    t.file("venv/lib/python3.12/site-packages/torch/_C.so", 999_999);
    t.write("models.yaml", "models:\n  - name: Qwen3.5-9B-MLX-4bit\n    path: /x\n");

    let found = VllmStore.scan(&t.root);
    assert!(
        found.iter().all(|a| !a.path.to_string_lossy().contains("venv")),
        "venv leaked into the scan: {found:?}"
    );
    assert_eq!(total_unique_bytes(&found), 2048, "only the model counts");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test inventory_scan vllm`
Expected: FAIL — unresolved import.

- [ ] **Step 3: Add the YAML dependency and implement**

Add to `Cargo.toml`:

```toml
serde_yaml = "0.9"
```

Create `src/inventory/scan/vllm.rs`:

```rust
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
/// Note `path:` is not tilde-expanded by vllm-mlx itself — a leading `~` is
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
        let mut out = scan_dir(&root.join("models"), Store::Vllm);
        for a in &mut out {
            if let Some(hint) = model_dir(&root.join("models"), &a.path) {
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inventory_scan`
Expected: PASS — 13 tests.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/inventory/scan/vllm.rs tests/inventory_scan.rs
git commit -m "feat: vLLM-MLX scanner with registry parsing and venv exclusion"
```

---

### Task 6: Ollama scanner — manifests to blobs

The only content-addressed store, and the only one where two names can share one file.

**Files:**
- Create: `src/inventory/scan/ollama.rs`
- Modify: `tests/inventory_scan.rs`

**Interfaces:** Produces `scan::ollama::OllamaStore`, `Manifest { tag: String, model_digest: String, bytes: u64 }`, `parse_manifest(&str) -> Option<Manifest>`, `blob_path(root, digest) -> PathBuf`.

- [ ] **Step 1: Write the failing test**

Append to `tests/inventory_scan.rs`:

```rust
use llm_harmony::inventory::scan::ollama::{parse_manifest, OllamaStore};

/// Manifest shape captured live 2026-09-09 from
/// ~/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest
#[test]
fn ollama_manifest_yields_the_model_layer_only() {
    let json = r#"{
      "schemaVersion": 2,
      "layers": [
        {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:970aa74c","size":274290656},
        {"mediaType":"application/vnd.ollama.image.license","digest":"sha256:c71d239d","size":11357},
        {"mediaType":"application/vnd.ollama.image.params","digest":"sha256:ce4a164f","size":17}
      ]
    }"#;
    let m = parse_manifest(json).expect("parsed");
    assert_eq!(m.model_digest, "sha256:970aa74c", "the model layer, not the license");
    assert_eq!(m.bytes, 274_290_656);
}

#[test]
fn ollama_manifest_without_a_model_layer_is_none() {
    assert!(parse_manifest(r#"{"layers":[{"mediaType":"x","digest":"d","size":1}]}"#).is_none());
    assert!(parse_manifest("not json").is_none());
}

/// Content-addressed dedup: two tags pointing at one blob is one allocation,
/// and removing either tag reclaims nothing.
#[test]
fn ollama_two_tags_sharing_a_blob_count_once() {
    let t = Tree::new("ollama");
    t.file("blobs/sha256-970aa74c", 4096);
    let manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:970aa74c","size":4096}]}"#;
    t.write("manifests/registry.ollama.ai/library/nomic-embed-text/latest", manifest);
    t.write("manifests/registry.ollama.ai/library/nomic-embed-text/v1.5", manifest);

    let found = OllamaStore.scan(&t.root);
    assert_eq!(found.len(), 2, "both tags are listed");
    assert_eq!(total_unique_bytes(&found), 4096, "but one allocation");
    assert!(found.iter().any(|a| a.name_hint.contains("nomic-embed-text")));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test inventory_scan ollama`
Expected: FAIL — unresolved import.

- [ ] **Step 3: Write the implementation**

Create `src/inventory/scan/ollama.rs`:

```rust
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

                // `<name>/<tag>` from the manifest path.
                let rel = e.path().strip_prefix(&manifests).ok()?;
                // `registry.ollama.ai/library/nomic-embed-text/latest`
                //   -> `nomic-embed-text:latest`
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
                    format: Format::Gguf, // Ollama serves GGUF
                    bits: None,           // not derivable from the manifest
                    bytes: m.bytes,
                    key: FileKey::of(&blob),
                    is_link: false,
                    name_hint,
                })
            })
            .collect()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inventory_scan`
Expected: PASS — 16 tests.

- [ ] **Step 5: Commit**

```bash
git add src/inventory/scan/ollama.rs tests/inventory_scan.rs
git commit -m "feat: Ollama scanner resolving manifests to content-addressed blobs"
```

---

### Task 7: The graph — aliases and lineage with provenance

**Files:**
- Create: `src/inventory/graph.rs`, `src/inventory/identity.rs`
- Test: `tests/inventory_graph.rs`

**Interfaces:**
- Produces: `Edge::{Aliases, DerivesFrom, RequiredBy}`, `Requirer`, `Graph::build(&[Artifact])`, `Graph::aliases_of()`, `ModelIdentity`, `canonical_name(&str) -> String`, `group(&[Artifact]) -> Vec<ModelIdentity>`.

- [ ] **Step 1: Write the failing test**

Create `tests/inventory_graph.rs`:

```rust
mod support;

use llm_harmony::inventory::artifact::{Artifact, Provenance, Store};
use llm_harmony::inventory::graph::Graph;
use llm_harmony::inventory::identity::{canonical_name, group};

use support::tree::Tree;

fn scan_two_stores(t: &Tree) -> Vec<Artifact> {
    use llm_harmony::inventory::scan::scan_dir;
    let mut v = scan_dir(&t.root.join("lmstudio"), Store::LmStudio);
    v.extend(scan_dir(&t.root.join("pool"), Store::LlamaCpp));
    v
}

/// The live case: llama.cpp's pool symlinks into LM Studio's store.
#[test]
fn artifacts_sharing_an_allocation_become_an_alias_edge() {
    let t = Tree::new("graph-alias");
    let target = t.file("lmstudio/TeichAI/repo/model.q4_k_m.gguf", 4096);
    t.link("pool/Qwen3-14B-Q4_K_M.gguf", &target);

    let g = Graph::build(&scan_two_stores(&t));
    let aliases = g.alias_groups();
    assert_eq!(aliases.len(), 1, "one shared allocation");
    assert_eq!(aliases[0].len(), 2, "reached from two stores");
}

#[test]
fn distinct_files_are_not_aliased() {
    let t = Tree::new("graph-noalias");
    t.file("lmstudio/a/repo/model.q4_k_m.gguf", 4096);
    t.file("pool/other-Q8_0.gguf", 4096); // same size, different allocation

    let g = Graph::build(&scan_two_stores(&t));
    assert!(g.alias_groups().is_empty(), "equal size is not equal identity");
}

/// Names differ across stores for the same logical model. Normalisation must
/// strip quantisation, format, and store-specific decoration.
#[test]
fn canonical_names_collapse_store_specific_decoration() {
    let a = canonical_name("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF");
    let b = canonical_name("Qwen3-14B-Claude-4.5-Opus-Distill-Q4_K_M.gguf");
    let c = canonical_name("Qwen3-14B-Claude-4.5-Opus-Distill-Q8_0.gguf");
    assert_eq!(b, c, "quantisation is not identity: {b} vs {c}");
    assert!(a.contains("qwen3-14b"), "publisher stripped, model kept: {a}");
}

#[test]
fn mlx_and_gguf_builds_of_one_model_group_together() {
    let names = [
        "Qwen3.5-9B-ultra-uncensored-heretic-Q4_K_M.gguf",
        "Qwen3.5-9B-ultra-uncensored-heretic-Q6_K.gguf",
        "Qwen3.5-9B-ultra-uncensored-heretic-MLX",
        "Qwen3.5-9B-ultra-uncensored-heretic-MLX-6bit",
    ];
    let canon: Vec<String> = names.iter().map(|n| canonical_name(n)).collect();
    assert!(
        canon.windows(2).all(|w| w[0] == w[1]),
        "four artifacts of one model must share a canonical name: {canon:?}"
    );
}

/// Grouping is inference, and inference must say so.
#[test]
fn every_grouping_is_marked_inferred_in_this_slice() {
    let t = Tree::new("group-prov");
    t.file("lmstudio/a/Qwen3.5-9B-heretic-GGUF/m.q4_k_m.gguf", 16);
    t.file("pool/Qwen3.5-9B-heretic-Q6_K.gguf", 16);

    let identities = group(&scan_two_stores(&t));
    assert!(!identities.is_empty());
    for id in &identities {
        assert!(
            !id.provenance.is_recorded(),
            "slice 2 performs no conversions, so nothing can be Recorded"
        );
        assert!(matches!(id.provenance, Provenance::Inferred { .. }));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test inventory_graph`
Expected: FAIL — unresolved imports.

- [ ] **Step 3: Write `src/inventory/identity.rs`**

```rust
use crate::inventory::artifact::{Artifact, ArtifactId, Provenance};

/// A logical model, and the artifacts that are builds of it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelIdentity {
    pub canonical: String,
    pub artifacts: Vec<ArtifactId>,
    /// Always `Inferred` in slice 2 — llm-harmony has performed no conversions,
    /// so no lineage is recorded. Placement (slice 4) must refuse to act on
    /// an inferred identity without an explicit override.
    pub provenance: Provenance,
}

/// Strip everything that varies between builds of one model: publisher,
/// extension, format tag, and quantisation.
///
/// This is inference. Names lie — docs/field-notes.md records two HF repos
/// whose names claimed to be builds of models they were not — which is why the
/// result is always wrapped in `Provenance::Inferred`.
pub fn canonical_name(raw: &str) -> String {
    let mut s = raw.to_ascii_lowercase();

    // Publisher prefix: "teichai/qwen3-14b-..." -> "qwen3-14b-..."
    if let Some((_, rest)) = s.split_once('/') {
        s = rest.to_string();
    }
    // Extension
    for ext in [".gguf", ".safetensors", ".npz", ".json"] {
        s = s.trim_end_matches(ext).to_string();
    }
    // Quantisation and format decoration, longest first so `-mlx-6bit` goes
    // before `-mlx`.
    const DECORATION: [&str; 18] = [
        "-q2_k", "-q3_k_m", "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q6_k", "-q8_0",
        "-mlx-4bit", "-mlx-6bit", "-mlx-8bit", "-4bit", "-6bit", "-8bit",
        "-gguf", "-mlx", "-imatrix", ".q4_k_m", ".q8_0",
    ];
    let mut changed = true;
    while changed {
        changed = false;
        for d in DECORATION {
            if let Some(stripped) = s.strip_suffix(d) {
                s = stripped.to_string();
                changed = true;
            }
            // Also handle mid-string decoration, e.g. "...-distill-q4_k_m"
            if s.contains(d) && !d.starts_with('.') {
                s = s.replace(d, "");
                changed = true;
            }
        }
    }
    // Ollama tags: "nomic-embed-text:latest" -> "nomic-embed-text"
    if let Some((base, _tag)) = s.split_once(':') {
        s = base.to_string();
    }
    s.trim_matches(['-', '.', '_']).to_string()
}

pub fn group(artifacts: &[Artifact]) -> Vec<ModelIdentity> {
    use std::collections::BTreeMap;
    let mut by_name: BTreeMap<String, Vec<ArtifactId>> = BTreeMap::new();
    for a in artifacts {
        let canon = canonical_name(&a.name_hint);
        if canon.is_empty() {
            continue;
        }
        by_name.entry(canon).or_default().push(a.id.clone());
    }
    by_name
        .into_iter()
        .map(|(canonical, artifacts)| ModelIdentity {
            canonical,
            artifacts,
            provenance: Provenance::Inferred {
                basis: "normalised filename".to_string(),
            },
        })
        .collect()
}
```

- [ ] **Step 4: Write `src/inventory/graph.rs`**

```rust
use std::collections::HashMap;

use crate::inventory::artifact::{Artifact, ArtifactId, FileKey, Provenance};
use crate::provider::ProviderKind;

/// Why an artifact may not be removed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Requirer {
    /// A provider currently has it resident. From the slice-1 memory ledger.
    ServedLive { provider: ProviderKind, model_id: String },
    /// Named in a registry, where a stale path is a startup failure.
    Registry { store: String, name: String },
    /// Inside a directory a provider scans at runtime — llama.cpp's router
    /// scans the HF cache, so cache entries are serving dependencies.
    Scanned { provider: ProviderKind },
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "edge", rename_all = "kebab-case")]
pub enum Edge {
    /// Same bytes reached by two paths. Removing one reclaims nothing.
    Aliases { a: ArtifactId, b: ArtifactId },
    DerivesFrom {
        child: ArtifactId,
        parent: ArtifactId,
        provenance: Provenance,
    },
    RequiredBy {
        artifact: ArtifactId,
        requirer: Requirer,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Graph {
    pub edges: Vec<Edge>,
    /// Skipped: serde_json cannot use a struct as a map key, and this is an
    /// index rather than part of the graph's public shape.
    #[serde(skip)]
    by_key: HashMap<FileKey, Vec<ArtifactId>>,
}

impl Graph {
    pub fn build(artifacts: &[Artifact]) -> Graph {
        let mut by_key: HashMap<FileKey, Vec<ArtifactId>> = HashMap::new();
        for a in artifacts {
            if let Some(k) = a.key {
                by_key.entry(k).or_default().push(a.id.clone());
            }
        }

        let mut edges = Vec::new();
        for ids in by_key.values() {
            for pair in ids.windows(2) {
                edges.push(Edge::Aliases {
                    a: pair[0].clone(),
                    b: pair[1].clone(),
                });
            }
        }

        Graph { edges, by_key }
    }

    /// Groups of artifact ids that are the same allocation.
    pub fn alias_groups(&self) -> Vec<Vec<ArtifactId>> {
        self.by_key
            .values()
            .filter(|ids| ids.len() > 1)
            .cloned()
            .collect()
    }

    pub fn aliases_of(&self, id: &str) -> Vec<ArtifactId> {
        self.alias_groups()
            .into_iter()
            .find(|g| g.iter().any(|x| x == id))
            .map(|g| g.into_iter().filter(|x| x != id).collect())
            .unwrap_or_default()
    }

    pub fn add_required_by(&mut self, artifact: ArtifactId, requirer: Requirer) {
        self.edges.push(Edge::RequiredBy { artifact, requirer });
    }

    pub fn requirers_of(&self, id: &str) -> Vec<&Requirer> {
        self.edges
            .iter()
            .filter_map(|e| match e {
                Edge::RequiredBy { artifact, requirer } if artifact == id => Some(requirer),
                _ => None,
            })
            .collect()
    }
}
```

Add `pub mod graph;` and `pub mod identity;` to `src/inventory/mod.rs`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test inventory_graph`
Expected: PASS — 5 tests.

- [ ] **Step 6: Commit**

```bash
git add src/inventory/graph.rs src/inventory/identity.rs src/inventory/mod.rs tests/inventory_graph.rs
git commit -m "feat: alias graph and inferred model identity grouping"
```

---

### Task 8: `REQUIRED_BY` — cross-referencing the live providers

The edge that makes removal unsafe, and the first place slice 2 uses slice 1.

**Files:**
- Modify: `src/inventory/mod.rs` (add `Inventory::scan`)
- Test: `tests/inventory_required.rs`

**Interfaces:** Produces `Inventory { artifacts, graph, identities }`, `Inventory::scan(&Config, &Http, Machine) -> Inventory`, `Inventory::scan_offline(Option<Vec<(Store, PathBuf)>>) -> Inventory`.

- [ ] **Step 1: Write the failing test**

Create `tests/inventory_required.rs`:

```rust
mod support;

use llm_harmony::inventory::graph::{Graph, Requirer};
use llm_harmony::inventory::artifact::{Artifact, Store};
use llm_harmony::provider::ProviderKind;

use support::tree::Tree;

fn artifact(t: &Tree, rel: &str, bytes: usize, store: Store) -> Artifact {
    let p = t.file(rel, bytes);
    Artifact::from_path(&p, store).unwrap()
}

/// docs/field-notes.md: "llama.cpp's router scans the HF cache, so cache
/// entries are runtime dependencies, not merely downloads." That makes every
/// HF blob a serving dependency whenever the router is up.
#[test]
fn hf_cache_entries_are_marked_as_scanned_by_llamacpp() {
    let t = Tree::new("req-scan");
    let a = artifact(&t, "models--x--y/blobs/aaa", 64, Store::HfCache);
    let mut g = Graph::build(&[a.clone()]);
    g.add_required_by(a.id.clone(), Requirer::Scanned { provider: ProviderKind::LlamaCpp });

    let reqs = g.requirers_of(&a.id);
    assert_eq!(reqs.len(), 1);
    assert!(matches!(reqs[0], Requirer::Scanned { provider: ProviderKind::LlamaCpp }));
}

#[test]
fn a_live_resident_model_becomes_a_served_live_requirer() {
    let t = Tree::new("req-live");
    let a = artifact(&t, "TeichAI/repo/model.q4_k_m.gguf", 64, Store::LmStudio);
    let mut g = Graph::build(&[a.clone()]);
    g.add_required_by(
        a.id.clone(),
        Requirer::ServedLive {
            provider: ProviderKind::LmStudio,
            model_id: "qwen3-14b-claude-4.5-opus-high-reasoning-distill".into(),
        },
    );
    assert_eq!(g.requirers_of(&a.id).len(), 1);
}

/// docs/inventory.md §2 question 3: a name in models.yaml is a dependency,
/// because a stale path there is a startup failure rather than a warning.
#[test]
fn a_registry_named_artifact_becomes_a_registry_requirer() {
    let t = Tree::new("req-registry");
    let a = artifact(&t, "Qwen3.5-9B-MLX-4bit/weights.npz", 64, Store::Vllm);
    let mut g = Graph::build(&[a.clone()]);
    g.add_required_by(
        a.id.clone(),
        Requirer::Registry {
            store: "vllm".into(),
            name: "Qwen3.5-9B-MLX-4bit".into(),
        },
    );
    let reqs = g.requirers_of(&a.id);
    assert_eq!(reqs.len(), 1);
    assert!(matches!(reqs[0], Requirer::Registry { .. }));
}

#[test]
fn an_artifact_with_no_requirers_reports_an_empty_list_not_an_error() {
    let t = Tree::new("req-none");
    let a = artifact(&t, "pool/orphan-Q4_K_M.gguf", 64, Store::LlamaCpp);
    let g = Graph::build(&[a.clone()]);
    assert!(g.requirers_of(&a.id).is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test inventory_required`
Expected: FAIL if `Requirer` is not exported; otherwise these pass against Task 7's graph and the task is confirming the API shape before `Inventory` wires it.

- [ ] **Step 3: Write `Inventory` in `src/inventory/mod.rs`**

```rust
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
use crate::inventory::identity::{group, ModelIdentity};
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
    /// Filesystem only. No provider is contacted, so nothing is `ServedLive`.
    pub fn scan_offline(roots: Option<Vec<(Store, PathBuf)>>) -> Inventory {
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
        // it is a potential serving dependency even with nothing loaded.
        for a in artifacts.iter().filter(|a| a.store == Store::HfCache) {
            graph.add_required_by(
                a.id.clone(),
                Requirer::Scanned {
                    provider: crate::provider::ProviderKind::LlamaCpp,
                },
            );
        }

        // vLLM-MLX names models in models.yaml, where a stale path is a
        // startup failure -- docs/inventory.md §2, question 3. An artifact
        // under a registered path may not be removed while the name stands.
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

        let identities = group(&artifacts);
        Inventory {
            artifacts,
            graph,
            identities,
        }
    }

    /// Filesystem plus the live memory ledger, so resident models become
    /// `ServedLive` requirers and cannot be planned for removal.
    pub fn scan(config: &Config, http: &Http, machine: Machine) -> Inventory {
        let mut inv = Inventory::scan_offline(None);
        let ledger = Ledger::assemble(config, http, machine);

        for row in &ledger.rows {
            let Outcome::Ok(models) = &row.outcome else { continue };
            for m in models.iter().filter(|m| m.state == crate::provider::State::Loaded) {
                // Match a resident model to artifacts by canonical name. This
                // is inference, and it is why `rm` refuses rather than warns.
                let want = identity::canonical_name(&m.id);
                for a in &inv.artifacts {
                    if identity::canonical_name(&a.name_hint) == want {
                        inv.graph.add_required_by(
                            a.id.clone(),
                            Requirer::ServedLive {
                                provider: row.kind,
                                model_id: m.id.clone(),
                            },
                        );
                    }
                }
            }
        }
        inv
    }

    pub fn total_bytes(&self) -> u64 {
        scan::total_unique_bytes(&self.artifacts)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inventory_required`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/inventory/mod.rs tests/inventory_required.rs
git commit -m "feat: required-by edges from live providers and scanned caches"
```

---

### Task 9: Safety classification

**Files:**
- Create: `src/inventory/safety.rs`
- Test: unit tests in `src/inventory/safety.rs`

**Interfaces:** Produces `Safety::{Redundant, Reproducible, Irreplaceable}`, `classify(&Artifact, &[Artifact]) -> Safety`.

- [ ] **Step 1: Write the failing test**

Create `src/inventory/safety.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::{Artifact, Format, Store};

    fn art(id: &str, name: &str, bits: Option<u8>, format: Format, store: Store) -> Artifact {
        Artifact {
            id: id.into(),
            path: format!("/x/{name}").into(),
            store,
            format,
            bits,
            bytes: 1024,
            key: None,
            is_link: false,
            name_hint: name.into(),
        }
    }

    /// A Q4 alongside a Q6 of the same model is redundant. Quantisation is
    /// one-way, so the higher-bit artifact is the one worth keeping.
    #[test]
    fn a_lower_bit_build_beside_a_higher_one_is_redundant() {
        let q4 = art("a", "Qwen3.5-9B-heretic-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        let q6 = art("b", "Qwen3.5-9B-heretic-Q6_K.gguf", Some(6), Format::Gguf, Store::LlamaCpp);
        let peers = vec![q4.clone(), q6.clone()];

        assert!(matches!(classify(&q4, &peers), Safety::Redundant { .. }));
    }

    /// The higher-bit build is not redundant just because a lower one exists.
    #[test]
    fn the_highest_bit_build_is_never_redundant() {
        let q4 = art("a", "m-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        let q6 = art("b", "m-Q6_K.gguf", Some(6), Format::Gguf, Store::LlamaCpp);
        assert!(!matches!(classify(&q6, &vec![q4, q6.clone()]), Safety::Redundant { .. }));
    }

    /// There is no MLX <-> GGUF path. An MLX build does not make a GGUF build
    /// redundant, whatever the bit depths.
    #[test]
    fn a_different_format_never_makes_an_artifact_redundant() {
        let gguf = art("a", "m-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        let mlx = art("b", "m-MLX-6bit", Some(6), Format::Mlx, Store::Vllm);
        assert!(!matches!(classify(&gguf, &vec![gguf.clone(), mlx]), Safety::Redundant { .. }));
    }

    /// A source in the HF cache can be re-downloaded; that is priced, not free.
    #[test]
    fn an_hf_source_is_reproducible() {
        let src = art("a", "org/model", None, Format::SafetensorsBf16, Store::HfCache);
        assert!(matches!(classify(&src, &vec![src.clone()]), Safety::Reproducible { .. }));
    }

    /// A lone converted artifact with no source present cannot be rebuilt from
    /// what remains.
    #[test]
    fn a_lone_build_with_no_source_is_irreplaceable() {
        let only = art("a", "m-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        assert!(matches!(classify(&only, &vec![only.clone()]), Safety::Irreplaceable { .. }));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib safety`
Expected: FAIL — `cannot find function classify`.

- [ ] **Step 3: Write the implementation above the test module**

```rust
use crate::inventory::artifact::{Artifact, ArtifactId, Format, Store};

/// Three kinds of "safe to delete", which are not the same risk.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "safety", rename_all = "kebab-case")]
pub enum Safety {
    /// A better artifact of the same model, same format, exists locally.
    Redundant { better: ArtifactId },
    /// Deleting costs a re-download or a re-convert, and the recipe is known.
    Reproducible { how: String },
    /// The only local copy of something that cannot be rebuilt from what
    /// remains. Never auto-selected.
    Irreplaceable { why: String },
}

/// Classify `a` against the artifacts sharing its canonical identity.
///
/// Two rules from docs/inventory.md are encoded here and must not be relaxed:
/// quantisation is one-way (Q4 cannot become Q8), and there is no MLX <-> GGUF
/// path — both descend from the same source and neither derives from the other.
pub fn classify(a: &Artifact, peers: &[Artifact]) -> Safety {
    // A strictly better build of the SAME format makes this redundant.
    if let (Some(bits), Format::Gguf | Format::Mlx) = (a.bits, a.format) {
        if let Some(better) = peers
            .iter()
            .filter(|p| p.id != a.id && p.format == a.format)
            .filter(|p| p.bits.map(|b| b > bits).unwrap_or(false))
            .max_by_key(|p| p.bits)
        {
            return Safety::Redundant {
                better: better.id.clone(),
            };
        }
    }

    // A source can be re-downloaded. Priced, not free.
    if a.store == Store::HfCache || a.format == Format::SafetensorsBf16 {
        return Safety::Reproducible {
            how: "re-download from Hugging Face".to_string(),
        };
    }

    // A converted artifact whose source is still present can be rebuilt.
    let source_present = peers
        .iter()
        .any(|p| p.format == Format::SafetensorsBf16 || p.store == Store::HfCache);
    if source_present {
        return Safety::Reproducible {
            how: "re-convert from the local source".to_string(),
        };
    }

    Safety::Irreplaceable {
        why: "no local source to rebuild from, and no better build of this format".to_string(),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib safety`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/inventory/safety.rs
git commit -m "feat: redundant reproducible irreplaceable classification"
```

---

### Task 10: Removal planning that cannot remove

**Files:**
- Create: `src/inventory/plan.rs`
- Test: unit tests in `src/inventory/plan.rs`

**Interfaces:** Produces `RemovalPlan { steps, reclaims_bytes, refusals }`, `RemovalStep`, `Refusal`, `build_plan(Vec<Artifact>, &Graph) -> RemovalPlan`.

- [ ] **Step 1: Write the failing test**

Create `src/inventory/plan.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::{Artifact, FileKey, Format, Store};
    use crate::inventory::graph::{Graph, Requirer};
    use crate::provider::ProviderKind;

    /// `ino` is explicit: a link and its target share one, which is what makes
    /// "an alias frees nothing" testable without touching the filesystem.
    fn art(id: &str, name: &str, bytes: u64, is_link: bool, ino: u64) -> Artifact {
        Artifact {
            id: id.into(),
            path: format!("/x/{name}").into(),
            store: Store::LlamaCpp,
            format: Format::Gguf,
            bits: Some(4),
            bytes,
            key: Some(FileKey { dev: 1, ino }),
            is_link,
            name_hint: name.into(),
        }
    }

    /// docs/inventory.md: "remove aliases before targets, or the store is
    /// briefly full of dangling links."
    #[test]
    fn symlinks_are_ordered_before_the_files_they_point_at() {
        let target = art("t", "model.gguf", 4096, false, 42);
        let link = art("l", "alias.gguf", 4096, true, 42);
        let plan = build_plan(vec![link, target], &Graph::build(&[]));

        let first = &plan.steps[0];
        assert!(first.is_link, "the alias must be removed first: {:?}", plan.steps);
    }

    /// Removing a symlink frees nothing; only the allocation counts.
    #[test]
    fn reclaimed_bytes_do_not_count_aliases() {
        let target = art("t", "model.gguf", 4096, false, 42);
        let link = art("l", "alias.gguf", 4096, true, 42);
        let plan = build_plan(vec![link, target], &Graph::build(&[]));
        assert_eq!(plan.reclaims_bytes, 4096, "one allocation, not two");
    }

    /// Refuses rather than warns, per docs/inventory.md §4.
    #[test]
    fn a_live_served_artifact_produces_a_refusal_and_no_steps() {
        let a = art("t", "model.gguf", 4096, false, 7);
        let mut g = Graph::build(&[a.clone()]);
        g.add_required_by(
            a.id.clone(),
            Requirer::ServedLive {
                provider: ProviderKind::LmStudio,
                model_id: "qwen3-14b".into(),
            },
        );
        let plan = build_plan(vec![a], &g);

        assert!(plan.steps.is_empty(), "nothing may be planned while it is served");
        assert_eq!(plan.refusals.len(), 1);
        assert!(plan.refusals[0].reason.contains("served"), "{:?}", plan.refusals);
    }

    #[test]
    fn an_artifact_scanned_by_a_router_is_also_refused() {
        let a = art("t", "blob", 64, false, 8);
        let mut g = Graph::build(&[a.clone()]);
        g.add_required_by(a.id.clone(), Requirer::Scanned { provider: ProviderKind::LlamaCpp });
        let plan = build_plan(vec![a], &g);
        assert!(plan.steps.is_empty());
        assert_eq!(plan.refusals.len(), 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib plan`
Expected: FAIL — `cannot find function build_plan`.

- [ ] **Step 3: Write the implementation above the test module**

```rust
use std::collections::HashSet;

use crate::inventory::artifact::{Artifact, ArtifactId, FileKey};
use crate::inventory::graph::{Graph, Requirer};

#[derive(Debug, Clone, serde::Serialize)]
pub struct RemovalStep {
    pub id: ArtifactId,
    pub path: String,
    pub is_link: bool,
    /// Bytes this step actually frees. Zero for an alias.
    pub frees_bytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Refusal {
    pub id: ArtifactId,
    pub reason: String,
}

/// A plan, never an execution.
///
/// There is deliberately no `apply`, and no `std::fs` removal call anywhere in
/// this module. Slice 2 cannot delete, for the same structural reason slice 1
/// cannot unload.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RemovalPlan {
    pub steps: Vec<RemovalStep>,
    pub reclaims_bytes: u64,
    pub refusals: Vec<Refusal>,
}

fn refusal_for(id: &str, graph: &Graph) -> Option<Refusal> {
    let reqs = graph.requirers_of(id);
    if reqs.is_empty() {
        return None;
    }
    let reason = reqs
        .iter()
        .map(|r| match r {
            Requirer::ServedLive { provider, model_id } => {
                format!("served live by {provider} as {model_id}")
            }
            Requirer::Registry { store, name } => {
                format!("named in the {store} registry as {name}")
            }
            Requirer::Scanned { provider } => {
                format!("inside a directory {provider} scans at runtime")
            }
        })
        .collect::<Vec<_>>()
        .join("; ");
    Some(Refusal {
        id: id.to_string(),
        reason,
    })
}

pub fn build_plan(artifacts: Vec<Artifact>, graph: &Graph) -> RemovalPlan {
    let mut refusals = Vec::new();
    let mut keep = Vec::new();

    for a in artifacts {
        match refusal_for(&a.id, graph) {
            Some(r) => refusals.push(r),
            None => keep.push(a),
        }
    }

    // Any refusal poisons the whole plan: removing half of an aliased pair
    // leaves a dangling link, and removing a source while a build is served
    // is exactly the case this exists to prevent.
    if !refusals.is_empty() {
        return RemovalPlan {
            steps: Vec::new(),
            reclaims_bytes: 0,
            refusals,
        };
    }

    // Aliases before targets, or the store is briefly full of dangling links.
    keep.sort_by_key(|a| !a.is_link);

    let mut counted: HashSet<FileKey> = HashSet::new();
    let mut reclaims = 0u64;
    let steps = keep
        .into_iter()
        .map(|a| {
            let frees = match (a.is_link, a.key) {
                (true, _) => 0,
                (false, Some(k)) => {
                    if counted.insert(k) {
                        a.bytes
                    } else {
                        0
                    }
                }
                (false, None) => 0,
            };
            reclaims += frees;
            RemovalStep {
                id: a.id,
                path: a.path.display().to_string(),
                is_link: a.is_link,
                frees_bytes: frees,
            }
        })
        .collect();

    RemovalPlan {
        steps,
        reclaims_bytes: reclaims,
        refusals: Vec::new(),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib plan`
Expected: PASS — 4 tests.

- [ ] **Step 5: Verify the no-deletion property**

Run:

```bash
grep -rnE "remove_file|remove_dir|remove_dir_all|std::fs::rename|File::create|Command::new" src/inventory/
```

Expected: **no matches.** The only `std::fs` calls in `src/inventory/` are `metadata`, `symlink_metadata`, `read_to_string`, and `exists`.

- [ ] **Step 6: Commit**

```bash
git add src/inventory/plan.rs
git commit -m "feat: removal planning with alias ordering and hard refusals"
```

---

### Task 11: `ls` and `rm --dry-run`, and live verification

**Files:**
- Create: `src/render_ls.rs`
- Modify: `src/main.rs`, `src/lib.rs`

**Interfaces:** Produces `render_ls(&Inventory) -> String`, `render_plan(&RemovalPlan) -> String`, and the `ls` / `rm` subcommands.

- [ ] **Step 1: Write `src/render_ls.rs`**

```rust
use crate::inventory::plan::RemovalPlan;
use crate::inventory::safety::{classify, Safety};
use crate::inventory::Inventory;
use crate::render::human_bytes;

pub fn render_ls(inv: &Inventory) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<44} {:>8} {:>9}  {}\n",
        "model", "builds", "on disk", "safety"
    ));

    for id in &inv.identities {
        let arts: Vec<_> = inv
            .artifacts
            .iter()
            .filter(|a| id.artifacts.contains(&a.id))
            .cloned()
            .collect();
        if arts.is_empty() {
            continue;
        }
        let bytes = crate::inventory::scan::total_unique_bytes(&arts);
        let worst = arts
            .iter()
            .map(|a| classify(a, &arts))
            .max_by_key(|s| match s {
                Safety::Redundant { .. } => 0,
                Safety::Reproducible { .. } => 1,
                Safety::Irreplaceable { .. } => 2,
            });
        let label = match worst {
            Some(Safety::Redundant { .. }) => "redundant",
            Some(Safety::Reproducible { .. }) => "reproducible",
            Some(Safety::Irreplaceable { .. }) => "IRREPLACEABLE",
            None => "-",
        };
        let name: String = id.canonical.chars().take(44).collect();
        out.push_str(&format!(
            "{:<44} {:>8} {:>9}  {}\n",
            name,
            arts.len(),
            human_bytes(bytes),
            label
        ));
    }

    out.push_str(&"─".repeat(78));
    out.push('\n');
    out.push_str(&format!(
        "{:<44} {:>8} {:>9}\n",
        "total",
        inv.artifacts.len(),
        human_bytes(inv.total_bytes())
    ));
    out.push_str("\nlineage is inferred from filenames; no conversion was recorded by llm-harmony\n");
    out
}

pub fn render_plan(plan: &RemovalPlan) -> String {
    let mut out = String::new();
    if !plan.refusals.is_empty() {
        out.push_str("refusing:\n");
        for r in &plan.refusals {
            out.push_str(&format!("  {} — {}\n", r.id, r.reason));
        }
        out.push_str("\nnothing planned.\n");
        return out;
    }
    out.push_str("would remove, in order:\n");
    for s in &plan.steps {
        let note = if s.is_link { "alias — frees nothing" } else { "" };
        out.push_str(&format!(
            "  {:>9}  {} {}\n",
            human_bytes(s.frees_bytes),
            s.path,
            note
        ));
    }
    out.push_str(&format!("\nreclaims {}\n", human_bytes(plan.reclaims_bytes)));
    out.push_str("dry run — llm-harmony cannot delete\n");
    out
}
```

- [ ] **Step 2: Wire the subcommands in `src/main.rs`**

Add to the `Command` enum:

```rust
    /// What is on disk, grouped by model.
    Ls {
        #[arg(long)]
        json: bool,
    },
    /// Print a removal plan. Never executes.
    Rm {
        model: String,
        #[arg(long, default_value_t = true)]
        dry_run: bool,
    },
```

And the match arms:

```rust
        Command::Ls { json } => {
            let inv = llm_harmony::inventory::Inventory::scan_offline(None);
            if json {
                println!("{}", serde_json::to_string_pretty(&inv.identities).unwrap());
            } else {
                print!("{}", llm_harmony::render_ls::render_ls(&inv));
            }
            ExitCode::SUCCESS
        }
        Command::Rm { model, dry_run } => {
            if !dry_run {
                eprintln!("llm-harmony: real removal is not implemented in this slice");
                return ExitCode::FAILURE;
            }
            let inv = llm_harmony::inventory::Inventory::scan_offline(None);
            let want = llm_harmony::inventory::identity::canonical_name(&model);
            let arts: Vec<_> = inv
                .artifacts
                .iter()
                .filter(|a| llm_harmony::inventory::identity::canonical_name(&a.name_hint) == want)
                .cloned()
                .collect();
            if arts.is_empty() {
                eprintln!("llm-harmony: no artifacts match `{model}`");
                return ExitCode::FAILURE;
            }
            let plan = llm_harmony::inventory::plan::build_plan(arts, &inv.graph);
            print!("{}", llm_harmony::render_ls::render_plan(&plan));
            ExitCode::SUCCESS
        }
```

- [ ] **Step 3: Run the whole suite**

Run: `cargo test`
Expected: PASS — slice 1's 44 tests plus roughly 30 new ones.

- [ ] **Step 4: Verify against the real machine**

Run: `cargo run --quiet -- ls`

Expected, from the stores probed 2026-09-09:

- `Qwen3.5-9B-Defiant-Heretic-NEO-IMATRIX-MAX-MTP` groups **5 artifacts** — an HF source, GGUF Q4_K_M (5.8 G) and Q6_K (7.6 G), and MLX 4-bit and 6-bit builds.
- `Qwen3.5-9B-ultra-uncensored-heretic` groups the same shape.
- `Qwen3-14B-Claude-4.5-Opus-Distill` groups the LM Studio GGUF, its llama.cpp alias, and the 16 G Q8_0.
- The total is **well under** `du -sh` across the four stores, because the HF snapshot symlinks and the cross-store alias are each counted once.

Cross-check:

```bash
du -shc ~/.cache/huggingface/hub ~/.lmstudio/models ~/.llamacpp/models ~/.vllm-mlx/models 2>/dev/null | tail -1
cargo run --quiet -- ls | tail -3
```

The `ls` total should be *smaller*, and the difference should be explainable by the alias. If it is larger, dedup is broken.

- [ ] **Step 5: Verify the refusal path with a live model**

With a model loaded in LM Studio:

```bash
cargo run --quiet -- rm Qwen3-14B-Claude-4.5-Opus-Distill
```

Expected: a refusal naming the live provider, and no steps. Note `scan_offline` does not contact providers, so this refusal comes from the HF-scanned edge; wiring `Inventory::scan` into the CLI so `ServedLive` refusals apply is the follow-up below.

- [ ] **Step 6: Verify the no-deletion property one more time**

Run: `grep -rnE "remove_file|remove_dir|rename|Command::new" src/`

Expected: no matches outside comments.

- [ ] **Step 7: Commit**

```bash
git add src/render_ls.rs src/main.rs src/lib.rs
git commit -m "feat: llm-harmony ls and rm --dry-run"
```

---

## What this slice cannot answer

`inventory.md` §2 lists five questions that make removal unsafe. This slice
answers four:

| Question | Covered by |
|---|---|
| 1. Is it served live? | `Requirer::ServedLive`, from the slice-1 ledger |
| 2. Is it a symlink target? | `FileKey` dedup and `Edge::Aliases` |
| 3. Is it named in a registry? | `Requirer::Registry`, from `models.yaml` |
| 4. Is it a conversion source, or the only copy? | `Safety::{Reproducible, Irreplaceable}` |
| 5. **Is it used by something that is not an LLM provider?** | **Not answered** |

Question 5 is the one that caught a real case: a 128 MB embedding model that
turned out to be `loci`'s search index backend, invisible to every provider and
easy to delete by accident. Nothing in this slice can see a consumer that is not
one of the four providers, and inventing a plugin interface for arbitrary tools
is out of scope here.

**Mitigation, not a solution:** `Safety::Irreplaceable` is never auto-selected,
and small non-GGUF artifacts in the HF cache — the shape that case took — land
in `Reproducible` rather than `Redundant`. That makes accidental removal
unlikely rather than impossible, and the gap should be stated in `ls` output
rather than hidden.

## Follow-ups, deliberately not in this slice

- **Wire `Inventory::scan` (live) into the CLI.** Task 11 uses `scan_offline`, so `ServedLive` refusals are not yet applied at the command line. Doing it means every `ls` polls four providers; whether that is the right default is a real question.
- **Re-capture the llama.cpp fixture** from slice 1, still marked UNVERIFIED.
- **Record the answer to §10 Q2** — does the disk ledger genuinely want the memory ledger's shape? This slice is the evidence; write the conclusion into `architecture.md` §2 either way.
- **Bits for Ollama artifacts** are `None`, because the manifest does not carry quantisation. It is in the blob's GGUF header, which means parsing GGUF metadata — a slice of its own, and the same parser intake will need.
