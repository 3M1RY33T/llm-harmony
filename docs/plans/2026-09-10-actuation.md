# llm-harmony Slice 5 — Actuation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `llm-harmony load`, `unload` and `switch` — harmony stops describing the memory ledger and starts changing it, without ever being able to wedge the machine.

**Architecture:** Three additions, each independently useful. **Pins** are runtime state that revoke harmony's permission to evict a named model, settable while it is already resident. **A computed estimate basis** replaces the weights-only floor that admission currently trusts. **A supervised load** watches real memory during the load and unloads if it crosses the floor, so the guarantee does not rest on a prediction being right. Serialised by one advisory lock held across admit → evict → load → verify, so admission and the load it authorises are a single critical section and no grant outlives a process.

**Tech Stack:** Same crate, same dependencies. `libc` (already present) supplies `flock`. No new crates — the GGUF reader is six keys of a documented header, not a parser.

**Spec:** [`../design.md`](../design.md) §§5–7, [`../architecture.md`](../architecture.md) §4. Supersedes the Evict column of [`../field-notes.md`](../field-notes.md), corrected in Task 1.

---

## What was verified on 2026-09-10, against the documentation

Every claim below was probed live against all four providers running at once. Two of them contradict what this project already wrote down, which is why Task 1 corrects the docs before any code is written.

**Two of four providers cannot be evicted at model granularity at all.**

| Provider | Documented in `field-notes.md` | Probed |
|---|---|---|
| LM Studio | `lms unload` | ✅ `lms load`/`unload` exist, with `-c`, `--ttl`, `--identifier` |
| Ollama | `ollama stop`, `keep_alive: 0` | documented only — **Task 6 must verify live** |
| llama.cpp | `POST /api/models/unload/<model>` | ❌ **404**, with a real model name, GET and POST alike. Router self-manages: `max_instances: 1`, `models_autoload: true` |
| vLLM-MLX | "registry eviction" | ❌ **no such route** in `/openapi.json`. `DELETE /v1/cache` is the prefix cache |

This is the third and fourth instance of the trap already recorded in `field-notes.md` under *an adapter written from documentation is a hypothesis*. `design.md` §6 anticipated the consequence — "a provider with no unload path is one the daemon can account for but never act on" — so the design survives; the docs do not.

**LM Studio's own estimator is a weights-only floor.** `lms load --estimate-only` on `qwen3-14b-claude-4.5-opus-high-reasoning-distill`:

```
-c 8192   → Estimated Total Memory: 8.38 GiB   Confidence: LOW
-c 40960  → Estimated Total Memory: 8.38 GiB   Confidence: LOW
```

Flat across a 5× context change, and 8.38 GiB is exactly the GGUF's size on disk. It is the `Declared` basis wearing a vendor badge, and it cannot carry admission. This is the second independent confirmation that the computed KV term is load-bearing.

**The corpus measures exactly one shape, and it was measured while swapping.** Counted 2026-09-10 over `~/.local/state/llm-harmony/observations.jsonl`:

| | |
|---|---|
| lines | 25 |
| usable (`schema: 2`, carrying `processes`) | 15 — the other 10 are dropped by `corpus::load` |
| distinct loaded shapes | **1** — `qwen3-14b-…-distill` at 40,960, nine times |

So `Basis::Measured` today covers one `(model, context)` pair on the whole machine, and **every other model on disk falls through to the weights-only `Declared` floor that `resolve` currently admits on.** The hazard in Task 4 is not a corner case; it is the normal path.

That single shape also reads `footprint_bytes: 9,604,429,576` with `swap_used_bytes: 5,507,448,832`. A resident-set measurement taken while the machine is paging is an **under**-count of what the model wanted. Task 4 must not calibrate against it without saying so.

---

## Global Constraints

- **A pin is a veto, never a reservation.** It prevents eviction; it cannot grant residency. Under pressure harmony refuses rather than crossing a pin — and therefore **every refusal names the pins that caused it**, or the operator sees an inexplicable "no room" on a half-empty machine.
- **An explicitly named model beats a pin; a search for room never does.** `unload X` on a pinned `X` proceeds, reports the override, and leaves the pin in place. Collateral eviction skips pinned models entirely.
- **Never evict a model with an in-flight request**, under any mode, pinned or not.
- **Never signal a process harmony does not own.** Eviction is always the provider's own verb. Process-level `stop` remains the separate authority slice 4 installed.
- **Bias high, always.** `design.md` §8: over-estimating wastes capacity, under-estimating wedges the machine. Where the computed basis is uncertain, round up.
- **Fail open.** Every failure — lock contention, a dead provider, a corrupt pin file — leaves the providers exactly as they were and exits non-zero with a reason. Delroy's `harmony.py` treats anything it cannot parse as no data; nothing here may break that.
- **`status` stays `schema: 1`.** New fields are additive only. Delroy pins `SCHEMA = 1` and treats a mismatch as no data, so a bump would silently disable the integration. The new verbs emit their own document.
- **No daemon, no socket, no proxy.** Decided 2026-09-10. If this slice grows a request path, the project has become llama-swap.
- macOS only, as slices 1–4.

---

### Task 1: Actuation capability, and correcting the record

**Files:**
- Modify: `src/provider.rs` (the `Adapter` trait, ~L133)
- Modify: `src/adapters/lmstudio.rs`, `src/adapters/ollama.rs`, `src/adapters/llamacpp.rs`, `src/adapters/vllm.rs`
- Modify: `src/main.rs` (new `verify` subcommand)
- Modify: `docs/field-notes.md`, `docs/design.md` §7
- Test: `tests/adapters.rs`

**Interfaces:**
- Produces: `provider::Actuation`, `Adapter::actuation()`. Every later task branches on this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/adapters.rs`:

```rust
use llm_harmony::provider::Actuation;

/// Probed live 2026-09-10: `lms load` and `lms unload` both exist.
#[test]
fn lmstudio_and_ollama_are_model_level() {
    assert!(matches!(LmStudio.actuation(), Actuation::ModelLevel));
    assert!(matches!(Ollama.actuation(), Actuation::ModelLevel));
}

/// Probed live 2026-09-10: `POST /api/models/unload/<real-model>` returns 404,
/// and vLLM-MLX publishes no load or unload route in its own openapi.json.
/// Their only ceilings are the ones they were started with.
#[test]
fn llamacpp_and_vllm_are_self_managed_and_name_their_ceiling() {
    match LlamaCpp.actuation() {
        Actuation::SelfManaged { ceiling } => assert_eq!(ceiling, "max_instances"),
        a => panic!("{a:?}"),
    }
    match Vllm.actuation() {
        Actuation::SelfManaged { ceiling } => assert_eq!(ceiling, "memory_budget_gb"),
        a => panic!("{a:?}"),
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test adapters`
Expected: FAIL — `no method named actuation`.

- [ ] **Step 3: Add the type and the trait method**

In `src/provider.rs`, above the `Adapter` trait:

```rust
/// What harmony may actually do to a provider's resident models.
///
/// This is a property of a **verified control surface**, never of config. A
/// provider whose unload path 404s is `SelfManaged` no matter what any TOML
/// file claims -- see `docs/field-notes.md`, *an adapter written from
/// documentation is a hypothesis*, which this enum exists to stop repeating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "actuation", rename_all = "kebab-case")]
pub enum Actuation {
    /// Load and unload one model at a time, by a verb probed against the server.
    ModelLevel,
    /// Loads on first request and evicts at its own ceiling. Harmony may admit,
    /// account, and bound it at install time -- never actuate it.
    SelfManaged { ceiling: &'static str },
}

impl Actuation {
    pub fn can_unload(&self) -> bool {
        matches!(self, Actuation::ModelLevel)
    }
}
```

Add to the trait (`src/provider.rs:133`):

```rust
    /// What this adapter may do beyond observing. Verified, not declared.
    fn actuation(&self) -> Actuation;
```

Implement in each adapter: `ModelLevel` for `LmStudio` and `Ollama`; `SelfManaged { ceiling: "max_instances" }` for `LlamaCpp`; `SelfManaged { ceiling: "memory_budget_gb" }` for `Vllm`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test adapters`
Expected: PASS.

- [ ] **Step 5: Add `llm-harmony verify`**

In `src/main.rs`, a subcommand that probes each configured provider and prints what it can really do — so the table in the docs can be regenerated instead of remembered:

```
llm-harmony verify [--json]

lmstudio   http://127.0.0.1:1234   reachable   model-level
ollama     http://127.0.0.1:11434  reachable   model-level
llamacpp   http://127.0.0.1:8080   reachable   self-managed (max_instances)
vllm       http://127.0.0.1:8000   reachable   self-managed (memory_budget_gb)
```

- [ ] **Step 6: Correct the documentation**

In `docs/field-notes.md`, replace the Evict column of the *Provider control surfaces* table with the probed results, and add an entry under *An adapter written from documentation is a hypothesis*:

> **Two evict paths in this table were never real.** Probed 2026-09-10 with all four providers up: llama.cpp's `POST /api/models/unload/<model>` returns 404 for a real model name on both GET and POST, and vLLM-MLX publishes no load or unload route in its own `/openapi.json`. Both self-manage — `max_instances: 1` and `memory_budget_gb` — so harmony can bound them at install time and account for them after, and can never unload one model from either. The original table was written from their documentation. Rule, third restatement: an evict path is verified by a 200 from the live server, not by its presence in a README.

In `docs/design.md` §7, correct the same table and add one line below it: *Eviction granularity is asymmetric. Two providers expose a model-level verb; two expose only a process and a ceiling.*

- [ ] **Step 7: Commit**

```bash
git add src/provider.rs src/adapters tests/adapters.rs src/main.rs docs/field-notes.md docs/design.md
git commit -m "declare actuation capability per provider, and correct two evict paths that were never real"
```

---

### Task 2: Pins

**Files:**
- Create: `src/pins.rs`
- Modify: `src/lib.rs` (`pub mod pins;`), `src/main.rs` (`pin`, `unpin`), `src/render.rs` (status column)
- Test: `src/pins.rs` unit tests, `tests/pins.rs`

**Interfaces:**
- Produces: `pins::{Pin, Pins}`; `Pins::is_pinned(provider, model) -> bool`, consumed by Task 7's planner.

- [ ] **Step 1: Write the failing tests**

Create `tests/pins.rs`:

```rust
use llm_harmony::pins::{Pin, Pins};
use llm_harmony::provider::ProviderKind;

fn pin(model: &str) -> Pin {
    Pin {
        provider: ProviderKind::LmStudio,
        model: model.to_string(),
        at: 1_788_900_000,
        note: None,
    }
}

#[test]
fn a_pin_round_trips_through_the_file() {
    let dir = tempdir();
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    p.save_to(&dir.join("pins.json")).unwrap();

    let back = Pins::load_from(&dir.join("pins.json"));
    assert!(back.is_pinned(ProviderKind::LmStudio, "qwen3-14b"));
}

/// A pin is per (provider, model): the same name on another provider is a
/// different model, and pinning one must not protect the other.
#[test]
fn a_pin_does_not_leak_across_providers() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(!p.is_pinned(ProviderKind::Ollama, "qwen3-14b"));
}

/// Sticky: a pin outlives the residency it was created for, so an unload --
/// deliberate or otherwise -- does not silently unprotect the next load.
#[test]
fn a_pin_survives_being_unloaded_and_is_cleared_only_by_unpin() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(p.is_pinned(ProviderKind::LmStudio, "qwen3-14b"));
    assert!(p.remove(ProviderKind::LmStudio, "qwen3-14b"));
    assert!(!p.is_pinned(ProviderKind::LmStudio, "qwen3-14b"));
}

#[test]
fn adding_the_same_pin_twice_is_idempotent() {
    let mut p = Pins::empty();
    assert!(p.add(pin("m")));
    assert!(!p.add(pin("m")), "already pinned");
    assert_eq!(p.all().len(), 1);
}

/// Fail open, per the global constraints: a corrupt pin file must not stop
/// harmony reading the ledger. It degrades to "nothing is pinned", which is
/// visible in `status` rather than silent.
#[test]
fn a_corrupt_pin_file_reads_as_no_pins_rather_than_an_error() {
    let dir = tempdir();
    let path = dir.join("pins.json");
    std::fs::write(&path, b"{ not json").unwrap();
    assert!(Pins::load_from(&path).all().is_empty());
}
```

Reuse the existing temp-directory helper from `tests/support/mod.rs` rather than writing a second one; if `tempdir()` is not already exported there, add it there and not here.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test pins`
Expected: FAIL — unresolved import `llm_harmony::pins`.

- [ ] **Step 3: Implement the store**

Create `src/pins.rs`:

```rust
use std::path::{Path, PathBuf};

use crate::provider::ProviderKind;

/// A model harmony may not evict to make room for something else.
///
/// Runtime state, deliberately not config: `docs/plans/2026-09-10-actuation.md`
/// requires a pin to be settable on a model that is *already loaded*, which a
/// file the user hand-edits cannot serve.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pin {
    pub provider: ProviderKind,
    pub model: String,
    /// Unix seconds, for `status` to show how long it has been held.
    pub at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Pins {
    #[serde(default = "one")]
    schema: u32,
    #[serde(default)]
    pins: Vec<Pin>,
}

fn one() -> u32 { 1 }

impl Pins {
    pub fn empty() -> Pins { Pins { schema: 1, pins: Vec::new() } }

    pub fn path() -> Option<PathBuf> {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".local/state/llm-harmony/pins.json"))
    }

    /// Never fails. A missing file is no pins; so is a corrupt one, because a
    /// pin store that can abort `status` would be worse than no pin store.
    pub fn load_from(path: &Path) -> Pins {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| Pins::empty()),
            Err(_) => Pins::empty(),
        }
    }

    pub fn load() -> Pins {
        match Pins::path() {
            Some(p) => Pins::load_from(&p),
            None => Pins::empty(),
        }
    }

    pub fn is_pinned(&self, provider: ProviderKind, model: &str) -> bool {
        self.pins.iter().any(|p| p.provider == provider && p.model == model)
    }

    pub fn all(&self) -> &[Pin] { &self.pins }

    /// `false` if it was already pinned.
    pub fn add(&mut self, pin: Pin) -> bool {
        if self.is_pinned(pin.provider, &pin.model) {
            return false;
        }
        self.pins.push(pin);
        true
    }

    /// `false` if it was not pinned.
    pub fn remove(&mut self, provider: ProviderKind, model: &str) -> bool {
        let before = self.pins.len();
        self.pins.retain(|p| !(p.provider == provider && p.model == model));
        self.pins.len() != before
    }

    /// Atomic: write a sibling temp file and rename, so an interrupted save
    /// cannot leave a half-written pin store behind.
    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, body).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self) -> Result<(), String> {
        match Pins::path() {
            Some(p) => self.save_to(&p),
            None => Err("no HOME; cannot locate the pin store".to_string()),
        }
    }
}
```

`ProviderKind` needs `Deserialize` for this; it already derives `Serialize` and has a `FromStr`. Add `#[derive(serde::Deserialize)]` with `#[serde(rename_all = "lowercase")]` so the on-disk spelling matches `as_str()`, and add a round-trip test in `src/provider.rs` asserting `serde_json::to_string` then `from_str` returns the same kind for all four.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test pins`
Expected: PASS, 5 tests.

- [ ] **Step 5: Wire the verbs and the status column**

`src/main.rs`:

```
llm-harmony pin   <model> [--provider P] [--note "why"]
llm-harmony unpin <model> [--provider P]
```

Resolution rule: with `--provider`, pin exactly that pair. Without it, resolve the model through `resolve::identity` and pin every candidate that names it — pinning "the model" rather than one provider's spelling of it. If nothing resolves, pin nothing and exit non-zero naming the model; a pin on a name no provider serves is a silent no-op waiting to surprise someone.

In `src/render.rs`, add a pin marker to the `status` rows, and `"pinned": true` to the JSON detail entries — additive, so the document stays `schema: 1` and `harmony.py` is unaffected.

- [ ] **Step 6: Commit**

```bash
git add src/pins.rs src/lib.rs src/main.rs src/render.rs src/provider.rs tests/pins.rs
git commit -m "pins: a runtime veto on eviction, settable while a model is resident"
```

---

### Task 3: Read a model's shape from its artifact

**Files:**
- Create: `src/estimate/shape.rs`
- Modify: `src/estimate/mod.rs`
- Test: `tests/shape.rs`

**Interfaces:**
- Produces: `estimate::shape::{ModelShape, from_artifact, from_gguf, from_mlx_config}`. Task 4 consumes `ModelShape`.

Nothing in the tree opens a model file today — artifact identity is by filename (`src/inventory/artifact.rs:45`). This is the first reader, and it is deliberately six keys wide, not a GGUF parser.

- [ ] **Step 1: Write the failing tests**

Create `tests/shape.rs`. The fixture is built in the test rather than committed, so no binary blob enters the repo:

```rust
use llm_harmony::estimate::shape::{self, ModelShape};

/// Minimal GGUF v3 header: magic, version, tensor count, kv count, then the
/// six keys the estimator needs. Format per the GGUF spec; only the metadata
/// block is written, since nothing here reads tensor data.
fn write_gguf(path: &std::path::Path, arch: &str, kv: &[(&str, u32)]) {
    let mut b: Vec<u8> = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());                       // tensor_count
    b.extend(((kv.len() + 1) as u64).to_le_bytes());    // kv_count, +1 for architecture

    let mut put_key = |b: &mut Vec<u8>, k: &str| {
        b.extend((k.len() as u64).to_le_bytes());
        b.extend(k.as_bytes());
    };
    put_key(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());                       // type 8 = string
    b.extend((arch.len() as u64).to_le_bytes());
    b.extend(arch.as_bytes());

    for (k, v) in kv {
        put_key(&mut b, k);
        b.extend(4u32.to_le_bytes());                   // type 4 = u32
        b.extend(v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

#[test]
fn a_gguf_header_yields_the_six_keys_the_estimator_needs() {
    let dir = tempdir();
    let p = dir.join("qwen3-14b-q4_k_m.gguf");
    write_gguf(&p, "qwen3", &[
        ("qwen3.block_count", 40),
        ("qwen3.attention.head_count", 40),
        ("qwen3.attention.head_count_kv", 8),
        ("qwen3.attention.key_length", 128),
        ("qwen3.embedding_length", 5120),
        ("qwen3.context_length", 40960),
    ]);

    let s: ModelShape = shape::from_gguf(&p).expect("readable header");
    assert_eq!(s.arch, "qwen3");
    assert_eq!(s.n_layers, 40);
    assert_eq!(s.n_kv_heads, 8);
    assert_eq!(s.head_dim, 128);
    assert_eq!(s.trained_context, Some(40960));
}

/// `key_length` is optional. When it is absent the head dimension is
/// embedding_length / head_count -- 5120/40 = 128 for this shape.
#[test]
fn head_dim_falls_back_to_embedding_over_heads() {
    let dir = tempdir();
    let p = dir.join("m.gguf");
    write_gguf(&p, "qwen3", &[
        ("qwen3.block_count", 40),
        ("qwen3.attention.head_count", 40),
        ("qwen3.attention.head_count_kv", 8),
        ("qwen3.embedding_length", 5120),
    ]);
    assert_eq!(shape::from_gguf(&p).unwrap().head_dim, 128);
}

/// Multi-head attention: kv heads default to attention heads when the key is
/// absent, which is what pre-GQA models publish.
#[test]
fn kv_heads_default_to_attention_heads() {
    let dir = tempdir();
    let p = dir.join("m.gguf");
    write_gguf(&p, "llama", &[
        ("llama.block_count", 32),
        ("llama.attention.head_count", 32),
        ("llama.embedding_length", 4096),
    ]);
    assert_eq!(shape::from_gguf(&p).unwrap().n_kv_heads, 32);
}

/// Never panic on a file that is not a GGUF, however it is malformed. This
/// reader runs over whatever a provider claims to be serving.
#[test]
fn a_file_that_is_not_a_gguf_is_none_not_a_panic() {
    let dir = tempdir();
    let p = dir.join("not.gguf");
    std::fs::write(&p, b"\x00\x01\x02").unwrap();
    assert!(shape::from_gguf(&p).is_none());
}

/// MLX ships plain JSON, so this half is trivial -- and it is the only half
/// that works for vLLM-MLX and LM Studio's MLX runtime.
#[test]
fn an_mlx_config_yields_the_same_shape() {
    let dir = tempdir();
    std::fs::write(dir.join("config.json"), r#"{
        "model_type": "qwen3",
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
        "max_position_embeddings": 40960
    }"#).unwrap();

    let s = shape::from_mlx_config(&dir).expect("readable config");
    assert_eq!((s.n_layers, s.n_kv_heads, s.head_dim), (40, 8, 128));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test shape`
Expected: FAIL — unresolved import.

- [ ] **Step 3: Implement the reader**

Create `src/estimate/shape.rs`. Read at most the first 1 MiB of the file: the metadata block precedes tensor data, and a bounded read means a 15 GB artifact costs one page-in rather than a scan. Walk the KV pairs, skipping values of types the estimator does not need (every GGUF value type has a known width; arrays carry an element type and a count, so they are skippable without being understood). Stop at the first of: all six keys found, KV count exhausted, or the bounded read ending — a truncated read is `None`, never a guess.

```rust
pub struct ModelShape {
    pub arch: String,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub weights_bytes: u64,
    pub trained_context: Option<u32>,
}

/// GGUF file or MLX directory, dispatched on what the path is.
pub fn from_artifact(path: &std::path::Path) -> Option<ModelShape>;
pub fn from_gguf(path: &std::path::Path) -> Option<ModelShape>;
pub fn from_mlx_config(dir: &std::path::Path) -> Option<ModelShape>;
```

`weights_bytes` is the artifact's size on disk — the GGUF's own length, or the sum of `*.safetensors` in the MLX directory. Both are facts already available from slice 2's scanner; take them from the filesystem here rather than threading the disk ledger in.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test shape`
Expected: PASS, 6 tests.

- [ ] **Step 5: Verify against a real artifact**

Run the reader over an artifact that is actually on this machine and check the numbers against what the provider reports:

```bash
cargo run -- ls --json | python3 -c "import json,sys; [print(a['path']) for m in json.load(sys.stdin)['models'] for a in m['artifacts']][:5]"
```

Take a real GGUF path and a real MLX directory, and confirm `arch`, `n_layers` and `n_kv_heads` match the model card. A header reader that passes synthetic fixtures and misreads a real file is the exact failure this project has already recorded twice.

- [ ] **Step 6: Commit**

```bash
git add src/estimate/shape.rs src/estimate/mod.rs tests/shape.rs
git commit -m "read layer count, kv heads and head dim from a GGUF header or an MLX config"
```

---

### Task 4: The computed basis, and demoting Declared

**Files:**
- Modify: `src/estimate/estimator.rs`, `src/resolve/decide.rs`
- Create: `src/estimate/computed.rs`
- Test: `tests/estimate_corpus.rs`, unit tests in `computed.rs`

**Interfaces:**
- Consumes: `shape::ModelShape` (Task 3).
- Produces: `Basis::Computed`; `computed::kv_bytes(&ModelShape, u32, u64) -> u64`; `estimator::ladder(measured, computed, declared) -> Estimate`, replacing today's `with_declared`.

This is where today's live hazard gets closed. `resolve::decide` currently admits on `Basis::Declared`, which `estimator.rs` documents in its own comment as weights-only — the under-estimate that wedges the machine.

- [ ] **Step 1: Write the failing tests**

In `src/estimate/computed.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::shape::ModelShape;

    fn qwen3_14b() -> ModelShape {
        ModelShape {
            arch: "qwen3".into(),
            n_layers: 40,
            n_kv_heads: 8,
            head_dim: 128,
            weights_bytes: 8_380_000_000,
            trained_context: Some(40_960),
        }
    }

    /// K and V, per layer, per kv head, per token, at the cache dtype.
    /// 2 x 40 x 8 x 128 x 8192 x 2 = 1,342,177,280.
    #[test]
    fn kv_scales_linearly_with_context() {
        let s = qwen3_14b();
        assert_eq!(kv_bytes(&s, 8_192, 2), 1_342_177_280);
        assert_eq!(kv_bytes(&s, 16_384, 2), 2_684_354_560);
    }

    /// The whole point of the basis: the figure must move when the window
    /// moves. LM Studio's own estimator does not, which is why it cannot
    /// carry admission -- see this plan's opening section.
    #[test]
    fn a_bigger_window_costs_more_than_a_smaller_one() {
        let s = qwen3_14b();
        assert!(computed(&s, 40_960, 2).bytes > computed(&s, 8_192, 2).bytes);
    }

    /// Weights are the floor and KV is the part that kills you; the estimate
    /// is both, plus the margin the constraints require.
    #[test]
    fn the_estimate_is_weights_plus_kv_plus_margin() {
        let s = qwen3_14b();
        let e = computed(&s, 8_192, 2);
        let floor = s.weights_bytes + kv_bytes(&s, 8_192, 2);
        assert!(e.bytes.expect("a computed estimate is always a number") > floor);
        assert_eq!(e.basis, Basis::Computed);
    }
}
```

In `src/resolve/decide.rs`, add to the existing test module:

```rust
/// The live hazard this task closes. A declared figure is weights-only by
/// construction -- vLLM's `memory_gb`, Ollama's `size`, a file's length on
/// disk, and `lms load --estimate-only` alike. Admitting on it is admitting
/// on an under-estimate.
#[test]
fn a_declared_figure_is_never_admitted_on_its_own() {
    let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
    let declared = Estimate {
        bytes: Some(2_000_000_000),
        basis: Basis::Declared,
        samples: 0,
        spread_bytes: None,
    };
    match decide(&c, &declared, &machine(20_000_000_000), 0) {
        Decision::Deny { reason, .. } => assert!(reason.contains("weights only"), "{reason}"),
        d => panic!("a declared floor must not admit: {d:?}"),
    }
}

#[test]
fn a_computed_estimate_that_fits_is_admitted() {
    let c = vec![cand(ProviderKind::LlamaCpp, State::NotLoaded, true)];
    let e = Estimate {
        bytes: Some(2_000_000_000),
        basis: Basis::Computed,
        samples: 0,
        spread_bytes: None,
    };
    match decide(&c, &e, &machine(20_000_000_000), 0) {
        Decision::Ready { resident, .. } => assert!(!resident),
        d => panic!("{d:?}"),
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test computed && cargo test --lib resolve`
Expected: FAIL — no `Basis::Computed`; the declared case still admits.

- [ ] **Step 3: Implement**

Create `src/estimate/computed.rs`:

```rust
/// f16, the cache dtype every one of these providers uses unless told
/// otherwise. llama.cpp can quantise it with `-ctk`/`-ctv` and publishes
/// nothing about the choice, so harmony assumes the expensive case and
/// documents the override. Biasing high is the rule (`design.md` section 8).
pub const KV_DTYPE_BYTES: u64 = 2;

/// Activations, compute buffers, the prefix cache, and the allocator's own
/// slack. A margin, not a measurement -- Step 5 is where it earns its value.
pub const MARGIN: f64 = 1.10;

pub fn kv_bytes(shape: &ModelShape, context_tokens: u32, kv_dtype_bytes: u64) -> u64 {
    2 // K and V
        * u64::from(shape.n_layers)
        * u64::from(shape.n_kv_heads)
        * u64::from(shape.head_dim)
        * u64::from(context_tokens)
        * kv_dtype_bytes
}

pub fn computed(shape: &ModelShape, context_tokens: u32, kv_dtype_bytes: u64) -> Estimate;
```

Add `Basis::Computed` between `Measured` and `Declared`, documented as *derived from the artifact's own metadata; bias-high, never observed*. Replace `with_declared` with `ladder(measured, computed, declared)`: return the first of measured, computed; if neither exists, return the declared figure **labelled but not admissible**. In `decide`, match `Basis::Measured | Basis::Computed` where it currently matches `Basis::Measured | Basis::Declared`, and give `Declared` its own denial: `"only a declared figure is available for this shape, which covers weights only"`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test`
Expected: PASS. The existing `resolve` tests that relied on declared admission will need updating — that is the behaviour change, not a regression.

- [ ] **Step 5: Calibrate the margin against the corpus, and write down what it says**

This step is not optional and its result is not predictable. Add to `tests/estimate_corpus.rs`:

```rust
/// Safety property: for every shape the corpus has actually measured, the
/// computed basis must not come in BELOW the measurement. Under-prediction is
/// the failure that wedges the machine; over-prediction only wastes room.
///
/// Ratio measured 2026-09-__ on the one shape the corpus covers: __x.
/// Fill that in from Step 5 -- a safety test whose margin nobody wrote down
/// is a safety test nobody can re-check.
#[test]
fn computed_never_under_predicts_a_measured_shape() {
    let t = Tree::new("corpus-calibration");
    let p = t.write("observations.jsonl", &format!("{V2}\n"));
    let obs = corpus::load(&p);

    let shape = ModelShape {
        arch: "qwen3".into(),
        n_layers: 40,
        n_kv_heads: 8,
        head_dim: 128,
        weights_bytes: 8_380_000_000,
        trained_context: Some(40_960),
    };

    let measured = estimator::for_model(&obs, "qwen3-14b", Some(40_960));
    let predicted = computed(&shape, 40_960, computed::KV_DTYPE_BYTES);

    let m = measured.bytes.expect("the fixture measures this shape");
    let c = predicted.bytes.expect("a computed estimate is always a number");
    assert!(
        c >= m,
        "computed {c} is below measured {m}: the estimator would admit a load \
         that the machine has already been observed unable to hold"
    );
}
```

Then run the comparison by hand and record the ratio in the test's doc comment:

```bash
cargo run -- estimate qwen3-14b-claude-4.5-opus-high-reasoning-distill --context 40960 --json
```

**Expect the formula to over-predict, possibly by a lot.** Hand-computed for the one measured shape: weights 8.38 GB + KV 6.71 GB at 40,960 = ~15 GB against a corpus observation of 9.60 GB. Three explanations are live, and the step is to find out which before tuning anything:

1. The observation was taken with **5.5 GB of swap in use** (see this plan's opening section), so it is an under-count of what the model wanted.
2. LM Studio may allocate the KV cache lazily, or quantise it by default.
3. The shape's real `n_kv_heads`/`head_dim` may differ from the hand-computed 8/128 — Task 3 Step 5 is what establishes them.

Record the finding in `docs/field-notes.md` under a new *KV allocation* heading whichever way it goes, and set `MARGIN` from data. If the formula over-predicts by more than ~1.5× on a real shape, add a config override — `kv_dtype_bytes` per provider — rather than shaving the margin, because a margin tuned down to fit one observation is how under-prediction gets shipped.

- [ ] **Step 6: Commit**

```bash
git add src/estimate/computed.rs src/estimate/estimator.rs src/resolve/decide.rs tests/estimate_corpus.rs docs/field-notes.md
git commit -m "compute a footprint from artifact metadata, and stop admitting on a weights-only floor"
```

---

### Task 5: The actuation lock

**Files:**
- Create: `src/actuate/mod.rs`, `src/actuate/lock.rs`
- Modify: `src/lib.rs`
- Test: `tests/lock.rs`

**Interfaces:**
- Produces: `actuate::lock::{ActuationLock, LockError, LockHolder}`; `ActuationLock::acquire(verb: &str, timeout: Duration)`. Task 9 wraps every actuating verb in one.

- [ ] **Step 1: Write the failing tests**

Create `tests/lock.rs`:

```rust
use std::time::Duration;
use llm_harmony::actuate::lock::{ActuationLock, LockError};

#[test]
fn a_second_acquirer_is_told_who_holds_it_and_since_when() {
    let dir = tempdir();
    let path = dir.join("actuation.lock");
    let _held = ActuationLock::acquire_at(&path, "switch", Duration::from_millis(0)).unwrap();

    match ActuationLock::acquire_at(&path, "load", Duration::from_millis(50)) {
        Err(LockError::Busy { holder, .. }) => {
            let h = holder.expect("the holder writes its identity into the file");
            assert_eq!(h.verb, "switch");
            assert_eq!(h.pid, std::process::id());
        }
        other => panic!("expected Busy, got {other:?}"),
    }
}

/// flock releases on process death, so a killed harmony cannot wedge the next
/// one. This is why the lock is flock and not a pidfile we have to reap.
#[test]
fn the_lock_is_released_when_the_holder_is_dropped() {
    let dir = tempdir();
    let path = dir.join("actuation.lock");
    {
        let _held = ActuationLock::acquire_at(&path, "switch", Duration::from_millis(0)).unwrap();
    }
    assert!(ActuationLock::acquire_at(&path, "load", Duration::from_millis(0)).is_ok());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test lock`
Expected: FAIL — unresolved import.

- [ ] **Step 3: Implement**

`flock(fd, LOCK_EX | LOCK_NB)` through `libc`, retried on a short interval until the timeout, holding the `File` in the struct so the lock releases on drop *and* on process death. After acquiring, truncate the file and write `{"pid":…,"verb":…,"started_at":…}` so a waiter can say who is holding it. Read that JSON on contention; a hold whose file is unreadable reports `holder: None` rather than failing.

```rust
pub struct ActuationLock { _file: std::fs::File }
pub struct LockHolder { pub pid: u32, pub verb: String, pub started_at: u64 }
pub enum LockError {
    Busy { holder: Option<LockHolder>, waited_s: u64 },
    Io(String),
}
impl ActuationLock {
    pub fn path() -> Option<std::path::PathBuf>;   // ~/.local/state/llm-harmony/actuation.lock
    pub fn acquire(verb: &str, timeout: std::time::Duration) -> Result<Self, LockError>;
    pub fn acquire_at(path: &std::path::Path, verb: &str, timeout: std::time::Duration) -> Result<Self, LockError>;
}
```

Default timeout for the verbs: 120 s. A load is slow, and failing fast here would mean two callers cannot queue behind each other at all.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test lock`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/actuate tests/lock.rs src/lib.rs
git commit -m "serialise actuation behind one advisory lock that dies with its holder"
```

---

### Task 6: Load and unload, per adapter

**Files:**
- Modify: `src/provider.rs`, `src/adapters/lmstudio.rs`, `src/adapters/ollama.rs`, `src/adapters/llamacpp.rs`, `src/adapters/vllm.rs`
- Test: `tests/adapters.rs`

**Interfaces:**
- Consumes: `Actuation` (Task 1).
- Produces: `Adapter::{load, unload, busy}`, `provider::{LoadRequest, ActuateError}`. Tasks 8 and 9 consume these.

- [ ] **Step 1: Write the failing tests**

```rust
/// A self-managed provider refuses in a way that names the lever that does
/// exist, so the operator is never told only what harmony cannot do.
#[test]
fn a_self_managed_provider_refuses_unload_and_names_its_ceiling() {
    match LlamaCpp.unload(&http(), "http://127.0.0.1:1", "any-model") {
        Err(ActuateError::NotSupported { ceiling }) => assert_eq!(ceiling, "max_instances"),
        other => panic!("{other:?}"),
    }
}

/// Ollama drops a model by asking for it with keep_alive 0. Verified against
/// a stub here and against the live server in Step 5.
#[test]
fn ollama_unloads_by_asking_for_a_zero_keep_alive() {
    let seen = support::RecordingServer::start();
    Ollama.unload(&http(), &seen.base_url(), "qwen3:14b").unwrap();
    let req = seen.last_request();
    assert_eq!(req.path, "/api/generate");
    assert_eq!(req.json["keep_alive"], 0);
    assert_eq!(req.json["model"], "qwen3:14b");
    assert_eq!(req.json["prompt"], "", "never carry content: docs/architecture.md");
}

/// LM Studio is driven by its CLI, which is the only verb it exposes. The
/// requested window must reach it, because the window is most of the estimate.
#[test]
fn lmstudio_load_passes_the_requested_context_to_the_cli() {
    let argv = LmStudio.load_argv(&LoadRequest {
        model: "qwen3-14b".into(),
        context_tokens: Some(8192),
    });
    assert_eq!(argv, vec!["lms", "load", "qwen3-14b", "-c", "8192", "-y"]);
}
```

`RecordingServer` is a stub that captures the last request; if `tests/support/mod.rs` has no such helper, add it there beside `StubServer`.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test adapters`
Expected: FAIL — no `load`/`unload`/`busy`.

- [ ] **Step 3: Implement**

```rust
pub struct LoadRequest { pub model: String, pub context_tokens: Option<u32> }

#[derive(Debug)]
pub enum ActuateError {
    /// This provider has no model-level verb. Carries the ceiling that bounds
    /// it instead, so the caller can say something useful.
    NotSupported { ceiling: &'static str },
    Failed { reason: String },
    Timeout { after_s: u64 },
}
```

- **LM Studio** — `lms load <model> [-c N] -y` and `lms unload <model>`, as subprocesses. Split `load_argv`/`unload_argv` out as pure functions so the argv is unit-testable without spawning. `lms` missing from `PATH` is `Failed { reason }` naming the binary, not a panic.
- **Ollama** — `POST /api/generate` with `{"model": m, "prompt": "", "keep_alive": 0}` to drop; the same with `"keep_alive": "10m"` to load. This needs `Http::post_json`, which does not exist yet — add it beside `get_json` in `src/http.rs`, with the same `ProbeError` mapping.
- **llama.cpp, vLLM-MLX** — both verbs return `NotSupported { ceiling }` immediately, without touching the network.
- **`busy`** — `true` when the provider reports an in-flight request for that model, `false` when it reports none, and `Err` when it cannot say. Callers treat `Err` as busy: the global constraints forbid evicting a model with a request in flight, so an unknown must not read as free.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test adapters`
Expected: PASS.

- [ ] **Step 5: Verify against the live servers**

Fixtures prove the adapter matches a hypothesis; only the server proves the hypothesis. With LM Studio and Ollama running:

```bash
# Ollama: load, confirm residency, drop, confirm it is gone.
cargo run -- load nomic-embed-text:latest --provider ollama
cargo run -- status | grep -i nomic
cargo run -- unload nomic-embed-text:latest --provider ollama
cargo run -- status | grep -i nomic

# LM Studio: the same, with a small model and an explicit window.
cargo run -- load text-embedding-nomic-embed-text-v1.5 --provider lmstudio --context 2048
cargo run -- status | grep -i embedding
cargo run -- unload text-embedding-nomic-embed-text-v1.5 --provider lmstudio
```

Use the embedding models for this, not a 14B: the point is to verify the control verb, and a small model makes a failed unload cheap to recover from. Record anything surprising in `docs/field-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add src/provider.rs src/adapters src/http.rs tests/adapters.rs tests/support
git commit -m "load and unload where a model-level verb exists; refuse by name where it does not"
```

---

### Task 7: The eviction planner

**Files:**
- Create: `src/actuate/plan.rs`
- Test: unit tests in `plan.rs`

**Interfaces:**
- Consumes: `resolve::identity::Candidate`, `pins::Pins`, `estimate::estimator::Estimate`, `memory::Machine`.
- Produces: `actuate::plan::{Request, Action, UnloadReason, Plan, plan}`. Task 9 executes what this returns.

A pure function: ledger plus pins plus estimate in, an ordered list of actions out. Everything interesting about this slice is decided here, and none of it needs a provider to be running.

- [ ] **Step 1: Write the failing tests**

Helpers first, defined once — every test below uses them:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn machine(free: u64) -> Machine {
        Machine {
            total_bytes: 24 * GB,
            used_bytes: 24 * GB - free,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        }
    }

    fn measured(bytes: u64) -> Estimate {
        Estimate { bytes: Some(bytes), basis: Basis::Measured, samples: 3, spread_bytes: Some(0) }
    }

    /// One candidate, on a provider that can actually be actuated.
    fn candidates(model: &str) -> Vec<Candidate> {
        vec![Candidate {
            provider: ProviderKind::LmStudio,
            provider_model_id: model.to_string(),
            url: "http://127.0.0.1:1234".into(),
            state: State::NotLoaded,
            artifact: Some("a".into()),
            provenance: Provenance::Recorded,
        }]
    }

    fn res(model: &str, bytes: u64, last_used: u64) -> Resident {
        Resident {
            provider: ProviderKind::LmStudio,
            model: model.to_string(),
            estimated_bytes: bytes,
            last_used,
            busy: false,
            actuation: Actuation::ModelLevel,
        }
    }

    fn resident(models: &[&str]) -> Vec<Resident> {
        models.iter().enumerate().map(|(i, m)| res(m, 9 * GB, i as u64)).collect()
    }

    fn pins_with(entries: &[(ProviderKind, &str)]) -> Pins {
        let mut p = Pins::empty();
        for (provider, model) in entries {
            p.add(Pin { provider: *provider, model: model.to_string(), at: 0, note: None });
        }
        p
    }

    fn req(model: &str) -> Request {
        Request { model: model.into(), context_tokens: None, free_first: None, pin_after: false }
    }

    fn req_switch(from: &str, to: &str) -> Request {
        Request {
            model: to.into(),
            context_tokens: None,
            free_first: Some(from.into()),
            pin_after: false,
        }
    }
```

Then the cases:

```rust
/// A pin is a veto. The room exists, the model is idle, and it still does not
/// move -- that is the whole point of the flag.
#[test]
fn a_pinned_model_is_never_chosen_to_make_room() {
    let p = pins_with(&[(ProviderKind::LmStudio, "big-pinned")]);
    match plan(&candidates("wanted"), &resident(&["big-pinned"]), &p,
               &measured(9_000_000_000), &machine(2_000_000_000), 0, &req("wanted")) {
        Plan::Refused { pinned_blockers, .. } => {
            assert_eq!(pinned_blockers, vec!["big-pinned on lmstudio"]);
        }
        p => panic!("{p:?}"),
    }
}

/// And because it is a veto rather than a reservation, the refusal has to say
/// so -- otherwise the operator reads "no room" on a machine that looks half
/// empty and has no way to discover why.
#[test]
fn a_refusal_caused_by_pins_names_them_in_its_reason() {
    let p = pins_with(&[(ProviderKind::LmStudio, "big-pinned")]);
    match plan(&candidates("wanted"), &resident(&["big-pinned"]), &p,
               &measured(9 * GB), &machine(2 * GB), 0, &req("wanted")) {
        Plan::Refused { reason, .. } => {
            assert!(reason.contains("pinned"), "{reason}");
            assert!(reason.contains("big-pinned"), "{reason}");
        }
        p => panic!("{p:?}"),
    }
}

/// An explicitly named model beats a pin: the human named it, that is consent.
/// The pin survives, so the next load is protected again.
#[test]
fn naming_a_pinned_model_for_unload_overrides_the_pin_but_keeps_it() {
    let p = pins_with(&[(ProviderKind::LmStudio, "pinned")]);
    match plan(&candidates("b"), &resident(&["pinned"]), &p,
               &measured(1_000_000_000), &machine(20_000_000_000), 0,
               &req_switch("pinned", "b")) {
        Plan::Actions(a) => {
            assert!(matches!(&a[0], Action::Unload { model, reason: UnloadReason::Named, .. } if model == "pinned"));
        }
        p => panic!("{p:?}"),
    }
}

/// Busy beats everything. A model serving a request is not a candidate, and a
/// provider that cannot say whether it is busy counts as busy (Task 6).
#[test]
fn a_busy_model_is_never_evicted_even_when_it_is_the_only_candidate() {
    let mut busy = res("serving", 9 * GB, 0);
    busy.busy = true;
    match plan(&candidates("wanted"), &[busy], &Pins::empty(),
               &measured(9 * GB), &machine(2 * GB), 0, &req("wanted")) {
        Plan::Refused { reason, pinned_blockers, .. } => {
            assert!(reason.contains("in flight"), "{reason}");
            assert!(pinned_blockers.is_empty(), "busy is not a pin");
        }
        p => panic!("a model with a request in flight must not be evicted: {p:?}"),
    }
}

/// `switch` means replace, per the 2026-09-10 decision: it frees the named
/// model first even when both would have fitted. `load` is how two models end
/// up resident together.
#[test]
fn switch_always_unloads_the_named_model_even_when_both_would_fit() {
    match plan(&candidates("b"), &resident(&["a"]), &Pins::empty(),
               &measured(1_000_000_000), &machine(20_000_000_000), 0, &req_switch("a", "b")) {
        Plan::Actions(a) => {
            assert_eq!(a.len(), 2);
            assert!(matches!(&a[0], Action::Unload { reason: UnloadReason::Named, .. }));
            assert!(matches!(&a[1], Action::Load { .. }));
        }
        p => panic!("{p:?}"),
    }
}

/// `load` adds. With room for both, nothing is disturbed -- this is the
/// multi-residency case, and it must cost zero unloads.
#[test]
fn load_does_not_evict_anything_when_the_model_fits_alongside() {
    match plan(&candidates("b"), &resident(&["a"]), &Pins::empty(),
               &measured(1_000_000_000), &machine(20_000_000_000), 0, &req("b")) {
        Plan::Actions(a) => assert!(a.iter().all(|x| matches!(x, Action::Load { .. }))),
        p => panic!("{p:?}"),
    }
}

/// Free the least it can. Evicting two models when one suffices is a bug that
/// looks like caution.
#[test]
fn making_room_evicts_the_smallest_sufficient_set_lru_first() {
    // 3 GB free, 4 GB wanted. Either resident model alone releases enough;
    // `old` was used longer ago, so it is the one that goes -- alone.
    let residents = vec![res("old", 5 * GB, 1), res("recent", 5 * GB, 99)];
    match plan(&candidates("wanted"), &residents, &Pins::empty(),
               &measured(4 * GB), &machine(3 * GB), 0, &req("wanted")) {
        Plan::Actions(a) => {
            let unloads: Vec<&str> = a.iter().filter_map(|x| match x {
                Action::Unload { model, .. } => Some(model.as_str()),
                _ => None,
            }).collect();
            assert_eq!(unloads, vec!["old"], "one unload, least-recently-used");
        }
        p => panic!("{p:?}"),
    }
}

/// A model resident on a self-managed provider cannot be evicted at all
/// (Task 1), so it can never appear in a plan. The refusal names the ceiling
/// and the `stop` verb instead of offering something harmony cannot do.
#[test]
fn a_self_managed_residency_is_never_planned_for_eviction() {
    let mut stuck = res("on-llamacpp", 9 * GB, 0);
    stuck.provider = ProviderKind::LlamaCpp;
    stuck.actuation = Actuation::SelfManaged { ceiling: "max_instances" };

    match plan(&candidates("wanted"), &[stuck], &Pins::empty(),
               &measured(9 * GB), &machine(2 * GB), 0, &req("wanted")) {
        Plan::Refused { reason, .. } => {
            assert!(reason.contains("max_instances"), "name the ceiling: {reason}");
            assert!(reason.contains("stop"), "name the lever that does exist: {reason}");
        }
        p => panic!("harmony cannot unload one model from llama.cpp: {p:?}"),
    }
}

/// Residency short-circuits everything, exactly as `resolve::decide` already
/// does: it costs what it costs whether or not we answer.
#[test]
fn a_model_already_resident_is_reported_not_reloaded() {
    let mut c = candidates("wanted");
    c[0].state = State::Loaded;
    // Deliberately no headroom at all -- residency must win anyway.
    match plan(&c, &resident(&["wanted"]), &Pins::empty(),
               &measured(9 * GB), &machine(0), 0, &req("wanted")) {
        Plan::AlreadyResident { model, .. } => assert_eq!(model, "wanted"),
        p => panic!("{p:?}"),
    }
}
}   // mod tests
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib actuate::plan`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```rust
pub struct Request {
    pub model: String,
    pub context_tokens: Option<u32>,
    /// `switch`: the model to free first, whatever the arithmetic says.
    pub free_first: Option<String>,
    pub pin_after: bool,
}

pub enum UnloadReason { Named, MakingRoom }

pub enum Action {
    Unload { provider: ProviderKind, model: String, reason: UnloadReason },
    Load { provider: ProviderKind, model: String, context_tokens: Option<u32> },
}

pub enum Plan {
    Actions(Vec<Action>),
    AlreadyResident { provider: ProviderKind, model: String, base_url: String },
    Refused { reason: String, pinned_blockers: Vec<String>, alternatives: Vec<String> },
}

pub fn plan(
    candidates: &[Candidate],
    resident: &[Resident],
    pins: &Pins,
    estimate: &Estimate,
    machine: &Machine,
    reserve_bytes: u64,
    request: &Request,
) -> Plan;
```

Order of reasoning: honour `free_first` unconditionally → return early if already resident → compute headroom as `free − reserve` minus whatever `free_first` releases → if it fits, load with no eviction → otherwise choose victims from `{ model-level provider, not pinned, not busy, not the target }`, LRU first, smallest sufficient set → if no such set exists, `Refused`, listing every pinned model that would have been a candidate.

`Resident` is the planner's view of one loaded model — assembled by Task 9 from the ledger, so this function stays pure and testable without a network:

```rust
pub struct Resident {
    pub provider: ProviderKind,
    pub model: String,
    /// What it is costing, by the same ladder admission uses (Task 4).
    pub estimated_bytes: u64,
    /// Unix seconds, for LRU. Providers that publish nothing get the time
    /// harmony first saw them resident, which is worse than a real figure and
    /// better than an arbitrary order.
    pub last_used: u64,
    /// A request is in flight. Unknown counts as busy (Task 6).
    pub busy: bool,
    pub actuation: Actuation,
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib actuate::plan`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/actuate/plan.rs
git commit -m "plan an eviction: pins veto, busy is untouchable, free the smallest sufficient set"
```

---

### Task 8: The load watchdog

**Files:**
- Create: `src/actuate/watchdog.rs`
- Test: unit tests in `watchdog.rs`

**Interfaces:**
- Produces: `actuate::watchdog::{Limits, Outcome, supervise}`. Task 9 wraps every load in it.

The estimate can be wrong. This is what makes the guarantee hold anyway. It takes its samples through an injected closure, so the failure paths are testable without allocating 9 GB.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    /// A machine sample with `n` GB free and no swap in use.
    fn free(n: u64) -> Machine {
        Machine {
            total_bytes: 24 * GB,
            used_bytes: 24 * GB - n * GB,
            swap_total_bytes: 8 * GB,
            swap_used_bytes: 0,
        }
    }

    /// A machine sample with plenty free and `n` GB of swap in use.
    fn swap(n: u64) -> Machine {
        Machine { swap_used_bytes: n * GB, ..free(12) }
    }

    fn limits() -> Limits {
        Limits {
            floor_bytes: 4 * GB,
            max_swap_growth_bytes: 2 * GB,
            poll: std::time::Duration::from_millis(0),
            timeout: std::time::Duration::from_millis(50),
        }
    }

    /// Hand back one prepared sample per poll, repeating the last one forever
    /// so a test never depends on how many times `supervise` looks.
    fn stepper(samples: Vec<Machine>) -> impl FnMut() -> Machine {
        let mut i = 0;
        move || {
            let s = samples[i.min(samples.len() - 1)].clone();
            i += 1;
            s
        }
    }

    fn after_n_polls(n: usize) -> impl FnMut() -> bool {
        let mut i = 0;
        move || { i += 1; i >= n }
    }

    fn never_resident() -> impl FnMut() -> bool { || false }

    /// The ordinary path: pressure stays fine, residency appears, done.
    #[test]
    fn a_load_that_stays_within_the_floor_completes() {
        let samples = vec![free(12), free(11), free(10)];
        let out = supervise(&limits(), stepper(samples), after_n_polls(3));
        assert!(matches!(out, Outcome::Loaded { .. }));
    }

    /// The case this exists for: free memory crosses the floor mid-load and
    /// the load is abandoned rather than completed.
    #[test]
    fn crossing_the_floor_aborts_the_load() {
        let samples = vec![free(12), free(6), free(1)];
        match supervise(&limits(), stepper(samples), never_resident()) {
            Outcome::Aborted { reason } => assert!(reason.contains("free memory"), "{reason}"),
            o => panic!("{o:?}"),
        }
    }

    /// Swap growth is the second signal, and on unified memory it is the one
    /// that means the machine is already losing. `field-notes.md` measured
    /// 30,187 pageouts with a 15 GB model resident.
    #[test]
    fn swap_growing_during_the_load_aborts_it_even_above_the_floor() {
        let samples = vec![swap(0), swap(1), swap(4)];
        match supervise(&limits(), stepper(samples), never_resident()) {
            Outcome::Aborted { reason } => assert!(reason.contains("swap"), "{reason}"),
            o => panic!("{o:?}"),
        }
    }

    /// A provider that never reports residency must not hang the caller.
    #[test]
    fn a_load_that_never_becomes_resident_times_out() {
        match supervise(&limits(), stepper(vec![free(12); 100]), never_resident()) {
            Outcome::TimedOut => {}
            o => panic!("{o:?}"),
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib actuate::watchdog`
Expected: FAIL.

- [ ] **Step 3: Implement**

```rust
pub struct Limits {
    /// Abort below this much free physical memory. Defaults to half the
    /// reserve: the reserve is what the OS is owed, and crossing half of it
    /// mid-load means the estimate was wrong by more than the margin.
    pub floor_bytes: u64,
    /// Abort if swap in use grows by more than this during the load.
    pub max_swap_growth_bytes: u64,
    pub poll: std::time::Duration,   // 250 ms
    pub timeout: std::time::Duration, // 180 s
}

pub enum Outcome {
    Loaded { took_s: u64, footprint_bytes: u64 },
    Aborted { reason: String },
    TimedOut,
}

pub fn supervise<S, R>(limits: &Limits, mut sample: S, mut resident: R) -> Outcome
where
    S: FnMut() -> Machine,
    R: FnMut() -> bool;
```

`supervise` only *decides*; the caller performs the abort, because only the caller knows the adapter. Task 9 wires `Aborted` to an immediate `unload` — and per Task 1's asymmetry, a `SelfManaged` provider has no undo, so Task 9 must require a stricter margin before triggering one of those loads rather than relying on being able to abort.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib actuate::watchdog`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/actuate/watchdog.rs
git commit -m "supervise a load against real memory and abort before the machine swaps"
```

---

### Task 9: The verbs

**Files:**
- Modify: `src/main.rs`
- Create: `src/render_actuate.rs`
- Test: `tests/integration.rs`

**Interfaces:**
- Consumes: every prior task.
- Produces: the `load`/`unload`/`switch` CLI and their JSON document.

- [ ] **Step 1: Write the failing tests**

In `tests/integration.rs`, exercising the binary with no provider running — the fail-open paths, which are the ones Delroy will actually hit:

```rust
/// Nothing running, so nothing can be loaded -- and the exit must be a clean
/// refusal with a reason, never a panic or a hang.
#[test]
fn load_with_no_provider_running_refuses_cleanly() {
    let out = run(&["load", "some-model", "--config", &empty_config()]);
    assert_ne!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no provider serves this model"));
}

/// The JSON document carries its own schema, so `status` can stay at 1 and
/// harmony.py keeps working untouched.
#[test]
fn the_actuate_document_carries_its_own_schema() {
    let out = run(&["load", "some-model", "--json", "--config", &empty_config()]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["schema"], 1);
    assert_eq!(v["verb"], "load");
    assert_eq!(v["outcome"]["status"], "refused");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test integration`
Expected: FAIL — unknown subcommand `load`.

- [ ] **Step 3: Implement the verbs**

```
llm-harmony load   <model> [--context N] [--provider P] [--pin] [--reserve BYTES] [--floor BYTES] [--json]
llm-harmony unload <model> [--provider P] [--json]
llm-harmony switch <from> <to> [--context N] [--provider P] [--pin] [--reserve BYTES] [--floor BYTES] [--json]
```

`--reserve` matches the flag `resolve` and `estimate` already take. `--floor` overrides the watchdog's abort threshold (Task 8 `Limits::floor_bytes`), defaulting to half the reserve; Task 10 Step 4 uses it to prove the watchdog fires without needing a load large enough to endanger the machine.

Each one, in order: acquire the lock (Task 5) → read the ledger and the pins → resolve candidates and estimate (Tasks 3–4) → `plan()` (Task 7) → execute actions in order, each unload verified by a re-read rather than trusted → wrap the load in `supervise()` (Task 8) → on `Aborted`, unload immediately and report it → verify residency → record an observation → apply `--pin` → render.

The output document:

```json
{
  "schema": 1,
  "verb": "switch",
  "outcome": {
    "status": "ready",
    "base_url": "http://127.0.0.1:1234",
    "provider": "lmstudio",
    "provider_model_id": "qwen3-14b-...",
    "resident": true,
    "took_s": 7
  },
  "unloaded": [{ "provider": "lmstudio", "model": "...", "reason": "named" }],
  "estimate": { "bytes": 9600000000, "basis": "computed" },
  "machine": { "free_bytes": 7000000000, "swap_used_bytes": 0 }
}
```

`outcome.status` is one of `ready` | `refused` | `failed` | `aborted`. Per the 2026-09-10 decision there is **no rollback**: when a switch unloads `from` and then fails to load `to`, the document reports both, and the human-readable render prints the exact command that restores what was unloaded. That is legibility, not recovery — do not add a retry.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test`
Expected: PASS, whole suite.

- [ ] **Step 5: The warm-up, and the promise it edits**

For a `SelfManaged` provider there is no load verb, so `ready` can only mean *resident* if harmony triggers the JIT load itself: a single `POST /v1/completions` with a fixed sentinel prompt and `max_tokens: 1`.

This puts harmony on a completion endpoint, against `architecture.md`'s "zero presence in traffic". Implement it, bounded — fixed sentinel, one token, never user content, only when the caller asked for residency — and amend `architecture.md` in the same commit to state the narrower claim that is actually load-bearing:

> **No user prompt passes through harmony.** It never proxies, never forwards, and never sees a conversation. It may send one fixed sentinel token to a provider that has no control-plane load verb, because otherwise `ready` would mean "advertised" rather than "resident" — the exact distinction that made `/v1/models` unusable as a residency signal (`field-notes.md`, *advertised is not resident*).

If that trade is unwanted, the alternative is to return `resident: false` for self-managed providers and let the caller's first request do the loading — at the cost of harmony's ledger being wrong until it happens. Do not implement both.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs src/render_actuate.rs tests/integration.rs docs/architecture.md
git commit -m "load, unload and switch, serialised and supervised"
```

---

### Task 10: Physical verification

**Files:**
- Modify: `docs/field-notes.md`, `docs/roadmap.md`

`design.md` §10 is explicit that the interesting behaviour here is physical, and no unit test reaches it.

- [ ] **Step 1: Prove the refusal**

With the 14B Q8 (15 GB) resident — the shape `field-notes.md` already measured at 8% free with 30,187 pageouts — ask for a second large model:

```bash
cargo run -- load <second-large-model> --context 32768
```

Expected: a refusal naming the arithmetic, with nothing unloaded and nothing loaded. Record the actual output.

- [ ] **Step 2: Prove the switch**

```bash
cargo run -- switch <resident-14b> <9b-on-another-provider> --context 8192
cargo run -- status
```

Expected: the first is gone, the second is resident, and `status` agrees. Time it — the wall-clock number belongs in the field notes, because "quick switch" is the claim being made.

- [ ] **Step 3: Prove the pin holds**

```bash
cargo run -- pin <resident-model> --note "in use"
cargo run -- load <something-that-needs-the-room>
```

Expected: refused, naming the pin. Then `unpin` and confirm the same command now succeeds.

- [ ] **Step 4: Prove the watchdog fires**

Set the floor artificially high (`--floor` or a config override) so an ordinary load crosses it, and confirm the load is aborted and the partially-loaded model unloaded. A watchdog that has never fired is a watchdog that has never been tested.

- [ ] **Step 5: Record and close the slice**

Write the measured numbers into `docs/field-notes.md` under a new *Actuation* heading, and update the `docs/roadmap.md` state table: slice 4 complete, this slice complete, and — if Task 4's calibration changed the picture — whatever it says about the reserve, which `design.md` §9 Q3 still lists as an open question.

- [ ] **Step 6: Commit**

```bash
git add docs/field-notes.md docs/roadmap.md
git commit -m "verify actuation against the machine, and record what it cost"
```

---

## What this slice deliberately does not do

- **No daemon and no socket.** Decided 2026-09-10: because harmony performs the load itself, admission and load are one critical section, so no grant has to outlive a process — which removes the daemon's justification for the second time (slice 3 removed it the first). Slice 6's hosting API is where a socket earns its keep.
- **No rollback.** A failed switch reports what is gone and how to restore it. Chosen over restore-on-failure because a rollback that itself fails is a worse position to be in than a clear report.
- **No disk eviction.** Clearing caches to make *storage* room is slice 5's half of the two-ledger claim.
- **No Delroy change.** `status` stays `schema: 1` with additive fields; `harmony.py` is untouched by this slice. Teaching Delroy to call `switch` is a separate piece of work with its own decision about whether a turn may ever wait.
- **No control over JIT loads.** llama.cpp and vLLM-MLX load on first request. Harmony bounds them at install time (`max_instances`, `memory_budget_gb`) and accounts for them after; it cannot stand in front of them, and this slice does not pretend otherwise.

## Risks

**The computed basis may over-predict badly enough to be useless.** Hand-computation for the one measured shape suggests ~15 GB predicted against a 9.6 GB observation. Task 4 Step 5 exists to find out why before tuning, and the answer changes what ships: a lazily-allocated or quantised KV cache means the formula needs a per-provider override, not a smaller margin. Shaving the margin to fit one observation is how under-prediction gets shipped, and under-prediction is the one failure that costs the machine.

**The corpus is 25 observations and one of them was taken while swapping.** Every measured admission rests on that. This slice grows the corpus with every supervised load, which is the only real fix, but early admissions will lean on `Computed` far more than on `Measured`.

**`lms` is a subprocess on `PATH`.** LM Studio's only control verb is its CLI, so harmony's eviction power there depends on a binary being findable. When it is not, the adapter must degrade to `Failed` with the binary named — never to a silent no-op. (A related trap, already live: `llm-harmony` itself is installed at `~/.cargo/bin`, which is not on this machine's `PATH`, which is why Delroy's memory notice has never once fired.)
