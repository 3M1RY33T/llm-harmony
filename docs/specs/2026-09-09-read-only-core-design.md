# Slice 1 — the read-only core

Spec, 2026-09-09. Companion to [`../design.md`](../design.md), which describes
the whole daemon. This describes the first slice of it that gets built.

Approved shape: one crate (`lib.rs` + `main.rs`), blocking HTTP, threads for
concurrency, Rust.

---

## 1. Scope

Four provider adapters, a ledger assembled from them, and one command:

```
llm-harmony status [--json]
```

It polls every configured provider once, attributes memory, prints a table, and
exits.

**In scope:** observation. Adapter discovery, listing loaded models, reading the
serving context window, per-provider memory attribution, machine totals.

**Out of scope, and each its own later slice:** the admission protocol
(`ASK`/`COMMIT`/`RELEASE`), eviction, the footprint estimator, the daemon and its
poll loop, persistence, and the entire inventory half of
[`../inventory.md`](../inventory.md).

### Why this slice first

Both halves of llm-harmony need to know what each provider is currently serving —
admission needs it to do arithmetic, inventory needs it to answer *"is this
model served live?"*. Building it once, first, and proving it against real
providers is cheaper than discovering the adapters are wrong from inside a
subsystem that also unloads things.

It is also the slice where being wrong is free. Nothing is unloaded and nothing
is deleted, so a bug is a wrong number on a terminal rather than a wedged
machine.

## 2. The safety property

`status` cannot unload a model. Not because policy forbids it — because the
`Adapter` trait has no method that could, and no adapter links the provider's
unload endpoint. The property is structural and is checked by reading the trait,
not by reading a config file.

This is what makes it reasonable to run slice 1 against a live machine on day
one, before any of the estimation work in `design.md` §5 is trustworthy.

## 3. Architecture

```
src/
  lib.rs           re-exports
  config.rs        endpoints, built-in defaults, TOML load
  provider.rs      trait Adapter, ProviderKind, LoadedModel, State
  adapters/
    lmstudio.rs
    llamacpp.rs
    vllm.rs
    ollama.rs
  memory.rs        machine total, per-pid phys_footprint, swap pressure
  ledger.rs        rows -> report
  main.rs          clap: status [--json]
```

### Why blocking HTTP and threads

Four endpoints, on loopback, on one host. `tokio` would add a large dependency
tree, slow builds, and `async` colouring on every adapter method in exchange for
concurrency that four `std::thread::spawn` calls already provide. The slice-2
daemon is `loop { poll; sleep }` around this same library and does not need
async either.

Dependencies, kept deliberately small:

| crate | for |
|---|---|
| `ureq` | blocking HTTP, no async runtime |
| `serde` / `serde_json` | provider response parsing |
| `toml` | config |
| `clap` | argument parsing |
| `libproc` | macOS `phys_footprint` per pid |

## 4. The adapter interface

```rust
pub trait Adapter {
    fn kind(&self) -> ProviderKind;
    fn probe(&self, http: &Http) -> Result<bool>;
    fn list(&self, http: &Http) -> Result<Vec<LoadedModel>>;
    fn pids(&self) -> Result<Vec<u32>>;
}
```

- `probe` — confirms an endpoint is the provider it was declared to be.
- `list` — every model the provider knows about, with its state.
- `pids` — the processes whose memory belongs to this provider.

`unload` and `busy` from `design.md` §7 are absent by design; they arrive with
the eviction slice.

### The model row

```rust
pub struct LoadedModel {
    pub id: String,
    pub state: State,                    // Loaded | Loading | NotLoaded
    pub context_tokens: Option<u32>,     // serving window; None when unloaded
    pub weights_bytes: Option<u64>,
}
```

`context_tokens` is `Option<u32>` because a serving window is only knowable once
a model is resident. This encodes the field note directly into the type system:

> Never budget against a capability figure.

Verified live on this machine, 2026-09-09 — LM Studio returns, for an unloaded
model:

```json
{ "id": "qwen3-14b-...", "state": "not-loaded", "max_context_length": 40960 }
```

`loaded_context_length` is simply absent. `max_context_length` (40,960) and
llama.cpp's `n_ctx_train` must never be read into `context_tokens`; a test
asserts it, because reading the first over-promises room by 5x on qwen3-8b.

### Per-provider sources

All four surfaces exist already and were verified 2026-09-08.

| Provider | Endpoint | State field | Serving window | Weights |
|---|---|---|---|---|
| LM Studio | `GET /api/v0/models` | `state` | `loaded_context_length` | not reported |
| llama.cpp | `GET /v1/models`, router `GET /running` | presence in `/running` | `meta.n_ctx` | not reported |
| vLLM-MLX | `GET /v1/models` | presence | not reported | not reported |
| Ollama | `GET /api/ps` | presence | `context_length` | `size` |

vLLM-MLX reports no window at all. It gets `None`, not a guess.

**Two figures on Ollama's `/api/tags` are traps and must not be read.** Verified
2026-09-09: `details.context_length` there is 2,048 for `nomic-embed-text` — a
*capability* figure attached to an unloaded model, the same class of mistake as
LM Studio's `max_context_length`. Only `/api/ps` reports a serving window, and
only for something resident. `/api/tags` `size` is safe to read, being a
property of the file rather than of a load.

## 5. Discovery

Config at `~/.config/llm-harmony/config.toml`, with a built-in default table so
zero-config works on this machine:

```toml
[[provider]]
kind = "lmstudio"
url  = "http://127.0.0.1:1234"
```

Defaults: `1234` lmstudio, `11434` ollama, `8080` llamacpp, `8000` vllm.

`kind` is declared rather than sniffed. `probe()` then *confirms* it, using each
provider's unique surface — `/api/v0/models`, `/api/ps`, `/props` — never the
shared `/v1/models`, which three of the four answer. A wrong declaration
produces a reported mismatch, not a silent misparse.

A provider that is not running is absent from the report. It is never an error.
On this machine right now, two of four are running, so that path is exercised
from the first run.

## 6. Memory attribution

The primary number is **real footprint against machine capacity** — the sum
`design.md` §1 says is unbounded today. It needs no per-model data and is
therefore available for every running provider without exception.

- **footprint** — real `phys_footprint` summed over the provider's pids, via
  `libproc`. On Apple Silicon this includes Metal buffers in unified memory,
  which is what makes it meaningful here and not merely RSS.
- **weights** — summed artifact size of the provider's loaded models, **only
  where the provider reports it**. Of the four, only Ollama does (`size`, in
  both `/api/ps` and `/api/tags`). LM Studio reports no size on `/api/v0/models`
  or `/v1/models`; verified live 2026-09-09.
- **gap** — `footprint - weights`, shown only where weights are known.
  KV cache, prefix cache, and activations: the term every per-provider budget
  omits, including vLLM-MLX's, which says so in its own startup log.

Resolving a model id to a file on disk — which would give weights for the other
three providers — is the inventory half's dependency graph, not this slice.
Slice 1 prints `?` rather than guessing at a path.

Machine line: `hw.memsize` total, free percentage, and swap used from
`vm.swapusage` — swap being the direct indicator of the failure this project
exists to prevent.

```
provider    loaded   footprint   weights      gap
lmstudio         2        9.8G         ?        ?
ollama           1        0.4G      0.3G    +0.1G
llamacpp         —   not running
vllm             —   not running
──────────────────────────────────────────────────
providers                10.2G
machine     24.0G total · 8% free · swap 1.2G used
```

The `footprint` column and its total are the artifact of this slice: the first
statement anywhere on this machine of what the providers *collectively* cost.
`gap`, where it can be computed, is the quantity slice 2's measured estimator
learns from.

`--json` emits the same data structurally, including per-model rows and any
per-provider errors.

## 7. Failure handling

Fail open, always, per `design.md` §8.

| Condition | Behaviour | Exit |
|---|---|---|
| Provider not listening | `not running` row | 0 |
| Provider answers unparseably | `unreadable: <reason>` row, error in `--json` | 0 |
| `probe` contradicts declared `kind` | `kind mismatch: expected X` row | 0 |
| Provider pids unreadable | footprint shown as `?`, models still listed | 0 |
| Config malformed | message to stderr | 1 |
| `hw.memsize` unreadable | message to stderr | 1 |

No single provider can make `status` fail. The command's job is to report what
it can see, including its own blind spots.

## 8. Testing

The physical tests in `design.md` §10 belong to the slices that admit and evict.
There is nothing to prove physically about a program that only reads. What is
worth testing here is parsing, because that is where the field notes say the
bugs are.

- **Adapter parsing against recorded fixtures.** Real JSON captured from real
  providers into `tests/fixtures/<provider>/`. LM Studio and Ollama can be
  captured now; llama.cpp and vLLM-MLX when they are next started. A fixture
  records the machine and date it came from.
- **The capability-vs-configuration test, named explicitly.** An unloaded LM
  Studio model yields `context_tokens: None` despite `max_context_length:
  40960`. A llama.cpp model yields `meta.n_ctx`, never `n_ctx_train`. This is
  the bug that already happened once by hand; it does not get to happen again.
- **Empty and absent cases.** `{"models":[]}` from Ollama, a refused connection,
  a 404, HTML from something that is not a provider at all.
- **Integration.** `status` against a stub HTTP server serving fixtures,
  asserting table and `--json` output.
- **One smoke test.** Machine total equals `sysctl hw.memsize`.

## 9. Done when

1. `llm-harmony status` prints a correct table on this machine with two
   providers running and two absent.
2. Starting llama.cpp and vLLM-MLX makes them appear with no config change.
3. Loading a model in LM Studio makes `context_tokens` change from `None` to the
   real serving window, and moves `footprint` by roughly the model's size.
4. `--json` round-trips through `jq` and carries per-provider errors.
5. No code path in the crate can unload a model.

## 10. What this slice deliberately does not answer

- What a model *will* cost before it is loaded. That is the estimator, and it
  needs the measured deltas only a daemon can collect.
- Whether there is room. That is admission.
- Whether it is safe to delete anything. That is inventory.

Slice 1 answers exactly one question — *what is loaded right now, and what is it
really costing?* — which nothing on this machine can answer today.
