# Field notes

Facts verified on one machine (M4 Pro, 24 GB, macOS 26.5.1) between 2026-09-07
and 2026-09-09, while setting up four local providers by hand. Everything here
was observed, not read. Each entry exists because getting it wrong cost time.

---

## Provider control surfaces

All four expose both halves an adapter needs. None had to be invented.

| Provider | Observe | Evict |
|---|---|---|
| LM Studio | `GET /api/v0/models` → `state`, `loaded_context_length`, `type`, `capabilities`; `lms ps` | `lms unload` |
| llama.cpp | `GET /v1/models` → `meta.n_ctx`, `meta.n_ctx_train`; router `GET /running` | **none** — see below |
| vLLM-MLX | `GET /v1/models`; registry `memory_budget_gb` | **none** — see below |
| Ollama | `ollama ps`, `GET /api/ps` | `ollama stop`, `keep_alive: 0` |

**Corrected 2026-09-10.** Two entries in the Evict column were written from
documentation and were never real. The originals are struck above and argued
immediately below.

## Context windows: capability vs configuration

The distinction that caused the first real bug of the session.

- **LM Studio** reports both `max_context_length` (what the model supports) and
  `loaded_context_length` (what it was actually loaded with). qwen3-8b: 40,960
  vs **8,192**. Reading the first over-promises room by 5×.
- **llama.cpp** reports `meta.n_ctx` (serving) and `meta.n_ctx_train` (trained).
  Only the first bounds a request.
- **vLLM-MLX** reports neither. A consumer must fall back to a conservative
  floor rather than guess.
- **Unloaded models report no serving window at all** — LM Studio omits
  `loaded_context_length`, llama.cpp's router returns `n_ctx: null`. A window is
  only knowable once something is resident.

**Rule:** never budget against a capability figure. If configuration is
unavailable, use a floor.

## Failure reported inside a 200

Three providers, three different ways of saying "this went wrong" without an
HTTP error. All were initially read as "the model had nothing to say".

| Host | Shape |
|---|---|
| LM Studio | HTTP 200, then an SSE frame `event: error` with `{"error":{"message":…}}` and no `choices` |
| Ollama Cloud | HTTP 200 with the model's raw unparsed token stream in `content` (Harmony channel markers) |
| vLLM-MLX | HTTP 200, `finish_reason: "error"`, `content: null`, usage all zeros |

`finish_reason: "error"` is not in the OpenAI schema — nothing legitimate sends
it.

## Advertised is not resident

vLLM-MLX serves two lists and they mean different things. Found 2026-09-10, the
first time the server was actually started.

| Endpoint | What it lists |
|---|---|
| `GET /v1/models` | every model the registry **can** serve |
| `GET /v1/status` | what is **resident**, via an explicit `loaded` boolean |

Reading the first as residency claimed five 9B models were loaded while the
server held 331 MB and none were. On a `memory_budget_gb: 14` pool that is not
merely wrong, it is arithmetically impossible — and it would have told a memory
ledger that ~25 GB was committed.

`/v1/status` carries more than residency, and vLLM-MLX is the only one of the
four that publishes any of it:

- **`memory_gb` per model** — the manager's own footprint figure (4.7 GB for a
  9B at 4-bit, 6.8 GB at 6-bit). Its startup log already warns this covers
  weights only, so it is a floor, not a peak.
- **`source`** — the artifact path on disk, which is the model-to-artifact link
  nothing else hands over for free. llama.cpp's `status.args` carries the same
  thing as `--model <path>`.
- **`memory_budget_gb`** — the provider's own ceiling, declared.

## An adapter written from documentation is a hypothesis

Both llama.cpp and vLLM-MLX had adapters written against documented shapes,
with fixtures marked UNVERIFIED because the servers were not running. Both were
wrong, and neither error was visible until the provider actually started:

| Provider | What the adapter assumed | What the server does |
|---|---|---|
| llama.cpp | a `meta` block, `meta.n_ctx`, a `/running` endpoint | none of the three exist; state is `status.value`, `/running` 404s, no window is published at all |
| vLLM-MLX | presence in `/v1/models` means resident | presence means *advertised*; `/v1/status` has the truth |

The llama.cpp one is the more instructive failure: `status` reported the
provider as **not running** while it was answering fine, and did so for the
whole life of three slices. The fixture caveat was written down at the time,
which is the only reason it was ever reconciled.

**Rule:** an adapter is not verified by its tests. It is verified by the server.

### Two evict paths in this document were never real

Found 2026-09-10, the first time anything tried to *use* the Evict column
rather than write it down. All four providers were running.

| Probe | Result |
|---|---|
| `POST /api/models/unload/Qwen3-14B-Claude-4.5-Opus-Distill-Q4_K_M` (llama.cpp) | **404** |
| the same path by `GET` | **404** |
| vLLM-MLX `/openapi.json`, every route | no load, no unload, no evict |

The llama.cpp probe used a **real model id**, so 404 means the route is absent
rather than the model being unknown. vLLM-MLX's own OpenAPI document lists 23
routes; the only `DELETE`s are `/v1/cache` and `/v1/cache/prefix`, which are
the prefix cache, not residency.

What they expose instead is a ceiling and an autoloader. llama.cpp's router
publishes `{"role":"router","max_instances":1,"models_autoload":true}` at
`/props`: it loads a model on first request and evicts past its own limit.
vLLM-MLX bounds itself with `memory_budget_gb` and evicts internally.

**Consequence for harmony:** eviction granularity is asymmetric. Two providers
expose a model-level verb; two expose only a process and a ceiling. Harmony can
bound the latter pair at install time and account for them after, and can never
unload one model from either. `design.md` §6 anticipated exactly this — *"a
provider configured with no ceiling and no unload path is a provider the daemon
can account for but never act on"* — so the design survived; the table did not.

**Rule, third restatement:** an evict path is verified by a 200 from the live
server, not by its presence in a README. `llm-harmony verify` prints the probed
answer so this table never has to be trusted from memory again.

### A measurement taken while swapping is not a measurement

Found 2026-09-11, calibrating the computed estimator against the corpus. Every
`loaded` observation this machine has recorded:

| provider | ctx | footprint | phys | rss | swap in use |
|---|---|---|---|---|---|
| lmstudio | 40,960 | 9.61 G | 7.40 G | 9.31 G | **5.0 G** |
| lmstudio | 40,960 | 9.61 G | 7.40 G | 9.31 G | **5.0 G** |
| lmstudio | 40,960 | 9.60 G | 7.40 G | 9.10 G | **7.7 G** |
| lmstudio | 40,960 | 9.64 G | 7.43 G | 9.22 G | **8.8 G** |

Four samples, one shape, and **not one of them taken on a calm machine.**

Read them against the artifact's own geometry — 40 layers, 8 kv heads, head_dim
128, from its header — and the KV cache at 40,960 tokens should be **6.71 GB**
on top of 9.00 GB of weights. The observations report a total of 9.6 GB. That
is a cache of approximately zero at a 40k window, which is not physically
possible.

The most likely reading is that the figure is depressed rather than the formula
inflated: a resident-set reading taken while the machine is paging records what
survived eviction, not what the model asked for.

**Consequence, and it inverts a ranking.** `design.md` §5 puts measured above
computed, which is right in general and wrong for a measurement taken under
pressure: a systematically depressed number would outrank a bias-high one, for
the single failure that costs the machine. So an observation with more than
1 GiB of swap in use is no longer a measurement — `estimator::is_trustworthy`
drops it, and the shape falls through to the computed rung.

On this machine that disqualifies the entire corpus. That is the honest state:
**there is currently no trustworthy measurement of any model here**, and the
`--record` path will only produce one on a calm machine.

**The margin was deliberately not tuned.** Computed comes in at 16.97 GB
against the 9.00 GB "measurement", 1.88×. Closing that gap by shaving the
margin would have been fitting the formula to a number already known to be
wrong — which is precisely how under-prediction gets shipped. The one knob
added instead is `kv_dtype_bytes` per provider, for anyone actually running a
quantised cache.

### A GGUF metadata block is megabytes, and the tokenizer is nearly all of it

Measured 2026-09-11 while building the shape reader, on
`Qwen3-14B-Claude-4.5-Opus-Distill-Q8_0.gguf`:

| | |
|---|---|
| metadata block ends at | **5,934,129 bytes** (5.9 MB) |
| `tokenizer.ggml.tokens` | 2,588,293 bytes |
| `tokenizer.ggml.merges` | 2,731,593 bytes |
| `tokenizer.ggml.token_type` | 607,793 bytes |
| everything else, 33 pairs | ~6 KB |

The six hyperparameters an estimator needs are in that last 6 KB, and they are
written **before** the tokenizer. So the cheap read is not "a bounded prefix"
but "stop at the first `tokenizer.` key" — with an escalating read behind it
for any writer that orders them differently.

The first version of the reader took a flat 1 MiB prefix. It passed every
synthetic fixture and returned `None` for **every real model on this machine**,
because 1 MiB lands in the middle of the token list. Same shape as the adapter
failures above, in a new place: *the fixture was the hypothesis.*

**Also:** `DirEntry::metadata()` does not follow symlinks, and every weight file
in the Hugging Face cache is a symlink into `blobs/`. Sizing an MLX snapshot
that way reported a 4.6 GB model as 0 bytes. `std::fs::metadata(entry.path())`
is the one that follows.

Verified against four real artifacts once fixed — geometry cross-checks against
the numbers already in this document:

```
lmstudio  Q4_K_M 14B   40 layers  8 kv heads  head_dim 128   9.00 GB  (= 8.38 GiB)
llamacpp  Q8_0   14B   40 layers  8 kv heads  head_dim 128  15.70 GB  (= "the 14B Q8")
mlx       4bit    8B   36 layers  8 kv heads  head_dim 128   4.61 GB
hf-cache  bf16   14B   40 layers  8 kv heads  head_dim 128  29.54 GB
```

### LM Studio's own estimator is weights-only

`lms load --estimate-only` looks like a free, authoritative footprint. It is
not — it is the `Declared` basis wearing a vendor badge. Same model, two
windows, 2026-09-10:

```
-c 8192   → Estimated Total Memory: 8.38 GiB   Confidence: LOW
-c 40960  → Estimated Total Memory: 8.38 GiB   Confidence: LOW
```

Flat across a 5× context change, and 8.38 GiB is exactly the GGUF's size on
disk. The corpus measured that same shape resident at **9.60 GB**. LM Studio
says `Confidence: LOW` itself, which is honest; the trap is that a consumer
reads the number and not the label.

**Rule:** a vendor's own estimate is still a declared figure. If it does not
move when the context window moves, it does not model the KV cache, and the KV
cache is the part that wedges the machine.

## Control surfaces lie too

Both found 2026-09-09 while building the slice-1 adapters. Both are the same
shape as traps already documented above, but on the *control* plane rather than
the completion plane -- which is where a memory daemon actually reads.

- **LM Studio answers every unknown path with HTTP 200.** `GET /api/ps`,
  `GET /props`, anything at all returns
  `{"error":"Unexpected endpoint or method. (GET /path)"}` with a 200 status.
  Ollama, by contrast, returns honest 404s. A fourth instance of *failure
  reported inside a 200* -- and it means provider identification cannot trust a
  status code. An adapter must validate the *shape* of what came back.
- **Ollama's `/api/tags` carries a capability window.**
  `details.context_length` is 2,048 for `nomic-embed-text` while the model is
  not loaded. It sits one field away from where a consumer would reach for a
  serving window, and it is exactly the `max_context_length` trap wearing a
  different name. Only `/api/ps` reports a serving window, and only for
  something resident. `size` on `/api/tags` is safe -- it is a property of the
  file, not of a load.

**Rule, extended:** never budget against a capability figure, and never identify
a provider by a status code.

## Neither macOS memory metric is sufficient alone

Measured 2026-09-09 while building the estimator. The two obvious metrics are
blind to different things, and one of the blind spots is dangerous.

| | LM Studio tree, 14B Q4_K_M loaded at 40,960 ctx |
|---|---|
| GGUF on disk | 8.38 GB |
| `llama-server` (the backend child) RSS alone | **8.39 GB** |
| Whole-tree RSS | 8.67 GB |
| Whole-tree `phys_footprint` | **6.89 GB** |

`llama.cpp` memory-maps the GGUF, and `phys_footprint` counts anonymous, dirty
and compressed pages while largely excluding clean file-backed ones. So the
weights are resident, RSS sees them, and `phys_footprint` does not. The provider
reported as costing **less than the model's own weights**, before counting a
40k-token KV cache.

This is the exact inverse of the Electron finding below, and both are real:

| Process kind | RSS | `phys_footprint` | Which is right |
|---|---|---|---|
| `llama-server` with an mmapped GGUF | 8.39 GB | ~6.3 GB | RSS |
| LM Studio main (Metal buffers) | 0.10 GB | 0.29 GB | `phys_footprint` |

**Rule:** take `max(phys_footprint, rss)` **per process**, then sum. Per tree
would lose one of the two, because a provider mixes an Electron shell whose cost
is anonymous with a backend whose cost is mmapped. On this machine that changes
the reported figure from 6.89 GB to 8.95 GB — against 8.38 GB of weights plus
overhead, which is the first number that is not obviously too small.

Crude, and deliberately biased: over-estimating wastes capacity, while
under-estimating is what wedges the machine.

## An idle provider is not free

LM Studio with **zero models loaded** occupies 592 MB across five processes
(main, GPU helper, renderer, a plain helper, and a `node`). Ollama idles at
16 MB in one. Measured via `phys_footprint`, 2026-09-09.

Two consequences for anything doing memory arithmetic:

- A provider's footprint has a floor that has nothing to do with models. An
  estimator that attributes all of a provider's memory to its resident models
  will over-attribute by that floor.
- `sysinfo`'s per-process memory is **not** `phys_footprint`. On LM Studio's
  main process the two read 104 MB and 285 MB -- a 2.7x undercount, and it is
  worst on precisely the GPU helper processes holding Metal buffers. On unified
  memory, `proc_pid_rusage` -> `ri_phys_footprint` is the figure that counts.

## Tool calling

- **llama.cpp `--jinja` is what makes tool calls work.** With it, a 1.5B model
  produced a correct call with correct arguments first try. LM Studio, given the
  same GGUF family without an equivalent, returned the tool-call *syntax as
  prose*.
- **A model can write a tool call instead of making one.** Observed: a DeepSeek-R1
  distill printed an **ASCII imitation** of its own tokens — `<|tool_calls_begin|>`
  (U+007C, U+005F) — inside a markdown fence. Its tokenizer's real tokens are
  fullwidth: `<｜tool▁calls▁begin｜>` (U+FF5C, U+2581). The host was right to
  report zero tool calls; the model simply never emitted them.
- **A host's advertised capability list describes the HOST, not the model.**
  LM Studio listed no `tool_use` for that model, and the model was not incapable
  — it just printed the format. Capability silence is not evidence of incapacity.

## Conversion

- **Pin the converter to the serving binary.** `convert_hf_to_gguf.py` from
  llama.cpp *master* against a `b10240` binary produced unloadable GGUFs:
  `block_count` 33 with tensors only to `blk.31`, and behind that a second
  33-length array (`attention.recurrent_layers`). Checking out tag `b10240` —
  matching `llama-server --version` — fixed both, and the second array stopped
  being emitted at all.
- **`mlx_lm.convert` needs the whole snapshot.** Its final `save()` calls
  `snapshot_download(local_files_only=True)`, which demands *every* file
  including `README.md` and a `.png`. Weight-only downloaders leave the cache
  incomplete and it dies at the last step, after doing the work.
- **Conversion is silently lossy.** MLX conversion dropped 15 MTP tensors **and**
  the vision tower, at 4-bit and 6-bit alike. Only the log line
  `Checkpoint has no weights for module(s): vision_tower` hinted at the second.
  GGUF conversion kept the MTP tensors.
- **Sources can be internally inconsistent.** One repo: 760 tensors, zero
  `mtp.*`, config still declaring an MTP layer. Every GGUF from it needs
  `block_count 32` and `nextn_predict_layers 0` before it loads. Detectable as
  *metadata promises a tensor group the weights do not contain.*
- **Quantization is one-way.** Q4 → Q8 is impossible. Upgrading bit depth means
  re-converting from bf16.
- **There is no MLX ↔ GGUF path.** Both descend from the same source; neither
  derives from the other.

## Provenance

Two HF repos looked like builds of a wanted model and were not:

- `pipenetwork/…-Uncensored-Heretic-MLX-4bit` declares
  `base_model: nightmedia/Qwen3.5-9B-DS9-USS-Defiant` — a different model with a
  near-identical name.
- `RuihanRZhao/…-ultra-uncensored-heretic-8bit` declares **no** `base_model`,
  has an empty README, and its config says **4** bits while the repo is named
  `-8bit`.

**Rule:** an artifact claiming to be a build of X must declare X.

## Runtime behaviour

- **`--continuous-batching` is mandatory for vllm-mlx**, not optional. Without
  it, generation runs on a FastAPI threadpool thread and MLX dies on the first
  request: `RuntimeError: There is no Stream(gpu, 1) in current thread`.
- **Prefix caching is worth ~55× on TTFT** with a long shared system prompt
  (2,775 ms → 49 ms), and nothing on generation speed — it only touches prefill.
- **Prefix cache hits are not deterministic.** At temperature 0, cache off gave
  byte-identical output every run; cache on matched only the first (cold)
  response, and later hits diverged — 222/149/152 chars, 0.48 similarity. Keep
  it off when reproducibility matters.
- **Sampling defaults are the variance.** llama-server defaults to `seed: -1`
  (fresh random seed per request), `top_k 40`, `top_p 0.95`, temp ~0.8. A client
  that sends no sampling parameters inherits all of it.
- **llama.cpp's router scans the HF cache**, so cache entries are live serving
  dependencies, not merely downloads.
- **vLLM-MLX registry `path:` is not tilde-expanded.** A leading `~` is read as
  part of a Hugging Face repo id and the server tries to download it.

## Measured performance

Same model, same 4-bit class, same harness, 5,858-char shared prefix:

| | cold TTFT | warm TTFT | gen tok/s |
|---|---|---|---|
| llama.cpp (GGUF Q4_K_M) | 787 ms | ~11 ms | ~144 |
| vLLM-MLX (MLX 4-bit) | 1,991 ms | ~32 ms | ~89 |

Bit depth, llama.cpp, 8k context:

| | TTFT | tok/s |
|---|---|---|
| 9B Q4_K_M | 92 ms | 38.4 |
| 9B Q6_K | 111 ms | 31.7 |
| 14B Q4_K_M | 120 ms | 26.4 |
| 14B Q8_0 | 197 ms | 16.6 |

**With the 14B Q8 (15 GB) resident: memory free 8%, 30,187 pageouts** — already
touching swap at an 8k context, on a 24 GB machine. This is the case llm-harmony
exists to catch.

Caveat on the first table: it compares *servers*, not formats. LM Studio also
runs an MLX backend and was never measured. Published comparisons claim MLX is
20–40% faster than GGUF, which is the opposite of what was observed here; the
untested configuration is the likely explanation.
