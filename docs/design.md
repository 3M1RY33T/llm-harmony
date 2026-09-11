# llm-harmony — design

Design sketch, 2026-09-08. Nothing here is built. The purpose of this document
is to be disagreed with before code exists.

---

## 1. What went wrong, precisely

The failure is not that any provider misbehaves. Each one is correct:

- vLLM-MLX honours `memory_budget_gb: 14` and says so at startup — *"the
  registry budget covers model weights only; the KV cache, the prefix cache and
  activations are additional and are not reserved by it."*
- llama.cpp's router evicts LRU past `--models-max`.
- LM Studio JIT-loads and unloads on an idle TTL.
- Ollama honours `OLLAMA_MAX_LOADED_MODELS` and `keep_alive`.

Four schedulers, four private ledgers, one physical memory. The sum is
unbounded. Two 9B models at 4-bit are ~5 GB each in weights alone; add KV cache
at a 32k context and the real figure is meaningfully higher. Three of those
across two providers fits every individual budget and none of the machine's.

Unified memory removes the usual safety margin. There is no VRAM ceiling that
fails a `cudaMalloc` and leaves the OS alive — the allocation succeeds, the
machine swaps, and everything stops.

**Design consequence:** the daemon's job is arithmetic across providers, not
better policy inside one.

## 2. Shape

```
   ┌──────────┐        ask / release           ┌───────────────┐
   │  Delroy  │ ─────────────────────────────► │               │
   └──────────┘                                │  llm-harmony  │
   ┌──────────┐                                │    (daemon)   │
   │ CLI, IDE │ ─────────────────────────────► │               │
   └──────────┘                                └───────┬───────┘
                                            poll state │ unload
                                                       ▼
              ┌─────────────┬──────────────┬───────────────┬──────────┐
              │  LM Studio  │  llama.cpp   │   vLLM-MLX    │  Ollama  │
              └─────────────┴──────────────┴───────────────┴──────────┘
                        (started by you, however you like)
```

The daemon talks to providers on their **control** surfaces only. Prompts and
completions never pass through it. If the daemon is down, every provider still
works exactly as it does today — you simply lose the arithmetic.

## 3. The ledger

One table, rebuilt by polling and amended by admissions.

| field | meaning | source |
|---|---|---|
| `provider` | lmstudio \| llamacpp \| vllm \| ollama | adapter |
| `model` | provider-local id | adapter |
| `state` | loaded \| loading \| idle | adapter |
| `weights_bytes` | on-disk size of the artefact | filesystem / catalog |
| `context_tokens` | what it was actually loaded with | adapter |
| `estimated_bytes` | admission figure (§5) | computed |
| `last_used` | for LRU eviction | adapter or daemon |
| `sanctioned` | did it come through an admission? | daemon |

Rebuilt on a poll cycle (~2 s idle, faster while an admission is pending). The
ledger is **derived state, never authoritative** — a provider that unloaded a
model behind the daemon's back must not leave a phantom reservation. This is why
polling exists at all rather than tracking only what the daemon admitted.

## 4. Admission protocol

Deliberately tiny. Three verbs over a unix socket:

```
ASK    provider model [context_tokens]   → GRANT | WAIT reason eta | DENY reason
COMMIT provider model                    → ack     (client loaded it)
RELEASE provider model                   → ack     (client is done; hint, not a promise)
```

A client that never speaks is not an error. The daemon discovers its model on
the next poll and marks it `sanctioned: false`. Asking is how you get a *useful*
answer — the daemon can make room *before* you load — rather than an accounting
correction afterwards.

`WAIT` carries a reason and an eta so a caller can decide for itself: Delroy
would surface it as a notice rather than silently blocking a turn, in the same
spirit as the step-budget notice it already appends to a conversation and the
`model_fallback` event it already emits to the operator.

(Corrected 2026-09-09: an earlier draft cited a "tool-trim notice" here. No
such thing exists in Delroy — the two real precedents are the ones named
above, and they go to different audiences, which turns out to matter.)

## 5. Estimating a footprint — the hard part

Weights are the floor, not the answer. From measurements on this machine:

- Q4_K_M 9B ≈ 5.2–5.4 GB of weights.
- KV cache scales with context × layers × kv-heads × head-dim × 2 × dtype-bytes.
  At 32k it is not a rounding error.
- vLLM-MLX's own log states the budget excludes KV cache, prefix cache and
  activations — the same trap, acknowledged upstream.
- llama.cpp reports `meta.n_ctx` (the window it is actually serving) alongside
  `n_ctx_train`; only the first predicts memory.

Proposal, in order of preference:

1. **Measured.** Record real RSS delta across a load, keyed by
   `(provider, model, context_tokens)`. After one load of a given shape, the
   daemon stops guessing. This is the only source that can be right.
2. **Computed.** `weights + kv(context, arch) × safety`, from config metadata,
   for shapes never seen.
3. **Declared.** A per-model override in config, for the cases both above get
   wrong.

Plus a **headroom floor**: never admit past `total − reserve`, where reserve
covers the OS and everything not an LLM. On 24 GB, a reserve of 8 GB is a
starting guess to be tuned by measurement, not a claim.

The honest statement of the rule, borrowed from `vramd`'s framing: *admit by the
real peak, not by the weights.* Every per-provider budget today gets this wrong
by omission.

## 6. Eviction

Only with permission, and only ever a **provider's own** unload command — never
a signal to a process the daemon does not own.

Policy, in order: prefer `sanctioned: false` models (nobody claimed them), then
LRU, then largest-fit. Never evict a model with an in-flight request; the
adapters can see busy state (`ollama ps`, llama.cpp `/running`, LM Studio's
state field).

Per-provider modes, so trust is granted rather than assumed:

- `observe` — account for it, never touch it. **Default.**
- `evict` — may unload this provider's models to make room.
- `manage` — may also pre-emptively unload on idle.
- `start` — may bring this provider up. Orthogonal to the three above rather
  than a rung on the same ladder: creating a server and destroying a resident
  model are different authorities, and wanting one is no reason to grant the
  other. Granted by declaring a `start` command in config, and used only via
  an explicit `llm-harmony install` (added 2026-09-09).

### The per-provider budgets are not superseded

It would be easy to read this design as replacing `OLLAMA_MAX_LOADED_MODELS`,
`--models-max`, LM Studio's idle TTL and `memory_budget_gb`. It does not. They
become **more** load-bearing, for two reasons:

1. **They are the mechanism.** The daemon owns no processes and holds no
   allocator. Every eviction it performs is one of those controls being invoked
   on its behalf. A provider configured with no ceiling and no unload path is a
   provider the daemon can account for but never act on.
2. **They are the fail-open floor.** §8 requires that a dead daemon degrade to
   today's behaviour rather than taking the providers with it — which means
   whenever the daemon is down, those per-provider budgets are the *only* thing
   standing between four servers and the machine. A correctly tuned local
   ceiling is what makes fail-open survivable rather than merely honest.

So the daemon's job is the arithmetic *between* those budgets, not instead of
them. Tuning each provider stays real work, and a setup with them switched off
is strictly worse than one with them on and no daemon at all.

## 7. Adapters

One per provider, each implementing `list()`, `unload(model)`, `busy(model)`.
All four surfaces exist already (verified 2026-09-08):

```
LM Studio   GET /api/v0/models  → state, loaded_context_length   |  lms unload
llama.cpp   GET /v1/models      → status.value ; GET /props      |  none (max_instances)
vLLM-MLX    GET /v1/status      → loaded, memory_gb, source      |  none (memory_budget_gb)
Ollama      ollama ps / GET /api/ps                              |  ollama stop, keep_alive: 0
```

**Corrected 2026-09-10.** The right-hand column originally claimed
`POST /api/models/unload/<model>` for llama.cpp and "registry eviction" for
vLLM-MLX. Probed with all four servers up, neither exists — see
[`field-notes.md`](field-notes.md), *two evict paths in this document were
never real*. **Eviction granularity is asymmetric:** two providers expose a
model-level verb, two expose only a process and a ceiling. That is not a gap
to close but a fact to design around, and §6 above already stated the
consequence before it was measured.

Discovery: scan a configured port list, identify by response shape. A provider
that is not running is simply absent from the ledger — never an error.

## 8. Failure modes worth designing for

| Failure | Response |
|---|---|
| Daemon dies | Providers unaffected. Clients that get no answer proceed as today. **Fail open, always.** |
| Provider dies mid-admission | Next poll drops its rows; reservation released. |
| Model loaded without asking | Seen on next poll, `sanctioned: false`, first eviction candidate. |
| Two clients ask at once | Serialised at the socket; second gets `WAIT`. |
| Estimate too low | The case that crashes the machine. Mitigated by the headroom floor and by preferring measured over computed. |
| Estimate too high | Wasted capacity — annoying, safe. **Bias here.** |

## 9. Open questions

1. **Advisory or enforcing by default?** `observe` everywhere is safest and
   also useless on day one — it can warn but never prevent. Suggestion: ship
   `observe` as the default and make `evict` a one-line opt-in per provider.
2. **Does Delroy block on `WAIT`?** Blocking a turn on a daemon is a new failure
   mode in a tool that currently has none. A notice plus proceeding may be the
   better default, accepting that the guarantee becomes advisory.
3. **What is the reserve on 24 GB?** 8 GB is a guess. It should be measured
   under real load before it is written down as a number anyone trusts.
4. **Is the ledger worth persisting?** Measured footprints are the expensive
   knowledge; losing them on restart means re-learning by loading. Probably yes,
   in the same spirit as Delroy's `model-windows.json`.

## 10. What would prove this works

Not unit tests — the interesting behaviour is physical:

- Load models across three providers past the machine's capacity **with** the
  daemon and show it refuses or evicts rather than swapping.
- The same sequence with the daemon stopped, to show the crash it prevents.
- Measured footprint converging on real RSS after one load of each shape.
- Inference latency identical with and without the daemon running, proving it
  is out of the request path.
