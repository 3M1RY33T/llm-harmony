# llm-harmony

The management centre for local models on one machine: **Hugging Face for
receiving, four providers for serving, and llm-harmony in between.**

It holds sole authority over supply — what arrives on disk, what derives from
what, which provider serves it, and what is resident right now — and has **zero
presence in traffic**. Prompts never pass through it. It hands a client the
coordinates of a ready model and gets out of the way.

Two resources, one set of machinery:

| | Memory ledger | Disk ledger |
|---|---|---|
| Question | is there room to **load** this? | is there room to **store** this? |
| Eviction | unload a model | clear a cache, remove an artifact |
| Cost of being wrong | the machine swaps, then wedges | an unrecoverable deletion |

Admission, eviction and refusal work the same way on both. See
[`docs/architecture.md`](docs/architecture.md) for why that is the load-bearing
claim of the project.

> **On the name.** Unrelated to *Harmony*, OpenAI's response format for
> gpt-oss models — the one whose `<|channel|>analysis<|message|>` markers appear
> in [`docs/field-notes.md`](docs/field-notes.md) as an example of a provider
> leaking its raw token stream. Same word, different thing: that Harmony is a
> wire format a model emits; this one decides what fits.

**Status: slices 1–5 implemented**, 281 tests. `status` reads all four
providers, `ls` and `rm --dry-run` cover 146 GB across five stores, `estimate`
predicts a model's footprint, `resolve` answers for a *model* rather than a
provider, and `start` brings a provider up under launchd. `clean --redundant`
asks `rm`'s question of every model at once — 19 GB of superseded builds here
— and `ls --duplicates` looks for the same bytes stored twice, of which this
machine has none.

Slice 5 made it act: `load`, `unload` and `switch` change what is resident,
admitted against the machine and supervised while they happen, with `pin` to
take a model off the table. Two of the four providers turned out to have no
model-level unload at all — `llm-harmony verify` prints what each one can
really do, probed rather than remembered. Intake and conversion are next.

| Document | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | the two planes, the two ledgers, and how the pieces fit |
| [`docs/roadmap.md`](docs/roadmap.md) | the arc from here to the goal, slice by slice |
| [`docs/design.md`](docs/design.md) | memory admission at runtime — the ledger, the protocol, eviction |
| [`docs/inventory.md`](docs/inventory.md) | safe add/remove of models, the dependency graph, the HF pipeline |
| [`docs/field-notes.md`](docs/field-notes.md) | provider behaviour verified by hand, with the traps that cost time |
| [`docs/prior-art.md`](docs/prior-art.md) | what exists and why none of it fits |

---

## The problem

Four local providers, each with a correct memory policy, and no way for any of
them to know the others exist:

| Provider | Its own budget | What it knows about the rest |
|---|---|---|
| LM Studio | JIT load, idle TTL | nothing |
| llama.cpp (router mode) | `--models-max`, LRU eviction | nothing |
| vLLM-MLX | `memory_budget_gb` + eviction | nothing |
| Ollama | `OLLAMA_MAX_LOADED_MODELS`, `keep_alive` | nothing |

Each stays under its own ceiling. Their **sum** does not stay under the
machine's. On a 24 GB Apple Silicon box (`hw.memsize` = 25,769,803,776) there is
no VRAM/RAM split to absorb the mistake: an overcommit swaps, then wedges, then
takes the machine down.

The same machine holds **148 GB of models across four stores** with tangled
symlinks between them, no record of what derives from what, and no way to answer
*"is it safe to delete this?"* without checking five things by hand.

Both halves are the same shape: a resource with no global accounting.

## The idea in one paragraph

A daemon keeps **two ledgers** — bytes resident and bytes stored — learned by
polling each provider's existing status surface and scanning each store. It
acts through each provider's existing unload command and each store's existing
layout. Clients ask for a **model**, not a provider: llm-harmony picks who can
serve it, frees room if needed, ensures it is loaded, and returns the endpoint
to talk to. Models arrive the same way, through one intake path that verifies
provenance and checks fit before a 28 GB download rather than after.

## The two planes

The distinction the whole project rests on:

- **Management plane — llm-harmony is the middle-man.** Receiving from HF,
  placing into stores, converting, registering, evicting, clearing. Sole
  authority, deliberately.
- **Request plane — llm-harmony is absent.** It returns a `base_url` and a
  provider-local model id. The client connects directly. No prompt, completion,
  or token ever passes through the daemon.

Returning coordinates is not routing. A router carries traffic and owns the
latency and the blast radius; this hands over an address and stops. That is why
[`docs/prior-art.md`](docs/prior-art.md)'s objection to llama-swap and LiteLLM
does not fold back onto this project — but the line is thin enough that it is
argued properly in [`docs/architecture.md`](docs/architecture.md) rather than
asserted here.

## Why this shape

- **Out of the request path.** Inference latency is untouched, and a daemon
  crash degrades serving to today's behaviour rather than taking every provider
  with it.
- **Your launch commands, not ours.** Harmony can start a provider, but only
  by running a command you declared — every flag stays in your own script. A
  provider you start by hand is discovered exactly as before.
- **Refuses rather than warns.** On both ledgers, the failure modes are
  expensive and asymmetric: over-estimating memory wastes capacity, and
  under-estimating it takes the machine down. Bias accordingly.

## What already exists to build on

Every provider ships both halves an adapter needs. None of it has to be
invented:

| Provider | Observe | Evict |
|---|---|---|
| LM Studio | `GET /api/v0/models` → `state`, `loaded_context_length`; `lms ps` | `lms unload <model>` |
| llama.cpp | `GET /v1/models` → `status.value` (**not** `meta.n_ctx`, and `/running` 404s — corrected 2026-09-09 against a live router) | `POST /api/models/unload/<model>` |
| vLLM-MLX | `GET /v1/models`; registry budget | registry eviction; process signal |
| Ollama | `ollama ps`, `GET /api/ps` | `ollama stop <model>`, `keep_alive: 0` |

Verified on this machine, 2026-09-08, and exercised by slice 1's adapters.

## What works today

Slice 1 is the read-only core: four adapters, the memory ledger, and one
command. It cannot unload a model — not by policy, but because no such code
path exists in the crate.

```
$ llm-harmony status
provider    loaded   footprint   weights      gap
lmstudio         1        8.9G         ?        ?
ollama           0       17.4M         ?        ?
llamacpp         0       36.1M         ?        ?
vllm             — not running
──────────────────────────────────────────────────
providers                 9.0G
machine    24.0G total · 12% free · swap 7.2G used```

`footprint` is per-process `max(phys_footprint, rss)`. Neither metric alone is
enough: `phys_footprint` cannot see a memory-mapped GGUF, and RSS cannot see
Metal buffers in unified memory. `weights` appears only where a provider
publishes an artifact size — of the four, only Ollama does — and `gap` is the KV
cache and activations every per-provider budget omits.

That output is the failure this project exists to prevent, caught live: one 14B
model on a 24 GB machine and swap already in the gigabytes.

## Open questions

Settled in the docs, not here:

1. ~~**If the right provider is not running, does harmony start it?**~~
   **Answered 2026-09-09: yes, opt-in, supervised by launchd.**
2. ~~**How is a footprint estimated?**~~ **Answered by slice 3:** measured from
   the largest process in a provider's tree, and *unknown* rather than guessed
   for a shape never seen.
3. **Is `DERIVES_FROM` recorded or inferred?** Still open, and now load-bearing:
   placement would depend on it, and slice 2 forbids inferred lineage from
   driving decisions.
4. **What is the memory reserve on 24 GB?** 8 GB remains a guess. A dozen
   observations is not a basis for tuning it.

## Non-goals

- **Not a router, gateway, or load balancer.** It returns coordinates; it never
  carries a request.
- **Starts and supervises inference servers only on request, per provider,
  opt-in.** Changed 2026-09-09; it was previously a non-goal. Harmony runs a
  `start` command *you* declare and installs it as a launchd agent, so macOS
  supervises it and a dead harmony never stops a running server. It still never
  learns a provider's flags — those stay in your scripts.
- **Does not transform requests.** (It *does* choose which provider serves a
  model — that is placement, and it is a goal.)
- **Not a scheduler for throughput.** It optimises *fitting*, not tokens/sec.
- **Not cross-machine.** One host, one daemon.
