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

**Status: slice 1 implemented.** `llm-harmony status` reads all four providers
and reports what they collectively cost. Everything else is design.

| Document | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | the two planes, the two ledgers, and how the pieces fit |
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
- **No server lifecycle ownership.** You keep starting servers however you
  like, with whatever flags. The daemon discovers them.
- **Refuses rather than warns.** On both ledgers, the failure modes are
  expensive and asymmetric: over-estimating memory wastes capacity, and
  under-estimating it takes the machine down. Bias accordingly.

## What already exists to build on

Every provider ships both halves an adapter needs. None of it has to be
invented:

| Provider | Observe | Evict |
|---|---|---|
| LM Studio | `GET /api/v0/models` → `state`, `loaded_context_length`; `lms ps` | `lms unload <model>` |
| llama.cpp | `GET /v1/models` → `meta.n_ctx`; router `GET /running` | `POST /api/models/unload/<model>` |
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
lmstudio         1        7.0G         ?        ?
ollama           0       15.8M         ?        ?
llamacpp         — not running
vllm             — not running
──────────────────────────────────────────────────
providers                 7.0G
machine    24.0G total · 9% free · swap 6.0G used
```

`footprint` is real `phys_footprint`, which on Apple Silicon includes Metal
buffers in unified memory. `weights` is shown only where a provider publishes an
artifact size — of the four, only Ollama does — and `gap` is the KV cache and
activations that every per-provider budget omits.

That output is the failure this project exists to prevent, caught live: one 14B
model on a 24 GB machine, three of four providers idle, and swap already at
6 GB.

## Open questions

Settled in the docs, not here:

1. **If the right provider is not running, does harmony start it?** Refusing is
   not seamless; starting it crosses the server-lifecycle line. The sharpest
   unresolved question in the design.
2. **How is a footprint estimated** on unified memory, where weights are the
   floor and the KV cache is the part that actually kills you?
3. **Is `DERIVES_FROM` recorded or inferred?** Inference was acceptable for an
   offline `rm`; it is riskier once placement depends on it.
4. **What is the memory reserve on 24 GB?** 8 GB is a guess that should be
   measured before anyone trusts it.

## Non-goals

- **Not a router, gateway, or load balancer.** It returns coordinates; it never
  carries a request.
- **Does not start or supervise inference servers**, and does not transform
  requests. (It *does* choose which provider serves a model — that is
  placement, and it is a goal.)
- **Not a scheduler for throughput.** It optimises *fitting*, not tokens/sec.
- **Not cross-machine.** One host, one daemon.
