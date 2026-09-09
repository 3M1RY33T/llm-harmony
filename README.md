# llm-harmony

A memory daemon for several local inference servers running at once.

It is **not** a proxy, a gateway, or a router. It never sees a prompt, never
sits in the request path, and adds no latency to inference. It answers one
question — *is there room for this model right now?* — and, when there is not,
frees some by unloading something else.

It has a second job on the same graph: knowing what is **on disk**, what
derives from what, and what breaks if a file is deleted — because "is it safe to
remove this model?" turned out to need the same dependency map as "is there room
to load it?".

> **On the name.** Unrelated to *Harmony*, OpenAI's response format for
> gpt-oss models — the one whose `<|channel|>analysis<|message|>` markers appear
> in [`docs/field-notes.md`](docs/field-notes.md) as an example of a provider
> leaking its raw token stream. Same word, different thing: that Harmony is a
> wire format a model emits; this one is a daemon that decides what fits in
> memory. Where the docs mean OpenAI's, they capitalise it and say so.

**Status: design sketch.** No implementation yet.

| Document | What it covers |
|---|---|
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
takes the machine down — which is the failure this exists to prevent.

Nothing that exists solves this shape (see [`docs/prior-art.md`](docs/prior-art.md)):
every tool found either routes requests through itself or launches the servers
itself. llm-harmony does neither.

## The idea in one paragraph

A small daemon keeps a **ledger** of what is loaded where and what it costs. It
learns the truth by polling each provider's existing status surface, and acts
through each provider's existing unload command. Clients — Delroy, a shell
alias, an editor plugin — **ask before loading**, and the daemon either says yes
or frees room and then says yes. A client that never asks is not broken: the
daemon still sees the model appear on its next poll and accounts for it. Asking
buys you a *good* answer instead of an after-the-fact one.

## Why this shape

- **Out of the request path.** Inference latency is untouched, and a daemon
  crash degrades to today's behaviour rather than taking every provider with it.
- **No lifecycle ownership.** You keep starting servers however you like, with
  whatever flags. The daemon discovers them.
- **Advisory first.** A tool that unloads someone's model without being asked
  is a tool people turn off. Eviction is opt-in per provider.

## What already exists to build on

Every provider ships both halves of what the daemon needs. None of this has to
be invented:

| Provider | Observe | Evict |
|---|---|---|
| LM Studio | `GET /api/v0/models` → `state`, `loaded_context_length`; `lms ps` | `lms unload <model>` |
| llama.cpp | `GET /v1/models` → `meta.n_ctx`; router `GET /running` | `POST /api/models/unload/<model>` |
| vLLM-MLX | `GET /v1/models`; registry budget | registry eviction; process signal |
| Ollama | `ollama ps`, `GET /api/ps` | `ollama stop <model>`, `keep_alive: 0` |

Verified on this machine, 2026-09-08.

## Open questions

Three things worth settling before any code — argued in the design doc, not
decided here:

1. **Advisory or enforcing?** Does the daemon ask a client to wait, or unload
   another provider's model on its own authority?
2. **How is a footprint estimated** on unified memory, where weights are the
   floor and the KV cache is the part that actually kills you?
3. **What about models the daemon did not sanction** — loaded before it started,
   or by someone who never asked?

## Why the inventory half exists

Answering *"is it safe to delete this model?"* on one machine required, by hand,
every time:

1. Is it **served live**? llama.cpp's router scans the HF cache, so downloads
   are runtime dependencies.
2. Is it a **symlink target**? Deleting the link frees nothing; deleting the
   target removes it from two tools at once.
3. Is it **named in a registry**, where a stale path is a startup failure?
4. Is it a **conversion source** whose artifacts exist — or the only copy of
   something that cannot be rebuilt?
5. Is it used by something that is **not an LLM provider**? One 128 MB embedding
   model was a search tool's index backend, invisible to every provider.

Five questions, four stores, no tool. See [`docs/inventory.md`](docs/inventory.md).

## Non-goals

- Not a unified endpoint. Not a router. Not a load balancer.
- Does not start servers, choose models, or transform requests.
- Not a scheduler for throughput. It optimises *fitting*, not tokens/sec.
- Not cross-machine. One host, one daemon.
