# Prior art

Searched 2026-09-08, before writing any design. The conclusion is that the
shape llm-harmony proposes does not exist — but two projects are close enough
to be worth reading, and one of them contributes the central rule.

## Closest: vramd

<https://pypi.org/project/vramd/0.3.6/>

Self-described as *"VRAM admission control for generative inference on consumer
GPUs. One process holds the GPU and decides who gets in."* Admits by real peak —
weights + activation + margin — queues with priority and affinity, evicts by
weight + LRU. Explicitly not an LLM server: *"it doesn't optimize token
throughput, it optimizes fitting."*

That is llm-harmony's thesis, and vramd got there first.

**Why it does not fit here:**

- **CUDA / NVIDIA.** Documentation says POSIX, *"Windows needs a different IO
  layer"*; no mention of Metal or Apple Silicon. Unified memory is not a
  supported target.
- **It owns the processes.** *"Each model in its own venv. A backend is a
  process with its own interpreter."* Integration is by implementing a
  `WorkerAdapter` with `load()` / `generate()` / `unload()`. That replaces
  LM Studio, llama.cpp and vLLM-MLX rather than coordinating them.

**What to steal:** *admit by the real peak, not by the weights.* Every
per-provider budget in use today violates this by omission — vLLM-MLX says so
in its own startup log.

## Rejected: request-path routers

- **llama-swap** — a Go proxy on a unified endpoint. Inspects `model`, starts
  the right `llama-server`, proxies, stops the old one, unloads on TTL.

  **The objection is now narrower, and honesty requires saying so.** As of
  2026-09-09 llm-harmony also starts servers, so "owns lifecycle" no longer
  separates them. What still does, and what the whole design rests on, is the
  request path: llama-swap proxies every token and therefore owns the latency,
  the buffering and the blast radius of its own crashes. llm-harmony hands back
  a `base_url` and gets out of the way. One of those can be killed mid-turn
  without anybody noticing.
- **LiteLLM, LocalAI** — unified gateways. Same objection, and no memory
  awareness at all.

These solve *"one endpoint, many models"*. llm-harmony solves *"many endpoints,
one machine"*. Different problem; they compose fine.

## Rejected: per-provider controls

Each provider already manages its own memory well:

| Provider | Mechanism |
|---|---|
| Ollama | `OLLAMA_MAX_LOADED_MODELS`, `keep_alive`, `ollama stop` |
| llama.cpp | router `--models-max` with LRU eviction |
| LM Studio | JIT load, idle TTL |
| vLLM-MLX | `memory_budget_gb` with eviction policy |

None can see the others. Four correct local decisions summing to one wrong
global one *is* the bug — so a fifth per-provider control cannot be the fix.

## Not applicable: the research

Searches for "admission control" in 2026 LLM literature return work on *memory
for agents* (what an agent should remember) and on *KV-cache admission inside
one serving system*. Neither addresses several independent inference servers
contending for one host's RAM.

## Conclusion

The gap is narrow and real: **every existing tool either routes requests through
itself or launches the servers itself.** A coordinator that does neither —
observing and evicting through control surfaces the providers already expose —
has no equivalent found.

Worth re-checking before implementation begins; this moves quickly.
