# Architecture

Written 2026-09-09, after slice 1 was built and the scope widened from *"keep
four providers from overcommitting memory"* to *"be the management centre for
local models on this machine."*

[`design.md`](design.md) and [`inventory.md`](inventory.md) remain the subsystem
specs. This document covers what sits above them: the boundary the project must
not cross, the two resources it accounts for, and the pieces neither existing
doc describes.

---

## 1. Two planes

The single most important sentence in the project:

> **Sole authority over supply. Zero presence in traffic.**

| | Management plane | Request plane |
|---|---|---|
| llm-harmony's role | the middle-man, deliberately | absent |
| What flows | HF downloads, conversions, store writes, registry edits, unload commands | prompts, completions, tokens |
| If the daemon dies | management stops | **nothing changes** |

Everything the project does — receiving, placing, converting, registering,
evicting, clearing — is management. A client that wants inference gets an
address and connects to the provider directly.

### The one exception, and the narrower claim that replaces it

Amended 2026-09-11, in slice 5. Two of the four providers — llama.cpp and
vLLM-MLX — expose **no control-plane load verb at all**: a model becomes
resident on its first inference request and no earlier
([`field-notes.md`](field-notes.md)). So `RESOLVE` returning `ready` for one of
them can only mean *advertised*, not *resident* — the exact distinction that
made `/v1/models` unusable as a residency signal — unless harmony triggers that
first request itself.

It does, bounded as tightly as the idea allows: **one fixed sentinel token**, to
`/v1/completions`, `max_tokens: 1`, only when the caller asked for residency,
never carrying anything a user wrote.

The honest claim is therefore not "zero bytes on the request plane" but the one
that was always load-bearing:

> **No user prompt passes through harmony.** It never proxies, never forwards,
> never sees a conversation, and never stands between a client and a provider.

The table above still holds for everything else. If this exception ever grows a
second case, that is the signal the project is becoming a router — and the
argument in *Why returning coordinates is not routing* stops applying.

### Why returning coordinates is not routing

[`prior-art.md`](prior-art.md) rejects llama-swap, LiteLLM and LocalAI because
they "either route requests through themselves or launch the servers
themselves." Once llm-harmony picks *which provider serves a model*, that
objection needs answering, because picking looks a lot like routing.

The difference is what happens after the pick:

```
router:      client → proxy → provider → proxy → client
llm-harmony: client → daemon (once, for an address)
             client → provider → client        (every request, forever)
```

A router owns the latency of every token, the blast radius of its own crashes,
and the memory of buffering streams. llm-harmony owns none of those. It answers
one question, once, and the answer is a `base_url` and a provider-local model
id.

This is a thin line, and it is thin in a specific direction: the temptation will
be to add "just a small proxy" for convenience — to normalise response shapes,
or retry a failed provider, or hide the fact that
[`field-notes.md`](field-notes.md) documents three different ways a provider
reports failure inside an HTTP 200. **Each of those is a reasonable feature and
each one crosses the line.** They belong in the client.

### What this costs the client

Being out of the request path means the client absorbs real work: it must speak
each provider's dialect, handle the 200-with-an-error shapes, and cope with a
model disappearing between `RESOLVE` and the request. That is a deliberate
trade — the alternative is inference latency and a shared crash domain.

## 2. Two ledgers

Memory and disk are the same problem at different timescales, and the machinery
generalises cleanly:

| | Memory ledger | Disk ledger |
|---|---|---|
| Unit | bytes resident | bytes stored |
| Truth source | provider status APIs + `phys_footprint` | filesystem scan of four stores |
| Admission | "room to load this?" | "room to download or convert this?" |
| Eviction | unload a model | clear a cache, remove an artifact |
| Reversibility | trivial — reload it | **none** — re-download or re-convert, if possible at all |
| Failure if under-estimated | machine swaps, then wedges | disk fills mid-conversion |
| Failure if over-estimated | wasted capacity | wasted capacity |

Both bias the same way: **over-estimate, refuse early.**

The disk ledger is not a new invention — [`inventory.md`](inventory.md) §4
already defines its safety classification (redundant / reproducible /
irreplaceable) and §5 already demands *"refuse before a 28 GB download, not
after."* Naming it a ledger makes explicit that it shares admission and eviction
logic with memory rather than being a separate feature.

### Why this is the unifying claim

[`inventory.md`](inventory.md) §1 justifies one daemon on the grounds that both
halves need the same dependency graph. That is true but weaker than it looks:
two subsystems sharing a data structure is an argument for a shared library, not
a shared process.

The stronger claim is that they share **policy**. "What do I remove to make room,
and what am I forbidden from removing?" is one question asked of two resources.
A conversion is the case that proves it: a 40 GB scratch job running while three
models are resident consumes *both* ledgers at once, and nothing in the
ecosystem accounts for that.

**This claim should be stress-tested before it is built on.** If the disk
ledger's schema turns out to want a different shape than the memory ledger's,
that is worth discovering at the schema stage, not after a hosting screen exists.

## 3. Model identity — the layer that does not exist

Everything above assumes a client can say *"I want Qwen3-14B"* without naming a
provider. Nothing today supports that.

Every id in slice 1 is provider-local **by design**, and the same model appears
under different names and formats in different stores:

| Provider | Id for the same underlying model |
|---|---|
| LM Studio | `qwen3-14b-claude-4.5-opus-high-reasoning-distill` |
| llama.cpp | `Qwen3-14B-Q4_K_M` |
| Ollama | `qwen3:14b` |

A **canonical model identity** mapping one logical model to N provider-local
artifacts is a prerequisite for placement. It is built from the graph
[`inventory.md`](inventory.md) §3 already specifies — `ALIASES` for symlinks and
duplicate registry names, `DERIVES_FROM` for conversion lineage — plus §6's
format capability matrix to answer "who *can* serve this artifact?"

Two consequences:

- **The inventory graph moves onto the hot path.** It was framed as offline
  ("what breaks if I delete this?"). Placement reads it on every switch, so it
  must be current and fast, not merely correct when `rm` runs.
- **`DERIVES_FROM` being inferred becomes riskier.** [`inventory.md`](inventory.md)
  §7 Q1 left recorded-vs-inferred open. Inferring from names was tolerable for a
  deletion warning; a wrong inference now sends a request to the wrong artifact.
  And [`field-notes.md`](field-notes.md) documents two HF repos whose names lied
  about exactly this.

## 4. `RESOLVE` — the protocol the client actually calls

[`design.md`](design.md) §4 specifies an *admission* protocol: the client names
a provider, asks permission, and loads the model itself.

```
ASK provider model [context_tokens] → GRANT | WAIT | DENY
```

Seamless switching inverts it. The client names a **model** and receives a place
to send requests:

```
RESOLVE model_ref [context_tokens] → READY base_url provider_model_id
                                   | WAIT  reason eta
                                   | DENY  reason
```

`RESOLVE` does four things `ASK` does not:

1. **Place** — choose a provider that can serve this model, from the identity
   graph and the capability matrix.
2. **Admit** — check the memory ledger; evict if permitted and necessary.
3. **Activate** — ensure the model is resident, via the provider's own load path.
4. **Return coordinates** — and then get out of the way.

`ASK` does not disappear. A client that already knows which provider it wants —
a shell alias, a script — still has a simpler question, and the advisory
contract in [`design.md`](design.md) §4 still holds: a client that never speaks
is not broken, it is merely accounted for after the fact instead of before.

### The unresolved question

**If the provider that should serve a model is not running, what happens?**

On this machine right now, llama.cpp and vLLM-MLX are both down. If a client
asks for a model only llama.cpp can serve:

- `DENY provider not running` is honest, and not seamless.
- Starting `llama-server` is seamless, and crosses the server-lifecycle line
  that [`prior-art.md`](prior-art.md) uses to reject llama-swap.

There is a third option worth considering: **place it somewhere else.** If the
identity graph knows an MLX build of the same model exists and vLLM-MLX is up,
`RESOLVE` can return that instead — different artifact, same logical model. That
preserves both the seamlessness and the boundary, at the cost of the client
silently getting a different quantisation than it might expect.

Unresolved. It gates whether "seamless" is achievable without owning lifecycle,
and it should be settled before the placement slice is built.

## 5. Caches are four unrelated things

"Clear cache" is a single verb over four mechanisms with wildly different risk:

| Cache | Size here | Risk | Correct mechanism |
|---|---|---|---|
| Conversion scratch | ~40 GB per 9B | **None** — purely transient | delete freely |
| HF hub cache | 69 GB | **High** — llama.cpp's router scans it, so entries are live serving dependencies | check residency, then treat as artifact removal |
| Ollama blob store | — | **Dedup hazard** — content-addressed by digest; one blob can back several tags | resolve references first |
| KV / prefix cache | not on disk | **Different resource entirely** — runtime memory | unload/reload or a provider call |

The second row is the dangerous one and is already documented:
[`field-notes.md`](field-notes.md) records that *"llama.cpp's router scans the HF
cache, so cache entries are runtime dependencies, not merely downloads."* A
"clear HF cache" button is one click from pulling a model out from under a live
server.

**Design consequence:** clearing a cache is **disk eviction**, not a maintenance
button. Rows two and three go through the same refuse-rather-than-warn path as
`llm-harmony rm`, with the same redundant / reproducible / irreplaceable
classification. Only row one is a simple delete, and row four does not belong in
this subsystem at all.

They should not share an implementation, and probably should not share a word in
the UI.

## 6. Jobs

Receiving and converting are long, heavy, and failure-prone: tens of GB
downloaded, ~40 GB of scratch for a 9B, converters that fail at the last step
(`mlx_lm.convert` calling `snapshot_download(local_files_only=True)` and dying
because a `README.md` is missing).

Slice 1 has nothing resembling a job model — it is a synchronous command that
prints a table. Intake needs queueing, progress, cancellation, and resumption,
and it needs them before a hosting screen can exist.

The useful unification: **a job is a ledger consumer like a load is.** Admitting
a conversion against both ledgers gives *"there is not room to convert this right
now"* for free, which is a real feature and one nothing else offers.

## 7. Being the intake path creates obligations

Once llm-harmony is the only way models arrive, duties that were previously
advice become mandatory:

- **Provenance verification becomes gating.** [`inventory.md`](inventory.md) §5
  records two HF repos that claimed to be builds of models they were not — one
  declaring a `base_model` with a near-identical name, one declaring none at all
  while its config contradicted its own repo name. As a CLI convenience that
  rule is guidance. As the intake path for a browsable hosting screen, it is the
  only thing between a search result and a wrong model on disk.
- **Integrity checking does not exist yet.** HF publishes hashes; nothing
  verifies them.
- **Conversion executes against remote-controlled metadata.**
  `convert_hf_to_gguf.py` parsing a config from an arbitrary repo is a modest
  surface, but it only becomes relevant once repo ids come from a search box
  rather than from someone who already trusts them.
- **Capability loss must be recorded, not just observed.** MLX conversion
  silently dropped 15 MTP tensors *and* a vision tower, with one log line
  hinting at the second. If harmony performs the conversion, it knows the delta
  and should store it.

## 8. Availability is now two-tiered

[`design.md`](design.md) §8 says *"fail open, always"* — a dead daemon degrades
to today's behaviour. That remains true for serving and is non-negotiable.

It cannot be true for management. If harmony is the only path by which models
are installed, converted, or cleared, then harmony being down blocks that work
entirely.

| Plane | Daemon down |
|---|---|
| Serving | Providers unaffected. Clients proceed without coordination, as today. |
| Management | Blocked. No intake, no conversion, no coordinated eviction. |

This is acceptable, but it must be stated rather than discovered. It also argues
for keeping the management surface's state on disk and reconstructible, so a
crash mid-conversion is recoverable rather than a corrupt store.

## 9. Build order

Slice 1 (read-only core) is built and holds up unchanged. What follows was
reordered by the widened scope: placement depends on identity, so the inventory
graph moves ahead of the daemon.

| Slice | What | Why here |
|---|---|---|
| 1 ✅ | Read-only core — adapters, memory ledger, `status` | Foundation for everything; cannot unload by construction |
| 2 | Disk ledger + model identity graph | Placement, install and clear all read from it |
| 3 | Daemon + measured estimator | Needs continuous observation to learn footprints |
| 4 | `RESOLVE` — placement, activation, eviction | The thing a client actually calls |
| 5 | Intake and conversion as admitted jobs | Reuses both ledgers for fit |
| 6 | Hosting screen API | An HTTP surface over 2, 4 and 5 |

## 10. Open questions

1. ~~**If the right provider is not running, does harmony start it, refuse, or
   place elsewhere?**~~ **Answered 2026-09-09: it starts it.** Opt-in per
   provider, running a command the user declares, supervised by launchd rather
   than by harmony — so the §8 fail-open promise survives, verified by removing
   the binary and watching the server keep serving. Placing elsewhere was
   rejected because slice 2 forbids inferred lineage from driving decisions,
   and every cross-format grouping is inferred.
2. **Does the disk ledger genuinely want the memory ledger's shape?** (§2) The
   unifying claim of the project; currently asserted, not demonstrated.
3. **Is `DERIVES_FROM` recorded or inferred?** (§3) Riskier now that placement
   depends on it.
4. **Does harmony own a canonical store, or place native copies per provider?**
   [`inventory.md`](inventory.md) §7 Q3, unresolved and now harder: the machine
   already has `~/.lmstudio/models` symlinking into the llama.cpp pool, so the
   answer is currently "accidentally, both."
5. **How much of a provider's footprint is not model memory?** Measured
   2026-09-09: LM Studio idles at 592 MB across five Electron processes with
   nothing loaded. The estimator must subtract a baseline it does not yet track.
