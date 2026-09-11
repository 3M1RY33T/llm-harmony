# Roadmap

Written 2026-09-09. The arc from what exists to the goal in
[`architecture.md`](architecture.md): Hugging Face for receiving, four providers
for serving, llm-harmony in between.

Each slice produces something that works and is verifiable on its own. Each one
is planned in detail only once the slice before it has answered the questions
that shape it — the slice-1 plan was written after probing the live providers,
and it caught four defects that guesswork would have shipped.

---

## Where we are

| Slice | State |
|---|---|
| 1 — Read-only core | **Built.** 44 tests. `llm-harmony status` reads four providers and reports real footprint against machine capacity. Cannot unload by construction. |
| 2 — Disk ledger + identity graph | **Built.** 39 tests. `ls` and `rm --dry-run` over 146 GB across five stores; cannot delete by construction. `clean --redundant` and `ls --duplicates` added 2026-09-11, same property. |
| 3 — Measured estimator | **Built.** `estimate` predicts a footprint and returns a fit verdict. |
| 4 — `RESOLVE` | **Built.** Placement by artifact identity; admission by the estimate ladder. |
| 5 — Actuation | **Built.** 249 tests. `load`, `unload`, `switch`, `pin`/`unpin`, `verify`. Planned in [`plans/2026-09-10-actuation.md`](plans/2026-09-10-actuation.md). |
| 6 — Arbitration and set admission | **Built.** `compare`, `lease`/`release`, `fit`, `status --budgets`. Planned in [`plans/2026-09-11-arbitration.md`](plans/2026-09-11-arbitration.md), which corrected two of the four rows below and records what execution changed. |
| 7 — Intake and conversion | **Half built.** `search` and `add` pull a pre-built GGUF or MLX repo, admitted against both ledgers before a byte moves. Conversion not started. |
| 8 — Hosting API | Sketched |

---

## Slice 2 — Disk ledger and model identity

**Delivers:** `llm-harmony ls` and `llm-harmony rm --dry-run`.

Scans all five stores in place, builds the artifact graph, groups artifacts
under canonical model identities, and classifies every artifact as redundant,
reproducible, or irreplaceable. Prints a removal plan it **cannot execute**.

**Why first:** placement (slice 4), intake (slice 7) and cache clearing all read
from this graph. It is also the second half of the two-ledger claim in
[`architecture.md`](architecture.md) §2 — and that claim is asserted, not yet
demonstrated. This slice is where it gets tested.

**Decisions taken:** index in place, never move files. Lineage inferred but
marked, and inferred edges may not drive placement or deletion. Read-only, with
the same structural safety property as slice 1.

**Settles:** whether the disk ledger genuinely wants the memory ledger's shape
(§10 Q2), and how much lineage is recoverable by inference (§10 Q3).

## Slice 3 — The measured estimator

**Delivers:** footprint estimates that stop being guesses, and a fit verdict.

**Reshaped 2026-09-09.** This was "daemon and estimator". Two findings removed
the daemon from it:

* **The corpus had no LM Studio baseline**, because LM Studio never sat idle
  while the recorder existed. That kills *whole-tree minus idle baseline* — a
  provider that never idles never yields one.
  **Superseded 2026-09-11:** it now holds 193 idle LM Studio readings, and
  §10 Q5 is answered from them. The finding stands as the reason this slice was
  reshaped; it is no longer true of the corpus.
* **Per-process attribution needs no baseline.** The model-bearing backend
  dwarfs the shell (8.39 GB against 0.28 GB, measured), so the largest process
  in the tree *is* the model. That works from a single observation, which
  removes the daemon's main justification: continuity.

Delroy already records an observation every turn, so coverage arrives without a
timer. The daemon is deferred until a missed short-lived state is actually
observed. Planned in
[`plans/2026-09-09-measured-estimator.md`](plans/2026-09-09-measured-estimator.md).

**Depends on:** slice 1's adapters, and the `--record` path added alongside the
Delroy integration.

**Does not settle the reserve.** The earlier draft claimed it would. Ten
observations cannot; it ships as a documented default with an override.

## Slice 4 — `RESOLVE`

**Delivers:** the call Delroy actually makes.

```
RESOLVE model_ref [context_tokens] → READY base_url provider_model_id
                                   | WAIT  reason eta
                                   | DENY  reason
```

Placement (from slice 2's graph), admission (from slice 3's estimator),
activation via the provider's own load path, and eviction. This is the first
slice that can unload a model, so it is the first that needs the
`observe`/`evict`/`manage` trust modes from [`design.md`](design.md) §6.

**Blocked on:** §10 Q1 — *if the provider that should serve a model is not
running, does harmony start it, refuse, or place the request on a different
artifact of the same model?* This determines whether seamless switching is
achievable without owning server lifecycle, and it cannot be deferred past the
design of this slice.

**Depends on:** slices 2 and 3, both.

## Slice 6 — Arbitration and set admission

**Delivers:** `compare`, `lease`, `fit`, and `status --budgets`.

Four commands, no new adapters, and no new I/O beyond one poll. Every one of
them exists only because the four providers are now read through one ledger and
one identity, and not one of them is available to a provider on its own.

| Command | The question it answers | Where the parts already are |
|---|---|---|
| `compare <model>` | every provider that could serve this, priced, with a fit verdict | the candidate list `resolve` builds and discards (`src/resolve/decide.rs`) |
| `lease <model> --ttl --owner` | may this be evicted out from under a caller still using it? | `src/pins.rs`, plus an owner and an expiry |
| `fit <model>...` | can these be resident *at once*, and under which assignment? | the estimate ladder, one machine read, and a search |
| `status --budgets` | what is each provider allowed to take, and in what unit? | each provider's own argv, environ, API or config — not harmony's |

**Cheapest first, and that is also the order of the value.** `compare` prints
data the resolver already computes and throws away.

**Corrected by the plan, 2026-09-11.** Two rows above were sketched wrong.
`status --budgets` cannot sum anything — the four ceilings are in three
different units, one of them is Ollama's own undocumented default and so
unknowable, and LM Studio has none in either unit; the command prints what each
provider is allowed with `?` where it cannot be read. And it is not a standalone
nicety: llama.cpp runs `--models-max 1` here, so a set can be refused by a count
ceiling with the bytes to spare, which makes the ceiling reader an input to
`fit` and moves it ahead of it in the build order.

**Why here:** it is composition over slices 1–5, and it closes a gap slice 5
opened. `src/pins.rs` says a pin is *a veto, never a reservation*, and it is
sticky until `unpin` — so a second caller's `resolve` may evict the model a
long-running client is still using, and the only defence today is a manual pin
that nobody remembers setting. A lease is that same store with an owner and an
expiry, which is arbitration between callers rather than protection from the
policy.

**`fit` is the load-bearing one.** It is admission generalised from one item to
a set, and a set has a dimension one model does not: the same logical model
exists as several artifacts at several prices, so *does this set fit* is a
selection as well as a sum. That is an assignment problem over the candidate
lists, and the candidate lists exist only because of the identity graph. It is
also the first question sequential admission cannot answer — three `load` calls
each admit against a ledger the previous one changed, so the third is denied
after the first two have already moved.

**Decisions taken:**

- Read-only, except `lease`. `fit` plans; actuating a planned set is slice 7's
  work or later. Slices 1 and 2 could not act by construction, and the planner
  keeps that property.
- **A set's basis is the weakest rung in it.** `decide` will not admit on
  `Declared` or `Unknown` ([`design.md`](design.md) §5), so a set holding one
  unpriced model is an unpriced set. The alternative is a total that looks
  authoritative because two of its three terms were measured.
- Cost is `Σ baseline(provider) + Σ cost(model)`, each provider's baseline
  counted once. Summing per-model footprints counts it once per model.
- The search is exhaustive, not heuristic. Six models with four candidates each
  is small, and exhaustive is what makes *why not the other one* printable.

**Unblocked 2026-09-11.** It needed [`architecture.md`](architecture.md) §10
Q5 — per-provider baselines, which `estimate` gets away without because one
model means one baseline and a set does not. Measured rather than decided, and
the corpus already held it: **≈1.0 GiB with all four providers idle**, taken at
p99 because the maximum is a single reading that drifts. Three of the four
figures are tight; vLLM-MLX's rests on two readings, so a plan that leans on it
has to say so.

**Settles:** §10 Q2, at the memory end. Set admission is the function slice 7
needs for jobs — *is there room to convert this* is `fit` over the disk ledger
with a scratch term — so building it once and reusing it there is what
demonstrates that the two ledgers share policy rather than merely a data
structure.

## Slice 7 — Intake and conversion

**Delivers:** `llm-harmony add <hf-id>`, conversion, and real deletion.

**Half built, 2026-09-11 — the pull, not the conversion.** `search` and `add`
handle a repo that is *already* built for a provider harmony can serve: GGUF
and MLX. The order is the design and the tests assert it — resolve the repo,
read its provenance, price it against both ledgers, choose its store, and only
then move a byte, because a refusal at 97% is not a refusal.

Planned as Task 8 of Delroy's `docs/hosting-page-r27-plan.md` rather than in
this repo, because the driver was the hosting page and the plan spans both
codebases. Worth knowing where to look; worth not repeating.

What running it caught, none of which a unit test would have:

- **A sharded GGUF is one build across several files.** "Smallest loadable
  file" chose `…-00007-of-00007.gguf` — 3.25 GB, which fit comfortably, so
  both ledgers passed and a pull started for one seventh of a model. Shards
  are now folded into a build priced at its total, and one unpriced part makes
  the whole total unknown.
- **An importance matrix is not a model.** `imatrix_unsloth.gguf` is a real,
  loadable GGUF used to *produce* a quantisation, and "smallest loadable"
  chose it. So is an `mmproj` projector.
- **Search priced disk against no directory** and reported `0B free`, so every
  row read *will not fit* on a machine with 161 GB spare.

Still out, and the reason this is a half rather than a slice: conversion, and
therefore the scratch accounting, the converter pinned per serving binary, the
defect patching, and the capability-loss diff.

The pipeline [`inventory.md`](inventory.md) §5 specifies: resolve → verify
provenance → check fit → download → convert → patch → place → register. Plus the
write half of slice 2: actual `rm`, and cache clearing as disk eviction.

**Needs a job model** that nothing in slices 1–6 has: queued, resumable,
cancellable, progress-reporting. A 9B conversion needs ~40 GB of scratch and can
fail at the last step.

**This is where the daemon returns**, decided 2026-09-11. Slice 3 removed it on
the grounds that per-process attribution needs no continuity. A job model does,
and no amount of per-invocation cleverness makes a resumable 40 GB conversion
work. The continuous behaviours that were the daemon's original case — a
standing swap guard, cross-provider LRU, one idle policy instead of four —
arrive as a dividend of that decision rather than as its justification.
Building a daemon for the warden alone would still fail slice 3's bar: a missed
short-lived state has to be observed first.

**The unification worth building for:** a job is admitted against *both* ledgers.
"There is not room to convert this right now" is a feature nothing else offers.

**Carries the supply-chain obligations** from [`architecture.md`](architecture.md)
§7 — provenance becomes gating rather than advice, and integrity checking has to
exist.

## Slice 8 — Hosting API

**Delivers:** the HTTP surface the hosting screen talks to. Search, browse,
install, convert, clear, switch.

Mostly composition over slices 2, 4, 6 and 7, plus streaming progress for jobs
and whatever search/browse needs from the HF API.

**Note:** this is the first slice where the daemon being down blocks a user
workflow rather than degrading gracefully — see
[`architecture.md`](architecture.md) §8.

---

## The gates between slices

Three questions each block a specific slice, and none can be answered by
thinking harder — they need the slice before them to exist.

| Question | Blocks | Answered by |
|---|---|---|
| Does the disk ledger want the memory ledger's shape? | the case for one daemon | slice 2 |
| What is the real memory reserve? | admission being trustworthy | a broader corpus, not slice 3 |
| Provider not running: start it, refuse, or place elsewhere? | slice 4's whole design | a decision, before slice 4 |

The third is the only one that is a judgement call rather than a measurement,
and it is the one worth arguing early.

## What could make this shorter

Two honest off-ramps, worth naming so they are chosen rather than drifted into:

- **If Delroy only ever needs a memory notice** — not gated switching — then
  slices 3 and 4 collapse into "expose `status` over a socket," and the project
  stays much closer to its original scope.
- **If the two-ledger claim fails in slice 2** — if the disk graph wants a
  genuinely different shape — then inventory should become a separate binary
  sharing a library, and [`architecture.md`](architecture.md) §2 needs rewriting
  rather than defending.
