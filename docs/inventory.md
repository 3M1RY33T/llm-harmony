# The model inventory — safe add and remove

Design sketch, 2026-09-09. Companion to [`design.md`](design.md), which covers
memory admission at runtime. This covers the artifacts on disk: what exists,
what derives from what, and what breaks if a file goes away.

---

## 1. Why this belongs in llm-harmony and not a separate tool

The two subsystems ask the same question at different timescales:

- **Admission (runtime):** *what breaks if I load this now?* → memory
- **Inventory (offline):** *what breaks if I delete this?* → dependencies

Both need one thing the machine does not currently record anywhere: a graph of
models, the places they live, and who depends on them. Building that graph twice
would guarantee two answers to the same question. Same daemon, same graph, two
readers.

## 2. The problem, from a real session

A single 24 GB Mac accumulated models across four stores with tangled
dependencies between them:

```
~/.cache/huggingface/hub    sources AND servable GGUFs — SCANNED LIVE by llama.cpp
~/.lmstudio/models          LM Studio's own, some symlinked into the llama.cpp pool
~/.llamacpp/models          pool: a mix of real files and symlinks
~/.vllm-mlx/models          MLX builds, referenced by name in models.yaml
```

Answering *"is it safe to delete this?"* required, every time, by hand:

1. Is it being **served live**? The llama.cpp router scans the HF cache, so
   cache entries are runtime dependencies, not just downloads.
2. Is it a **symlink target**? Deleting the link frees nothing; deleting the
   target removes the model from two tools at once.
3. Is it **named in a registry** — `models.yaml` — where a stale path is a
   startup failure?
4. Is it a **conversion source** whose artifacts already exist, or the only
   copy of something irreplaceable?
5. Is it used by something that is **not an LLM provider at all**? One 128 MB
   embedding model turned out to be `loci`'s search index backend, invisible to
   every provider and easy to delete by accident.

Five questions, four stores, no tool. That is the feature.

## 3. The dependency graph

One node per artifact, edges for the three relationships that matter.

| Node field | Meaning |
|---|---|
| `path` | where it actually is |
| `store` | hf-cache \| lmstudio \| llamacpp \| vllm \| other |
| `format` | safetensors-bf16 \| mlx \| gguf |
| `bits` | 4, 6, 8, 16 … |
| `bytes` | real size, following symlinks once |
| `capabilities` | mtp, vision, tool_use — what this artifact still has |

Edges:

- **`DERIVES_FROM`** — this GGUF/MLX build came from that source. Recorded at
  conversion time, when it is knowable for free; inferred later only as a guess.
- **`ALIASES`** — a symlink, or two registry names pointing at one directory.
  Deduplicates the size report and makes the "frees nothing" case obvious.
- **`REQUIRED_BY`** — a provider serves it, a registry names it, or a non-LLM
  tool depends on it. The edge that makes removal unsafe.

## 4. Safe removal

`llm-harmony rm <model>` answers the five questions before doing anything, and
refuses rather than warns when an answer is bad:

```
$ llm-harmony rm Qwen3.6-27B-A3B-Coder
  14G  ~/.lmstudio/models/mradermacher/Qwen3.6-27B-A3B-Coder-GGUF
       ALIASED  ~/.llamacpp/models/Qwen3.6-27B-A3B-Coder-Q4_K_S.gguf (symlink)
       SERVED   llama.cpp pool, currently loaded
       reclaims 14G — removes it from LM Studio AND the llama.cpp pool
  refusing while loaded; `llm-harmony unload` first, or --force
```

Ordering matters and is easy to get wrong by hand: **remove aliases before
targets**, or the store is briefly full of dangling links. A `--dry-run` that
prints the plan should be the default posture for anything above a threshold.

The report must distinguish three kinds of "safe", because they are not the same
risk:

- **Redundant** — a better artifact of the same model exists (Q4 when Q6 is
  present). Safe.
- **Reproducible** — deleting costs a re-download or a re-convert, and the
  recipe is known. Safe but priced.
- **Irreplaceable** — the only local copy of something that cannot be rebuilt
  from what remains. Never auto-selected.

That last category is not hypothetical. One 9B source held 15 MTP tensors that
**every MLX conversion silently dropped**; only its GGUF builds kept them. An
inventory that reasoned purely about "same model, smaller file" would have
happily deleted the one artifact carrying a capability.

### `clean --redundant` — the same question, asked of everything

`rm` names one model. `clean --redundant` classifies every model and plans the
builds a better one supersedes, keeping the best of each format. Only
`Redundant` is selected: `Reproducible` is a *priced* deletion and belongs to a
person, not to a sweep.

One difference from `rm` is load-bearing. A refusal poisons **that model's**
plan and nothing else — a served model must not hold the other forty hostage —
so `clean` builds one plan per identity group and reports the poisoned ones as
skipped.

Measured here 2026-09-11: 19 GB of superseded builds across three models. One
is the cross-store case from §2 — LM Studio's Q4 with the llama.cpp pool
symlink pointing into it, the alias ordered first and priced at zero, exactly
as this section requires.

**The report is not the whole risk.** Cleaning that Q4 leaves llama.cpp's Q8_0
as the model's only build, and it is 14.6 GiB where the Q4 was 8.4 GiB — a disk
clean that raises a model's memory floor by ~6 GB on a 24 GB machine. Whether
the disk ledger may do that to the memory ledger is open; §7 Q5.

### `ls --duplicates` — the same bytes, stored twice

`FileKey` collapses two paths that share an allocation, so the ledger never
double-counts a symlink. It says nothing about two *separate* copies, which is
the failure a store-per-provider layout invites: pull one repo into LM Studio
and llama.cpp independently and the disk pays twice with nothing aliased.

Candidates are same-size allocations above 64 MB; each is then compared on its
first and last 4 MB. The floor is the only other filter, and that is
deliberate — an HF blob is named by its sha256 with no extension, so filtering
on `Format` first hid the entire 69 GB cache, the store most likely to hold two
revisions of one repo.

Sampling is not proof, and the report says which basis it used rather than the
word *identical*. It also never proposes an action: replacing a copy with a
symlink is a write, and this slice has none.

Measured here 2026-09-11: **none**. 37 candidate allocations above 64 MB, five
same-size groups, and all five different models once sampled — two MLX
conversions of sibling 9B models quantise to byte-identical *sizes*. Equal size
is a candidate and never an answer.

## 5. Safe addition

`llm-harmony add <hf-id>` is the pipeline this session ran by hand, five times:

```
resolve → verify provenance → check fit → download → convert → patch → place → register
```

**Resolve.** Is it already MLX or GGUF, or PyTorch needing conversion?
`looks_like_mlx_artifact` / `needs_conversion` answers this without downloading
weights, and should gate everything after it.

**Verify provenance.** The step most easily skipped and most costly to skip.
Two HF repos this session *looked* like builds of a wanted model and were not:

- one declared `base_model: nightmedia/…-DS9-USS-Defiant` — a different model
  with a near-identical name;
- one declared no `base_model` at all **and** was named `-8bit` while its config
  said 4 bits.

Rule: an artifact claiming to be a build of X must **declare** X, or be treated
as an unrelated model that happens to share a name.

**Amended 2026-09-11: this warns, it does not refuse.** Behind a search box the
rule was going to be the only thing between a result and a wrong model on
disk — which argued for gating, and is what
[`architecture.md`](architecture.md) §7 said. Put to the user with the search
box built, the answer was the other way: show the declared and the expected
`base_model` side by side and let the pull proceed, because a refusal is worked
around in a terminal where nothing checks at all. The obligation moves onto the
warning instead — it names both models and is carried on the result, not only
in the progress stream.

**Check fit.** Download size, conversion scratch (~40 GB for a 9B), and the
resulting footprint against free disk *and* the memory budget from `design.md`.
Refuse before a 28 GB download, not after.

**Which ledger refuses, amended 2026-09-11.** Disk refuses; memory warns. A
build that will land but cannot be loaded *as things stand* is still worth
having — making room in memory is a different act from making room on disk, and
the memory reading changes minute to minute while the disk one does not. So
`add` plans it and carries the reason outward. The memory half is still checked
against the **same bar the loader uses** (`free − reserve`, not raw free
memory): two admission paths over one resource holding each other to different
numbers is how a pull gets admitted that `load` then refuses.

**Convert — with the converter that matches the runtime.** `convert_hf_to_gguf.py`
from llama.cpp *master* against a `b10240` binary produced GGUFs that failed to
load: `block_count` 33 with tensors only to 31, then a second 33-length array
behind it. Re-converting at the matching tag fixed both. The rule: **pin the
converter to the serving binary's build**, and record which build produced each
artifact.

**Patch known source defects.** Some sources are internally inconsistent and
every conversion inherits it. One had 760 tensors, zero `mtp.*`, and a config
still declaring an MTP layer — so each GGUF needed `block_count 32` and
`nextn_predict_layers 0` before it would load. Detectable mechanically:
*metadata promises a tensor group the weights do not contain.*

**Record what was lost.** Conversion is lossy in ways nothing reports.
`mlx_lm.convert` dropped MTP heads **and** the vision tower; the log said only
`Checkpoint has no weights for module(s): vision_tower`. The inventory should
diff source capabilities against artifact capabilities and store the delta, so
`ls` can show that an artifact is less capable than its parent.

## 6. Format capability matrix

**Superseded as a source of truth, 2026-09-11.** `llm-harmony verify --json`
now carries a `formats` array per provider, answered by the adapter rather than
read from here. The table below is kept as the reasoning that produced it, and
because the two consequences after it are still true.

It was wrong for the machine it was written on. The vLLM row says safetensors
bf16 via "convert"; the vLLM there is **vllm-mlx**, which serves MLX and loads
no safetensors in any dtype — while a *stock* vLLM loads compressed-tensors,
AWQ and GPTQ directly. One row cannot describe both, and the difference decides
whether a pull is refused. `Actuation` had already made the same move in slice
5, for the same reason: an adapter written from documentation is a hypothesis.

Which providers can serve what — the table that makes "safe to delete" answerable
across stores:

| Format | llama.cpp | vLLM-MLX | LM Studio | Ollama |
|---|---|---|---|---|
| GGUF | ✅ | ❌ | ✅ | ✅ |
| MLX | ❌ | ✅ | ✅ | ✅ (MLX backend) |
| safetensors bf16 | convert | convert | ❌ | ❌ |

**And "safetensors" is not one row.** `.safetensors` is a container: bf16, FP8,
compressed-tensors, AWQ, GPTQ and bitsandbytes all wear the extension, and only
`config.json` says which. Until 2026-09-11 harmony read the extension, so every
quantised repo on Hugging Face was reported as bf16 needing a conversion —
including the ones a stock vLLM loads without one. Found by pulling
`mconcat/…-FP8-Dynamic`, which is 8-bit compressed-tensors and was described as
bf16.

Two consequences worth encoding rather than rediscovering:

- **There is no MLX ↔ GGUF path.** Both are quantized forms of the same
  original; neither derives from the other. "Convert my MLX model to GGUF" means
  re-quantizing from the source, which is why sources have value after their
  artifacts exist.
- **Quantization is one-way.** Q4 → Q8 is impossible; discarded precision does
  not come back. Upgrading a bit depth is a fresh conversion from bf16.

## 7. Open questions

1. ~~**Is `DERIVES_FROM` recorded or inferred?**~~ **Answered 2026-09-11:
   both, split by who made the artifact.** Recorded at conversion time for
   anything harmony converts — which Q3's answer now makes harmony's job —
   and inferred from names, marked as inferred, for everything already on
   disk. The guess survives only where there is nothing better, and it still
   may not drive placement or deletion.
2. **Does `rm` ever act without confirmation?** Deletion is unrecoverable and
   re-download is expensive. Dry-run-by-default is the safe posture; the cost is
   friction on the common, harmless case.
3. ~~**Should it manage the stores, or only report on them?**~~ Moving models
   to one pool would deduplicate, but it means owning layout — and every
   provider is currently happy pointing wherever the user likes. **Narrowed
   2026-09-11:** `ls --duplicates` measured what a pool would deduplicate here,
   and the answer is nothing — 146 GB, zero duplicate allocations, because the
   cross-store sharing on this machine is already done with symlinks. The case
   for a pool cannot be made from deduplication; it would have to be made from
   owning intake, where placement is harmony's anyway. **Answered 2026-09-11:
   own what harmony makes, index what it did not.** Conversions go into a store
   harmony owns and lays out, because that is the only place where it knows the
   lineage and the capability delta first-hand. Everything a provider's own
   downloader put on disk is reported on and never moved. See
   [`architecture.md`](architecture.md) §10 Q4.
4. **Does capability-loss detection generalise?** MTP and vision tower were
   caught by diffing tensor names. Whether that finds the next kind of loss, or
   only the two already known, is untested.
5. **May a disk clean raise a model's memory floor?** Opened 2026-09-11 by
   `clean --redundant`, which is right that a Q8_0 is the better artifact and
   silent that it costs ~6 GB more to load than the Q4 it supersedes. Three
   shapes: note it in the report, refuse without an explicit override, or
   admit the survivor against the memory ledger the way slice 7 admits a job
   against both. The third is the one that fits the two-ledger claim, and it is
   the most work.
