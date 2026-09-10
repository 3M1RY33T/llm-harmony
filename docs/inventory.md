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

**Check fit.** Download size, conversion scratch (~40 GB for a 9B), and the
resulting footprint against free disk *and* the memory budget from `design.md`.
Refuse before a 28 GB download, not after.

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

Which providers can serve what — the table that makes "safe to delete" answerable
across stores:

| Format | llama.cpp | vLLM-MLX | LM Studio | Ollama |
|---|---|---|---|---|
| GGUF | ✅ | ❌ | ✅ | ✅ |
| MLX | ❌ | ✅ | ✅ | ✅ (MLX backend) |
| safetensors bf16 | convert | convert | ❌ | ❌ |

Two consequences worth encoding rather than rediscovering:

- **There is no MLX ↔ GGUF path.** Both are quantized forms of the same
  original; neither derives from the other. "Convert my MLX model to GGUF" means
  re-quantizing from the source, which is why sources have value after their
  artifacts exist.
- **Quantization is one-way.** Q4 → Q8 is impossible; discarded precision does
  not come back. Upgrading a bit depth is a fresh conversion from bf16.

## 7. Open questions

1. **Is `DERIVES_FROM` recorded or inferred?** Recording at conversion time is
   exact and needs llm-harmony to have done the conversion. Inferring from names
   is available for existing files and is a guess — and names lie, as §5 shows.
2. **Does `rm` ever act without confirmation?** Deletion is unrecoverable and
   re-download is expensive. Dry-run-by-default is the safe posture; the cost is
   friction on the common, harmless case.
3. **Should it manage the stores, or only report on them?** Moving models to one
   pool would deduplicate, but it means owning layout — and every provider is
   currently happy pointing wherever the user likes.
4. **Does capability-loss detection generalise?** MTP and vision tower were
   caught by diffing tensor names. Whether that finds the next kind of loss, or
   only the two already known, is untested.
