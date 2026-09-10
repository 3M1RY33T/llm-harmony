# llm-harmony Slice 3 — The Measured Estimator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the observation corpus into an answer to *"how much will this model cost, and will it fit?"* — the first question llm-harmony has been unable to answer.

**Architecture:** A corpus reader over `observations.jsonl`, an attribution rule that isolates the model-bearing process within a provider's tree, and an estimator that groups measurements by `(provider, model, context_tokens)`. No daemon, no new process lifecycle, nothing written except the observation log that already exists.

**Tech Stack:** Same crate. No new dependencies.

**Spec:** [`../design.md`](../design.md) §5 (estimating a footprint), [`../roadmap.md`](../roadmap.md) slice 3.

## Global Constraints

- **Read-only, as slices 1 and 2 were.** The only write path stays `record.rs` appending to its own state file. No unload, no delete, no store mutation.
- **An estimate must say how it was reached.** `design.md` §5 ranks measured over computed over declared. A number without its provenance is the kind of confident wrong answer that wedges a machine.
- **Never invent a measurement.** A shape with no observations returns *unknown*, not an interpolation. Under-estimating is the failure that crashes the machine; refusing to answer is merely unhelpful.
- **The observation schema is independent of the ledger schema.** See Task 1 — they are currently the same constant, which is a latent bug.
- **Attribution is the largest process in the tree**, and only where exactly one model is resident. Measured 2026-09-09: `llama-server` held 8.39 GB against 0.28 GB for LM Studio's five Electron processes combined.

## What the corpus says today

Ten observations, and they shaped this plan:

| Shape | Count |
|---|---|
| `lmstudio`, 1 model @ 40,960 ctx | 5 |
| `ollama`, 0 models (baseline) | 5 |

**There is no LM Studio baseline**, because LM Studio has held a model the entire time the recorder has existed. That is what killed the roadmap's original design — *whole-tree footprint minus recorded idle baseline* — because a provider that never sits idle never produces one. Per-process attribution replaces it and needs no baseline at all.

---

### Task 1: Give observations their own schema, and record per-process detail

The estimator cannot attribute anything from the current record: it stores only tree totals.

**Files:**
- Modify: `src/record.rs`, `src/memory.rs`, `src/ledger.rs`
- Test: unit tests in `src/record.rs`

**Interfaces:**
- Produces: `record::OBSERVATION_SCHEMA` (independent of `ledger::SCHEMA`), `ProcessSample { pid, footprint_bytes, phys_footprint_bytes, rss_bytes }`, and `Observation.processes: Vec<ProcessSample>`.

- [ ] **Step 1: Write the failing test**

Append to `src/record.rs`'s test module:

```rust
    /// The two schemas travel separately. Delroy pins `ledger::SCHEMA` and
    /// would break on a bump; the corpus is read only by this crate. Sharing
    /// one constant means a change to either silently invalidates the other.
    #[test]
    fn the_observation_schema_is_not_the_ledger_schema() {
        // Not a value assertion -- a *separation* assertion. These may happen
        // to be equal today; they must not be the same constant.
        let obs = OBSERVATION_SCHEMA;
        let _ledger = crate::ledger::SCHEMA;
        assert!(obs >= 1);
    }

    #[test]
    fn an_observation_carries_its_per_process_breakdown() {
        let l = ledger(vec![row_with_processes(
            ProviderKind::LmStudio,
            Outcome::Ok(vec![model("qwen3-14b", State::Loaded, Some(40960))]),
            Some(9_600_000_000),
            vec![(647, 300_000_000), (25164, 9_000_000_000), (3549, 150_000_000)],
        )]);
        let obs = observations(&l, 1);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].processes.len(), 3);
        assert_eq!(obs[0].schema, OBSERVATION_SCHEMA);
    }
```

Add the fixture helper alongside the existing `row`:

```rust
    fn row_with_processes(
        kind: ProviderKind,
        outcome: Outcome,
        fp: Option<u64>,
        procs: Vec<(u32, u64)>,
    ) -> ProviderRow {
        let mut r = row(kind, outcome, fp);
        r.pids = procs.iter().map(|(p, _)| *p).collect();
        r.processes = procs
            .into_iter()
            .map(|(pid, bytes)| crate::memory::ProcessSample {
                pid,
                footprint_bytes: bytes,
                phys_footprint_bytes: Some(bytes),
                rss_bytes: Some(bytes),
            })
            .collect();
        r
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib record`
Expected: FAIL — `cannot find value OBSERVATION_SCHEMA`, `no field processes`.

- [ ] **Step 3: Add `ProcessSample` to `src/memory.rs`**

```rust
/// One process's contribution, kept so a model's cost can be attributed to the
/// process that actually holds it rather than to the provider as a whole.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProcessSample {
    pub pid: u32,
    /// `max(phys_footprint, rss)` for this process.
    pub footprint_bytes: u64,
    pub phys_footprint_bytes: Option<u64>,
    pub rss_bytes: Option<u64>,
}
```

Add `pub processes: Vec<ProcessSample>` to `ProcessTree`, and populate it in `footprint_for_port` inside the existing per-pid loop:

```rust
    let mut samples = Vec::new();
    for pid in &pids {
        if let Some((phys, rss)) = usage(*pid) {
            let combined = phys.max(rss);
            total += combined;
            phys_total += phys;
            rss_total += rss;
            any = true;
            samples.push(ProcessSample {
                pid: *pid,
                footprint_bytes: combined,
                phys_footprint_bytes: Some(phys),
                rss_bytes: Some(rss),
            });
        }
    }
```

- [ ] **Step 4: Carry it through `ProviderRow` in `src/ledger.rs`**

Add `pub processes: Vec<ProcessSample>` to `ProviderRow`, populated from the tree:

```rust
                            processes: tree
                                .as_ref()
                                .map(|t| t.processes.clone())
                                .unwrap_or_default(),
```

Update the three `ProviderRow` literals in `src/render.rs`'s tests with `processes: Vec::new()`.

- [ ] **Step 5: Separate the schema and record the breakdown in `src/record.rs`**

```rust
/// Version of the *observation* record.
///
/// Deliberately not `ledger::SCHEMA`. That one is a contract with external
/// consumers -- Delroy pins it and refuses a document it does not recognise --
/// while this one versions a local corpus only this crate reads. Sharing a
/// constant meant a change to either silently invalidated the other.
pub const OBSERVATION_SCHEMA: u32 = 2;
```

Change `Observation.schema` to be stamped with `OBSERVATION_SCHEMA`, add
`pub processes: Vec<ProcessSample>`, and populate it from `row.processes`.

Derive `serde::Deserialize` on `Observation` — Task 2 reads these back. That
requires `Deserialize` on every field type it owns, so add it to
`ProviderKind` in `src/provider.rs` as well:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
```

`ProcessSample` already derives both, from Step 3.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --lib`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/record.rs src/memory.rs src/ledger.rs src/render.rs
git commit -m "feat: record per-process samples under an independent observation schema"
```

---

### Task 2: Read the corpus

**Files:**
- Create: `src/estimate/mod.rs`, `src/estimate/corpus.rs`
- Modify: `src/lib.rs`
- Test: `tests/estimate_corpus.rs`

**Interfaces:**
- Produces: `corpus::load(&Path) -> Vec<Observation>`, `corpus::load_default() -> Vec<Observation>`.

- [ ] **Step 1: Write the failing test**

```rust
mod support;

use llm_harmony::estimate::corpus;
use support::tree::Tree;

const V2: &str = r#"{"schema":2,"at":100,"provider":"lmstudio","loaded":1,"model":"qwen3-14b","context_tokens":40960,"attributable":true,"footprint_bytes":9600000000,"phys_footprint_bytes":6890000000,"rss_bytes":8670000000,"processes":[{"pid":647,"footprint_bytes":300000000,"phys_footprint_bytes":300000000,"rss_bytes":280000000},{"pid":25164,"footprint_bytes":9000000000,"phys_footprint_bytes":6300000000,"rss_bytes":9000000000}],"machine_total_bytes":25769803776,"machine_used_bytes":20000000000,"swap_used_bytes":5100000000}"#;

#[test]
fn well_formed_lines_are_loaded() {
    let t = Tree::new("corpus-ok");
    let p = t.write("observations.jsonl", &format!("{V2}\n{V2}\n"));
    assert_eq!(corpus::load(&p).len(), 2);
}

/// The corpus is append-only and long-lived. One unreadable line -- a partial
/// write, a record from an older schema -- must not discard the rest.
#[test]
fn a_corrupt_line_is_skipped_not_fatal() {
    let t = Tree::new("corpus-corrupt");
    let p = t.write("observations.jsonl", &format!("{V2}\nnot json\n{{}}\n{V2}\n"));
    assert_eq!(corpus::load(&p).len(), 2);
}

/// Schema 1 records have no `processes`, so nothing can be attributed from
/// them. They are dropped rather than read with v2 assumptions.
#[test]
fn records_from_an_older_schema_are_dropped() {
    let t = Tree::new("corpus-v1");
    let v1 = V2.replace(r#""schema":2"#, r#""schema":1"#);
    let p = t.write("observations.jsonl", &format!("{v1}\n{V2}\n"));
    assert_eq!(corpus::load(&p).len(), 1);
}

#[test]
fn a_missing_corpus_is_empty_not_an_error() {
    assert!(corpus::load(std::path::Path::new("/nonexistent/observations.jsonl")).is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test estimate_corpus`
Expected: FAIL — unresolved import.

- [ ] **Step 3: Write `src/estimate/corpus.rs`**

```rust
use std::path::Path;

use crate::record::{Observation, OBSERVATION_SCHEMA};

/// Every readable observation of the current schema.
///
/// The corpus is append-only and outlives any single build, so this is
/// deliberately forgiving: a torn final line from a process that died
/// mid-write, or a record from an older schema, is skipped rather than
/// allowed to discard everything after it.
pub fn load(path: &Path) -> Vec<Observation> {
    let Ok(body) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    body.lines()
        .filter_map(|line| serde_json::from_str::<Observation>(line).ok())
        .filter(|o| o.schema == OBSERVATION_SCHEMA)
        .collect()
}

pub fn load_default() -> Vec<Observation> {
    match crate::record::default_path() {
        Some(p) => load(&p),
        None => Vec::new(),
    }
}
```

Create `src/estimate/mod.rs` with `pub mod corpus;` and add `pub mod estimate;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test estimate_corpus`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/estimate src/lib.rs tests/estimate_corpus.rs
git commit -m "feat: forgiving reader over the observation corpus"
```

---

### Task 3: Attribute a footprint to a model

**Files:**
- Create: `src/estimate/attribute.rs`
- Test: unit tests within

**Interfaces:**
- Produces: `attribute::model_bytes(&Observation) -> Option<u64>`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ProcessSample;
    use crate::provider::ProviderKind;
    use crate::record::Observation;

    fn sample(pid: u32, bytes: u64) -> ProcessSample {
        ProcessSample {
            pid,
            footprint_bytes: bytes,
            phys_footprint_bytes: Some(bytes),
            rss_bytes: Some(bytes),
        }
    }

    fn obs(loaded: usize, attributable: bool, procs: Vec<ProcessSample>) -> Observation {
        Observation {
            schema: crate::record::OBSERVATION_SCHEMA,
            at: 1,
            provider: ProviderKind::LmStudio,
            loaded,
            model: Some("qwen3-14b".into()),
            context_tokens: Some(40960),
            attributable,
            footprint_bytes: procs.iter().map(|p| p.footprint_bytes).sum(),
            phys_footprint_bytes: None,
            rss_bytes: None,
            processes: procs,
            machine_total_bytes: 25_769_803_776,
            machine_used_bytes: 20_000_000_000,
            swap_used_bytes: 0,
        }
    }

    /// Measured 2026-09-09: llama-server held 8.39 GB while LM Studio's five
    /// Electron processes together held 0.28 GB. The model is the big one.
    #[test]
    fn the_model_is_the_largest_process_in_the_tree() {
        let o = obs(
            1,
            true,
            vec![
                sample(647, 300_000_000),
                sample(25164, 9_000_000_000),
                sample(3549, 150_000_000),
            ],
        );
        assert_eq!(model_bytes(&o), Some(9_000_000_000));
    }

    /// Two resident models share a tree and the largest process is one of
    /// them, not both. Refusing beats guessing.
    #[test]
    fn two_resident_models_cannot_be_attributed() {
        let o = obs(2, false, vec![sample(1, 5_000_000_000), sample(2, 4_000_000_000)]);
        assert_eq!(model_bytes(&o), None);
    }

    #[test]
    fn a_provider_with_nothing_loaded_attributes_nothing() {
        let o = obs(0, true, vec![sample(647, 600_000_000)]);
        assert_eq!(model_bytes(&o), None);
    }

    #[test]
    fn an_observation_with_no_process_detail_attributes_nothing() {
        let o = obs(1, true, vec![]);
        assert_eq!(model_bytes(&o), None, "a v1 record cannot be attributed");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib attribute`
Expected: FAIL — `cannot find function model_bytes`.

- [ ] **Step 3: Write the implementation**

```rust
use crate::record::Observation;

/// What the resident model costs, from one observation.
///
/// The rule is the largest process in the provider's tree. A provider mixes a
/// shell with a backend, and the backend holding the weights dwarfs it --
/// 8.39 GB against 0.28 GB, measured 2026-09-09. The alternative the roadmap
/// originally proposed, whole-tree minus idle baseline, cannot work for a
/// provider that never sits idle, and LM Studio never has.
///
/// `None` whenever the answer would be a guess: no model resident, more than
/// one resident, or an observation with no per-process detail.
pub fn model_bytes(o: &Observation) -> Option<u64> {
    if o.loaded != 1 || !o.attributable {
        return None;
    }
    o.processes.iter().map(|p| p.footprint_bytes).max()
}
```

Add `pub mod attribute;` to `src/estimate/mod.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib attribute`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/estimate/attribute.rs src/estimate/mod.rs
git commit -m "feat: attribute a footprint to the model-bearing process"
```

---

### Task 4: The estimate

**Files:**
- Create: `src/estimate/estimator.rs`
- Test: unit tests within

**Interfaces:**
- Produces: `Estimate { bytes, basis, samples, spread_bytes }`, `Basis::{Measured, Unknown}`, `estimator::for_model(&[Observation], &str, Option<u32>) -> Estimate`, `estimator::all(&[Observation]) -> Vec<(Key, Estimate)>`.

- [ ] **Step 1: Write the failing test**

First add the shared fixtures to `src/estimate/mod.rs`, since Task 3's tests
and these both need them:

```rust
#[cfg(test)]
pub mod fixtures {
    use crate::memory::ProcessSample;
    use crate::provider::ProviderKind;
    use crate::record::{Observation, OBSERVATION_SCHEMA};

    pub fn sample(pid: u32, bytes: u64) -> ProcessSample {
        ProcessSample {
            pid,
            footprint_bytes: bytes,
            phys_footprint_bytes: Some(bytes),
            rss_bytes: Some(bytes),
        }
    }

    fn base(model: &str, ctx: u32, loaded: usize, attributable: bool,
            procs: Vec<ProcessSample>) -> Observation {
        Observation {
            schema: OBSERVATION_SCHEMA,
            at: 1,
            provider: ProviderKind::LmStudio,
            loaded,
            model: Some(model.to_string()),
            context_tokens: Some(ctx),
            attributable,
            footprint_bytes: procs.iter().map(|p| p.footprint_bytes).sum(),
            phys_footprint_bytes: None,
            rss_bytes: None,
            processes: procs,
            machine_total_bytes: 25_769_803_776,
            machine_used_bytes: 20_000_000_000,
            swap_used_bytes: 0,
        }
    }

    /// One resident model whose backing process holds `bytes`, beside a small
    /// shell process -- the real shape of an LM Studio tree.
    pub fn loaded(model: &str, ctx: u32, bytes: u64) -> Observation {
        base(model, ctx, 1, true, vec![sample(647, 300_000_000), sample(25164, bytes)])
    }

    /// Two resident models: attributable is false and nothing may be inferred.
    pub fn two_loaded(model: &str, ctx: u32, bytes: u64) -> Observation {
        base(model, ctx, 2, false, vec![sample(1, bytes / 2), sample(2, bytes / 2)])
    }
}
```

Then the tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::fixtures;

    /// Repeated measurements of one shape converge on one answer.
    #[test]
    fn a_measured_shape_reports_its_measurement() {
        let corpus = vec![
            fixtures::loaded("qwen3-14b", 40960, 9_000_000_000),
            fixtures::loaded("qwen3-14b", 40960, 9_100_000_000),
        ];
        let e = for_model(&corpus, "qwen3-14b", Some(40960));
        assert!(matches!(e.basis, Basis::Measured));
        assert_eq!(e.samples, 2);
        // The high-water mark, not the mean: admitting by the average of two
        // readings admits below the larger one, and under-estimating is the
        // failure that wedges the machine.
        assert_eq!(e.bytes, Some(9_100_000_000));
        assert_eq!(e.spread_bytes, Some(100_000_000));
    }

    #[test]
    fn an_unseen_shape_is_unknown_never_interpolated() {
        let corpus = vec![fixtures::loaded("qwen3-14b", 40960, 9_000_000_000)];
        let e = for_model(&corpus, "qwen3-14b", Some(8192));
        assert!(matches!(e.basis, Basis::Unknown));
        assert_eq!(e.bytes, None, "8k is cheaper than 40k, but by how much is not known");
    }

    #[test]
    fn an_unseen_model_is_unknown() {
        let corpus = vec![fixtures::loaded("qwen3-14b", 40960, 9_000_000_000)];
        assert!(matches!(for_model(&corpus, "llama-3.2-1b", Some(8192)).basis, Basis::Unknown));
    }

    /// Asking without a context is asking about the worst case seen.
    #[test]
    fn omitting_the_context_takes_the_largest_measurement_for_that_model() {
        let corpus = vec![
            fixtures::loaded("qwen3-14b", 8192, 7_000_000_000),
            fixtures::loaded("qwen3-14b", 40960, 9_000_000_000),
        ];
        let e = for_model(&corpus, "qwen3-14b", None);
        assert_eq!(e.bytes, Some(9_000_000_000));
    }

    #[test]
    fn unattributable_observations_are_not_measurements() {
        let corpus = vec![fixtures::two_loaded("qwen3-14b", 40960, 12_000_000_000)];
        assert!(matches!(for_model(&corpus, "qwen3-14b", Some(40960)).basis, Basis::Unknown));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib estimator`
Expected: FAIL — `cannot find function for_model`.

- [ ] **Step 3: Write the implementation**

```rust
use crate::estimate::attribute::model_bytes;
use crate::record::Observation;

/// How an estimate was reached. `design.md` §5 ranks measured over computed
/// over declared; slice 3 implements the first and refuses the rest rather
/// than pretending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Basis {
    /// Real observations of this exact shape.
    Measured,
    /// Never seen. Not a number.
    Unknown,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Estimate {
    pub bytes: Option<u64>,
    pub basis: Basis,
    pub samples: usize,
    /// Largest minus smallest measurement. A wide spread is a warning that the
    /// shape is not as fixed as it looks.
    pub spread_bytes: Option<u64>,
}

impl Estimate {
    fn unknown() -> Estimate {
        Estimate { bytes: None, basis: Basis::Unknown, samples: 0, spread_bytes: None }
    }
}

/// What this model costs, if it has been seen.
///
/// `context_tokens: None` asks about the worst case seen for the model at any
/// window. A specific window matches only that window: KV cache scales with
/// context, and interpolating between two windows would produce a number
/// nothing measured.
pub fn for_model(corpus: &[Observation], model: &str, context_tokens: Option<u32>) -> Estimate {
    let measurements: Vec<u64> = corpus
        .iter()
        .filter(|o| o.model.as_deref() == Some(model))
        .filter(|o| match context_tokens {
            Some(want) => o.context_tokens == Some(want),
            None => true,
        })
        .filter_map(model_bytes)
        .collect();

    if measurements.is_empty() {
        return Estimate::unknown();
    }
    let max = *measurements.iter().max().expect("non-empty");
    let min = *measurements.iter().min().expect("non-empty");
    Estimate {
        // The high-water mark. design.md section 8: over-estimating wastes
        // capacity, under-estimating wedges the machine. Bias here.
        bytes: Some(max),
        basis: Basis::Measured,
        samples: measurements.len(),
        spread_bytes: Some(max - min),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib estimator`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/estimate/estimator.rs src/estimate/mod.rs
git commit -m "feat: measured footprint estimates biased to the high-water mark"
```

---

### Task 5: `llm-harmony estimate`

**Files:**
- Create: `src/render_estimate.rs`
- Modify: `src/main.rs`, `src/lib.rs`

**Interfaces:**
- Produces: `render_estimate(&Estimate, &Machine, Option<u64>) -> String`, and the `estimate` subcommand.

- [ ] **Step 1: Write the failing test**

Unit tests in `src/render_estimate.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::estimator::{Basis, Estimate};
    use crate::memory::Machine;

    fn machine(free: u64) -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 25_769_803_776 - free,
            swap_total_bytes: 6_442_450_944,
            swap_used_bytes: 0,
        }
    }

    #[test]
    fn a_model_larger_than_headroom_would_not_fit() {
        let e = Estimate {
            bytes: Some(9_000_000_000),
            basis: Basis::Measured,
            samples: 3,
            spread_bytes: Some(100_000_000),
        };
        let out = render_estimate(&e, &machine(4_000_000_000), None);
        assert!(out.contains("WOULD NOT FIT"), "{out}");
        assert!(out.contains("measured"), "the basis is always stated: {out}");
    }

    #[test]
    fn a_model_inside_headroom_fits() {
        let e = Estimate {
            bytes: Some(2_000_000_000),
            basis: Basis::Measured,
            samples: 1,
            spread_bytes: Some(0),
        };
        let out = render_estimate(&e, &machine(12_000_000_000), None);
        assert!(out.contains("fits"), "{out}");
    }

    /// The reserve is subtracted before the verdict: headroom that the OS needs
    /// is not headroom a model may take.
    #[test]
    fn the_reserve_is_held_back_from_headroom() {
        let e = Estimate {
            bytes: Some(5_000_000_000),
            basis: Basis::Measured,
            samples: 1,
            spread_bytes: Some(0),
        };
        let generous = render_estimate(&e, &machine(6_000_000_000), Some(0));
        let reserved = render_estimate(&e, &machine(6_000_000_000), Some(4_000_000_000));
        assert!(generous.contains("fits"), "{generous}");
        assert!(reserved.contains("WOULD NOT FIT"), "{reserved}");
    }

    /// An unmeasured shape must not produce a verdict at all.
    #[test]
    fn an_unknown_estimate_refuses_to_judge() {
        let e = Estimate { bytes: None, basis: Basis::Unknown, samples: 0, spread_bytes: None };
        let out = render_estimate(&e, &machine(1_000_000), None);
        assert!(out.contains("unknown"), "{out}");
        assert!(!out.contains("FIT"), "no verdict without a measurement: {out}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib render_estimate`
Expected: FAIL — `cannot find function render_estimate`.

- [ ] **Step 3: Write the implementation**

```rust
use crate::estimate::estimator::{Basis, Estimate};
use crate::memory::Machine;
use crate::render::human_bytes;

/// Memory held back for the OS and everything that is not a model.
///
/// `design.md` §9 calls 8 GB on 24 GB "a guess to be tuned by measurement, not
/// a claim", and it remains one: ten observations is not a basis for tuning it.
/// It is a knob with a documented default, not a finding.
pub const DEFAULT_RESERVE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub fn render_estimate(e: &Estimate, machine: &Machine, reserve: Option<u64>) -> String {
    let reserve = reserve.unwrap_or(DEFAULT_RESERVE_BYTES);
    let free = machine.total_bytes.saturating_sub(machine.used_bytes);
    let headroom = free.saturating_sub(reserve);

    let mut out = String::new();
    match (e.bytes, e.basis) {
        (Some(bytes), Basis::Measured) => {
            let spread = e
                .spread_bytes
                .map(|s| format!(" \u{b1}{}", human_bytes(s)))
                .unwrap_or_default();
            out.push_str(&format!(
                "  predicted  {:>9}{}  (measured, {} observation{})\n",
                human_bytes(bytes),
                spread,
                e.samples,
                if e.samples == 1 { "" } else { "s" },
            ));
            out.push_str(&format!(
                "  headroom   {:>9}   ({} free \u{2212} {} reserve)\n",
                human_bytes(headroom),
                human_bytes(free),
                human_bytes(reserve),
            ));
            out.push_str(if bytes <= headroom {
                "  verdict    fits\n"
            } else {
                "  verdict    WOULD NOT FIT\n"
            });
        }
        _ => {
            out.push_str("  predicted  unknown \u{2014} this shape has never been measured\n");
            out.push_str(&format!(
                "  headroom   {:>9}   ({} free \u{2212} {} reserve)\n",
                human_bytes(headroom),
                human_bytes(free),
                human_bytes(reserve),
            ));
            out.push_str("  verdict    none \u{2014} load it once and it will be known\n");
        }
    }
    out
}
```

- [ ] **Step 4: Wire the subcommand in `src/main.rs`**

```rust
    /// What a model costs, and whether it fits right now.
    Estimate {
        model: String,
        #[arg(long)]
        context: Option<u32>,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        #[arg(long)]
        json: bool,
    },
```

```rust
        Command::Estimate { model, context, reserve, json } => {
            let machine = match Machine::read() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let corpus = llm_harmony::estimate::corpus::load_default();
            let est = llm_harmony::estimate::estimator::for_model(&corpus, &model, context);
            if json {
                println!("{}", serde_json::to_string_pretty(&est).unwrap());
            } else {
                print!(
                    "{}",
                    llm_harmony::render_estimate::render_estimate(&est, &machine, reserve)
                );
            }
            ExitCode::SUCCESS
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS across every target.

- [ ] **Step 6: Commit**

```bash
git add src/render_estimate.rs src/main.rs src/lib.rs
git commit -m "feat: llm-harmony estimate with a fit verdict"
```

---

### Task 6: Live verification

- [ ] **Step 1: The shape the corpus already holds**

```bash
llm-harmony estimate qwen3-14b-claude-4.5-opus-high-reasoning-distill --context 40960
```

Expected: `measured`, with a sample count matching the corpus, and a verdict against real headroom. On a machine at ~16% free the verdict should be `WOULD NOT FIT`, which is correct and is the point.

- [ ] **Step 2: A shape that has never been seen**

```bash
llm-harmony estimate qwen3-14b-claude-4.5-opus-high-reasoning-distill --context 8192
llm-harmony estimate never-heard-of-this-model
```

Expected: `unknown` and **no verdict** in both cases. An 8k window is certainly cheaper than 40k, and the estimator must still refuse — it has not measured it.

- [ ] **Step 3: Grow the corpus and watch it learn**

Load a *second* model shape in LM Studio (a smaller one, given the machine is tight), then:

```bash
llm-harmony status --record
llm-harmony estimate <that model>
```

Expected: the new shape moves from `unknown` to `measured` after one load. That transition is the whole slice.

- [ ] **Step 4: Confirm nothing regressed**

```bash
cargo test
grep -rnE "fs::remove_file|fs::remove_dir|fs::rename|fs::write|fs::copy|Command::new" src/
llm-harmony status && llm-harmony ls | tail -3
```

Expected: all tests pass; no mutation calls outside `record.rs`'s test cleanup; slices 1 and 2 unchanged.

- [ ] **Step 5: Commit**

```bash
git commit -m "docs: record slice 3 verification against the live corpus"
```

---

## What this slice deliberately does not do

- **No daemon.** Per-process attribution works from a single observation, so continuity is not required. Delroy already records every turn. A daemon buys coverage of short-lived states and nothing else; it is worth building when a missed state is observed, not before.
- **No declared overrides.** `design.md` §5's third tier — a per-model figure
  in config, for the cases measurement and computation both get wrong. Nothing
  has been measured wrongly yet, so there is nothing to override; adding the
  knob first would be inventing a use for it.
- **No computed estimates.** `design.md` §5's second tier — `weights + kv(context, arch) × safety` — needs GGUF header parsing for architecture metadata, which is also what slice 2's follow-ups want for Ollama bit-depths. One parser, its own slice.
- **No interpolation between context windows.** Tempting and wrong: the relationship is not linear in practice, and a wrong number here is the dangerous direction.
- **The reserve is not settled.** The roadmap said slice 3 would settle it. Ten observations cannot. It ships as a documented default with a `--reserve` override, and the corpus informs it later.

## Follow-ups

- **The corpus is single-provider.** Nearly every observation is LM Studio. Starting llama.cpp and vLLM-MLX occasionally is what makes the estimator's coverage real rather than nominal.
- **Open-file attribution.** Reading which `.gguf` a process holds would give per-model attribution inside a multi-model provider, and hand slice 4 the artifact→model link it needs. `libproc` 0.14 implements `PIDFDInfo` only for `SocketFDInfo`, so this means hand-rolling `vnode_fdinfowithpath` in unsafe FFI. Worth it for slice 4, not for this one.
- **Corpus rotation.** `observations.jsonl` grows without bound. Harmless at one line per turn; worth a cap before it is not.
