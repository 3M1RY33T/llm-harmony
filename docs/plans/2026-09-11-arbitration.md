# llm-harmony Slice 6 — Arbitration and set admission

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** four commands that exist only because four providers are read through one ledger and one identity — `compare` (every candidate, priced), `lease` (a pin that expires), `fit` (can these be resident at once), `status --budgets` (what each provider is allowed to take).

**Architecture:** composition. No new adapter, no new poll, no new state file. `compare` prices each candidate through the *same* `estimate_for` the resolver uses and marks the pick by calling the *same* `decide`. `lease` adds two defaulted fields to `Pin` and one time-aware accessor, consulted at the single place eviction already consults pins. `fit` adds a per-provider baseline term derived from the corpus, a per-provider ceiling read from wherever that ceiling actually lives, and an exhaustive search over assignments.

**Tech Stack:** same crate, **no new dependencies.** `sysinfo` 0.39 is already a dependency and already exposes `Process::cmd()` and `Process::environ()` behind `ProcessRefreshKind::with_cmd()` / `with_environ()`, which is everything the ceiling reader needs.

**Spec:** [`../roadmap.md`](../roadmap.md) slice 6, [`../architecture.md`](../architecture.md) §10 Q5, [`../design.md`](../design.md) §5 and §6.

---

## What the probe changed

Probed live 2026-09-11 with all four providers up. The roadmap sketched four
rows; two of them were wrong, and both were wrong in ways worth keeping.

### 1. The four ceilings do not sum

The roadmap row read *"do the four per-provider ceilings sum to more than the
machine? — config that is already read."* **Both halves are wrong.**

Harmony's config carries no ceiling at all, and `ProviderConfig::start` says
why: every flag that matters "lives in the user's own script … reproducing them
here would be a second, worse source of truth." So each ceiling has to be read
from where it actually is:

| Provider | Ceiling, on this machine | Where it lives | Unit |
|---|---|---|---|
| vLLM-MLX | `memory_budget_gb: 14.0` | `/v1/status` → `model_manager` | **bytes** |
| llama.cpp | `--models-max 1` | process argv | count |
| Ollama | `OLLAMA_MAX_LOADED_MODELS` **not set** | process environ | count, **unknown** |
| LM Studio | unbounded: JIT on, idle TTL off | `~/.lmstudio/.internal/http-server-config.json` and `model-data.json` | policy, not a ceiling |

`gb` in vLLM-MLX's figure is read as **GiB**, the larger of the two readings.
The project's bias inverts for a ceiling: over-estimating a cost is safe, while
over-estimating a ceiling under-states what the provider may take.

Verbatim from the probe:

```
llama-server --models-dir ~/.llamacpp/models --models-max 1 --jinja --port 8080
/v1/status → {"model_manager":{"memory_budget_gb":14.0, …}}
ollama environ → OLLAMA_MODELS, OLLAMA_NO_CLOUD          (no MAX_LOADED_MODELS)
http-server-config.json → "justInTimeModelLoading": true
model-data.json → "uiTTL":{"enabled":false,"ttlSeconds":3600}   per model
```

There is no sum to print. One ceiling is in bytes, two are counts, one of those
counts is Ollama's own undocumented default — which harmony must not guess, by
the same rule that makes a footprint *unknown* rather than invented — and the
fourth provider has no ceiling in either unit. `--budgets` therefore prints what
each provider is allowed, with `?` where it cannot be known (the convention
`status` already uses for `weights` and `gap`), and one verdict line:

> the only byte ceiling on this machine is vLLM-MLX's **14.0 G of 24.0 G**, and
> nothing bounds the other three in bytes at all.

That is a better feature than the sum was. It is the README's opening
screenshot explained rather than restated: LM Studio held 8.9 G because nothing
told it not to.

### 2. `--budgets` is not a nicety — it is `fit`'s second admission input

llama.cpp runs `--models-max 1`. **A set that fits the machine in bytes can
still be impossible**, because no arithmetic over memory will get two GGUFs
into a router that holds one. So the ceiling reader is a dependency of `fit`,
not a separate cheap command, and the roadmap's ordering moves: ceilings before
`fit`, everything else in the stated order.

Where a count ceiling is *unknown* (Ollama), `fit` **marks and proceeds** rather
than refusing. The asymmetry that justifies refusing on memory does not apply
here: exceeding a count ceiling makes the provider evict one of its own models
by its own policy, which is a surprise, not a wedged machine.

### 3. A lease is a pin with an expiry, and the store already tolerates it

`src/pins.rs` keys on `(provider, model)`, saves atomically, and carries two
compatibility tests — `a_document_without_a_schema_field_still_loads` and
`an_unknown_field_does_not_discard_the_store` — which is exactly the licence to
add defaulted fields. Eviction consults the store in **one** place,
`actuate::plan::victims_for` (`src/actuate/plan.rs:300`), and already returns
`pinned_blockers` so a refusal can name its cause.

So: `Pin` gains `owner` and `expires_at`, both `#[serde(default)]`, and **a pin
is a lease with no expiry.** Six construction sites move:
`src/render.rs:284`, `src/main.rs:548`, `src/actuate/plan.rs:400`,
`src/actuate/run.rs:483`, `tests/pins.rs:14`, `tests/pins.rs:93`.

### 4. The baseline is measured, and the maximum is the wrong statistic

Answered in [`../architecture.md`](../architecture.md) §10 Q5 from ~800 recorded
readings: p99 per provider — LM Studio 637 MiB, Ollama 28 MiB, llama.cpp 22 MiB,
vLLM-MLX 332 MiB on two samples — **≈1.0 GiB with all four idle.** The maximum
was rejected because it drifts: LM Studio's grew 637 → 788 MiB inside an hour on
a single sample while p95 held at 571 MiB.

---

## Global Constraints

- **No new authority.** Slice 5 can already evict. This slice adds nothing that
  can: `lease` *restricts* eviction, and `compare`, `fit` and `--budgets` print.
  The structural check in Task 7 is that no new unload path exists.
- **One pricing path.** `compare` and `fit` call `estimate_for` — the helper
  `estimate` and `resolve` already share — never a second implementation. Pass a
  one-element slice to price a single candidate; the signature already allows it.
- **One preference order.** `compare` marks which candidate `resolve` would pick
  by calling `decide`, not by re-deriving the ranking. A menu that disagreed
  with the resolver would be worse than no menu.
- **Unknown is printed, never guessed.** A ceiling that cannot be read is `?`.
  A model with no measurement is `unknown`, and a set containing one is an
  unpriced set.
- **A refusal names its cause.** `pins.rs` already requires that of a pin. A
  lease refusal additionally names the **owner** and the time remaining, because
  "something invisible is holding memory you can see" is the failure mode.
- **Set cost is not the sum of model costs.** Each provider's baseline is
  counted once, and **anything already resident is not charged at all** — it is
  in `machine.used_bytes` already, and `decide` has always treated a resident
  model as needing no admission.
- **Seconds, not a duration grammar.** `lease --ttl` takes `u64` seconds like
  `load --ttl` does. The project has no duration parser and this is not the
  place to introduce its only one.
- **Commit messages are subject lines only.** The `feat:`-prefixed examples in
  the slice 1–5 plans are superseded by the standing rule of 2026-09-11:
  lowercase, behaviour-first, no prefix, no body.
- macOS only, as slices 1–5.

---

### Task 1: `compare` — every candidate, priced

**Files:** create `src/render_compare.rs`; `src/main.rs`, `src/lib.rs`
**Interfaces:** `render_compare::render(&[Priced], &Machine, u64, &Decision) -> String`, `render_compare::render_json(…) -> String`, `struct Priced { candidate: Candidate, estimate: Estimate }`

- [ ] **Step 1: Write the failing tests**

In-module tests, as `src/render_estimate.rs` does:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the command: `resolve` computes this list and throws
    /// all but one away.
    #[test]
    fn every_candidate_is_listed_not_only_the_chosen_one() {
        let out = render(&[priced(LlamaCpp, 8_400_000_000, Basis::Measured),
                           priced(Vllm,     4_700_000_000, Basis::Declared)],
                         &machine(13 << 30), RESERVE, &ready(LlamaCpp));
        assert!(out.contains("llamacpp"));
        assert!(out.contains("vllm"), "the unchosen candidate is the point: {out}");
    }

    /// A price is worthless without its rung. design.md §5.
    #[test]
    fn each_candidate_carries_its_own_basis() {
        let out = render(&[priced(LlamaCpp, 8_400_000_000, Basis::Measured),
                           priced(Vllm,     4_700_000_000, Basis::Declared)],
                         &machine(13 << 30), RESERVE, &ready(LlamaCpp));
        assert!(out.contains("measured"));
        assert!(out.contains("declared"));
    }

    /// A candidate that cannot be admitted is still information.
    #[test]
    fn a_candidate_that_does_not_fit_is_marked_rather_than_dropped() {
        let out = render(&[priced(LlamaCpp, 20 << 30, Basis::Measured)],
                         &machine(9 << 30), RESERVE, &deny());
        assert!(out.contains("llamacpp"));
        assert!(out.contains("does not fit"), "{out}");
    }

    /// Slice 2's rule: an inferred edge may not silently drive a decision.
    #[test]
    fn an_inferred_candidate_says_so() {
        let out = render(&[inferred(LmStudio, 8_900_000_000)],
                         &machine(13 << 30), RESERVE, &ready(LmStudio));
        assert!(out.contains("inferred"), "{out}");
    }

    /// The invariant that keeps the menu honest: the marked row is whatever
    /// `decide` returned, never a ranking re-derived here.
    #[test]
    fn the_marked_candidate_is_the_one_resolve_would_pick() {
        let out = render(&[priced(LlamaCpp, 8_400_000_000, Basis::Measured),
                           priced(Vllm,     4_700_000_000, Basis::Measured)],
                         &machine(13 << 30), RESERVE, &ready(Vllm));
        let marked = out.lines().find(|l| l.trim_start().starts_with('>')).expect("a marked row");
        assert!(marked.contains("vllm"), "cheaper is not the rule; decide is: {out}");
    }

    #[test]
    fn a_model_nothing_serves_says_so_rather_than_printing_an_empty_table() { /* … */ }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib render_compare`
Expected: FAIL — no module `render_compare`.

- [ ] **Step 3: Implement the renderer**

Shape it on `render_resolve.rs`, which already computes
`headroom = total − used − reserve`:

```
$ llm-harmony compare qwen3.5-9b-ultra-uncensored-heretic
  provider   model id                             price    basis      fit
> llamacpp   Qwen3.5-9B-ultra-...-heretic-Q4_K_M   5.2G    measured   fits      (resident)
  vllm       Qwen3.5-9B-ultra-...-heretic-MLX-4bit 4.7G    declared   fits      (weights only)
  vllm       Qwen3.5-9B-ultra-...-heretic-MLX-6bit 6.8G    declared   fits      (weights only)
  headroom   13.1G  (21.1G free − 8.0G reserve)
  > is where resolve would send this request
```

- [ ] **Step 4: Wire the command**

`src/main.rs`: assemble as `Command::Resolve` does —
`Ledger::assemble`, `Inventory::scan_offline(None)`,
`identity::candidates`, then `estimate_for(…, &[c.clone()], …)` per candidate
and one `decide` over the whole list. `--json` carries its own `schema`,
separate from the ledger's. Exit 0 always: `compare` reports, it does not admit.

- [ ] **Step 5: Run tests, then commit**

```bash
cargo test --lib render_compare && cargo test
git add src/render_compare.rs src/main.rs src/lib.rs
git commit -m "price every provider that could serve a model, and mark the one resolve would pick"
```

---

### Task 2: `lease` — a pin that expires

**Files:** `src/pins.rs`, `src/actuate/plan.rs`, `src/actuate/run.rs`, `src/render.rs`, `src/main.rs`, `tests/pins.rs`
**Interfaces:** `Pin { owner: Option<String>, expires_at: Option<u64> }`, `Pins::holds(ProviderKind, &str, now: u64) -> Option<&Pin>`, `Pins::sweep(now: u64) -> usize`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pins.rs`:

```rust
fn lease(model: &str, expires_at: u64, owner: &str) -> Pin {
    Pin {
        provider: ProviderKind::LmStudio,
        model: model.to_string(),
        at: 1_788_900_000,
        note: None,
        owner: Some(owner.to_string()),
        expires_at: Some(expires_at),
    }
}

/// The gap slice 5 left: a caller still using a model has no way to say so
/// except a manual pin that outlives its reason.
#[test]
fn a_lease_protects_until_it_expires_and_then_stops() {
    let mut p = Pins::empty();
    p.add(lease("qwen3-14b", 1_788_903_600, "delroy"));
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", 1_788_903_599).is_some());
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", 1_788_903_601).is_none());
}

/// A pin is a lease with no expiry, and that is the only difference.
#[test]
fn a_pin_has_no_expiry_and_protects_at_any_time() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", u64::MAX).is_some());
}

/// An expired lease is not an error and not a refusal; it is simply not a
/// veto any more. Sweeping is housekeeping, not policy.
#[test]
fn an_expired_lease_is_dropped_by_a_sweep_and_a_pin_is_not() {
    let mut p = Pins::empty();
    p.add(lease("expired", 100, "delroy"));
    p.add(pin("forever"));
    assert_eq!(p.sweep(1_000), 1);
    assert_eq!(p.all().len(), 1);
}

/// Protection must never weaken by accident: a lease over an existing pin is
/// refused, exactly as a second pin is.
#[test]
fn a_lease_cannot_downgrade_an_existing_pin() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(!p.add(lease("qwen3-14b", 1_788_903_600, "delroy")));
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", u64::MAX).is_some());
}

/// The store predates both fields. Losing every pin on upgrade would be a
/// poor trade for strictness -- the rule pins.rs already states.
#[test]
fn a_store_written_before_leases_existed_still_loads() {
    let p: Pins = serde_json::from_str(
        r#"{"schema":1,"pins":[{"provider":"ollama","model":"m","at":1}]}"#,
    ).unwrap();
    assert!(p.holds(ProviderKind::Ollama, "m", u64::MAX).is_some());
}
```

And in `src/actuate/plan.rs`'s test module:

```rust
/// "Something invisible is holding memory you can see" is the failure mode, so
/// a lease refusal names the owner and what is left of it.
#[test]
fn a_refusal_caused_by_a_lease_names_its_owner_and_remaining_time() { /* … */ }

/// Expiry returns a model to the eviction pool. It does not cause an eviction:
/// a lease is a veto, never a reservation.
#[test]
fn an_expired_lease_makes_its_model_a_candidate_again() { /* … */ }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test pins`
Expected: FAIL — no field `owner`, no method `holds`.

- [ ] **Step 3: Extend the store**

```rust
    /// Who asked for this protection, when anyone did.
    ///
    /// A pin set by hand has none. A lease taken by a client has one, and a
    /// refusal names it: the operator has to be able to find out who is
    /// holding the memory without reading a state file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Unix seconds after which this stops protecting anything.
    ///
    /// `None` is a pin: protection with no end, cleared only by `unpin`. The
    /// two are the same mechanism because eviction has exactly one place to
    /// ask, and two stores would be two answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
```

`holds(provider, model, now)` replaces `is_pinned` at the planner's single
consult point; `is_pinned` stays for `status` rendering, expressed as
`holds(p, m, now).is_some()`. `now` is a parameter, never read inside — the
convention `record.rs` states ("passed in rather than read, so this is
testable").

- [ ] **Step 4: Wire the commands**

```
llm-harmony lease <model> --ttl <SECONDS> [--owner NAME] [--provider P] [--note ..]
llm-harmony release <model> [--provider P]
```

`--ttl` here is **harmony's** lease, unrelated to `load --ttl`, which is the
provider's own idle timer — the help text must say so, because the two flags
share a name and nothing else. Leases appear on the `status` `pinned` line with
their remaining time, and a `sweep` runs on every save.

- [ ] **Step 5: Run tests, then commit**

```bash
cargo test --test pins && cargo test
git add src/pins.rs src/actuate src/render.rs src/main.rs tests/pins.rs
git commit -m "let a caller hold a model for a bounded time, and name the holder in every refusal it causes"
```

---

### Task 3: The per-provider idle baseline

**Files:** create `src/estimate/baseline.rs`; `src/estimate/mod.rs`
**Interfaces:** `baseline::from_corpus(&[Observation], ProviderKind) -> Option<Baseline>`, `struct Baseline { bytes: u64, p50_bytes: u64, samples: usize, thin: bool }`

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Only a reading with nothing loaded prices the provider itself.
    #[test]
    fn a_reading_with_a_model_loaded_is_not_a_baseline() {
        let c = vec![obs(LmStudio, 1, 9 << 30), obs(LmStudio, 0, 560 << 20)];
        assert_eq!(from_corpus(&c, LmStudio).unwrap().samples, 1);
    }

    /// Measured 2026-09-11: LM Studio's largest idle reading moved 637 -> 788
    /// MiB inside an hour, on one sample, while p95 held at 571. The maximum
    /// is a tail that grows with the corpus; p99 is a statistic.
    #[test]
    fn the_figure_is_p99_and_not_the_maximum() {
        let mut c: Vec<_> = (0..99).map(|_| obs(LmStudio, 0, 560 << 20)).collect();
        c.push(obs(LmStudio, 0, 788 << 20));
        let b = from_corpus(&c, LmStudio).unwrap();
        assert!(b.bytes < 788 << 20, "the lone spike must not set the figure");
    }

    /// vLLM-MLX has two idle readings and that is structural: with no
    /// model-level unload it is idle only between start and first request.
    /// A figure resting on two samples has to say so.
    #[test]
    fn a_provider_with_too_few_idle_readings_is_marked_thin() {
        let c = vec![obs(Vllm, 0, 332 << 20), obs(Vllm, 0, 331 << 20)];
        assert!(from_corpus(&c, Vllm).unwrap().thin);
    }

    #[test]
    fn a_provider_never_observed_idle_has_no_baseline() {
        assert!(from_corpus(&[obs(Vllm, 1, 5 << 30)], Vllm).is_none());
    }

    /// An observation harmony could not attribute is not evidence.
    #[test]
    fn an_unattributable_reading_is_skipped() { /* … */ }
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test --lib baseline`

- [ ] **Step 3–4: Implement, run, commit**

p99 over `loaded == 0 && attributable`, `thin` below 10 samples. No corpus
filtering by date: the corpus is append-only and a baseline is not a footprint —
it does not move with the model.

```bash
git commit -m "derive each provider's idle baseline from the corpus at p99, and mark a figure resting on too few readings"
```

---

### Task 4: Read each provider's own ceiling, and `status --budgets`

**Files:** create `src/budget.rs`; `src/render.rs`, `src/main.rs`
**Interfaces:** `budget::ceiling(kind, &ProviderConfig, &Ledger, &System) -> Ceiling`, `enum Ceiling { Bytes(u64), Count(u32), Unbounded, Unknown { why: String } }`

- [ ] **Step 1: Write the failing tests**

Fixture-driven, as the adapter tests are — the argv and environ shapes below
are the ones captured live on 2026-09-11:

```rust
/// llama.cpp's ceiling is a count and it is in the argv, because the flag
/// lives in the user's own start script and nowhere else.
#[test]
fn llamacpp_ceiling_comes_from_models_max_in_the_argv() {
    let argv = ["llama-server", "--models-dir", "/m", "--models-max", "1", "--port", "8080"];
    assert_eq!(from_argv(&argv), Ceiling::Count(1));
}

/// vLLM-MLX is the only provider that publishes a ceiling in bytes, and it
/// publishes it over the API rather than in a file.
///
/// `gb` is read as **GiB**, the larger reading. The bias inverts for a
/// ceiling: over-estimating a *cost* is safe, but over-estimating a *ceiling*
/// under-states how much the provider may take. 14.0 as GiB is 15.03e9 bytes,
/// so that is the number to hold it to until vLLM-MLX's own source says
/// otherwise.
#[test]
fn vllm_ceiling_comes_from_the_status_endpoint_and_gb_means_gib() {
    let body = fixture("vllm/v1-status.json");   // memory_budget_gb: 14.0
    assert_eq!(from_status(&body), Ceiling::Bytes(14 * 1024 * 1024 * 1024));
}

/// Measured: the variable is simply not set, so Ollama's own default applies
/// and harmony does not know it. Guessing here would be the same mistake as
/// guessing a footprint.
#[test]
fn an_unset_ollama_variable_is_unknown_and_not_a_default() {
    let env = ["OLLAMA_MODELS=/m", "OLLAMA_NO_CLOUD=0"];
    assert!(matches!(from_environ(&env), Ceiling::Unknown { .. }));
}

#[test]
fn a_set_ollama_variable_is_a_count() {
    assert_eq!(from_environ(&["OLLAMA_MAX_LOADED_MODELS=2"]), Ceiling::Count(2));
}

/// LM Studio has JIT loading on and its per-model idle TTL off, which is not
/// a ceiling in either unit. Saying "none" is the finding, not a gap.
#[test]
fn lmstudio_is_unbounded_in_both_units() {
    assert_eq!(from_lmstudio_config(fixture("lmstudio/http-server-config.json")), Ceiling::Unbounded);
}

/// A provider that is down has no process and no endpoint to ask.
#[test]
fn a_provider_that_is_not_running_has_an_unknown_ceiling() { /* … */ }
```

- [ ] **Step 2: Run to verify it fails** — `cargo test --lib budget`

- [ ] **Step 3: Implement the reader**

`sysinfo` with `ProcessRefreshKind::new().with_cmd(UpdateKind::Always)
.with_environ(UpdateKind::Always)`; find the provider's process the way slice 1
already finds it, then read `cmd()` and `environ()`. No shelling out to `ps`,
no new dependency. vLLM's figure comes from the `/v1/status` body the adapter
already fetches — do not add a second request.

- [ ] **Step 4: Render it**

```
$ llm-harmony status --budgets
provider    allowed        unit     source
lmstudio    none           —        JIT on, idle TTL off
llamacpp    1 model        count    --models-max, from the argv
ollama      ?              count    OLLAMA_MAX_LOADED_MODELS unset; its own default applies
vllm        14.0G          bytes    memory_budget_gb, from /v1/status
──────────────────────────────────────────────────
machine     24.0G total
the only byte ceiling here is vllm's 14.0G — 58% of the machine.
nothing bounds the other three in bytes.
```

No total row. The units do not add, and a sum would be the exact kind of
confident wrong number this project exists to avoid.

- [ ] **Step 5: Run tests, then commit**

```bash
cargo test --lib budget && cargo test
git add src/budget.rs src/render.rs src/main.rs tests/fixtures
git commit -m "read each provider's own ceiling from where it actually lives, and refuse to add units that do not add"
```

---

### Task 5: `fit` — set admission

**Files:** create `src/fit/mod.rs`, `src/fit/search.rs`, `src/render_fit.rs`; `src/main.rs`, `src/lib.rs`
**Interfaces:** `fit::plan(&[String], &Ledger, &Inventory, &Machine, u64, &[Observation], &Pins, u64) -> Plan`, `struct Plan { assigned: Vec<Assigned>, baselines: Vec<(ProviderKind, u64)>, charged_bytes: u64, headroom_bytes: u64, verdict: Verdict }`, `enum Verdict { Fits, Unpriced { model: String }, DoesNotFit { by_bytes: u64, smallest_drop: Vec<String> }, CeilingRefusal { provider: ProviderKind, allowed: u32 } }`

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_that_fits_names_the_assignment_it_fits_under() { /* … */ }

    /// The dimension one model does not have: the same logical model exists at
    /// several prices, so this is a selection as well as a sum.
    #[test]
    fn the_same_model_on_two_providers_is_priced_twice_and_the_cheaper_assignment_wins() { /* … */ }

    /// The arithmetic the whole feature turns on. Two models on one provider
    /// pay that provider's baseline once, not twice.
    #[test]
    fn a_providers_baseline_is_charged_once_however_many_of_its_models_are_chosen() { /* … */ }

    /// A resident model is in `used_bytes` already, and `decide` has always
    /// treated it as needing no admission. Charging it again would refuse sets
    /// that are already satisfied.
    #[test]
    fn a_model_already_resident_is_not_charged_again() { /* … */ }

    /// Same argument, one level up: a running provider's baseline is spent.
    #[test]
    fn a_running_providers_baseline_is_not_charged_again() { /* … */ }

    /// design.md §5 and decide's own rule: one unpriced member makes the set
    /// unpriced. A total that looks authoritative because two of three terms
    /// were measured is the under-estimate this project biases against.
    #[test]
    fn a_set_holding_one_unpriced_model_is_an_unpriced_set() { /* … */ }

    #[test]
    fn a_set_that_does_not_fit_names_the_smallest_drop_that_would_fit() { /* … */ }

    /// llama.cpp runs --models-max 1 here. No arithmetic over bytes gets two
    /// GGUFs into a router that holds one.
    #[test]
    fn two_models_on_a_provider_that_allows_one_is_a_ceiling_refusal_not_a_byte_refusal() { /* … */ }

    /// An unknown count ceiling is not an unknown memory cost: exceeding it
    /// makes the provider evict by its own policy. Mark it, proceed.
    #[test]
    fn an_unknown_count_ceiling_is_marked_and_does_not_refuse_the_set() { /* … */ }

    /// Exhaustive, so the rejected assignment is printable and "why not the
    /// other one" is answerable.
    #[test]
    fn the_runner_up_assignment_is_reported_with_what_it_would_have_cost() { /* … */ }

    #[test]
    fn a_pinned_or_leased_resident_is_fixed_cost_and_never_a_drop_candidate() { /* … */ }
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test --lib fit`

- [ ] **Step 3: Implement**

`charged = Σ baseline(p) for p newly needed + Σ estimate(m) for m not resident`,
against `headroom = total − used − reserve`. Search: the cartesian product of
each model's candidate list, filtered by per-provider count ceilings, minimised
on `charged`. Cap the product at a size that cannot hang (the roadmap's bound is
six models × four candidates) and **`log` the cap if it is ever hit** — a silent
truncation would read as "no assignment fits" when the truth is "we stopped
looking."

- [ ] **Step 4: Render**

```
$ llm-harmony fit qwen3-14b nomic-embed granite-3b
  qwen3-14b      llamacpp   Qwen3-14B-Q4_K_M        8.4G   measured (7)
  nomic-embed    ollama     nomic-embed-text:v1.5   0.3G   measured (2)
  granite-3b     lmstudio   granite-3.1-3b-a800m    2.1G   computed
  baselines      lmstudio 0.62G · ollama 0.03G                0.65G
  charged                                                    11.5G
  headroom       (21.1G free − 8.0G reserve)                  13.1G
  fits
  not chosen: qwen3-14b on lmstudio (8.9G, measured) — 0.5G dearer
```

`--json` with its own `schema`. Exit 0 for `fits`, 1 for every other verdict —
a caller must be able to tell from the exit code, as `resolve` already allows.

- [ ] **Step 5: Run tests, then commit**

```bash
cargo test --lib fit && cargo test
git add src/fit src/render_fit.rs src/main.rs src/lib.rs
git commit -m "answer whether a set of models can be resident at once, charging each provider's baseline once and nothing that is already loaded"
```

---

### Task 6: Live verification

The interesting behaviour is physical, as `design.md` §10 says. All four
providers up, then:

- [ ] **The cross-provider menu.** `compare qwen3.5-9b-ultra-uncensored-heretic`
  — the one group on this machine genuinely served by two providers (llama.cpp
  GGUF Q4/Q6, vLLM-MLX MLX 4/6-bit). Confirm every candidate is listed, each
  with its own basis, and that the marked row is what `resolve` returns for the
  same argument. **Run both and diff the choice.**
- [ ] **The lease, end to end.** `lease <resident model> --ttl 60 --owner test`,
  then a `load` that needs its room: expect a refusal naming `test` and the
  seconds remaining. Wait out the expiry, repeat: expect it to succeed. Confirm
  no unload happened while the lease held.
- [ ] **The lease does not reserve.** Lease a model that is *not* resident and
  confirm nothing loads. A veto, never a reservation.
- [ ] **The baselines match the record.** `fit` on a single model, and confirm
  the baseline line agrees with `architecture.md` §10 Q5 — 637 / 28 / 22 / 332
  MiB at p99 — and that vLLM-MLX is marked thin.
- [ ] **The ceiling refusal.** `fit` two GGUF models that both resolve to
  llama.cpp. Expect a ceiling refusal naming `--models-max 1`, **not** a byte
  refusal, even though the bytes fit.
- [ ] **The double-charge check.** `fit <model that is already resident>` and
  confirm `charged` is ~0 and the verdict is `fits` — the arithmetic error that
  would make the feature useless.
- [ ] **`--budgets` with a provider down.** `llm-harmony stop vllm`, then
  `status --budgets`: vLLM's ceiling becomes `?` rather than a remembered 14.0G.
  Restart it and confirm the figure returns. (This also banks vLLM-MLX idle
  readings, which is the only way its thin baseline improves.)
- [ ] **The safety property.**
  `grep -rn "unload\|lms unload\|ollama stop\|keep_alive" src/fit src/budget.rs src/render_compare.rs`
  — no eviction path in anything this slice added.
- [ ] **Fail-open.** `mv target/release/llm-harmony /tmp` and confirm all four
  providers keep serving, as slice 4's verification did.

---

## Deliberately not in this slice

- **`use` / working sets.** Actuating a planned set is a new authority over
  several models at once; `fit` plans and stops, as slices 1 and 2 did.
- **The daemon, cross-provider LRU, the standing swap guard.** Slice 7, by the
  roadmap's decision of 2026-09-11: the daemon is justified by jobs, and the
  continuous behaviours arrive as its dividend.
- **Enforcing a lease against a foreign load.** A lease binds harmony's own
  eviction. Making it bind anything else needs reconciliation speed, which
  needs the daemon.
- **Writing a provider's ceiling.** Harmony reads `--models-max`; it will not
  set it. `ProviderConfig::start` is explicit that provider flags live in the
  user's script, and `design.md` §6 is explicit that those ceilings are the
  fail-open floor.
- **Guessing Ollama's default.** `?` is the answer until the variable is set.
- **Prefix-cache-aware placement.** Refused with the request path,
  `architecture.md` §1.
