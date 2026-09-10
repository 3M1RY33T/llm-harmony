# llm-harmony Slice 4 — Starting a provider

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `llm-harmony start <provider>` — bring a provider up, supervised, and wait until it actually answers. The first capability that creates rather than observes.

**Architecture:** Harmony never learns a provider's flags. It runs a `start` command you declare, installed as a launchd agent so macOS supervises it. Readiness is confirmed by polling the adapter's existing `probe()` — the method slice 1 defined and nothing has used since.

**Tech Stack:** Same crate. No new dependencies; `launchctl` is invoked as a subprocess.

**Spec:** [`../design.md`](../design.md) §6 (trust modes), [`../architecture.md`](../architecture.md) §4 (the unresolved question, now decided).

## The decision this implements

*If the provider that should serve a model isn't running, does harmony start it?*
**Yes.** Decided 2026-09-09, over refusing and over placing on a different
artifact.

Two documented positions change, and both need rewriting rather than quietly
contradicting:

- **README non-goals** currently say harmony "does not start or supervise
  inference servers." It now does both, on request, per provider.
- **[`prior-art.md`](../prior-art.md)** rejects llama-swap partly for owning
  lifecycle. The rejection stands on the *other* half — llama-swap sits in the
  request path — and that objection must be restated so it does not read as
  self-contradiction.

What does **not** change: harmony still never carries a request, and still
never learns a provider's flags.

## Global Constraints

- **Harmony runs a command you declared; it never composes one.** Every flag
  lives in your script. Verified 2026-09-09: `~/.llamacpp/serve-pool` carries
  `--models-dir --models-max 1 --jinja --reasoning-format deepseek`, and
  `~/.vllm-mlx/serve-pool` carries `--continuous-batching --models-config`.
  Harmony reproducing those would be a second, worse source of truth.
- **Starting is opt-in per provider and off by default**, in the spirit of
  `design.md` §6's `observe` default. A provider with no `start` declared can
  be observed and never launched.
- **Installing the agent is a separate act from starting it.** `install` writes
  to `~/Library/LaunchAgents/` — the first time harmony writes outside its own
  state directory — and must be requested explicitly, never as a side effect of
  `start`.
- **Fail-open survives.** launchd owns the process, not harmony. Killing
  harmony must leave every server running, as `design.md` §8 requires.
- **A start that cannot be confirmed is a failure.** "Launched" is not
  "serving": the command may exit, the port may never open, the model may fail
  to load. Success means `probe()` answered.
- macOS only, as slices 1–3.

## Launch surfaces, verified on this machine

| Provider | Command | Who wrote it |
|---|---|---|
| LM Studio | `lms server start` | vendor CLI at `~/.lmstudio/bin/lms` |
| llama.cpp | `~/.llamacpp/serve-pool` | you |
| vLLM-MLX | `~/.vllm-mlx/serve-pool` | you |
| Ollama | `ollama serve` | vendor CLI |

`launchctl list` already shows `com.ollama.ollama` and
`application.ai.elementlabs.lmstudio…`, so launchd is the platform's own answer
to this, not an invention of ours.

---

### Task 1: Declare a start command

**Files:** `src/config.rs`
**Interfaces:** `ProviderConfig { kind, url, start: Option<String>, launchd_label: Option<String> }`

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn a_provider_may_declare_how_to_start_it() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "llamacpp"
            url = "http://127.0.0.1:8080"
            start = "~/.llamacpp/serve-pool --port 8080"
            "#,
        )
        .unwrap();
        assert_eq!(
            c.providers[0].start.as_deref(),
            Some("~/.llamacpp/serve-pool --port 8080")
        );
    }

    /// Starting is opt-in. A provider that declares nothing can be observed
    /// and never launched, which is the `observe` default of design.md §6.
    #[test]
    fn a_provider_without_a_start_command_cannot_be_started() {
        let c = Config::defaults();
        assert!(c.providers.iter().all(|p| p.start.is_none()));
    }

    /// The label is derived, not invented, so `install` and `stop` agree.
    #[test]
    fn the_launchd_label_defaults_to_a_stable_derivation() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "llamacpp"
            url = "http://127.0.0.1:8080"
            start = "x"
            "#,
        )
        .unwrap();
        assert_eq!(c.providers[0].label(), "dev.llm-harmony.llamacpp");
    }

    #[test]
    fn an_explicit_label_wins_so_an_existing_agent_can_be_adopted() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "ollama"
            url = "http://127.0.0.1:11434"
            launchd_label = "com.ollama.ollama"
            "#,
        )
        .unwrap();
        assert_eq!(c.providers[0].label(), "com.ollama.ollama");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config`
Expected: FAIL — no field `start`.

- [ ] **Step 3: Implement**

Add to `ProviderConfig`:

```rust
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub url: String,
    /// The command that brings this provider up. Harmony runs it verbatim and
    /// never composes one: every flag that matters lives in the user's own
    /// script, and reproducing them here would be a second source of truth.
    pub start: Option<String>,
    /// Adopt an existing launchd agent instead of installing one. Useful for
    /// providers the platform already supervises -- `com.ollama.ollama` and
    /// LM Studio's own agent are both registered on this machine already.
    pub launchd_label: Option<String>,
}

impl ProviderConfig {
    pub fn label(&self) -> String {
        self.launchd_label
            .clone()
            .unwrap_or_else(|| format!("dev.llm-harmony.{}", self.kind.as_str()))
    }
}
```

Add both to `RawProvider` as `#[serde(default)] Option<String>`, and carry them
through `from_toml` and `defaults` (both `None` in defaults).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "feat: declare a per-provider start command and launchd label"
```

---

### Task 2: Generate the launchd agent

**Files:** Create `src/launch/mod.rs`, `src/launch/plist.rs`
**Interfaces:** `plist::for_provider(&ProviderConfig) -> Result<String, String>`, `plist::path(&str) -> Option<PathBuf>`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use crate::provider::ProviderKind;

    fn cfg(start: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            kind: ProviderKind::LlamaCpp,
            url: "http://127.0.0.1:8080".into(),
            start: start.map(str::to_string),
            launchd_label: None,
        }
    }

    #[test]
    fn the_agent_runs_the_declared_command_through_a_shell() {
        let xml = for_provider(&cfg(Some("~/.llamacpp/serve-pool --port 8080"))).unwrap();
        assert!(xml.contains("dev.llm-harmony.llamacpp"), "{xml}");
        assert!(xml.contains("/bin/sh"), "a shell, so ~ and flags behave as typed: {xml}");
        assert!(xml.contains("serve-pool --port 8080"), "{xml}");
    }

    /// KeepAlive is the whole point of supervising: a crashed provider comes
    /// back without anyone watching.
    #[test]
    fn the_agent_restarts_on_crash_but_not_on_a_clean_stop() {
        let xml = for_provider(&cfg(Some("x"))).unwrap();
        assert!(xml.contains("KeepAlive"), "{xml}");
        assert!(xml.contains("SuccessfulExit"), "a deliberate stop must stay stopped: {xml}");
    }

    #[test]
    fn a_provider_with_no_start_command_cannot_produce_an_agent() {
        assert!(for_provider(&cfg(None)).is_err());
    }

    /// The command is XML-escaped. A `&` in a path would otherwise produce a
    /// plist launchd silently refuses to load.
    #[test]
    fn the_command_is_escaped_for_xml() {
        let xml = for_provider(&cfg(Some("run --a=1 && echo <ok>"))).unwrap();
        assert!(xml.contains("&amp;&amp;"), "{xml}");
        assert!(xml.contains("&lt;ok&gt;"), "{xml}");
        assert!(!xml.contains("&& echo <ok>"), "raw markup leaked: {xml}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib plist`
Expected: FAIL — `cannot find function for_provider`.

- [ ] **Step 3: Implement**

```rust
use std::path::PathBuf;

use crate::config::ProviderConfig;

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// A LaunchAgent that runs the provider's declared command.
///
/// Run through `/bin/sh -c` deliberately: the declared command is what the
/// user would type, complete with `~` and quoting, and splitting it into an
/// argv here would mean reimplementing shell parsing badly.
///
/// `KeepAlive.SuccessfulExit = false` means: restart it if it crashes, leave it
/// alone if it exited cleanly. Without the qualifier a deliberate stop would be
/// undone immediately by launchd.
pub fn for_provider(p: &ProviderConfig) -> Result<String, String> {
    let Some(command) = p.start.as_deref() else {
        return Err(format!(
            "{} declares no `start` command; add one to config to start it",
            p.kind
        ));
    };
    let label = p.label();
    Ok(format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-lc</string>
    <string>{command}</string>
  </array>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{log}.out.log</string>
  <key>StandardErrorPath</key><string>{log}.err.log</string>
</dict>
</plist>
"#,
        label = escape(&label),
        command = escape(command),
        log = escape(&log_stem(&label)),
    ))
}

fn log_stem(label: &str) -> String {
    match std::env::var_os("HOME") {
        Some(h) => format!("{}/.local/state/llm-harmony/{label}", PathBuf::from(h).display()),
        None => format!("/tmp/{label}"),
    }
}

/// `~/Library/LaunchAgents/<label>.plist`
pub fn path(label: &str) -> Option<PathBuf> {
    Some(
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/LaunchAgents")
            .join(format!("{label}.plist")),
    )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib plist`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/launch
git commit -m "feat: generate a launchd agent from a declared start command"
```

---

### Task 3: Drive launchctl

**Files:** Create `src/launch/launchctl.rs`
**Interfaces:** `launchctl::bootstrap(&Path)`, `kickstart(&str)`, `bootout(&str)`, `is_loaded(&str) -> bool`, all returning `Result<(), String>`.

- [ ] **Step 1: Write the failing test**

Only the argv construction is unit-testable; running `launchctl` is Task 5's
live verification.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// gui/<uid> is the modern domain. `launchctl load` is deprecated and
    /// silently does the wrong thing under some session types.
    #[test]
    fn commands_target_the_gui_domain_of_the_current_user() {
        let uid = 501;
        assert_eq!(
            bootstrap_argv(uid, std::path::Path::new("/x/a.plist")),
            vec!["bootstrap", "gui/501", "/x/a.plist"]
        );
        assert_eq!(kickstart_argv(uid, "dev.llm-harmony.llamacpp"),
                   vec!["kickstart", "-k", "gui/501/dev.llm-harmony.llamacpp"]);
        assert_eq!(bootout_argv(uid, "dev.llm-harmony.llamacpp"),
                   vec!["bootout", "gui/501/dev.llm-harmony.llamacpp"]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib launchctl`
Expected: FAIL.

- [ ] **Step 3: Implement**

```rust
use std::path::Path;
use std::process::Command;

fn uid() -> u32 {
    // SAFETY: getuid cannot fail and has no side effects.
    unsafe { libc::getuid() }
}

pub fn bootstrap_argv(uid: u32, plist: &Path) -> Vec<String> {
    vec!["bootstrap".into(), format!("gui/{uid}"), plist.display().to_string()]
}

pub fn kickstart_argv(uid: u32, label: &str) -> Vec<String> {
    vec!["kickstart".into(), "-k".into(), format!("gui/{uid}/{label}")]
}

pub fn bootout_argv(uid: u32, label: &str) -> Vec<String> {
    vec!["bootout".into(), format!("gui/{uid}/{label}")]
}

fn run(args: Vec<String>) -> Result<(), String> {
    let out = Command::new("launchctl")
        .args(&args)
        .output()
        .map_err(|e| format!("launchctl: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "launchctl {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr).trim()
    ))
}

pub fn bootstrap(plist: &Path) -> Result<(), String> {
    run(bootstrap_argv(uid(), plist))
}

/// `-k` restarts it if already running, which makes `start` idempotent.
pub fn kickstart(label: &str) -> Result<(), String> {
    run(kickstart_argv(uid(), label))
}

pub fn bootout(label: &str) -> Result<(), String> {
    run(bootout_argv(uid(), label))
}

pub fn is_loaded(label: &str) -> bool {
    Command::new("launchctl")
        .args(["print", &format!("gui/{}/{label}", uid())])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
```

Add `libc = "0.2"` to `Cargo.toml` as a **direct** dependency. It is currently
reachable only through `clang-sys` -> `bindgen`, which is a *build* dependency
of something else and not linkable from our code -- verified 2026-09-09 with
`cargo tree -i libc`.

**Note the constraint this breaks:** slices 1–3 had no `Command::new` anywhere
in `src/`. This is the first, and it is deliberate. Update the safety grep in
Task 5 to expect exactly these, and nowhere else.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib launchctl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/launch/launchctl.rs Cargo.toml
git commit -m "feat: drive launchctl in the gui domain"
```

---

### Task 4: Wait until it actually answers

**Files:** Create `src/launch/ready.rs`
**Interfaces:** `ready::wait_for(&dyn Adapter, &Http, &str, Duration) -> Result<Duration, String>`

This is what `Adapter::probe()` was for. Slice 1 defined it and flagged it as
exercised only by tests; confirming a provider you just started is its job.

- [ ] **Step 1: Write the failing test**

```rust
mod support;

use std::time::Duration;

use llm_harmony::adapters::lmstudio::LmStudio;
use llm_harmony::http::Http;
use llm_harmony::launch::ready;

#[test]
fn a_provider_that_answers_is_ready_immediately() {
    let body = std::fs::read_to_string("tests/fixtures/lmstudio/models-none-loaded.json").unwrap();
    let s = support::StubServer::start_lmstudio_style(support::routes(&[("/api/v0/models", &body)]));
    let http = Http::new(Duration::from_millis(500));
    assert!(ready::wait_for(&LmStudio, &http, &s.base_url(), Duration::from_secs(2)).is_ok());
}

/// "Launched" is not "serving". A port that never opens must time out with a
/// message naming the provider, not hang.
#[test]
fn a_provider_that_never_answers_times_out() {
    let http = Http::new(Duration::from_millis(100));
    let err = ready::wait_for(
        &LmStudio,
        &http,
        "http://127.0.0.1:1",
        Duration::from_millis(400),
    )
    .unwrap_err();
    assert!(err.contains("did not answer"), "{err}");
}

/// A server that answers with something else is not this provider, and
/// must not be reported ready.
#[test]
fn a_wrong_provider_on_the_port_is_not_ready() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let http = Http::new(Duration::from_millis(200));
    assert!(ready::wait_for(&LmStudio, &http, &s.base_url(), Duration::from_millis(400)).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test launch_ready`
Expected: FAIL — unresolved import.

- [ ] **Step 3: Implement**

```rust
use std::time::{Duration, Instant};

use crate::http::Http;
use crate::provider::Adapter;

/// Poll until the provider answers as itself, or give up.
///
/// A start is only successful when `probe()` succeeds. The command exiting 0
/// proves nothing: llama-server takes seconds to map a model, and a port that
/// never opens is the common failure.
pub fn wait_for(
    adapter: &dyn Adapter,
    http: &Http,
    base: &str,
    timeout: Duration,
) -> Result<Duration, String> {
    let started = Instant::now();
    let step = Duration::from_millis(200);
    loop {
        if adapter.probe(http, base).is_ok() {
            return Ok(started.elapsed());
        }
        if started.elapsed() >= timeout {
            return Err(format!(
                "{} did not answer at {base} within {:.0}s",
                adapter.kind(),
                timeout.as_secs_f64()
            ));
        }
        std::thread::sleep(step);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test launch_ready`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/launch/ready.rs tests/launch_ready.rs
git commit -m "feat: confirm a started provider by probing until it answers"
```

---

### Task 5: `install`, `start`, `stop`

**Files:** `src/main.rs`, `src/launch/mod.rs`

Three commands, deliberately separate. `install` grants the authority and
writes the agent; `start` and `stop` use it.

- [ ] **Step 1: Wire the subcommands**

```rust
    /// Write a launchd agent for a provider. Writes to ~/Library/LaunchAgents.
    Install {
        provider: String,
        /// Print the agent instead of writing it.
        #[arg(long)]
        dry_run: bool,
    },
    /// Start a provider and wait until it answers.
    Start {
        provider: String,
        #[arg(long, default_value_t = 60)]
        timeout_s: u64,
    },
    /// Stop a provider harmony installed.
    Stop { provider: String },
```

Each resolves the provider from config by `ProviderKind`, erroring clearly when
it is unknown or declares no `start`.

`install` must:
1. refuse when `start` is absent, naming the config key to add;
2. print the full destination path before writing, so the one write outside
   harmony's own state directory is never silent;
3. write the plist, then `bootstrap` it.

`start` must: `kickstart`, then `ready::wait_for`, and report the elapsed time.
A timeout is exit 1 with the log paths named, because the reason will be in
them.

`stop` must: `bootout`, and say plainly that a provider harmony did not install
is not harmony's to stop.

- [ ] **Step 2: Verify the safety property changed as intended**

Run:

```bash
grep -rn "Command::new" src/
```

Expected: matches **only** in `src/launch/launchctl.rs`. Slices 1–3 had none
anywhere; this slice adds exactly one call site, and no other module may spawn
a process.

Run:

```bash
grep -rnE "fs::write|File::create" src/
```

Expected: `src/launch/mod.rs` (the plist) and `src/record.rs` (the corpus), and
nothing else.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs src/launch/mod.rs
git commit -m "feat: llm-harmony install, start and stop for a provider"
```

---

### Task 6: Live verification

llama.cpp and vLLM-MLX are both down on this machine, which makes them the
honest test.

- [ ] **Step 1: Declare llama.cpp's start command**

In `~/.config/llm-harmony/config.toml`:

```toml
[[provider]]
kind = "lmstudio"
url  = "http://127.0.0.1:1234"

[[provider]]
kind = "ollama"
url  = "http://127.0.0.1:11434"

[[provider]]
kind = "llamacpp"
url  = "http://127.0.0.1:8080"
start = "$HOME/.llamacpp/serve-pool --port 8080"

[[provider]]
kind = "vllm"
url  = "http://127.0.0.1:8000"
start = "$HOME/.vllm-mlx/serve-pool --port 8000"
```

- [ ] **Step 2: Inspect before writing anything**

```bash
llm-harmony install llamacpp --dry-run
```

Expected: the plist on stdout, nothing written. Confirm the command line inside
it matches your script exactly.

- [ ] **Step 3: Install and start**

```bash
llm-harmony install llamacpp
llm-harmony start llamacpp
llm-harmony status
```

Expected: `start` reports the time it took to answer, and `status` shows
`llamacpp` as a live row rather than `not running`. This also finally satisfies
slice 1's criterion 2 — a provider appearing with no code change — which has
been outstanding since the first slice.

- [ ] **Step 4: Verify supervision is real**

```bash
pkill -f 'llama-server --models-dir'
sleep 5
llm-harmony status
```

Expected: `llamacpp` is up again. launchd restarted it; nothing harmony did.

- [ ] **Step 5: Verify fail-open survived the change**

```bash
llm-harmony stop llamacpp   # then start it again
llm-harmony start llamacpp
mv "$(which llm-harmony)" /tmp/ && sleep 2
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/v1/models
mv /tmp/llm-harmony "$(dirname "$(command -v cargo)")/llm-harmony"
```

Expected: `200`. **Harmony being gone does not stop what it started** — the
property that made supervising acceptable rather than a violation of §8.

- [ ] **Step 6: Verify a clean stop stays stopped**

```bash
llm-harmony stop llamacpp
sleep 5
llm-harmony status
```

Expected: `llamacpp` reports `not running` and stays that way. If launchd
restarts it, `KeepAlive.SuccessfulExit` is wrong.

- [ ] **Step 7: Grow the estimator's corpus**

With llama.cpp finally up:

```bash
llm-harmony status --record
llm-harmony estimate <a model llama.cpp serves>
```

Expected: the corpus stops being single-provider, which slice 3 flagged as its
main weakness.

---

## Documentation this slice must change

Not optional, and not cosmetic — both currently assert the opposite of what the
code does:

- [ ] **README non-goals**: "does not start or supervise inference servers"
  becomes a description of the opt-in, with the boundary that survives stated
  plainly: harmony starts servers on request and never carries a request.
- [ ] **[`prior-art.md`](../prior-art.md)**: restate the llama-swap rejection on
  the request-path objection alone, and say explicitly that the lifecycle
  objection was dropped, when, and why.
- [ ] **[`architecture.md`](../architecture.md) §4 and §10 Q1**: record the
  decision and its date. The question is answered; leaving it open would be
  the docs lying.
- [ ] **[`design.md`](../design.md) §6**: `start` joins the trust vocabulary
  beside `observe` / `evict` / `manage`.

## What this slice is not

- **Not `RESOLVE`.** Placement, admission and activation are the next slice.
  This one only brings a provider up.
- **Not model loading.** Starting `llama-server` in router mode makes models
  *available*; making one resident is activation, and belongs with `RESOLVE`.
- **Not eviction.** Harmony still cannot unload anything.
- **Not cross-machine.** `gui/<uid>` is this user's session on this host.
