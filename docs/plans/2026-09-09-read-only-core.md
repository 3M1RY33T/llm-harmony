# llm-harmony Slice 1 — Read-Only Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llm-harmony status`, a command that polls four local inference providers, attributes real memory to each, and prints what the machine is collectively spending on them.

**Architecture:** One crate with a `lib.rs` (the reusable core the slice-2 daemon will import) and a thin `main.rs`. Blocking HTTP via `ureq`; the four providers are polled concurrently with `std::thread::scope`. Memory attribution is pure-Rust via `libproc` (port → pid → descendants → `phys_footprint`) with machine totals from `sysinfo`. No async runtime, no persistence, no code path that can unload a model.

**Tech Stack:** Rust 1.98.1 (edition 2021), `ureq` 3.4, `serde`/`serde_json` 1, `toml` 1.1, `clap` 4.6, `libproc` 0.14, `sysinfo` 0.39.

**Spec:** [`../specs/2026-09-09-read-only-core-design.md`](../specs/2026-09-09-read-only-core-design.md)

## Global Constraints

- **Target is macOS on Apple Silicon only.** `libproc` and `phys_footprint` are macOS APIs. Do not add `#[cfg]` branches for other platforms; this slice does not claim to support them.
- **No adapter may expose an unload, stop, evict, or delete method.** The `Adapter` trait is the enforcement point. If a task's code would add one, the task is wrong.
- **`context_tokens` is only ever populated from a *serving* window.** Never from `max_context_length` (LM Studio), `n_ctx_train` (llama.cpp), or `details.context_length` (Ollama `/api/tags`). These are capability figures and over-promise room — on `qwen3-8b` by 5×.
- **Fail open.** No single provider may cause a nonzero exit. Only a malformed config or an unreadable machine total exits `1`.
- **Never trust an HTTP status alone.** Verified 2026-09-09: LM Studio returns `200` with `{"error":"Unexpected endpoint or method. (GET /path)"}` for *any* unrecognised path. Every `probe()` validates response *shape*.
- **All timings and byte counts in tests are fixtures, never live calls.** Only `tests/smoke.rs` may touch the real machine.
- Binary name is `llm-harmony`; the library crate is `llm_harmony`.

---

## File Structure

| File | Responsibility |
|---|---|
| `Cargo.toml` | Deps, `[lib]` + `[[bin]]` targets |
| `src/lib.rs` | Module declarations and re-exports |
| `src/provider.rs` | `ProviderKind`, `State`, `LoadedModel`, `ProbeError`, `trait Adapter` |
| `src/config.rs` | `Config`, `ProviderConfig`, built-in defaults, TOML loading |
| `src/http.rs` | `Http` — a timeout-bounded `ureq` wrapper returning `ProbeError` |
| `src/adapters/mod.rs` | `all_adapters()` registry |
| `src/adapters/lmstudio.rs` | LM Studio adapter |
| `src/adapters/ollama.rs` | Ollama adapter |
| `src/adapters/llamacpp.rs` | llama.cpp adapter |
| `src/adapters/vllm.rs` | vLLM-MLX adapter |
| `src/memory.rs` | `Machine` totals, port→pid, descendants, `phys_footprint` summing |
| `src/ledger.rs` | Concurrent assembly of `ProviderRow`s into a `Ledger` |
| `src/render.rs` | Table and JSON rendering |
| `src/main.rs` | `clap` wiring |
| `tests/support/mod.rs` | Canned-response stub HTTP server |
| `tests/adapters.rs` | Adapter parsing against fixtures |
| `tests/integration.rs` | `Ledger` assembly against stub servers |
| `tests/smoke.rs` | The only test that touches the real machine |
| `tests/fixtures/**` | Recorded provider JSON |

**Deviation from the spec, deliberate:** the spec put `pids()` on the `Adapter` trait. Probing showed port→pid→descendants is *uniform* across all four providers (verified: pid 647 owns `:1234`, pid 4140 owns `:11434`, and in both cases model memory lives in descendants). So pid discovery lives in `memory.rs` and the trait loses a method. Fewer methods on the trait is also a stronger version of the safety property.

---

### Task 1: Crate skeleton and core domain types

**Files:**
- Create: `Cargo.toml`, `src/lib.rs`, `src/provider.rs`
- Test: unit tests inside `src/provider.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `ProviderKind` (with `as_str`, `default_port`, `FromStr`), `State`, `LoadedModel`, `ProbeError`, `trait Adapter`.

- [ ] **Step 1: Write `Cargo.toml`**

```toml
[package]
name = "llm-harmony"
version = "0.1.0"
edition = "2021"

[lib]
name = "llm_harmony"
path = "src/lib.rs"

[[bin]]
name = "llm-harmony"
path = "src/main.rs"

[dependencies]
ureq = { version = "3.4", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "1.1"
clap = { version = "4.6", features = ["derive"] }
libproc = "0.14"
sysinfo = "0.39"
```

- [ ] **Step 2: Write the failing test**

Create `src/provider.rs` containing only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_round_trips_through_str() {
        for k in ProviderKind::ALL {
            assert_eq!(k.as_str().parse::<ProviderKind>().unwrap(), k);
        }
    }

    #[test]
    fn kind_rejects_unknown_names() {
        assert!("vllm-mlx".parse::<ProviderKind>().is_err());
        assert!("".parse::<ProviderKind>().is_err());
    }

    #[test]
    fn default_ports_match_the_machine() {
        assert_eq!(ProviderKind::LmStudio.default_port(), 1234);
        assert_eq!(ProviderKind::Ollama.default_port(), 11434);
        assert_eq!(ProviderKind::LlamaCpp.default_port(), 8080);
        assert_eq!(ProviderKind::Vllm.default_port(), 8000);
    }

    #[test]
    fn an_unloaded_model_has_no_serving_window() {
        let m = LoadedModel::unloaded("qwen3-14b");
        assert_eq!(m.state, State::NotLoaded);
        assert_eq!(m.context_tokens, None, "a window is only knowable once resident");
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib`
Expected: FAIL — `cannot find type ProviderKind in this scope`.

- [ ] **Step 4: Write `src/lib.rs`**

```rust
pub mod provider;

pub use provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};
```

- [ ] **Step 5: Write the implementation above the test module in `src/provider.rs`**

```rust
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    LmStudio,
    LlamaCpp,
    Vllm,
    Ollama,
}

impl ProviderKind {
    /// Fixed reporting order. Also the order rows appear in `status`.
    pub const ALL: [ProviderKind; 4] = [
        ProviderKind::LmStudio,
        ProviderKind::Ollama,
        ProviderKind::LlamaCpp,
        ProviderKind::Vllm,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderKind::LmStudio => "lmstudio",
            ProviderKind::LlamaCpp => "llamacpp",
            ProviderKind::Vllm => "vllm",
            ProviderKind::Ollama => "ollama",
        }
    }

    /// Verified on this machine 2026-09-09: LM Studio on 1234, Ollama on 11434.
    pub fn default_port(&self) -> u16 {
        match self {
            ProviderKind::LmStudio => 1234,
            ProviderKind::LlamaCpp => 8080,
            ProviderKind::Vllm => 8000,
            ProviderKind::Ollama => 11434,
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ProviderKind::ALL
            .into_iter()
            .find(|k| k.as_str() == s)
            .ok_or_else(|| format!("unknown provider kind `{s}`"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum State {
    Loaded,
    Loading,
    NotLoaded,
}

/// One model as a provider reports it.
///
/// `context_tokens` is `Option` on purpose: a serving window is only knowable
/// once a model is resident. Capability figures (`max_context_length`,
/// `n_ctx_train`, `/api/tags` `details.context_length`) must never land here.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LoadedModel {
    pub id: String,
    pub state: State,
    pub context_tokens: Option<u32>,
    pub weights_bytes: Option<u64>,
}

impl LoadedModel {
    pub fn unloaded(id: impl Into<String>) -> Self {
        LoadedModel {
            id: id.into(),
            state: State::NotLoaded,
            context_tokens: None,
            weights_bytes: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "error", rename_all = "kebab-case")]
pub enum ProbeError {
    /// Nothing is listening. Not an error condition — the provider is simply off.
    NotListening,
    /// Listening, but did not answer in time.
    Timeout,
    /// Answered with an unexpected status.
    Status { code: u16 },
    /// Answered, but not with what this provider is supposed to answer with.
    Malformed { reason: String },
    /// Answered like a different provider than the config declared.
    KindMismatch { expected: ProviderKind },
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeError::NotListening => write!(f, "not running"),
            ProbeError::Timeout => write!(f, "unreadable: timed out"),
            ProbeError::Status { code } => write!(f, "unreadable: HTTP {code}"),
            ProbeError::Malformed { reason } => write!(f, "unreadable: {reason}"),
            ProbeError::KindMismatch { expected } => {
                write!(f, "kind mismatch: expected {expected}")
            }
        }
    }
}

/// Read-only by construction.
///
/// There is deliberately no `unload`, `stop`, or `evict` method. Slice 1 cannot
/// free memory, and that is checked by reading this trait rather than a config.
pub trait Adapter: Send + Sync {
    fn kind(&self) -> ProviderKind;

    /// Confirm the endpoint is the provider the config declared it to be.
    fn probe(&self, http: &crate::http::Http, base: &str) -> Result<(), ProbeError>;

    /// Every model the provider knows about, with its state.
    fn list(&self, http: &crate::http::Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError>;
}
```

Note: `src/lib.rs` must also declare `pub mod http;` before this compiles. Add a placeholder `src/http.rs` containing `pub struct Http;` for now — Task 3 replaces it.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --lib`
Expected: PASS — 4 tests.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/provider.rs src/http.rs
git commit -m "feat: core provider types with read-only adapter trait"
```

---

### Task 2: Config with built-in defaults

**Files:**
- Create: `src/config.rs`
- Modify: `src/lib.rs` (add `pub mod config;`)
- Test: unit tests inside `src/config.rs`

**Interfaces:**
- Consumes: `ProviderKind` from Task 1.
- Produces: `Config { providers: Vec<ProviderConfig> }`, `ProviderConfig { kind: ProviderKind, url: String }`, `Config::defaults()`, `Config::from_toml(&str) -> Result<Config, String>`, `Config::load(Option<&Path>) -> Result<Config, String>`.

- [ ] **Step 1: Write the failing test**

Create `src/config.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_cover_all_four_providers() {
        let c = Config::defaults();
        assert_eq!(c.providers.len(), 4);
        let lms = c.providers.iter().find(|p| p.kind == ProviderKind::LmStudio).unwrap();
        assert_eq!(lms.url, "http://127.0.0.1:1234");
    }

    #[test]
    fn toml_replaces_defaults_entirely() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "ollama"
            url = "http://127.0.0.1:99"
            "#,
        )
        .unwrap();
        assert_eq!(c.providers.len(), 1, "an explicit config is not merged with defaults");
        assert_eq!(c.providers[0].url, "http://127.0.0.1:99");
    }

    #[test]
    fn empty_toml_falls_back_to_defaults() {
        assert_eq!(Config::from_toml("").unwrap().providers.len(), 4);
    }

    #[test]
    fn unknown_kind_is_a_config_error() {
        let err = Config::from_toml(
            r#"
            [[provider]]
            kind = "vllm-mlx"
            url = "http://127.0.0.1:8000"
            "#,
        )
        .unwrap_err();
        assert!(err.contains("vllm-mlx"), "error should name the bad kind, got: {err}");
    }

    #[test]
    fn trailing_slashes_are_stripped_so_urls_join_cleanly() {
        let c = Config::from_toml(
            r#"
            [[provider]]
            kind = "ollama"
            url = "http://127.0.0.1:11434/"
            "#,
        )
        .unwrap();
        assert_eq!(c.providers[0].url, "http://127.0.0.1:11434");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config`
Expected: FAIL — `cannot find type Config in this scope`.

- [ ] **Step 3: Write the implementation above the test module**

```rust
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::provider::ProviderKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub providers: Vec<ProviderConfig>,
}

/// The on-disk shape. Kept separate from `Config` so `kind` can be validated
/// with a message that names the offending string.
#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    provider: Vec<RawProvider>,
}

#[derive(Debug, Deserialize)]
struct RawProvider {
    kind: String,
    url: String,
}

impl Config {
    /// Zero-config behaviour: every provider on its conventional port.
    pub fn defaults() -> Self {
        Config {
            providers: ProviderKind::ALL
                .into_iter()
                .map(|kind| ProviderConfig {
                    kind,
                    url: format!("http://127.0.0.1:{}", kind.default_port()),
                })
                .collect(),
        }
    }

    pub fn from_toml(s: &str) -> Result<Self, String> {
        let raw: RawConfig = toml::from_str(s).map_err(|e| format!("invalid config: {e}"))?;
        if raw.provider.is_empty() {
            return Ok(Config::defaults());
        }
        let providers = raw
            .provider
            .into_iter()
            .map(|p| {
                let kind = p.kind.parse::<ProviderKind>()?;
                Ok(ProviderConfig {
                    kind,
                    url: p.url.trim_end_matches('/').to_string(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Config { providers })
    }

    /// `~/.config/llm-harmony/config.toml`, or defaults if it is absent.
    pub fn load(explicit: Option<&Path>) -> Result<Self, String> {
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => match Self::default_path() {
                Some(p) => p,
                None => return Ok(Config::defaults()),
            },
        };
        match std::fs::read_to_string(&path) {
            Ok(s) => Config::from_toml(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && explicit.is_none() => {
                Ok(Config::defaults())
            }
            Err(e) => Err(format!("cannot read {}: {e}", path.display())),
        }
    }

    fn default_path() -> Option<PathBuf> {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".config/llm-harmony/config.toml"))
    }
}
```

Add `pub mod config;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/lib.rs
git commit -m "feat: config with built-in provider defaults"
```

---

### Task 3: HTTP client and the test stub server

**Files:**
- Rewrite: `src/http.rs` (replacing the Task 1 placeholder)
- Create: `tests/support/mod.rs`
- Test: `tests/http.rs`

**Interfaces:**
- Consumes: `ProbeError` from Task 1.
- Produces: `Http::new(Duration) -> Http`, `Http::get_json(&self, url: &str) -> Result<serde_json::Value, ProbeError>`. And for tests, `support::StubServer::start(routes) -> StubServer` with `.base_url() -> String`.

- [ ] **Step 1: Write the stub server**

Create `tests/support/mod.rs`:

```rust
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// A canned-response HTTP server. Routes map a request path to
/// `(status, body)`. Any path not in the map gets a 404 — except when
/// `always_200` is set, which reproduces LM Studio's behaviour of answering
/// unknown paths with a 200 and an error body.
pub struct StubServer {
    port: u16,
    stop: Arc<AtomicBool>,
}

impl StubServer {
    pub fn start(routes: HashMap<String, (u16, String)>) -> Self {
        Self::start_inner(routes, false)
    }

    /// Verified against LM Studio 2026-09-09: unknown paths return
    /// `200 {"error":"Unexpected endpoint or method. (GET /path)"}`.
    pub fn start_lmstudio_style(routes: HashMap<String, (u16, String)>) -> Self {
        Self::start_inner(routes, true)
    }

    fn start_inner(routes: HashMap<String, (u16, String)>, always_200: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();

        thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_thread.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                handle(stream, &routes, always_200);
            }
        });

        StubServer { port, stop }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Unblock the accept loop.
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn handle(mut stream: TcpStream, routes: &HashMap<String, (u16, String)>, always_200: bool) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/").to_string();

    let (status, body) = match routes.get(&path) {
        Some((s, b)) => (*s, b.clone()),
        None if always_200 => (
            200,
            format!(r#"{{"error":"Unexpected endpoint or method. (GET {path})"}}"#),
        ),
        None => (404, "404 page not found".to_string()),
    };

    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// Convenience for the common single-route case.
pub fn routes(pairs: &[(&str, &str)]) -> HashMap<String, (u16, String)> {
    pairs
        .iter()
        .map(|(p, b)| (p.to_string(), (200u16, b.to_string())))
        .collect()
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/http.rs`:

```rust
mod support;

use std::time::Duration;

use llm_harmony::http::Http;
use llm_harmony::provider::ProbeError;

fn http() -> Http {
    Http::new(Duration::from_millis(1500))
}

#[test]
fn parses_a_json_body() {
    let s = support::StubServer::start(support::routes(&[("/ok", r#"{"models":[]}"#)]));
    let v = http().get_json(&format!("{}/ok", s.base_url())).unwrap();
    assert!(v["models"].is_array());
}

#[test]
fn a_closed_port_is_not_listening_not_an_error() {
    // Port 1 is reserved and nothing binds it.
    let err = http().get_json("http://127.0.0.1:1/anything").unwrap_err();
    assert_eq!(err, ProbeError::NotListening);
}

#[test]
fn a_404_is_reported_with_its_code() {
    let s = support::StubServer::start(support::routes(&[]));
    let err = http().get_json(&format!("{}/missing", s.base_url())).unwrap_err();
    assert_eq!(err, ProbeError::Status { code: 404 });
}

#[test]
fn a_non_json_body_is_malformed_not_a_panic() {
    let mut r = support::routes(&[]);
    r.insert("/html".to_string(), (200, "<html>hello</html>".to_string()));
    let s = support::StubServer::start(r);
    let err = http().get_json(&format!("{}/html", s.base_url())).unwrap_err();
    assert!(matches!(err, ProbeError::Malformed { .. }), "got {err:?}");
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --test http`
Expected: FAIL — `Http::new` not found / `http` module is private.

- [ ] **Step 4: Write the implementation**

Replace `src/http.rs` entirely:

```rust
use std::time::Duration;

use crate::provider::ProbeError;

/// A timeout-bounded HTTP client.
///
/// Every failure becomes a `ProbeError` rather than propagating, because no
/// provider may make `status` fail.
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    pub fn new(timeout: Duration) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build()
            .into();
        Http { agent }
    }

    pub fn get_json(&self, url: &str) -> Result<serde_json::Value, ProbeError> {
        let mut response = self.agent.get(url).call().map_err(map_err)?;
        response
            .body_mut()
            .read_json::<serde_json::Value>()
            .map_err(|e| ProbeError::Malformed {
                reason: format!("not JSON: {e}"),
            })
    }
}

fn map_err(e: ureq::Error) -> ProbeError {
    match e {
        ureq::Error::StatusCode(code) => ProbeError::Status { code },
        ureq::Error::Timeout(_) => ProbeError::Timeout,
        ureq::Error::Io(io) => match io.kind() {
            std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::AddrNotAvailable
            | std::io::ErrorKind::ConnectionReset => ProbeError::NotListening,
            std::io::ErrorKind::TimedOut => ProbeError::Timeout,
            _ => ProbeError::Malformed {
                reason: io.to_string(),
            },
        },
        other => ProbeError::Malformed {
            reason: other.to_string(),
        },
    }
}
```

Ensure `src/lib.rs` contains `pub mod http;`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test http`
Expected: PASS — 4 tests.

- [ ] **Step 6: Commit**

```bash
git add src/http.rs src/lib.rs tests/support/mod.rs tests/http.rs
git commit -m "feat: timeout-bounded http client with fail-open error mapping"
```

---

### Task 4: LM Studio adapter

The flagship adapter, and the one carrying the capability-vs-configuration test.

**Files:**
- Create: `src/adapters/mod.rs`, `src/adapters/lmstudio.rs`
- Create: `tests/fixtures/lmstudio/models-none-loaded.json`
- Modify: `src/lib.rs` (add `pub mod adapters;`)
- Test: `tests/adapters.rs`

**Interfaces:**
- Consumes: `Adapter`, `LoadedModel`, `State`, `ProbeError`, `Http`.
- Produces: `adapters::lmstudio::LmStudio` (unit struct implementing `Adapter`).

- [ ] **Step 1: Record the fixture**

This is real captured output. Create `tests/fixtures/lmstudio/models-none-loaded.json`:

```json
{
  "data": [
    {
      "id": "locateanything-3b-mlx",
      "object": "model",
      "type": "vlm",
      "publisher": "andai-labs",
      "arch": "locateanything",
      "compatibility_type": "mlx",
      "quantization": "bf16",
      "state": "not-loaded",
      "max_context_length": 32768
    },
    {
      "id": "qwen3-14b-claude-4.5-opus-high-reasoning-distill",
      "object": "model",
      "type": "llm",
      "publisher": "TeichAI",
      "arch": "qwen3",
      "compatibility_type": "gguf",
      "quantization": "Q4_K_M",
      "state": "not-loaded",
      "max_context_length": 40960,
      "capabilities": ["tool_use"]
    },
    {
      "id": "text-embedding-nomic-embed-text-v1.5",
      "object": "model",
      "type": "embeddings",
      "publisher": "nomic-ai",
      "arch": "nomic-bert",
      "compatibility_type": "gguf",
      "quantization": "Q4_K_M",
      "state": "not-loaded",
      "max_context_length": 2048
    }
  ],
  "object": "list"
}
```

Also create `tests/fixtures/lmstudio/models-one-loaded.json` — the same three
models, but with the second one loaded. Note `loaded_context_length` is present
*only* on the loaded entry, and is 8192 while `max_context_length` is 40960:

```json
{
  "data": [
    {
      "id": "locateanything-3b-mlx",
      "object": "model",
      "state": "not-loaded",
      "max_context_length": 32768
    },
    {
      "id": "qwen3-14b-claude-4.5-opus-high-reasoning-distill",
      "object": "model",
      "state": "loaded",
      "max_context_length": 40960,
      "loaded_context_length": 8192
    },
    {
      "id": "text-embedding-nomic-embed-text-v1.5",
      "object": "model",
      "state": "not-loaded",
      "max_context_length": 2048
    }
  ],
  "object": "list"
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/adapters.rs`:

```rust
mod support;

use std::time::Duration;

use llm_harmony::adapters::lmstudio::LmStudio;
use llm_harmony::http::Http;
use llm_harmony::provider::{Adapter, ProbeError, ProviderKind, State};

fn http() -> Http {
    Http::new(Duration::from_millis(1500))
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{name}")).expect("fixture exists")
}

#[test]
fn lmstudio_lists_every_model_with_its_state() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();

    assert_eq!(models.len(), 3);
    assert!(models.iter().all(|m| m.state == State::NotLoaded));
    assert_eq!(models[1].id, "qwen3-14b-claude-4.5-opus-high-reasoning-distill");
}

/// The bug that already happened once by hand. It does not get to happen again.
#[test]
fn lmstudio_never_reads_max_context_length_as_a_serving_window() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();

    for m in &models {
        assert_eq!(
            m.context_tokens, None,
            "{} is not loaded; max_context_length must not leak into context_tokens",
            m.id
        );
    }
}

#[test]
fn lmstudio_reads_the_serving_window_only_when_loaded() {
    let body = fixture("lmstudio/models-one-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();

    let loaded: Vec<_> = models.iter().filter(|m| m.state == State::Loaded).collect();
    assert_eq!(loaded.len(), 1);
    assert_eq!(
        loaded[0].context_tokens,
        Some(8192),
        "must be loaded_context_length (8192), never max_context_length (40960)"
    );
    assert!(models.iter().filter(|m| m.state != State::Loaded).all(|m| m.context_tokens.is_none()));
}

#[test]
fn lmstudio_reports_no_weights_because_it_does_not_publish_them() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();
    assert!(models.iter().all(|m| m.weights_bytes.is_none()));
}

/// LM Studio answers unknown paths with 200 and an error body, so `probe`
/// must validate shape rather than status. Verified against the real server
/// 2026-09-09.
#[test]
fn lmstudio_probe_rejects_a_two_hundred_that_is_actually_an_error() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = LmStudio.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::LmStudio });
}

#[test]
fn lmstudio_probe_accepts_a_real_model_list() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    assert!(LmStudio.probe(&http(), &s.base_url()).is_ok());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --test adapters`
Expected: FAIL — `unresolved import llm_harmony::adapters`.

- [ ] **Step 4: Write `src/adapters/mod.rs`**

```rust
pub mod llamacpp;
pub mod lmstudio;
pub mod ollama;
pub mod vllm;

use crate::provider::{Adapter, ProviderKind};

/// The adapter for a given kind. One place, so the ledger never matches on kind.
pub fn adapter_for(kind: ProviderKind) -> Box<dyn Adapter> {
    match kind {
        ProviderKind::LmStudio => Box::new(lmstudio::LmStudio),
        ProviderKind::LlamaCpp => Box::new(llamacpp::LlamaCpp),
        ProviderKind::Vllm => Box::new(vllm::Vllm),
        ProviderKind::Ollama => Box::new(ollama::Ollama),
    }
}
```

The other three modules need stubs so this compiles. Each returns `NotListening` so nothing silently reports success before its real task lands. Write all three verbatim.

`src/adapters/llamacpp.rs` — replaced by Task 6:

```rust
use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind};

pub struct LlamaCpp;

impl Adapter for LlamaCpp {
    fn kind(&self) -> ProviderKind { ProviderKind::LlamaCpp }
    fn probe(&self, _http: &Http, _base: &str) -> Result<(), ProbeError> {
        Err(ProbeError::NotListening)
    }
    fn list(&self, _http: &Http, _base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        Err(ProbeError::NotListening)
    }
}
```

`src/adapters/ollama.rs` — replaced by Task 5:

```rust
use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind};

pub struct Ollama;

impl Adapter for Ollama {
    fn kind(&self) -> ProviderKind { ProviderKind::Ollama }
    fn probe(&self, _http: &Http, _base: &str) -> Result<(), ProbeError> {
        Err(ProbeError::NotListening)
    }
    fn list(&self, _http: &Http, _base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        Err(ProbeError::NotListening)
    }
}
```

`src/adapters/vllm.rs` — replaced by Task 7:

```rust
use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind};

pub struct Vllm;

impl Adapter for Vllm {
    fn kind(&self) -> ProviderKind { ProviderKind::Vllm }
    fn probe(&self, _http: &Http, _base: &str) -> Result<(), ProbeError> {
        Err(ProbeError::NotListening)
    }
    fn list(&self, _http: &Http, _base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        Err(ProbeError::NotListening)
    }
}
```

- [ ] **Step 5: Write `src/adapters/lmstudio.rs`**

```rust
use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct LmStudio;

impl LmStudio {
    fn fetch(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/api/v0/models"))?;
        // LM Studio answers *every* unknown path with 200 and an error body,
        // so a status code proves nothing. Validate the shape instead.
        if v.get("error").is_some() || !v["data"].is_array() {
            return Err(ProbeError::KindMismatch {
                expected: ProviderKind::LmStudio,
            });
        }
        Ok(v)
    }
}

impl Adapter for LmStudio {
    fn kind(&self) -> ProviderKind {
        ProviderKind::LmStudio
    }

    fn probe(&self, http: &Http, base: &str) -> Result<(), ProbeError> {
        self.fetch(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let v = self.fetch(http, base)?;
        let entries = v["data"].as_array().expect("checked in fetch");

        Ok(entries
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                let state = match m["state"].as_str() {
                    Some("loaded") => State::Loaded,
                    Some("loading") => State::Loading,
                    _ => State::NotLoaded,
                };
                // `loaded_context_length` is the serving window and is absent
                // unless the model is resident. `max_context_length` is a
                // capability figure (40960 vs 8192 on qwen3-14b) and is never
                // read here.
                let context_tokens = match state {
                    State::Loaded => m["loaded_context_length"].as_u64().map(|n| n as u32),
                    _ => None,
                };
                Some(LoadedModel {
                    id,
                    state,
                    context_tokens,
                    // LM Studio publishes no artifact size on this endpoint or
                    // on /v1/models. Verified 2026-09-09.
                    weights_bytes: None,
                })
            })
            .collect())
    }
}
```

Add `pub mod adapters;` to `src/lib.rs`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --test adapters`
Expected: PASS — 6 tests.

- [ ] **Step 7: Commit**

```bash
git add src/adapters src/lib.rs tests/adapters.rs tests/fixtures/lmstudio
git commit -m "feat: LM Studio adapter with shape-validating probe"
```

---

### Task 5: Ollama adapter

**Files:**
- Rewrite: `src/adapters/ollama.rs`
- Create: `tests/fixtures/ollama/ps-empty.json`, `tests/fixtures/ollama/ps-one-loaded.json`, `tests/fixtures/ollama/tags.json`
- Modify: `tests/adapters.rs` (append tests)

**Interfaces:**
- Consumes: same as Task 4.
- Produces: `adapters::ollama::Ollama`.

- [ ] **Step 1: Record the fixtures**

`tests/fixtures/ollama/ps-empty.json` — captured live 2026-09-09:

```json
{"models":[]}
```

`tests/fixtures/ollama/ps-one-loaded.json` — `/api/ps` shape with a resident model. `context_length` here *is* the serving window:

```json
{
  "models": [
    {
      "name": "nomic-embed-text:latest",
      "model": "nomic-embed-text:latest",
      "size": 274302450,
      "size_vram": 274302450,
      "digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
      "details": {
        "format": "gguf",
        "family": "nomic-bert",
        "parameter_size": "137M",
        "quantization_level": "F16"
      },
      "context_length": 2048,
      "expires_at": "2026-09-09T22:05:00.000000000-04:00"
    }
  ]
}
```

`tests/fixtures/ollama/tags.json` — captured live 2026-09-09. **This endpoint is a trap** and exists in the fixtures only so a test can prove it is not read:

```json
{
  "models": [
    {
      "name": "nomic-embed-text:latest",
      "model": "nomic-embed-text:latest",
      "modified_at": "2026-06-08T19:30:33.144558419-04:00",
      "size": 274302450,
      "digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "nomic-bert",
        "families": ["nomic-bert"],
        "parameter_size": "137M",
        "quantization_level": "F16",
        "context_length": 2048,
        "embedding_length": 768
      },
      "capabilities": ["embedding"]
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/adapters.rs`:

```rust
use llm_harmony::adapters::ollama::Ollama;

#[test]
fn ollama_reports_nothing_when_nothing_is_resident() {
    let ps = fixture("ollama/ps-empty.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();
    assert!(models.iter().all(|m| m.state != State::Loaded));
}

#[test]
fn ollama_reads_serving_window_and_weights_for_a_resident_model() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();

    let m = models.iter().find(|m| m.state == State::Loaded).expect("one loaded");
    assert_eq!(m.id, "nomic-embed-text:latest");
    assert_eq!(m.context_tokens, Some(2048), "from /api/ps, the serving window");
    assert_eq!(m.weights_bytes, Some(274_302_450), "Ollama is the one provider that publishes size");
}

/// `/api/tags` carries `details.context_length` for models that are NOT
/// loaded. Same class of trap as LM Studio's `max_context_length`.
/// Verified live 2026-09-09.
#[test]
fn ollama_never_takes_a_window_from_the_tags_endpoint() {
    let ps = fixture("ollama/ps-empty.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();

    let m = models.iter().find(|m| m.id == "nomic-embed-text:latest").unwrap();
    assert_eq!(m.state, State::NotLoaded);
    assert_eq!(
        m.context_tokens, None,
        "details.context_length is 2048 in tags.json and must not be read"
    );
    assert_eq!(m.weights_bytes, Some(274_302_450), "size is a file property and IS safe to read");
}

#[test]
fn ollama_probe_rejects_a_server_that_is_not_ollama() {
    // LM Studio's port: 200 with an error body on /api/ps.
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = Ollama.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::Ollama });
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --test adapters ollama`
Expected: FAIL — the placeholder `Ollama` returns `NotListening`.

- [ ] **Step 4: Write the implementation**

Replace `src/adapters/ollama.rs`:

```rust
use std::collections::HashMap;

use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct Ollama;

impl Ollama {
    fn fetch_ps(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/api/ps"))?;
        if !v["models"].is_array() {
            return Err(ProbeError::KindMismatch {
                expected: ProviderKind::Ollama,
            });
        }
        Ok(v)
    }
}

impl Adapter for Ollama {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    fn probe(&self, http: &Http, base: &str) -> Result<(), ProbeError> {
        self.fetch_ps(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let ps = self.fetch_ps(http, base)?;

        // Resident models: /api/ps is the ONLY source of a serving window.
        let mut out: Vec<LoadedModel> = Vec::new();
        let mut resident: HashMap<String, ()> = HashMap::new();
        for m in ps["models"].as_array().expect("checked in fetch_ps") {
            let Some(id) = m["name"].as_str().or_else(|| m["model"].as_str()) else {
                continue;
            };
            resident.insert(id.to_string(), ());
            out.push(LoadedModel {
                id: id.to_string(),
                state: State::Loaded,
                context_tokens: m["context_length"].as_u64().map(|n| n as u32),
                weights_bytes: m["size"].as_u64(),
            });
        }

        // Everything on disk. `size` here is a file property and safe to read.
        // `details.context_length` is a CAPABILITY figure and is deliberately
        // not read — see tests. A failure here is not fatal: the resident list
        // is the part that matters for memory.
        if let Ok(tags) = http.get_json(&format!("{base}/api/tags")) {
            for m in tags["models"].as_array().unwrap_or(&Vec::new()) {
                let Some(id) = m["name"].as_str().or_else(|| m["model"].as_str()) else {
                    continue;
                };
                if resident.contains_key(id) {
                    continue;
                }
                out.push(LoadedModel {
                    id: id.to_string(),
                    state: State::NotLoaded,
                    context_tokens: None,
                    weights_bytes: m["size"].as_u64(),
                });
            }
        }

        Ok(out)
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test adapters`
Expected: PASS — 10 tests.

- [ ] **Step 6: Commit**

```bash
git add src/adapters/ollama.rs tests/adapters.rs tests/fixtures/ollama
git commit -m "feat: Ollama adapter reading windows only from /api/ps"
```

---

### Task 6: llama.cpp adapter

**Files:**
- Rewrite: `src/adapters/llamacpp.rs`
- Create: `tests/fixtures/llamacpp/v1-models.json`, `tests/fixtures/llamacpp/running.json`
- Modify: `tests/adapters.rs` (append tests)

**Interfaces:** Produces `adapters::llamacpp::LlamaCpp`.

> **Fixture caveat — read this.** llama.cpp was not running on this machine when
> the plan was written, so these two fixtures are constructed from the shapes
> documented in `docs/field-notes.md` rather than captured from a live server.
> They are the plan's only unverified fixtures. When llama.cpp next runs,
> re-capture `GET /v1/models` and `GET /running` and reconcile. Add
> `// UNVERIFIED — constructed from field-notes.md, re-capture when llama.cpp runs`
> at the top of each fixture and delete the comment once confirmed.

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/llamacpp/v1-models.json` — note `meta.n_ctx` (serving) beside `meta.n_ctx_train` (capability):

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-14B-Q4_K_M",
      "object": "model",
      "created": 1780961433,
      "owned_by": "llamacpp",
      "meta": {
        "n_ctx": 8192,
        "n_ctx_train": 40960,
        "n_params": 14770033664,
        "size": 8988392448
      }
    },
    {
      "id": "Qwen3-1.7B-Q8_0",
      "object": "model",
      "created": 1780961433,
      "owned_by": "llamacpp",
      "meta": {
        "n_ctx": null,
        "n_ctx_train": 40960,
        "n_params": 1720000000,
        "size": 1830000000
      }
    }
  ]
}
```

`tests/fixtures/llamacpp/running.json` — the router's list of what is resident:

```json
{
  "running": [
    { "model": "Qwen3-14B-Q4_K_M" }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/adapters.rs`:

```rust
use llm_harmony::adapters::llamacpp::LlamaCpp;

#[test]
fn llamacpp_marks_only_running_models_as_loaded() {
    let models = fixture("llamacpp/v1-models.json");
    let running = fixture("llamacpp/running.json");
    let s = support::StubServer::start(support::routes(&[
        ("/v1/models", &models),
        ("/running", &running),
    ]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();

    assert_eq!(out.len(), 2);
    let loaded: Vec<_> = out.iter().filter(|m| m.state == State::Loaded).collect();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, "Qwen3-14B-Q4_K_M");
}

/// `n_ctx` is the window being served; `n_ctx_train` is a capability figure.
#[test]
fn llamacpp_never_reads_n_ctx_train_as_a_serving_window() {
    let models = fixture("llamacpp/v1-models.json");
    let running = fixture("llamacpp/running.json");
    let s = support::StubServer::start(support::routes(&[
        ("/v1/models", &models),
        ("/running", &running),
    ]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();

    let loaded = out.iter().find(|m| m.state == State::Loaded).unwrap();
    assert_eq!(loaded.context_tokens, Some(8192), "n_ctx, not n_ctx_train (40960)");

    let idle = out.iter().find(|m| m.id == "Qwen3-1.7B-Q8_0").unwrap();
    assert_eq!(idle.context_tokens, None, "n_ctx is null when not resident");
}

#[test]
fn llamacpp_reads_artifact_size_from_meta() {
    let models = fixture("llamacpp/v1-models.json");
    let running = fixture("llamacpp/running.json");
    let s = support::StubServer::start(support::routes(&[
        ("/v1/models", &models),
        ("/running", &running),
    ]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();
    let loaded = out.iter().find(|m| m.state == State::Loaded).unwrap();
    assert_eq!(loaded.weights_bytes, Some(8_988_392_448));
}

/// A plain llama-server has no router, so /running 404s. Models are then
/// reported with an unknown state rather than the whole provider failing.
#[test]
fn llamacpp_survives_a_missing_router_endpoint() {
    let models = fixture("llamacpp/v1-models.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &models)]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();
    assert_eq!(out.len(), 2, "models still listed without /running");
    assert!(out.iter().all(|m| m.state == State::NotLoaded));
}

#[test]
fn llamacpp_probe_rejects_lmstudio_on_the_same_shape() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = LlamaCpp.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::LlamaCpp });
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --test adapters llamacpp`
Expected: FAIL — placeholder returns `NotListening`.

- [ ] **Step 4: Write the implementation**

Replace `src/adapters/llamacpp.rs`:

```rust
use std::collections::HashSet;

use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct LlamaCpp;

impl LlamaCpp {
    fn fetch_models(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/v1/models"))?;
        // Three providers answer /v1/models, so shape alone is not enough:
        // require the `meta` block llama.cpp attaches and nobody else does.
        let is_llamacpp = v["data"]
            .as_array()
            .map(|a| a.iter().any(|m| m.get("meta").is_some()))
            .unwrap_or(false);
        if !is_llamacpp {
            return Err(ProbeError::KindMismatch {
                expected: ProviderKind::LlamaCpp,
            });
        }
        Ok(v)
    }

    /// The router's resident set. Absent on a plain `llama-server`.
    fn running(&self, http: &Http, base: &str) -> HashSet<String> {
        let Ok(v) = http.get_json(&format!("{base}/running")) else {
            return HashSet::new();
        };
        v["running"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|m| m["model"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Adapter for LlamaCpp {
    fn kind(&self) -> ProviderKind {
        ProviderKind::LlamaCpp
    }

    fn probe(&self, http: &Http, base: &str) -> Result<(), ProbeError> {
        self.fetch_models(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let v = self.fetch_models(http, base)?;
        let running = self.running(http, base);

        Ok(v["data"]
            .as_array()
            .expect("checked in fetch_models")
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                let state = if running.contains(&id) {
                    State::Loaded
                } else {
                    State::NotLoaded
                };
                // meta.n_ctx is the served window and is null when not
                // resident. meta.n_ctx_train is what the model was trained
                // with and never bounds a request.
                let context_tokens = match state {
                    State::Loaded => m["meta"]["n_ctx"].as_u64().map(|n| n as u32),
                    _ => None,
                };
                Some(LoadedModel {
                    id,
                    state,
                    context_tokens,
                    weights_bytes: m["meta"]["size"].as_u64(),
                })
            })
            .collect())
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test adapters`
Expected: PASS — 15 tests.

- [ ] **Step 6: Commit**

```bash
git add src/adapters/llamacpp.rs tests/adapters.rs tests/fixtures/llamacpp
git commit -m "feat: llama.cpp adapter reading n_ctx never n_ctx_train"
```

---

### Task 7: vLLM-MLX adapter

**Files:**
- Rewrite: `src/adapters/vllm.rs`
- Create: `tests/fixtures/vllm/v1-models.json`
- Modify: `tests/adapters.rs` (append tests)

**Interfaces:** Produces `adapters::vllm::Vllm`.

> **Fixture caveat:** vLLM-MLX was not running when the plan was written. Mark
> the fixture `UNVERIFIED` and re-capture when it next runs, as in Task 6.

- [ ] **Step 1: Create the fixture**

vLLM-MLX serves a bare OpenAI-shaped list — no `meta`, no size, no window:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-9B-MLX-4bit",
      "object": "model",
      "created": 1780961433,
      "owned_by": "vllm-mlx"
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/adapters.rs`:

```rust
use llm_harmony::adapters::vllm::Vllm;

/// vLLM-MLX reports neither a window nor a size. It gets None, not a guess.
#[test]
fn vllm_reports_a_model_with_no_window_and_no_weights() {
    let body = fixture("vllm/v1-models.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, "Qwen3.5-9B-MLX-4bit");
    assert_eq!(out[0].state, State::Loaded, "presence in /v1/models is residency");
    assert_eq!(out[0].context_tokens, None, "vLLM-MLX publishes no window; never guess one");
    assert_eq!(out[0].weights_bytes, None);
}

#[test]
fn vllm_probe_rejects_lmstudio_style_error_bodies() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = Vllm.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::Vllm });
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --test adapters vllm`
Expected: FAIL — placeholder returns `NotListening`.

- [ ] **Step 4: Write the implementation**

Replace `src/adapters/vllm.rs`:

```rust
use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct Vllm;

impl Vllm {
    fn fetch(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/v1/models"))?;
        if v.get("error").is_some() || !v["data"].is_array() {
            return Err(ProbeError::KindMismatch {
                expected: ProviderKind::Vllm,
            });
        }
        Ok(v)
    }
}

impl Adapter for Vllm {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Vllm
    }

    fn probe(&self, http: &Http, base: &str) -> Result<(), ProbeError> {
        self.fetch(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let v = self.fetch(http, base)?;
        Ok(v["data"]
            .as_array()
            .expect("checked in fetch")
            .iter()
            .filter_map(|m| {
                Some(LoadedModel {
                    id: m["id"].as_str()?.to_string(),
                    // The registry only lists what it is serving, so presence
                    // is residency.
                    state: State::Loaded,
                    // vLLM-MLX reports no serving window at all. A floor would
                    // be a guess, and a guess here is the estimate-too-low case
                    // that crashes the machine.
                    context_tokens: None,
                    weights_bytes: None,
                })
            })
            .collect())
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test adapters`
Expected: PASS — 17 tests.

- [ ] **Step 6: Commit**

```bash
git add src/adapters/vllm.rs tests/adapters.rs tests/fixtures/vllm
git commit -m "feat: vLLM-MLX adapter reporting no window rather than a guess"
```

---

### Task 8: Memory attribution

**Files:**
- Create: `src/memory.rs`
- Modify: `src/lib.rs` (add `pub mod memory;`)
- Test: `tests/smoke.rs` — the only test that touches the real machine

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Machine { total_bytes, used_bytes, swap_total_bytes, swap_used_bytes }`, `Machine::read() -> Result<Machine, String>`, `port_from_url(&str) -> Option<u16>`, `pid_listening_on(u16) -> Option<u32>`, `ProcessTree { pids: Vec<u32>, footprint_bytes: Option<u64> }`, `footprint_for_port(u16) -> Option<ProcessTree>`.

- [ ] **Step 1: Write the failing test**

Create `tests/smoke.rs`:

```rust
use llm_harmony::memory::{port_from_url, Machine};

#[test]
fn machine_total_matches_sysctl_hw_memsize() {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .expect("sysctl runs on macOS");
    let expected: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap();

    let m = Machine::read().unwrap();
    assert_eq!(m.total_bytes, expected);
}

#[test]
fn machine_reports_plausible_usage() {
    let m = Machine::read().unwrap();
    assert!(m.used_bytes > 0);
    assert!(m.used_bytes <= m.total_bytes);
    assert!(m.swap_used_bytes <= m.swap_total_bytes);
}

#[test]
fn port_is_parsed_out_of_a_config_url() {
    assert_eq!(port_from_url("http://127.0.0.1:1234"), Some(1234));
    assert_eq!(port_from_url("http://127.0.0.1:11434/"), Some(11434));
    assert_eq!(port_from_url("http://localhost"), None);
    assert_eq!(port_from_url("garbage"), None);
}

/// Nothing binds port 1, so this must be None rather than a panic.
#[test]
fn an_unbound_port_has_no_pid() {
    assert_eq!(llm_harmony::memory::pid_listening_on(1), None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test smoke`
Expected: FAIL — `unresolved import llm_harmony::memory`.

- [ ] **Step 3: Write the implementation**

Create `src/memory.rs`:

```rust
use libproc::libproc::file_info::{pidfdinfo, ListFDs, ProcFDType};
use libproc::libproc::net_info::{SocketFDInfo, SocketInfoKind};
use libproc::libproc::pid_rusage::{pidrusage, RUsageInfoV2};
use libproc::libproc::proc_pid::listpidinfo;
use libproc::processes::{pids_by_type, ProcFilter};
use sysinfo::{MemoryRefreshKind, ProcessRefreshKind, RefreshKind, System};

/// Machine-wide totals. `total_bytes` equals `sysctl hw.memsize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Machine {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
}

impl Machine {
    pub fn read() -> Result<Machine, String> {
        let sys = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        let total = sys.total_memory();
        if total == 0 {
            return Err("cannot read machine memory total".to_string());
        }
        Ok(Machine {
            total_bytes: total,
            used_bytes: sys.used_memory(),
            swap_total_bytes: sys.total_swap(),
            swap_used_bytes: sys.used_swap(),
        })
    }

    pub fn free_fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.total_bytes.saturating_sub(self.used_bytes)) as f64 / self.total_bytes as f64
    }
}

/// The processes attributable to one provider, and what they cost.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ProcessTree {
    pub pids: Vec<u32>,
    /// `None` when no pid's usage could be read — e.g. a root-owned process.
    /// Never silently zero.
    pub footprint_bytes: Option<u64>,
}

pub fn port_from_url(url: &str) -> Option<u16> {
    let after_scheme = url.split("://").nth(1)?;
    let hostport = after_scheme.split('/').next()?;
    hostport.rsplit(':').next()?.parse().ok()
}

/// The pid holding a listening TCP socket on `port`.
///
/// Uniform across all four providers, which is why this is here and not on the
/// `Adapter` trait: verified 2026-09-09, pid 647 owns :1234 (LM Studio) and pid
/// 4140 owns :11434 (Ollama), with model memory in their descendants.
pub fn pid_listening_on(port: u16) -> Option<u32> {
    let all = pids_by_type(ProcFilter::All).ok()?;
    for pid in all {
        let Ok(fds) = listpidinfo::<ListFDs>(pid as i32, 4096) else {
            continue;
        };
        for fd in fds {
            if !matches!(fd.proc_fdtype.into(), ProcFDType::Socket) {
                continue;
            }
            let Ok(sock) = pidfdinfo::<SocketFDInfo>(pid as i32, fd.proc_fd) else {
                continue;
            };
            if !matches!(sock.psi.soi_kind.into(), SocketInfoKind::Tcp) {
                continue;
            }
            // SAFETY: the union discriminant was just checked to be Tcp.
            let tcp = unsafe { sock.psi.soi_proto.pri_tcp };
            if u16::from_be(tcp.tcpsi_ini.insi_lport as u16) == port {
                return Some(pid);
            }
        }
    }
    None
}

fn descendants(sys: &System, root: u32) -> Vec<u32> {
    let mut out = vec![root];
    let mut frontier = vec![root];
    while let Some(parent) = frontier.pop() {
        for (pid, proc_) in sys.processes() {
            if proc_.parent().map(|p| p.as_u32()) == Some(parent) {
                let child = pid.as_u32();
                if !out.contains(&child) {
                    out.push(child);
                    frontier.push(child);
                }
            }
        }
    }
    out
}

/// `phys_footprint` for a pid. On Apple Silicon this includes Metal buffers in
/// unified memory, which is what makes it the right metric here — sysinfo's
/// `Process::memory()` is resident size and undercounts by roughly 3x on the
/// GPU helper processes that actually hold model weights (verified 2026-09-09:
/// LM Studio pid 647 reported 104 MB resident against 285 MB footprint).
fn phys_footprint(pid: u32) -> Option<u64> {
    pidrusage::<RUsageInfoV2>(pid as i32)
        .ok()
        .map(|r| r.ri_phys_footprint)
}

/// The process tree serving `port`, and its summed footprint.
///
/// Note this includes a provider's non-model overhead — LM Studio idles at
/// ~0.62 GB across five Electron processes with nothing loaded. That is real
/// memory the machine cannot give to a model, so it belongs in the total.
pub fn footprint_for_port(port: u16) -> Option<ProcessTree> {
    let root = pid_listening_on(port)?;
    let sys = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
    );
    let pids = descendants(&sys, root);

    let mut total: u64 = 0;
    let mut any = false;
    for pid in &pids {
        if let Some(bytes) = phys_footprint(*pid) {
            total += bytes;
            any = true;
        }
    }

    Some(ProcessTree {
        pids,
        footprint_bytes: any.then_some(total),
    })
}
```

Add `pub mod memory;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test smoke`
Expected: PASS — 4 tests.

- [ ] **Step 5: Confirm port→pid resolves against the live machine**

The smoke tests only prove an *unbound* port yields `None`. Confirm the positive case too, since it is the part that silently returns `None` if the libproc union handling is wrong:

```bash
cargo test --test smoke -- --nocapture
lsof -nP -iTCP:1234 -sTCP:LISTEN | awk 'NR>1 {print "lsof says pid", $2}'
```

Expected: smoke tests pass, and `lsof` names a pid for LM Studio's port. If `lsof` reports a pid, `pid_listening_on(1234)` must find the same one — verified 2026-09-09 as pid 647. If nothing is on 1234, start LM Studio or skip this check.

- [ ] **Step 6: Commit**

```bash
git add src/memory.rs src/lib.rs tests/smoke.rs
git commit -m "feat: memory attribution via port to pid tree phys_footprint"
```

---

### Task 9: Ledger assembly

**Files:**
- Create: `src/ledger.rs`
- Modify: `src/lib.rs` (add `pub mod ledger;`)
- Test: `tests/integration.rs`

**Interfaces:**
- Consumes: `Config`, `adapter_for`, `Http`, `Machine`, `footprint_for_port`.
- Produces: `ProviderRow { kind, url, outcome, pids, footprint_bytes }` with methods `loaded()`, `weights_bytes()`, `gap_bytes()`; `Outcome::{Ok(Vec<LoadedModel>), Failed(ProbeError)}`; `Ledger { machine, rows }` with `total_footprint_bytes()`; `Ledger::assemble(&Config, &Http, Machine) -> Ledger`; and `Ledger::assemble_with(&Config, &Http, Machine, F) -> Ledger` where `F: Fn(u16) -> Option<(Vec<u32>, Option<u64>)> + Sync`, so tests never touch real processes.

**On `probe` vs `list`:** the ledger calls `list` only. Each adapter's private `fetch` performs the shape validation, and `list` and `probe` both go through it — so a kind mismatch surfaces through `list` without a second HTTP round trip per provider. `probe` remains on the trait as the explicit kind-confirmation entry point (it is what a future `llm-harmony doctor` calls) and is exercised directly by the Task 4–7 tests.

- [ ] **Step 1: Write the failing test**

Create `tests/integration.rs`:

```rust
mod support;

use std::time::Duration;

use llm_harmony::config::{Config, ProviderConfig};
use llm_harmony::http::Http;
use llm_harmony::ledger::{Ledger, Outcome};
use llm_harmony::memory::Machine;
use llm_harmony::provider::{ProbeError, ProviderKind, State};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{name}")).expect("fixture exists")
}

fn machine() -> Machine {
    Machine {
        total_bytes: 25_769_803_776,
        used_bytes: 17_343_315_968,
        swap_total_bytes: 6_442_450_944,
        swap_used_bytes: 4_749_852_672,
    }
}

#[test]
fn a_provider_that_is_down_is_a_row_not_an_error() {
    let cfg = Config {
        providers: vec![ProviderConfig {
            kind: ProviderKind::LlamaCpp,
            // Nothing binds port 1.
            url: "http://127.0.0.1:1".to_string(),
        }],
    };
    let ledger = Ledger::assemble_with(&cfg, &Http::new(Duration::from_millis(500)), machine(), |_| None);

    assert_eq!(ledger.rows.len(), 1);
    assert!(matches!(
        ledger.rows[0].outcome,
        Outcome::Failed(ProbeError::NotListening)
    ));
}

#[test]
fn two_live_providers_are_both_assembled() {
    let lms_body = fixture("lmstudio/models-one-loaded.json");
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");

    let lms = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &lms_body,
    )]));
    let oll = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));

    let cfg = Config {
        providers: vec![
            ProviderConfig { kind: ProviderKind::LmStudio, url: lms.base_url() },
            ProviderConfig { kind: ProviderKind::Ollama, url: oll.base_url() },
        ],
    };
    let ledger = Ledger::assemble_with(
        &cfg,
        &Http::new(Duration::from_millis(1500)),
        machine(),
        |_port| Some((vec![647], Some(9_800_000_000))),
    );

    assert_eq!(ledger.rows.len(), 2);
    assert_eq!(ledger.rows[0].kind, ProviderKind::LmStudio, "reporting order is stable");

    let loaded_total: usize = ledger
        .rows
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Ok(models) => Some(models.iter().filter(|m| m.state == State::Loaded).count()),
            Outcome::Failed(_) => None,
        })
        .sum();
    assert_eq!(loaded_total, 2, "one loaded in each provider");
}

#[test]
fn provider_footprints_sum_into_a_total() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let oll = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));

    let cfg = Config {
        providers: vec![ProviderConfig { kind: ProviderKind::Ollama, url: oll.base_url() }],
    };
    let ledger = Ledger::assemble_with(
        &cfg,
        &Http::new(Duration::from_millis(1500)),
        machine(),
        |_port| Some((vec![4140], Some(400_000_000))),
    );

    assert_eq!(ledger.total_footprint_bytes(), Some(400_000_000));
}

/// A provider whose pids cannot be read still lists its models.
#[test]
fn an_unreadable_footprint_does_not_hide_the_models() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let oll = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));

    let cfg = Config {
        providers: vec![ProviderConfig { kind: ProviderKind::Ollama, url: oll.base_url() }],
    };
    let ledger = Ledger::assemble_with(
        &cfg,
        &Http::new(Duration::from_millis(1500)),
        machine(),
        |_port| Some((vec![4140], None)),
    );

    assert_eq!(ledger.rows[0].footprint_bytes, None);
    assert!(matches!(&ledger.rows[0].outcome, Outcome::Ok(m) if !m.is_empty()));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test integration`
Expected: FAIL — `unresolved import llm_harmony::ledger`.

- [ ] **Step 3: Write the implementation**

Create `src/ledger.rs`:

```rust
use crate::adapters::adapter_for;
use crate::config::Config;
use crate::http::Http;
use crate::memory::{footprint_for_port, port_from_url, Machine};
use crate::provider::{LoadedModel, ProbeError, ProviderKind, State};

#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum Outcome {
    Ok(Vec<LoadedModel>),
    Failed(ProbeError),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProviderRow {
    pub kind: ProviderKind,
    pub url: String,
    pub outcome: Outcome,
    pub pids: Vec<u32>,
    pub footprint_bytes: Option<u64>,
}

impl ProviderRow {
    pub fn loaded(&self) -> Vec<&LoadedModel> {
        match &self.outcome {
            Outcome::Ok(models) => models.iter().filter(|m| m.state == State::Loaded).collect(),
            Outcome::Failed(_) => Vec::new(),
        }
    }

    /// Summed weights of resident models — `None` unless *every* resident model
    /// publishes a size. A partial sum would understate, and understating is
    /// the failure mode that crashes the machine.
    pub fn weights_bytes(&self) -> Option<u64> {
        let loaded = self.loaded();
        if loaded.is_empty() {
            return None;
        }
        loaded.iter().map(|m| m.weights_bytes).sum()
    }

    /// KV cache, prefix cache, and activations: what the weights figure omits.
    pub fn gap_bytes(&self) -> Option<u64> {
        let fp = self.footprint_bytes?;
        let w = self.weights_bytes()?;
        Some(fp.saturating_sub(w))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Ledger {
    pub machine: Machine,
    pub rows: Vec<ProviderRow>,
}

impl Ledger {
    /// Poll every configured provider. Concurrent because four sequential
    /// timeouts against dead ports would otherwise dominate the runtime.
    pub fn assemble(config: &Config, http: &Http, machine: Machine) -> Ledger {
        Self::assemble_with(config, http, machine, |port| {
            footprint_for_port(port).map(|t| (t.pids, t.footprint_bytes))
        })
    }

    /// `footprint` is injected so tests never touch real processes.
    pub fn assemble_with<F>(config: &Config, http: &Http, machine: Machine, footprint: F) -> Ledger
    where
        F: Fn(u16) -> Option<(Vec<u32>, Option<u64>)> + Sync,
    {
        let footprint = &footprint;
        let rows = std::thread::scope(|scope| {
            let handles: Vec<_> = config
                .providers
                .iter()
                .map(|p| {
                    scope.spawn(move || {
                        let adapter = adapter_for(p.kind);
                        let outcome = match adapter.list(http, &p.url) {
                            Ok(models) => Outcome::Ok(models),
                            Err(e) => Outcome::Failed(e),
                        };
                        // Only attribute memory to a provider that answered.
                        let (pids, footprint_bytes) = match (&outcome, port_from_url(&p.url)) {
                            (Outcome::Ok(_), Some(port)) => {
                                footprint(port).unwrap_or((Vec::new(), None))
                            }
                            _ => (Vec::new(), None),
                        };
                        ProviderRow {
                            kind: p.kind,
                            url: p.url.clone(),
                            outcome,
                            pids,
                            footprint_bytes,
                        }
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect::<Vec<_>>()
        });

        Ledger { machine, rows }
    }

    /// `None` only when no provider reported a readable footprint.
    pub fn total_footprint_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for r in &self.rows {
            if let Some(b) = r.footprint_bytes {
                total += b;
                any = true;
            }
        }
        any.then_some(total)
    }
}
```

Add `pub mod ledger;` to `src/lib.rs`. Note `Machine` must derive `Clone, Copy` (it does, from Task 8).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test integration`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ledger.rs src/lib.rs tests/integration.rs
git commit -m "feat: concurrent ledger assembly across providers"
```

---

### Task 10: Rendering

**Files:**
- Create: `src/render.rs`
- Modify: `src/lib.rs` (add `pub mod render;`)
- Test: unit tests inside `src/render.rs`

**Interfaces:**
- Consumes: `Ledger`, `ProviderRow`, `Outcome`, `Machine`.
- Produces: `render_table(&Ledger) -> String`, `render_json(&Ledger) -> String`, `human_bytes(u64) -> String`.

- [ ] **Step 1: Write the failing test**

Create `src/render.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Machine;
    use crate::provider::{LoadedModel, ProbeError, ProviderKind, State};
    use crate::ledger::{Ledger, Outcome, ProviderRow};

    fn machine() -> Machine {
        Machine {
            total_bytes: 25_769_803_776,
            used_bytes: 17_343_315_968,
            swap_total_bytes: 6_442_450_944,
            swap_used_bytes: 4_749_852_672,
        }
    }

    fn loaded(id: &str, weights: Option<u64>) -> LoadedModel {
        LoadedModel {
            id: id.to_string(),
            state: State::Loaded,
            context_tokens: Some(8192),
            weights_bytes: weights,
        }
    }

    fn ledger() -> Ledger {
        Ledger {
            machine: machine(),
            rows: vec![
                ProviderRow {
                    kind: ProviderKind::LmStudio,
                    url: "http://127.0.0.1:1234".into(),
                    outcome: Outcome::Ok(vec![loaded("qwen3-14b", None)]),
                    pids: vec![647],
                    footprint_bytes: Some(9_800_000_000),
                },
                ProviderRow {
                    kind: ProviderKind::Ollama,
                    url: "http://127.0.0.1:11434".into(),
                    outcome: Outcome::Ok(vec![loaded("nomic-embed-text:latest", Some(274_302_450))]),
                    pids: vec![4140],
                    footprint_bytes: Some(400_000_000),
                },
                ProviderRow {
                    kind: ProviderKind::LlamaCpp,
                    url: "http://127.0.0.1:8080".into(),
                    outcome: Outcome::Failed(ProbeError::NotListening),
                    pids: vec![],
                    footprint_bytes: None,
                },
            ],
        }
    }

    /// Binary units, so `hw.memsize` reads as the 24G everyone calls it.
    #[test]
    fn bytes_render_in_readable_units() {
        assert_eq!(human_bytes(0), "0B");
        assert_eq!(human_bytes(274_302_450), "261.6M");
        assert_eq!(human_bytes(9_800_000_000), "9.1G");
        assert_eq!(human_bytes(25_769_803_776), "24.0G");
    }

    #[test]
    fn a_down_provider_says_not_running_and_shows_no_numbers() {
        let out = render_table(&ledger());
        let line = out.lines().find(|l| l.contains("llamacpp")).unwrap();
        assert!(line.contains("not running"), "got: {line}");
        assert!(!line.contains('G'), "a down provider must not show a size: {line}");
    }

    #[test]
    fn a_provider_without_published_weights_shows_a_question_mark() {
        let out = render_table(&ledger());
        let line = out.lines().find(|l| l.contains("lmstudio")).unwrap();
        assert!(line.contains("9.1G"), "footprint is always known: {line}");
        assert!(line.contains('?'), "weights are unknown for LM Studio: {line}");
    }

    #[test]
    fn a_provider_with_weights_shows_the_gap() {
        let out = render_table(&ledger());
        let line = out.lines().find(|l| l.contains("ollama")).unwrap();
        assert!(line.contains("261.6M"), "weights: {line}");
        assert!(line.contains("+119.9M"), "gap is footprint minus weights, signed: {line}");
    }

    #[test]
    fn the_machine_line_names_swap() {
        let out = render_table(&ledger());
        assert!(out.contains("24.0G total"), "{out}");
        assert!(out.contains("swap"), "{out}");
    }

    #[test]
    fn json_is_valid_and_carries_errors() {
        let json = render_json(&ledger());
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["rows"].as_array().unwrap().len(), 3);
        assert_eq!(v["rows"][2]["outcome"]["error"], "not-listening");
        assert_eq!(v["machine"]["total_bytes"], 25_769_803_776u64);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib render`
Expected: FAIL — `cannot find function human_bytes`.

- [ ] **Step 3: Write the implementation above the test module**

```rust
use crate::ledger::{Ledger, Outcome};

/// Binary units. `hw.memsize` is 25,769,803,776, which is 24.0 GiB and 25.8 GB
/// — and every document in this project, plus the machine's own spec sheet,
/// calls it a 24 GB machine. Rendering 25.8G would contradict all of them.
pub fn human_bytes(b: u64) -> String {
    const UNITS: [(u64, &str); 4] = [
        (1024 * 1024 * 1024 * 1024, "T"),
        (1024 * 1024 * 1024, "G"),
        (1024 * 1024, "M"),
        (1024, "K"),
    ];
    for (scale, suffix) in UNITS {
        if b >= scale {
            return format!("{:.1}{suffix}", b as f64 / scale as f64);
        }
    }
    format!("{b}B")
}

fn opt(v: Option<u64>) -> String {
    v.map(human_bytes).unwrap_or_else(|| "?".to_string())
}

pub fn render_table(ledger: &Ledger) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<10} {:>7} {:>11} {:>9} {:>8}\n",
        "provider", "loaded", "footprint", "weights", "gap"
    ));

    for row in &ledger.rows {
        match &row.outcome {
            Outcome::Failed(e) => {
                out.push_str(&format!("{:<10} {:>7} {}\n", row.kind.as_str(), "—", e));
            }
            Outcome::Ok(_) => {
                let gap = row
                    .gap_bytes()
                    .map(|g| format!("+{}", human_bytes(g)))
                    .unwrap_or_else(|| "?".to_string());
                out.push_str(&format!(
                    "{:<10} {:>7} {:>11} {:>9} {:>8}\n",
                    row.kind.as_str(),
                    row.loaded().len(),
                    opt(row.footprint_bytes),
                    opt(row.weights_bytes()),
                    gap,
                ));
            }
        }
    }

    out.push_str(&"─".repeat(50));
    out.push('\n');
    out.push_str(&format!(
        "{:<10} {:>7} {:>11}\n",
        "providers",
        "",
        opt(ledger.total_footprint_bytes())
    ));

    let m = &ledger.machine;
    out.push_str(&format!(
        "{:<10} {} total · {:.0}% free · swap {} used\n",
        "machine",
        human_bytes(m.total_bytes),
        m.free_fraction() * 100.0,
        human_bytes(m.swap_used_bytes),
    ));

    out
}

pub fn render_json(ledger: &Ledger) -> String {
    serde_json::to_string_pretty(ledger).expect("ledger is serialisable")
}
```

Add `pub mod render;` to `src/lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib render`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/render.rs src/lib.rs
git commit -m "feat: table and json rendering with an explicit unknown marker"
```

---

### Task 11: CLI wiring and live verification

**Files:**
- Create: `src/main.rs`

**Interfaces:**
- Consumes: everything.
- Produces: the `llm-harmony` binary with `status [--json] [--config PATH] [--timeout-ms N]`.

- [ ] **Step 1: Write `src/main.rs`**

```rust
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};

use llm_harmony::config::Config;
use llm_harmony::http::Http;
use llm_harmony::ledger::Ledger;
use llm_harmony::memory::Machine;
use llm_harmony::render::{render_json, render_table};

#[derive(Parser)]
#[command(name = "llm-harmony", about = "Memory accounting across local inference providers")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show what every provider currently has loaded, and what it costs.
    Status {
        #[arg(long)]
        json: bool,
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1500)]
        timeout_ms: u64,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Status { json, config, timeout_ms } => {
            // Only two things may fail the command: a broken config and an
            // unreadable machine total. No provider can.
            let config = match Config::load(config.as_deref()) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let machine = match Machine::read() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };

            let http = Http::new(Duration::from_millis(timeout_ms));
            let ledger = Ledger::assemble(&config, &http, machine);

            if json {
                println!("{}", render_json(&ledger));
            } else {
                print!("{}", render_table(&ledger));
            }
            ExitCode::SUCCESS
        }
    }
}
```

- [ ] **Step 2: Build and run against the live machine**

Run: `cargo run -- status`

Expected: a table with `lmstudio` and `ollama` reporting real footprints, and `llamacpp` / `vllm` reporting `not running`. Machine line shows `24.0G total`.

This satisfies spec criterion 1. Note LM Studio will report ~0.6G of footprint with zero models loaded — that is its five Electron processes, and it is correct: real memory the machine cannot give to a model.

- [ ] **Step 3: Verify the fail-open contract**

Run:

```bash
cargo run -- status; echo "exit=$?"
cargo run -- status --json | python3 -m json.tool > /dev/null && echo "json ok"
cargo run -- status --config /nonexistent.toml; echo "exit=$?"
```

Expected: first `exit=0` despite two providers being down; `json ok`; third `exit=1` with a message naming the path.

- [ ] **Step 4: Verify the safety property by inspection**

Run: `grep -rniE "unload|evict|delete|/api/models/unload|ollama stop|lms unload" src/`

Expected: matches only in comments explaining what slice 1 does *not* do. No HTTP call, no `Command::new`, no trait method.

- [ ] **Step 5: Run the whole suite**

Run: `cargo test`
Expected: PASS — all tests across `--lib`, `http`, `adapters`, `integration`, `smoke`.

- [ ] **Step 6: Verify spec criterion 2 — a provider appears with no config change**

1. Run `cargo run -- status` and confirm `llamacpp` reports `not running`.
2. Start a llama.cpp server on port 8080 with any model.
3. Run `cargo run -- status` again, changing nothing else.

Expected: `llamacpp` becomes a real row with a footprint, purely from the built-in default port table. If it instead reports `kind mismatch`, the `meta`-block discrimination in Task 6 is wrong for this build — re-capture the fixture and reconcile, per the fixture caveat.

If llama.cpp cannot be started right now, this step is deferred, not skipped — record it as outstanding rather than marking the task done.

- [ ] **Step 7: Verify criterion 3 from the spec, by hand**

1. Run `cargo run -- status` and note LM Studio's footprint and that every model shows no window.
2. In LM Studio, load `qwen3-14b-claude-4.5-opus-high-reasoning-distill`.
3. Run `cargo run -- status` again.

Expected: that model's `state` becomes loaded, `context_tokens` becomes the real
serving window (the field notes saw 8192 against a `max_context_length` of
40960), and `footprint` rises by roughly the model's size. Record both outputs
in the commit message.

- [ ] **Step 8: Commit**

```bash
git add src/main.rs
git commit -m "feat: llm-harmony status command"
```

---

## Post-Implementation

- [ ] **Re-capture the unverified fixtures.** Tasks 6 and 7 ship fixtures constructed from `field-notes.md` rather than captured live. Next time llama.cpp and vLLM-MLX are running, capture `GET /v1/models`, `GET /running` (llama.cpp) and `GET /v1/models` (vLLM-MLX), reconcile against the fixtures, and remove the `UNVERIFIED` comments.
- [ ] **Append the new finding to `docs/field-notes.md`.** LM Studio returns `200` with `{"error":"Unexpected endpoint or method. (GET /path)"}` for any unknown path — a fourth instance of "failure reported inside a 200", and the first on a control surface rather than a completion surface. Also: Ollama's `/api/tags` `details.context_length` is a capability figure on an unloaded model, the same trap as `max_context_length`.
- [ ] **Record the idle baseline.** LM Studio idles at ~0.62 GB across five processes with nothing loaded (verified 2026-09-09). Worth noting in the design doc, because slice 2's estimator must subtract it to attribute memory to models rather than to Electron.

## Deferred to later slices

Explicitly not in this plan, and not gaps:

- The `ASK`/`COMMIT`/`RELEASE` socket protocol (`design.md` §4).
- The footprint estimator, measured or computed (`design.md` §5).
- Eviction and the `observe`/`evict`/`manage` modes (`design.md` §6).
- The daemon, its poll loop, and persistence (`design.md` §3).
- The entire inventory subsystem (`inventory.md`).
- Resolving a model id to a path on disk — which is what would give weights for the three providers that do not publish a size.
