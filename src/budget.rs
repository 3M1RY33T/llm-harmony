//! What each provider is allowed to take, read from wherever that limit
//! actually lives.
//!
//! Harmony's own config carries none of these, deliberately:
//! [`crate::config::ProviderConfig::start`] says every flag that matters lives
//! in the user's own script, because "reproducing them here would be a second,
//! worse source of truth." So this module reads the first source rather than
//! keeping a copy -- a provider's argv, a provider's environment, a provider's
//! API, a provider's config file, one each.
//!
//! ## The four are not comparable, and there is no total
//!
//! Probed live 2026-09-11 with all four running:
//!
//! | Provider | Ceiling | Where | Unit |
//! |---|---|---|---|
//! | vLLM-MLX | `memory_budget_gb: 14.0` | `/v1/status` | bytes |
//! | llama.cpp | `--models-max 1` | process argv | count |
//! | Ollama | `OLLAMA_MAX_LOADED_MODELS` unset | process environ | count, unknown |
//! | LM Studio | none; JIT on, idle TTL off | its own config files | neither |
//!
//! One ceiling is in bytes, two are counts, one of those counts is Ollama's own
//! undocumented default, and the fourth provider has no ceiling in either unit.
//! **Nothing here sums them.** A total would be a confident wrong number, which
//! is the failure this project exists to avoid -- and the fact worth printing is
//! not a sum but that the only byte ceiling on a 24 GB machine is vLLM-MLX's
//! 14 GiB, while nothing bounds the other three in bytes at all.
//!
//! ## Why the bias inverts here
//!
//! Everywhere else in the project an uncertain figure is rounded *up*, because
//! under-estimating a cost is what wedges a machine. A ceiling is the opposite:
//! over-stating what a provider is allowed to take *under*-states the danger.
//! So vLLM-MLX's `gb` is read as GiB -- the larger reading -- and an unreadable
//! limit is `Unknown` rather than a default someone hoped for.

use crate::provider::ProviderKind;

/// How much a provider may hold, in whichever unit it expresses.
///
/// Adjacently tagged, for `ledger::Outcome`'s reason -- a consumer branches on
/// `kind` rather than on what shape `value` happens to be -- and because serde
/// cannot internally-tag a variant whose payload is a bare integer, which a
/// byte ceiling is.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum Ceiling {
    /// A real memory limit. Only vLLM-MLX publishes one.
    Bytes(u64),
    /// A limit on how *many* models, which says nothing about their size.
    Count(u32),
    /// No limit in either unit. LM Studio: JIT loading with no cap, so the
    /// only thing bounding it is the machine -- which is the failure in this
    /// project's opening screenshot.
    Unbounded,
    /// The limit exists and harmony cannot read it. Never replaced by a
    /// guess: `design.md` §5's rule for a footprint applies to a ceiling too.
    Unknown { why: String },
}

/// A ceiling and where it was found, so the report can be audited rather than
/// believed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Budget {
    pub provider: ProviderKind,
    pub ceiling: Ceiling,
    pub source: String,
}

/// Everything harmony managed to read about one provider.
///
/// Gathered by the caller so every decision below is a pure function of bytes
/// that were actually observed -- the same split `actuate::plan` uses.
#[derive(Debug, Clone, Default)]
pub struct Readings {
    pub running: bool,
    /// The provider process's argv, when it could be read.
    pub argv: Vec<String>,
    /// The provider process's environment, as `KEY=VALUE` strings.
    pub environ: Vec<String>,
    /// vLLM-MLX's `/v1/status` body.
    pub status: Option<serde_json::Value>,
    /// LM Studio's `http-server-config.json`.
    pub lmstudio_config: Option<String>,
}

const LLAMACPP_FLAG: &str = "--models-max";
const OLLAMA_VAR: &str = "OLLAMA_MAX_LOADED_MODELS";
const VLLM_FIELD: &str = "memory_budget_gb";

/// llama.cpp's router takes its cap on the command line and publishes it
/// nowhere else. The flag is in the user's start script, which is exactly
/// where `ProviderConfig::start` says it belongs.
pub fn from_argv(argv: &[String]) -> Ceiling {
    match argv.iter().position(|a| a == LLAMACPP_FLAG) {
        Some(i) => match argv.get(i + 1).and_then(|v| v.parse::<u32>().ok()) {
            Some(n) => Ceiling::Count(n),
            None => Ceiling::Unknown {
                why: format!("{LLAMACPP_FLAG} was given without a number"),
            },
        },
        None if argv.is_empty() => Ceiling::Unknown {
            why: "the process command line could not be read".to_string(),
        },
        // Not passed at all: llama.cpp applies its own default, and harmony
        // does not know what that is on this build.
        None => Ceiling::Unknown {
            why: format!("{LLAMACPP_FLAG} is not set; llama.cpp's own default applies"),
        },
    }
}

/// Ollama's cap is an environment variable, and on this machine it is simply
/// not set -- measured 2026-09-11, where only `OLLAMA_MODELS` and
/// `OLLAMA_NO_CLOUD` were present. Its own default then applies, and that
/// default is not published anywhere harmony can read.
pub fn from_environ(environ: &[String]) -> Ceiling {
    if environ.is_empty() {
        return Ceiling::Unknown {
            why: "the process environment could not be read".to_string(),
        };
    }
    let found = environ
        .iter()
        .find_map(|e| e.strip_prefix(&format!("{OLLAMA_VAR}=")))
        .map(|v| v.trim());
    match found {
        Some(v) => match v.parse::<u32>() {
            Ok(n) => Ceiling::Count(n),
            Err(_) => Ceiling::Unknown { why: format!("{OLLAMA_VAR}={v} is not a number") },
        },
        None => Ceiling::Unknown {
            why: format!("{OLLAMA_VAR} is unset; Ollama's own default applies"),
        },
    }
}

/// The only byte ceiling any of the four publishes.
///
/// `gb` is read as **GiB**, which is both the larger reading -- over-stating a
/// ceiling under-states the danger -- and what the vLLM-MLX adapter already
/// does with the sibling `memory_gb` field on the same objects
/// (`adapters/vllm.rs`, `const GIB`). Two readings of one unit inside one
/// provider would be worse than either.
pub fn from_status(status: &serde_json::Value) -> Ceiling {
    match status["model_manager"][VLLM_FIELD].as_f64() {
        Some(gb) if gb > 0.0 => Ceiling::Bytes((gb * 1024.0 * 1024.0 * 1024.0) as u64),
        Some(_) => Ceiling::Unbounded,
        None => Ceiling::Unknown {
            why: format!("{VLLM_FIELD} was not in /v1/status"),
        },
    }
}

/// LM Studio has no ceiling in either unit.
///
/// It has a *policy*: `justInTimeModelLoading` decides whether a request may
/// load a model, and a per-model `uiTTL` decides whether one is dropped when
/// idle. Measured 2026-09-11: JIT on, and every model's `uiTTL.enabled` false.
/// Neither of those caps anything, which is the finding rather than a gap --
/// LM Studio is the provider that held 8.9 G in this project's opening
/// screenshot, and nothing was configured to stop it.
pub fn from_lmstudio_config(config: Option<&str>) -> Ceiling {
    match config {
        // Whatever the file says, none of it is a cap. The distinction worth
        // keeping is only between having read it and not.
        Some(_) => Ceiling::Unbounded,
        None => Ceiling::Unknown {
            why: "LM Studio's http-server-config.json could not be read".to_string(),
        },
    }
}

/// The ceiling for one provider, with the source that produced it.
pub fn budget(kind: ProviderKind, r: &Readings) -> Budget {
    if !r.running {
        return Budget {
            provider: kind,
            ceiling: Ceiling::Unknown { why: "not running".to_string() },
            source: "\u{2014}".to_string(),
        };
    }
    let (ceiling, source) = match kind {
        ProviderKind::LlamaCpp => (from_argv(&r.argv), format!("{LLAMACPP_FLAG}, from the argv")),
        ProviderKind::Ollama => (from_environ(&r.environ), format!("{OLLAMA_VAR}, from the environment")),
        ProviderKind::Vllm => match &r.status {
            Some(s) => (from_status(s), format!("{VLLM_FIELD}, from /v1/status")),
            None => (
                Ceiling::Unknown { why: "/v1/status did not answer".to_string() },
                "/v1/status".to_string(),
            ),
        },
        ProviderKind::LmStudio => (
            from_lmstudio_config(r.lmstudio_config.as_deref()),
            "JIT on, idle TTL off".to_string(),
        ),
    };
    Budget { provider: kind, ceiling, source }
}

/// Gather what can be read about one provider. The only I/O in this module.
///
/// Untested by design, like `memory.rs`'s samplers: every decision made from
/// these bytes is a pure function above, and a test of this function would be
/// a test of the machine it runs on.
///
/// A provider can be several processes -- LM Studio is five -- so each pid is
/// tried in turn and the first whose argv or environment actually yields a
/// ceiling wins. Where none does, the last readable one is kept, so the report
/// can distinguish *the limit is unset* from *the process could not be read*.
pub fn read(row: &crate::ledger::ProviderRow, http: &crate::http::Http) -> Readings {
    use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

    let running = matches!(row.outcome, crate::ledger::Outcome::Ok(_));
    let mut r = Readings { running, ..Default::default() };
    if !running {
        return r;
    }

    let sys = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
    );
    for pid in &row.pids {
        let Some(p) = sys.process(Pid::from_u32(*pid)) else { continue };
        let argv: Vec<String> =
            p.cmd().iter().map(|s| s.to_string_lossy().into_owned()).collect();
        let environ: Vec<String> =
            p.environ().iter().map(|s| s.to_string_lossy().into_owned()).collect();
        let decisive = match row.kind {
            ProviderKind::LlamaCpp => argv.iter().any(|a| a == LLAMACPP_FLAG),
            ProviderKind::Ollama => environ.iter().any(|e| e.starts_with(OLLAMA_VAR)),
            _ => false,
        };
        if decisive || r.argv.is_empty() {
            r.argv = argv;
            r.environ = environ;
        }
        if decisive {
            break;
        }
    }

    match row.kind {
        // The one ceiling that is published rather than configured. This is a
        // second request to an endpoint `status` already polled, which is the
        // price of not widening the `Adapter` trait for a value one provider
        // publishes and one opt-in flag consumes.
        ProviderKind::Vllm => {
            r.status = http.get_json(&format!("{}/v1/status", row.url)).ok();
        }
        ProviderKind::LmStudio => {
            r.lmstudio_config = std::env::var_os("HOME").and_then(|h| {
                std::fs::read_to_string(
                    std::path::PathBuf::from(h).join(".lmstudio/.internal/http-server-config.json"),
                )
                .ok()
            });
        }
        _ => {}
    }
    r
}

/// How many models this provider may hold at once, when that is known.
///
/// `fit` needs this and nothing else from the module: a count ceiling is the
/// one that can refuse a set the machine has room for. `None` covers both
/// "unbounded" and "unknown", which are different facts with the same
/// consequence here -- neither of them refuses anything.
pub fn count_limit(b: &Budget) -> Option<u32> {
    match b.ceiling {
        Ceiling::Count(n) => Some(n),
        _ => None,
    }
}

impl Ceiling {
    /// What to print in a table. `?` for unknown, matching `status`'s existing
    /// convention for `weights` and `gap`.
    pub fn cell(&self) -> String {
        match self {
            Ceiling::Bytes(b) => crate::render::human_bytes(*b),
            Ceiling::Count(1) => "1 model".to_string(),
            Ceiling::Count(n) => format!("{n} models"),
            Ceiling::Unbounded => "none".to_string(),
            Ceiling::Unknown { .. } => "?".to_string(),
        }
    }

    pub fn unit(&self) -> &'static str {
        match self {
            Ceiling::Bytes(_) => "bytes",
            Ceiling::Count(_) => "count",
            Ceiling::Unbounded => "\u{2014}",
            Ceiling::Unknown { .. } => "\u{2014}",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    /// Captured live 2026-09-11, pid 1733.
    #[test]
    fn llamacpp_ceiling_comes_from_models_max_in_the_argv() {
        let a = argv(&[
            "llama-server",
            "--models-dir",
            "/Users/x/.llamacpp/models",
            "--models-max",
            "1",
            "--jinja",
            "--port",
            "8080",
        ]);
        assert_eq!(from_argv(&a), Ceiling::Count(1));
    }

    /// Absent is not zero and not unlimited: llama.cpp has a default and
    /// harmony does not know it.
    #[test]
    fn llamacpp_without_the_flag_is_unknown_rather_than_unbounded() {
        assert!(matches!(from_argv(&argv(&["llama-server", "--port", "8080"])), Ceiling::Unknown { .. }));
    }

    #[test]
    fn a_flag_with_no_number_after_it_is_unknown() {
        assert!(matches!(from_argv(&argv(&["llama-server", "--models-max"])), Ceiling::Unknown { .. }));
    }

    /// Measured: the variable is simply not set, so Ollama's own default
    /// applies and harmony does not know it. Guessing here would be the same
    /// mistake as guessing a footprint.
    #[test]
    fn an_unset_ollama_variable_is_unknown_and_not_a_default() {
        let env = argv(&["OLLAMA_MODELS=/Users/x/.ollama/models", "OLLAMA_NO_CLOUD=0"]);
        match from_environ(&env) {
            Ceiling::Unknown { why } => assert!(why.contains("unset"), "{why}"),
            other => panic!("expected unknown, got {other:?}"),
        }
    }

    #[test]
    fn a_set_ollama_variable_is_a_count() {
        assert_eq!(from_environ(&argv(&["OLLAMA_MAX_LOADED_MODELS=2"])), Ceiling::Count(2));
    }

    #[test]
    fn an_unreadable_environment_is_unknown_for_a_different_reason() {
        match from_environ(&[]) {
            Ceiling::Unknown { why } => assert!(why.contains("could not be read"), "{why}"),
            other => panic!("expected unknown, got {other:?}"),
        }
    }

    /// vLLM-MLX is the only provider publishing a ceiling in bytes, and it
    /// publishes it over the API rather than in a file. `gb` is read as GiB:
    /// the bias inverts for a ceiling, because over-stating what a provider
    /// may take under-states the danger.
    #[test]
    fn vllm_ceiling_comes_from_the_status_endpoint_and_gb_means_gib() {
        let s = serde_json::json!({"model_manager": {"memory_budget_gb": 14.0, "models": []}});
        assert_eq!(from_status(&s), Ceiling::Bytes(14 * 1024 * 1024 * 1024));
    }

    #[test]
    fn a_status_body_without_the_field_is_unknown() {
        let s = serde_json::json!({"model_manager": {"models": []}});
        assert!(matches!(from_status(&s), Ceiling::Unknown { .. }));
    }

    /// JIT loading with no cap and every idle TTL disabled is not a ceiling in
    /// either unit. Saying so is the finding, not a gap.
    #[test]
    fn lmstudio_is_unbounded_in_both_units() {
        let body = r#"{"port":1234,"justInTimeModelLoading": true,"cors":false}"#;
        assert_eq!(from_lmstudio_config(Some(body)), Ceiling::Unbounded);
    }

    /// A provider that is down has no process to read and no endpoint to ask.
    /// Remembering its last known ceiling would be the wrong kind of helpful.
    #[test]
    fn a_provider_that_is_not_running_has_an_unknown_ceiling() {
        let b = budget(ProviderKind::Vllm, &Readings::default());
        match b.ceiling {
            Ceiling::Unknown { why } => assert_eq!(why, "not running"),
            other => panic!("expected unknown, got {other:?}"),
        }
    }

    /// Only a count ceiling can refuse a set on grounds other than bytes, so
    /// only a count ceiling reaches `fit`.
    #[test]
    fn only_a_count_reaches_fit_as_a_limit() {
        let count = Budget {
            provider: ProviderKind::LlamaCpp,
            ceiling: Ceiling::Count(1),
            source: "x".into(),
        };
        let bytes = Budget {
            provider: ProviderKind::Vllm,
            ceiling: Ceiling::Bytes(1 << 30),
            source: "x".into(),
        };
        let unknown = Budget {
            provider: ProviderKind::Ollama,
            ceiling: Ceiling::Unknown { why: "unset".into() },
            source: "x".into(),
        };
        assert_eq!(count_limit(&count), Some(1));
        assert_eq!(count_limit(&bytes), None, "a byte ceiling is admission's business, not the search's");
        assert_eq!(count_limit(&unknown), None, "unknown marks; it does not refuse");
    }

    /// The table prints `?` for what it cannot know -- the convention `status`
    /// already uses for `weights` and `gap`.
    #[test]
    fn an_unknown_ceiling_prints_as_a_question_mark() {
        assert_eq!(Ceiling::Unknown { why: "x".into() }.cell(), "?");
        assert_eq!(Ceiling::Unbounded.cell(), "none");
        assert_eq!(Ceiling::Count(1).cell(), "1 model");
    }
}
