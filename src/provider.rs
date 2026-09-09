use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
    /// Nothing is listening. Not an error condition -- the provider is simply off.
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
