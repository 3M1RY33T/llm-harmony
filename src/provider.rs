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
    /// When the provider intends to drop this model, as a unix timestamp.
    ///
    /// Ollama publishes `expires_at` on `/api/ps` and moves it forward on
    /// every use, which makes it the only ordering signal any of the four
    /// gives for "least recently used". It is an expiry, not a last-use time --
    /// two models with different `keep_alive` values are not comparable by
    /// it -- so it ranks eviction order and is never called a last-use.
    pub expires_at_unix: Option<u64>,

    /// What kind of model this is, as the provider classifies it.
    ///
    /// Only LM Studio publishes one (`llm`, `vlm`, `embeddings`), and it is
    /// carried because the served id depends on it: an embedding model is
    /// served as `text-embedding-<repo>` while its directory is just `<repo>`.
    /// Matching that on name shape alone would be a guess; this makes it a
    /// lookup.
    pub model_type: Option<String>,

    /// Where the weights live, when the provider says so.
    ///
    /// This is the identity placement keys on. Names differ across providers
    /// for the same model -- measured 2026-09-10, canonical-name matching
    /// bridged one of three cross-provider cases -- while a path resolves to a
    /// `FileKey` the disk ledger already indexes, which is proof rather than
    /// inference. LM Studio publishes none, and is bridged by name with the
    /// inference marked.
    pub artifact_path: Option<String>,
}

impl LoadedModel {
    pub fn unloaded(id: impl Into<String>) -> Self {
        LoadedModel {
            id: id.into(),
            state: State::NotLoaded,
            context_tokens: None,
            weights_bytes: None,
            expires_at_unix: None,
            model_type: None,
            artifact_path: None,
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

/// What harmony may actually do to a provider's resident models.
///
/// This is a property of a **verified control surface**, never of config. A
/// provider whose unload path 404s is `SelfManaged` no matter what any TOML
/// file claims -- see `docs/field-notes.md`, *an adapter written from
/// documentation is a hypothesis*, which this enum exists to stop repeating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "actuation", rename_all = "kebab-case")]
pub enum Actuation {
    /// Load and unload one model at a time, by a verb probed against the
    /// server. LM Studio (`lms load`/`lms unload`) and Ollama (`keep_alive`).
    ModelLevel,
    /// Loads on first request and evicts at its own ceiling. Harmony may
    /// admit, account, and bound it at install time -- never actuate it.
    /// llama.cpp (`max_instances`) and vLLM-MLX (`memory_budget_gb`), both
    /// verified 2026-09-10 to publish no model-level load or unload route.
    SelfManaged { ceiling: &'static str },
}

impl Actuation {
    /// The question the eviction planner asks. Kept on the type so no caller
    /// has to match on `ProviderKind` to answer it.
    pub fn can_unload(&self) -> bool {
        matches!(self, Actuation::ModelLevel)
    }
}

/// Observation, plus a declaration of what could be actuated.
///
/// Through slice 4 this trait was read-only by construction: no `unload`, no
/// `evict`, checked by reading the trait rather than a config. Slice 5 ends
/// that, and `Actuation` is what replaces it -- the constraint moves from
/// "the method does not exist" to "the provider says whether it can", which
/// is stronger, because two of the four genuinely cannot and said so only
/// when probed.
pub trait Adapter: Send + Sync {
    fn kind(&self) -> ProviderKind;

    /// Confirm the endpoint is the provider the config declared it to be.
    fn probe(&self, http: &crate::http::Http, base: &str) -> Result<(), ProbeError>;

    /// Every model the provider knows about, with its state.
    fn list(&self, http: &crate::http::Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError>;

    /// What this adapter may do beyond observing. Verified, not declared.
    fn actuation(&self) -> Actuation;

    /// Make this model resident, at the window asked for.
    ///
    /// Returns once the provider has accepted the instruction; residency is
    /// confirmed by re-reading `list`, never by trusting this to have worked.
    fn load(
        &self,
        http: &crate::http::Http,
        base: &str,
        request: &LoadRequest,
    ) -> Result<(), ActuateError>;

    /// Make this model not resident.
    fn unload(
        &self,
        http: &crate::http::Http,
        base: &str,
        model: &str,
    ) -> Result<(), ActuateError>;

    /// Is a request in flight against this model?
    ///
    /// An `Err` means the provider could not say, and **every caller must treat
    /// that as busy**: the slice's constraints forbid evicting a model that is
    /// serving, so an unknown may never read as free.
    fn busy(
        &self,
        http: &crate::http::Http,
        base: &str,
        model: &str,
    ) -> Result<bool, ProbeError>;
}

/// What to load, and how much window to give it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadRequest {
    pub model: String,
    /// `None` leaves the provider's own default alone -- a better guess than
    /// any this project could invent.
    pub context_tokens: Option<u32>,
}

#[derive(Debug)]
pub enum ActuateError {
    /// This provider has no model-level verb. Carries the ceiling that bounds
    /// it instead, so the caller can say something useful rather than only
    /// what harmony cannot do.
    NotSupported { ceiling: &'static str },
    Failed { reason: String },
    Timeout { after_s: u64 },
}

impl std::fmt::Display for ActuateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActuateError::NotSupported { ceiling } => write!(
                f,
                "this provider has no model-level unload; it self-manages under {ceiling}"
            ),
            ActuateError::Failed { reason } => write!(f, "{reason}"),
            ActuateError::Timeout { after_s } => write!(f, "timed out after {after_s}s"),
        }
    }
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
