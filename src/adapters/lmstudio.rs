use crate::http::Http;
use crate::provider::{Actuation, Adapter, LoadedModel, ProbeError, ProviderKind, State};

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
                    // Nor a path: its entries carry publisher, arch,
                    // quantization and state, and nothing that locates the
                    // file. Verified live 2026-09-10. Placement bridges this
                    // provider by name and marks the result inferred.
                    expires_at_unix: None,
                    model_type: m["type"].as_str().map(str::to_string),
                    artifact_path: None,
                })
            })
            .collect())
    }

    /// Verified 2026-09-10: `lms load <model> [-c N]` and
    /// `lms unload <model>` both exist. The CLI is the only control
    /// surface -- the REST API has no load or unload route.
    fn actuation(&self) -> Actuation {
        Actuation::ModelLevel
    }

    /// LM Studio's REST API has no load or unload route; `lms` is the whole
    /// control plane. So harmony shells out -- and the binary being findable
    /// on PATH is therefore a hard dependency of eviction on this provider.
    fn load(
        &self,
        http: &Http,
        base: &str,
        request: &crate::provider::LoadRequest,
    ) -> Result<(), crate::provider::ActuateError> {
        // Loading an already-resident model does not no-op: LM Studio starts a
        // SECOND instance, identifier `<model>:2`, with its own weights and its
        // own KV cache, and `lms unload <model>` then removes only the first.
        // Verified 2026-09-11. On a 24 GB machine that is a silent doubling --
        // the exact failure this project exists to prevent -- so the cost of
        // one extra read here is not a question.
        if self.is_resident(http, base, &request.model) {
            return Ok(());
        }
        run(&LmStudio::load_argv(request))
    }

    fn unload(
        &self,
        _http: &Http,
        _base: &str,
        model: &str,
    ) -> Result<(), crate::provider::ActuateError> {
        run(&LmStudio::unload_argv(model))
    }

    /// `/api/v0/models` carries `state` but nothing about in-flight requests.
    /// Unknown, which every caller must read as busy.
    fn busy(&self, _http: &Http, _base: &str, _model: &str) -> Result<bool, ProbeError> {
        Err(ProbeError::Malformed { reason: "LM Studio publishes no request state".into() })
    }

}

impl LmStudio {
    /// Is this model already loaded?
    ///
    /// An unreadable provider answers `false`: refusing to load because the
    /// list could not be read would make a transient blip look like a
    /// permanent refusal, and the duplicate-instance trap it guards against
    /// is only reachable when the model really is resident.
    fn is_resident(&self, http: &Http, base: &str, model: &str) -> bool {
        self.list(http, base)
            .map(|ms| ms.iter().any(|m| m.id == model && m.state == State::Loaded))
            .unwrap_or(false)
    }

    /// Split out as a pure function so the argv is testable without spawning.
    /// `--yes` because an interactive disambiguation prompt would hang a
    /// non-interactive caller forever.
    pub fn load_argv(request: &crate::provider::LoadRequest) -> Vec<String> {
        let mut argv = vec![
            BINARY.to_string(),
            "load".to_string(),
            request.model.clone(),
            "--yes".to_string(),
        ];
        if let Some(ctx) = request.context_tokens {
            argv.push("--context-length".to_string());
            argv.push(ctx.to_string());
        }
        // LM Studio's own idle timer, in seconds. Omitted when nothing asked
        // for one, which leaves whatever the app is configured with alone.
        if let Some(ttl) = request.ttl_seconds {
            argv.push("--ttl".to_string());
            argv.push(ttl.to_string());
        }
        argv
    }

    pub fn unload_argv(model: &str) -> Vec<String> {
        vec![BINARY.to_string(), "unload".to_string(), model.to_string()]
    }
}

/// LM Studio's CLI, as installed at ~/.lmstudio/bin/lms.
const BINARY: &str = "lms";

/// A missing binary is a named failure, never a silent no-op: harmony would
/// otherwise report an eviction it never performed.
fn run(argv: &[String]) -> Result<(), crate::provider::ActuateError> {
    use crate::provider::ActuateError;

    let out = std::process::Command::new(&argv[0]).args(&argv[1..]).output();
    match out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(ActuateError::Failed {
            reason: format!(
                "`{}` exited {}: {}",
                argv.join(" "),
                o.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&o.stderr).trim()
            ),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(ActuateError::Failed {
            reason: format!("`{BINARY}` is not on PATH; LM Studio has no other control surface"),
        }),
        Err(e) => Err(ActuateError::Failed { reason: e.to_string() }),
    }
}
