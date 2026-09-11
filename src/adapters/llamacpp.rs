use crate::http::Http;
use crate::provider::{Actuation, Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct LlamaCpp;

impl LlamaCpp {
    fn fetch_models(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/v1/models"))?;
        // Three providers answer /v1/models, so shape alone is not enough.
        // llama.cpp's router attaches a `status` object to every entry and
        // nobody else does.
        //
        // Captured live 2026-09-09. An earlier version of this adapter looked
        // for a `meta` block, from a fixture constructed out of docs rather
        // than captured from a server. No such block exists, and the mistake
        // only surfaced the first time llama.cpp was actually started.
        let is_llamacpp = v["data"]
            .as_array()
            .map(|a| a.iter().any(|m| m.get("status").is_some()))
            .unwrap_or(false);
        if !is_llamacpp {
            return Err(ProbeError::KindMismatch {
                expected: ProviderKind::LlamaCpp,
            });
        }
        Ok(v)
    }
}

/// The path the router will serve this entry from.
///
/// `status.args` is the full argv the router would exec, and `--model <path>`
/// is in it whether or not the model is resident. Captured live 2026-09-10.
fn model_arg(m: &serde_json::Value) -> Option<String> {
    let args = m["status"]["args"].as_array()?;
    let i = args.iter().position(|a| a.as_str() == Some("--model"))?;
    Some(args.get(i + 1)?.as_str()?.to_string())
}

/// This provider's own limit, named in every refusal so the operator learns
/// the lever that does exist.
const CEILING: &str = "max_instances";

impl Adapter for LlamaCpp {
    fn kind(&self) -> ProviderKind {
        ProviderKind::LlamaCpp
    }

    fn probe(&self, http: &Http, base: &str) -> Result<(), ProbeError> {
        self.fetch_models(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let v = self.fetch_models(http, base)?;

        Ok(v["data"]
            .as_array()
            .expect("checked in fetch_models")
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                let state = match m["status"]["value"].as_str() {
                    Some("loaded") => State::Loaded,
                    Some("loading") | Some("starting") => State::Loading,
                    _ => State::NotLoaded,
                };
                Some(LoadedModel {
                    id,
                    state,
                    // This build publishes no context window anywhere -- not
                    // on the model entry, not under `architecture`. Verified
                    // live 2026-09-09 against llama.cpp b10240. `None` is the
                    // honest answer; the previous `meta.n_ctx` was invented.
                    context_tokens: None,
                    // No size either. The path below is what lets the disk
                    // ledger supply one.
                    weights_bytes: None,
                    expires_at_unix: None,
                    model_type: None,
                    artifact_path: model_arg(m),
                })
            })
            .collect())
    }

    /// Verified 2026-09-10: `POST /api/models/unload/<model>` returns
    /// 404 for a real model id, on GET and POST alike. The router
    /// self-manages -- `models_autoload: true`, `max_instances: 1` --
    /// so loading a second model is what evicts the first.
    fn actuation(&self) -> Actuation {
        Actuation::SelfManaged { ceiling: CEILING }
    }

    /// GGUF only. `llama-server` loads nothing else, in any dtype.
    fn formats(&self) -> Vec<crate::inventory::artifact::Format> {
        vec![crate::inventory::artifact::Format::Gguf]
    }

    fn load(
        &self,
        _http: &Http,
        _base: &str,
        _request: &crate::provider::LoadRequest,
    ) -> Result<(), crate::provider::ActuateError> {
        // No network call: there is no endpoint to reach, and a timeout would
        // misreport "cannot" as "did not answer". `ttl_seconds` is dropped
        // here for the same reason the window is: there is no load verb to
        // carry either, and this provider's idle behaviour is its own flag.
        Err(crate::provider::ActuateError::NotSupported { ceiling: CEILING })
    }

    fn unload(
        &self,
        _http: &Http,
        _base: &str,
        _model: &str,
    ) -> Result<(), crate::provider::ActuateError> {
        Err(crate::provider::ActuateError::NotSupported { ceiling: CEILING })
    }

    /// The router publishes no per-model request state -- `/running` 404s,
    /// verified 2026-09-10. Unknown, which every caller must read as busy.
    fn busy(&self, _http: &Http, _base: &str, _model: &str) -> Result<bool, ProbeError> {
        Err(ProbeError::Malformed { reason: "llama.cpp publishes no request state".into() })
    }
}
