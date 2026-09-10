use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

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
                    artifact_path: model_arg(m),
                })
            })
            .collect())
    }
}
