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
