use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct Vllm;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

impl Vllm {
    /// `/v1/status` is the only endpoint that reports real residency.
    ///
    /// `/v1/models` lists everything the registry *can* serve, which is not the
    /// same thing at all: verified live 2026-09-10, it advertised five models
    /// while the server held 0.32 GB and none were resident. An earlier version
    /// of this adapter treated presence as residency and would have told the
    /// ledger that ~25 GB of weights were committed.
    fn fetch_status(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/v1/status"))?;
        if !v["model_manager"]["models"].is_array() {
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
        self.fetch_status(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let v = self.fetch_status(http, base)?;
        Ok(v["model_manager"]["models"]
            .as_array()
            .expect("checked in fetch_status")
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                // The explicit boolean, not the presence of a row.
                let state = match m["loaded"].as_bool() {
                    Some(true) => State::Loaded,
                    _ => State::NotLoaded,
                };
                Some(LoadedModel {
                    id,
                    state,
                    // vLLM-MLX still publishes no serving window anywhere.
                    context_tokens: None,
                    // It is the one provider that publishes a per-model memory
                    // figure of its own. Its own startup log warns this covers
                    // weights only -- no KV cache, no prefix cache, no
                    // activations -- so it is a floor, and the ledger's `gap`
                    // column is exactly the distance it omits.
                    weights_bytes: m["memory_gb"]
                        .as_f64()
                        .map(|g| (g * GIB) as u64),
                    // The manager publishes the model directory outright.
                    artifact_path: m["source"].as_str().map(str::to_string),
                })
            })
            .collect())
    }
}
