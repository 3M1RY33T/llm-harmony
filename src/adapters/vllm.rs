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
