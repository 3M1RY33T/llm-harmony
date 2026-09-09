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
