use std::collections::HashMap;

use crate::http::Http;
use crate::provider::{Adapter, LoadedModel, ProbeError, ProviderKind, State};

pub struct Ollama;

impl Ollama {
    fn fetch_ps(&self, http: &Http, base: &str) -> Result<serde_json::Value, ProbeError> {
        let v = http.get_json(&format!("{base}/api/ps"))?;
        if !v["models"].is_array() {
            return Err(ProbeError::KindMismatch {
                expected: ProviderKind::Ollama,
            });
        }
        Ok(v)
    }
}

impl Adapter for Ollama {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    fn probe(&self, http: &Http, base: &str) -> Result<(), ProbeError> {
        self.fetch_ps(http, base).map(|_| ())
    }

    fn list(&self, http: &Http, base: &str) -> Result<Vec<LoadedModel>, ProbeError> {
        let ps = self.fetch_ps(http, base)?;

        // Resident models: /api/ps is the ONLY source of a serving window.
        let mut out: Vec<LoadedModel> = Vec::new();
        let mut resident: HashMap<String, ()> = HashMap::new();
        for m in ps["models"].as_array().expect("checked in fetch_ps") {
            let Some(id) = m["name"].as_str().or_else(|| m["model"].as_str()) else {
                continue;
            };
            resident.insert(id.to_string(), ());
            out.push(LoadedModel {
                id: id.to_string(),
                state: State::Loaded,
                context_tokens: m["context_length"].as_u64().map(|n| n as u32),
                weights_bytes: m["size"].as_u64(),
                // /api/ps carries a digest, not a path. The blob it names is
                // resolvable, but only through slice 2's scanner, which owns
                // the blobs directory layout.
                artifact_path: None,
            });
        }

        // Everything on disk. `size` here is a file property and safe to read.
        // `details.context_length` is a CAPABILITY figure and is deliberately
        // not read -- see tests. A failure here is not fatal: the resident list
        // is the part that matters for memory.
        if let Ok(tags) = http.get_json(&format!("{base}/api/tags")) {
            let empty = Vec::new();
            for m in tags["models"].as_array().unwrap_or(&empty) {
                let Some(id) = m["name"].as_str().or_else(|| m["model"].as_str()) else {
                    continue;
                };
                if resident.contains_key(id) {
                    continue;
                }
                out.push(LoadedModel {
                    id: id.to_string(),
                    state: State::NotLoaded,
                    context_tokens: None,
                    weights_bytes: m["size"].as_u64(),
                    artifact_path: None,
                });
            }
        }

        Ok(out)
    }
}
