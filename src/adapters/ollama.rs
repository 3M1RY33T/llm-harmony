use std::collections::HashMap;

use crate::http::Http;
use crate::provider::{Actuation, Adapter, LoadedModel, ProbeError, ProviderKind, State};

/// How long a harmony-initiated load stays resident without use.
///
/// Not `-1`: a model harmony loaded and nobody used should eventually leave,
/// and Ollama's own TTL is the mechanism for that. Pins are how a model is
/// kept deliberately.
const KEEP_ALIVE: &str = "10m";

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

    /// Verified 2026-09-10: residency is controlled by `keep_alive`
    /// on an ordinary request -- 0 drops a model, a duration holds it.
    fn actuation(&self) -> Actuation {
        Actuation::ModelLevel
    }

    /// Ollama's whole control plane is `keep_alive` on an ordinary request:
    /// a duration holds the model, zero drops it. The prompt is empty -- this
    /// is a control call and carries no content, ever.
    fn load(
        &self,
        http: &Http,
        base: &str,
        request: &crate::provider::LoadRequest,
    ) -> Result<(), crate::provider::ActuateError> {
        let path = Ollama::endpoint(http, base, &request.model);
        let mut body = Ollama::residency_body(path, &request.model, KEEP_ALIVE.into());
        // num_ctx is where Ollama takes a window, and the window is most of
        // what a load costs.
        if let Some(ctx) = request.context_tokens {
            body["options"] = serde_json::json!({ "num_ctx": ctx });
        }
        Ollama::post_residency(http, base, path, body)
    }

    fn unload(
        &self,
        http: &Http,
        base: &str,
        model: &str,
    ) -> Result<(), crate::provider::ActuateError> {
        let path = Ollama::endpoint(http, base, model);
        let body = Ollama::residency_body(path, model, 0.into());
        Ollama::post_residency(http, base, path, body)
    }

    /// `/api/ps` lists what is resident but says nothing about in-flight
    /// requests. Unknown, which every caller must read as busy.
    fn busy(&self, _http: &Http, _base: &str, _model: &str) -> Result<bool, ProbeError> {
        Err(ProbeError::Malformed { reason: "ollama publishes no per-model request state".into() })
    }
}

impl Ollama {
    /// Which endpoint controls this model's residency.
    ///
    /// Verified 2026-09-11: `/api/generate` answers **HTTP 400** for an
    /// embedding model -- `"nomic-embed-text:latest" does not support
    /// generate` -- so there is no single endpoint that loads anything Ollama
    /// serves. `/api/show` publishes `capabilities`, and `/api/embed` with an
    /// empty input loads the embedding models, so the server is asked rather
    /// than guessed at.
    ///
    /// A failed lookup falls back to `/api/generate`: it is the common case,
    /// and a wrong guess costs a clear 400 rather than a silent no-op.
    fn endpoint(http: &Http, base: &str, model: &str) -> &'static str {
        let shown = http.post_json(
            &format!("{base}/api/show"),
            &serde_json::json!({ "model": model }),
        );
        let embedding = shown
            .ok()
            .and_then(|v| {
                v["capabilities"]
                    .as_array()
                    .map(|caps| caps.iter().any(|c| c.as_str() == Some("embedding")))
            })
            .unwrap_or(false);

        if embedding {
            "/api/embed"
        } else {
            "/api/generate"
        }
    }

    /// The request that changes residency and carries no content.
    ///
    /// `prompt`/`input` are empty by design: this is a control call. See
    /// `docs/architecture.md` -- no user prompt passes through harmony.
    fn residency_body(
        path: &str,
        model: &str,
        keep_alive: serde_json::Value,
    ) -> serde_json::Value {
        match path {
            "/api/embed" => {
                serde_json::json!({ "model": model, "input": "", "keep_alive": keep_alive })
            }
            _ => serde_json::json!({ "model": model, "prompt": "", "keep_alive": keep_alive }),
        }
    }

    fn post_residency(
        http: &Http,
        base: &str,
        path: &str,
        body: serde_json::Value,
    ) -> Result<(), crate::provider::ActuateError> {
        http.post_json(&format!("{base}{path}"), &body)
            .map(|_| ())
            .map_err(|e| crate::provider::ActuateError::Failed { reason: e.to_string() })
    }
}
