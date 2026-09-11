use std::collections::HashMap;

use crate::http::Http;
use crate::provider::{Actuation, Adapter, LoadedModel, ProbeError, ProviderKind, State};

/// How long a harmony-initiated load stays resident without use, when the
/// caller named no TTL of its own.
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
                //
                // The expiry, though, is published right here, and leaving it
                // None cost more than a missing UI field: `residents()` feeds
                // it into `Resident.evict_rank`, so every resident model on
                // every provider ranked 0 and `plan.rs`'s "least-recently-used
                // first" was ledger order.
                expires_at_unix: m["expires_at"].as_str().and_then(parse_rfc3339),
                model_type: None,
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
                    expires_at_unix: None,
                    model_type: None,
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
    /// GGUF, and MLX through its own backend. Separate from whether a file
    /// can be dropped into its store -- it cannot, and `place::store_for`
    /// still says so.
    fn formats(&self) -> Vec<crate::inventory::artifact::Format> {
        use crate::inventory::artifact::Format;
        vec![Format::Gguf, Format::Mlx]
    }

    fn load(
        &self,
        http: &Http,
        base: &str,
        request: &crate::provider::LoadRequest,
    ) -> Result<(), crate::provider::ActuateError> {
        let path = Ollama::endpoint(http, base, &request.model);
        // Seconds when the caller named a TTL, and Ollama's duration string
        // otherwise: the API takes either, and a bare number is seconds.
        let keep_alive = match request.ttl_seconds {
            Some(secs) => serde_json::Value::from(secs),
            None => KEEP_ALIVE.into(),
        };
        let mut body = Ollama::residency_body(path, &request.model, keep_alive);
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

/// `2026-09-11T01:10:00.952437-04:00` -> unix seconds.
///
/// Hand-rolled rather than pulling in a date crate, for the same reason the
/// GGUF header reader is: one well-understood function is cheaper to own than
/// a dependency, and this parses exactly one shape from exactly one producer.
/// Anything it does not recognise is `None` -- a wrong timestamp would reorder
/// evictions silently, which is worse than having no ordering at all.
fn parse_rfc3339(s: &str) -> Option<u64> {
    let bytes = s.as_bytes();
    if bytes.len() < 19 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
        return None;
    }
    let num = |a: usize, b: usize| s.get(a..b)?.parse::<i64>().ok();

    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, sec) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }

    // Everything after the seconds is an optional fraction then the offset.
    let rest = &s[19..];
    let rest = match rest.find(['Z', '+', '-']) {
        Some(i) => &rest[i..],
        None => return None,
    };
    let offset_seconds = if rest.starts_with('Z') {
        0
    } else {
        let sign = if rest.starts_with('-') { -1 } else { 1 };
        let oh: i64 = rest.get(1..3)?.parse().ok()?;
        let om: i64 = rest.get(4..6)?.parse().ok()?;
        sign * (oh * 3600 + om * 60)
    };

    let secs = days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec - offset_seconds;
    u64::try_from(secs).ok()
}

/// Days since 1970-01-01 for a proleptic Gregorian date. Howard Hinnant's
/// `days_from_civil`, which is the standard formulation of this.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod timestamp_tests {
    use super::*;

    /// Reference values computed independently, not by this function.
    #[test]
    fn known_timestamps_round_trip() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2026-09-11T05:10:00Z"), Some(1_789_103_400));
        assert_eq!(parse_rfc3339("2000-02-29T12:00:00+02:00"), Some(951_818_400));
    }

    /// The exact shape Ollama emits: fractional seconds and a negative offset.
    /// Same instant as the Z form above.
    #[test]
    fn ollamas_own_format_parses_to_the_same_instant() {
        assert_eq!(
            parse_rfc3339("2026-09-11T01:10:00.952437-04:00"),
            Some(1_789_103_400)
        );
    }

    /// Anything unrecognised is None. A wrong timestamp would silently
    /// reorder evictions, which is worse than having no ordering.
    #[test]
    fn unparseable_input_is_none_rather_than_a_guess() {
        for bad in ["", "not a date", "2026-09-11", "2026-09-11T01:10:00", "2026-13-40T99:99:99Z"] {
            assert_eq!(parse_rfc3339(bad), None, "{bad}");
        }
    }
}
