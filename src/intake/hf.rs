//! The Hugging Face read: search, and what a repo holds.
//!
//! Deliberately the *only* place that knows HF's URLs and JSON. Delroy has no
//! HF client at all and must not grow one — `Delroy/docs/hosting-page-r27-plan.md`
//! §2 rejects that explicitly, because the provenance rule and the placement
//! rule would then live in two repos and drift.
//!
//! Nothing here writes. Downloading is `download.rs`, so a search that returns
//! nonsense costs a wasted request and never a wasted disk.

use serde::Serialize;

use crate::http::Http;
use crate::inventory::artifact::{bits_from_name, Format};
use crate::provider::ProbeError;

const API: &str = "https://huggingface.co/api";

/// One file in a repo that a provider here could actually load.
#[derive(Debug, Clone, Serialize)]
pub struct RepoFile {
    pub name: String,
    pub format: Format,
    pub bits: Option<u8>,
    /// `None` when the listing did not carry a size. Not zero: a missing size
    /// and a zero-byte file are very different things to admit against.
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Repo {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    /// From the repo card. `None` is the common honest case, not a red flag —
    /// see `provenance`.
    pub base_model: Option<String>,
    pub files: Vec<RepoFile>,
}

impl Repo {
    /// Whether this repo holds something loadable as it stands.
    ///
    /// A bf16-only repo is *shown* and marked, never hidden: a user searching
    /// for a model should learn that a build exists and that harmony cannot yet
    /// use it, not conclude there is none.
    pub fn needs_conversion(&self) -> bool {
        !self.files.is_empty()
            && self
                .files
                .iter()
                .all(|f| matches!(f.format, Format::SafetensorsBf16))
    }

    pub fn has_loadable_build(&self) -> bool {
        self.files.iter().any(|f| matches!(f.format, Format::Gguf | Format::Mlx))
    }
}

/// A JSON value that may be a string or a list of them, as one string.
fn first_string(v: &serde_json::Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    v.as_array()?.iter().find_map(|e| e.as_str()).map(|s| s.to_string())
}

fn classify(name: &str) -> Option<RepoFile> {
    let format = Format::from_path_str(name);
    if matches!(format, Format::Other) {
        return None;
    }
    Some(RepoFile { name: name.to_string(), format, bits: bits_from_name(name), size_bytes: None })
}

/// Parse one repo out of HF's model JSON. Split from the fetch so the shape is
/// testable without a network.
pub fn repo_from_json(value: &serde_json::Value) -> Option<Repo> {
    let id = value["id"].as_str().or_else(|| value["modelId"].as_str())?.to_string();
    // `base_model` is a string on some repos and an ARRAY on others — verified
    // against the live API 2026-09-11, where `unsloth/Qwen3.8-27B-GGUF`
    // publishes `["Qwen/Qwen3.8-27B"]`. Reading only the string form reported
    // every one of those as declaring nothing, which turned the provenance
    // warning into noise on exactly the repos that were being careful.
    let base_model = first_string(&value["cardData"]["base_model"])
        .or_else(|| first_string(&value["config"]["base_model"]));

    let mut files = Vec::new();
    if let Some(siblings) = value["siblings"].as_array() {
        for s in siblings {
            let Some(name) = s["rfilename"].as_str() else { continue };
            let Some(mut f) = classify(name) else { continue };
            f.size_bytes = s["size"].as_u64().or_else(|| s["lfs"]["size"].as_u64());
            files.push(f);
        }
    }
    Some(Repo {
        id,
        downloads: value["downloads"].as_u64().unwrap_or(0),
        likes: value["likes"].as_u64().unwrap_or(0),
        base_model,
        files,
    })
}

/// Search HF for models, as HF ranks them.
///
/// Returns the ids only. The list endpoint carries neither `cardData` nor file
/// sizes even with `full=true` — verified against the live API 2026-09-11 — so
/// a row built from it can say nothing about provenance and nothing about fit,
/// which are the two things a search result exists to carry. `search_detailed`
/// is what callers want; this is the half that is one request.
pub fn search_ids(http: &Http, query: &str, limit: usize) -> Result<Vec<String>, ProbeError> {
    let url = format!(
        "{API}/models?search={}&limit={}",
        urlencode(query),
        limit.clamp(1, 50)
    );
    let value = http.get_json(&url)?;
    let empty = Vec::new();
    Ok(value
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter_map(|v| v["id"].as_str().or_else(|| v["modelId"].as_str()))
        .map(|s| s.to_string())
        .collect())
}

/// Search, then read each hit in full.
///
/// One request per result on top of the search. That is the price of §6's
/// decision that every row arrives priced, and it is paid behind a debounce in
/// a subprocess, not on Delroy's event loop. A repo whose detail cannot be read
/// is dropped rather than shown unpriced — a row that cannot answer the
/// question the panel exists to answer is not a row.
pub fn search(http: &Http, query: &str, limit: usize) -> Result<Vec<Repo>, ProbeError> {
    let ids = search_ids(http, query, limit)?;
    Ok(ids.iter().filter_map(|id| repo(http, id).ok()).collect())
}

/// One repo in full, with its file listing **and sizes**.
///
/// `?blobs=true` is not optional: without it `siblings` carries filenames and
/// no sizes, and a file with no size cannot be admitted against either ledger —
/// which would make every pull unpriceable and every search row blank.
pub fn repo(http: &Http, id: &str) -> Result<Repo, ProbeError> {
    let value = http.get_json(&format!("{API}/models/{id}?blobs=true"))?;
    repo_from_json(&value).ok_or(ProbeError::Malformed {
        reason: format!("`{id}` did not come back as a model"),
    })
}

/// The URL a file's bytes come from.
pub fn download_url(repo_id: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/main/{file}?download=true")
}

/// Percent-encode a query. Small by hand rather than a dependency: the only
/// thing crossing this boundary is a search box's contents.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn only_loadable_files_are_listed() {
        let r = repo_from_json(&json(
            r#"{"id":"x/y","siblings":[
                 {"rfilename":"README.md"},
                 {"rfilename":"tokenizer.json"},
                 {"rfilename":"model.q4_k_m.gguf","size":8192}
               ]}"#,
        ))
        .unwrap();
        assert_eq!(r.files.len(), 1, "a repo is mostly not weights");
        assert_eq!(r.files[0].bits, Some(4));
        assert_eq!(r.files[0].size_bytes, Some(8192));
    }

    #[test]
    fn a_size_that_was_not_published_is_none_and_not_zero() {
        // A missing size and a zero-byte file admit very differently.
        let r = repo_from_json(&json(
            r#"{"id":"x/y","siblings":[{"rfilename":"m.gguf"}]}"#,
        ))
        .unwrap();
        assert_eq!(r.files[0].size_bytes, None);
    }

    #[test]
    fn an_lfs_size_is_read_where_the_plain_one_is_absent() {
        let r = repo_from_json(&json(
            r#"{"id":"x/y","siblings":[{"rfilename":"m.gguf","lfs":{"size":999}}]}"#,
        ))
        .unwrap();
        assert_eq!(r.files[0].size_bytes, Some(999));
    }

    #[test]
    fn a_bf16_only_repo_is_shown_and_marked_rather_than_hidden() {
        let r = repo_from_json(&json(
            r#"{"id":"x/y","siblings":[{"rfilename":"model.safetensors","size":1}]}"#,
        ))
        .unwrap();
        assert!(r.needs_conversion(), "it exists; harmony just cannot use it yet");
        assert!(!r.has_loadable_build());
    }

    #[test]
    fn a_base_model_published_as_a_list_is_read_like_a_string() {
        // Verified live 2026-09-11: `unsloth/Qwen3.8-27B-GGUF` publishes
        // `["Qwen/Qwen3.8-27B"]`. Reading only the string form reported every
        // such repo as declaring nothing, which is noise on the careful ones.
        let r = repo_from_json(&json(
            r#"{"id":"x/y","cardData":{"base_model":["Qwen/Qwen3.8-27B"]},"siblings":[]}"#,
        ))
        .unwrap();
        assert_eq!(r.base_model.as_deref(), Some("Qwen/Qwen3.8-27B"));
    }

    #[test]
    fn the_declared_base_model_is_read_from_the_card() {
        let r = repo_from_json(&json(
            r#"{"id":"x/y","cardData":{"base_model":"Qwen/Qwen3-14B"},"siblings":[]}"#,
        ))
        .unwrap();
        assert_eq!(r.base_model.as_deref(), Some("Qwen/Qwen3-14B"));
    }

    #[test]
    fn a_query_with_spaces_and_slashes_survives_the_url() {
        assert_eq!(urlencode("qwen3 14b/gguf"), "qwen3+14b%2Fgguf");
    }
}
