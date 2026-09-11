//! Models harmony may not evict to make room for something else.
//!
//! A pin is a **veto, never a reservation**. It cannot make a model resident
//! and it cannot reserve memory for one; it only removes a model from the set
//! of things that may be unloaded on someone else's behalf. The consequence is
//! that a pin can cause a refusal on a machine that looks half empty, which is
//! why every refusal that a pin caused has to name it.
//!
//! Runtime state, deliberately not config: the point is to protect a model
//! that is *already loaded*, which a file the user hand-edits before starting
//! anything cannot serve.

use std::path::{Path, PathBuf};

use crate::provider::ProviderKind;

/// One protected model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pin {
    pub provider: ProviderKind,
    pub model: String,
    /// Unix seconds, so `status` can say how long it has been held. A pin
    /// nobody remembers setting is the failure mode worth making visible.
    pub at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The pin store, as it lives on disk.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Pins {
    #[serde(default = "schema")]
    schema: u32,
    #[serde(default)]
    pins: Vec<Pin>,
}

fn schema() -> u32 {
    1
}

impl Default for Pins {
    fn default() -> Self {
        Pins::empty()
    }
}

impl Pins {
    pub fn empty() -> Pins {
        Pins { schema: schema(), pins: Vec::new() }
    }

    pub fn path() -> Option<PathBuf> {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".local/state/llm-harmony/pins.json"))
    }

    /// Never fails. A missing file is no pins; so is a corrupt one, because a
    /// pin store that can abort `status` would be worse than no pin store at
    /// all -- and the visible consequence (nothing is protected) is safer than
    /// the alternative (nothing can be read).
    pub fn load_from(path: &Path) -> Pins {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| Pins::empty()),
            Err(_) => Pins::empty(),
        }
    }

    pub fn load() -> Pins {
        match Pins::path() {
            Some(p) => Pins::load_from(&p),
            None => Pins::empty(),
        }
    }

    /// Keyed on the pair: the same model id on another provider is a different
    /// artifact with a different footprint, and protecting one must not
    /// protect the other.
    pub fn is_pinned(&self, provider: ProviderKind, model: &str) -> bool {
        self.pins.iter().any(|p| p.provider == provider && p.model == model)
    }

    pub fn all(&self) -> &[Pin] {
        &self.pins
    }

    /// `false` if it was already pinned.
    pub fn add(&mut self, pin: Pin) -> bool {
        if self.is_pinned(pin.provider, &pin.model) {
            return false;
        }
        self.pins.push(pin);
        true
    }

    /// `false` if it was not pinned. Reported rather than swallowed so
    /// `unpin` can tell the operator nothing changed.
    pub fn remove(&mut self, provider: ProviderKind, model: &str) -> bool {
        let before = self.pins.len();
        self.pins.retain(|p| !(p.provider == provider && p.model == model));
        self.pins.len() != before
    }

    /// Atomic: write a sibling temp file and rename, so an interrupted save
    /// leaves either the old store or the new one and never half of either.
    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, body).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self) -> Result<(), String> {
        match Pins::path() {
            Some(p) => self.save_to(&p),
            None => Err("no HOME; cannot locate the pin store".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An older store with no `schema` field still reads, because the field
    /// defaults. The first version of this file shipped before anyone thought
    /// to write one, and losing every pin on upgrade would be a poor trade for
    /// strictness.
    #[test]
    fn a_document_without_a_schema_field_still_loads() {
        let p: Pins = serde_json::from_str(r#"{"pins":[]}"#).unwrap();
        assert_eq!(p.schema, 1);
    }

    #[test]
    fn an_unknown_field_does_not_discard_the_store() {
        let p: Pins = serde_json::from_str(
            r#"{"schema":1,"pins":[{"provider":"ollama","model":"m","at":1}],"future":true}"#,
        )
        .unwrap();
        assert!(p.is_pinned(ProviderKind::Ollama, "m"));
    }
}
