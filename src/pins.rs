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
    /// Who asked for this protection, when anyone did.
    ///
    /// A pin set by hand has none. A lease taken by a client has one, and
    /// every refusal it causes names it: "something invisible is holding
    /// memory you can see" is the failure mode, and an operator has to be able
    /// to find the holder without reading a state file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Unix seconds after which this protects nothing.
    ///
    /// `None` is a pin: protection with no end, cleared only by `unpin`. **A
    /// pin is a lease with no expiry**, and they share one store because
    /// eviction has exactly one place to ask -- two stores would be two
    /// answers to one question.
    ///
    /// Expiry does not cause anything. A lease is still a veto and never a
    /// reservation, so the only thing that happens when one lapses is that its
    /// model rejoins the set harmony may consider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
}

impl Pin {
    /// Still protecting, as of `now`.
    pub fn live_at(&self, now: u64) -> bool {
        self.expires_at.is_none_or(|e| now < e)
    }

    /// How this entry should be named in a refusal it caused.
    pub fn blame(&self, now: u64) -> String {
        match (&self.owner, self.expires_at) {
            (Some(who), Some(e)) => format!(
                "{} on {} (leased by {who}, {} left)",
                self.model,
                self.provider,
                human_seconds(e.saturating_sub(now))
            ),
            (None, Some(e)) => format!(
                "{} on {} (leased, {} left)",
                self.model,
                self.provider,
                human_seconds(e.saturating_sub(now))
            ),
            (Some(who), None) => format!("{} on {} (pinned by {who})", self.model, self.provider),
            (None, None) => format!("{} on {}", self.model, self.provider),
        }
    }
}

/// Coarse on purpose: a refusal needs the order of magnitude, and a lease
/// measured to the second would imply a precision the expiry does not have.
pub fn human_seconds(s: u64) -> String {
    match s {
        0 => "expired".to_string(),
        s if s < 90 => format!("{s}s"),
        s if s < 5400 => format!("{}m", s / 60),
        // One decimal, because flooring turns a two-hour lease into "1h" the
        // moment it starts -- which reads as half the protection it has.
        s => format!("{:.1}h", s as f64 / 3600.0),
    }
}

/// What taking a lease did. Reported rather than swallowed, so a caller can
/// tell "you now hold it" from "a pin already protects this, and I left it
/// alone".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseOutcome {
    Taken,
    Extended,
    /// Asked for less time than the lease already had. Nothing changed.
    AlreadyLonger,
    /// A pin holds this model. A lease would be a downgrade, so it was refused.
    RefusedPinned,
}

/// What releasing did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseOutcome {
    Released,
    /// Nothing held it. Asking for a state you are already in is a success,
    /// the posture `unpin` already takes.
    NotHeld,
    /// A pin held it, and a pin is not a lease's to clear. `unpin` is.
    WasPinned,
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
    ///
    /// **Ignores expiry, and is for display only.** Every decision goes
    /// through [`Pins::holds`], which takes the time. A store that has not
    /// been written since a lease lapsed still lists it here; `sweep` removes
    /// it on the next save.
    pub fn is_pinned(&self, provider: ProviderKind, model: &str) -> bool {
        self.pins.iter().any(|p| p.provider == provider && p.model == model)
    }

    /// The entry protecting this model right now, if any.
    ///
    /// The one question eviction asks. `now` is a parameter rather than a
    /// clock read, for the reason `record.rs` states about its own timestamp:
    /// passed in, so this is testable.
    pub fn holds(&self, provider: ProviderKind, model: &str, now: u64) -> Option<&Pin> {
        self.pins
            .iter()
            .find(|p| p.provider == provider && p.model == model && p.live_at(now))
    }

    /// Drop every lapsed lease. Returns how many went.
    ///
    /// Housekeeping, not policy: it changes nothing about what is protected,
    /// because [`Pins::holds`] already ignores a lapsed entry. It exists so
    /// the store does not accumulate dead rows, and so `status` stops naming
    /// them once anything writes the file.
    pub fn sweep(&mut self, now: u64) -> usize {
        let before = self.pins.len();
        self.pins.retain(|p| p.live_at(now));
        before - self.pins.len()
    }

    pub fn all(&self) -> &[Pin] {
        &self.pins
    }

    /// Take or renew a lease, refusing to weaken anything.
    ///
    /// A client that holds a lease and is still working has to be able to
    /// extend it, so a second lease over a live one moves the expiry to
    /// whichever is later -- never earlier, because shortening protection by
    /// asking for it again would be a trap. A lease over a *pin* is refused
    /// outright: a pin has no expiry, and replacing it with one would quietly
    /// convert permanent protection into temporary protection.
    pub fn lease(&mut self, pin: Pin) -> LeaseOutcome {
        match self.pins.iter_mut().find(|p| p.provider == pin.provider && p.model == pin.model) {
            Some(existing) if existing.expires_at.is_none() => LeaseOutcome::RefusedPinned,
            Some(existing) => {
                let was = existing.expires_at;
                existing.expires_at = existing.expires_at.max(pin.expires_at);
                existing.owner = pin.owner.or_else(|| existing.owner.clone());
                if existing.expires_at == was {
                    LeaseOutcome::AlreadyLonger
                } else {
                    LeaseOutcome::Extended
                }
            }
            None => {
                self.pins.push(pin);
                LeaseOutcome::Taken
            }
        }
    }

    /// `false` if it was already pinned.
    pub fn add(&mut self, pin: Pin) -> bool {
        if self.is_pinned(pin.provider, &pin.model) {
            return false;
        }
        self.pins.push(pin);
        true
    }

    /// Give a lease back early.
    ///
    /// Refuses to touch a pin, for the mirror of the reason `lease` refuses to
    /// replace one: `unpin` is the verb that clears permanent protection, and
    /// a client releasing its own lease must not be able to clear a pin
    /// somebody set by hand.
    pub fn release(&mut self, provider: ProviderKind, model: &str) -> ReleaseOutcome {
        match self.pins.iter().position(|p| p.provider == provider && p.model == model) {
            None => ReleaseOutcome::NotHeld,
            Some(i) if self.pins[i].expires_at.is_none() => ReleaseOutcome::WasPinned,
            Some(i) => {
                self.pins.remove(i);
                ReleaseOutcome::Released
            }
        }
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
