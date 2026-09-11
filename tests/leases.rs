//! A lease is a pin that expires, and that is the only difference.
//!
//! Slice 5 left a gap: a pin is "a veto, never a reservation" and sticky until
//! `unpin`, so a caller that is still using a model has no way to say so
//! except a manual pin that outlives its reason -- and `pins.rs` names exactly
//! that failure, "a pin nobody remembers setting". A lease closes it by
//! putting an owner and an expiry on the same store, because eviction has one
//! place to ask and two stores would be two answers.
//!
//! Kept out of `tests/pins.rs` deliberately: that file is the store's own
//! round-trip contract, and expiry is a separate question about time.

mod support;

use llm_harmony::pins::{LeaseOutcome, Pin, Pins};
use llm_harmony::provider::ProviderKind;
use support::tree::Tree;

const NOW: u64 = 1_789_000_000;
const HOUR: u64 = 3600;

fn lease(model: &str, expires_at: u64, owner: &str) -> Pin {
    Pin {
        provider: ProviderKind::LmStudio,
        model: model.to_string(),
        at: NOW,
        note: None,
        owner: Some(owner.to_string()),
        expires_at: Some(expires_at),
    }
}

fn pin(model: &str) -> Pin {
    Pin {
        provider: ProviderKind::LmStudio,
        model: model.to_string(),
        at: NOW,
        note: None,
        owner: None,
        expires_at: None,
    }
}

/// The gap slice 5 left, closed.
#[test]
fn a_lease_protects_until_it_expires_and_then_stops() {
    let mut p = Pins::empty();
    assert_eq!(p.lease(lease("qwen3-14b", NOW + HOUR, "delroy")), LeaseOutcome::Taken);
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", NOW + HOUR - 1).is_some());
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", NOW + HOUR).is_none());
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", NOW + HOUR + 1).is_none());
}

/// A pin is a lease with no expiry, and nothing about it is time-dependent.
#[test]
fn a_pin_has_no_expiry_and_protects_at_any_time() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", u64::MAX - 1).is_some());
}

/// Protection is keyed on the pair, as `pins.rs` requires: the same id on
/// another provider is a different artifact with a different footprint.
#[test]
fn a_lease_does_not_protect_the_same_name_on_another_provider() {
    let mut p = Pins::empty();
    p.lease(lease("qwen3-14b", NOW + HOUR, "delroy"));
    assert!(p.holds(ProviderKind::Ollama, "qwen3-14b", NOW).is_none());
}

/// Housekeeping, not policy: `holds` already ignores a lapsed lease, so a
/// sweep only stops the store accumulating dead rows.
#[test]
fn a_sweep_drops_lapsed_leases_and_never_a_pin() {
    let mut p = Pins::empty();
    p.lease(lease("expired", NOW - 1, "delroy"));
    p.add(pin("forever"));
    assert_eq!(p.sweep(NOW), 1);
    assert_eq!(p.all().len(), 1);
    assert_eq!(p.all()[0].model, "forever");
}

/// A client still working has to be able to extend, and asking again must
/// never shorten what it already holds.
#[test]
fn a_second_lease_extends_and_never_shortens() {
    let mut p = Pins::empty();
    p.lease(lease("m", NOW + HOUR, "delroy"));
    assert_eq!(p.lease(lease("m", NOW + 2 * HOUR, "delroy")), LeaseOutcome::Extended);
    assert_eq!(p.all()[0].expires_at, Some(NOW + 2 * HOUR));
    assert_eq!(p.lease(lease("m", NOW + 60, "delroy")), LeaseOutcome::AlreadyLonger);
    assert_eq!(p.all()[0].expires_at, Some(NOW + 2 * HOUR), "asking again must not shorten it");
}

/// Protection may never weaken by accident: a lease over a pin would quietly
/// convert permanent protection into temporary protection.
#[test]
fn a_lease_cannot_downgrade_an_existing_pin() {
    let mut p = Pins::empty();
    p.add(pin("qwen3-14b"));
    assert_eq!(p.lease(lease("qwen3-14b", NOW + HOUR, "delroy")), LeaseOutcome::RefusedPinned);
    assert_eq!(p.all()[0].expires_at, None);
    assert!(p.holds(ProviderKind::LmStudio, "qwen3-14b", u64::MAX - 1).is_some());
}

/// The store predates both fields, and `pins.rs` already holds that losing
/// every pin on upgrade would be a poor trade for strictness.
#[test]
fn a_store_written_before_leases_existed_still_loads() {
    let t = Tree::new("leases-compat");
    let path = t.root.join("pins.json");
    std::fs::write(&path, r#"{"schema":1,"pins":[{"provider":"ollama","model":"m","at":1}]}"#)
        .unwrap();
    let p = Pins::load_from(&path);
    assert!(p.holds(ProviderKind::Ollama, "m", u64::MAX - 1).is_some());
}

/// A lease survives the round trip, owner and expiry included, or a restart
/// would silently release everything.
#[test]
fn a_lease_round_trips_through_the_file() {
    let t = Tree::new("leases-roundtrip");
    let path = t.root.join("pins.json");
    let mut p = Pins::empty();
    p.lease(lease("qwen3-14b", NOW + HOUR, "delroy"));
    p.save_to(&path).unwrap();

    let back = Pins::load_from(&path);
    let held = back.holds(ProviderKind::LmStudio, "qwen3-14b", NOW).expect("still held");
    assert_eq!(held.owner.as_deref(), Some("delroy"));
    assert_eq!(held.expires_at, Some(NOW + HOUR));
}

/// "Something invisible is holding memory you can see" is the failure mode, so
/// a refusal has to be traceable to a holder without reading a state file.
#[test]
fn a_blame_line_names_the_owner_and_the_time_left() {
    let l = lease("qwen3-14b", NOW + 2 * HOUR, "delroy");
    let line = l.blame(NOW);
    assert!(line.contains("delroy"), "{line}");
    assert!(line.contains("qwen3-14b"), "{line}");
    assert!(line.contains("2.0h"), "{line}");
}

/// A hand-set pin has no owner and no clock, and its line stays the plain one
/// slice 5 already prints.
#[test]
fn a_plain_pin_blames_itself_exactly_as_before() {
    assert_eq!(pin("big-pinned").blame(NOW), "big-pinned on lmstudio");
}
