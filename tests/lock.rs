//! One advisory lock, held across admit -> evict -> load -> verify.
//!
//! This is what removes the daemon. A grant never has to outlive a process
//! because the process that was granted room is the one that fills it, so
//! there is no window between the arithmetic and the load for a second caller
//! to slip through.

mod support;

use std::time::Duration;

use llm_harmony::actuate::lock::{ActuationLock, LockError};
use support::tree::Tree;

#[test]
fn a_second_acquirer_is_told_who_holds_it_and_since_when() {
    let t = Tree::new("lock-busy");
    let path = t.root.join("actuation.lock");

    let _held = ActuationLock::acquire_at(&path, "switch", Duration::from_millis(0))
        .expect("uncontended");

    match ActuationLock::acquire_at(&path, "load", Duration::from_millis(50)) {
        Err(LockError::Busy { holder, .. }) => {
            let h = holder.expect("the holder writes its identity into the file");
            assert_eq!(h.verb, "switch");
            assert_eq!(h.pid, std::process::id());
        }
        other => panic!("expected Busy, got {other:?}"),
    }
}

/// flock releases on process death, so a killed harmony cannot wedge the next
/// one. This is why the lock is flock and not a pidfile somebody has to reap.
#[test]
fn the_lock_is_released_when_the_holder_is_dropped() {
    let t = Tree::new("lock-drop");
    let path = t.root.join("actuation.lock");
    {
        let _held = ActuationLock::acquire_at(&path, "switch", Duration::from_millis(0)).unwrap();
    }
    assert!(
        ActuationLock::acquire_at(&path, "load", Duration::from_millis(0)).is_ok(),
        "a dropped lock is a free lock"
    );
}

/// A caller that waits must actually wait -- the point of the timeout is that
/// two callers queue rather than one failing instantly.
#[test]
fn waiting_takes_at_least_the_timeout_before_giving_up() {
    let t = Tree::new("lock-waits");
    let path = t.root.join("actuation.lock");
    let _held = ActuationLock::acquire_at(&path, "switch", Duration::from_millis(0)).unwrap();

    let started = std::time::Instant::now();
    let r = ActuationLock::acquire_at(&path, "load", Duration::from_millis(120));
    assert!(r.is_err());
    assert!(started.elapsed() >= Duration::from_millis(100), "gave up early");
}

/// The lock file lives beside the pins and the observation log, and creating
/// it must not require the directory to exist already.
#[test]
fn the_lock_directory_is_created_on_demand() {
    let t = Tree::new("lock-mkdir");
    let path = t.root.join("nested/deeper/actuation.lock");
    assert!(ActuationLock::acquire_at(&path, "load", Duration::from_millis(0)).is_ok());
    assert!(path.exists());
}

/// An unreadable holder record must not turn a Busy into a crash: the lock is
/// still held, and that is the part the caller needs.
#[test]
fn a_lock_held_with_an_unreadable_record_still_reports_busy() {
    let t = Tree::new("lock-garbled");
    let path = t.root.join("actuation.lock");
    let _held = ActuationLock::acquire_at(&path, "switch", Duration::from_millis(0)).unwrap();
    std::fs::write(&path, b"not json").unwrap();

    match ActuationLock::acquire_at(&path, "load", Duration::from_millis(10)) {
        Err(LockError::Busy { holder, .. }) => assert!(holder.is_none()),
        other => panic!("expected Busy, got {other:?}"),
    }
}
