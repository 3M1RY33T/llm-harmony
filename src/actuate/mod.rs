//! Changing the ledger rather than describing it.
//!
//! Everything here can unload a model, which is the authority slices 1-4 were
//! built to withhold. Three things make that safe to grant:
//!
//! * **one lock** ([`lock`]), held across admit -> evict -> load -> verify, so
//!   admission and the load it authorises cannot be separated;
//! * **a plan** ([`plan`]), computed before anything is touched, which pins
//!   veto and busy models are exempt from;
//! * **a watchdog** ([`watchdog`]), which decides an in-flight load has gone
//!   wrong on evidence rather than on the estimate that authorised it.

pub mod lock;
