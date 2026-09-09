use std::time::{Duration, Instant};

use crate::http::Http;
use crate::provider::Adapter;

/// Poll until the provider answers as itself, or give up.
///
/// A start is only successful when `probe()` succeeds. The command exiting 0
/// proves nothing: llama-server takes seconds to map a model, and a port that
/// never opens is the common failure. This is what `Adapter::probe` was
/// defined for in slice 1 -- until now nothing but tests had called it.
pub fn wait_for(
    adapter: &dyn Adapter,
    http: &Http,
    base: &str,
    timeout: Duration,
) -> Result<Duration, String> {
    let started = Instant::now();
    let step = Duration::from_millis(200);
    loop {
        if adapter.probe(http, base).is_ok() {
            return Ok(started.elapsed());
        }
        if started.elapsed() >= timeout {
            return Err(format!(
                "{} did not answer at {base} within {:.0}s",
                adapter.kind(),
                timeout.as_secs_f64()
            ));
        }
        std::thread::sleep(step);
    }
}
