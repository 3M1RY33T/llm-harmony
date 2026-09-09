use std::time::Duration;

use crate::provider::ProbeError;

/// A timeout-bounded HTTP client.
///
/// Every failure becomes a `ProbeError` rather than propagating, because no
/// provider may make `status` fail.
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    pub fn new(timeout: Duration) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build()
            .into();
        Http { agent }
    }

    pub fn get_json(&self, url: &str) -> Result<serde_json::Value, ProbeError> {
        let mut response = self.agent.get(url).call().map_err(map_err)?;
        response
            .body_mut()
            .read_json::<serde_json::Value>()
            .map_err(|e| ProbeError::Malformed {
                reason: format!("not JSON: {e}"),
            })
    }
}

fn map_err(e: ureq::Error) -> ProbeError {
    match e {
        ureq::Error::StatusCode(code) => ProbeError::Status { code },
        ureq::Error::Timeout(_) => ProbeError::Timeout,
        ureq::Error::Io(io) => match io.kind() {
            std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::AddrNotAvailable
            | std::io::ErrorKind::ConnectionReset => ProbeError::NotListening,
            std::io::ErrorKind::TimedOut => ProbeError::Timeout,
            _ => ProbeError::Malformed {
                reason: io.to_string(),
            },
        },
        other => ProbeError::Malformed {
            reason: other.to_string(),
        },
    }
}
