mod support;

use std::time::Duration;

use llm_harmony::adapters::lmstudio::LmStudio;
use llm_harmony::http::Http;
use llm_harmony::launch::ready;

#[test]
fn a_provider_that_answers_is_ready_immediately() {
    let body = std::fs::read_to_string("tests/fixtures/lmstudio/models-none-loaded.json").unwrap();
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let http = Http::new(Duration::from_millis(500));
    assert!(ready::wait_for(&LmStudio, &http, &s.base_url(), Duration::from_secs(2)).is_ok());
}

/// "Launched" is not "serving". A port that never opens must time out with a
/// message naming the provider, not hang.
#[test]
fn a_provider_that_never_answers_times_out() {
    let http = Http::new(Duration::from_millis(100));
    let err = ready::wait_for(
        &LmStudio,
        &http,
        "http://127.0.0.1:1",
        Duration::from_millis(400),
    )
    .unwrap_err();
    assert!(err.contains("did not answer"), "{err}");
}

/// A server that answers with something else is not this provider, and must
/// not be reported ready.
#[test]
fn a_wrong_provider_on_the_port_is_not_ready() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let http = Http::new(Duration::from_millis(200));
    assert!(ready::wait_for(&LmStudio, &http, &s.base_url(), Duration::from_millis(400)).is_err());
}
