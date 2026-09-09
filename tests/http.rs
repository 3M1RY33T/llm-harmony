mod support;

use std::time::Duration;

use llm_harmony::http::Http;
use llm_harmony::provider::ProbeError;

fn http() -> Http {
    Http::new(Duration::from_millis(1500))
}

#[test]
fn parses_a_json_body() {
    let s = support::StubServer::start(support::routes(&[("/ok", r#"{"models":[]}"#)]));
    let v = http().get_json(&format!("{}/ok", s.base_url())).unwrap();
    assert!(v["models"].is_array());
}

#[test]
fn a_closed_port_is_not_listening_not_an_error() {
    let err = http().get_json("http://127.0.0.1:1/anything").unwrap_err();
    assert_eq!(err, ProbeError::NotListening);
}

#[test]
fn a_404_is_reported_with_its_code() {
    let s = support::StubServer::start(support::routes(&[]));
    let err = http().get_json(&format!("{}/missing", s.base_url())).unwrap_err();
    assert_eq!(err, ProbeError::Status { code: 404 });
}

#[test]
fn a_non_json_body_is_malformed_not_a_panic() {
    let mut r = support::routes(&[]);
    r.insert("/html".to_string(), (200, "<html>hello</html>".to_string()));
    let s = support::StubServer::start(r);
    let err = http().get_json(&format!("{}/html", s.base_url())).unwrap_err();
    assert!(matches!(err, ProbeError::Malformed { .. }), "got {err:?}");
}
