mod support;

use std::time::Duration;

use llm_harmony::adapters::lmstudio::LmStudio;
use llm_harmony::http::Http;
use llm_harmony::provider::{Adapter, ProbeError, ProviderKind, State};

fn http() -> Http {
    Http::new(Duration::from_millis(1500))
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{name}")).expect("fixture exists")
}

#[test]
fn lmstudio_lists_every_model_with_its_state() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();

    assert_eq!(models.len(), 3);
    assert!(models.iter().all(|m| m.state == State::NotLoaded));
    assert_eq!(models[1].id, "qwen3-14b-claude-4.5-opus-high-reasoning-distill");
}

/// The bug that already happened once by hand. It does not get to happen again.
#[test]
fn lmstudio_never_reads_max_context_length_as_a_serving_window() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();

    for m in &models {
        assert_eq!(
            m.context_tokens, None,
            "{} is not loaded; max_context_length must not leak into context_tokens",
            m.id
        );
    }
}

#[test]
fn lmstudio_reads_the_serving_window_only_when_loaded() {
    let body = fixture("lmstudio/models-one-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();

    let loaded: Vec<_> = models.iter().filter(|m| m.state == State::Loaded).collect();
    assert_eq!(loaded.len(), 1);
    assert_eq!(
        loaded[0].context_tokens,
        Some(8192),
        "must be loaded_context_length (8192), never max_context_length (40960)"
    );
    assert!(models.iter().filter(|m| m.state != State::Loaded).all(|m| m.context_tokens.is_none()));
}

#[test]
fn lmstudio_reports_no_weights_because_it_does_not_publish_them() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let models = LmStudio.list(&http(), &s.base_url()).unwrap();
    assert!(models.iter().all(|m| m.weights_bytes.is_none()));
}

/// LM Studio answers unknown paths with 200 and an error body, so `probe`
/// must validate shape rather than status. Verified against the real server
/// 2026-09-09.
#[test]
fn lmstudio_probe_rejects_a_two_hundred_that_is_actually_an_error() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = LmStudio.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::LmStudio });
}

#[test]
fn lmstudio_probe_accepts_a_real_model_list() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    assert!(LmStudio.probe(&http(), &s.base_url()).is_ok());
}

use llm_harmony::adapters::ollama::Ollama;

#[test]
fn ollama_reports_nothing_when_nothing_is_resident() {
    let ps = fixture("ollama/ps-empty.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();
    assert!(models.iter().all(|m| m.state != State::Loaded));
}

#[test]
fn ollama_reads_serving_window_and_weights_for_a_resident_model() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();

    let m = models.iter().find(|m| m.state == State::Loaded).expect("one loaded");
    assert_eq!(m.id, "nomic-embed-text:latest");
    assert_eq!(m.context_tokens, Some(2048), "from /api/ps, the serving window");
    assert_eq!(m.weights_bytes, Some(274_302_450), "Ollama is the one provider that publishes size");
}

/// `/api/tags` carries `details.context_length` for models that are NOT
/// loaded. Same class of trap as LM Studio's `max_context_length`.
/// Verified live 2026-09-09.
#[test]
fn ollama_never_takes_a_window_from_the_tags_endpoint() {
    let ps = fixture("ollama/ps-empty.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();

    let m = models.iter().find(|m| m.id == "nomic-embed-text:latest").unwrap();
    assert_eq!(m.state, State::NotLoaded);
    assert_eq!(
        m.context_tokens, None,
        "details.context_length is 2048 in tags.json and must not be read"
    );
    assert_eq!(m.weights_bytes, Some(274_302_450), "size is a file property and IS safe to read");
}

#[test]
fn ollama_probe_rejects_a_server_that_is_not_ollama() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = Ollama.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::Ollama });
}

use llm_harmony::adapters::llamacpp::LlamaCpp;

/// Every llamacpp fixture below was captured from a live router on
/// 2026-09-09. The previous set was constructed from documentation and was
/// wrong in three ways -- a `meta` block that does not exist, an `n_ctx` that
/// is never published, and a `/running` endpoint that 404s.
#[test]
fn llamacpp_reads_state_from_the_status_object() {
    let models = fixture("llamacpp/v1-models-one-loaded.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &models)]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();

    assert_eq!(out.len(), 2);
    let loaded: Vec<_> = out.iter().filter(|m| m.state == State::Loaded).collect();
    assert_eq!(loaded.len(), 1, "status.value == loaded is the only signal");
}

#[test]
fn llamacpp_lists_everything_the_router_scanned_as_not_loaded() {
    let models = fixture("llamacpp/v1-models.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &models)]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|m| m.state == State::NotLoaded));
}

/// This build publishes no window at all. `None` beats a number nothing
/// measured -- the same rule vLLM-MLX gets.
#[test]
fn llamacpp_publishes_no_serving_window_so_reports_none() {
    let models = fixture("llamacpp/v1-models-one-loaded.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &models)]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();
    assert!(out.iter().all(|m| m.context_tokens.is_none()));
}

/// It no longer calls /running at all: that endpoint 404s on this build.
#[test]
fn llamacpp_needs_only_the_models_endpoint() {
    let models = fixture("llamacpp/v1-models.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &models)]));
    assert!(LlamaCpp.list(&http(), &s.base_url()).is_ok(), "no /running route served");
}

#[test]
fn llamacpp_probe_rejects_lmstudio_on_the_same_shape() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = LlamaCpp.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::LlamaCpp });
}

use llm_harmony::adapters::vllm::Vllm;

/// vLLM-MLX reports neither a window nor a size. It gets None, not a guess.
#[test]
fn vllm_reports_a_model_with_no_window_and_no_weights() {
    let body = fixture("vllm/v1-models.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, "Qwen3.5-9B-MLX-4bit");
    assert_eq!(out[0].state, State::Loaded, "presence in /v1/models is residency");
    assert_eq!(out[0].context_tokens, None, "vLLM-MLX publishes no window; never guess one");
    assert_eq!(out[0].weights_bytes, None);
}

#[test]
fn vllm_probe_rejects_lmstudio_style_error_bodies() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = Vllm.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::Vllm });
}
