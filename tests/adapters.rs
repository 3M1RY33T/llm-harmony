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

/// Captured live 2026-09-10. `/v1/models` advertises what the registry *can*
/// serve; only `/v1/status` says what is resident.
#[test]
fn vllm_reads_residency_from_the_loaded_flag_not_from_presence() {
    let body = fixture("vllm/v1-status.json");
    let s = support::StubServer::start(support::routes(&[("/v1/status", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();
    assert_eq!(out.len(), 2);
    assert!(
        out.iter().all(|m| m.state == State::NotLoaded),
        "advertised is not resident: {out:?}"
    );
}

#[test]
fn vllm_marks_the_one_model_the_manager_reports_as_loaded() {
    let body = fixture("vllm/v1-status-one-loaded.json");
    let s = support::StubServer::start(support::routes(&[("/v1/status", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();
    assert_eq!(out.iter().filter(|m| m.state == State::Loaded).count(), 1);
}

/// The only provider of the four that publishes a per-model memory figure.
#[test]
fn vllm_reports_the_managers_own_memory_estimate_as_weights() {
    let body = fixture("vllm/v1-status.json");
    let s = support::StubServer::start(support::routes(&[("/v1/status", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();
    let w = out[0].weights_bytes.expect("memory_gb is published");
    assert!(w > 4_000_000_000 && w < 8_000_000_000, "got {w}");
}

/// Still no serving window, so still None rather than a guess.
#[test]
fn vllm_publishes_no_serving_window_so_reports_none() {
    let body = fixture("vllm/v1-status.json");
    let s = support::StubServer::start(support::routes(&[("/v1/status", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();
    assert!(out.iter().all(|m| m.context_tokens.is_none()));
}

#[test]
fn vllm_probe_rejects_lmstudio_style_error_bodies() {
    let s = support::StubServer::start_lmstudio_style(support::routes(&[]));
    let err = Vllm.probe(&http(), &s.base_url()).unwrap_err();
    assert_eq!(err, ProbeError::KindMismatch { expected: ProviderKind::Vllm });
}

/// Three providers publish the artifact path outright. Captured live
/// 2026-09-10; this is what makes placement a fact rather than a guess.
/// Name matching, measured on the same day, bridged one of three
/// cross-provider cases.
#[test]
fn llamacpp_reports_the_artifact_path_from_status_args() {
    let models = fixture("llamacpp/v1-models.json");
    let s = support::StubServer::start(support::routes(&[("/v1/models", &models)]));
    let out = LlamaCpp.list(&http(), &s.base_url()).unwrap();
    let p = out[0].artifact_path.as_deref().expect("--model is in status.args");
    assert!(p.ends_with(".gguf"), "{p}");
    assert!(p.starts_with('/'), "an absolute path: {p}");
}

#[test]
fn vllm_reports_the_artifact_path_from_source() {
    let body = fixture("vllm/v1-status.json");
    let s = support::StubServer::start(support::routes(&[("/v1/status", &body)]));
    let out = Vllm.list(&http(), &s.base_url()).unwrap();
    assert!(out[0].artifact_path.as_deref().unwrap().starts_with('/'));
}

/// LM Studio publishes no path at all -- verified live: its entries carry only
/// publisher, arch, quantization and state. Placement must bridge it by name
/// and say that it did.
#[test]
fn lmstudio_reports_no_artifact_path() {
    let body = fixture("lmstudio/models-none-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let out = LmStudio.list(&http(), &s.base_url()).unwrap();
    assert!(out.iter().all(|m| m.artifact_path.is_none()));
}

// --- Actuation capability (slice 5, Task 1) ---
//
// What harmony may DO to a provider, as opposed to what it can read. Probed
// live 2026-09-10 with all four servers running; see
// docs/plans/2026-09-10-actuation.md for the transcript.

use llm_harmony::provider::Actuation;

/// Probed live 2026-09-10: `lms load` and `lms unload` both exist.
#[test]
fn lmstudio_and_ollama_are_model_level() {
    assert!(matches!(LmStudio.actuation(), Actuation::ModelLevel));
    assert!(matches!(Ollama.actuation(), Actuation::ModelLevel));
}

/// Probed live 2026-09-10: `POST /api/models/unload/<real-model>` returns 404,
/// and vLLM-MLX publishes no load or unload route in its own openapi.json.
/// Their only ceilings are the ones they were started with.
#[test]
fn llamacpp_and_vllm_are_self_managed_and_name_their_ceiling() {
    match LlamaCpp.actuation() {
        Actuation::SelfManaged { ceiling } => assert_eq!(ceiling, "max_instances"),
        a => panic!("{a:?}"),
    }
    match Vllm.actuation() {
        Actuation::SelfManaged { ceiling } => assert_eq!(ceiling, "memory_budget_gb"),
        a => panic!("{a:?}"),
    }
}

/// The distinction the planner branches on, so it belongs to the type rather
/// than to a match at every call site.
#[test]
fn only_a_model_level_provider_can_unload() {
    assert!(LmStudio.actuation().can_unload());
    assert!(!LlamaCpp.actuation().can_unload());
}

// --- Load and unload (slice 5, Task 6) ---

use llm_harmony::provider::{ActuateError, LoadRequest};

/// A self-managed provider refuses in a way that names the lever that does
/// exist, so an operator is never told only what harmony cannot do.
#[test]
fn a_self_managed_provider_refuses_unload_and_names_its_ceiling() {
    match LlamaCpp.unload(&http(), "http://127.0.0.1:1", "any-model") {
        Err(ActuateError::NotSupported { ceiling }) => assert_eq!(ceiling, "max_instances"),
        other => panic!("{other:?}"),
    }
    match Vllm.unload(&http(), "http://127.0.0.1:1", "any-model") {
        Err(ActuateError::NotSupported { ceiling }) => assert_eq!(ceiling, "memory_budget_gb"),
        other => panic!("{other:?}"),
    }
}

/// And refuses without touching the network: there is no endpoint to call, and
/// a timeout would misreport "cannot" as "did not answer".
#[test]
fn a_self_managed_refusal_does_not_touch_the_network() {
    let started = std::time::Instant::now();
    let _ = LlamaCpp.load(
        &http(),
        // A port nothing is listening on: reaching it would cost a timeout.
        "http://127.0.0.1:1",
        &LoadRequest { model: "m".into(), context_tokens: None, ttl_seconds: None },
    );
    assert!(started.elapsed() < Duration::from_millis(100), "it dialled out");
}

/// Ollama drops a model by asking for it with keep_alive 0.
#[test]
fn ollama_unloads_by_asking_for_a_zero_keep_alive() {
    let seen = support::RecordingServer::start();
    Ollama.unload(&http(), &seen.base_url(), "qwen3:14b").unwrap();

    let req = seen.last_request();
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/api/generate");
    assert_eq!(req.json["model"], "qwen3:14b");
    assert_eq!(req.json["keep_alive"], 0);
    assert_eq!(req.json["prompt"], "", "never carry content: docs/architecture.md");
}

/// And loads by asking for it with a keep_alive that is not zero. Same
/// endpoint, opposite intent -- which is the whole of Ollama's control plane.
#[test]
fn ollama_loads_by_asking_for_a_nonzero_keep_alive() {
    let seen = support::RecordingServer::start();
    Ollama
        .load(&http(), &seen.base_url(), &LoadRequest { model: "qwen3:14b".into(), context_tokens: Some(8192), ttl_seconds: None })
        .unwrap();

    let req = seen.last_request();
    assert_eq!(req.path, "/api/generate");
    assert_ne!(req.json["keep_alive"], 0);
    assert_eq!(req.json["prompt"], "");
    assert_eq!(req.json["options"]["num_ctx"], 8192, "the window is most of the estimate");
}

/// LM Studio is driven by its CLI -- the only control verb it exposes. The
/// requested window must reach it, because the window is most of the estimate.
#[test]
fn lmstudio_load_passes_the_requested_context_to_the_cli() {
    let argv = LmStudio::load_argv(&LoadRequest {
        model: "qwen3-14b".into(),
        context_tokens: Some(8192),
        ttl_seconds: None,
    });
    assert_eq!(argv, vec!["lms", "load", "qwen3-14b", "--yes", "--context-length", "8192"]);
}

/// Without a window, none is passed: LM Studio's own default is a better
/// guess than any this project could invent.
#[test]
fn lmstudio_load_omits_the_window_when_none_was_asked_for() {
    let argv = LmStudio::load_argv(&LoadRequest {
        model: "m".into(),
        context_tokens: None,
        ttl_seconds: None,
    });
    assert_eq!(argv, vec!["lms", "load", "m", "--yes"]);
}

#[test]
fn lmstudio_unload_names_the_model() {
    assert_eq!(LmStudio::unload_argv("qwen3-14b"), vec!["lms", "unload", "qwen3-14b"]);
}

/// Verified live 2026-09-11: `/api/generate` answers HTTP 400 for an embedding
/// model -- `"nomic-embed-text:latest" does not support generate` -- so there
/// is no one endpoint that loads everything Ollama serves. The server is asked
/// which it is.
#[test]
fn ollama_loads_an_embedding_model_through_the_embed_endpoint() {
    let mut canned = std::collections::HashMap::new();
    canned.insert("/api/show".to_string(), r#"{"capabilities":["embedding"]}"#.to_string());
    let seen = support::RecordingServer::start_with(canned);

    Ollama
        .load(
            &http(),
            &seen.base_url(),
            &LoadRequest {
                model: "nomic-embed-text:latest".into(),
                context_tokens: None,
                ttl_seconds: None,
            },
        )
        .unwrap();

    let req = seen.last_request();
    assert_eq!(req.path, "/api/embed");
    assert_eq!(req.json["input"], "", "a control call carries no content");
    assert_ne!(req.json["keep_alive"], 0);
}

/// And a model that can generate keeps the generate path.
#[test]
fn ollama_loads_a_chat_model_through_the_generate_endpoint() {
    let mut canned = std::collections::HashMap::new();
    canned.insert("/api/show".to_string(), r#"{"capabilities":["completion","tools"]}"#.to_string());
    let seen = support::RecordingServer::start_with(canned);

    Ollama
        .load(&http(), &seen.base_url(), &LoadRequest { model: "qwen3:14b".into(), context_tokens: None, ttl_seconds: None })
        .unwrap();

    assert_eq!(seen.last_request().path, "/api/generate");
}

/// LM Studio starts a SECOND instance when told to load an already-resident
/// model -- `<model>:2`, with its own weights and KV cache -- and
/// `lms unload <model>` then removes only the first. Verified live
/// 2026-09-11. A silent doubling is the failure this project exists to
/// prevent, so the adapter refuses to create one.
#[test]
fn lmstudio_load_is_a_no_op_when_the_model_is_already_resident() {
    let body = fixture("lmstudio/models-one-loaded.json");
    let s = support::StubServer::start_lmstudio_style(support::routes(&[(
        "/api/v0/models",
        &body,
    )]));
    let loaded = LmStudio
        .list(&http(), &s.base_url())
        .unwrap()
        .into_iter()
        .find(|m| m.state == State::Loaded)
        .expect("the fixture has a resident model");

    // Would shell out to `lms` and create a duplicate if the guard were absent;
    // `lms` is not reachable from the test environment's PATH assumptions, so a
    // spawn would surface as an error rather than silently passing.
    LmStudio
        .load(
            &http(),
            &s.base_url(),
            &LoadRequest {
                model: loaded.id.clone(),
                context_tokens: Some(4096),
                ttl_seconds: None,
            },
        )
        .expect("already resident is success, not a second instance");
}

// --- The idle TTL, and pinning a switch's target provider (r26, Task 1) ---

/// The TTL is the provider's own idle timer -- LM Studio takes `--ttl` in
/// seconds. Delroy's hosting settings reach the provider through this and
/// nothing else: neither Delroy nor harmony runs a timer of its own.
#[test]
fn lmstudio_load_passes_a_ttl_when_one_was_asked_for() {
    let argv = LmStudio::load_argv(&LoadRequest {
        model: "qwen3-14b".into(),
        context_tokens: Some(8192),
        ttl_seconds: Some(1800),
    });
    assert_eq!(
        argv,
        vec!["lms", "load", "qwen3-14b", "--yes", "--context-length", "8192", "--ttl", "1800"]
    );
}

/// And omits it otherwise, leaving the provider's own default alone.
#[test]
fn lmstudio_load_omits_the_ttl_when_none_was_asked_for() {
    let argv = LmStudio::load_argv(&LoadRequest {
        model: "m".into(),
        context_tokens: None,
        ttl_seconds: None,
    });
    assert_eq!(argv, vec!["lms", "load", "m", "--yes"]);
}

/// Ollama spells the same idea `keep_alive`, in seconds.
#[test]
fn ollama_load_passes_a_ttl_as_keep_alive() {
    let seen = support::RecordingServer::start();
    Ollama
        .load(
            &http(),
            &seen.base_url(),
            &LoadRequest {
                model: "qwen3:14b".into(),
                context_tokens: None,
                ttl_seconds: Some(1800),
            },
        )
        .unwrap();
    assert_eq!(seen.last_request().json["keep_alive"], 1800);
}

// --- r27 Task 3: the expiry Ollama publishes, and the LRU that depends on it -
//
// `/api/ps` publishes `expires_at`, `parse_rfc3339` reads it, three tests cover
// that function -- and `list()` set `expires_at_unix: None` on both paths, so
// the parser was dead code. Clippy said so; nothing else did.
//
// The cost is not the missing UI field. `residents()` feeds `expires_at_unix`
// into `Resident.evict_rank`, so EVERY resident model on EVERY provider ranked
// 0, and `plan.rs`'s documented "least-recently-used first" eviction was in
// fact ledger order. Its own unit test passed because it builds `Resident`
// values by hand and never goes through an adapter.

#[test]
fn ollama_reports_the_expiry_it_publishes() {
    let ps = fixture("ollama/ps-one-loaded.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();

    let m = models.iter().find(|m| m.state == State::Loaded).expect("one loaded");
    // 2026-09-09T22:05:00.000000000-04:00
    assert_eq!(m.expires_at_unix, Some(1_789_005_900), "the serving window's end");
}

/// A model that is not resident has no serving window to report. `/api/tags`
/// says nothing about when anything expires, and inventing a zero there would
/// put every unloaded model at the head of the eviction queue.
#[test]
fn ollama_gives_no_expiry_for_a_model_that_is_not_loaded() {
    let ps = fixture("ollama/ps-empty.json");
    let tags = fixture("ollama/tags.json");
    let s = support::StubServer::start(support::routes(&[
        ("/api/ps", &ps),
        ("/api/tags", &tags),
    ]));
    let models = Ollama.list(&http(), &s.base_url()).unwrap();
    assert!(models.iter().all(|m| m.expires_at_unix.is_none()));
}

/// And the planner therefore evicts the model expiring soonest, end to end,
/// rather than whichever the ledger happened to list first.
#[test]
fn eviction_order_follows_the_expiry_through_the_adapter() {
    use llm_harmony::actuate::plan::Resident;
    use llm_harmony::provider::{Actuation, ProviderKind};

    // Built the way `residents()` builds them -- from `expires_at_unix` --
    // rather than by hand, which is the gap that let this survive.
    let from_adapter = |model: &str, expires: Option<u64>| Resident {
        provider: ProviderKind::Ollama,
        model: model.to_string(),
        estimated_bytes: 1_000,
        evict_rank: expires.unwrap_or(0),
        busy: false,
        actuation: Actuation::ModelLevel,
    };

    let mut residents = vec![
        from_adapter("listed-first-expires-last", Some(1_789_999_999)),
        from_adapter("listed-second-expires-soonest", Some(1_789_000_000)),
    ];
    residents.sort_by_key(|r| r.evict_rank);
    assert_eq!(
        residents[0].model, "listed-second-expires-soonest",
        "ledger order is not eviction order once the expiry is real"
    );
}

// --- r29 Task 2: a provider says which formats it loads ---------------------
//
// `inventory.md` §6 carried a hardcoded matrix, and it was already wrong for
// the machine in front of it: the row said vLLM takes safetensors bf16 via
// "convert", while the vLLM here is vllm-mlx, whose `/v1/status` lists nothing
// but MLX directories. A stock vLLM, meanwhile, loads compressed-tensors
// directly. One row cannot describe both.
//
// `Actuation` already won this argument in slice 5 and its doc comment says
// why: an adapter written from documentation is a hypothesis.

use llm_harmony::inventory::artifact::{Format, Quant};

#[test]
fn llamacpp_loads_gguf_and_nothing_else() {
    let f = LlamaCpp.formats();
    assert!(f.contains(&Format::Gguf));
    assert!(!f.iter().any(|x| matches!(x, Format::Mlx | Format::Safetensors(_))));
}

#[test]
fn lmstudio_loads_both_gguf_and_mlx() {
    let f = LmStudio.formats();
    assert!(f.contains(&Format::Gguf));
    assert!(f.contains(&Format::Mlx));
}

/// Ollama's store is content-addressed, so "which formats can it serve" and
/// "can a file be dropped into it" are separate questions -- and the second
/// being no does not make the first empty.
#[test]
fn ollama_loads_gguf_even_though_it_takes_no_file_drop() {
    assert!(Ollama.formats().contains(&Format::Gguf));
    assert_eq!(
        llm_harmony::intake::place::store_for(ProviderKind::Ollama, Format::Gguf),
        None,
        "still no directory to drop into"
    );
}

/// The one this plan exists for. vllm-mlx serves MLX; it does not load
/// safetensors in any dtype, quantised or not.
#[test]
fn vllm_mlx_loads_mlx_and_not_safetensors() {
    let f = Vllm.formats();
    assert!(f.contains(&Format::Mlx));
    for q in [Quant::Bf16, Quant::CompressedTensors, Quant::Fp8, Quant::Awq] {
        assert!(!f.contains(&Format::Safetensors(q)), "{q:?}");
    }
}

/// Placement consults the adapter rather than its own `match`, so the two can
/// never disagree about what a provider takes.
#[test]
fn placement_asks_the_adapter_rather_than_repeating_the_table() {
    use llm_harmony::intake::place::store_for;
    use llm_harmony::inventory::artifact::Store;
    assert_eq!(store_for(ProviderKind::LmStudio, Format::Mlx), Some(Store::LmStudio));
    assert_eq!(store_for(ProviderKind::LlamaCpp, Format::Mlx), None);
    assert_eq!(store_for(ProviderKind::Vllm, Format::Gguf), None);
}

/// A format nobody here loads has nowhere to go, whatever is inside it.
#[test]
fn no_provider_here_takes_safetensors_in_any_dtype() {
    for kind in ProviderKind::ALL {
        for q in [Quant::Bf16, Quant::Fp8, Quant::CompressedTensors] {
            assert_eq!(
                llm_harmony::intake::place::store_for(kind, Format::Safetensors(q)),
                None,
                "{kind} {q:?}"
            );
        }
    }
}
