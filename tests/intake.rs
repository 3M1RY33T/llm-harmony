//! Task 8: pulling a built artifact from Hugging Face.
//!
//! The ordering is what these assert. Everything that can refuse a pull refuses
//! *before* a byte moves — provenance is read, both ledgers are checked, the
//! store is chosen — because a refusal at 97% is not a refusal, it is a waste.
//!
//! One test here is the inverse of what the plan first specified. The plan's
//! `a_repo_that_does_not_declare_its_base_model_is_refused_by_default` came
//! from `inventory.md` §5 and `architecture.md` §7, which both state provenance
//! as gating. That was **overridden on 2026-09-11** (`hosting-page-r27-plan.md`
//! §6): it warns and the pull proceeds, on the reasoning that a refusal is
//! worked around by downloading in a terminal where there is no check at all.
//! The test below therefore asserts the warning survives, which is now the
//! whole safety story.

mod support;

use llm_harmony::intake::download::{fetch, transfer_agent, Progress};
use llm_harmony::intake::provenance::{check, Finding};
use llm_harmony::intake::{hf, place};
use llm_harmony::inventory::artifact::{Format, Store};
use llm_harmony::provider::ProviderKind;
use support::tree::Tree;

fn json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap()
}

#[test]
fn a_repo_that_does_not_declare_its_base_model_is_pulled_with_a_warning() {
    // Was "refused by default" until the rule was changed to inform rather
    // than gate. What must not regress is the warning itself.
    let f = check(None, Some("qwen3-14b"));
    assert!(f.is_warning(), "the absence is surfaced");
    assert!(!matches!(f, Finding::Declared { .. }));
    assert!(f.message().contains("qwen3-14b"), "it names what was asked for");
}

#[test]
fn a_repo_declaring_a_different_model_names_both_so_a_person_can_judge() {
    let f = check(Some("Qwen/Qwen2.5-14B"), Some("qwen3-14b"));
    let m = f.message();
    assert!(m.contains("Qwen2.5-14B") && m.contains("qwen3-14b"), "{m}");
}

#[test]
fn a_repo_needing_conversion_is_reported_as_such_rather_than_refused_silently() {
    // bf16 is visible and honestly labelled: a user searching for a model
    // should see that a build exists and that harmony cannot yet use it.
    let repo = hf::repo_from_json(&json(
        r#"{"id":"x/y","siblings":[{"rfilename":"model.safetensors","size":10}]}"#,
    ))
    .unwrap();
    assert!(repo.needs_conversion());
    assert!(!repo.has_loadable_build());
}

#[test]
fn a_file_is_placed_in_the_store_for_the_provider_that_will_serve_it() {
    // Placement is the disk ledger's answer, not the user's guess: the same
    // GGUF belongs in different stores for different providers.
    assert_eq!(place::store_for(ProviderKind::LmStudio, Format::Gguf), Some(Store::LmStudio));
    assert_eq!(place::store_for(ProviderKind::LlamaCpp, Format::Gguf), Some(Store::LlamaCpp));
    let dir = place::directory_for(Store::LmStudio, "TheBloke/Qwen3-14B-GGUF").unwrap();
    assert!(dir.ends_with("TheBloke/Qwen3-14B-GGUF"), "{dir:?}");
}

#[test]
fn add_emits_one_json_document_per_progress_event() {
    // The caller is a UI polling a job. A percentage printed over itself is
    // not readable by one; a document per line is.
    let events = [
        Progress::Started { file: "m.gguf".into(), bytes_total: Some(10), target: "/s".into() },
        Progress::Advanced { bytes_done: 5, bytes_total: Some(10) },
        Progress::Finished { file: "m.gguf".into(), bytes_done: 10, path: "/s/m.gguf".into() },
    ];
    let stream: Vec<String> = events.iter().map(|e| serde_json::to_string(e).unwrap()).collect();
    for line in &stream {
        assert!(!line.contains('\n'), "one document per LINE: {line}");
        serde_json::from_str::<serde_json::Value>(line).expect("each line parses alone");
    }
    let kinds: Vec<String> = stream
        .iter()
        .map(|l| json(l)["event"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(kinds, ["started", "advanced", "finished"]);
}

#[test]
fn an_aborted_download_leaves_nothing_a_provider_could_load() {
    // A server that promises ten bytes and delivers three. Without the
    // length check the rename would publish a truncated GGUF as a finished
    // one, and LM Studio would try to load it.
    let t = Tree::new("intake-abort");
    let dir = t.root.join("lmstudio/pub/repo");
    let s = support::StubServer::start_truncating("/m.gguf", "abc", 10);

    let mut seen = Vec::new();
    let err = fetch(
        &transfer_agent(),
        &format!("{}/m.gguf", s.base_url()),
        &dir,
        "m.gguf",
        &mut |p| seen.push(p),
    )
    .expect_err("a short read is a failed download, not a small model");

    // Which layer catches it is not the contract — ureq notices the peer
    // disconnect before the length check does, and on a clean short close the
    // length check is what fires. The contract is what is left on disk.
    assert!(!err.is_empty(), "a failure states a reason");
    assert!(!dir.join("m.gguf").exists(), "nothing a scanner would index");
    assert!(!dir.join("m.gguf.part").exists(), "and no leftover either");
    assert!(
        !seen.iter().any(|p| matches!(p, Progress::Finished { .. })),
        "and it never claimed to have finished"
    );
}

#[test]
fn a_completed_download_is_renamed_into_place_atomically() {
    let t = Tree::new("intake-complete");
    let dir = t.root.join("lmstudio/pub/repo");
    let s = support::StubServer::start_bytes("/m.gguf", "hello");

    let mut seen = Vec::new();
    let path = fetch(
        &transfer_agent(),
        &format!("{}/m.gguf", s.base_url()),
        &dir,
        "m.gguf",
        &mut |p| seen.push(p),
    )
    .expect("it completes");

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    assert!(!dir.join("m.gguf.part").exists(), "the .part is gone, not left beside it");
    assert!(matches!(seen.first(), Some(Progress::Started { .. })));
    assert!(matches!(seen.last(), Some(Progress::Finished { .. })));
}

// --- Task 11: Ollama's transport (r27 Phase C) ------------------------------
//
// `add` refused Ollama outright, and the refusal was right about the wrong
// thing: Ollama will not read a loose GGUF out of a directory, but that is a
// statement about the *transport*, not about the provider. Its store is
// content-addressed behind `ollama pull`, and Ollama pulls a GGUF repo straight
// from Hugging Face under an `hf.co/…` name — so the bridge is exact rather
// than a guess at a name in Ollama's own library.
//
// Admission does not move. Provenance is read and both ledgers are checked
// before `ollama` is invoked at all: a second transport must not become a hole
// in the first one's rules.

use llm_harmony::intake::ollama as intake_ollama;
use llm_harmony::intake::run::Build;

fn build(files: &[(&str, Option<u64>)]) -> Build {
    let files: Vec<hf::RepoFile> = files
        .iter()
        .map(|(n, s)| hf::RepoFile {
            name: (*n).into(),
            format: Format::from_path_str(n),
            bits: llm_harmony::inventory::artifact::bits_from_name(n),
            size_bytes: *s,
        })
        .collect();
    let bytes = files.iter().try_fold(0u64, |acc, f| f.size_bytes.map(|b| acc + b));
    Build { files, bytes }
}

#[test]
fn an_ollama_target_is_planned_as_a_pull_rather_than_refused() {
    // The old `store_for` answer stays correct and stops being the end of the
    // road: there is no store to place into, and that is why there is a pull.
    assert_eq!(place::store_for(ProviderKind::Ollama, Format::Gguf), None);
    let t = intake_ollama::transport_for("unsloth/Qwen3.5-4B-GGUF", &build(&[("Qwen3.5-4B-Q4_K_M.gguf", Some(2))]));
    assert_eq!(t.unwrap(), "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M");
}

#[test]
fn the_registry_name_carries_the_quantisation_the_user_chose() {
    // Without the tag Ollama picks its own default, which is a different build
    // from the one that was priced and admitted -- so the number the user was
    // shown would describe a file they did not get.
    let t = intake_ollama::transport_for("x/y-GGUF", &build(&[("y-Q8_0.gguf", Some(2))])).unwrap();
    assert!(t.ends_with(":Q8_0"), "{t}");
}

#[test]
fn a_build_with_no_quantisation_in_its_name_is_pulled_untagged() {
    // Ollama's own default then applies. Inventing a tag would name a build
    // that may not exist in the repo at all.
    let t = intake_ollama::transport_for("x/y-GGUF", &build(&[("y.gguf", Some(2))])).unwrap();
    assert_eq!(t, "hf.co/x/y-GGUF");
}

#[test]
fn a_sharded_build_is_refused_for_ollama_by_name() {
    // `hf.co/…` resolves to ONE file. A split build pulled through it is one
    // shard wearing the whole build's name -- the exact failure `shard_of`
    // exists to prevent on the download path.
    let err = intake_ollama::transport_for(
        "x/y-GGUF",
        &build(&[
            ("y-Q4_K_M-00001-of-00002.gguf", Some(2)),
            ("y-Q4_K_M-00002-of-00002.gguf", Some(2)),
        ]),
    )
    .unwrap_err();
    assert!(err.contains("part"), "{err}");
}

#[test]
fn an_mlx_build_is_refused_for_ollama_rather_than_pulled_as_a_gguf() {
    // `hf.co/` is GGUF-only. Ollama serves MLX through its own backend, from
    // its own library, which is not this repo id.
    let err = intake_ollama::transport_for("x/y-MLX", &build(&[("model.safetensors", Some(2))]))
        .unwrap_err();
    assert!(err.contains("GGUF"), "{err}");
}

#[test]
fn the_pull_command_names_the_model_and_nothing_else() {
    assert_eq!(
        intake_ollama::pull_argv("hf.co/x/y:Q4_K_M"),
        vec!["ollama", "pull", "hf.co/x/y:Q4_K_M"]
    );
}

#[test]
fn a_missing_ollama_binary_is_reported_rather_than_skipped() {
    // The rule `adapters/lmstudio.rs` already states for `lms`: a missing
    // binary is a named failure, never a silent no-op, or harmony reports a
    // pull it never performed.
    let mut seen: Vec<Progress> = Vec::new();
    let err = intake_ollama::run_pull(
        &["definitely-not-ollama".into(), "pull".into(), "m".into()],
        &mut |p| seen.push(p),
    )
    .unwrap_err();
    assert!(err.to_lowercase().contains("path"), "{err}");
}

#[test]
fn an_ollama_pull_emits_the_same_progress_documents_as_a_download() {
    // Same contract, so a caller polling a job never special-cases the
    // provider: one document per line, `started` first and a terminal event
    // last. `sh` stands in for `ollama` so the shape is asserted without
    // pulling gigabytes.
    let mut seen: Vec<Progress> = Vec::new();
    let script = "printf 'pulling manifest\\n'; printf 'pulling 1a2b3c... 50%% 1.0 GB/2.0 GB\\n'; printf 'success\\n'";
    intake_ollama::run_pull(
        &["sh".into(), "-c".into(), script.into()],
        &mut |p| seen.push(p),
    )
    .unwrap();

    assert!(matches!(seen.first(), Some(Progress::Started { .. })), "{seen:?}");
    assert!(matches!(seen.last(), Some(Progress::Finished { .. })), "{seen:?}");
    assert!(
        seen.iter().any(|p| matches!(p, Progress::Advanced { .. })),
        "a percentage line becomes an Advanced event: {seen:?}"
    );
}

#[test]
fn a_failing_pull_is_an_error_and_not_a_silent_success() {
    let mut seen: Vec<Progress> = Vec::new();
    let err = intake_ollama::run_pull(
        &["sh".into(), "-c".into(), "printf 'Error: file does not exist\\n' >&2; exit 1".into()],
        &mut |p| seen.push(p),
    )
    .unwrap_err();
    assert!(err.contains("does not exist"), "the reason survives: {err}");
}

// --- r29 Task 1: a format is what the repo declares -------------------------
//
// `.safetensors` is a CONTAINER. What is inside it -- bf16, FP8, AWQ, GPTQ,
// INT4 -- is declared in `config.json`, and reading the extension instead
// reported every quantised repo on Hugging Face as bf16 needing a conversion.
//
// Found by pulling `mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-
// FP8-Dynamic` on 2026-09-11, which answered "publishes only bf16 safetensors"
// about a model quantised to 8 bits with vLLM's own llm-compressor. A stock
// vLLM loads that repo directly; harmony refused it for a conversion that was
// never needed.

use llm_harmony::inventory::artifact::Quant;

fn compressed_tensors_repo() -> serde_json::Value {
    json(
        r#"{"id":"mconcat/Qwen3.5-27B-FP8-Dynamic",
            "siblings":[{"rfilename":"model.safetensors","size":15000000000},
                        {"rfilename":"config.json"},{"rfilename":"recipe.yaml"}],
            "config":{"quantization_config":{
                "format":"float-quantized",
                "quant_method":"compressed-tensors",
                "config_groups":{"group_0":{"weights":{"num_bits":8}}}}}}"#,
    )
}

#[test]
fn a_compressed_tensors_repo_is_not_reported_as_bf16() {
    let repo = hf::repo_from_json(&compressed_tensors_repo()).unwrap();
    let weights = repo.files.iter().find(|f| f.name.ends_with(".safetensors")).unwrap();
    assert_eq!(weights.format, Format::Safetensors(Quant::CompressedTensors));
    assert_ne!(weights.format, Format::Safetensors(Quant::Bf16));
    // It DOES need converting -- on this machine, where no provider loads
    // safetensors. That is a fact about the providers, not about the repo,
    // and `formats()` is what settles it per provider (r29 Task 2). What
    // changed here is the reason: a convertible quantised source, rather
    // than a bf16 file it never was.
    assert!(repo.needs_conversion());
    assert!(matches!(weights.format, Format::Safetensors(q) if q.is_convertible()));
}

#[test]
fn awq_and_gptq_are_read_from_the_declaration_too() {
    for (method, want) in [("awq", Quant::Awq), ("gptq", Quant::Gptq),
                           ("fp8", Quant::Fp8), ("bitsandbytes", Quant::BitsAndBytes)] {
        let v = json(&format!(
            r#"{{"id":"x/y","siblings":[{{"rfilename":"model.safetensors","size":1}}],
                 "config":{{"quantization_config":{{"quant_method":"{method}"}}}}}}"#
        ));
        let repo = hf::repo_from_json(&v).unwrap();
        assert_eq!(repo.files[0].format, Format::Safetensors(want), "{method}");
    }
}

/// No extra request: `GET /api/models/{id}?blobs=true` already returns
/// `config`, and `repo_from_json` was parsing that very response and keeping
/// only the file list.
#[test]
fn the_declaration_is_read_from_the_response_already_fetched() {
    let repo = hf::repo_from_json(&compressed_tensors_repo()).unwrap();
    assert_eq!(repo.quant_method.as_deref(), Some("compressed-tensors"));
    assert_eq!(repo.quant_bits, Some(8));
}

/// A repo that declares nothing quantised IS bf16, and the old answer was
/// right for it. The fix must not make the common case worse.
#[test]
fn a_repo_with_no_quantization_config_is_still_bf16() {
    let v = json(r#"{"id":"x/y","siblings":[{"rfilename":"model.safetensors","size":1}]}"#);
    let repo = hf::repo_from_json(&v).unwrap();
    assert_eq!(repo.files[0].format, Format::Safetensors(Quant::Bf16));
    assert!(repo.needs_conversion(), "bf16 still needs one");
}

/// A declaration this build does not recognise is `Unknown`, never bf16.
/// Guessing "probably bf16" is how a 55 GB dequantisation gets planned for a
/// file that was never 16-bit.
#[test]
fn an_unrecognised_quant_method_is_unknown_rather_than_assumed() {
    let v = json(
        r#"{"id":"x/y","siblings":[{"rfilename":"model.safetensors","size":1}],
            "config":{"quantization_config":{"quant_method":"some-new-thing"}}}"#,
    );
    let repo = hf::repo_from_json(&v).unwrap();
    assert_eq!(repo.files[0].format, Format::Safetensors(Quant::Unknown));
    // The raw declaration survives, so a refusal can name what it saw.
    assert_eq!(repo.quant_method.as_deref(), Some("some-new-thing"));
    assert!(!repo.needs_conversion(), "harmony does not know it is convertible either");
}

/// A GGUF repo is unaffected by any of this.
#[test]
fn a_gguf_repo_is_still_a_gguf_repo() {
    let v = json(
        r#"{"id":"x/y-GGUF","siblings":[{"rfilename":"y-Q4_K_M.gguf","size":1}],
            "config":{"quantization_config":{"quant_method":"awq"}}}"#,
    );
    let repo = hf::repo_from_json(&v).unwrap();
    assert_eq!(repo.files[0].format, Format::Gguf);
    assert!(repo.has_loadable_build());
}

/// MLX on Hugging Face is **safetensors**, not `.npz`.
///
/// `Format::from_path_str` only ever produces `Mlx` for a `.npz`, which is the
/// old format nobody publishes any more. Every current MLX repo ships
/// `model-0000N-of-0000M.safetensors` and declares itself with
/// `library_name: "mlx"` — so intake classified all of them as PyTorch
/// safetensors and could not pull a single MLX model, on a machine where two
/// of four providers serve MLX natively.
///
/// Found 2026-09-11 reading `Jackrong/MLX-Qwen3.5-27B-…-4bit`, whose files are
/// safetensors and whose `library_name` says what they are.
#[test]
fn an_mlx_repo_is_mlx_however_its_files_are_named() {
    let v = json(
        r#"{"id":"Jackrong/MLX-Qwen3.5-27B-4bit","library_name":"mlx","tags":["mlx"],
            "siblings":[{"rfilename":"model-00001-of-00003.safetensors","size":5000000000},
                        {"rfilename":"model-00002-of-00003.safetensors","size":5000000000}],
            "config":{"quantization_config":{"bits":4}}}"#,
    );
    let repo = hf::repo_from_json(&v).unwrap();
    assert!(repo.files.iter().all(|f| f.format == Format::Mlx), "{:?}", repo.files);
    assert!(repo.has_loadable_build(), "two of four providers serve this natively");
    assert!(!repo.needs_conversion());
}

/// The tag alone is enough where `library_name` is absent — HF sets one or
/// both, and a repo that says `mlx` in either place is saying it.
#[test]
fn an_mlx_tag_is_read_when_library_name_is_missing() {
    let v = json(
        r#"{"id":"x/y","tags":["mlx","text-generation"],
            "siblings":[{"rfilename":"model.safetensors","size":1}]}"#,
    );
    assert_eq!(hf::repo_from_json(&v).unwrap().files[0].format, Format::Mlx);
}

/// And a GGUF file in an MLX-tagged repo is still a GGUF file. The library
/// declaration resolves what a safetensors container holds; it does not
/// overrule an extension that is already unambiguous.
#[test]
fn an_unambiguous_extension_is_not_overruled_by_the_library_tag() {
    let v = json(
        r#"{"id":"x/y","library_name":"mlx",
            "siblings":[{"rfilename":"y-Q4_K_M.gguf","size":1}]}"#,
    );
    assert_eq!(hf::repo_from_json(&v).unwrap().files[0].format, Format::Gguf);
}

// --- Two regressions found pulling a real MLX repo, 2026-09-11 -------------
//
// `igorvibes/Qwen3.8-27B-UD-Q6_K_XL-AWQ-MTP-mlx`: six safetensors shards
// totalling 25.6 GB, correctly declaring `base_model: Qwen/Qwen3.8-27B`.
// `add` answered with two warnings, both wrong, and planned a pull of 752 MB.

/// Sharding is not a GGUF-only idea.
///
/// `shard_of` stripped `.gguf` and nothing else, so MLX safetensors shards —
/// `model-00001-of-00006.safetensors` — were six separate "builds" and
/// "smallest loadable" chose `…-00006-of-00006`: **752 MB of a 25.6 GB
/// model**, one sixth, which nothing could ever load.
///
/// This is the exact failure `shard_of`'s own doc comment records against
/// `unsloth/DeepSeek-V3.1-GGUF`, arriving through a door that opened the day
/// MLX repos became recognisable at all (r29 Task 1): before that they were
/// misclassified as PyTorch and refused, so the gap could not show.
#[test]
fn safetensors_shards_are_folded_into_one_build_like_gguf_shards() {
    let v = json(
        r#"{"id":"igorvibes/Q-mlx","library_name":"mlx",
            "siblings":[
              {"rfilename":"model-00001-of-00006.safetensors","size":5330000000},
              {"rfilename":"model-00002-of-00006.safetensors","size":5350000000},
              {"rfilename":"model-00003-of-00006.safetensors","size":5340000000},
              {"rfilename":"model-00004-of-00006.safetensors","size":5320000000},
              {"rfilename":"model-00005-of-00006.safetensors","size":5370000000},
              {"rfilename":"model-00006-of-00006.safetensors","size":788940645}]}"#,
    );
    let repo = hf::repo_from_json(&v).unwrap();
    let build = llm_harmony::intake::run::best_build(&repo).expect("one build");
    assert_eq!(build.files.len(), 6, "a build is the SET, not its smallest part");
    assert_eq!(build.bytes, Some(27_498_940_645));
    assert!(build.label().contains("6 parts"), "{}", build.label());
}

/// And a repo that publishes several whole builds still offers a choice.
#[test]
fn unsharded_builds_are_still_separate_builds() {
    let v = json(
        r#"{"id":"x/y-GGUF","siblings":[
              {"rfilename":"y-Q4_K_M.gguf","size":5000000000},
              {"rfilename":"y-Q8_0.gguf","size":9000000000}]}"#,
    );
    let repo = hf::repo_from_json(&v).unwrap();
    assert_eq!(llm_harmony::intake::run::builds(&repo).len(), 2);
}

/// A build's `base_model` points at its PARENT. Differing from the repo's own
/// name is the entire point of the field, so comparing the two called every
/// correctly-labelled quantisation a mismatch:
///
///   "this repo declares base_model `Qwen/Qwen3.8-27B`, but you asked for
///    `igorvibes/Qwen3.8-27B-UD-Q6_K_XL-AWQ-MTP-mlx` — they are not builds of
///    the same model"
///
/// They are. This module's own docstring had already recorded the rule — the
/// call site in `plan_add` was not following it.
#[test]
fn a_build_naming_its_parent_is_not_a_mismatch() {
    let v = json(
        r#"{"id":"igorvibes/Qwen3.8-27B-UD-Q6_K_XL-AWQ-MTP-mlx","library_name":"mlx",
            "cardData":{"base_model":"Qwen/Qwen3.8-27B"},
            "siblings":[{"rfilename":"model-00001-of-00002.safetensors","size":5000000000},
                        {"rfilename":"model-00002-of-00002.safetensors","size":5000000000}]}"#,
    );
    let repo = hf::repo_from_json(&v).unwrap();
    let found = llm_harmony::intake::provenance::check_repo(
        repo.base_model.as_deref(),
        &repo.id,
    );
    assert!(!found.is_warning(), "{}", found.message());
}

/// The check that remains: a declaration the repo's own name does not sit
/// under. `inventory.md` §5's first case — a repo declaring a *different*
/// model with a near-identical name.
#[test]
fn a_declaration_the_repo_name_does_not_descend_from_is_still_a_warning() {
    let found = llm_harmony::intake::provenance::check_repo(
        Some("nightmedia/Qwen3-14B-DS9-USS-Defiant"),
        "someone/Llama-3.3-70B-Instruct-GGUF",
    );
    assert!(found.is_warning(), "unrelated lineage must still be shown");
    assert!(found.message().contains("Defiant"), "{}", found.message());
}

/// And declaring nothing is still worth saying, which is the case that
/// actually bites: most hand-converted GGUF repos fill in nothing at all.
#[test]
fn a_repo_declaring_nothing_is_still_a_warning() {
    assert!(llm_harmony::intake::provenance::check_repo(None, "x/y-GGUF").is_warning());
}
