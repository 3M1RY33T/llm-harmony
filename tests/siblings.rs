//! `siblings`: another build of the same model, in a format this machine serves.
//!
//! The cheap answer to "this format cannot be served here" is usually not a
//! conversion. Thirty-four builds of the repo that triggered r29 already exist
//! on Hugging Face, several of them MLX, which this machine serves natively.
//! Converting its FP8 instead means 15 GB down, dequantised to ~55 GB of bf16,
//! requantised to 4-bit — an hour and two lossy steps to arrive at an artifact
//! somebody already published.
//!
//! The rule that shapes every test here: **a candidate is confirmed by reading
//! its repo.** llmfit's catalog labels the FP8 repo `format: gguf,
//! quantization: Q4_K_M` — both wrong — and calls repos merely *named* `MLX-…`
//! gguf. A sibling offered on a name would reproduce the bug r29 exists to fix,
//! one layer out.

mod support;

use llm_harmony::intake::hf;
use llm_harmony::intake::siblings::{self, Sibling};
use llm_harmony::inventory::artifact::{Format, Quant};
use llm_harmony::provider::ProviderKind;

fn json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap()
}

fn repo(id: &str, base: &str, file: &str, size: u64, quant: Option<&str>) -> hf::Repo {
    let q = quant
        .map(|m| format!(r#","config":{{"quantization_config":{{"quant_method":"{m}"}}}}"#))
        .unwrap_or_default();
    hf::repo_from_json(&json(&format!(
        r#"{{"id":"{id}","cardData":{{"base_model":"{base}"}},
             "siblings":[{{"rfilename":"{file}","size":{size}}}]{q}}}"#
    )))
    .unwrap()
}

/// The whole point: a build of the same model in a format that loads here.
///
/// The shape is the live one. `mconcat/…-FP8-Dynamic` declares its immediate
/// parent; the builds that replace it are published under that parent's name
/// with a format suffix, which is exactly what `canonical_name` strips.
#[test]
fn a_sibling_is_a_build_of_the_same_model_in_a_format_this_machine_serves() {
    let source = repo("mconcat/Qwen3.5-27B-Distilled-FP8-Dynamic",
                      "Jackrong/Qwen3.5-27B-Distilled",
                      "model.safetensors", 15_000_000_000, Some("compressed-tensors"));
    let candidates = vec![
        repo("Jackrong/Qwen3.5-27B-Distilled-GGUF", "Qwen/Qwen3.5-27B",
             "q4_k_m.gguf", 16_000_000_000, None),
        repo("other/Unrelated-Model-GGUF", "Qwen/Qwen3.5-27B", "q4.gguf", 9_000_000_000, None),
    ];
    let found = siblings::from_candidates(&source, &candidates, &[Format::Gguf]);
    assert_eq!(found.iter().map(|s| s.repo.as_str()).collect::<Vec<_>>(),
               vec!["Jackrong/Qwen3.5-27B-Distilled-GGUF"],
               "the build of the model the source names, not everything sharing a root");
}

/// And a format this machine does not serve is not an answer, however right
/// the model is.
#[test]
fn a_sibling_in_a_format_nothing_here_loads_is_not_offered() {
    let source = repo("mconcat/M-FP8", "org/M", "model.safetensors", 1, Some("fp8"));
    let exl3 = repo("someone/M-exl3", "org/M", "model.safetensors", 9_000_000_000, Some("awq"));
    assert!(siblings::from_candidates(&source, &[exl3], &[Format::Gguf, Format::Mlx]).is_empty());
}

/// Every fine-tune of Qwen3.5-27B declares `Qwen/Qwen3.5-27B`. Matching on a
/// shared declaration would make two unrelated fine-tunes siblings —
/// `inventory.md` §5's trap arriving through the declaration door rather than
/// the name one. So a shared base is deliberately NOT a ground.
#[test]
fn two_unrelated_finetunes_of_one_root_are_not_siblings() {
    let a = repo("alice/Storyteller-FP8", "Qwen/Qwen3.5-27B",
                 "model.safetensors", 1, Some("fp8"));
    let b = repo("bob/Coder-GGUF", "Qwen/Qwen3.5-27B", "q4.gguf", 9_000_000_000, None);
    assert!(siblings::from_candidates(&a, &[b], &[Format::Gguf]).is_empty());
}

/// Every candidate is confirmed by reading its repo, never by its name.
#[test]
fn a_candidate_is_confirmed_by_reading_its_repo_not_by_its_name() {
    let source = repo("x/Model-FP8", "org/Model", "model.safetensors", 1, Some("fp8"));
    // Named MLX, publishes safetensors. llmfit's catalog calls exactly this
    // kind of repo `gguf`; reading it says otherwise.
    let liar = repo("someone/MLX-Model-4bit", "org/Model", "model.safetensors", 1, Some("awq"));
    let found = siblings::from_candidates(&source, &[liar], &[Format::Mlx, Format::Gguf]);
    assert!(found.is_empty(), "a name is not a format");
}

/// Same model, by the identity rule `resolve` already uses — a shared
/// `base_model`, not a similar string. `inventory.md` §5 records two repos
/// that looked like builds of a model and were not.
#[test]
fn a_model_with_a_similar_name_is_not_a_sibling() {
    let source = repo("x/Model-FP8", "org/Model", "model.safetensors", 1, Some("fp8"));
    let other = repo("y/Model-v2-GGUF", "org/Model-v2", "q4.gguf", 1, None);
    assert!(siblings::from_candidates(&source, &[other], &[Format::Gguf]).is_empty());
}

/// A candidate that declares nothing is still offered when its **id** is the
/// model the source named.
///
/// The anchor is never a declaration alone: it is a repo id, which is a thing
/// that exists rather than a claim, canonicalised by the same function the
/// disk ledger uses to group one model's artifacts across five stores. The
/// publisher prefix is stripped there too, deliberately —
/// `TheBloke/Qwen3-14B-GGUF` and `Qwen/Qwen3-14B` are one model.
#[test]
fn a_candidate_is_anchored_on_its_id_rather_than_on_a_declaration() {
    let source = repo("x/Model-FP8", "org/Model", "model.safetensors", 1, Some("fp8"));
    let silent = hf::repo_from_json(&json(
        r#"{"id":"z/Model-GGUF","siblings":[{"rfilename":"q4.gguf","size":9000000000}]}"#,
    ))
    .unwrap();
    let found = siblings::from_candidates(&source, &[silent], &[Format::Gguf]);
    assert_eq!(found.len(), 1);
    // And the ground it was matched on is recorded, so a row can show it.
    assert_eq!(found[0].base_model, "org/Model");
}

/// The source itself is never its own sibling.
#[test]
fn the_repo_being_replaced_is_not_offered_as_its_own_replacement() {
    let source = repo("x/Model-GGUF", "org/Model", "q4.gguf", 1, None);
    let found = siblings::from_candidates(&source, &[source.clone()], &[Format::Gguf]);
    assert!(found.is_empty());
}

/// Smallest first: the reason anyone is looking at this list is that the
/// thing they asked for would not fit.
#[test]
fn siblings_are_ordered_by_what_they_cost() {
    let source = repo("x/M-FP8", "org/M", "model.safetensors", 1, Some("fp8"));
    let big = repo("a/M-GGUF", "org/M", "q8.gguf", 20_000_000_000, None);
    let small = repo("b/M-GGUF", "org/M", "q4.gguf", 5_000_000_000, None);
    let found = siblings::from_candidates(&source, &[big, small], &[Format::Gguf]);
    assert_eq!(found.iter().map(|s| s.repo.as_str()).collect::<Vec<_>>(),
               vec!["b/M-GGUF", "a/M-GGUF"]);
}

/// A build whose size the repo never published cannot be admitted against
/// either ledger, so it is not offered as the cheap way out.
#[test]
fn a_sibling_with_no_published_size_is_not_offered() {
    let source = repo("x/M-FP8", "org/M", "model.safetensors", 1, Some("fp8"));
    let no_size = hf::repo_from_json(&json(
        r#"{"id":"a/M-GGUF","cardData":{"base_model":"org/M"},
            "siblings":[{"rfilename":"q4.gguf"}]}"#,
    ))
    .unwrap();
    assert!(siblings::from_candidates(&source, &[no_size], &[Format::Gguf]).is_empty());
}

/// And a source that IS servable has no siblings to look for.
#[test]
fn a_servable_source_needs_no_sibling() {
    let source = repo("x/M-GGUF", "org/M", "q4.gguf", 1, None);
    let other = repo("y/M-GGUF", "org/M", "q8.gguf", 2, None);
    assert!(siblings::for_repo(&source, &[other], &[Format::Gguf], false).is_empty(),
            "nothing to replace");
}

/// Unless it will not fit. Found 2026-09-11: a 25.6 GiB MLX build of a 27B on
/// a 24 GiB machine got no alternatives at all, because the guard asked
/// whether the format was servable and stopped there. A format this machine
/// serves and a size this machine can hold are different facts, and it is the
/// second one that sends a reader looking for a smaller build.
#[test]
fn a_servable_source_too_big_for_the_machine_still_wants_a_smaller_build() {
    let source = repo("x/M-GGUF", "org/M", "q8.gguf", 27_000_000_000, None);
    let smaller = repo("y/M-GGUF", "org/M", "q4.gguf", 9_000_000_000, None);
    let found = siblings::for_repo(&source, &[smaller], &[Format::Gguf], true);
    assert_eq!(found.len(), 1, "the whole answer to a model that is too big");
    assert_eq!(found[0].repo, "y/M-GGUF");
}

/// A build is a set, not a file. `edgefloor/Qwen-Qwen3.8-27B-MTPLX` publishes
/// its model in three shards beside an 810 MB `mtp.safetensors`, and this
/// module — which reimplemented "smallest servable file" rather than sharing
/// the fold — offered 810 MB as a replacement for a 25 GB model. Found
/// 2026-09-11.
#[test]
fn a_sharded_sibling_is_priced_as_the_whole_set_not_its_smallest_part() {
    let source = repo("x/M-FP8", "org/M", "model.safetensors", 1, Some("fp8"));
    let sharded = hf::repo_from_json(&json(
        r#"{"id":"a/M-MLX","library_name":"mlx","cardData":{"base_model":"org/M"},
            "siblings":[
              {"rfilename":"model-00001-of-00003.safetensors","size":9000000000},
              {"rfilename":"model-00002-of-00003.safetensors","size":9000000000},
              {"rfilename":"model-00003-of-00003.safetensors","size":7000000000},
              {"rfilename":"mtp.safetensors","size":810000000}]}"#,
    ))
    .unwrap();
    let found = siblings::from_candidates(&source, &[sharded], &[Format::Mlx]);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].bytes, 25_000_000_000, "three shards, not one, and not the head");
    assert!(found[0].file.contains("3 parts"), "and it says so: {}", found[0].file);
}

#[test]
fn a_sibling_carries_what_it_is_so_a_row_can_say_so() {
    let source = repo("x/M-FP8", "org/M", "model.safetensors", 1, Some("fp8"));
    let mlx = repo("a/M-MLX", "org/M", "model.npz", 9_000_000_000, None);
    let found: Vec<Sibling> = siblings::from_candidates(&source, &[mlx], &[Format::Mlx]);
    assert_eq!(found[0].format, Format::Mlx);
    assert_eq!(found[0].bytes, 9_000_000_000);
    assert_eq!(found[0].file, "model.npz");
    let _ = (ProviderKind::Vllm, Quant::Fp8);
}
