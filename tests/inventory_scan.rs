mod support;

use llm_harmony::inventory::artifact::{Artifact, Store};
use llm_harmony::inventory::scan::{is_excluded, total_unique_bytes};

use support::tree::Tree;

fn artifacts(paths: &[std::path::PathBuf], store: Store) -> Vec<Artifact> {
    paths.iter().filter_map(|p| Artifact::from_path(p, store)).collect()
}

/// The rule the disk ledger rests on. A symlink and its target are one
/// allocation, so `du`-style summing overstates by the size of every link.
#[test]
fn a_symlink_and_its_target_count_once() {
    let t = Tree::new("dedupe");
    let real = t.file("store-a/model.gguf", 4096);
    let link = t.link("store-b/model.gguf", &real);

    let all = artifacts(&[real, link], Store::LlamaCpp);
    assert_eq!(all.len(), 2, "both paths are real artifacts");
    assert_eq!(total_unique_bytes(&all), 4096, "but they are one allocation, not two");
}

#[test]
fn a_symlink_is_marked_so_removing_it_is_known_to_reclaim_nothing() {
    let t = Tree::new("islink");
    let real = t.file("a/model.gguf", 128);
    let link = t.link("b/model.gguf", &real);

    let all = artifacts(&[real, link], Store::LlamaCpp);
    assert!(!all[0].is_link);
    assert!(all[1].is_link);
    assert_eq!(all[0].key, all[1].key, "same allocation");
}

/// Verified 2026-09-09: ~/.vllm-mlx/venv holds 47,588 files.
#[test]
fn python_virtualenvs_and_caches_are_excluded() {
    assert!(is_excluded(std::path::Path::new("/x/.vllm-mlx/venv/lib/python3.12/foo.py")));
    assert!(is_excluded(std::path::Path::new("/x/venv")));
    assert!(is_excluded(std::path::Path::new("/x/__pycache__/y.pyc")));
    assert!(is_excluded(std::path::Path::new("/x/.git/config")));
    assert!(!is_excluded(std::path::Path::new("/x/models/model.gguf")));
}

#[test]
fn a_store_that_does_not_exist_yields_nothing_and_does_not_panic() {
    use llm_harmony::inventory::scan::scan_dir;
    let out = scan_dir(std::path::Path::new("/nonexistent/store"), Store::LmStudio);
    assert!(out.is_empty());
}

use llm_harmony::inventory::scan::hf::{repo_id_from_dir, HfCache};
use llm_harmony::inventory::scan::StoreScanner;

#[test]
fn hf_directory_names_decode_to_repo_ids() {
    assert_eq!(repo_id_from_dir("models--BAAI--bge-small-en-v1.5").as_deref(), Some("BAAI/bge-small-en-v1.5"));
    assert_eq!(repo_id_from_dir("models--mlx-community--Llama-3.2-1B-Instruct-4bit").as_deref(), Some("mlx-community/Llama-3.2-1B-Instruct-4bit"));
    assert_eq!(repo_id_from_dir("CACHEDIR.TAG"), None);
    assert_eq!(repo_id_from_dir("datasets--foo--bar"), None);
}

/// Real HF layout: snapshots/<sha>/x -> ../../blobs/<hash>.
#[test]
fn hf_snapshot_symlinks_do_not_double_count_the_blobs() {
    let t = Tree::new("hf");
    let blob = t.file("models--BAAI--bge/blobs/3c9f31665447", 8192);
    t.link("models--BAAI--bge/snapshots/5c38ec/model.safetensors", &blob);
    t.write("models--BAAI--bge/refs/main", "5c38ec");

    let found = HfCache.scan(&t.root);
    assert_eq!(total_unique_bytes(&found), 8192, "blob and its snapshot link are one allocation");
    assert!(found.len() >= 2, "both paths are still listed: {found:?}");
}

#[test]
fn hf_artifacts_carry_their_repo_id_as_the_name_hint() {
    let t = Tree::new("hf-hint");
    let blob = t.file("models--mlx-community--Llama-3.2-1B-Instruct-4bit/blobs/aaaa", 64);
    t.link("models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/s1/model.safetensors", &blob);

    let found = HfCache.scan(&t.root);
    assert!(found.iter().any(|a| a.name_hint.contains("Llama-3.2-1B-Instruct-4bit")),
        "repo id must reach the artifact for identity grouping: {found:?}");
}

use llm_harmony::inventory::scan::llamacpp::LlamaCppPool;
use llm_harmony::inventory::scan::lmstudio::LmStudioStore;

#[test]
fn lmstudio_hints_with_publisher_and_repo() {
    let t = Tree::new("lms");
    t.file("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF/model.q4_k_m.gguf", 512);
    t.write("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF/config.json", "{}");

    let found = LmStudioStore.scan(&t.root);
    // Every file in a repo shares the repo's name hint, so select by format.
    let gguf = found
        .iter()
        .find(|a| a.format == llm_harmony::inventory::artifact::Format::Gguf)
        .expect("the gguf");
    assert!(gguf.name_hint.contains("TeichAI"), "publisher must survive: {}", gguf.name_hint);
    assert_eq!(gguf.bits, Some(4));
}

/// The live case on this machine, 2026-09-09.
#[test]
fn llamacpp_pool_sees_symlinks_into_other_stores_as_zero_reclaim() {
    let t = Tree::new("pool");
    let target = t.file("lmstudio/TeichAI/repo/model.q4_k_m.gguf", 4096);
    t.link("pool/Qwen3-14B-Claude-4.5-Opus-Distill-Q4_K_M.gguf", &target);
    t.file("pool/Qwen3-14B-Claude-4.5-Opus-Distill-Q8_0.gguf", 8192);

    let found = LlamaCppPool.scan(&t.root.join("pool"));
    assert_eq!(found.len(), 2);
    let link = found.iter().find(|a| a.is_link).expect("the symlink");
    let real = found.iter().find(|a| !a.is_link).expect("the real file");
    assert_eq!(link.bits, Some(4));
    assert_eq!(real.bits, Some(8));
    assert_eq!(total_unique_bytes(&found), 4096 + 8192, "distinct allocations");
}

#[test]
fn llamacpp_pool_tolerates_a_broken_symlink() {
    let t = Tree::new("broken");
    t.link("pool/gone.gguf", std::path::Path::new("/nonexistent/model.gguf"));
    let found = LlamaCppPool.scan(&t.root.join("pool"));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].key, None, "unstattable");
    assert_eq!(total_unique_bytes(&found), 0, "a dangling link occupies nothing");
}

use llm_harmony::inventory::scan::vllm::{parse_registry, VllmStore};

#[test]
fn vllm_registry_names_are_parsed() {
    let yaml = r#"
manager:
  memory_budget_gb: 14
models:
  - name: Qwen3.5-9B-ultra-uncensored-heretic-MLX
    path: /Users/x/.vllm-mlx/models/Qwen3.5-9B-ultra-uncensored-heretic-MLX
  - name: Qwen3.5-9B-Defiant-Heretic-NEO-IMATRIX-MAX-MTP-MLX-6bit
    path: /Users/x/.vllm-mlx/models/Qwen3.5-9B-Defiant-Heretic-NEO-IMATRIX-MAX-MTP-MLX-6bit
"#;
    let entries = parse_registry(yaml);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].name, "Qwen3.5-9B-ultra-uncensored-heretic-MLX");
    assert!(entries[1].path.ends_with("MLX-6bit"));
}

#[test]
fn vllm_registry_survives_a_file_with_no_models_key() {
    assert!(parse_registry("manager:\n  memory_budget_gb: 14\n").is_empty());
    assert!(parse_registry("not: valid: yaml: [").is_empty());
}

/// Verified 2026-09-09: ~/.vllm-mlx/venv holds 47,588 files.
#[test]
fn vllm_scan_skips_the_virtualenv_entirely() {
    let t = Tree::new("vllm");
    t.file("models/Qwen3.5-9B-MLX-4bit/weights.npz", 2048);
    t.file("venv/lib/python3.12/site-packages/torch/_C.so", 999_999);
    t.write("models.yaml", "models:\n  - name: Qwen3.5-9B-MLX-4bit\n    path: /x\n");

    let found = VllmStore.scan(&t.root);
    assert!(found.iter().all(|a| !a.path.to_string_lossy().contains("venv")), "venv leaked: {found:?}");
    assert_eq!(total_unique_bytes(&found), 2048, "only the model counts");
}

use llm_harmony::inventory::scan::ollama::{parse_manifest, OllamaStore};

/// Manifest shape captured live 2026-09-09.
#[test]
fn ollama_manifest_yields_the_model_layer_only() {
    let json = r#"{
      "schemaVersion": 2,
      "layers": [
        {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:970aa74c","size":274290656},
        {"mediaType":"application/vnd.ollama.image.license","digest":"sha256:c71d239d","size":11357},
        {"mediaType":"application/vnd.ollama.image.params","digest":"sha256:ce4a164f","size":17}
      ]
    }"#;
    let m = parse_manifest(json).expect("parsed");
    assert_eq!(m.model_digest, "sha256:970aa74c", "the model layer, not the license");
    assert_eq!(m.bytes, 274_290_656);
}

#[test]
fn ollama_manifest_without_a_model_layer_is_none() {
    assert!(parse_manifest(r#"{"layers":[{"mediaType":"x","digest":"d","size":1}]}"#).is_none());
    assert!(parse_manifest("not json").is_none());
}

#[test]
fn ollama_two_tags_sharing_a_blob_count_once() {
    let t = Tree::new("ollama");
    t.file("blobs/sha256-970aa74c", 4096);
    let manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:970aa74c","size":4096}]}"#;
    t.write("manifests/registry.ollama.ai/library/nomic-embed-text/latest", manifest);
    t.write("manifests/registry.ollama.ai/library/nomic-embed-text/v1.5", manifest);

    let found = OllamaStore.scan(&t.root);
    assert_eq!(found.len(), 2, "both tags are listed");
    assert_eq!(total_unique_bytes(&found), 4096, "but one allocation");
    assert!(found.iter().any(|a| a.name_hint.contains("nomic-embed-text")));
}

/// The HF cache root also holds `.locks/` and `*.lock` files, which are
/// coordination artifacts rather than model artifacts. Found live 2026-09-09
/// when they leaked into `ls` as dozens of zero-byte "models".
#[test]
fn hf_lock_files_and_non_model_dirs_are_not_artifacts() {
    let t = Tree::new("hf-locks");
    let blob = t.file("models--BAAI--bge/blobs/abc", 1024);
    t.link("models--BAAI--bge/snapshots/s1/model.safetensors", &blob);
    t.file(".locks/models--BAAI--bge/abc.lock", 0);
    t.file("models--BAAI--bge/blobs/abc.lock", 0);
    t.file("CACHEDIR.TAG", 5);

    let found = HfCache.scan(&t.root);
    assert!(
        found.iter().all(|a| !a.path.to_string_lossy().contains(".lock")),
        "lock files leaked: {:?}",
        found.iter().map(|a| &a.name_hint).collect::<Vec<_>>()
    );
    assert!(found.iter().all(|a| a.name_hint != "CACHEDIR.TAG"));
    assert_eq!(total_unique_bytes(&found), 1024);
}
