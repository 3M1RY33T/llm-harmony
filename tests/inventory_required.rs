mod support;

use llm_harmony::inventory::artifact::{Artifact, Store};
use llm_harmony::inventory::graph::{Graph, Requirer};
use llm_harmony::provider::ProviderKind;

use support::tree::Tree;

fn artifact(t: &Tree, rel: &str, bytes: usize, store: Store) -> Artifact {
    let p = t.file(rel, bytes);
    Artifact::from_path(&p, store).unwrap()
}

/// docs/field-notes.md: "llama.cpp's router scans the HF cache, so cache
/// entries are runtime dependencies, not merely downloads."
#[test]
fn hf_cache_entries_are_marked_as_scanned_by_llamacpp() {
    let t = Tree::new("req-scan");
    let a = artifact(&t, "models--x--y/blobs/aaa", 64, Store::HfCache);
    let mut g = Graph::build(&[a.clone()]);
    g.add_required_by(a.id.clone(), Requirer::Scanned { provider: ProviderKind::LlamaCpp });

    let reqs = g.requirers_of(&a.id);
    assert_eq!(reqs.len(), 1);
    assert!(matches!(reqs[0], Requirer::Scanned { provider: ProviderKind::LlamaCpp }));
}

#[test]
fn a_live_resident_model_becomes_a_served_live_requirer() {
    let t = Tree::new("req-live");
    let a = artifact(&t, "TeichAI/repo/model.q4_k_m.gguf", 64, Store::LmStudio);
    let mut g = Graph::build(&[a.clone()]);
    g.add_required_by(a.id.clone(), Requirer::ServedLive {
        provider: ProviderKind::LmStudio,
        model_id: "qwen3-14b-claude-4.5-opus-high-reasoning-distill".into(),
    });
    assert_eq!(g.requirers_of(&a.id).len(), 1);
}

/// docs/inventory.md §2 question 3: a name in models.yaml is a dependency,
/// because a stale path there is a startup failure rather than a warning.
#[test]
fn a_registry_named_artifact_becomes_a_registry_requirer() {
    let t = Tree::new("req-registry");
    let a = artifact(&t, "Qwen3.5-9B-MLX-4bit/weights.npz", 64, Store::Vllm);
    let mut g = Graph::build(&[a.clone()]);
    g.add_required_by(a.id.clone(), Requirer::Registry {
        store: "vllm".into(),
        name: "Qwen3.5-9B-MLX-4bit".into(),
    });
    let reqs = g.requirers_of(&a.id);
    assert_eq!(reqs.len(), 1);
    assert!(matches!(reqs[0], Requirer::Registry { .. }));
}

#[test]
fn an_artifact_with_no_requirers_reports_an_empty_list_not_an_error() {
    let t = Tree::new("req-none");
    let a = artifact(&t, "pool/orphan-Q4_K_M.gguf", 64, Store::LlamaCpp);
    let g = Graph::build(&[a.clone()]);
    assert!(g.requirers_of(&a.id).is_empty());
}
