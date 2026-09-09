mod support;

use llm_harmony::inventory::artifact::{Artifact, Provenance, Store};
use llm_harmony::inventory::graph::Graph;
use llm_harmony::inventory::identity::{canonical_name, group};

use support::tree::Tree;

fn scan_two_stores(t: &Tree) -> Vec<Artifact> {
    use llm_harmony::inventory::scan::scan_dir;
    let mut v = scan_dir(&t.root.join("lmstudio"), Store::LmStudio);
    v.extend(scan_dir(&t.root.join("pool"), Store::LlamaCpp));
    v
}

/// The live case: llama.cpp's pool symlinks into LM Studio's store.
#[test]
fn artifacts_sharing_an_allocation_become_an_alias_edge() {
    let t = Tree::new("graph-alias");
    let target = t.file("lmstudio/TeichAI/repo/model.q4_k_m.gguf", 4096);
    t.link("pool/Qwen3-14B-Q4_K_M.gguf", &target);

    let g = Graph::build(&scan_two_stores(&t));
    let aliases = g.alias_groups();
    assert_eq!(aliases.len(), 1, "one shared allocation");
    assert_eq!(aliases[0].len(), 2, "reached from two stores");
}

#[test]
fn distinct_files_are_not_aliased() {
    let t = Tree::new("graph-noalias");
    t.file("lmstudio/a/repo/model.q4_k_m.gguf", 4096);
    t.file("pool/other-Q8_0.gguf", 4096);

    let g = Graph::build(&scan_two_stores(&t));
    assert!(g.alias_groups().is_empty(), "equal size is not equal identity");
}

#[test]
fn canonical_names_collapse_store_specific_decoration() {
    let a = canonical_name("TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF");
    let b = canonical_name("Qwen3-14B-Claude-4.5-Opus-Distill-Q4_K_M.gguf");
    let c = canonical_name("Qwen3-14B-Claude-4.5-Opus-Distill-Q8_0.gguf");
    assert_eq!(b, c, "quantisation is not identity: {b} vs {c}");
    assert!(a.contains("qwen3-14b"), "publisher stripped, model kept: {a}");
}

#[test]
fn mlx_and_gguf_builds_of_one_model_group_together() {
    let names = [
        "Qwen3.5-9B-ultra-uncensored-heretic-Q4_K_M.gguf",
        "Qwen3.5-9B-ultra-uncensored-heretic-Q6_K.gguf",
        "Qwen3.5-9B-ultra-uncensored-heretic-MLX",
        "Qwen3.5-9B-ultra-uncensored-heretic-MLX-6bit",
    ];
    let canon: Vec<String> = names.iter().map(|n| canonical_name(n)).collect();
    assert!(canon.windows(2).all(|w| w[0] == w[1]),
        "four artifacts of one model must share a canonical name: {canon:?}");
}

#[test]
fn every_grouping_is_marked_inferred_in_this_slice() {
    let t = Tree::new("group-prov");
    t.file("lmstudio/a/Qwen3.5-9B-heretic-GGUF/m.q4_k_m.gguf", 16);
    t.file("pool/Qwen3.5-9B-heretic-Q6_K.gguf", 16);

    let identities = group(&scan_two_stores(&t));
    assert!(!identities.is_empty());
    for id in &identities {
        assert!(!id.provenance.is_recorded(),
            "slice 2 performs no conversions, so nothing can be Recorded");
        assert!(matches!(id.provenance, Provenance::Inferred { .. }));
    }
}

/// Found live 2026-09-09: LM Studio serves the repo name
/// `...Opus-High-Reasoning-Distill-GGUF` while the file inside it, and the
/// llama.cpp symlink to it, are named `...Opus-Distill`. Name normalisation
/// cannot connect those, so `rm --live` failed to refuse a served model.
///
/// Sharing an allocation is proof of identity that names cannot override.
#[test]
fn artifacts_that_are_one_allocation_group_together_despite_unlike_names() {
    use llm_harmony::inventory::identity::group_with_aliases;

    let t = Tree::new("alias-identity");
    let target = t.file("lmstudio/TeichAI/Qwen3-14B-Opus-High-Reasoning-Distill-GGUF/m.q4_k_m.gguf", 4096);
    t.link("pool/Qwen3-14B-Opus-Distill-Q4_K_M.gguf", &target);

    let arts = scan_two_stores(&t);
    let g = Graph::build(&arts);

    let by_name = group(&arts);
    assert!(by_name.len() >= 2, "names alone split them: {:?}",
        by_name.iter().map(|i| &i.canonical).collect::<Vec<_>>());

    let merged = group_with_aliases(&arts, &g);
    let holding_both = merged.iter().filter(|i| i.artifacts.len() >= 2).count();
    assert_eq!(holding_both, 1, "the allocation must unify them: {:?}",
        merged.iter().map(|i| (&i.canonical, i.artifacts.len())).collect::<Vec<_>>());
}
