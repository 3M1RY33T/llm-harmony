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

/// The substring list lost to a quantisation name with a suffix. `-q6_k`
/// matched inside `-Q6_K_XL` and left `_xl` fused to the model name, so a
/// build of Qwen3.8-27B canonicalised to something no other build of
/// Qwen3.8-27B could equal — and `siblings`, which asks this function whether
/// two repos are one model, found nothing for it. Found 2026-09-11.
#[test]
fn a_quantisation_name_with_a_suffix_comes_off_whole() {
    let decorated = canonical_name("igorvibes/Qwen3.8-27B-UD-Q6_K_XL-AWQ-MTP-mlx");
    assert_eq!(decorated, "qwen3.8-27b", "every marker here is how, not what: {decorated}");
    assert_eq!(canonical_name("Qwen/Qwen3.8-27B"), decorated);
    assert_eq!(canonical_name("unsloth/Gemma-4-12B-IQ2_XXS-GGUF"), "gemma-4-12b");
}

/// And the opposite guard: a word that carries meaning is not a decoration.
/// Stripping one is how two different models become one — inventory.md §5.
#[test]
fn a_word_that_names_the_model_survives() {
    let a = canonical_name("TeichAI/Qwen3-14B-Claude-Distill-Q4_K_M.gguf");
    let b = canonical_name("TeichAI/Qwen3-14B-Hermes-Distill-Q4_K_M.gguf");
    assert_ne!(a, b, "two fine-tunes are two models: {a} vs {b}");
    assert_eq!(a, "qwen3-14b-claude-distill");
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

// --- r27 Task 2: `ls --json` carries what its table computes ----------------
//
// The JSON form emitted artifact PATH STRINGS and nothing else, while the table
// beside it derived builds, unique bytes and a safety class per model. A UI
// reading the document had to re-derive all three, and one of them it cannot:
// a pool symlink and its target are one allocation, so bytes summed from paths
// double-counts every model llama.cpp shares with LM Studio.

use llm_harmony::inventory::Inventory;

/// LM Studio holding two quantisations of one model, with llama.cpp's pool
/// symlinked into the Q4 — this machine's actual layout, in miniature.
fn two_builds_one_aliased(t: &Tree) -> Inventory {
    let q4 = t.file("lmstudio/TeichAI/Qwen3-14B-GGUF/qwen3-14b.q4_k_m.gguf", 8192);
    t.file("lmstudio/TeichAI/Qwen3-14B-GGUF/qwen3-14b.q8_0.gguf", 16384);
    t.link("pool/Qwen3-14B-Q4_K_M.gguf", &q4);
    Inventory::scan_offline(Some(vec![
        (Store::LmStudio, t.root.join("lmstudio")),
        (Store::LlamaCpp, t.root.join("pool")),
    ]))
}

fn model_row<'a>(doc: &'a serde_json::Value, canonical_fragment: &str) -> &'a serde_json::Value {
    doc["models"]
        .as_array()
        .expect("models is an array")
        .iter()
        .find(|m| m["canonical"].as_str().is_some_and(|c| c.contains(canonical_fragment)))
        .unwrap_or_else(|| panic!("no model matching `{canonical_fragment}` in {doc}"))
}

#[test]
fn ls_json_reports_bytes_builds_and_safety_per_model() {
    let t = Tree::new("ls-json-columns");
    let inv = two_builds_one_aliased(&t);
    let doc = llm_harmony::render_ls::document(&inv);

    let row = model_row(&doc, "qwen3-14b");
    assert_eq!(row["builds"], 2, "two quantisations, not every file in the repo: {row}");
    assert_eq!(
        row["bytes"], 8192 + 16384,
        "the alias is one allocation and is counted once: {row}"
    );
    assert!(
        row["safety"].as_str().is_some_and(|s| !s.is_empty()),
        "every model carries the class the table prints: {row}"
    );
    assert!(row["provenance"].is_string() || row["provenance"].is_object(), "{row}");
}

#[test]
fn ls_json_reports_each_artifact_with_its_format_store_and_bits() {
    let t = Tree::new("ls-json-artifacts");
    let inv = two_builds_one_aliased(&t);
    let doc = llm_harmony::render_ls::document(&inv);

    let arts = model_row(&doc, "qwen3-14b")["artifacts"].as_array().expect("artifacts array");
    let q8 = arts
        .iter()
        .find(|a| a["path"].as_str().is_some_and(|p| p.contains("q8_0")))
        .expect("the Q8 build is listed");

    // Format and store are what decide which providers could serve a model,
    // so they are what a hosting screen groups and filters by.
    assert_eq!(q8["format"], "gguf");
    assert_eq!(q8["store"], "lmstudio");
    assert_eq!(q8["bits"], 8);
    assert_eq!(q8["bytes"], 16384);
    assert!(q8["id"].as_str().is_some_and(|s| !s.is_empty()));
}

#[test]
fn ls_json_marks_links_so_reclaimable_space_is_not_double_counted() {
    let t = Tree::new("ls-json-links");
    let inv = two_builds_one_aliased(&t);
    let doc = llm_harmony::render_ls::document(&inv);

    let arts = model_row(&doc, "qwen3-14b")["artifacts"].as_array().unwrap();
    let link = arts
        .iter()
        .find(|a| a["store"] == "llamacpp")
        .expect("the pool entry is listed");
    // Removing a link reclaims nothing. Saying so is the difference between a
    // truthful reclaimable figure and one that counts the same bytes twice.
    assert_eq!(link["is_link"], true, "{link}");

    let summed: u64 = arts.iter().filter_map(|a| a["bytes"].as_u64()).sum();
    assert!(
        summed > model_row(&doc, "qwen3-14b")["bytes"].as_u64().unwrap(),
        "summing artifact bytes double-counts, which is exactly why `bytes` is reported"
    );
}
