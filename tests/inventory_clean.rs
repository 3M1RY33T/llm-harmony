mod support;

use std::path::PathBuf;

use llm_harmony::inventory::artifact::Store;
use llm_harmony::inventory::clean;
use llm_harmony::inventory::Inventory;

use support::tree::Tree;

fn scan(roots: Vec<(Store, PathBuf)>) -> Inventory {
    Inventory::scan_offline(Some(roots))
}

/// The whole feature, end to end on a real tree: a Q4 goes, its pool symlink
/// goes with it and frees nothing, and the Q6 stays.
#[test]
fn a_superseded_build_and_its_symlink_are_planned_and_the_better_build_stays() {
    let t = Tree::new("clean-pool");
    let q4 = t.file("pool/m-Q4_K_M.gguf", 9_000);
    t.file("pool/m-Q6_K.gguf", 12_000);
    // The real tangle, verified 2026-09-09: a pool entry whose name carries no
    // bit depth, pointing at a build in another store.
    t.link("pool/served-model.gguf", &q4);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    let plan = clean::redundant(&inv);

    assert_eq!(plan.models.len(), 1, "{:?}", plan.models);
    let steps = &plan.models[0].plan.steps;
    assert_eq!(steps.len(), 2, "the Q4 and its alias: {steps:?}");
    assert!(steps[0].is_link, "aliases first: {steps:?}");
    assert_eq!(steps[0].frees_bytes, 0);
    assert_eq!(plan.reclaims_bytes, 9_000, "one allocation, not two");
    assert!(
        !steps.iter().any(|s| s.path.ends_with("m-Q6_K.gguf")),
        "the surviving build must not be planned: {steps:?}"
    );
}

/// Offline, every HF-cache entry is assumed to be scanned by the llama.cpp
/// router, so a clean over the cache plans nothing and says why. This is the
/// behaviour `--live` exists to narrow.
#[test]
fn hf_cache_builds_are_skipped_while_the_router_may_be_scanning_them() {
    let t = Tree::new("clean-hf");
    t.file("hub/models--org--m/snapshots/abc/m-Q4_K_M.gguf", 9_000);
    t.file("hub/models--org--m/snapshots/abc/m-Q6_K.gguf", 12_000);

    let inv = scan(vec![(Store::HfCache, t.root.join("hub"))]);
    let plan = clean::redundant(&inv);

    assert!(plan.models.is_empty(), "{:?}", plan.models);
    assert_eq!(plan.skipped.len(), 1);
    assert_eq!(plan.reclaims_bytes, 0);
    let reasons: Vec<&str> = plan.skipped[0]
        .plan
        .refusals
        .iter()
        .map(|r| r.reason.as_str())
        .collect();
    assert!(
        reasons.iter().any(|r| r.contains("scans at runtime")),
        "{reasons:?}"
    );
}

/// A clean over a store where nothing is superseded must plan nothing rather
/// than fall back to a weaker safety class.
#[test]
fn a_single_build_per_model_plans_nothing() {
    let t = Tree::new("clean-none");
    t.file("pool/a-Q4_K_M.gguf", 9_000);
    t.file("pool/b-Q6_K.gguf", 12_000);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    let plan = clean::redundant(&inv);

    assert!(plan.models.is_empty());
    assert!(plan.skipped.is_empty());
    assert_eq!(plan.reclaims_bytes, 0);
    assert_eq!(
        llm_harmony::render_ls::render_clean(&plan),
        "nothing is redundant: every build is the best of its format\n"
    );
}
