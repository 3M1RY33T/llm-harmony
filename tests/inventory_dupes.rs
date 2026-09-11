mod support;

use std::path::PathBuf;

use llm_harmony::inventory::artifact::Store;
use llm_harmony::inventory::dupes::{duplicates_above, reclaimable_bytes};
use llm_harmony::inventory::Inventory;

use support::tree::Tree;

fn scan(roots: Vec<(Store, PathBuf)>) -> Inventory {
    Inventory::scan_offline(Some(roots))
}

fn write(t: &Tree, rel: &str, byte: u8, len: usize) -> PathBuf {
    let p = t.root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, vec![byte; len]).unwrap();
    p
}

/// The case this exists for: one repo pulled into two stores separately, so
/// two allocations hold the same bytes and the disk pays twice.
#[test]
fn the_same_bytes_in_two_stores_are_reported_as_one_group() {
    let t = Tree::new("dupes-two-stores");
    write(&t, "pool/m-Q4_K_M.gguf", 7, 4096);
    write(&t, "lms/pub/repo/m.q4_k_m.gguf", 7, 4096);

    let inv = scan(vec![
        (Store::LlamaCpp, t.root.join("pool")),
        (Store::LmStudio, t.root.join("lms")),
    ]);
    let groups = duplicates_above(&inv, 1024);

    assert_eq!(groups.len(), 1, "{groups:?}");
    assert_eq!(groups[0].members.len(), 2);
    assert_eq!(groups[0].reclaimable_bytes, 4096, "one of the two copies");
    assert_eq!(reclaimable_bytes(&groups), 4096);
}

/// The finding that makes sampling necessary, measured 2026-09-11: four
/// same-size pairs on this machine were conversions of sibling models --
/// identical architecture and quantisation, different weights. Equal size is
/// a candidate, never an answer.
#[test]
fn two_same_size_files_with_different_content_are_not_duplicates() {
    let t = Tree::new("dupes-same-size");
    write(&t, "pool/a-Q6_K.gguf", 1, 4096);
    write(&t, "pool/b-Q6_K.gguf", 2, 4096);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    assert!(duplicates_above(&inv, 1024).is_empty());
}

/// A symlink is the same allocation reached twice. The ledger already counts
/// it once and `rm` already prices it at zero; reporting it here would be the
/// same bytes counted as waste.
#[test]
fn a_symlink_and_its_target_are_not_a_duplicate_pair() {
    let t = Tree::new("dupes-alias");
    let target = write(&t, "pool/m-Q4_K_M.gguf", 7, 4096);
    t.link("pool/alias.gguf", &target);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    assert!(duplicates_above(&inv, 1024).is_empty());
}

/// Every HF repo carries an identical `tokenizer_config.json`. Without a floor
/// the report is thousands of 4 KB twins and none of the 9 GB ones.
#[test]
fn copies_below_the_floor_are_not_compared() {
    let t = Tree::new("dupes-floor");
    write(&t, "pool/a-Q4_K_M.gguf", 7, 512);
    write(&t, "pool/b-Q4_K_M.gguf", 7, 512);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    assert!(duplicates_above(&inv, 1024).is_empty());
    assert_eq!(duplicates_above(&inv, 256).len(), 1, "found once the floor allows it");
}

#[test]
fn three_copies_reclaim_two_of_them() {
    let t = Tree::new("dupes-three");
    write(&t, "pool/a-Q4_K_M.gguf", 7, 4096);
    write(&t, "pool/b-Q4_K_M.gguf", 7, 4096);
    write(&t, "pool/c-Q4_K_M.gguf", 7, 4096);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    let groups = duplicates_above(&inv, 1024);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].members.len(), 3);
    assert_eq!(groups[0].reclaimable_bytes, 8192, "keep one, link two");
}

/// A match on a sampled comparison must not be reported as a full one.
#[test]
fn the_basis_of_a_match_is_stated_and_is_not_a_claim_of_identity() {
    let t = Tree::new("dupes-basis");
    write(&t, "pool/a-Q4_K_M.gguf", 7, 4096);
    write(&t, "pool/b-Q4_K_M.gguf", 7, 4096);

    let inv = scan(vec![(Store::LlamaCpp, t.root.join("pool"))]);
    let groups = duplicates_above(&inv, 1024);
    assert!(!groups[0].basis.contains("identical"), "{}", groups[0].basis);
    assert!(groups[0].basis.contains("equal size"), "{}", groups[0].basis);
}

/// The gap a `Format` filter opened, found 2026-09-11: an HF blob is named by
/// its sha256 and has no extension, so filtering on format hid the largest
/// store on the disk. Two repos holding one model's weights is exactly the
/// duplication worth finding.
#[test]
fn hf_blobs_are_compared_even_though_their_names_carry_no_format() {
    let t = Tree::new("dupes-hf-blobs");
    write(&t, "hub/models--org--a/blobs/aaaaaaaa", 7, 4096);
    write(&t, "hub/models--org--b/blobs/bbbbbbbb", 7, 4096);

    let inv = scan(vec![(Store::HfCache, t.root.join("hub"))]);
    let groups = duplicates_above(&inv, 1024);

    assert_eq!(groups.len(), 1, "{groups:?}");
    assert_eq!(groups[0].reclaimable_bytes, 4096);
}
