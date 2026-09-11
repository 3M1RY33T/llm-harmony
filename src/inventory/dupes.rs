//! `ls --duplicates` -- distinct allocations holding the same bytes.
//!
//! The one disk question `FileKey` cannot answer. A symlink and its target are
//! one allocation and the ledger already counts them once; two separate
//! downloads of one GGUF are two allocations, two counts, and twice the disk,
//! and nothing in slices 1-5 reports them.
//!
//! Measured here 2026-09-11: none. 146 GB across five stores and not one
//! byte-identical pair, because the cross-store sharing on this machine is
//! done with symlinks. That is why this reports and never moves: the
//! duplication it looks for is the habit of a *different* machine -- one where
//! the same repo was pulled into LM Studio and llama.cpp separately -- and it
//! should be found rather than assumed.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::inventory::artifact::{Artifact, ArtifactId, FileKey, Store};
use crate::inventory::Inventory;

/// Read from each end of a candidate. Whole files are not hashed: the pairs
/// worth finding are 5 to 18 GB, and reading 146 GB to answer a report nobody
/// asked to be slow is the wrong trade.
const SAMPLE_BYTES: u64 = 4 * 1024 * 1024;

/// Below this a match is noise rather than news: every HF repo carries an
/// identical `tokenizer_config.json`, and thousands of 4 KB twins bury the
/// 9 GB pair that matters.
///
/// The floor is also the *only* filter on what gets compared, and that is
/// deliberate. Filtering on `Format` first looked right and was wrong: an HF
/// blob is named by its sha256 with no extension at all, so it reads as
/// `Format::Other` and a format filter hides the whole 69 GB cache -- the
/// store most likely to hold two revisions of one repo. Above this floor,
/// anything in a model store is weights.
pub const MIN_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DuplicateMember {
    pub id: ArtifactId,
    pub path: String,
    pub store: Store,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DuplicateGroup {
    /// Size of one copy.
    pub bytes: u64,
    /// What keeping one and linking the rest would free: `(n - 1) * bytes`.
    pub reclaimable_bytes: u64,
    pub members: Vec<DuplicateMember>,
    /// What the match rests on. Never the word "identical" unless the whole
    /// file was read, which for a 9 GB build it was not.
    pub basis: String,
}

/// The first and last `SAMPLE_BYTES`, or the whole file when the two would
/// overlap.
fn sample(path: &Path, bytes: u64) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(path).ok()?;
    if bytes <= SAMPLE_BYTES * 2 {
        let mut all = Vec::new();
        f.read_to_end(&mut all).ok()?;
        return Some(all);
    }
    let mut buf = vec![0u8; SAMPLE_BYTES as usize];
    f.read_exact(&mut buf).ok()?;
    f.seek(SeekFrom::End(-(SAMPLE_BYTES as i64))).ok()?;
    let mut tail = vec![0u8; SAMPLE_BYTES as usize];
    f.read_exact(&mut tail).ok()?;
    buf.extend(tail);
    Some(buf)
}

fn basis_for(bytes: u64) -> String {
    if bytes <= SAMPLE_BYTES * 2 {
        "equal size and full content".to_string()
    } else {
        "equal size and the first and last 4.0M".to_string()
    }
}

pub fn duplicates(inv: &Inventory) -> Vec<DuplicateGroup> {
    duplicates_above(inv, MIN_BYTES)
}

pub fn duplicates_above(inv: &Inventory, min_bytes: u64) -> Vec<DuplicateGroup> {
    // One entry per allocation. An alias is not a duplicate -- it is the same
    // bytes reached twice, which `FileKey` already collapses and `rm` already
    // prices at zero.
    let mut seen: HashSet<FileKey> = HashSet::new();
    let mut candidates: Vec<&Artifact> = Vec::new();
    for a in &inv.artifacts {
        if a.is_link || a.bytes < min_bytes {
            continue;
        }
        let Some(k) = a.key else { continue };
        if seen.insert(k) {
            candidates.push(a);
        }
    }

    let mut by_size: BTreeMap<u64, Vec<&Artifact>> = BTreeMap::new();
    for a in candidates {
        by_size.entry(a.bytes).or_default().push(a);
    }

    let mut out: Vec<DuplicateGroup> = Vec::new();
    for (bytes, arts) in by_size {
        if arts.len() < 2 {
            continue;
        }
        // Equal size is a candidate and not an answer. Four same-size pairs on
        // this machine turned out to be conversions of sibling models: same
        // architecture, same quantisation, same tensor sizes, different
        // weights. Sampling is what separated them.
        let mut buckets: Vec<(Vec<u8>, Vec<&Artifact>)> = Vec::new();
        for a in arts {
            let Some(s) = sample(&a.path, bytes) else { continue };
            match buckets.iter_mut().find(|(b, _)| *b == s) {
                Some((_, group)) => group.push(a),
                None => buckets.push((s, vec![a])),
            }
        }
        for (_, group) in buckets {
            if group.len() < 2 {
                continue;
            }
            out.push(DuplicateGroup {
                bytes,
                reclaimable_bytes: bytes * (group.len() as u64 - 1),
                members: group
                    .iter()
                    .map(|a| DuplicateMember {
                        id: a.id.clone(),
                        path: a.path.display().to_string(),
                        store: a.store,
                    })
                    .collect(),
                basis: basis_for(bytes),
            });
        }
    }

    out.sort_by_key(|g| std::cmp::Reverse(g.reclaimable_bytes));
    out
}

pub fn reclaimable_bytes(groups: &[DuplicateGroup]) -> u64 {
    groups.iter().map(|g| g.reclaimable_bytes).sum()
}
