use crate::inventory::artifact::{Artifact, ArtifactId, Provenance};

/// A logical model, and the artifacts that are builds of it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelIdentity {
    pub canonical: String,
    pub artifacts: Vec<ArtifactId>,
    /// Always `Inferred` in slice 2 -- llm-harmony has performed no conversions,
    /// so no lineage is recorded. Placement (slice 4) must refuse to act on
    /// an inferred identity without an explicit override.
    pub provenance: Provenance,
}

/// Strip everything that varies between builds of one model: publisher,
/// extension, format tag, and quantisation.
///
/// This is inference. Names lie -- docs/field-notes.md records two HF repos
/// whose names claimed to be builds of models they were not -- which is why the
/// result is always wrapped in `Provenance::Inferred`.
pub fn canonical_name(raw: &str) -> String {
    let mut s = raw.to_ascii_lowercase();

    // Publisher prefix: "teichai/qwen3-14b-..." -> "qwen3-14b-..."
    if let Some((_, rest)) = s.split_once('/') {
        s = rest.to_string();
    }
    for ext in [".gguf", ".safetensors", ".npz", ".json"] {
        s = s.trim_end_matches(ext).to_string();
    }
    // Longest first, so `-mlx-6bit` is stripped before `-mlx`.
    const DECORATION: [&str; 18] = [
        "-mlx-4bit", "-mlx-6bit", "-mlx-8bit",
        "-q2_k", "-q3_k_m", "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q6_k", "-q8_0",
        ".q4_k_m", ".q8_0",
        "-4bit", "-6bit", "-8bit",
        "-gguf", "-mlx", "-imatrix",
    ];
    let mut changed = true;
    while changed {
        changed = false;
        for d in DECORATION {
            if s.contains(d) {
                s = s.replace(d, "");
                changed = true;
            }
        }
    }
    // Ollama tags: "nomic-embed-text:latest" -> "nomic-embed-text"
    if let Some((base, _tag)) = s.split_once(':') {
        s = base.to_string();
    }
    s.trim_matches(['-', '.', '_']).to_string()
}

pub fn group(artifacts: &[Artifact]) -> Vec<ModelIdentity> {
    use std::collections::BTreeMap;
    let mut by_name: BTreeMap<String, Vec<ArtifactId>> = BTreeMap::new();
    for a in artifacts {
        let canon = canonical_name(&a.name_hint);
        if canon.is_empty() {
            continue;
        }
        by_name.entry(canon).or_default().push(a.id.clone());
    }
    by_name
        .into_iter()
        .map(|(canonical, artifacts)| ModelIdentity {
            canonical,
            artifacts,
            provenance: Provenance::Inferred {
                basis: "normalised filename".to_string(),
            },
        })
        .collect()
}

/// Group by name, then merge any groups joined by an alias edge.
///
/// Two paths that resolve to one allocation are the same model however they
/// are named — which matters because they routinely are not. Verified
/// 2026-09-09: LM Studio's repo is `...Opus-High-Reasoning-Distill-GGUF` and
/// the llama.cpp symlink into it is `...Opus-Distill-Q4_K_M.gguf`. Name
/// normalisation splits them; the shared inode does not.
///
/// This is the one identity signal in slice 2 that is *not* inference, so
/// merged groups keep the strongest provenance of their parts.
pub fn group_with_aliases(
    artifacts: &[Artifact],
    graph: &crate::inventory::graph::Graph,
) -> Vec<ModelIdentity> {
    use std::collections::BTreeMap;

    let mut groups = group(artifacts);

    // Which group holds each artifact.
    let mut owner: BTreeMap<ArtifactId, usize> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        for a in &g.artifacts {
            owner.insert(a.clone(), i);
        }
    }

    // Union groups joined by a shared allocation.
    let mut parent: Vec<usize> = (0..groups.len()).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            let r = find(parent, parent[x]);
            parent[x] = r;
        }
        parent[x]
    }
    for alias_group in graph.alias_groups() {
        let mut it = alias_group.iter().filter_map(|id| owner.get(id).copied());
        if let Some(first) = it.next() {
            for other in it {
                let (a, b) = (find(&mut parent, first), find(&mut parent, other));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }

    let mut merged: BTreeMap<usize, ModelIdentity> = BTreeMap::new();
    for i in 0..groups.len() {
        let root = find(&mut parent, i);
        let g = std::mem::replace(
            &mut groups[i],
            ModelIdentity {
                canonical: String::new(),
                artifacts: Vec::new(),
                provenance: Provenance::Inferred { basis: String::new() },
            },
        );
        match merged.get_mut(&root) {
            Some(existing) => {
                // Keep the longer name: it is the more specific one, and the
                // short form is usually a truncation in a pool filename.
                if g.canonical.len() > existing.canonical.len() {
                    existing.canonical = g.canonical.clone();
                }
                existing.artifacts.extend(g.artifacts);
                existing.provenance = Provenance::Inferred {
                    basis: "normalised filename, unified by shared allocation".to_string(),
                };
            }
            None => {
                merged.insert(root, g);
            }
        }
    }

    merged.into_values().filter(|g| !g.artifacts.is_empty()).collect()
}
