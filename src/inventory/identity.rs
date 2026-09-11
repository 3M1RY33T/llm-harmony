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

/// One token that says how a model was built, never which model it is.
///
/// Every marker here can come off a name without changing what the weights are
/// a build *of*: the container (`gguf`, `mlx`), the quantisation and the method
/// that produced it (`q4_k_m`, `iq2_xxs`, `awq`, `gptq`, `fp8`), the width
/// (`4bit`), and the prediction head (`mtp`). Deliberately **not** anything
/// that distinguishes two fine-tunes -- `inventory.md` section 5's trap is
/// calling two different models one model, and the way in is stripping a word
/// that carried meaning.
fn is_decoration(token: &str) -> bool {
    if matches!(
        token,
        "gguf" | "mlx" | "imatrix" | "awq" | "gptq" | "fp8" | "bnb" | "bitsandbytes"
            | "bf16" | "fp16" | "f16" | "int4" | "int8" | "ud" | "mtp"
    ) {
        return true;
    }
    // `4bit`, `6bit`, `8bit`.
    if let Some(width) = token.strip_suffix("bit") {
        if !width.is_empty() && width.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    // `q8_0`, `q4_k_m`, `q6_k_xl`, `iq2_xxs`: a `q` or `iq`, a digit, then
    // only the alphabet a quantisation name is written in.
    let rest = token.strip_prefix("iq").or_else(|| token.strip_prefix('q'));
    match rest {
        Some(r) => {
            let mut chars = r.chars();
            chars.next().is_some_and(|c| c.is_ascii_digit())
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => false,
    }
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
    // Decorations are stripped a hyphen-separated token at a time, not as
    // substrings. A literal list of substrings lost this race as soon as
    // quantisation names grew suffixes: `-q6_k` matched inside `-q6_k_xl` and
    // left a `_xl` welded to the model name, so
    // `Qwen3.8-27B-UD-Q6_K_XL-AWQ-MTP-mlx` canonicalised to
    // `qwen3.8-27b-ud_xl-awq-mtp` and matched no other build of Qwen3.8-27B.
    // Found 2026-09-11, on the repo that also proved `siblings` needed it.
    s = s
        .split('-')
        .filter(|t| !t.is_empty() && !is_decoration(t))
        .collect::<Vec<_>>()
        .join("-");
    // The dot-joined forms the token split cannot see: `model.Q4_K_M`.
    for d in [".q4_k_m", ".q8_0", ".f16", ".bf16"] {
        s = s.replace(d, "");
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
