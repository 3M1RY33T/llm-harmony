use std::collections::HashSet;

use crate::inventory::artifact::{Artifact, ArtifactId, FileKey};
use crate::inventory::graph::{Graph, Requirer};

#[derive(Debug, Clone, serde::Serialize)]
pub struct RemovalStep {
    pub id: ArtifactId,
    pub path: String,
    pub is_link: bool,
    /// Bytes this step actually frees. Zero for an alias.
    pub frees_bytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Refusal {
    pub id: ArtifactId,
    pub reason: String,
}

/// A plan, never an execution.
///
/// There is deliberately no `apply`, and no `std::fs` removal call anywhere in
/// this module. Slice 2 cannot delete, for the same structural reason slice 1
/// cannot unload.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RemovalPlan {
    pub steps: Vec<RemovalStep>,
    pub reclaims_bytes: u64,
    pub refusals: Vec<Refusal>,
}

fn refusal_for(id: &str, graph: &Graph) -> Option<Refusal> {
    let reqs = graph.requirers_of(id);
    if reqs.is_empty() {
        return None;
    }
    let reason = reqs
        .iter()
        .map(|r| match r {
            Requirer::ServedLive { provider, model_id } => {
                format!("served live by {provider} as {model_id}")
            }
            Requirer::Registry { store, name } => {
                format!("named in the {store} registry as {name}")
            }
            Requirer::Scanned { provider } => {
                format!("inside a directory {provider} scans at runtime")
            }
        })
        .collect::<Vec<_>>()
        .join("; ");
    Some(Refusal {
        id: id.to_string(),
        reason,
    })
}

pub fn build_plan(artifacts: Vec<Artifact>, graph: &Graph) -> RemovalPlan {
    let mut refusals = Vec::new();
    let mut keep = Vec::new();

    for a in artifacts {
        match refusal_for(&a.id, graph) {
            Some(r) => refusals.push(r),
            None => keep.push(a),
        }
    }

    // Any refusal poisons the whole plan: removing half of an aliased pair
    // leaves a dangling link, and removing a source while a build is served
    // is exactly the case this exists to prevent.
    if !refusals.is_empty() {
        return RemovalPlan {
            steps: Vec::new(),
            reclaims_bytes: 0,
            refusals,
        };
    }

    // Aliases before targets, or the store is briefly full of dangling links.
    keep.sort_by_key(|a| !a.is_link);

    let mut counted: HashSet<FileKey> = HashSet::new();
    let mut reclaims = 0u64;
    let steps = keep
        .into_iter()
        .map(|a| {
            let frees = match (a.is_link, a.key) {
                (true, _) => 0,
                (false, Some(k)) => {
                    if counted.insert(k) {
                        a.bytes
                    } else {
                        0
                    }
                }
                (false, None) => 0,
            };
            reclaims += frees;
            RemovalStep {
                id: a.id,
                path: a.path.display().to_string(),
                is_link: a.is_link,
                frees_bytes: frees,
            }
        })
        .collect();

    RemovalPlan {
        steps,
        reclaims_bytes: reclaims,
        refusals: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::{Artifact, FileKey, Format, Store};
    use crate::inventory::graph::{Graph, Requirer};
    use crate::provider::ProviderKind;

    /// `ino` is explicit: a link and its target share one, which is what makes
    /// "an alias frees nothing" testable without touching the filesystem.
    fn art(id: &str, name: &str, bytes: u64, is_link: bool, ino: u64) -> Artifact {
        Artifact {
            id: id.into(),
            path: format!("/x/{name}").into(),
            store: Store::LlamaCpp,
            format: Format::Gguf,
            bits: Some(4),
            bytes,
            key: Some(FileKey { dev: 1, ino }),
            is_link,
            name_hint: name.into(),
        }
    }

    /// docs/inventory.md: "remove aliases before targets, or the store is
    /// briefly full of dangling links."
    #[test]
    fn symlinks_are_ordered_before_the_files_they_point_at() {
        let target = art("t", "model.gguf", 4096, false, 42);
        let link = art("l", "alias.gguf", 4096, true, 42);
        let plan = build_plan(vec![link, target], &Graph::build(&[]));
        assert!(plan.steps[0].is_link, "the alias must be removed first: {:?}", plan.steps);
    }

    #[test]
    fn reclaimed_bytes_do_not_count_aliases() {
        let target = art("t", "model.gguf", 4096, false, 42);
        let link = art("l", "alias.gguf", 4096, true, 42);
        let plan = build_plan(vec![link, target], &Graph::build(&[]));
        assert_eq!(plan.reclaims_bytes, 4096, "one allocation, not two");
    }

    /// Refuses rather than warns, per docs/inventory.md §4.
    #[test]
    fn a_live_served_artifact_produces_a_refusal_and_no_steps() {
        let a = art("t", "model.gguf", 4096, false, 7);
        let mut g = Graph::build(&[a.clone()]);
        g.add_required_by(a.id.clone(), Requirer::ServedLive {
            provider: ProviderKind::LmStudio,
            model_id: "qwen3-14b".into(),
        });
        let plan = build_plan(vec![a], &g);
        assert!(plan.steps.is_empty(), "nothing may be planned while it is served");
        assert_eq!(plan.refusals.len(), 1);
        assert!(plan.refusals[0].reason.contains("served"), "{:?}", plan.refusals);
    }

    #[test]
    fn an_artifact_scanned_by_a_router_is_also_refused() {
        let a = art("t", "blob", 64, false, 8);
        let mut g = Graph::build(&[a.clone()]);
        g.add_required_by(a.id.clone(), Requirer::Scanned { provider: ProviderKind::LlamaCpp });
        let plan = build_plan(vec![a], &g);
        assert!(plan.steps.is_empty());
        assert_eq!(plan.refusals.len(), 1);
    }
}
