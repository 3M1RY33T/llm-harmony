use std::collections::HashMap;

use crate::inventory::artifact::{Artifact, ArtifactId, FileKey, Provenance};
use crate::provider::ProviderKind;

/// Why an artifact may not be removed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Requirer {
    /// A provider currently has it resident. From the slice-1 memory ledger.
    ServedLive { provider: ProviderKind, model_id: String },
    /// Named in a registry, where a stale path is a startup failure.
    Registry { store: String, name: String },
    /// Inside a directory a provider scans at runtime -- llama.cpp's router
    /// scans the HF cache, so cache entries are serving dependencies.
    Scanned { provider: ProviderKind },
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "edge", rename_all = "kebab-case")]
pub enum Edge {
    /// Same bytes reached by two paths. Removing one reclaims nothing.
    Aliases { a: ArtifactId, b: ArtifactId },
    DerivesFrom {
        child: ArtifactId,
        parent: ArtifactId,
        provenance: Provenance,
    },
    RequiredBy {
        artifact: ArtifactId,
        requirer: Requirer,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Graph {
    pub edges: Vec<Edge>,
    /// Skipped: serde_json cannot use a struct as a map key, and this is an
    /// index rather than part of the graph's public shape.
    #[serde(skip)]
    by_key: HashMap<FileKey, Vec<ArtifactId>>,
}

impl Graph {
    pub fn build(artifacts: &[Artifact]) -> Graph {
        let mut by_key: HashMap<FileKey, Vec<ArtifactId>> = HashMap::new();
        for a in artifacts {
            if let Some(k) = a.key {
                by_key.entry(k).or_default().push(a.id.clone());
            }
        }

        let mut edges = Vec::new();
        for ids in by_key.values() {
            for pair in ids.windows(2) {
                edges.push(Edge::Aliases {
                    a: pair[0].clone(),
                    b: pair[1].clone(),
                });
            }
        }

        Graph { edges, by_key }
    }

    /// Groups of artifact ids that are the same allocation.
    pub fn alias_groups(&self) -> Vec<Vec<ArtifactId>> {
        self.by_key
            .values()
            .filter(|ids| ids.len() > 1)
            .cloned()
            .collect()
    }

    pub fn aliases_of(&self, id: &str) -> Vec<ArtifactId> {
        self.alias_groups()
            .into_iter()
            .find(|g| g.iter().any(|x| x == id))
            .map(|g| g.into_iter().filter(|x| x != id).collect())
            .unwrap_or_default()
    }

    pub fn add_required_by(&mut self, artifact: ArtifactId, requirer: Requirer) {
        self.edges.push(Edge::RequiredBy { artifact, requirer });
    }

    pub fn requirers_of(&self, id: &str) -> Vec<&Requirer> {
        self.edges
            .iter()
            .filter_map(|e| match e {
                Edge::RequiredBy { artifact, requirer } if artifact == id => Some(requirer),
                _ => None,
            })
            .collect()
    }
}
