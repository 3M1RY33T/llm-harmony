//! `clean --redundant` -- `rm` asked of every model at once.
//!
//! `rm` answers *may I remove this model?*. `clean` asks the same question of
//! the whole inventory and keeps only the builds nothing supersedes. The
//! selection is `safety::classify`'s `Redundant` and nothing else, so the two
//! rules that make it safe are the ones already encoded there: quantisation is
//! one-way, and there is no MLX <-> GGUF path. An artifact is selected only
//! when a strictly better build of the *same format* survives it.
//!
//! The unit of refusal is one model, not the run. `plan::build_plan` poisons a
//! whole plan when any artifact in it is required -- correct for `rm`, whose
//! plan is one model. Here a served model must not block the other forty, so
//! each model gets its own plan and a poisoned one is reported as skipped.

use crate::inventory::artifact::{Artifact, ArtifactId};
use crate::inventory::plan::{build_plan, RemovalPlan};
use crate::inventory::safety::{classify, Safety};
use crate::inventory::Inventory;

/// One model's share of a clean.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelClean {
    pub model: String,
    pub plan: RemovalPlan,
    /// `removed -> the build that supersedes it`, so the report can say what
    /// survives rather than only what goes.
    pub superseded: Vec<(ArtifactId, ArtifactId)>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CleanPlan {
    /// Models with an executable plan, largest reclaim first: the reason
    /// anyone runs this is to find space.
    pub models: Vec<ModelClean>,
    /// Models a requirer took off the table. Not an error, and not empty in
    /// normal use: offline, every HF-cache entry is assumed to be scanned by
    /// the llama.cpp router, which is what `--live` exists to check.
    pub skipped: Vec<ModelClean>,
    pub reclaims_bytes: u64,
}

impl CleanPlan {
    pub fn artifact_count(&self) -> usize {
        self.models.iter().map(|m| m.plan.steps.len()).sum()
    }
}

/// Plan the removal of every superseded build in the inventory.
pub fn redundant(inv: &Inventory) -> CleanPlan {
    let mut models = Vec::new();
    let mut skipped = Vec::new();

    for ident in &inv.identities {
        // Peers are the identity group, matching `ls`: a build is redundant
        // relative to the other builds of its own model and nothing wider.
        let peers: Vec<Artifact> = inv
            .artifacts
            .iter()
            .filter(|a| ident.artifacts.contains(&a.id))
            .cloned()
            .collect();

        let mut superseded: Vec<(ArtifactId, ArtifactId)> = Vec::new();
        let mut going: Vec<Artifact> = Vec::new();
        for a in &peers {
            if let Safety::Redundant { better } = classify(a, &peers) {
                superseded.push((a.id.clone(), better));
                going.push(a.clone());
            }
        }
        if going.is_empty() {
            continue;
        }

        // Whatever else reaches those bytes goes with them. A symlink left
        // behind is a dangling entry in another provider's pool -- the tangle
        // docs/inventory.md section 2 describes, deepened rather than cleaned
        // up. Pulling it in explicitly is necessary because an alias need not
        // classify as redundant itself: `~/.llamacpp/models/...-Q4_K_M.gguf`
        // carries a bit depth in its name, and the pool filenames next to it
        // do not. `build_plan` then orders the links first.
        let mut also: Vec<Artifact> = Vec::new();
        for a in &peers {
            if going.iter().any(|g| g.id == a.id) {
                continue;
            }
            if going
                .iter()
                .any(|g| inv.graph.aliases_of(&g.id).contains(&a.id))
            {
                also.push(a.clone());
            }
        }
        going.extend(also);

        let entry = ModelClean {
            model: ident.canonical.clone(),
            plan: build_plan(going, &inv.graph),
            superseded,
        };
        if entry.plan.refusals.is_empty() {
            models.push(entry);
        } else {
            skipped.push(entry);
        }
    }

    models.sort_by_key(|m| std::cmp::Reverse(m.plan.reclaims_bytes));
    skipped.sort_by(|a, b| a.model.cmp(&b.model));

    // Summing across models is safe: `group_with_aliases` puts every path
    // that reaches one allocation into a single identity, so no two models can
    // claim the same bytes, and `build_plan` already dedupes within one.
    let reclaims_bytes = models.iter().map(|m| m.plan.reclaims_bytes).sum();

    CleanPlan {
        models,
        skipped,
        reclaims_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::{FileKey, Format, Store};
    use crate::inventory::graph::{Graph, Requirer};
    use crate::inventory::identity::group_with_aliases;
    use crate::provider::ProviderKind;

    struct Build {
        name: &'static str,
        bits: Option<u8>,
        bytes: u64,
        ino: u64,
        is_link: bool,
        store: Store,
        format: Format,
    }

    fn gguf(name: &'static str, bits: u8, bytes: u64, ino: u64) -> Build {
        Build {
            name,
            bits: Some(bits),
            bytes,
            ino,
            is_link: false,
            store: Store::LlamaCpp,
            format: Format::Gguf,
        }
    }

    fn artifact(b: &Build) -> Artifact {
        Artifact {
            id: format!("{}:/x/{}", b.store.as_str(), b.name),
            path: format!("/x/{}", b.name).into(),
            store: b.store,
            format: b.format,
            bits: b.bits,
            bytes: b.bytes,
            key: Some(FileKey { dev: 1, ino: b.ino }),
            is_link: b.is_link,
            name_hint: b.name.to_string(),
        }
    }

    fn inventory(builds: &[Build]) -> Inventory {
        let artifacts: Vec<Artifact> = builds.iter().map(artifact).collect();
        let graph = Graph::build(&artifacts);
        let identities = group_with_aliases(&artifacts, &graph);
        Inventory {
            artifacts,
            graph,
            identities,
        }
    }

    fn planned_paths(plan: &CleanPlan) -> Vec<String> {
        plan.models
            .iter()
            .flat_map(|m| m.plan.steps.iter().map(|s| s.path.clone()))
            .collect()
    }

    #[test]
    fn a_superseded_build_is_selected_and_the_better_one_is_left_alone() {
        let plan = redundant(&inventory(&[
            gguf("m-Q4_K_M.gguf", 4, 9_000, 1),
            gguf("m-Q6_K.gguf", 6, 12_000, 2),
        ]));
        assert_eq!(planned_paths(&plan), vec!["/x/m-Q4_K_M.gguf"]);
        assert_eq!(plan.reclaims_bytes, 9_000);
    }

    /// The survivor guarantee: every step must leave a better build of the
    /// same format behind, so the highest-bit build is never in a plan.
    #[test]
    fn the_best_build_of_a_format_is_never_planned() {
        let plan = redundant(&inventory(&[
            gguf("m-Q4_K_M.gguf", 4, 9_000, 1),
            gguf("m-Q6_K.gguf", 6, 12_000, 2),
            gguf("m-Q8_0.gguf", 8, 15_000, 3),
        ]));
        let paths = planned_paths(&plan);
        assert!(paths.contains(&"/x/m-Q4_K_M.gguf".to_string()), "{paths:?}");
        assert!(paths.contains(&"/x/m-Q6_K.gguf".to_string()), "{paths:?}");
        assert!(!paths.contains(&"/x/m-Q8_0.gguf".to_string()), "{paths:?}");
        assert_eq!(plan.reclaims_bytes, 21_000);
    }

    /// docs/inventory.md section 6: there is no MLX <-> GGUF path, so a 6-bit
    /// MLX build cannot make a 4-bit GGUF redundant. This is the case that
    /// deleted a capability by hand -- the MTP tensors only the GGUF kept.
    #[test]
    fn a_build_in_another_format_is_never_selected() {
        let mut mlx = gguf("m-MLX-6bit", 6, 12_000, 2);
        mlx.format = Format::Mlx;
        mlx.store = Store::Vllm;
        let plan = redundant(&inventory(&[gguf("m-Q4_K_M.gguf", 4, 9_000, 1), mlx]));
        assert!(plan.models.is_empty(), "{:?}", planned_paths(&plan));
        assert_eq!(plan.reclaims_bytes, 0);
    }

    /// A symlink into another store carries no bit depth in its name, so it
    /// does not classify as redundant on its own. It still has to go, or the
    /// pool is left holding a dangling link.
    #[test]
    fn an_alias_of_a_selected_build_is_removed_with_it_and_frees_nothing() {
        let mut alias = gguf("pool-entry.gguf", 4, 9_000, 1);
        alias.bits = None;
        alias.is_link = true;
        alias.store = Store::LmStudio;
        let plan = redundant(&inventory(&[
            gguf("m-Q4_K_M.gguf", 4, 9_000, 1),
            gguf("m-Q6_K.gguf", 6, 12_000, 2),
            alias,
        ]));
        let steps = &plan.models[0].plan.steps;
        assert_eq!(steps.len(), 2, "the alias and its target: {steps:?}");
        assert!(steps[0].is_link, "aliases first: {steps:?}");
        assert_eq!(steps[0].frees_bytes, 0);
        assert_eq!(plan.reclaims_bytes, 9_000, "one allocation, not two");
    }

    /// The difference from `rm`: a refusal poisons one model's plan, not the
    /// run. Forty models must not be held hostage by the one being served.
    #[test]
    fn a_refusal_skips_only_the_model_it_belongs_to() {
        let inv0 = inventory(&[
            gguf("served-Q4_K_M.gguf", 4, 9_000, 1),
            gguf("served-Q6_K.gguf", 6, 12_000, 2),
            gguf("idle-Q4_K_M.gguf", 4, 5_000, 3),
            gguf("idle-Q6_K.gguf", 6, 7_000, 4),
        ]);
        let mut inv = inv0;
        inv.graph.add_required_by(
            "llamacpp:/x/served-Q4_K_M.gguf".to_string(),
            Requirer::ServedLive {
                provider: ProviderKind::LlamaCpp,
                model_id: "served".into(),
            },
        );

        let plan = redundant(&inv);
        assert_eq!(planned_paths(&plan), vec!["/x/idle-Q4_K_M.gguf"]);
        assert_eq!(plan.skipped.len(), 1);
        assert_eq!(plan.skipped[0].model, "served");
        assert_eq!(
            plan.reclaims_bytes, 5_000,
            "a skipped model contributes nothing to the total"
        );
    }

    #[test]
    fn an_inventory_with_one_build_per_model_plans_nothing() {
        let plan = redundant(&inventory(&[
            gguf("a-Q4_K_M.gguf", 4, 9_000, 1),
            gguf("b-Q6_K.gguf", 6, 12_000, 2),
        ]));
        assert!(plan.models.is_empty());
        assert!(plan.skipped.is_empty());
        assert_eq!(plan.reclaims_bytes, 0);
    }
}
