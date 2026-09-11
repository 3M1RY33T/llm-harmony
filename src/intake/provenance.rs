//! Is this repo really a build of the model it claims to be?
//!
//! `inventory.md` §5 records two repos that were not: one declaring a
//! `base_model` with a near-identical name, one declaring none at all while its
//! own config contradicted its repo name. As a CLI convenience that rule was
//! guidance. Behind a search box it is the last thing between a result and a
//! wrong model on disk.
//!
//! **It warns; it does not refuse.** Decided 2026-09-11 and recorded in
//! `Delroy/docs/hosting-page-r27-plan.md` §6, overriding `inventory.md` §5 and
//! `architecture.md` §7, which both state it as gating. The reasoning was that
//! a refusal is worked around by downloading in a terminal, where there is no
//! check at all, while a warning showing both names side by side informs the
//! person who can actually judge it.
//!
//! That decision puts weight on this module it would not otherwise carry. A
//! warning nobody reads is worse than a gate, so a `Finding` is built to
//! survive: it names both models, and it is carried on the pull's *result*
//! document rather than only through its progress stream, because a line that
//! scrolled past during an 8 GB download was never seen.

use serde::Serialize;

/// What the repo's own metadata says about its lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "provenance", rename_all = "kebab-case")]
pub enum Finding {
    /// The repo declares a base model, and it is the one asked for.
    Declared { base_model: String },
    /// It declares one, and it is something else. The loudest case: the repo
    /// name and the metadata disagree, and the name is the one that lies.
    Mismatch { declared: String, expected: String },
    /// It declares nothing. Not proof of anything — most GGUF repos are
    /// converted by hand and never fill the field in — which is exactly why
    /// this cannot be a gate.
    Undeclared { expected: String },
}

impl Finding {
    /// Whether a person should be shown this before the bytes move.
    pub fn is_warning(&self) -> bool {
        !matches!(self, Finding::Declared { .. })
    }

    /// One line, both names, no jargon. This is the whole safety story now.
    pub fn message(&self) -> String {
        match self {
            Finding::Declared { base_model } => {
                format!("declares base_model {base_model}")
            }
            Finding::Mismatch { declared, expected } => format!(
                "this repo declares base_model `{declared}`, but you asked for `{expected}` \
                 — they are not builds of the same model"
            ),
            Finding::Undeclared { expected } if expected.is_empty() => {
                "this repo declares no base_model, so nothing says what it is a build of"
                    .to_string()
            }
            Finding::Undeclared { expected } => format!(
                "this repo declares no base_model, so nothing confirms it is a build of \
                 `{expected}`"
            ),
        }
    }
}

/// What a repo declares, reported as a fact.
///
/// **`expected` is the model the *user* named, and is `None` when they named
/// none.** That distinction was got wrong once and is worth stating: comparing
/// a declared `base_model` against the repo's own id is wrong by construction,
/// because differing from the repo name is the entire point of the field.
/// `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` declaring `Qwen/Qwen3.8-27B` is the
/// system working, and the first version of this function called it a mismatch
/// — verified against the live API 2026-09-11, where it warned on every
/// correctly-labelled repo and stayed quiet on one that said nothing at all.
/// A warning that fires on the careful repos and not the careless ones is
/// worse than no warning.
///
/// So a mismatch is only reportable when there is something to mismatch
/// *with*: a model the user asked for by name. Browsing has no such thing, and
/// there the honest finding is "it declares X" or "it declares nothing".
///
/// Both sides are canonicalised before comparison, because legitimate names
/// differ by publisher, format tag and quantisation in every real case.
pub fn check(declared: Option<&str>, expected: Option<&str>) -> Finding {
    match (declared, expected) {
        (Some(d), _) if d.trim().is_empty() => Finding::Undeclared {
            expected: expected.unwrap_or("").to_string(),
        },
        (Some(d), Some(want)) => {
            let a = crate::inventory::identity::canonical_name(d);
            let b = crate::inventory::identity::canonical_name(want);
            if a == b {
                Finding::Declared { base_model: d.to_string() }
            } else {
                Finding::Mismatch { declared: d.to_string(), expected: want.to_string() }
            }
        }
        (Some(d), None) => Finding::Declared { base_model: d.to_string() },
        (None, _) => Finding::Undeclared { expected: expected.unwrap_or("").to_string() },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_publisher_prefix_is_not_a_mismatch() {
        // The common honest case: TheBloke builds Qwen's model and says so.
        assert!(matches!(
            check(Some("Qwen/Qwen3-14B"), Some("Qwen3-14B-GGUF")),
            Finding::Declared { .. }
        ));
    }

    #[test]
    fn a_near_identical_name_is_still_a_mismatch() {
        // `inventory.md` §5's first case: one character of version apart.
        let f = check(Some("Qwen/Qwen2.5-14B"), Some("qwen3-14b"));
        assert!(matches!(f, Finding::Mismatch { .. }), "{f:?}");
        assert!(f.message().contains("Qwen2.5-14B"), "both names are named");
        assert!(f.message().contains("qwen3-14b"));
    }

    #[test]
    fn a_declaration_with_nothing_to_compare_it_to_is_a_fact_not_a_warning() {
        // Browsing: the user typed a query, not a model id. A GGUF repo
        // declaring its upstream base is the system working, and calling that
        // a mismatch warned on every careful repo while staying quiet on the
        // careless ones.
        let f = check(Some("Qwen/Qwen3.8-27B"), None);
        assert!(matches!(f, Finding::Declared { .. }), "{f:?}");
        assert!(!f.is_warning());
    }

    #[test]
    fn declaring_nothing_warns_rather_than_passing() {
        let f = check(None, Some("qwen3-14b"));
        assert!(f.is_warning());
        assert!(f.message().contains("no base_model"));
        // And still warns while browsing, where there is no expected name.
        assert!(check(None, None).is_warning());
    }

    #[test]
    fn an_empty_declaration_is_the_same_as_none() {
        assert_eq!(check(Some("  "), Some("m")), check(None, Some("m")));
    }
}
