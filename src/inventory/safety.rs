use crate::inventory::artifact::{Artifact, ArtifactId, Format, Store};

/// Three kinds of "safe to delete", which are not the same risk.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "safety", rename_all = "kebab-case")]
pub enum Safety {
    /// A better artifact of the same model, same format, exists locally.
    Redundant { better: ArtifactId },
    /// Deleting costs a re-download or a re-convert, and the recipe is known.
    Reproducible { how: String },
    /// The only local copy of something that cannot be rebuilt from what
    /// remains. Never auto-selected.
    Irreplaceable { why: String },
}

/// Classify `a` against the artifacts sharing its canonical identity.
///
/// Two rules from docs/inventory.md are encoded here and must not be relaxed:
/// quantisation is one-way (Q4 cannot become Q8), and there is no MLX <-> GGUF
/// path -- both descend from the same source and neither derives from the other.
pub fn classify(a: &Artifact, peers: &[Artifact]) -> Safety {
    // A strictly better build of the SAME format makes this redundant.
    if let (Some(bits), Format::Gguf | Format::Mlx) = (a.bits, a.format) {
        if let Some(better) = peers
            .iter()
            .filter(|p| p.id != a.id && p.format == a.format)
            .filter(|p| p.bits.map(|b| b > bits).unwrap_or(false))
            .max_by_key(|p| p.bits)
        {
            return Safety::Redundant {
                better: better.id.clone(),
            };
        }
    }

    // A source can be re-downloaded. Priced, not free.
    if a.store == Store::HfCache || a.format == Format::SafetensorsBf16 {
        return Safety::Reproducible {
            how: "re-download from Hugging Face".to_string(),
        };
    }

    // A converted artifact whose source is still present can be rebuilt.
    let source_present = peers
        .iter()
        .any(|p| p.format == Format::SafetensorsBf16 || p.store == Store::HfCache);
    if source_present {
        return Safety::Reproducible {
            how: "re-convert from the local source".to_string(),
        };
    }

    Safety::Irreplaceable {
        why: "no local source to rebuild from, and no better build of this format".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::artifact::{Artifact, Format, Store};

    fn art(id: &str, name: &str, bits: Option<u8>, format: Format, store: Store) -> Artifact {
        Artifact {
            id: id.into(),
            path: format!("/x/{name}").into(),
            store,
            format,
            bits,
            bytes: 1024,
            key: None,
            is_link: false,
            name_hint: name.into(),
        }
    }

    /// A Q4 alongside a Q6 of the same model is redundant. Quantisation is
    /// one-way, so the higher-bit artifact is the one worth keeping.
    #[test]
    fn a_lower_bit_build_beside_a_higher_one_is_redundant() {
        let q4 = art("a", "Qwen3.5-9B-heretic-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        let q6 = art("b", "Qwen3.5-9B-heretic-Q6_K.gguf", Some(6), Format::Gguf, Store::LlamaCpp);
        let peers = vec![q4.clone(), q6.clone()];
        assert!(matches!(classify(&q4, &peers), Safety::Redundant { .. }));
    }

    #[test]
    fn the_highest_bit_build_is_never_redundant() {
        let q4 = art("a", "m-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        let q6 = art("b", "m-Q6_K.gguf", Some(6), Format::Gguf, Store::LlamaCpp);
        assert!(!matches!(classify(&q6, &vec![q4, q6.clone()]), Safety::Redundant { .. }));
    }

    /// There is no MLX <-> GGUF path. An MLX build does not make a GGUF build
    /// redundant, whatever the bit depths.
    #[test]
    fn a_different_format_never_makes_an_artifact_redundant() {
        let gguf = art("a", "m-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        let mlx = art("b", "m-MLX-6bit", Some(6), Format::Mlx, Store::Vllm);
        assert!(!matches!(classify(&gguf, &vec![gguf.clone(), mlx]), Safety::Redundant { .. }));
    }

    #[test]
    fn an_hf_source_is_reproducible() {
        let src = art("a", "org/model", None, Format::SafetensorsBf16, Store::HfCache);
        assert!(matches!(classify(&src, &vec![src.clone()]), Safety::Reproducible { .. }));
    }

    #[test]
    fn a_lone_build_with_no_source_is_irreplaceable() {
        let only = art("a", "m-Q4_K_M.gguf", Some(4), Format::Gguf, Store::LlamaCpp);
        assert!(matches!(classify(&only, &vec![only.clone()]), Safety::Irreplaceable { .. }));
    }
}
