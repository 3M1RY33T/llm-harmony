//! Where a pulled file goes.
//!
//! Placement is the disk ledger's answer, not the user's guess: a GGUF for
//! Ollama and a GGUF for LM Studio are the same bytes and belong in different
//! stores, and a file in the wrong one is a file the provider that was supposed
//! to serve it cannot see.
//!
//! Reuses the scanners' own `roots()` rather than repeating the paths. Two
//! stores that disagree about where they are is the bug this avoids by
//! construction — the thing that indexes a directory and the thing that writes
//! into it are the same source.

use std::path::PathBuf;

use crate::inventory::artifact::{Format, Store};
use crate::inventory::scan::StoreScanner;
use crate::provider::ProviderKind;

/// Which store serves a provider, for a format it can actually load.
///
/// `None` where the pairing is not real: Ollama will not read a loose GGUF out
/// of a directory (its store is content-addressed behind `ollama pull`), and no
/// provider here loads bf16 safetensors without a conversion this slice does
/// not do.
pub fn store_for(provider: ProviderKind, format: Format) -> Option<Store> {
    match (provider, format) {
        (ProviderKind::LmStudio, Format::Gguf) => Some(Store::LmStudio),
        (ProviderKind::LmStudio, Format::Mlx) => Some(Store::LmStudio),
        (ProviderKind::LlamaCpp, Format::Gguf) => Some(Store::LlamaCpp),
        (ProviderKind::Vllm, Format::Mlx) => Some(Store::Vllm),
        // Ollama's store is a blob database keyed by digest, written by its own
        // pull. Dropping a file into it produces something nothing will load.
        (ProviderKind::Ollama, _) => None,
        _ => None,
    }
}

/// The directory a repo's files land in, or `None` when the store has no root
/// on this machine.
///
/// The repo id becomes the directory, so `TheBloke/Qwen3-14B-GGUF` lands at
/// `<store>/TheBloke/Qwen3-14B-GGUF/` — the layout every scanner here already
/// expects, and the one that keeps a publisher's two builds apart.
pub fn directory_for(store: Store, repo_id: &str) -> Option<PathBuf> {
    let root = scanner_for(store)?.root()?;
    let mut out = root;
    for segment in repo_id.split('/').filter(|s| !s.is_empty() && *s != "." && *s != "..") {
        out.push(segment);
    }
    Some(out)
}

fn scanner_for(store: Store) -> Option<Box<dyn StoreScanner>> {
    use crate::inventory::scan::{
        hf::HfCache, llamacpp::LlamaCppPool, lmstudio::LmStudioStore, ollama::OllamaStore,
        vllm::VllmStore,
    };
    let s: Box<dyn StoreScanner> = match store {
        Store::HfCache => Box::new(HfCache),
        Store::LmStudio => Box::new(LmStudioStore),
        Store::LlamaCpp => Box::new(LlamaCppPool),
        Store::Vllm => Box::new(VllmStore),
        Store::Ollama => Box::new(OllamaStore),
    };
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gguf_goes_to_the_store_of_the_provider_that_will_serve_it() {
        assert_eq!(store_for(ProviderKind::LmStudio, Format::Gguf), Some(Store::LmStudio));
        assert_eq!(store_for(ProviderKind::LlamaCpp, Format::Gguf), Some(Store::LlamaCpp));
        // Same bytes, different store. This is the whole point of the module.
        assert_ne!(
            store_for(ProviderKind::LmStudio, Format::Gguf),
            store_for(ProviderKind::LlamaCpp, Format::Gguf)
        );
    }

    #[test]
    fn ollama_takes_no_loose_files() {
        // Its store is a digest-keyed blob database, so a file written into it
        // is a file nothing will ever load.
        assert_eq!(store_for(ProviderKind::Ollama, Format::Gguf), None);
    }

    #[test]
    fn a_format_no_provider_here_can_load_has_nowhere_to_go() {
        assert_eq!(store_for(ProviderKind::LmStudio, Format::SafetensorsBf16), None);
    }

    #[test]
    fn a_repo_id_becomes_its_directory_and_cannot_escape_the_store() {
        let d = directory_for(Store::LmStudio, "../../etc/passwd").expect("a root");
        assert!(!d.to_string_lossy().contains(".."), "{d:?}");
        assert!(d.ends_with("etc/passwd"), "the segments survive, the escape does not: {d:?}");
    }
}
