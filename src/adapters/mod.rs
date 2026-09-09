pub mod llamacpp;
pub mod lmstudio;
pub mod ollama;
pub mod vllm;

use crate::provider::{Adapter, ProviderKind};

/// The adapter for a given kind. One place, so the ledger never matches on kind.
pub fn adapter_for(kind: ProviderKind) -> Box<dyn Adapter> {
    match kind {
        ProviderKind::LmStudio => Box::new(lmstudio::LmStudio),
        ProviderKind::LlamaCpp => Box::new(llamacpp::LlamaCpp),
        ProviderKind::Vllm => Box::new(vllm::Vllm),
        ProviderKind::Ollama => Box::new(ollama::Ollama),
    }
}
