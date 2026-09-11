use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Store {
    HfCache,
    LmStudio,
    LlamaCpp,
    Vllm,
    Ollama,
}

impl Store {
    pub const ALL: [Store; 5] = [
        Store::HfCache,
        Store::LmStudio,
        Store::LlamaCpp,
        Store::Vllm,
        Store::Ollama,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Store::HfCache => "hf-cache",
            Store::LmStudio => "lmstudio",
            Store::LlamaCpp => "llamacpp",
            Store::Vllm => "vllm",
            Store::Ollama => "ollama",
        }
    }
}

/// What is inside a safetensors container.
///
/// `.safetensors` is a container and nothing more: bf16, FP8, AWQ, GPTQ and
/// several others all wear the same extension, and which one it is decides
/// whether a provider can load the file, whether it needs converting, and what
/// converting it would even mean.
///
/// Deliberately `Copy` and payload-free. The raw string a repo declared is
/// carried on `hf::Repo` instead, where a refusal can name it -- putting it
/// here would cost `Format` its `Copy` across a dozen comparison sites for a
/// value only error messages read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Quant {
    /// Nothing declared. That genuinely means bf16 -- it is the unquantised
    /// default, and the only case where silence is an answer.
    Bf16,
    /// vLLM's llm-compressor output. `recipe.yaml` beside the weights is its
    /// signature, and a stock vLLM loads it directly.
    CompressedTensors,
    Awq,
    Gptq,
    Fp8,
    BitsAndBytes,
    /// Declared, and not one this build knows.
    ///
    /// **Never folded into `Bf16`.** Guessing "probably bf16" for an
    /// unrecognised method is how a 55 GB dequantisation gets planned for a
    /// file that was never 16-bit.
    Unknown,
}

impl Quant {
    /// What `config.json`'s `quantization_config.quant_method` says.
    pub fn from_method(method: &str) -> Quant {
        match method.trim().to_ascii_lowercase().as_str() {
            "compressed-tensors" | "compressed_tensors" => Quant::CompressedTensors,
            "awq" => Quant::Awq,
            "gptq" => Quant::Gptq,
            "fp8" => Quant::Fp8,
            "bitsandbytes" | "bnb" => Quant::BitsAndBytes,
            _ => Quant::Unknown,
        }
    }

    /// Is this something a converter can read back to full precision?
    ///
    /// bf16 needs no dequantisation; the two the installed toolchains handle
    /// are FP8 (`convert_hf_to_gguf.py --fp8-as-q8`) and, through
    /// `mlx_lm.convert(dequantize=True)`, the compressed-tensors family. An
    /// `Unknown` is not convertible **because harmony does not know it is**,
    /// which is a different claim from it being impossible.
    pub fn is_convertible(&self) -> bool {
        matches!(self, Quant::Bf16 | Quant::Fp8 | Quant::CompressedTensors)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    Gguf,
    Mlx,
    /// Safetensors, with what the repo said is inside it.
    Safetensors(Quant),
    Other,
}

impl Format {
    /// The format of a file **on disk**, where there is no config beside it.
    ///
    /// A path is all this can see, so a `.safetensors` here is assumed
    /// unquantised. `hf::repo_from_json` is the door for a repo being
    /// considered, and it reads the declaration rather than guessing.
    pub fn from_path_str(name: &str) -> Format {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".gguf") {
            Format::Gguf
        } else if lower.ends_with(".safetensors") {
            Format::Safetensors(Quant::Bf16)
        } else if lower.ends_with(".npz") {
            Format::Mlx
        } else {
            Format::Other
        }
    }

    /// Weight-bearing, whatever is inside it.
    pub fn is_weights(&self) -> bool {
        matches!(self, Format::Gguf | Format::Mlx | Format::Safetensors(_))
    }
}

/// Where a claim came from.
///
/// Nothing in slice 2 can produce `Recorded` -- llm-harmony has not performed a
/// conversion. The variant exists so slice 5 has somewhere to put the truth,
/// and so consumers must pattern-match rather than assume.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "provenance", rename_all = "kebab-case")]
pub enum Provenance {
    Recorded,
    Inferred { basis: String },
}

impl Provenance {
    pub fn is_recorded(&self) -> bool {
        matches!(self, Provenance::Recorded)
    }
}

/// Filesystem allocation identity. Two paths with the same `FileKey` are the
/// same bytes and must be counted once.
///
/// This matters concretely: every file under an HF `snapshots/<sha>/` is a
/// symlink into `../../blobs/`, and `~/.llamacpp/models` holds a symlink into
/// `~/.lmstudio/models`. Verified 2026-09-09.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct FileKey {
    pub dev: u64,
    pub ino: u64,
}

impl FileKey {
    pub fn of(path: &Path) -> Option<FileKey> {
        use std::os::unix::fs::MetadataExt;
        // metadata() follows symlinks, which is what we want: we are asking
        // "which allocation is this?", not "is this a link?".
        let md = std::fs::metadata(path).ok()?;
        Some(FileKey {
            dev: md.dev(),
            ino: md.ino(),
        })
    }
}

pub type ArtifactId = String;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Artifact {
    pub id: ArtifactId,
    pub path: PathBuf,
    pub store: Store,
    pub format: Format,
    pub bits: Option<u8>,
    /// Real bytes of the allocation. Identical for every path sharing a `key`.
    pub bytes: u64,
    pub key: Option<FileKey>,
    /// True when this path is a symlink. Removing it reclaims nothing.
    pub is_link: bool,
    /// The name used for identity inference.
    pub name_hint: String,
}

/// Quantisation from a filename. GGUF uses `Q4_K_M`; MLX uses `4bit`.
pub fn bits_from_name(name: &str) -> Option<u8> {
    let upper = name.to_ascii_uppercase();
    for b in [2u8, 3, 4, 5, 6, 8] {
        if upper.contains(&format!("Q{b}_")) || upper.contains(&format!("-{b}BIT")) {
            return Some(b);
        }
    }
    for b in [2u8, 3, 4, 5, 6, 8] {
        if upper.ends_with(&format!("Q{b}")) || upper.contains(&format!("{b}BIT")) {
            return Some(b);
        }
    }
    None
}

impl Artifact {
    pub fn from_path(path: &Path, store: Store) -> Option<Artifact> {
        let name = path.file_name()?.to_string_lossy().to_string();
        let is_link = std::fs::symlink_metadata(path).ok()?.file_type().is_symlink();
        let key = FileKey::of(path);
        let bytes = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
        Some(Artifact {
            id: format!("{}:{}", store.as_str(), path.display()),
            path: path.to_path_buf(),
            store,
            format: Format::from_path_str(&name),
            bits: bits_from_name(&name),
            bytes,
            key,
            is_link,
            name_hint: name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_is_read_from_the_file_shape() {
        assert_eq!(Format::from_path_str("model.q4_k_m.gguf"), Format::Gguf);
        assert_eq!(Format::from_path_str("model.safetensors"), Format::Safetensors(Quant::Bf16));
        assert_eq!(Format::from_path_str("weights.npz"), Format::Mlx);
        assert_eq!(Format::from_path_str("README.md"), Format::Other);
    }

    #[test]
    fn bits_are_parsed_from_the_naming_conventions_in_use() {
        assert_eq!(bits_from_name("Qwen3-14B-Q4_K_M.gguf"), Some(4));
        assert_eq!(bits_from_name("Qwen3.5-9B-Heretic-Q6_K.gguf"), Some(6));
        assert_eq!(bits_from_name("Qwen3-14B-Q8_0.gguf"), Some(8));
        assert_eq!(bits_from_name("Qwen3.5-9B-MLX-6bit"), Some(6));
        assert_eq!(bits_from_name("Llama-3.2-1B-Instruct-4bit"), Some(4));
        assert_eq!(bits_from_name("bge-small-en-v1.5"), None);
    }

    #[test]
    fn provenance_cannot_be_silently_treated_as_fact() {
        let inferred = Provenance::Inferred { basis: "name match".into() };
        assert!(!inferred.is_recorded());
        assert!(Provenance::Recorded.is_recorded());
    }
}
