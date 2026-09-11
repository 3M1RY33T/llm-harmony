//! What a model is shaped like, read from the artifact itself.
//!
//! The KV cache is the part of a footprint that scales with the context window,
//! and it scales with the model's geometry: layers, kv heads, head dimension.
//! None of the four providers publish any of those. The artifact does.
//!
//! This is the first code in the project that opens a model file. Everything
//! before it identified artifacts by path, size and name, which answers *which*
//! model something is and nothing about what loading it will cost.
//!
//! Deliberately six keys wide, not a GGUF parser. It reads a bounded prefix of
//! the header, takes what the estimator needs, and gives up rather than guesses.

use std::path::Path;

/// How much of a file to read, escalating, before giving up on the metadata.
///
/// Measured 2026-09-11 on a real Q8_0 14B: the metadata block ends at **5.9 MB**,
/// of which 5.3 MB is three tokenizer arrays -- `tokens` (2.6 MB), `merges`
/// (2.7 MB) and `token_type`. A single 1 MiB bound looked generous against the
/// synthetic fixtures and truncated every real artifact on this machine, which
/// is this project's own rule arriving on schedule: a reader is verified by the
/// file, not by its tests.
///
/// Escalating rather than one large read, because the keys that matter are
/// written before the tokenizer and the walk stops when it reaches it: the
/// common case never reads past the first step.
const HEADER_READ_STEPS: [usize; 3] = [256 * 1024, 8 * 1024 * 1024, 48 * 1024 * 1024];

/// Why a header walk stopped, so `from_gguf` knows whether reading more would
/// help. Truncation is worth a retry; a file that is not a GGUF never is.
enum Stop {
    NotGguf,
    Truncated,
    Incomplete,
}

/// The geometry an estimate needs, however the artifact spells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShape {
    pub arch: String,
    pub n_layers: u32,
    /// Key/value heads. Equal to attention heads without GQA; on this
    /// machine's models it is 5x smaller, so defaulting it wrong under-counts
    /// the cache fivefold.
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// The artifact's own size on disk. The floor of any estimate.
    pub weights_bytes: u64,
    /// What the model was trained for -- a capability figure, never a budget.
    /// Carried so a caller can notice it asked for more than the model has.
    pub trained_context: Option<u32>,
}

/// A shape as it arrives on the command line, before it is checked.
///
/// Separate from `ModelShape` on purpose: every field is optional here and
/// none are there, so the check below is the only way across and cannot be
/// skipped by a caller in a hurry.
#[derive(serde::Deserialize)]
struct SuppliedShape {
    arch: Option<String>,
    n_layers: Option<u32>,
    n_kv_heads: Option<u32>,
    head_dim: Option<u32>,
    weights_bytes: Option<u64>,
    trained_context: Option<u32>,
}

impl ModelShape {
    /// A shape handed in rather than read off an artifact.
    ///
    /// Everything else in this module derives a shape from bytes harmony read
    /// itself. This is the one door for a shape that came from somewhere else
    /// -- a catalog that describes a model nobody has downloaded yet -- and it
    /// exists because the alternative is pricing that model from its published
    /// file size and reporting `Declared`, which covers weights and not the
    /// cache that scales with the window.
    ///
    /// **All four geometry fields or none.** A missing `n_kv_heads` defaulted
    /// to the attention-head count under-counts the cache fivefold on this
    /// machine's models, and zero is refused for the same reason a missing
    /// value is: a shape claiming no layers would price a model at its weights
    /// and call the answer computed. Weights are the floor of every estimate,
    /// so their absence is fatal too.
    pub fn from_json(raw: &str) -> Result<ModelShape, String> {
        let supplied: SuppliedShape =
            serde_json::from_str(raw).map_err(|e| format!("shape is not readable: {e}"))?;

        let arch = supplied.arch.unwrap_or_default();
        if arch.trim().is_empty() {
            return Err("shape is missing `arch`".to_string());
        }
        let mut missing: Vec<&str> = Vec::new();
        let mut need = |name: &'static str, value: Option<u32>| -> u32 {
            match value {
                Some(v) if v > 0 => v,
                _ => {
                    missing.push(name);
                    0
                }
            }
        };
        let n_layers = need("n_layers", supplied.n_layers);
        let n_kv_heads = need("n_kv_heads", supplied.n_kv_heads);
        let head_dim = need("head_dim", supplied.head_dim);
        let weights_bytes = match supplied.weights_bytes {
            Some(v) if v > 0 => v,
            _ => {
                missing.push("weights_bytes");
                0
            }
        };
        if !missing.is_empty() {
            return Err(format!(
                "shape is incomplete: {} -- a partial geometry is no geometry, \
                 because a defaulted head count under-counts the cache fivefold",
                missing.join(", ")
            ));
        }
        Ok(ModelShape {
            arch,
            n_layers,
            n_kv_heads,
            head_dim,
            weights_bytes,
            trained_context: supplied.trained_context.filter(|c| *c > 0),
        })
    }
}

/// A GGUF file, an MLX directory, or a file inside one.
///
/// One entry point so no caller has to branch on format. A path that is
/// neither is `None`, which every caller must already handle.
pub fn from_artifact(path: &Path) -> Option<ModelShape> {
    if path.is_dir() {
        return from_mlx_config(path);
    }
    // Sniffed, not guessed from the extension: Ollama stores its weights as
    // `blobs/sha256-<digest>` with no extension at all, and they are ordinary
    // GGUFs. Keying on `.gguf` made every Ollama model unpriceable, which
    // showed up as "only a declared figure is available" on a model harmony
    // could see perfectly well. Found 2026-09-11.
    if let Some(shape) = from_gguf(path) {
        return Some(shape);
    }
    // A shard inside an MLX snapshot: its config is a sibling.
    path.parent().and_then(from_mlx_config)
}

// --- GGUF ------------------------------------------------------------------
//
// Layout, from the format's own specification:
//
//   "GGUF" | version u32 | tensor_count u64 | kv_count u64 | kv pairs...
//   kv pair: key_len u64 | key bytes | value_type u32 | value
//
// Every value type has a knowable width, so a value this reader does not want
// can be stepped over without being understood -- which is most of them: a
// real header carries the whole tokenizer.

const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let s = self.b.get(self.at..end)?;
        self.at = end;
        Some(s)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<String> {
        let n = self.u64()? as usize;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    /// Width of a fixed-size value, or `None` for the two that are not fixed.
    fn fixed_width(t: u32) -> Option<usize> {
        match t {
            T_UINT8 | T_INT8 | T_BOOL => Some(1),
            T_UINT16 | T_INT16 => Some(2),
            T_UINT32 | T_INT32 | T_FLOAT32 => Some(4),
            T_UINT64 | T_INT64 | T_FLOAT64 => Some(8),
            _ => None,
        }
    }

    /// Step over one value of the given type, whatever it is.
    fn skip_value(&mut self, t: u32) -> Option<()> {
        if let Some(w) = Self::fixed_width(t) {
            self.take(w)?;
            return Some(());
        }
        match t {
            T_STRING => {
                let n = self.u64()? as usize;
                self.take(n)?;
                Some(())
            }
            T_ARRAY => {
                let elem = self.u32()?;
                let count = self.u64()?;
                if let Some(w) = Self::fixed_width(elem) {
                    // Multiplied in usize: a corrupt count must not wrap.
                    let total = (count as usize).checked_mul(w)?;
                    self.take(total)?;
                    return Some(());
                }
                if elem != T_STRING {
                    // A nested array. Nothing this reader needs lives inside
                    // one, and walking it would be the parser this is not.
                    return None;
                }
                for _ in 0..count {
                    let n = self.u64()? as usize;
                    self.take(n)?;
                }
                Some(())
            }
            _ => None,
        }
    }

    /// Read a value as u32 when it is an integer type, else step over it.
    fn u32_value(&mut self, t: u32) -> Option<u32> {
        match t {
            T_UINT32 | T_INT32 => self.u32(),
            T_UINT64 | T_INT64 => self.u64().map(|v| v as u32),
            T_UINT16 | T_INT16 => Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?).into()),
            T_UINT8 | T_INT8 => Some(self.take(1)?[0].into()),
            _ => {
                self.skip_value(t)?;
                None
            }
        }
    }
}

pub fn from_gguf(path: &Path) -> Option<ModelShape> {
    let weights_bytes = std::fs::metadata(path).ok()?.len();
    for limit in HEADER_READ_STEPS {
        let bytes = read_prefix(path, limit)?;
        let short = bytes.len() < limit;
        match parse_gguf(&bytes, weights_bytes) {
            Ok(shape) => return Some(shape),
            // Nothing longer will make it a GGUF, and an incomplete header is
            // incomplete however much of it we read.
            Err(Stop::NotGguf) | Err(Stop::Incomplete) => return None,
            // Already had the whole file; more bytes do not exist.
            Err(Stop::Truncated) if short => return None,
            Err(Stop::Truncated) => continue,
        }
    }
    None
}

fn parse_gguf(bytes: &[u8], weights_bytes: u64) -> Result<ModelShape, Stop> {
    let mut c = Cursor { b: bytes, at: 0 };
    if c.take(4).ok_or(Stop::Truncated)? != b"GGUF" {
        return Err(Stop::NotGguf);
    }
    let _version = c.u32().ok_or(Stop::Truncated)?;
    let _tensor_count = c.u64().ok_or(Stop::Truncated)?;
    let kv_count = c.u64().ok_or(Stop::Truncated)?;

    let mut arch: Option<String> = None;
    let mut vals: Vec<(String, u32)> = Vec::new();

    for _ in 0..kv_count {
        let key = c.string().ok_or(Stop::Truncated)?;
        let t = c.u32().ok_or(Stop::Truncated)?;

        // The tokenizer is where the megabytes are, and it is written after
        // the hyperparameters. Reaching it with everything in hand means the
        // rest of the header is 5 MB of arrays nobody here needs.
        if key.starts_with("tokenizer.") && arch.is_some() && has_required(&arch, &vals) {
            break;
        }

        if key == "general.architecture" {
            arch = Some(c.string().ok_or(Stop::Truncated)?);
            continue;
        }
        // Only the numeric hyperparameters are worth keeping.
        if key.contains("block_count")
            || key.contains("head_count")
            || key.contains("key_length")
            || key.contains("value_length")
            || key.contains("embedding_length")
            || key.ends_with("context_length")
        {
            if let Some(v) = c.u32_value(t) {
                vals.push((key, v));
            }
        } else {
            c.skip_value(t).ok_or(Stop::Truncated)?;
        }
    }

    let arch = arch.ok_or(Stop::Incomplete)?;
    let get = |suffix: &str| -> Option<u32> {
        vals.iter().find(|(k, _)| k == &format!("{arch}.{suffix}")).map(|(_, v)| *v)
    };

    let n_layers = get("block_count").ok_or(Stop::Incomplete)?;
    let n_heads = get("attention.head_count").ok_or(Stop::Incomplete)?;
    // Without GQA a model publishes no kv head count, and the two are equal.
    let n_kv_heads = get("attention.head_count_kv").unwrap_or(n_heads);
    let head_dim = head_dim(get("attention.key_length"), get("embedding_length"), n_heads)
        .ok_or(Stop::Incomplete)?;
    let trained_context = get("context_length");

    Ok(ModelShape {
        arch,
        n_layers,
        n_kv_heads,
        head_dim,
        weights_bytes,
        trained_context,
    })
}

/// Enough to stop walking: the geometry the estimator cannot do without.
fn has_required(arch: &Option<String>, vals: &[(String, u32)]) -> bool {
    let Some(arch) = arch else { return false };
    let has = |suffix: &str| vals.iter().any(|(k, _)| k == &format!("{arch}.{suffix}"));
    has("block_count")
        && has("attention.head_count")
        && (has("attention.key_length") || has("embedding_length"))
}

// --- MLX -------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct MlxConfig {
    model_type: Option<String>,
    num_hidden_layers: Option<u32>,
    num_attention_heads: Option<u32>,
    num_key_value_heads: Option<u32>,
    head_dim: Option<u32>,
    hidden_size: Option<u32>,
    max_position_embeddings: Option<u32>,
}

pub fn from_mlx_config(dir: &Path) -> Option<ModelShape> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let c: MlxConfig = serde_json::from_str(&raw).ok()?;

    let n_layers = c.num_hidden_layers?;
    let n_heads = c.num_attention_heads?;
    let n_kv_heads = c.num_key_value_heads.unwrap_or(n_heads);
    let head_dim = head_dim(c.head_dim, c.hidden_size, n_heads)?;

    // A config beside no weights is a config, not a model. Returning a shape
    // with `weights_bytes: 0` would price a 9 GB artifact at the size of its
    // KV cache alone -- confidently, and an order of magnitude low. Found
    // 2026-09-11 when an LM Studio candidate resolved to the `config.json`
    // sitting next to a GGUF, and the estimate came back as 1.3 GB.
    let weights_bytes = safetensors_bytes(dir);
    if weights_bytes == 0 {
        return None;
    }

    Some(ModelShape {
        arch: c.model_type.unwrap_or_else(|| "unknown".to_string()),
        n_layers,
        n_kv_heads,
        head_dim,
        weights_bytes,
        trained_context: c.max_position_embeddings,
    })
}

/// Every shard, summed. An MLX model is a directory, so its size is the sum of
/// its weight files rather than one file's length.
fn safetensors_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
        })
        // `DirEntry::metadata` does NOT follow symlinks, and every weight file
        // in the Hugging Face cache is a symlink into `blobs/` -- which read
        // as a few dozen bytes each and made a 4 GB model weigh nothing.
        .filter_map(|e| std::fs::metadata(e.path()).ok().map(|m| m.len()))
        .sum()
}

// --- shared ----------------------------------------------------------------

/// Declared if the artifact says so, else embedding width over head count.
///
/// Both are common: GGUF publishes `key_length` for some architectures and not
/// others, and plenty of MLX configs omit `head_dim`. A zero head count would
/// divide by zero, so it is refused rather than defaulted.
fn head_dim(declared: Option<u32>, embedding: Option<u32>, n_heads: u32) -> Option<u32> {
    if let Some(d) = declared.filter(|d| *d > 0) {
        return Some(d);
    }
    let e = embedding?;
    if n_heads == 0 {
        return None;
    }
    Some(e / n_heads).filter(|d| *d > 0)
}

/// At most `n` bytes of a file, without reading the rest of it.
fn read_prefix(path: &Path, n: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let f = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    f.take(n as u64).read_to_end(&mut buf).ok()?;
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_prefers_what_the_artifact_declares() {
        assert_eq!(head_dim(Some(128), Some(5120), 40), Some(128));
    }

    #[test]
    fn head_dim_ignores_a_declared_zero() {
        assert_eq!(head_dim(Some(0), Some(5120), 40), Some(128));
    }

    /// No embedding width and no declared dimension is not a shape. Returning
    /// a default here would put a plausible number into a multiplication that
    /// decides whether the machine swaps.
    #[test]
    fn head_dim_refuses_rather_than_defaulting() {
        assert_eq!(head_dim(None, None, 40), None);
        assert_eq!(head_dim(None, Some(5120), 0), None);
    }

    #[test]
    fn an_array_of_fixed_width_values_is_stepped_over_exactly() {
        let mut b: Vec<u8> = Vec::new();
        b.extend(T_UINT32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        b.extend(2u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(b"after");

        let mut c = Cursor { b: &b, at: 0 };
        c.skip_value(T_ARRAY).expect("steps over the array");
        assert_eq!(c.take(5).unwrap(), b"after");
    }

    /// A corrupt count must not be multiplied into a wrapped offset.
    #[test]
    fn an_absurd_array_count_is_refused_not_wrapped() {
        let mut b: Vec<u8> = Vec::new();
        b.extend(T_UINT64.to_le_bytes());
        b.extend(u64::MAX.to_le_bytes());
        let mut c = Cursor { b: &b, at: 0 };
        assert!(c.skip_value(T_ARRAY).is_none());
    }
}
