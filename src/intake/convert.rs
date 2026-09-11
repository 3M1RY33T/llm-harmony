//! Converting a model with the provider's own converter.
//!
//! harmony shells out to the tool that owns the format and never reimplements
//! one — the same rule `lms load` and `ollama pull` already follow. Both tools
//! are already installed wherever a provider is properly set up:
//!
//! | Target | Tool | Found through |
//! |---|---|---|
//! | GGUF | `convert_hf_to_gguf.py` | the llama.cpp checkout beside the server |
//! | MLX  | `mlx_lm.convert` | the vLLM-MLX venv |
//!
//! **The converter is found through the provider, not through a search path.**
//! `inventory.md` §5 records why: converting with llama.cpp *master* against a
//! `b10240` binary produced GGUFs that would not load — `block_count` 33 with
//! tensors only to 31 — and the rule it states is *pin the converter to the
//! serving binary's build*. That pinning is a property of the user's own
//! provider setup, which is exactly why `ProviderConfig::start` points at their
//! script rather than harmony composing a command. A mismatch is refused here
//! rather than attempted.
//!
//! **Both converters already read quantised input**, which is the part usually
//! assumed to be the blocker:
//!
//! * `convert_hf_to_gguf.py --fp8-as-q8` — "Store tensors dequantized from FP8
//!   as Q8_0 instead of BF16/F16."
//! * `mlx_lm.convert(dequantize=True)`, with `quantize`, `q_bits`, `q_group_size`.
//!
//! So this is the expensive route, not the impossible one — and the cost is
//! why `siblings` is offered first. Dequantising an FP8 27B to full precision
//! is tens of gigabytes of scratch before a single output byte is written,
//! which is the figure `Plan` exists to put in front of someone.

use serde::Serialize;

use crate::inventory::artifact::{Format, Quant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    Gguf,
    Mlx,
}

impl Target {
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Gguf => "gguf",
            Target::Mlx => "mlx",
        }
    }
}

/// Where the converter for a target lives, and which build it is.
#[derive(Debug, Clone)]
pub enum Converter {
    /// A llama.cpp checkout whose `convert_hf_to_gguf.py` is at `build`.
    LlamaCpp { script: String, python: String, build: String },
    /// A venv with `mlx_lm` in it.
    MlxLm { python: String },
    /// Present, but at a different build than the server it would feed.
    /// `inventory.md` §5's trap, refused rather than attempted.
    BuildMismatch { converter: String, server: String },
    Missing { looked_in: String, target: Target },
}

impl Converter {
    /// A fixture standing in for a correctly set up toolchain for `target`.
    pub fn for_test(target: Target) -> Converter {
        match target {
            Target::Gguf => Converter::LlamaCpp {
                script: "/home/.llamacpp/repo/convert_hf_to_gguf.py".into(),
                python: "/home/.llamacpp/venv/bin/python".into(),
                build: "b10240".into(),
            },
            Target::Mlx => Converter::MlxLm { python: "/home/.vllm-mlx/venv/bin/python".into() },
        }
    }

    pub fn mismatched_for_test() -> Converter {
        Converter::BuildMismatch { converter: "master".into(), server: "b10240".into() }
    }

    pub fn build(&self) -> Option<&str> {
        match self {
            Converter::LlamaCpp { build, .. } => Some(build),
            _ => None,
        }
    }
}

/// Find the converter for a target through the provider that would serve it.
///
/// Deliberately not a PATH search. `inventory.md` §5's rule is that the
/// converter must match the serving binary's build, and which binary that is
/// is a fact about the user's setup — `ProviderConfig::start` points at their
/// own script precisely because harmony composing a command would be a second,
/// worse source of truth. So the converter is looked for beside the server,
/// and a `start` command is the thread that leads there.
pub fn discover(config: &crate::config::Config, target: Target) -> Converter {
    use crate::provider::ProviderKind;
    let want = match target {
        Target::Gguf => ProviderKind::LlamaCpp,
        Target::Mlx => ProviderKind::Vllm,
    };
    let home = std::env::var("HOME").unwrap_or_default();
    // The provider's own directory, taken from its start command where there
    // is one, and from the conventional layout otherwise.
    let root = config
        .providers
        .iter()
        .find(|p| p.kind == want)
        .and_then(|p| p.start.as_deref())
        .and_then(provider_root)
        .unwrap_or_else(|| match target {
            Target::Gguf => format!("{home}/.llamacpp"),
            Target::Mlx => format!("{home}/.vllm-mlx"),
        });

    match target {
        Target::Gguf => {
            let script = format!("{root}/repo/convert_hf_to_gguf.py");
            let python = format!("{root}/venv/bin/python");
            if !std::path::Path::new(&script).exists() {
                return Converter::Missing { looked_in: script, target };
            }
            Converter::LlamaCpp {
                build: git_describe(&format!("{root}/repo")).unwrap_or_default(),
                script,
                python,
            }
        }
        Target::Mlx => {
            let python = format!("{root}/venv/bin/python");
            if !std::path::Path::new(&python).exists() {
                return Converter::Missing { looked_in: python, target };
            }
            Converter::MlxLm { python }
        }
    }
}

/// `$HOME/.llamacpp/serve-pool --port 8080` -> `/Users/x/.llamacpp`.
fn provider_root(start: &str) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let first = start.split_whitespace().next()?.replace("$HOME", &home);
    let path = std::path::Path::new(&first);
    Some(path.parent()?.to_string_lossy().to_string())
}

/// The checkout's build tag, which is what has to match the serving binary.
fn git_describe(repo: &str) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["-C", repo, "describe", "--tags"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let tag = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!tag.is_empty()).then_some(tag)
}

/// The weight files a conversion would read, and what they weigh together.
///
/// Not `run::best_build`, which answers "what could a provider load here" and
/// so never returns safetensors — the one thing a conversion is always FROM.
/// Sharded sources are summed: a 27B repo is a dozen `model-0000N-of-0000M`
/// files and the converter reads every one, so pricing the smallest would
/// under-state the job by an order of magnitude.
pub fn source_of(repo: &crate::intake::hf::Repo) -> Option<(Format, u64)> {
    let parts: Vec<_> = repo
        .files
        .iter()
        .filter(|f| matches!(f.format, Format::Safetensors(_)))
        .collect();
    let first = parts.first()?;
    // One missing size makes the whole total unknown, the same rule
    // `run::builds` applies: summing what is published and calling it the
    // source's weight under-states it, which is the direction that admits a
    // conversion that should not have been.
    let bytes = parts.iter().try_fold(0u64, |acc, f| f.size_bytes.map(|b| acc + b))?;
    Some((first.format, bytes))
}

pub struct Request {
    pub repo: String,
    pub target: Target,
    /// What the repo declares it publishes.
    pub source_format: Format,
    pub download_bytes: u64,
    /// Output bit depth, where the target takes one.
    pub bits: Option<u8>,
    pub converter: Converter,
    pub free_disk_bytes: u64,
    /// Where the result lands. Named by the caller because placement is the
    /// disk ledger's answer, not this module's.
    pub out_path: String,
}

/// What a conversion would cost and how it would be run. Moves nothing.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub repo: String,
    pub target: Target,
    pub argv: Vec<String>,
    /// The source, as fetched.
    pub download_bytes: u64,
    /// Peak working space. For a quantised source this is the dequantised
    /// model, which is several times the download — the figure that makes
    /// converting the expensive option rather than the obvious one.
    pub scratch_bytes: u64,
    pub result_bytes: u64,
    /// Which build produced it. Recorded on the artifact, which is the first
    /// time `inventory.md` §7 Q1's `Recorded` provenance has been reachable:
    /// lineage is recorded when harmony did the conversion, inferred
    /// otherwise.
    pub converter_build: Option<String>,
}

impl Plan {
    /// Every byte that has to be free at once. The download is still on disk
    /// while the scratch is written, and the scratch while the result is.
    pub fn total_bytes(&self) -> u64 {
        self.download_bytes
            .saturating_add(self.scratch_bytes)
            .saturating_add(self.result_bytes)
    }
}

/// How much larger than its download a source becomes at full precision.
///
/// bf16 is already full precision, so its scratch is the model itself. FP8 and
/// the compressed-tensors family roughly double on the way back to 16-bit, and
/// the margin is deliberately generous: under-estimating scratch fills a disk
/// at 70% through an hour of work, which is worse than refusing.
fn dequantised_multiple(source: Quant) -> u64 {
    match source {
        // Already full precision: the converter streams the source into the
        // output and needs no intermediate copy of its own.
        Quant::Bf16 => 0,
        _ => 3,
    }
}

pub fn plan(request: &Request) -> Result<Plan, String> {
    let Format::Safetensors(source) = request.source_format else {
        return Err(format!(
            "`{}` is already in a format a provider here loads; there is nothing to convert",
            request.repo
        ));
    };
    if !source.is_convertible() {
        return Err(format!(
            "`{}` declares a quantisation this build does not know how to read, so harmony \
             cannot promise to dequantise it — refusing rather than starting an hour of work \
             on a guess",
            request.repo
        ));
    }

    let argv = match (&request.converter, request.target) {
        (Converter::BuildMismatch { converter, server }, _) => {
            return Err(format!(
                "the converter is at build `{converter}` and the server it would feed is at \
                 `{server}`; converting across a build boundary produced GGUFs that would not \
                 load (docs/inventory.md §5), so this is refused rather than attempted"
            ))
        }
        (Converter::Missing { looked_in, target }, _) => {
            return Err(format!(
                "no {} converter found; looked in `{looked_in}`",
                target.as_str()
            ))
        }
        (Converter::LlamaCpp { script, python, .. }, Target::Gguf) => {
            // `--outtype` is the output PRECISION, not a quantisation step:
            // llama.cpp quantises with a separate binary (`llama-quantize`),
            // and the converter's only quantised outtype is q8_0. Asking for
            // 4-bit here and silently writing bf16 would hand back a 28 GB
            // file where 8 GB was requested, so it is refused instead, naming
            // the step that is missing.
            if let Some(bits) = request.bits {
                if bits != 8 && bits != 16 {
                    return Err(format!(
                        "`convert_hf_to_gguf.py` writes bf16 or q8_0 and nothing else; \
                         {bits}-bit GGUF needs a second pass through `llama-quantize`, \
                         which harmony does not run yet"
                    ));
                }
            }
            let mut argv = vec![python.clone(), script.clone(), "--outfile".into()];
            argv.push(request.out_path.clone());
            argv.push("--outtype".into());
            argv.push(if request.bits == Some(8) { "q8_0".into() } else { "bf16".into() });
            // "Store tensors dequantized from FP8 as Q8_0 instead of
            // BF16/F16" — the flag that makes a float-quantised repo
            // convertible at all, and at a fraction of the scratch a full
            // bf16 round trip would need. compressed-tensors carries FP8 in
            // its `float-quantized` form, which is the repo that produced
            // r29, so it takes the flag too; it is a no-op where no FP8
            // tensors are present.
            if matches!(source, Quant::Fp8 | Quant::CompressedTensors) {
                argv.push("--fp8-as-q8".into());
            }
            argv
        }
        (Converter::MlxLm { python }, Target::Mlx) => {
            let mut argv = vec![
                python.clone(),
                "-m".into(),
                "mlx_lm.convert".into(),
                "--hf-path".into(),
                request.repo.clone(),
                "--mlx-path".into(),
                request.out_path.clone(),
            ];
            if source != Quant::Bf16 {
                // Back to full precision before requantising. Lossy twice, and
                // the reason `siblings` is offered above this.
                argv.push("--dequantize".into());
            }
            if let Some(bits) = request.bits {
                argv.push("--quantize".into());
                argv.push("--q-bits".into());
                argv.push(bits.to_string());
            }
            argv
        }
        (have, want) => {
            return Err(format!(
                "the {} converter cannot produce {}",
                match have {
                    Converter::LlamaCpp { .. } => "llama.cpp",
                    Converter::MlxLm { .. } => "mlx_lm",
                    _ => "configured",
                },
                want.as_str()
            ))
        }
    };

    let scratch_bytes = request
        .download_bytes
        .saturating_mul(dequantised_multiple(source));
    // The result is the download's order of magnitude: a requantisation lands
    // near where it started, and over-stating it refuses conversions that
    // would have fitted.
    let result_bytes = request.download_bytes;

    let plan = Plan {
        repo: request.repo.clone(),
        target: request.target,
        argv,
        download_bytes: request.download_bytes,
        scratch_bytes,
        result_bytes,
        converter_build: request.converter.build().map(str::to_string),
    };

    if plan.total_bytes() > request.free_disk_bytes {
        // All three, named. "Not enough disk" against a figure the user can
        // see is actionable; against one they cannot is not.
        return Err(format!(
            "needs {} at once — {} download, {} scratch, {} result — and {} is free",
            crate::render::human_bytes(plan.total_bytes()),
            crate::render::human_bytes(plan.download_bytes),
            crate::render::human_bytes(plan.scratch_bytes),
            crate::render::human_bytes(plan.result_bytes),
            crate::render::human_bytes(request.free_disk_bytes),
        ));
    }
    Ok(plan)
}
