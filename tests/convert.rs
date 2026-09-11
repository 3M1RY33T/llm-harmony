//! `convert`: the provider's own converter, admitted against the disk it fills.
//!
//! harmony shells out to the tool that owns the format and never reimplements
//! one — the same rule `lms load` and `ollama pull` already follow. Both tools
//! are installed on the machine this was written against, and both already
//! read quantised input, which is the part everyone assumes is the blocker:
//!
//!   convert_hf_to_gguf.py --fp8-as-q8   "Store tensors dequantized from FP8
//!                                        as Q8_0 instead of BF16/F16"
//!   mlx_lm.convert(dequantize=True)     with quantize / q_bits / q_group_size
//!
//! `inventory.md` §5 records the trap this must not fall into: converting with
//! llama.cpp *master* against a `b10240` binary produced GGUFs that would not
//! load — `block_count` 33 with tensors only to 31. The rule it states is
//! **pin the converter to the serving binary's build**, and the converter is
//! therefore found through the provider's own configuration rather than
//! anywhere harmony feels like looking.

use llm_harmony::intake::convert::{self, Plan, Target};
use llm_harmony::inventory::artifact::{Format, Quant};

const GB: u64 = 1024 * 1024 * 1024;

fn plan(target: Target, source: Quant, down: u64, free: u64) -> Result<Plan, String> {
    convert::plan(&convert::Request {
        repo: "x/Model".into(),
        target,
        source_format: Format::Safetensors(source),
        download_bytes: down,
        bits: Some(8),
        converter: convert::Converter::for_test(target),
        free_disk_bytes: free,
        out_path: "/store/out".into(),
    })
}

#[test]
fn converting_to_gguf_invokes_the_llamacpp_converter() {
    let p = plan(Target::Gguf, Quant::Bf16, 15 * GB, 400 * GB).unwrap();
    assert!(p.argv.iter().any(|a| a.contains("convert_hf_to_gguf.py")), "{:?}", p.argv);
}

/// `convert_hf_to_gguf.py` writes bf16 or q8_0 and nothing else; 4-bit GGUF
/// needs a second pass through `llama-quantize`. Silently writing bf16 would
/// hand back a 28 GB file where 8 GB was asked for.
#[test]
fn a_gguf_bit_depth_the_converter_cannot_write_is_refused_by_name() {
    let err = convert::plan(&convert::Request {
        repo: "x/Model".into(),
        target: Target::Gguf,
        source_format: Format::Safetensors(Quant::Bf16),
        download_bytes: GB,
        bits: Some(4),
        converter: convert::Converter::for_test(Target::Gguf),
        free_disk_bytes: 400 * GB,
        out_path: "/store/out".into(),
    })
    .unwrap_err();
    assert!(err.contains("llama-quantize"), "{err}");
}

#[test]
fn converting_to_mlx_invokes_mlx_lm_convert() {
    let p = plan(Target::Mlx, Quant::Bf16, 15 * GB, 400 * GB).unwrap();
    assert!(p.argv.iter().any(|a| a == "mlx_lm.convert"), "{:?}", p.argv);
    assert!(p.argv.iter().any(|a| a == "--quantize"));
    assert!(p.argv.iter().any(|a| a == "8"), "the bit depth reaches it: {:?}", p.argv);
}

/// FP8 does not need to be refused: the converter reads it. This is the case
/// that produced r29 — a repo described as "bf16 needing a conversion
/// llm-harmony does not do yet", which was neither bf16 nor beyond the tools.
#[test]
fn an_fp8_source_is_converted_rather_than_refused() {
    let p = plan(Target::Gguf, Quant::Fp8, 15 * GB, 400 * GB).unwrap();
    assert!(p.argv.iter().any(|a| a == "--fp8-as-q8"), "{:?}", p.argv);
}

#[test]
fn a_quantised_source_is_dequantised_on_the_way_to_mlx() {
    let p = plan(Target::Mlx, Quant::CompressedTensors, 15 * GB, 400 * GB).unwrap();
    assert!(p.argv.iter().any(|a| a == "--dequantize"), "{:?}", p.argv);
}

/// A declaration harmony does not recognise is not one it can promise to
/// dequantise. Refusing beats starting an hour of work on a guess.
#[test]
fn an_unknown_source_quantisation_is_refused_by_name() {
    let err = plan(Target::Gguf, Quant::Unknown, 15 * GB, 400 * GB).unwrap_err();
    assert!(err.contains("does not know"), "{err}");
}

/// Three figures, not one. A conversion admitted on the download alone fills
/// the disk at 70%, which is the failure this project exists to prevent.
#[test]
fn a_conversion_is_admitted_against_download_scratch_and_result() {
    let p = plan(Target::Gguf, Quant::Fp8, 15 * GB, 400 * GB).unwrap();
    assert_eq!(p.download_bytes, 15 * GB);
    assert!(p.scratch_bytes > 0, "dequantising needs room of its own");
    assert!(p.result_bytes > 0);
    assert_eq!(p.total_bytes(), p.download_bytes + p.scratch_bytes + p.result_bytes);
}

/// A source already at full precision needs no intermediate copy: the
/// converter streams it into the output. Charging for one would refuse
/// conversions that fit.
#[test]
fn a_bf16_source_needs_no_dequantisation_scratch() {
    let p = plan(Target::Gguf, Quant::Bf16, 15 * GB, 400 * GB).unwrap();
    assert_eq!(p.scratch_bytes, 0);
    assert_eq!(p.total_bytes(), 30 * GB);
}

#[test]
fn a_conversion_that_will_not_fit_on_disk_is_refused_with_all_three_figures() {
    let err = plan(Target::Gguf, Quant::Bf16, 15 * GB, 20 * GB).unwrap_err();
    for want in ["download", "scratch", "result"] {
        assert!(err.contains(want), "{want} missing from: {err}");
    }
}

/// An FP8 source has to be expanded to full precision before it can be
/// requantised, so its scratch is far larger than its download — the figure
/// that makes "just convert it" the expensive option.
#[test]
fn a_quantised_source_needs_more_scratch_than_its_download() {
    let quantised = plan(Target::Gguf, Quant::Fp8, 15 * GB, 400 * GB).unwrap();
    let plain = plan(Target::Gguf, Quant::Bf16, 15 * GB, 400 * GB).unwrap();
    assert!(quantised.scratch_bytes > plain.scratch_bytes);
}

/// `inventory.md` §5: converting with a converter from a different build than
/// the server produced GGUFs that would not load. A mismatch is refused rather
/// than attempted.
#[test]
fn a_converter_from_a_different_build_than_the_server_is_refused() {
    let err = convert::plan(&convert::Request {
        repo: "x/Model".into(),
        target: Target::Gguf,
        source_format: Format::Safetensors(Quant::Bf16),
        download_bytes: GB,
        bits: None,
        converter: convert::Converter::mismatched_for_test(),
        free_disk_bytes: 400 * GB,
        out_path: "/store/out".into(),
    })
    .unwrap_err();
    assert!(err.contains("build"), "{err}");
}

/// A missing toolchain is a named refusal that says where it was looked for —
/// never a silent skip. The rule `adapters/lmstudio.rs` already states for
/// `lms`.
#[test]
fn a_missing_converter_is_reported_with_the_path_it_was_looked_for() {
    let err = convert::plan(&convert::Request {
        repo: "x/Model".into(),
        target: Target::Mlx,
        source_format: Format::Safetensors(Quant::Bf16),
        download_bytes: GB,
        bits: None,
        converter: convert::Converter::Missing {
            looked_in: "/nowhere/venv/bin/python".into(),
            target: Target::Mlx,
        },
        free_disk_bytes: 400 * GB,
        out_path: "/store/out".into(),
    })
    .unwrap_err();
    assert!(err.contains("/nowhere/venv/bin/python"), "{err}");
}

/// The build that produced an artifact is recorded, which is the first time
/// `inventory.md` §7 Q1's `Recorded` provenance has been reachable: lineage is
/// recorded when harmony did the conversion, and inferred otherwise.
#[test]
fn the_converters_build_is_carried_on_the_plan() {
    let p = plan(Target::Gguf, Quant::Bf16, GB, 400 * GB).unwrap();
    assert_eq!(p.converter_build.as_deref(), Some("b10240"));
}
