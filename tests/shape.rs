//! Reading a model's shape from the artifact itself.
//!
//! The first code in this project that opens a model file. Everything before
//! it identified artifacts by filename and size, which is enough to say *which*
//! model something is and nothing at all about what it will cost to load.
//!
//! Fixtures are built here rather than committed: a real GGUF is gigabytes, and
//! the six keys that matter are a few hundred bytes of header.

mod support;

use llm_harmony::estimate::shape;
use support::tree::Tree;

/// GGUF value type tags, from the format's own spec.
const T_U32: u32 = 4;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_F32: u32 = 6;

fn key(out: &mut Vec<u8>, k: &str) {
    out.extend((k.len() as u64).to_le_bytes());
    out.extend(k.as_bytes());
}

fn u32_kv(out: &mut Vec<u8>, k: &str, v: u32) {
    key(out, k);
    out.extend(T_U32.to_le_bytes());
    out.extend(v.to_le_bytes());
}

fn string_kv(out: &mut Vec<u8>, k: &str, v: &str) {
    key(out, k);
    out.extend(T_STRING.to_le_bytes());
    out.extend((v.len() as u64).to_le_bytes());
    out.extend(v.as_bytes());
}

/// A metadata block a real GGUF would carry and this reader must step over:
/// the tokenizer's token list is an array of tens of thousands of strings.
fn array_kv(out: &mut Vec<u8>, k: &str, items: &[&str]) {
    key(out, k);
    out.extend(T_ARRAY.to_le_bytes());
    out.extend(T_STRING.to_le_bytes());
    out.extend((items.len() as u64).to_le_bytes());
    for it in items {
        out.extend((it.len() as u64).to_le_bytes());
        out.extend(it.as_bytes());
    }
}

fn f32_kv(out: &mut Vec<u8>, k: &str, v: f32) {
    key(out, k);
    out.extend(T_F32.to_le_bytes());
    out.extend(v.to_le_bytes());
}

/// Header only: magic, version, tensor count, kv count, then the pairs.
/// Nothing here reads tensor data, so none is written.
fn gguf(t: &Tree, name: &str, arch: &str, kvs: &[(&str, u32)], extras: bool) -> std::path::PathBuf {
    let mut b: Vec<u8> = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());

    let count = kvs.len() as u64 + 1 + if extras { 2 } else { 0 };
    b.extend(count.to_le_bytes());

    string_kv(&mut b, "general.architecture", arch);
    if extras {
        array_kv(&mut b, "tokenizer.ggml.tokens", &["<s>", "</s>", "hello"]);
        f32_kv(&mut b, "qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    }
    for (k, v) in kvs {
        u32_kv(&mut b, k, *v);
    }

    let p = t.root.join(name);
    std::fs::write(&p, b).unwrap();
    p
}

#[test]
fn a_gguf_header_yields_the_keys_the_estimator_needs() {
    let t = Tree::new("shape-gguf");
    let p = gguf(
        &t,
        "qwen3-14b-q4_k_m.gguf",
        "qwen3",
        &[
            ("qwen3.block_count", 40),
            ("qwen3.attention.head_count", 40),
            ("qwen3.attention.head_count_kv", 8),
            ("qwen3.attention.key_length", 128),
            ("qwen3.embedding_length", 5120),
            ("qwen3.context_length", 40960),
        ],
        false,
    );

    let s = shape::from_gguf(&p).expect("readable header");
    assert_eq!(s.arch, "qwen3");
    assert_eq!(s.n_layers, 40);
    assert_eq!(s.n_kv_heads, 8);
    assert_eq!(s.head_dim, 128);
    assert_eq!(s.trained_context, Some(40960));
    assert!(s.weights_bytes > 0, "the artifact's own size on disk");
}

/// A real header carries tokenizer arrays and float hyperparameters between
/// the keys that matter. Stepping over a value whose type the reader does not
/// care about is most of the work.
#[test]
fn values_of_uninteresting_types_are_stepped_over() {
    let t = Tree::new("shape-skip");
    let p = gguf(
        &t,
        "m.gguf",
        "qwen3",
        &[
            ("qwen3.block_count", 40),
            ("qwen3.attention.head_count", 40),
            ("qwen3.attention.head_count_kv", 8),
            ("qwen3.embedding_length", 5120),
        ],
        true,
    );
    let s = shape::from_gguf(&p).expect("arrays and floats must not stop the walk");
    assert_eq!(s.n_layers, 40);
    assert_eq!(s.n_kv_heads, 8);
}

/// `key_length` is optional. Without it the head dimension is
/// embedding_length / head_count -- 5120/40 = 128 for this shape.
#[test]
fn head_dim_falls_back_to_embedding_over_heads() {
    let t = Tree::new("shape-headdim");
    let p = gguf(
        &t,
        "m.gguf",
        "qwen3",
        &[
            ("qwen3.block_count", 40),
            ("qwen3.attention.head_count", 40),
            ("qwen3.attention.head_count_kv", 8),
            ("qwen3.embedding_length", 5120),
        ],
        false,
    );
    assert_eq!(shape::from_gguf(&p).unwrap().head_dim, 128);
}

/// Multi-head attention: kv heads default to attention heads when the key is
/// absent, which is what pre-GQA models publish. Getting this backwards would
/// under-count the cache by the GQA ratio -- 5x on this machine's models.
#[test]
fn kv_heads_default_to_attention_heads() {
    let t = Tree::new("shape-mha");
    let p = gguf(
        &t,
        "m.gguf",
        "llama",
        &[
            ("llama.block_count", 32),
            ("llama.attention.head_count", 32),
            ("llama.embedding_length", 4096),
        ],
        false,
    );
    assert_eq!(shape::from_gguf(&p).unwrap().n_kv_heads, 32);
}

/// Never panic on a file that is not a GGUF, however it is malformed. This
/// reader runs over whatever path a provider claims to be serving.
#[test]
fn a_file_that_is_not_a_gguf_is_none_not_a_panic() {
    let t = Tree::new("shape-notgguf");
    let p = t.write("not.gguf", "\u{0}\u{1}\u{2}");
    assert!(shape::from_gguf(&p).is_none());
}

#[test]
fn a_truncated_header_is_none_not_a_guess() {
    let t = Tree::new("shape-truncated");
    let mut b: Vec<u8> = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(9u64.to_le_bytes()); // promises nine pairs, carries none
    let p = t.root.join("short.gguf");
    std::fs::write(&p, b).unwrap();
    assert!(shape::from_gguf(&p).is_none());
}

#[test]
fn a_missing_file_is_none() {
    assert!(shape::from_gguf(std::path::Path::new("/nonexistent/m.gguf")).is_none());
}

/// A header with no layer count is not a shape. Returning a partial one would
/// put a zero into a multiplication and produce a confident, tiny estimate.
#[test]
fn a_header_missing_the_layer_count_is_none() {
    let t = Tree::new("shape-nolayers");
    let p = gguf(
        &t,
        "m.gguf",
        "qwen3",
        &[("qwen3.attention.head_count", 40), ("qwen3.embedding_length", 5120)],
        false,
    );
    assert!(shape::from_gguf(&p).is_none());
}

/// MLX ships plain JSON, and it is the only half that works for vLLM-MLX and
/// LM Studio's MLX runtime.
#[test]
fn an_mlx_config_yields_the_same_shape() {
    let t = Tree::new("shape-mlx");
    t.write(
        "config.json",
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "max_position_embeddings": 40960
        }"#,
    );
    t.file("model-00001-of-00002.safetensors", 4096);
    t.file("model-00002-of-00002.safetensors", 2048);

    let s = shape::from_mlx_config(&t.root).expect("readable config");
    assert_eq!(s.arch, "qwen3");
    assert_eq!((s.n_layers, s.n_kv_heads, s.head_dim), (40, 8, 128));
    assert_eq!(s.trained_context, Some(40960));
    assert_eq!(s.weights_bytes, 6144, "every safetensors shard, summed");
}

/// `head_dim` is absent from plenty of configs; the same fallback applies.
#[test]
fn an_mlx_config_without_head_dim_derives_it() {
    let t = Tree::new("shape-mlx-nohead");
    t.write(
        "config.json",
        r#"{"model_type":"llama","num_hidden_layers":32,"num_attention_heads":32,"hidden_size":4096}"#,
    );
    let s = shape::from_mlx_config(&t.root).unwrap();
    assert_eq!(s.head_dim, 128);
    assert_eq!(s.n_kv_heads, 32, "no GQA key means kv heads equal attention heads");
}

/// One entry point, so callers never branch on the artifact's format.
#[test]
fn from_artifact_dispatches_on_what_the_path_is() {
    let t = Tree::new("shape-dispatch");
    let g = gguf(
        &t,
        "m.gguf",
        "qwen3",
        &[
            ("qwen3.block_count", 40),
            ("qwen3.attention.head_count", 40),
            ("qwen3.embedding_length", 5120),
        ],
        false,
    );
    assert!(shape::from_artifact(&g).is_some(), "a gguf file");

    let d = Tree::new("shape-dispatch-dir");
    d.write(
        "config.json",
        r#"{"model_type":"qwen3","num_hidden_layers":40,"num_attention_heads":40,"hidden_size":5120}"#,
    );
    assert!(shape::from_artifact(&d.root).is_some(), "an mlx directory");

    // A safetensors file inside an MLX snapshot: the config is its sibling.
    let f = d.file("model.safetensors", 16);
    assert!(shape::from_artifact(&f).is_some(), "a file beside a config");
}

/// The defect the synthetic fixtures could not catch, and the live artifacts
/// caught immediately.
///
/// A real GGUF's metadata block is **5.9 MB**, nearly all of it three
/// tokenizer arrays written after the hyperparameters. Measured 2026-09-11 on
/// `Qwen3-14B-...-Q8_0.gguf`. The first version of this reader took a flat
/// 1 MiB prefix, passed every fixture here, and returned `None` for every
/// model on the machine.
#[test]
fn a_header_larger_than_the_first_read_step_still_reads() {
    let t = Tree::new("shape-bigheader");

    let mut b: Vec<u8> = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(6u64.to_le_bytes());

    string_kv(&mut b, "general.architecture", "qwen3");
    u32_kv(&mut b, "qwen3.block_count", 40);
    u32_kv(&mut b, "qwen3.attention.head_count", 40);
    u32_kv(&mut b, "qwen3.attention.head_count_kv", 8);
    u32_kv(&mut b, "qwen3.embedding_length", 5120);

    // ~1.5 MB of tokenizer, past the first read step.
    let tokens: Vec<&str> = vec!["a-token-of-some-length"; 64_000];
    array_kv(&mut b, "tokenizer.ggml.tokens", &tokens);
    assert!(b.len() > 1024 * 1024, "the fixture must exceed the first step");

    let p = t.root.join("big.gguf");
    std::fs::write(&p, b).unwrap();

    let s = shape::from_gguf(&p).expect("a 1.5 MB header is an ordinary header");
    assert_eq!((s.n_layers, s.n_kv_heads, s.head_dim), (40, 8, 128));
}

/// And when the tokenizer comes first, the early exit cannot fire, so the
/// escalating read is what has to carry it.
#[test]
fn a_header_whose_arrays_precede_the_hyperparameters_still_reads() {
    let t = Tree::new("shape-arrays-first");

    let mut b: Vec<u8> = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(5u64.to_le_bytes());

    let tokens: Vec<&str> = vec!["a-token-of-some-length"; 64_000];
    array_kv(&mut b, "tokenizer.ggml.tokens", &tokens);
    string_kv(&mut b, "general.architecture", "qwen3");
    u32_kv(&mut b, "qwen3.block_count", 40);
    u32_kv(&mut b, "qwen3.attention.head_count", 40);
    u32_kv(&mut b, "qwen3.embedding_length", 5120);

    let p = t.root.join("arrays-first.gguf");
    std::fs::write(&p, b).unwrap();

    assert_eq!(shape::from_gguf(&p).expect("escalates past the array").n_layers, 40);
}

/// Weight files in the Hugging Face cache are symlinks into `blobs/`. Sizing
/// them without following the link reported a 4 GB model as a few hundred
/// bytes -- found 2026-09-11 against the real cache.
#[test]
fn mlx_weights_are_sized_through_a_symlink() {
    let blobs = Tree::new("shape-blobs");
    let real = blobs.file("sha256-abc", 8192);

    let snap = Tree::new("shape-snapshot");
    snap.write(
        "config.json",
        r#"{"model_type":"qwen3","num_hidden_layers":40,"num_attention_heads":40,"hidden_size":5120}"#,
    );
    snap.link("model.safetensors", &real);

    let s = shape::from_mlx_config(&snap.root).unwrap();
    assert_eq!(s.weights_bytes, 8192, "the blob's size, not the link's");
}
