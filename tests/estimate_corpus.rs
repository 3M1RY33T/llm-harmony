mod support;

use llm_harmony::estimate::corpus;
use support::tree::Tree;

const V2: &str = r#"{"schema":2,"at":100,"provider":"lmstudio","loaded":1,"model":"qwen3-14b","context_tokens":40960,"attributable":true,"footprint_bytes":9600000000,"phys_footprint_bytes":6890000000,"rss_bytes":8670000000,"processes":[{"pid":647,"footprint_bytes":300000000,"phys_footprint_bytes":300000000,"rss_bytes":280000000},{"pid":25164,"footprint_bytes":9000000000,"phys_footprint_bytes":6300000000,"rss_bytes":9000000000}],"machine_total_bytes":25769803776,"machine_used_bytes":20000000000,"swap_used_bytes":5100000000}"#;

#[test]
fn well_formed_lines_are_loaded() {
    let t = Tree::new("corpus-ok");
    let p = t.write("observations.jsonl", &format!("{V2}\n{V2}\n"));
    assert_eq!(corpus::load(&p).len(), 2);
}

/// The corpus is append-only and long-lived. One unreadable line -- a partial
/// write, a record from an older schema -- must not discard the rest.
#[test]
fn a_corrupt_line_is_skipped_not_fatal() {
    let t = Tree::new("corpus-corrupt");
    let p = t.write("observations.jsonl", &format!("{V2}\nnot json\n{{}}\n{V2}\n"));
    assert_eq!(corpus::load(&p).len(), 2);
}

/// Schema 1 records have no `processes`, so nothing can be attributed from
/// them. They are dropped rather than read with v2 assumptions.
#[test]
fn records_from_an_older_schema_are_dropped() {
    let t = Tree::new("corpus-v1");
    let v1 = V2.replace(r#""schema":2"#, r#""schema":1"#);
    let p = t.write("observations.jsonl", &format!("{v1}\n{V2}\n"));
    assert_eq!(corpus::load(&p).len(), 1);
}

#[test]
fn a_missing_corpus_is_empty_not_an_error() {
    assert!(corpus::load(std::path::Path::new("/nonexistent/observations.jsonl")).is_empty());
}

/// The safety property the whole slice rests on: for any shape that has a
/// trustworthy measurement, the computed basis must not come in **below** it.
/// Under-prediction is the failure that wedges the machine; over-prediction
/// only wastes room.
///
/// Ratio on the one shape this machine has data for, measured 2026-09-11:
/// computed 16.97 GB against a measured 9.00 GB, **1.88x**. That gap is not
/// evidence the formula is wrong -- every observation behind the 9.00 GB was
/// taken with 5.0-8.8 GB of swap in use, and now fails `is_trustworthy`. The
/// margin was deliberately NOT tuned to close it. See the module docs on
/// `TRUSTWORTHY_SWAP_CEILING_BYTES`.
#[test]
fn computed_never_under_predicts_a_trustworthy_measurement() {
    use llm_harmony::estimate::computed::{computed, KV_DTYPE_BYTES};
    use llm_harmony::estimate::estimator;
    use llm_harmony::estimate::shape::ModelShape;

    // The real geometry of this machine's 14B, read from its own header.
    let shape = ModelShape {
        arch: "qwen3".into(),
        n_layers: 40,
        n_kv_heads: 8,
        head_dim: 128,
        weights_bytes: 9_004_072_960,
        trained_context: Some(40_960),
    };

    let calm = V2.replace(r#""swap_used_bytes":5100000000"#, r#""swap_used_bytes":0"#);
    let t = Tree::new("corpus-safety");
    let p = t.write("observations.jsonl", &format!("{calm}\n"));
    let obs = corpus::load(&p);

    let measured = estimator::for_model(&obs, "qwen3-14b", Some(40_960));
    let m = measured.bytes.expect("the fixture measures this shape when calm");
    let c = computed(&shape, 40_960, KV_DTYPE_BYTES)
        .bytes
        .expect("a shape that parsed is a shape that can be priced");

    assert!(
        c >= m,
        "computed {c} is below measured {m}: the estimator would admit a load the \
         machine has already been observed unable to hold"
    );
}

/// And the corpus this machine actually has contains no trustworthy
/// measurement at all -- every line was recorded while paging. Kept as a test
/// so the day that stops being true is visible.
#[test]
fn every_observation_in_the_shipped_fixture_was_taken_under_pressure() {
    let t = Tree::new("corpus-pressure");
    let p = t.write("observations.jsonl", &format!("{V2}\n"));
    let obs = corpus::load(&p);
    assert!(!obs.is_empty(), "the fixture parses");
    assert!(
        obs.iter().all(|o| !llm_harmony::estimate::estimator::is_trustworthy(o)),
        "the fixture was captured at 5.1 GB of swap, like every real observation"
    );
}
