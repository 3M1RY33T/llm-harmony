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
