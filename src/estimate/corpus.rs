use std::path::Path;

use crate::record::{Observation, OBSERVATION_SCHEMA};

/// Every readable observation of the current schema.
///
/// The corpus is append-only and outlives any single build, so this is
/// deliberately forgiving: a torn final line from a process that died
/// mid-write, or a record from an older schema, is skipped rather than
/// allowed to discard everything after it.
pub fn load(path: &Path) -> Vec<Observation> {
    let Ok(body) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    body.lines()
        .filter_map(|line| serde_json::from_str::<Observation>(line).ok())
        .filter(|o| o.schema == OBSERVATION_SCHEMA)
        .collect()
}

pub fn load_default() -> Vec<Observation> {
    match crate::record::default_path() {
        Some(p) => load(&p),
        None => Vec::new(),
    }
}
