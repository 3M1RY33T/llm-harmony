//! Which providers could take this model, and where each would put it.
//!
//! **Not every provider handles every format**, and until 2026-09-11 nothing
//! said so before a pull was attempted: Delroy offered a destination picker
//! listing every provider on the machine, and choosing one that could not read
//! the build produced a refusal *after* the click. A picker whose options are
//! not all valid is not a picker, it is a guess with a menu.
//!
//! So the question is asked here, once, against the repo's own build — and
//! answered with `run::placement`, the same function `plan_add` uses to decide
//! where bytes go. A second copy of that rule would offer a provider the pull
//! then refused, which is the exact failure this verb exists to prevent.
//!
//! Every provider is reported, including the ones that cannot take it, each
//! with its reason. A destination that silently disappears teaches nobody why
//! their model has nowhere to go.

use serde::Serialize;

use crate::http::Http;
use crate::intake::{fit, hf, run};
use crate::inventory::artifact::Format;
use crate::memory::Machine;
use crate::provider::ProviderKind;

#[derive(Debug, Clone, Serialize)]
pub struct Destination {
    pub provider: String,
    /// Whether a pull into it would be planned at all. The fit is a separate
    /// question and a softer one — `add` warns and proceeds on a model that
    /// lands but cannot load — so a `true` here does not promise it will run.
    pub ok: bool,
    /// Why not. Empty when `ok`.
    pub reason: String,
    pub store: Option<String>,
    pub target_dir: Option<String>,
    /// What Ollama would be asked to pull, for the transport that is not a
    /// file drop.
    pub registry_name: Option<String>,
    pub fit: Option<fit::Fit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DestinationsDoc {
    pub schema: u32,
    pub verb: &'static str,
    pub repo: String,
    /// The build every destination was judged against: one file, or a shard
    /// set folded into one. Named so a caller shows what it priced.
    pub file: String,
    pub format: Option<Format>,
    pub bytes: Option<u64>,
    pub destinations: Vec<Destination>,
    /// Present when the repo has no build at all — a format nothing here
    /// reads. Then `destinations` is empty and this is the whole answer.
    pub refusal: Option<String>,
}

/// Ask every provider, off one read of the repo.
pub fn for_repo(
    http: &Http,
    machine: &Machine,
    repo_id: &str,
    providers: &[ProviderKind],
    want_file: Option<&str>,
) -> DestinationsDoc {
    let mut doc = DestinationsDoc {
        schema: 1,
        verb: "destinations",
        repo: repo_id.to_string(),
        file: String::new(),
        format: None,
        bytes: None,
        destinations: Vec::new(),
        refusal: None,
    };
    let repo = match hf::repo(http, repo_id) {
        Ok(r) => r,
        Err(e) => {
            doc.refusal = Some(format!("could not read `{repo_id}`: {e:?}"));
            return doc;
        }
    };
    // The same choice `plan_add` makes, so what is offered here is what would
    // actually be pulled. An explicit `--file` is honoured for the same reason
    // it is there: the caller has looked.
    let build = match want_file {
        Some(name) => repo
            .files
            .iter()
            .find(|f| f.name == name)
            .map(|f| run::Build { files: vec![f.clone()], bytes: f.size_bytes }),
        None => run::best_build(&repo),
    };
    let Some(build) = build else {
        doc.refusal = Some(format!(
            "`{repo_id}` publishes no build any provider here can load as published"
        ));
        return doc;
    };
    doc.file = build.label();
    doc.bytes = build.bytes;
    doc.format = Some(build.files[0].format);

    for provider in providers {
        doc.destinations.push(match run::placement(*provider, repo_id, &build) {
            Ok(p) => Destination {
                provider: provider.as_str().to_string(),
                ok: true,
                reason: String::new(),
                store: Some(p.store.as_str().to_string()),
                target_dir: p.target_dir.as_ref().map(|d| d.display().to_string()),
                registry_name: p.registry_name,
                // Priced per destination, because the stores are not all on
                // one filesystem and disk is half the admission.
                fit: run::price_build(machine, &build, Some(&p.filesystem)),
            },
            Err(reason) => Destination {
                provider: provider.as_str().to_string(),
                ok: false,
                reason,
                store: None,
                target_dir: None,
                registry_name: None,
                fit: None,
            },
        });
    }
    doc
}
