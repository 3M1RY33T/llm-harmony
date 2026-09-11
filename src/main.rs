use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};

use llm_harmony::config::Config;
use llm_harmony::http::Http;
use llm_harmony::ledger::Ledger;
use llm_harmony::memory::Machine;
use llm_harmony::render::{render_json, render_table};

#[derive(Parser)]
#[command(name = "llm-harmony", about = "Memory accounting across local inference providers")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show what every provider currently has loaded, and what it costs.
    Status {
        #[arg(long)]
        json: bool,
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1500)]
        timeout_ms: u64,
        /// Append this reading to the observation log for the estimator.
        #[arg(long)]
        record: bool,
        /// Where to append. Defaults to ~/.local/state/llm-harmony/observations.jsonl
        #[arg(long, value_name = "PATH")]
        record_path: Option<PathBuf>,
    },
    /// What is on disk, grouped by model.
    Ls {
        #[arg(long)]
        json: bool,
        /// Also poll providers, so resident models become removal blockers.
        #[arg(long)]
        live: bool,
        /// Report separate allocations holding the same bytes instead of the
        /// per-model table. Two paths that share an allocation are not
        /// duplicates and never appear here.
        #[arg(long)]
        duplicates: bool,
    },
    /// Where should a request for this model go?
    Resolve {
        model: String,
        #[arg(long)]
        context: Option<u32>,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Make a model resident, evicting unpinned models if that is what it takes.
    Load {
        model: String,
        #[arg(long)]
        context: Option<u32>,
        /// Serve it from this provider, rather than whichever one resolves.
        #[arg(long)]
        provider: Option<String>,
        /// How long the provider should hold it without use. Its own default
        /// otherwise -- harmony runs no timer.
        #[arg(long, value_name = "SECONDS")]
        ttl: Option<u64>,
        /// Protect it from eviction once it is loaded.
        #[arg(long)]
        pin: bool,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        /// Override the watchdog's abort threshold. Half the reserve by default.
        #[arg(long, value_name = "BYTES")]
        floor: Option<u64>,
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Free a model. Naming it is consent, so a pin does not stop this.
    Unload {
        model: String,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Replace one resident model with another, across providers or not.
    ///
    /// Always frees the first, even when both would have fitted: `switch`
    /// means "I am done with this one". `load` is how two models end up
    /// resident together.
    Switch {
        from: String,
        to: String,
        #[arg(long)]
        context: Option<u32>,
        /// Pin the *target* to this provider. `from` is still matched by model
        /// id across providers, which is what makes a stale one harmless.
        #[arg(long)]
        provider: Option<String>,
        /// How long the provider should hold the target without use.
        #[arg(long, value_name = "SECONDS")]
        ttl: Option<u64>,
        #[arg(long)]
        pin: bool,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        #[arg(long, value_name = "BYTES")]
        floor: Option<u64>,
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Protect a model from being evicted to make room for another.
    ///
    /// Works whether or not it is resident, and the pin is sticky: it survives
    /// an unload, so the next load is protected too. Cleared only by `unpin`.
    Pin {
        model: String,
        /// Pin this provider's spelling only. Without it, every provider that
        /// could serve the model is pinned.
        #[arg(long)]
        provider: Option<String>,
        /// Why, for whoever reads `status` in a fortnight.
        #[arg(long)]
        note: Option<String>,
    },
    /// Remove that protection.
    Unpin {
        model: String,
        #[arg(long)]
        provider: Option<String>,
    },
    /// What each provider can actually do, probed rather than assumed.
    Verify {
        #[arg(long)]
        json: bool,
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1500)]
        timeout_ms: u64,
    },
    /// Write a launchd agent for a provider. Writes to ~/Library/LaunchAgents.
    Install {
        provider: String,
        /// Print the agent instead of writing it.
        #[arg(long)]
        dry_run: bool,
    },
    /// Start a provider and wait until it answers.
    Start {
        provider: String,
        #[arg(long, default_value_t = 60)]
        timeout_s: u64,
    },
    /// Stop a provider harmony installed.
    Stop { provider: String },
    /// What a model costs, and whether it fits right now.
    Estimate {
        model: String,
        #[arg(long)]
        context: Option<u32>,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Print a removal plan. Never executes.
    Rm {
        model: String,
        #[arg(long, default_value_t = true)]
        dry_run: bool,
        /// Poll providers so a served model produces a refusal.
        #[arg(long)]
        live: bool,
    },
    /// Print one removal plan per model for every superseded build. Never
    /// executes.
    ///
    /// `rm` asks "may I remove this model?"; `clean` asks it of all of them.
    /// A refusal takes one model off the table rather than the whole run.
    Clean {
        /// Select builds a better one supersedes: lower bits, same model, same
        /// format, with the better build left in place. The only selector
        /// there is -- `reproducible` is a priced deletion, not a clean.
        #[arg(long)]
        redundant: bool,
        /// Print the plan and stop. The only mode this slice has, and the
        /// posture docs/inventory.md section 7 question 2 argues for.
        #[arg(long, default_value_t = true)]
        dry_run: bool,
        /// Poll providers, so a served model produces a refusal and the HF
        /// cache stops being assumed to be scanned by a router that is down.
        #[arg(long)]
        live: bool,
        #[arg(long)]
        json: bool,
    },
}

fn inventory(live: bool) -> llm_harmony::inventory::Inventory {
    if !live {
        return llm_harmony::inventory::Inventory::scan_offline(None);
    }
    let config = Config::load(None).unwrap_or_else(|_| Config::defaults());
    let machine = Machine::read().expect("machine memory readable");
    let http = Http::new(Duration::from_millis(1500));
    llm_harmony::inventory::Inventory::scan(&config, &http, machine)
}

/// The configured entry for a provider named on the command line.
fn provider_config(
    name: &str,
) -> Result<llm_harmony::config::ProviderConfig, String> {
    let kind: llm_harmony::provider::ProviderKind = name.parse()?;
    let config = Config::load(None)?;
    config
        .providers
        .into_iter()
        .find(|p| p.kind == kind)
        .ok_or_else(|| format!("`{name}` is not in your config"))
}

/// Which (provider, model-id) pairs a `pin`/`unpin` names.
///
/// With `--provider` the model string is taken as that provider's own id and
/// used verbatim. Without it the name is resolved the way `resolve` resolves
/// it -- by artifact identity -- and every provider that could serve it is
/// pinned, because "protect this model" means the model, not one server's
/// spelling of it.
fn pin_targets(
    model: &str,
    provider: Option<&str>,
) -> Result<Vec<(llm_harmony::provider::ProviderKind, String)>, String> {
    if let Some(name) = provider {
        let kind: llm_harmony::provider::ProviderKind = name.parse()?;
        return Ok(vec![(kind, model.to_string())]);
    }

    let config = Config::load(None)?;
    let machine = Machine::read()?;
    let http = Http::new(Duration::from_millis(1500));
    let ledger = Ledger::assemble(&config, &http, machine);
    let inventory = llm_harmony::inventory::Inventory::scan_offline(None);
    let candidates = llm_harmony::resolve::identity::candidates(&ledger, &inventory, model);

    if candidates.is_empty() {
        // Refusing beats pinning nothing: a pin on a name no provider serves
        // is a silent no-op waiting to surprise whoever set it.
        return Err(format!(
            "no provider serves `{model}`; name one with --provider to pin it anyway"
        ));
    }
    Ok(candidates
        .into_iter()
        .map(|c| (c.provider, c.provider_model_id))
        .collect())
}

/// Config and machine, the two things every actuating verb needs and the only
/// two whose failure is fatal before anything is touched.
fn setup(config: Option<&std::path::Path>) -> Result<(Config, Machine), String> {
    Ok((Config::load(config)?, Machine::read()?))
}

/// Print a report and turn it into an exit code.
///
/// Anything that is not `Ready` exits non-zero: a caller must be able to tell
/// "the model is there" from "the model is not there" without parsing prose.
fn emit(report: &llm_harmony::actuate::run::Report, json: bool) -> ExitCode {
    if json {
        println!("{}", serde_json::to_string_pretty(report).unwrap());
    } else {
        print!("{}", llm_harmony::render_actuate::render(report));
    }
    if report.is_success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// `load` and `switch`: the same machinery, differing only in whether a model
/// is freed first.
#[allow(clippy::too_many_arguments)]
fn actuate(
    verb: &'static str,
    mut request: llm_harmony::actuate::plan::Request,
    provider: Option<String>,
    pin: bool,
    reserve: Option<u64>,
    floor: Option<u64>,
    config_path: Option<&std::path::Path>,
    json: bool,
) -> ExitCode {
    // A named provider is a constraint on the target, carried into the plan
    // rather than validated and dropped: a caller that says where the model
    // lives must not be quietly resolved somewhere else.
    if let Some(p) = &provider {
        match p.parse::<llm_harmony::provider::ProviderKind>() {
            Ok(kind) => request.provider = Some(kind),
            Err(e) => {
                eprintln!("llm-harmony: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let (config, machine) = match setup(config_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("llm-harmony: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Generous: a load is slow, and the read timeouts that protect `status`
    // would abandon a perfectly healthy one.
    let http = Http::new(Duration::from_secs(30));
    let options = llm_harmony::actuate::run::Options {
        reserve_bytes: reserve.unwrap_or(llm_harmony::render_estimate::DEFAULT_RESERVE_BYTES),
        floor_bytes: floor,
        pin_after: pin,
        ..Default::default()
    };

    let corpus = llm_harmony::estimate::corpus::load_default();
    let inventory = llm_harmony::inventory::Inventory::scan_offline(None);
    let model = request.model.clone();
    let context = request.context_tokens;
    let estimate = |ledger: &Ledger, candidates: &[llm_harmony::resolve::identity::Candidate]| {
        estimate_for(&config, ledger, &inventory, candidates, &model, context, &corpus)
    };

    let report = llm_harmony::actuate::run::load_or_switch(
        verb, &config, &http, machine, &request, &estimate, &options,
    );
    emit(&report, json)
}

/// What a model costs, by the ladder in `design.md` section 5: measured, then
/// computed, then a declared floor.
///
/// Shared by `estimate` and `resolve` so the two verbs can never disagree --
/// the number one prints is the number the other admits on.
#[allow(clippy::too_many_arguments)]
fn estimate_for(
    config: &Config,
    ledger: &Ledger,
    inventory: &llm_harmony::inventory::Inventory,
    candidates: &[llm_harmony::resolve::identity::Candidate],
    model: &str,
    context: Option<u32>,
    corpus: &[llm_harmony::record::Observation],
) -> llm_harmony::estimate::estimator::Estimate {
    // A declared floor beats a refusal when nothing has been measured.
    // Provider figures first (vLLM-MLX publishes memory_gb, Ollama
    // size), then the artifact's own size from the disk ledger.
    let declared = candidates.iter().find_map(|c| {
        ledger.rows.iter().find(|r| r.kind == c.provider).and_then(|r| {
            match &r.outcome {
                llm_harmony::ledger::Outcome::Ok(ms) => ms
                    .iter()
                    .find(|m| m.id == c.provider_model_id)
                    .and_then(|m| m.weights_bytes),
                _ => None,
            }
        }).or_else(|| {
            c.artifact.as_ref().and_then(|id| {
                inventory.artifacts.iter().find(|a| &a.id == id).map(|a| a.bytes)
            })
        })
    });
    // The computed rung: the artifact's own geometry, priced at the
    // window that was asked for. Without `--context`, the model's
    // trained window is used -- the same "worst case seen" convention
    // `for_model` already applies to a `None` context, and the
    // direction `design.md` section 8 requires being wrong in.
    let computed = candidates
        .iter()
        .find_map(|c| {
            let id = c.artifact.as_ref()?;
            // The artifact the provider named, if it is itself a model file --
            // Ollama's extensionless blobs are, and no format table would say
            // so. Only when it is not do we go looking for the weights beside
            // it, which is the LM Studio `config.json` case.
            let given = inventory.artifacts.iter().find(|a| &a.id == id);
            let shape = given
                .and_then(|a| llm_harmony::estimate::shape::from_artifact(&a.path))
                .or_else(|| {
                    let a = inventory.weights_artifact(id)?;
                    llm_harmony::estimate::shape::from_artifact(&a.path)
                })?;
            let window = context.or(shape.trained_context)?;
            // f16 unless this provider declared a quantised cache.
            let kv_dtype = config
                .providers
                .iter()
                .find(|p| p.kind == c.provider)
                .and_then(|p| p.kv_dtype_bytes)
                .unwrap_or(llm_harmony::estimate::computed::KV_DTYPE_BYTES);
            Some(llm_harmony::estimate::computed::computed(&shape, window, kv_dtype))
        })
        .unwrap_or(llm_harmony::estimate::estimator::Estimate {
            bytes: None,
            basis: llm_harmony::estimate::estimator::Basis::Unknown,
            samples: 0,
            spread_bytes: None,
        });

    llm_harmony::estimate::estimator::ladder(
        llm_harmony::estimate::estimator::for_model(corpus, model, context),
        computed,
        declared,
    )
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Status { json, config, timeout_ms, record, record_path } => {
            // Only two things may fail the command: a broken config and an
            // unreadable machine total. No provider can.
            let config = match Config::load(config.as_deref()) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let machine = match Machine::read() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };

            let http = Http::new(Duration::from_millis(timeout_ms));
            let ledger = Ledger::assemble(&config, &http, machine);
            let pins = llm_harmony::pins::Pins::load();

            if record {
                // Recording must never fail the command: it is data
                // collection, not the job.
                let path = record_path.or_else(llm_harmony::record::default_path);
                if let Some(path) = path {
                    let obs = llm_harmony::record::observations(
                        &ledger,
                        llm_harmony::record::now_unix(),
                    );
                    let n = obs.len();
                    match llm_harmony::record::append(&path, &obs) {
                        Ok(()) if !json => {
                            eprintln!("recorded {n} observation(s) to {}", path.display())
                        }
                        Ok(()) => {}
                        Err(e) => eprintln!("llm-harmony: could not record: {e}"),
                    }
                }
            }

            if json {
                println!("{}", render_json(&ledger, &pins));
            } else {
                print!("{}", render_table(&ledger, &pins));
            }
            ExitCode::SUCCESS
        }
        Command::Load { model, context, provider, ttl, pin, reserve, floor, config, json } => {
            let request = llm_harmony::actuate::plan::Request {
                model,
                context_tokens: context,
                free_first: None,
                provider: None,
                ttl_seconds: ttl,
            };
            actuate("load", request, provider, pin, reserve, floor, config.as_deref(), json)
        }
        Command::Switch { from, to, context, provider, ttl, pin, reserve, floor, config, json } => {
            let request = llm_harmony::actuate::plan::Request {
                model: to,
                context_tokens: context,
                free_first: Some(from),
                provider: None,
                ttl_seconds: ttl,
            };
            actuate("switch", request, provider, pin, reserve, floor, config.as_deref(), json)
        }
        Command::Unload { model, provider, config, json } => {
            let kind = match provider.as_deref().map(str::parse::<llm_harmony::provider::ProviderKind>) {
                Some(Ok(k)) => Some(k),
                Some(Err(e)) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
                None => None,
            };
            let (cfg, machine) = match setup(config.as_deref()) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let http = Http::new(Duration::from_secs(30));
            let options = llm_harmony::actuate::run::Options::default();
            let report =
                llm_harmony::actuate::run::unload(&cfg, &http, machine, &model, kind, &options);
            emit(&report, json)
        }
        Command::Pin { model, provider, note } => {
            let targets = match pin_targets(&model, provider.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let mut pins = llm_harmony::pins::Pins::load();
            let at = llm_harmony::record::now_unix();
            let mut added = 0;
            for (kind, id) in &targets {
                if pins.add(llm_harmony::pins::Pin {
                    provider: *kind,
                    model: id.clone(),
                    at,
                    note: note.clone(),
                }) {
                    println!("pinned {id} on {}", kind.as_str());
                    added += 1;
                } else {
                    println!("{id} on {} was already pinned", kind.as_str());
                }
            }
            if added > 0 {
                if let Err(e) = pins.save() {
                    eprintln!("llm-harmony: could not save pins: {e}");
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Command::Unpin { model, provider } => {
            let targets = match pin_targets(&model, provider.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let mut pins = llm_harmony::pins::Pins::load();
            let mut removed = 0;
            for (kind, id) in &targets {
                if pins.remove(*kind, id) {
                    println!("unpinned {id} on {}", kind.as_str());
                    removed += 1;
                }
            }
            if removed == 0 {
                println!("nothing was pinned for `{model}`");
                return ExitCode::SUCCESS;
            }
            if let Err(e) = pins.save() {
                eprintln!("llm-harmony: could not save pins: {e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Command::Verify { json, config, timeout_ms } => {
            // The answer to "why did it refuse to unload that?", in one place.
            // Two of the four providers publish no model-level verb at all,
            // and that was only ever discovered by probing -- so this prints
            // what the servers say, never what the docs remember.
            let config = match Config::load(config.as_deref()) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let http = Http::new(Duration::from_millis(timeout_ms));
            let rows: Vec<_> = config
                .providers
                .iter()
                .map(|p| {
                    let adapter = llm_harmony::adapters::adapter_for(p.kind);
                    let reachable = adapter.probe(&http, &p.url).is_ok();
                    (p.kind, p.url.clone(), reachable, adapter.actuation())
                })
                .collect();

            if json {
                let docs: Vec<_> = rows
                    .iter()
                    .map(|(kind, url, reachable, actuation)| {
                        serde_json::json!({
                            "provider": kind.as_str(),
                            "url": url,
                            "reachable": reachable,
                            "actuation": actuation,
                        })
                    })
                    .collect();
                let doc = serde_json::json!({ "schema": 1, "providers": docs });
                println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            } else {
                print!("{}", llm_harmony::render::render_verify(&rows));
            }
            ExitCode::SUCCESS
        }
        Command::Ls { json, live, duplicates } => {
            let inv = inventory(live);
            if duplicates {
                let groups = llm_harmony::inventory::dupes::duplicates(&inv);
                if json {
                    let doc = serde_json::json!({
                        "schema": llm_harmony::ledger::SCHEMA,
                        "min_bytes": llm_harmony::inventory::dupes::MIN_BYTES,
                        "reclaimable_bytes":
                            llm_harmony::inventory::dupes::reclaimable_bytes(&groups),
                        "groups": groups,
                    });
                    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
                } else {
                    print!(
                        "{}",
                        llm_harmony::render_ls::render_duplicates(
                            &groups,
                            llm_harmony::inventory::dupes::MIN_BYTES
                        )
                    );
                }
                return ExitCode::SUCCESS;
            }
            if json {
                let doc = serde_json::json!({
                    "schema": llm_harmony::ledger::SCHEMA,
                    "total_bytes": inv.total_bytes(),
                    "artifact_count": inv.artifacts.len(),
                    "models": inv.identities,
                });
                println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            } else {
                print!("{}", llm_harmony::render_ls::render_ls(&inv));
            }
            ExitCode::SUCCESS
        }
        Command::Resolve { model, context, reserve, json } => {
            let config = match Config::load(None) {
                Ok(c) => c,
                Err(e) => { eprintln!("llm-harmony: {e}"); return ExitCode::FAILURE; }
            };
            let machine = match Machine::read() {
                Ok(m) => m,
                Err(e) => { eprintln!("llm-harmony: {e}"); return ExitCode::FAILURE; }
            };
            let reserve = reserve
                .unwrap_or(llm_harmony::render_estimate::DEFAULT_RESERVE_BYTES);
            let http = Http::new(Duration::from_millis(1500));
            let ledger = Ledger::assemble(&config, &http, machine);
            let inventory = llm_harmony::inventory::Inventory::scan_offline(None);

            let candidates =
                llm_harmony::resolve::identity::candidates(&ledger, &inventory, &model);
            let corpus = llm_harmony::estimate::corpus::load_default();
            let est = estimate_for(&config, &ledger, &inventory, &candidates, &model, context, &corpus);

            let decision =
                llm_harmony::resolve::decide::decide(&candidates, &est, &machine, reserve);

            let denied = matches!(decision, llm_harmony::resolve::decide::Decision::Deny { .. });
            if json {
                let doc = serde_json::json!({
                    // Its own schema: Delroy pins this, and slice 3 already
                    // learned what sharing a version constant costs.
                    "schema": 1,
                    "model": model,
                    "decision": decision,
                    "estimate": est,
                });
                println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            } else {
                print!(
                    "{}",
                    llm_harmony::render_resolve::render(&decision, &est, &machine, reserve)
                );
            }
            if denied { ExitCode::FAILURE } else { ExitCode::SUCCESS }
        }
        Command::Install { provider, dry_run } => {
            let p = match provider_config(&provider) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            // The environment the user would have run it in. A launchd agent
            // inherits nothing, so capture it here rather than hoping.
            let path_env = std::env::var("PATH").unwrap_or_default();
            let xml = match llm_harmony::launch::plist::for_provider(&p, &path_env) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let Some(dest) = llm_harmony::launch::plist::path(&p.label()) else {
                eprintln!("llm-harmony: cannot locate ~/Library/LaunchAgents");
                return ExitCode::FAILURE;
            };
            if dry_run {
                println!("{xml}");
                eprintln!("would write {}", dest.display());
                return ExitCode::SUCCESS;
            }
            // The one write outside harmony's own state directory. Never
            // silent: the destination is printed before the write happens.
            eprintln!("writing {}", dest.display());
            if let Some(parent) = dest.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            }
            if let Err(e) = std::fs::write(&dest, xml) {
                eprintln!("llm-harmony: {e}");
                return ExitCode::FAILURE;
            }
            match llm_harmony::launch::launchctl::bootstrap(&dest) {
                Ok(()) => {
                    println!("installed {}", p.label());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Start { provider, timeout_s } => {
            let p = match provider_config(&provider) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let label = p.label();
            if let Err(e) = llm_harmony::launch::launchctl::kickstart(&label) {
                eprintln!("llm-harmony: {e}");
                eprintln!("  has it been installed? try `llm-harmony install {provider}`");
                return ExitCode::FAILURE;
            }
            let http = Http::new(Duration::from_millis(1500));
            let adapter = llm_harmony::adapters::adapter_for(p.kind);
            match llm_harmony::launch::ready::wait_for(
                adapter.as_ref(),
                &http,
                &p.url,
                Duration::from_secs(timeout_s),
            ) {
                Ok(took) => {
                    println!("{} answered after {:.1}s", p.kind, took.as_secs_f64());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    // Launched is not serving. The reason is in the logs the
                    // agent writes, so name them rather than guessing.
                    eprintln!("llm-harmony: {e}");
                    if let Some(code) = llm_harmony::launch::launchctl::last_exit_code(&label) {
                        eprintln!("  the start command exited {code}{}", match code {
                            127 => " (command not found -- was it installed with the right PATH?)",
                            126 => " (not executable)",
                            _ => "",
                        });
                    }
                    eprintln!("  logs: ~/.local/state/llm-harmony/{label}.err.log");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Stop { provider } => {
            let p = match provider_config(&provider) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match llm_harmony::launch::launchctl::bootout(&p.label()) {
                Ok(()) => {
                    println!("stopped {}", p.label());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    eprintln!("  a provider harmony did not install is not harmony's to stop");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Estimate { model, context, reserve, json } => {
            let machine = match Machine::read() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let corpus = llm_harmony::estimate::corpus::load_default();
            // The same ladder `resolve` admits on, so the two verbs can never
            // disagree about what a model costs. Polling costs ~10 ms and buys
            // the computed rung, which is the whole point of asking.
            let config = match Config::load(None) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let http = Http::new(Duration::from_millis(1500));
            let ledger = Ledger::assemble(&config, &http, machine);
            let inventory = llm_harmony::inventory::Inventory::scan_offline(None);
            let candidates =
                llm_harmony::resolve::identity::candidates(&ledger, &inventory, &model);
            let est =
                estimate_for(&config, &ledger, &inventory, &candidates, &model, context, &corpus);
            if json {
                println!("{}", serde_json::to_string_pretty(&est).unwrap());
            } else {
                print!(
                    "{}",
                    llm_harmony::render_estimate::render_estimate(&est, &machine, reserve)
                );
            }
            ExitCode::SUCCESS
        }
        Command::Rm { model, dry_run, live } => {
            if !dry_run {
                eprintln!("llm-harmony: real removal is not implemented in this slice");
                return ExitCode::FAILURE;
            }
            let inv = inventory(live);
            let want = llm_harmony::inventory::identity::canonical_name(&model);
            let arts: Vec<_> = inv
                .artifacts
                .iter()
                .filter(|a| llm_harmony::inventory::identity::canonical_name(&a.name_hint) == want)
                .cloned()
                .collect();
            if arts.is_empty() {
                eprintln!("llm-harmony: no artifacts match `{model}`");
                return ExitCode::FAILURE;
            }
            let plan = llm_harmony::inventory::plan::build_plan(arts, &inv.graph);
            print!("{}", llm_harmony::render_ls::render_plan(&plan));
            ExitCode::SUCCESS
        }
        Command::Clean { redundant, dry_run, live, json } => {
            // No default selector. `--redundant` is the safest class there is
            // and still the one the user has to name, so adding
            // `--reproducible` later cannot silently widen an old command.
            if !redundant {
                eprintln!("llm-harmony: nothing selected \u{2014} pass --redundant");
                return ExitCode::FAILURE;
            }
            if !dry_run {
                eprintln!("llm-harmony: real removal is not implemented in this slice");
                return ExitCode::FAILURE;
            }
            let inv = inventory(live);
            let plan = llm_harmony::inventory::clean::redundant(&inv);
            if json {
                let doc = serde_json::json!({
                    "schema": llm_harmony::ledger::SCHEMA,
                    "selector": "redundant",
                    "reclaims_bytes": plan.reclaims_bytes,
                    "artifact_count": plan.artifact_count(),
                    "models": plan.models,
                    "skipped": plan.skipped,
                });
                println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            } else {
                print!("{}", llm_harmony::render_ls::render_clean(&plan));
            }
            ExitCode::SUCCESS
        }
    }
}
