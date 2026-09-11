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
        /// What each provider is allowed to take, read from its own argv,
        /// environment, API or config -- never from harmony's config, which
        /// deliberately keeps no copy.
        #[arg(long)]
        budgets: bool,
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
    /// Every provider that could serve this model, priced, with the row
    /// `resolve` would pick marked.
    ///
    /// The resolver computes this list and returns one row of it. This prints
    /// all of them, using the same prices and the same decision, so the menu
    /// can never disagree with the verb.
    Compare {
        model: String,
        #[arg(long, value_name = "TOKENS")]
        context: Option<u32>,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Can these models be resident at the same time, and under which
    /// assignment?
    ///
    /// Admission for a set rather than one model. Charges nothing for what is
    /// already resident, charges each provider's own overhead once, and
    /// refuses on a provider's model-count ceiling where the bytes would have
    /// fitted.
    Fit {
        #[arg(required = true)]
        models: Vec<String>,
        #[arg(long, value_name = "TOKENS")]
        context: Option<u32>,
        #[arg(long, value_name = "BYTES")]
        reserve: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Hold a model for a bounded time, so another caller's request cannot
    /// evict it.
    ///
    /// A lease is a pin that expires. Like a pin it is a veto and never a
    /// reservation: it cannot make a model resident and reserves no memory for
    /// one.
    Lease {
        model: String,
        /// Seconds. Harmony's own clock, and unrelated to `load --ttl`, which
        /// sets the *provider's* idle timer -- this one only decides how long
        /// harmony refuses to evict, and touches no server.
        #[arg(long, value_name = "SECONDS")]
        ttl: u64,
        /// Who is holding it. Named in every refusal the lease causes, because
        /// a hold nobody can trace is the failure mode worth preventing.
        #[arg(long)]
        owner: Option<String>,
        /// Lease this provider's spelling only. Without it, every provider
        /// that could serve the model is leased.
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Give a lease back before it expires. Will not clear a pin -- that is
    /// `unpin`.
    Release {
        model: String,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        json: bool,
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
        #[arg(long)]
        json: bool,
    },
    /// Remove that protection.
    Unpin {
        model: String,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        json: bool,
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
        #[arg(long)]
        json: bool,
    },
    /// Start a provider and wait until it answers.
    Start {
        provider: String,
        #[arg(long, default_value_t = 60)]
        timeout_s: u64,
        #[arg(long)]
        json: bool,
    },
    /// Stop a provider harmony installed.
    Stop {
        provider: String,
        #[arg(long)]
        json: bool,
    },
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
    /// Search Hugging Face for builds this machine could serve.
    Search {
        query: String,
        /// Only repos publishing this format.
        #[arg(long)]
        format: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Pull a built artifact from Hugging Face into a provider's store.
    Add {
        /// A Hugging Face repo id, e.g. `TheBloke/Qwen3-14B-GGUF`.
        hf_id: String,
        /// Which build. The smallest loadable one otherwise.
        #[arg(long)]
        file: Option<String>,
        #[arg(long, default_value = "lmstudio")]
        provider: String,
        /// Answer the whole verdict and move nothing.
        #[arg(long)]
        dry_run: bool,
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
        Command::Status { json, config, timeout_ms, record, record_path, budgets } => {
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

            if budgets {
                let rows: Vec<llm_harmony::budget::Budget> = ledger
                    .rows
                    .iter()
                    .map(|r| {
                        let readings = llm_harmony::budget::read(r, &http);
                        llm_harmony::budget::budget(r.kind, &readings)
                    })
                    .collect();
                if json {
                    println!("{}", llm_harmony::render_budget::render_json(&rows, &machine));
                } else {
                    print!("{}", llm_harmony::render_budget::render(&rows, &machine));
                }
                return ExitCode::SUCCESS;
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
        Command::Pin { model, provider, note, json } => {
            let targets = match pin_targets(&model, provider.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("pin", provider.as_deref().unwrap_or(""), e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            };
            let mut pins = llm_harmony::pins::Pins::load();
            let at = llm_harmony::record::now_unix();
            let mut rows = Vec::new();
            for (kind, id) in &targets {
                let changed = pins.add(llm_harmony::pins::Pin {
                    provider: *kind,
                    model: id.clone(),
                    at,
                    note: note.clone(),
                    owner: None,
                    // A pin is a lease with no expiry. `lease` is the verb
                    // that sets one.
                    expires_at: None,
                });
                if !json {
                    if changed {
                        println!("pinned {id} on {}", kind.as_str());
                    } else {
                        println!("{id} on {} was already pinned", kind.as_str());
                    }
                }
                rows.push(llm_harmony::render_lifecycle::Target {
                    provider: kind.as_str().to_string(),
                    model: id.clone(),
                    changed,
                });
            }
            if rows.iter().any(|r| r.changed) {
                if let Err(e) = pins.save() {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("pin", "", format!("could not save pins: {e}")).print();
                    } else {
                        eprintln!("llm-harmony: could not save pins: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            }
            if json {
                llm_harmony::render_lifecycle::Lifecycle::pins("pin", rows, llm_harmony::render_lifecycle::Outcome::Pinned).print();
            }
            ExitCode::SUCCESS
        }
        Command::Unpin { model, provider, json } => {
            let targets = match pin_targets(&model, provider.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("unpin", provider.as_deref().unwrap_or(""), e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            };
            let mut pins = llm_harmony::pins::Pins::load();
            let mut rows = Vec::new();
            for (kind, id) in &targets {
                let changed = pins.remove(*kind, id);
                if changed && !json {
                    println!("unpinned {id} on {}", kind.as_str());
                }
                rows.push(llm_harmony::render_lifecycle::Target {
                    provider: kind.as_str().to_string(),
                    model: id.clone(),
                    changed,
                });
            }
            let removed = rows.iter().filter(|r| r.changed).count();
            if removed == 0 {
                // Asking for a state you are already in is a success. A UI
                // that clears a pin twice should hear "nothing changed", not
                // an error it has to explain to someone.
                if json {
                    llm_harmony::render_lifecycle::Lifecycle::pins("unpin", rows, llm_harmony::render_lifecycle::Outcome::NotPinned).print();
                } else {
                    println!("nothing was pinned for `{model}`");
                }
                return ExitCode::SUCCESS;
            }
            if let Err(e) = pins.save() {
                if json {
                    llm_harmony::render_lifecycle::Lifecycle::failed("unpin", "", format!("could not save pins: {e}")).print();
                } else {
                    eprintln!("llm-harmony: could not save pins: {e}");
                }
                return ExitCode::FAILURE;
            }
            if json {
                llm_harmony::render_lifecycle::Lifecycle::pins("unpin", rows, llm_harmony::render_lifecycle::Outcome::Unpinned).print();
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&llm_harmony::render_ls::document(&inv)).unwrap()
                );
            } else {
                print!("{}", llm_harmony::render_ls::render_ls(&inv));
            }
            ExitCode::SUCCESS
        }
        Command::Compare { model, context, reserve, json } => {
            let (config, machine) = match setup(None) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let reserve =
                reserve.unwrap_or(llm_harmony::render_estimate::DEFAULT_RESERVE_BYTES);
            let http = Http::new(Duration::from_millis(1500));
            let ledger = Ledger::assemble(&config, &http, machine);
            let inventory = llm_harmony::inventory::Inventory::scan_offline(None);
            let candidates =
                llm_harmony::resolve::identity::candidates(&ledger, &inventory, &model);
            let corpus = llm_harmony::estimate::corpus::load_default();

            // One price per candidate, through the ladder `estimate` and
            // `resolve` already share -- a one-element slice, so no second
            // pricing path can drift from the first.
            let priced: Vec<llm_harmony::render_compare::Priced> = candidates
                .iter()
                .map(|c| llm_harmony::render_compare::Priced {
                    candidate: c.clone(),
                    estimate: estimate_for(
                        &config,
                        &ledger,
                        &inventory,
                        std::slice::from_ref(c),
                        &model,
                        context,
                        &corpus,
                    ),
                })
                .collect();

            // And the pick comes from `decide` rather than from any ranking
            // applied here.
            let est = estimate_for(
                &config, &ledger, &inventory, &candidates, &model, context, &corpus,
            );
            let decision =
                llm_harmony::resolve::decide::decide(&candidates, &est, &machine, reserve);

            if json {
                println!(
                    "{}",
                    llm_harmony::render_compare::render_json(
                        &model, &priced, &machine, reserve, &decision
                    )
                );
            } else {
                print!(
                    "{}",
                    llm_harmony::render_compare::render(
                        &model, &priced, &machine, reserve, &decision
                    )
                );
            }
            // `compare` reports; it admits nothing, so it refuses nothing.
            ExitCode::SUCCESS
        }
        Command::Fit { models, context, reserve, json } => {
            let (config, machine) = match setup(None) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let reserve =
                reserve.unwrap_or(llm_harmony::render_estimate::DEFAULT_RESERVE_BYTES);
            let http = Http::new(Duration::from_millis(1500));
            let ledger = Ledger::assemble(&config, &http, machine);
            let inventory = llm_harmony::inventory::Inventory::scan_offline(None);
            let corpus = llm_harmony::estimate::corpus::load_default();
            let baselines = llm_harmony::estimate::baseline::all(&corpus);
            let budgets: Vec<llm_harmony::budget::Budget> = ledger
                .rows
                .iter()
                .map(|r| {
                    let readings = llm_harmony::budget::read(r, &http);
                    llm_harmony::budget::budget(r.kind, &readings)
                })
                .collect();
            let running: Vec<llm_harmony::provider::ProviderKind> = ledger
                .rows
                .iter()
                .filter(|r| matches!(r.outcome, llm_harmony::ledger::Outcome::Ok(_)))
                .map(|r| r.kind)
                .collect();
            let pins = llm_harmony::pins::Pins::load();

            let cands =
                |m: &str| llm_harmony::resolve::identity::candidates(&ledger, &inventory, m);
            let price = |c: &llm_harmony::resolve::identity::Candidate, m: &str| {
                estimate_for(
                    &config,
                    &ledger,
                    &inventory,
                    std::slice::from_ref(c),
                    m,
                    context,
                    &corpus,
                )
            };
            let inputs = llm_harmony::fit::Inputs {
                candidates: &cands,
                price: &price,
                baselines: &baselines,
                budgets: &budgets,
                running: &running,
                machine: &machine,
                reserve_bytes: reserve,
                pins: &pins,
                now: llm_harmony::record::now_unix(),
            };
            let plan = llm_harmony::fit::plan(&models, &inputs);
            let fits = plan.verdict == llm_harmony::fit::Verdict::Fits;
            if json {
                println!("{}", llm_harmony::render_fit::render_json(&models, &plan));
            } else {
                print!("{}", llm_harmony::render_fit::render(&plan));
            }
            // A caller has to be able to tell "this set is possible" from
            // "it is not" without parsing prose, as `resolve` already allows.
            if fits { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Command::Lease { model, ttl, owner, provider, json } => {
            let targets = match pin_targets(&model, provider.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let mut pins = llm_harmony::pins::Pins::load();
            let now = llm_harmony::record::now_unix();
            pins.sweep(now);
            let expires_at = now + ttl;
            let mut rows = Vec::new();
            let mut refused = false;
            for (kind, id) in &targets {
                let outcome = pins.lease(llm_harmony::pins::Pin {
                    provider: *kind,
                    model: id.clone(),
                    at: now,
                    note: None,
                    owner: owner.clone(),
                    expires_at: Some(expires_at),
                });
                use llm_harmony::pins::LeaseOutcome::*;
                if outcome == RefusedPinned {
                    refused = true;
                }
                if !json {
                    let left = llm_harmony::pins::human_seconds(ttl);
                    match outcome {
                        Taken => println!("leased {id} on {} for {left}", kind.as_str()),
                        Extended => println!("extended the lease on {id} on {} to {left}", kind.as_str()),
                        AlreadyLonger => println!("{id} on {} is already leased for longer", kind.as_str()),
                        RefusedPinned => println!(
                            "{id} on {} is pinned; leaving the pin alone (a lease would be a downgrade)",
                            kind.as_str()
                        ),
                    }
                }
                rows.push(serde_json::json!({
                    "provider": kind.as_str(),
                    "model": id,
                    "outcome": format!("{outcome:?}").to_lowercase(),
                }));
            }
            if let Err(e) = pins.save() {
                eprintln!("llm-harmony: could not save pins: {e}");
                return ExitCode::FAILURE;
            }
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "schema": 1,
                        "verb": "lease",
                        "expires_at": expires_at,
                        "owner": owner,
                        "targets": rows,
                    })
                );
            }
            // Every target refused means nothing was leased, which a caller
            // must be able to see in the exit code.
            if refused && rows.len() == 1 { ExitCode::FAILURE } else { ExitCode::SUCCESS }
        }
        Command::Release { model, provider, json } => {
            let targets = match pin_targets(&model, provider.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let mut pins = llm_harmony::pins::Pins::load();
            let now = llm_harmony::record::now_unix();
            pins.sweep(now);
            let mut rows = Vec::new();
            let mut released = 0usize;
            for (kind, id) in &targets {
                let outcome = pins.release(*kind, id);
                use llm_harmony::pins::ReleaseOutcome::*;
                if outcome == Released {
                    released += 1;
                }
                if !json {
                    match outcome {
                        Released => println!("released {id} on {}", kind.as_str()),
                        NotHeld => {}
                        WasPinned => println!(
                            "{id} on {} is pinned, not leased -- use `unpin`",
                            kind.as_str()
                        ),
                    }
                }
                rows.push(serde_json::json!({
                    "provider": kind.as_str(),
                    "model": id,
                    "outcome": format!("{outcome:?}").to_lowercase(),
                }));
            }
            if released > 0 {
                if let Err(e) = pins.save() {
                    eprintln!("llm-harmony: could not save pins: {e}");
                    return ExitCode::FAILURE;
                }
            } else if !json {
                println!("nothing was leased for `{model}`");
            }
            if json {
                println!(
                    "{}",
                    serde_json::json!({"schema": 1, "verb": "release", "targets": rows})
                );
            }
            // Asking for a state you are already in is a success, the posture
            // `unpin` already takes.
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
        Command::Install { provider, dry_run, json } => {
            let p = match provider_config(&provider) {
                Ok(p) => p,
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("install", &provider, e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            };
            // The environment the user would have run it in. A launchd agent
            // inherits nothing, so capture it here rather than hoping.
            let path_env = std::env::var("PATH").unwrap_or_default();
            let xml = match llm_harmony::launch::plist::for_provider(&p, &path_env) {
                Ok(x) => x,
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("install", &provider, e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            };
            let Some(dest) = llm_harmony::launch::plist::path(&p.label()) else {
                let msg = "cannot locate ~/Library/LaunchAgents";
                if json {
                    llm_harmony::render_lifecycle::Lifecycle::failed("install", &provider, msg).print();
                } else {
                    eprintln!("llm-harmony: {msg}");
                }
                return ExitCode::FAILURE;
            };
            if dry_run {
                if json {
                    // `changed: false` is the whole point of a dry run, and it
                    // is the field a caller checks rather than the verb name.
                    llm_harmony::render_lifecycle::Lifecycle::provider(
                        "install",
                        &provider,
                        llm_harmony::render_lifecycle::Outcome::Planned {
                            path: dest.display().to_string(),
                            plist: xml,
                        },
                        false,
                    )
                    .print();
                } else {
                    println!("{xml}");
                    eprintln!("would write {}", dest.display());
                }
                return ExitCode::SUCCESS;
            }
            // The one write outside harmony's own state directory. Never
            // silent: the destination is printed before the write happens.
            if !json {
                eprintln!("writing {}", dest.display());
            }
            if let Some(parent) = dest.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("install", &provider, e.to_string()).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            }
            if let Err(e) = std::fs::write(&dest, xml) {
                if json {
                    llm_harmony::render_lifecycle::Lifecycle::failed("install", &provider, e.to_string()).print();
                } else {
                    eprintln!("llm-harmony: {e}");
                }
                return ExitCode::FAILURE;
            }
            match llm_harmony::launch::launchctl::bootstrap(&dest) {
                Ok(()) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::provider(
                            "install",
                            &provider,
                            llm_harmony::render_lifecycle::Outcome::Installed { path: dest.display().to_string() },
                            true,
                        )
                        .print();
                    } else {
                        println!("installed {}", p.label());
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("install", &provider, e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    ExitCode::FAILURE
                }
            }
        }
        Command::Start { provider, timeout_s, json } => {
            let p = match provider_config(&provider) {
                Ok(p) => p,
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("start", &provider, e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            };
            let label = p.label();
            // `start`, not `kickstart`: `stop` boots the service out of the
            // domain entirely, so the obvious stop/start pair needs the agent
            // bootstrapped again first. See `launchctl::start`.
            let plist = llm_harmony::launch::plist::path(&label).unwrap_or_default();
            if let Err(e) = llm_harmony::launch::launchctl::start(&label, &plist) {
                let hint = format!("{e}; has it been installed? try `llm-harmony install {provider}`");
                if json {
                    llm_harmony::render_lifecycle::Lifecycle::failed("start", &provider, hint).print();
                } else {
                    eprintln!("llm-harmony: {e}");
                    eprintln!("  has it been installed? try `llm-harmony install {provider}`");
                }
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
                    if json {
                        // "Started" and "answering" are not the same claim.
                        // Only the second one is reported as ready, and the
                        // wait is carried so a UI shows what it cost.
                        llm_harmony::render_lifecycle::Lifecycle::provider(
                            "start",
                            &provider,
                            llm_harmony::render_lifecycle::Outcome::Ready {
                                url: p.url.clone(),
                                took_s: took.as_secs(),
                            },
                            true,
                        )
                        .print();
                    } else {
                        println!("{} answered after {:.1}s", p.kind, took.as_secs_f64());
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    // Launched is not serving. The reason is in the logs the
                    // agent writes, so name them rather than guessing.
                    let code = llm_harmony::launch::launchctl::last_exit_code(&label);
                    let log = format!("~/.local/state/llm-harmony/{label}.err.log");
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::provider(
                            "start",
                            &provider,
                            llm_harmony::render_lifecycle::Outcome::Failed {
                                reason: e,
                                exit_code: code,
                                log: Some(log),
                            },
                            false,
                        )
                        .print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                        if let Some(code) = code {
                            eprintln!("  the start command exited {code}{}", match code {
                                127 => " (command not found -- was it installed with the right PATH?)",
                                126 => " (not executable)",
                                _ => "",
                            });
                        }
                        eprintln!("  logs: {log}");
                    }
                    ExitCode::FAILURE
                }
            }
        }
        Command::Stop { provider, json } => {
            let p = match provider_config(&provider) {
                Ok(p) => p,
                Err(e) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("stop", &provider, e).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                    }
                    return ExitCode::FAILURE;
                }
            };
            match llm_harmony::launch::launchctl::bootout(&p.label()) {
                Ok(()) => {
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::provider("stop", &provider, llm_harmony::render_lifecycle::Outcome::Stopped, true)
                            .print();
                    } else {
                        println!("stopped {}", p.label());
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    // Passed through rather than flattened: the page can say
                    // *why* instead of "failed".
                    let reason =
                        format!("{e}; a provider harmony did not install is not harmony's to stop");
                    if json {
                        llm_harmony::render_lifecycle::Lifecycle::failed("stop", &provider, reason).print();
                    } else {
                        eprintln!("llm-harmony: {e}");
                        eprintln!("  a provider harmony did not install is not harmony's to stop");
                    }
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&llm_harmony::render_estimate::document(&est))
                        .unwrap()
                );
            } else {
                print!(
                    "{}",
                    llm_harmony::render_estimate::render_estimate(&est, &machine, reserve)
                );
            }
            ExitCode::SUCCESS
        }
        Command::Search { query, format, limit, json } => {
            let machine = Machine::read().unwrap_or_else(|_| Machine::zero());
            let http = Http::new(Duration::from_secs(20));
            let mut doc = llm_harmony::intake::run::search(&http, &machine, &query, limit);
            if let Some(want) = format.as_deref() {
                let want = want.to_ascii_lowercase();
                doc.hits.retain(|h| {
                    h.files.iter().any(|f| format!("{:?}", f.format).to_ascii_lowercase() == want)
                });
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            } else {
                for hit in &doc.hits {
                    println!("{}", hit.repo);
                    if hit.needs_conversion {
                        // Shown, not hidden: a build exists and harmony cannot
                        // use it yet, which is different from there being none.
                        println!("  needs conversion (bf16 only)");
                    }
                    if let Some(f) = &hit.fit {
                        println!("  {}", f.message());
                    }
                    if hit.provenance.is_warning() {
                        println!("  ! {}", hit.provenance.message());
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Command::Add { hf_id, file, provider, dry_run, json } => {
            let kind: llm_harmony::provider::ProviderKind = match provider.parse() {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("llm-harmony: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let machine = Machine::read().unwrap_or_else(|_| Machine::zero());
            let http = Http::new(Duration::from_secs(20));
            let mut doc = llm_harmony::intake::run::plan_add(
                &http, &machine, &hf_id, file.as_deref(), kind,
            );

            let refused = matches!(
                doc.outcome,
                llm_harmony::intake::run::AddOutcome::Refused { .. }
            );
            if dry_run || refused {
                if json {
                    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
                } else {
                    for w in &doc.warnings {
                        eprintln!("! {w}");
                    }
                    match &doc.outcome {
                        llm_harmony::intake::run::AddOutcome::Refused { reason } => {
                            eprintln!("refused: {reason}");
                        }
                        _ => {
                            println!("{} -> {}", doc.file, doc.target_dir.clone().unwrap_or_default());
                            if let Some(f) = &doc.fit {
                                println!("{}", f.message());
                            }
                        }
                    }
                }
                return if refused { ExitCode::FAILURE } else { ExitCode::SUCCESS };
            }

            // Past every refusal this verb can make. Only now do bytes move.
            //
            // Which way they move is the one thing the two transports do not
            // share: Ollama takes the repo id through its own registry, every
            // other provider takes a file drop into the store placement chose.
            if let Some(name) = doc.registry_name.clone() {
                let mut emit = |p: llm_harmony::intake::download::Progress| {
                    if json {
                        p.print();
                    } else if let llm_harmony::intake::download::Progress::Advanced {
                        bytes_done, ..
                    } = p
                    {
                        eprintln!("  {}", llm_harmony::render::human_bytes(bytes_done));
                    }
                };
                for w in &doc.warnings {
                    if !json {
                        eprintln!("! {w}");
                    }
                }
                let argv = llm_harmony::intake::ollama::pull_argv(&name);
                match llm_harmony::intake::ollama::run_pull(&argv, &mut emit) {
                    Ok(()) => {
                        doc.outcome =
                            llm_harmony::intake::run::AddOutcome::Added { path: name.clone() };
                        if json {
                            println!("{}", serde_json::to_string_pretty(&doc).unwrap());
                        } else {
                            println!("added {name}");
                        }
                        return ExitCode::SUCCESS;
                    }
                    Err(e) => {
                        doc.outcome =
                            llm_harmony::intake::run::AddOutcome::Refused { reason: e.clone() };
                        if json {
                            println!("{}", serde_json::to_string_pretty(&doc).unwrap());
                        } else {
                            eprintln!("llm-harmony: {e}");
                        }
                        return ExitCode::FAILURE;
                    }
                }
            }
            let dir = std::path::PathBuf::from(doc.target_dir.clone().unwrap_or_default());
            let agent = llm_harmony::intake::download::transfer_agent();
            let mut emit = |p: llm_harmony::intake::download::Progress| {
                if json {
                    p.print();
                } else if let llm_harmony::intake::download::Progress::Advanced {
                    bytes_done, bytes_total,
                } = p
                {
                    let pct = bytes_total
                        .map(|t| format!(" ({}%)", bytes_done * 100 / t.max(1)))
                        .unwrap_or_default();
                    eprintln!("  {}{pct}", llm_harmony::render::human_bytes(bytes_done));
                }
            };
            for w in &doc.warnings {
                if !json {
                    eprintln!("! {w}");
                }
            }
            // Every part, in order. A sharded build is one thing to load and
            // several things to fetch, and stopping after the first would
            // leave a directory that looks populated and loads nothing.
            let mut last = std::path::PathBuf::new();
            for name in &doc.files {
                let url = llm_harmony::intake::hf::download_url(&hf_id, name);
                match llm_harmony::intake::download::fetch(&agent, &url, &dir, name, &mut emit) {
                    Ok(path) => last = path,
                    Err(e) => {
                        let failed = llm_harmony::intake::download::Progress::Failed {
                            file: name.clone(),
                            reason: e.clone(),
                        };
                        if json {
                            failed.print();
                        }
                        doc.outcome =
                            llm_harmony::intake::run::AddOutcome::Refused { reason: e.clone() };
                        if json {
                            println!("{}", serde_json::to_string_pretty(&doc).unwrap());
                        } else {
                            eprintln!("llm-harmony: {e}");
                        }
                        return ExitCode::FAILURE;
                    }
                }
            }
            doc.outcome = llm_harmony::intake::run::AddOutcome::Added {
                path: last.display().to_string(),
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&doc).unwrap());
            } else {
                println!("added {}", last.display());
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
