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
                println!("{}", render_json(&ledger));
            } else {
                print!("{}", render_table(&ledger));
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
        Command::Ls { json, live } => {
            let inv = inventory(live);
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
            let est = llm_harmony::estimate::estimator::with_declared(
                llm_harmony::estimate::estimator::for_model(&corpus, &model, context),
                declared,
            );
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
            let est = llm_harmony::estimate::estimator::for_model(&corpus, &model, context);
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
    }
}
