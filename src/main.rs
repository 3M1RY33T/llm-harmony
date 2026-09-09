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
