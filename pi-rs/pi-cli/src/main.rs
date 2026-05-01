//! pi-rs: The CLI entry point for pi, the AI coding agent.

#[macro_use]
extern crate clap_derive;

use clap::{Parser, Subcommand};
use tracing::info;

mod args;
mod client;
mod server;
mod plugin;

use args::{Cli, Commands, ServerSubcommand, PluginSubcommand};
use pi_core::Agent;
use pi_modes::InteractiveMode;
use pi_tools::{BashTool, EditTool, FindTool, GrepTool, LsTool, ReadTool, WriteTool};
use std::env;


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Print { prompt, model }) => {
            // Determine model and API key
            let model_id = model.unwrap_or_else(|| {
                // Auto-detect based on available API keys
                if env::var("ANTHROPIC_API_KEY").is_ok() {
                    "claude-sonnet-4-20250514".to_string()
                } else if env::var("OPENAI_API_KEY").is_ok() {
                    "gpt-4o".to_string()
                } else {
                    eprintln!("Error: No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.");
                    std::process::exit(1);
                }
            });

            // Resolve API key from environment based on model prefix
            let api_key = if model_id.to_lowercase().contains("claude") || model_id.to_lowercase().contains("anthropic") {
                env::var("ANTHROPIC_API_KEY").ok()
            } else {
                env::var("OPENAI_API_KEY").ok()
            };
            let api_key = match api_key {
                Some(k) => k,
                None => {
                    eprintln!("Error: API key not found in environment.");
                    std::process::exit(1);
                }
            };

            // Create agent with built-in tools
            let mut agent = Agent::new(&model_id)
                .with_api_key(&api_key)
                .with_tool(Box::new(BashTool))
                .with_tool(Box::new(ReadTool))
                .with_tool(Box::new(WriteTool))
                .with_tool(Box::new(EditTool))
                .with_tool(Box::new(GrepTool))
                .with_tool(Box::new(FindTool))
                .with_tool(Box::new(LsTool));

            // Run prompt
            match agent.prompt(&prompt).await {
                Ok(response) => {
                    println!("{}", response);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Rpc { session }) => {
            // RPC mode: JSON-RPC over stdin/stdout
            info!("RPC mode: {:?}", session);
            // Will be implemented in Phase 5.7
            println!("Not yet implemented: RPC mode");
        }
        Some(Commands::New { name, cwd, model }) => {
            // Create and attach to a new session (daemon mode)
            info!("New session: {:?}, cwd: {:?}, model: {:?}", name, cwd, model);
            // Will be implemented in Phase 5.3
            println!("Not yet implemented: new session");
        }
        Some(Commands::Attach { name }) => {
            // Attach to an existing session
            info!("Attaching to session: {:?}", name);
            // Will be implemented in Phase 5.3
            println!("Not yet implemented: attach");
        }
        Some(Commands::List) => {
            // List all sessions
            info!("Listing sessions");
            // Will be implemented in Phase 5.3
            println!("Not yet implemented: list");
        }
        Some(Commands::Server { subcommand }) => {
            // Daemon control
            match subcommand {
                ServerSubcommand::Start => {
                    info!("Starting daemon");
                    println!("Not yet implemented: server start");
                }
                ServerSubcommand::Stop => {
                    info!("Stopping daemon");
                    println!("Not yet implemented: server stop");
                }
                ServerSubcommand::Status => {
                    info!("Checking daemon status");
                    println!("Not yet implemented: server status");
                }
                ServerSubcommand::Logs => {
                    info!("Viewing daemon logs");
                    println!("Not yet implemented: server logs");
                }
            }
        }
        Some(Commands::Plugin { subcommand }) => {
            // Plugin management
            match subcommand {
                PluginSubcommand::Install { name } => {
                    info!("Installing plugin: {}", name);
                    println!("Not yet implemented: plugin install");
                }
                PluginSubcommand::List => {
                    info!("Listing plugins");
                    println!("Not yet implemented: plugin list");
                }
                PluginSubcommand::Remove { name } => {
                    info!("Removing plugin: {}", name);
                    println!("Not yet implemented: plugin remove");
                }
            }
        }
        None => {
            // Interactive mode: default
            info!("Starting interactive mode");
            
            // Create a basic agent with tools for interactive mode
            let model_id = cli.model.unwrap_or_else(|| {
                if env::var("ANTHROPIC_API_KEY").is_ok() {
                    "claude-sonnet-4-20250514".to_string()
                } else if env::var("OPENAI_API_KEY").is_ok() {
                    "gpt-4o".to_string()
                } else {
                    eprintln!("Warning: No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for full functionality.");
                    "claude-sonnet-4-20250514".to_string()
                }
            });

            let api_key = if model_id.to_lowercase().contains("claude") || model_id.to_lowercase().contains("anthropic") {
                env::var("ANTHROPIC_API_KEY").ok()
            } else {
                env::var("OPENAI_API_KEY").ok()
            };

            let mut agent = Agent::new(&model_id);
            if let Some(key) = api_key {
                agent = agent.with_api_key(&key);
            }

            // Add tools unless disabled
            if !cli.no_tools {
                agent = agent
                    .with_tool(Box::new(BashTool))
                    .with_tool(Box::new(ReadTool))
                    .with_tool(Box::new(WriteTool))
                    .with_tool(Box::new(EditTool))
                    .with_tool(Box::new(GrepTool))
                    .with_tool(Box::new(FindTool))
                    .with_tool(Box::new(LsTool));
            }

            // Launch interactive mode
            match InteractiveMode::new(agent) {
                Ok(mut mode) => {
                    if let Err(e) = mode.run().await {
                        eprintln!("Interactive mode error: {}", e);
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to start interactive mode: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}
