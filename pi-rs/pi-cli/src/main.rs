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
use pi_core::model_registry::ModelRegistry;
use pi_core::Agent;
use pi_modes::InteractiveMode;
use pi_tools::{BashTool, EditTool, FindTool, GrepTool, LsTool, ReadTool, WriteTool};

/// Priority order when auto-picking a default provider from env vars.
/// AWS credentials come last because `AWS_ACCESS_KEY_ID` / `AWS_PROFILE`
/// are typically present in dev shells for unrelated reasons; picking
/// bedrock alphabetically caused auth failures when the user intended a
/// different provider (see issue with StepFun/OpenRouter being routed to
/// Bedrock). Mirrors the ordering used by the TypeScript version's
/// `defaultModelPerProvider`.
const DEFAULT_PROVIDER_PRIORITY: &[&str] = &[
    "anthropic",
    "openai",
    "openrouter",
    "vercel-ai-gateway",
    "google",
    "google-vertex",
    "xai",
    "groq",
    "cerebras",
    "deepseek",
    "mistral",
    "fireworks",
    "huggingface",
    "zai",
    "github-copilot",
    "opencode",
    "opencode-go",
    "minimax",
    "minimax-cn",
    "kimi-coding",
    "cloudflare-workers-ai",
    "azure-openai-responses",
    "openai-codex",
    "google-gemini-cli",
    "google-antigravity",
    "amazon-bedrock",
];

/// Resolve a `(model_id, api_key)` pair from CLI flags + registry-aware env
/// lookup. `strict = true` exits the process on missing credentials; otherwise
/// a warning is printed and the TUI still starts (useful for read-only UI
/// navigation before auth is configured).
///
/// The returned model id is always fully qualified (`provider_id/model_id`)
/// so downstream `resolve_model` never has to guess which provider owns a
/// bare id that happens to exist under several providers (e.g.
/// `anthropic.claude-opus-4-6-v1` exists under both anthropic and bedrock).
fn resolve_model_and_key(
    cli_model: Option<String>,
    strict: bool,
) -> (String, Option<String>) {
    let registry = ModelRegistry::global();

    // 1. Explicit --model flag: accept either bare or qualified form. Look
    //    up the provider for env-var resolution; keep the user-supplied id
    //    as-is so error messages match what they typed.
    if let Some(model_id) = cli_model {
        let found = if let Some((pid, mid)) = model_id.split_once('/') {
            registry.find_by_provider(pid, mid)
        } else {
            registry.find_model(&model_id)
        };

        let api_key = found.and_then(|(p, _)| p.resolve_env_key().map(|(_, v)| v));
        if api_key.is_none() && strict {
            let hint = found
                .map(|(p, _)| {
                    if p.env_vars.is_empty() {
                        format!("provider '{}' has no env keys configured", p.id)
                    } else {
                        format!("set one of: {}", p.env_vars.join(", "))
                    }
                })
                .unwrap_or_else(|| {
                    "model not in registry; set the matching provider env var".to_string()
                });
            eprintln!("Error: no API key for model '{}': {}", model_id, hint);
            std::process::exit(1);
        }
        return (model_id, api_key);
    }

    // 2. No --model: consult saved settings first — pi remembers the
    //    last-used model so users don't have to retype `--model` every
    //    session. If the saved id is still in the registry and still has
    //    auth configured, use it.
    if let Ok(settings) = pi_core::settings::Settings::load_global() {
        if let Some(saved) = settings.model.as_deref() {
            let found = if let Some((pid, mid)) = saved.split_once('/') {
                registry.find_by_provider(pid, mid)
            } else {
                registry.find_model(saved)
            };
            if let Some((provider, model)) = found {
                if let Some((_, key)) = provider.resolve_env_key() {
                    return (format!("{}/{}", provider.id, model.id), Some(key));
                }
                // Saved id exists but auth is gone; fall through to the
                // priority list rather than insisting on it.
            }
        }
    }

    // 3. No saved model (or it's unusable): walk the priority list and
    //    pick the first provider whose env key is configured. Always
    //    return a qualified id so the agent knows exactly which
    //    provider to dispatch to.
    for pid in DEFAULT_PROVIDER_PRIORITY {
        if let Some(provider) = registry.provider(pid) {
            if let Some((_, key)) = provider.resolve_env_key() {
                if let Some(model) = provider.models.first() {
                    return (format!("{}/{}", provider.id, model.id), Some(key));
                }
            }
        }
    }

    // 3. Fallback: try *any* provider the registry knows about that has an
    //    env key configured (covers community providers added via
    //    ~/.pi/providers/ that aren't in DEFAULT_PROVIDER_PRIORITY).
    for provider in registry.providers() {
        if let Some((_, key)) = provider.resolve_env_key() {
            if let Some(model) = provider.models.first() {
                return (format!("{}/{}", provider.id, model.id), Some(key));
            }
        }
    }

    // 4. Nothing configured.
    let examples: Vec<&str> = DEFAULT_PROVIDER_PRIORITY
        .iter()
        .take(4)
        .filter_map(|pid| registry.provider(pid))
        .flat_map(|p| p.env_vars.iter().map(|s| s.as_str()))
        .take(4)
        .collect();
    let hint = if examples.is_empty() {
        "no providers have env keys configured".to_string()
    } else {
        format!("set one of: {}", examples.join(", "))
    };

    if strict {
        eprintln!("Error: no API key found. {}", hint);
        std::process::exit(1);
    }
    eprintln!("Warning: no API key found. {}. TUI will start in read-only mode.", hint);

    // Fall back to an arbitrary model so the TUI still renders.
    let fallback = registry
        .all_models()
        .next()
        .map(|(p, m)| format!("{}/{}", p.id, m.id))
        .unwrap_or_else(|| "anthropic/claude-sonnet-4-6".to_string());
    (fallback, None)
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Print { prompt, model }) => {
            let (model_id, api_key) = resolve_model_and_key(model, true);
            let api_key = api_key.expect("resolve_model_and_key(strict=true) should exit if None");

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

            let (model_id, api_key) = resolve_model_and_key(cli.model, false);

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
