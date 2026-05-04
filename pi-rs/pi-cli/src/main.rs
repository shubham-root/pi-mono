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
use pi_core::skills::Skill;
use pi_core::system_prompt::{compose_prompt_and_skills, describe_host_os, ComposePromptOptions};
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
/// Priority order (highest to lowest):
///
///   1. `--model <id>` flag on the command line. Always wins for the
///      current session; does NOT get persisted as the new default
///      unless the user also runs `/model` at runtime to save it.
///   2. `.pi/config.toml` in the current working directory (project-
///      pinned model). Overrides the global config so a repo can pin
///      a model for everyone who opens pi there.
///   3. `~/Library/Application Support/pi/config.toml` on macOS
///      (`~/.config/pi/config.toml` on Linux) — the global
///      last-used model. Written by `/model` when there's no
///      project `.pi/` folder.
///   4. `DEFAULT_PROVIDER_PRIORITY` scan for the first provider with
///      a configured env key.
///   5. Any registry-known provider with a configured env key.
///
/// If all five produce nothing, strict-mode exits; otherwise the TUI
/// starts with a warning and falls back to an arbitrary model id so
/// the UI can still render.
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

    // 1. Explicit --model flag wins for the current session. We do
    //    NOT persist it to config — that's the job of `/model` at
    //    runtime. Passing `--model X` for a one-off session should
    //    not rewrite the saved default.
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
    //    session. Project `.pi/config.toml` (in cwd) wins over the
    //    global config so a repo can pin a model for everyone who
    //    opens pi in it. If the resolved id is still in the registry
    //    and still has auth configured, use it.
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    if let Ok(settings) = pi_core::settings::Settings::load_merged(&cwd) {
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

/// Build the system prompt + load skills for the current run.
/// Respects `--system-prompt`, `--skill`, and `--no-skills` flags.
/// Returns `(system_prompt, skills)` so the caller can feed the
/// prompt into the agent and the skill list into the interactive
/// mode (for `/skill:<name>` expansion).
fn build_system_prompt_and_skills(cli: &Cli) -> (String, Vec<Skill>) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

    let cli_skill_paths: Vec<std::path::PathBuf> = cli
        .skill
        .iter()
        .map(|s| pi_core::resource_loader::expand_user(s))
        .collect();

    // Tool lines are built per-agent, not here, because the CLI
    // hands a strongly-typed tool list to `Agent::with_tool`. We
    // mirror the default list here so the system prompt has
    // accurate "Available tools" rows.
    let tool_lines = if cli.no_tools {
        Vec::new()
    } else {
        default_tool_lines()
    };

    let composed = compose_prompt_and_skills(ComposePromptOptions {
        cwd,
        cli_skill_paths,
        no_skills: cli.no_skills,
        system_prompt_override: cli.system_prompt.clone(),
        os_info: describe_host_os(),
        custom_instructions: None,
        include_date: true,
        tool_lines,
    });

    for d in &composed.diagnostics {
        eprintln!(
            "warning: skill {}: {} ({})",
            match d.kind {
                pi_core::skills::DiagnosticKind::Warning => "warn",
                pi_core::skills::DiagnosticKind::Collision => "collision",
            },
            d.message,
            d.path.display(),
        );
    }

    (composed.prompt, composed.skills)
}

/// Mirror `main()`'s hard-coded tool set so the system prompt's
/// "Available tools" section lists exactly what the agent gets.
fn default_tool_lines() -> Vec<String> {
    use pi_tools::Tool;
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(BashTool),
        Box::new(ReadTool),
        Box::new(WriteTool),
        Box::new(EditTool),
        Box::new(GrepTool),
        Box::new(FindTool),
        Box::new(LsTool),
    ];
    tools
        .iter()
        .map(|t| {
            let desc = t.description();
            let one = desc.lines().next().unwrap_or("").trim();
            format!("- {}: {}", t.name(), one)
        })
        .collect()
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command.clone() {
        Some(Commands::Print { prompt, model }) => {
            let (model_id, api_key) = resolve_model_and_key(model, true);
            let api_key = api_key.expect("resolve_model_and_key(strict=true) should exit if None");

            let (system_prompt, _skills) = build_system_prompt_and_skills(&cli);

            // Create agent with built-in tools
            let mut agent = Agent::new(&model_id)
                .with_api_key(&api_key)
                .with_system_prompt(&system_prompt)
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

            let (model_id, api_key) = resolve_model_and_key(cli.model.clone(), false);

            let (system_prompt, skills) = build_system_prompt_and_skills(&cli);

            let mut agent = Agent::new(&model_id).with_system_prompt(&system_prompt);
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

            // Resolve a resume target if the user asked for one. We
            // accept three shapes:
            //   --continue               -> newest saved session (or new)
            //   --resume <id>            -> specific saved session
            //   --resume                 -> newest session (no picker CLI yet)
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            let resume_session: Option<pi_core::session::Session> = if cli.continue_ {
                match pi_core::session::SessionManager::for_cwd(&cwd) {
                    Ok(mgr) => match mgr.most_recent() {
                        Ok(Some(path)) => match mgr.open_path(&path) {
                            Ok(s) => {
                                info!(
                                    "Continuing session {} ({} messages)",
                                    s.id(),
                                    s.messages().len()
                                );
                                Some(s)
                            }
                            Err(e) => {
                                eprintln!("Warning: --continue failed to open {e}; starting fresh");
                                None
                            }
                        },
                        Ok(None) => None,
                        Err(e) => {
                            eprintln!("Warning: --continue scan failed: {e}");
                            None
                        }
                    },
                    Err(e) => {
                        eprintln!("Warning: session dir unavailable: {e}");
                        None
                    }
                }
            } else if let Some(resume_arg) = cli.resume.clone() {
                match pi_core::session::SessionManager::for_cwd(&cwd) {
                    Ok(mgr) => {
                        let resolved = if resume_arg.is_empty() {
                            mgr.most_recent().ok().flatten()
                        } else {
                            // Allow either a bare id (no extension) or a
                            // full path. If bare, look for <id>.jsonl in
                            // the session dir.
                            let direct = std::path::PathBuf::from(&resume_arg);
                            if direct.is_file() {
                                Some(direct)
                            } else {
                                Some(mgr.dir().join(format!("{resume_arg}.jsonl")))
                            }
                        };
                        match resolved {
                            Some(path) if path.exists() => match mgr.open_path(&path) {
                                Ok(s) => {
                                    info!(
                                        "Resuming session {} ({} messages)",
                                        s.id(),
                                        s.messages().len()
                                    );
                                    Some(s)
                                }
                                Err(e) => {
                                    eprintln!("Warning: --resume failed to open {e}");
                                    None
                                }
                            },
                            _ => {
                                if !resume_arg.is_empty() {
                                    eprintln!(
                                        "Warning: no session matching '{resume_arg}' in {}",
                                        mgr.dir().display()
                                    );
                                }
                                None
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: session dir unavailable: {e}");
                        None
                    }
                }
            } else {
                None
            };

            // If we're resuming, align the agent with the saved
            // session's model + message history so the first follow-up
            // turn sees the full prior conversation.
            if let Some(session) = resume_session.as_ref() {
                let saved_model = session.metadata().model.clone();
                if !saved_model.is_empty() {
                    let _ = agent.set_model(&saved_model);
                }
                agent.set_messages(session.messages().to_vec());
            }

            // Launch interactive mode
            match InteractiveMode::new_with_session(agent, resume_session) {
                Ok(mut mode) => {
                    mode.set_skills(skills);
                    let cli_skill_paths: Vec<std::path::PathBuf> = cli
                        .skill
                        .iter()
                        .map(|s| pi_core::resource_loader::expand_user(s))
                        .collect();
                    mode.set_cli_skill_context(
                        cli_skill_paths,
                        cli.no_skills,
                        cli.system_prompt.clone(),
                    );
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
