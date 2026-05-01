//! CLI argument definitions using clap.

use clap::{Parser, Subcommand};

/// Main CLI arguments.
#[derive(Parser, Debug)]
#[clap(name = "pi-rs")]
#[clap(about = "pi - AI coding agent")]
#[clap(long_about = None)]
#[clap(version)]
pub struct Cli {
    /// Subcommand to run
    #[clap(subcommand)]
    pub command: Option<Commands>,

    /// Working directory for the session
    #[clap(short, long, default_value = ".")]
    pub cwd: Option<String>,

    /// Model to use
    #[clap(short, long)]
    pub model: Option<String>,

    /// System prompt override
    #[clap(long)]
    pub system_prompt: Option<String>,

    /// Thinking level (minimal, low, medium, high, xhigh)
    #[clap(long)]
    pub thinking: Option<String>,

    /// Disable all tools
    #[clap(long)]
    pub no_tools: bool,

    /// Disable plugins
    #[clap(long)]
    pub no_plugins: bool,

    /// Additional plugin to load
    #[clap(long)]
    pub plugin: Vec<String>,

    /// Run in single-process mode (no daemon)
    #[clap(long)]
    pub no_daemon: bool,

    /// Print only (non-interactive)
    #[clap(short, long)]
    pub print: bool,

    /// JSON output (for RPC mode)
    #[clap(long)]
    pub json: bool,

    /// Verbose logging
    #[clap(short, long)]
    pub verbose: bool,

    /// Session directory
    #[clap(long)]
    pub session_dir: Option<String>,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Print a single response
    Print {
        /// The prompt text
        prompt: String,
        /// Model to use
        #[clap(short, long)]
        model: Option<String>,
    },

    /// Start a new session
    New {
        /// Session name (auto-generated if not provided)
        name: Option<String>,
        /// Working directory
        #[clap(short, long)]
        cwd: Option<String>,
        /// Model to use
        #[clap(short, long)]
        model: Option<String>,
    },

    /// Attach to an existing session
    Attach {
        /// Session name or ID
        name: String,
    },

    /// List all sessions
    List,

    /// RPC mode: JSON-RPC over stdin/stdout
    Rpc {
        /// Session ID to attach to
        session: Option<String>,
    },

    /// Daemon control
    Server {
        #[clap(subcommand)]
        subcommand: ServerSubcommand,
    },

    /// Plugin management
    Plugin {
        #[clap(subcommand)]
        subcommand: PluginSubcommand,
    },
}

#[derive(Subcommand, Debug)]
pub enum ServerSubcommand {
    Start,
    Stop,
    Status,
    Logs,
}

#[derive(Subcommand, Debug)]
pub enum PluginSubcommand {
    Install {
        name: String,
    },
    List,
    Remove {
        name: String,
    },
}
