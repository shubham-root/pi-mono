//! System prompt assembly.
//!
//! Builds the full system prompt Pi sends on every turn, mirroring
//! what the TS `coding-agent/src/core/system-prompt.ts` does but
//! trimmed to what the Rust port currently supports:
//!
//!   1. Base instructions (agent identity + operating guidance).
//!   2. Environment block: cwd, OS, current date.
//!   3. Available tools (one line per tool).
//!   4. `<available_skills>` XML block (progressive-disclosure
//!      descriptions).
//!   5. AGENTS.md project instructions when present.
//!   6. Optional custom instructions from settings / CLI.
//!
//! Every block can be omitted — the builder is additive.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use pi_tools::Tool;

use crate::resource_loader::{ResourceLoader, ResourceLoaderOptions};
use crate::settings::Settings;
use crate::skills::{format_skills_for_prompt, LoadSkillsResult, Skill};

/// Assembles the system prompt from a collection of additive
/// building blocks. Every `with_*` method returns `self` for
/// chaining and is safe to skip.
pub struct SystemPromptBuilder {
    cwd: Option<PathBuf>,
    os_info: Option<String>,
    pub(crate) tool_lines: Vec<String>,
    skills: Vec<Skill>,
    custom_instructions: Option<String>,
    include_date: bool,
    agents_md: Option<String>,
}

impl SystemPromptBuilder {
    /// Start with nothing — every block is additive. The base
    /// agent-identity preamble is always emitted by `build`.
    pub fn new() -> Self {
        Self {
            cwd: None,
            os_info: None,
            tool_lines: Vec::new(),
            skills: Vec::new(),
            custom_instructions: None,
            include_date: true,
            agents_md: None,
        }
    }

    /// Record the working directory. Appears in the environment
    /// block so the model can reason about relative paths.
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Short OS description (e.g. `macOS 15.3 (darwin/arm64)`).
    pub fn with_os_info(mut self, os: impl Into<String>) -> Self {
        self.os_info = Some(os.into());
        self
    }

    /// Add the tools the model can call. Each tool contributes a
    /// `- <name>: <description>` line to the prompt.
    pub fn with_tools(mut self, tools: &[&dyn Tool]) -> Self {
        for t in tools {
            self.tool_lines.push(format!(
                "- {}: {}",
                t.name(),
                first_line(&t.description()),
            ));
        }
        self
    }

    /// Skills to expose via progressive disclosure.
    pub fn with_skills(mut self, skills: impl IntoIterator<Item = Skill>) -> Self {
        self.skills.extend(skills);
        self
    }

    /// Extra custom instructions appended after the built-in
    /// blocks. Matches the TS `customInstructions` hook.
    pub fn with_custom_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.custom_instructions = Some(instructions.into());
        self
    }

    /// Include / exclude the `Today is …` date line. Defaults on.
    pub fn with_date(mut self, include: bool) -> Self {
        self.include_date = include;
        self
    }

    /// Inline AGENTS.md style project instructions (the caller is
    /// responsible for reading the file, typically the nearest one
    /// walking up from `cwd`).
    pub fn with_agents_md(mut self, content: impl Into<String>) -> Self {
        self.agents_md = Some(content.into());
        self
    }

    /// Try reading `AGENTS.md` from `cwd` (or its ancestors up to a
    /// `.git` directory). No-op if the file isn't found.
    pub fn with_agents_md_from(mut self, cwd: &Path) -> Self {
        if let Some(content) = find_agents_md(cwd) {
            self.agents_md = Some(content);
        }
        self
    }

    /// Assemble the final prompt. Blocks that aren't configured get
    /// skipped cleanly; an empty builder still produces the base
    /// identity preamble.
    pub fn build(self) -> String {
        let mut out = String::new();
        out.push_str(BASE_INSTRUCTIONS);

        // Environment block.
        let mut env_lines: Vec<String> = Vec::new();
        if let Some(cwd) = &self.cwd {
            env_lines.push(format!("Working directory: {}", cwd.display()));
        }
        if let Some(os) = &self.os_info {
            env_lines.push(format!("Operating system: {os}"));
        }
        if self.include_date {
            env_lines.push(format!("Today is {}", Utc::now().format("%Y-%m-%d")));
        }
        if !env_lines.is_empty() {
            out.push_str("\n\n## Environment\n");
            for line in env_lines {
                out.push_str(&line);
                out.push('\n');
            }
        }

        // Tools.
        if !self.tool_lines.is_empty() {
            out.push_str("\n## Available tools\n");
            for line in &self.tool_lines {
                out.push_str(line);
                out.push('\n');
            }
        }

        // Skills (progressive disclosure).
        let skills_block = format_skills_for_prompt(&self.skills);
        if !skills_block.is_empty() {
            out.push_str(&skills_block);
            out.push('\n');
        }

        // AGENTS.md.
        if let Some(content) = &self.agents_md {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                out.push_str("\n## Project instructions (AGENTS.md)\n");
                out.push_str(trimmed);
                out.push('\n');
            }
        }

        // Caller-provided custom instructions.
        if let Some(custom) = &self.custom_instructions {
            let trimmed = custom.trim();
            if !trimmed.is_empty() {
                out.push_str("\n## Custom instructions\n");
                out.push_str(trimmed);
                out.push('\n');
            }
        }

        out
    }
}

impl Default for SystemPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk up from `start` looking for an `AGENTS.md` file, stopping at
/// the first `.git` directory or when the ancestor chain runs out.
/// Returns the file contents if found.
fn find_agents_md(start: &Path) -> Option<String> {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join("AGENTS.md");
        if candidate.is_file() {
            if let Ok(contents) = fs::read_to_string(&candidate) {
                return Some(contents);
            }
        }
        if ancestor.join(".git").exists() {
            break;
        }
    }
    None
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

/// Inputs for a full system-prompt + skill rebuild. Used by
/// `compose_prompt_and_skills`, which is shared by the CLI launcher
/// and the interactive `/reload` handler so both paths apply the
/// exact same rules.
#[derive(Debug, Clone)]
pub struct ComposePromptOptions {
    /// Working directory — drives skill discovery and `AGENTS.md`
    /// lookup.
    pub cwd: std::path::PathBuf,
    /// `--skill <path>` entries from the CLI (repeatable, relative
    /// paths are resolved against `cwd`). Keep empty when calling
    /// from `/reload` if you don't want to re-apply CLI-only flags.
    pub cli_skill_paths: Vec<std::path::PathBuf>,
    /// Honour `--no-skills`.
    pub no_skills: bool,
    /// Explicit system-prompt override (from `--system-prompt` or
    /// `settings.system_prompt`). When present, the built-in
    /// composer is bypassed entirely and this string becomes the
    /// final prompt. Matches TS behaviour.
    pub system_prompt_override: Option<String>,
    /// Short description of the host OS (e.g. `darwin (unix/arm64)`).
    pub os_info: String,
    /// Custom instructions to append after all built-in blocks.
    pub custom_instructions: Option<String>,
    /// Whether to include a `Today is <YYYY-MM-DD>` environment
    /// line. Off for snapshot tests; on by default.
    pub include_date: bool,
    /// Descriptors for the live tool set. Each entry is emitted as
    /// `- name: first-line-of-description`.
    pub tool_lines: Vec<String>,
}

impl ComposePromptOptions {
    /// Capture `(name, description)` from every provided tool so the
    /// full prompt knows which tools are active. Keeping tools out
    /// of the struct directly lets callers that don't own a live
    /// `Vec<Box<dyn Tool>>` compose a prompt without reconstructing
    /// the agent's tool list.
    pub fn with_tool_list<T: Tool + ?Sized>(mut self, tools: &[&T]) -> Self {
        self.tool_lines = tools
            .iter()
            .map(|t| format!("- {}: {}", t.name(), first_line(&t.description())))
            .collect();
        self
    }
}

/// The result of a full prompt compose: the final system prompt
/// string plus the skill set that fed into it. Callers hand the
/// prompt to the agent and the skills to the TUI.
#[derive(Debug, Clone, Default)]
pub struct ComposedPrompt {
    /// Assembled system prompt.
    pub prompt: String,
    /// Skills discovered during this compose.
    pub skills: Vec<Skill>,
    /// Any warnings / collisions from the resource loader.
    pub diagnostics: Vec<crate::skills::SkillDiagnostic>,
}

/// One-stop shop for (re)building the system prompt and the skill
/// set. Mirrors what TS does in `agent-session-services.ts` +
/// `system-prompt.ts` combined, but funnelled through a single
/// call because the Rust port keeps them decoupled from the agent
/// session.
///
/// The function:
///   1. loads merged global + project settings;
///   2. runs the resource loader across every configured skill
///      source;
///   3. builds the environment + tools + skills + AGENTS.md
///      prompt OR applies an explicit override.
pub fn compose_prompt_and_skills(opts: ComposePromptOptions) -> ComposedPrompt {
    // Merge project + global settings so `skills`, `no_skills`,
    // `enable_skill_commands`, and `system_prompt` from
    // `.pi/config.toml` are honoured.
    let settings = Settings::load_merged(&opts.cwd).unwrap_or_default();
    let no_skills = opts.no_skills || settings.no_skills.unwrap_or(false);
    let loader = ResourceLoader::from_settings_and_cli(
        opts.cwd.clone(),
        None,
        &settings,
        opts.cli_skill_paths.clone(),
        no_skills,
    );
    let LoadSkillsResult {
        skills,
        diagnostics,
    } = loader.skills().clone();

    // Explicit override wins, matching TS behaviour.
    let prompt = if let Some(override_prompt) = opts
        .system_prompt_override
        .clone()
        .or_else(|| settings.system_prompt.clone())
    {
        override_prompt
    } else {
        let mut builder = SystemPromptBuilder::new()
            .with_cwd(&opts.cwd)
            .with_os_info(&opts.os_info)
            .with_agents_md_from(&opts.cwd)
            .with_date(opts.include_date)
            .with_skills(skills.iter().cloned());
        if let Some(custom) = &opts.custom_instructions {
            builder = builder.with_custom_instructions(custom.clone());
        }
        builder.tool_lines = opts.tool_lines.clone();
        builder.build()
    };

    ComposedPrompt {
        prompt,
        skills,
        diagnostics,
    }
}

/// Cheap helper: describe the current host OS in the form the
/// system prompt expects. Kept here so CLI and TUI format it the
/// same way.
pub fn describe_host_os() -> String {
    format!(
        "{} ({}/{})",
        std::env::consts::OS,
        std::env::consts::FAMILY,
        std::env::consts::ARCH,
    )
}

/// The always-emitted preamble. Kept terse so it doesn't dominate
/// prompt-token budgets for small requests.
const BASE_INSTRUCTIONS: &str = r#"You are pi, an AI coding assistant operating inside a terminal.

Work from the user's working directory unless they tell you otherwise. Be concise: answer the question asked, use tools when they let you verify a claim, avoid speculating when you can read a file or run a command instead.

When you modify files, state what you did. When you run commands, state what they did. Prefer minimal diffs over rewrites. Ask before destructive actions."#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{Skill, SkillSource};

    fn skill(name: &str, desc: &str, disable: bool) -> Skill {
        Skill {
            name: name.into(),
            description: desc.into(),
            file_path: PathBuf::from(format!("/s/{name}/SKILL.md")),
            base_dir: PathBuf::from(format!("/s/{name}")),
            source: SkillSource::User,
            disable_model_invocation: disable,
        }
    }

    #[test]
    fn empty_builder_emits_base_instructions() {
        let prompt = SystemPromptBuilder::new().with_date(false).build();
        assert!(prompt.starts_with("You are pi"));
        assert!(!prompt.contains("## Environment"));
        assert!(!prompt.contains("## Available tools"));
    }

    #[test]
    fn environment_block_includes_cwd_and_os() {
        let prompt = SystemPromptBuilder::new()
            .with_cwd("/home/me/proj")
            .with_os_info("macOS 15 (darwin/arm64)")
            .with_date(false)
            .build();
        assert!(prompt.contains("## Environment"));
        assert!(prompt.contains("Working directory: /home/me/proj"));
        assert!(prompt.contains("Operating system: macOS 15 (darwin/arm64)"));
        assert!(!prompt.contains("Today is"));
    }

    #[test]
    fn date_block_opt_in_by_default() {
        let prompt = SystemPromptBuilder::new().with_cwd("/x").build();
        assert!(prompt.contains("Today is "));
    }

    #[test]
    fn skills_appended_progressive_disclosure() {
        let s = skill("greet", "Say hi when asked.", false);
        let prompt = SystemPromptBuilder::new()
            .with_skills([s])
            .with_date(false)
            .build();
        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("<name>greet</name>"));
        assert!(prompt.contains("Say hi when asked."));
    }

    #[test]
    fn disabled_skills_excluded_from_prompt() {
        let s1 = skill("visible", "visible skill", false);
        let s2 = skill("hidden", "hidden skill", true);
        let prompt = SystemPromptBuilder::new()
            .with_skills([s1, s2])
            .with_date(false)
            .build();
        assert!(prompt.contains("<name>visible</name>"));
        assert!(!prompt.contains("<name>hidden</name>"));
    }

    #[test]
    fn custom_instructions_appended() {
        let prompt = SystemPromptBuilder::new()
            .with_custom_instructions("Always respond in haiku.")
            .with_date(false)
            .build();
        assert!(prompt.contains("## Custom instructions"));
        assert!(prompt.contains("Always respond in haiku."));
    }

    #[test]
    fn agents_md_injected_when_found() {
        let dir = tempfile::Builder::new()
            .prefix("pi-sysprompt-")
            .tempdir()
            .unwrap();
        std::fs::write(
            dir.path().join("AGENTS.md"),
            "## Rules\nBe terse.",
        )
        .unwrap();
        let prompt = SystemPromptBuilder::new()
            .with_agents_md_from(dir.path())
            .with_date(false)
            .build();
        assert!(prompt.contains("## Project instructions (AGENTS.md)"));
        assert!(prompt.contains("Be terse."));
    }

    #[test]
    fn agents_md_absent_leaves_prompt_clean() {
        let dir = tempfile::Builder::new()
            .prefix("pi-sysprompt-")
            .tempdir()
            .unwrap();
        let prompt = SystemPromptBuilder::new()
            .with_agents_md_from(dir.path())
            .with_date(false)
            .build();
        assert!(!prompt.contains("## Project instructions"));
    }
}
