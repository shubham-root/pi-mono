//! Interactive mode - TUI matching TypeScript variant
//! - Enter to send message
//! - Alt+Enter to queue follow-up
//! - Escape to cancel
//! - Ctrl+L for model selector
//! - Ctrl+T for thinking toggle
//! - Ctrl+O for tool output toggle
//! - Ctrl+C to clear, Ctrl+C twice to quit
//! - / to open command palette

use crate::editor::InputEditor;
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use pi_core::model_registry::ModelRegistry;
use pi_core::Agent;
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Wrap};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Slash command info
#[derive(Clone, Debug)]
pub struct SlashCommand {
    pub name: String,
    pub description: String,
}

fn get_all_slash_commands() -> Vec<SlashCommand> {
    vec![
        SlashCommand { name: "help".to_string(), description: "Show keyboard shortcuts".to_string() },
        SlashCommand { name: "hotkeys".to_string(), description: "Show all keyboard shortcuts".to_string() },
        SlashCommand { name: "model".to_string(), description: "Switch models".to_string() },
        SlashCommand { name: "settings".to_string(), description: "Edit settings".to_string() },
        SlashCommand { name: "login".to_string(), description: "OAuth authentication".to_string() },
        SlashCommand { name: "logout".to_string(), description: "Clear authentication".to_string() },
        SlashCommand { name: "new".to_string(), description: "Start fresh session".to_string() },
        SlashCommand { name: "tree".to_string(), description: "Show session tree".to_string() },
        SlashCommand { name: "session".to_string(), description: "Show session info".to_string() },
        SlashCommand { name: "fork".to_string(), description: "Fork session".to_string() },
        SlashCommand { name: "export".to_string(), description: "Export to HTML file".to_string() },
        SlashCommand { name: "share".to_string(), description: "Share as GitHub gist".to_string() },
        SlashCommand { name: "compact".to_string(), description: "Manual context compaction".to_string() },
        SlashCommand { name: "thinking".to_string(), description: "Set reasoning level".to_string() },
        SlashCommand { name: "copy".to_string(), description: "Copy last assistant message".to_string() },
        SlashCommand { name: "reload".to_string(), description: "Reload config and extensions".to_string() },
        SlashCommand { name: "quit".to_string(), description: "Exit pi".to_string() },
    ]
}

fn filter_commands(query: &str, all: &[SlashCommand]) -> Vec<SlashCommand> {
    let q = query.to_lowercase();
    all.iter().filter(|cmd| cmd.name.contains(&q)).cloned().collect()
}

/// Flattened model entry for display (includes provider display name + id).
#[derive(Clone, Debug)]
pub struct ModelRow {
    pub provider_id: String,
    pub provider_display: String,
    pub model_id: String,
    pub model_name: String,
    pub context_window: u32,
    pub reasoning: bool,
    pub env_configured: bool,
}

fn build_model_rows(registry: &ModelRegistry) -> Vec<ModelRow> {
    let mut rows: Vec<ModelRow> = Vec::new();
    for provider in registry.providers() {
        let env_configured = provider.resolve_env_key().is_some();
        for model in &provider.models {
            rows.push(ModelRow {
                provider_id: provider.id.clone(),
                provider_display: provider.display_name.clone(),
                model_id: model.id.clone(),
                model_name: model.name.clone(),
                context_window: model.context_window,
                reasoning: model.reasoning,
                env_configured,
            });
        }
    }
    rows
}

fn filter_model_rows(query: &str, all: &[ModelRow]) -> Vec<ModelRow> {
    if query.is_empty() {
        return all.to_vec();
    }
    let q = query.to_lowercase();
    all.iter()
        .filter(|row| {
            row.model_id.to_lowercase().contains(&q)
                || row.model_name.to_lowercase().contains(&q)
                || row.provider_id.to_lowercase().contains(&q)
                || row.provider_display.to_lowercase().contains(&q)
        })
        .cloned()
        .collect()
}

/// Setting info
#[derive(Clone, Debug)]
pub struct SettingInfo {
    pub name: String,
    pub description: String,
    pub current: String,
    pub options: String,
}

fn get_settings_list() -> Vec<SettingInfo> {
    vec![
        SettingInfo { name: "defaultThinkingLevel".to_string(), description: "Thinking level".to_string(), current: "medium".to_string(), options: "off|minimal|low|medium|high|xhigh".to_string() },
        SettingInfo { name: "theme".to_string(), description: "UI theme".to_string(), current: "auto".to_string(), options: "dark|light|auto".to_string() },
        SettingInfo { name: "hideThinkingBlock".to_string(), description: "Hide thinking blocks".to_string(), current: "false".to_string(), options: "true|false".to_string() },
        SettingInfo { name: "steeringMode".to_string(), description: "Steering delivery".to_string(), current: "one-at-a-time".to_string(), options: "all|one-at-a-time".to_string() },
        SettingInfo { name: "followUpMode".to_string(), description: "Follow-up delivery".to_string(), current: "one-at-a-time".to_string(), options: "all|one-at-a-time".to_string() },
        SettingInfo { name: "transport".to_string(), description: "Provider transport".to_string(), current: "auto".to_string(), options: "sse|websocket|auto".to_string() },
        SettingInfo { name: "doubleEscapeAction".to_string(), description: "Double escape action".to_string(), current: "tree".to_string(), options: "fork|tree|none".to_string() },
    ]
}

/// UI display mode
#[derive(Clone, Copy, Debug, PartialEq)]
enum DisplayMode {
    Chat,
    ModelList,
    SettingsList,
    /// Pick a provider to log into (enter an API key for).
    LoginList,
    /// After picking a provider, type the key inline.
    LoginKeyEntry,
    /// Transcript tree viewer.
    TreeView,
    /// Thinking-level cycling overlay.
    ThinkingList,
}

/// Conversation entry rendered in the chat area. We remember enough
/// structure that the renderer can show thinking blocks separately, tool
/// calls inline, and stream partial assistant text live as it arrives.
#[derive(Clone, Debug, Default)]
pub struct ConversationMessage {
    pub role: String, // "user" | "assistant"
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<ConversationToolCall>,
    /// True until the agent emits `Done` / `TurnComplete` for this
    /// assistant message. Lets the renderer show a live caret while
    /// tokens arrive.
    pub streaming: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ConversationToolCall {
    pub id: String,
    pub name: String,
    pub input_preview: String,
    /// `None` while the tool is running, `Some(output)` once it returns.
    pub output: Option<String>,
    pub is_error: bool,
}

/// Per-session usage accumulator surfaced in the footer.
#[derive(Clone, Debug, Default)]
pub struct UsageStats {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub total_tokens: u64,
}

/// Re-export the streaming event type so callers can keep the namespace
/// local to this module if they prefer.
pub use pi_core::AgentEvent;

pub struct InteractiveMode {
    event_loop: EventLoop,
    agent: Option<Agent>,
    messages: Vec<ConversationMessage>,
    editor: InputEditor,
    queued_messages: Vec<String>,
    status: String,
    executing: bool,
    show_thinking: bool,
    show_tools: bool,

    // Command palette
    palette_active: bool,
    palette_items: Vec<SlashCommand>,
    palette_selected: usize,
    all_commands: Vec<SlashCommand>,

    // Display mode for model/settings lists
    display_mode: DisplayMode,
    model_list: Vec<ModelRow>,
    model_filter: String,
    model_selected: usize,
    settings_list: Vec<SettingInfo>,
    settings_selected: usize,

    // Login flow state
    login_list: Vec<LoginRow>,
    login_selected: usize,
    login_provider_id: Option<String>,
    login_key_input: String,

    // Tree viewer
    tree_rows: Vec<TreeRow>,
    tree_selected: usize,

    // Thinking-level selector
    thinking_options: Vec<&'static str>,
    thinking_selected: usize,

    // Cached sessions directory for save/fork
    sessions_dir: PathBuf,

    // Escape/Ctrl+C tracking
    last_escape_time: Option<std::time::Instant>,
    ctrl_c_count: u32,

    // Active prompt request. When this is `Some`, `agent` is `None` because
    // the agent has been moved into the background task; it returns via the
    // join handle along with the result.
    active_job: Option<PromptJob>,

    /// Cumulative usage across the current session. Updated on every
    /// `AgentEvent::Usage` so the footer's `0.0%/?` can show the real
    /// percentage of context used.
    usage: UsageStats,
}

struct PromptJob {
    /// Joins to `(Agent, Result<()>)` once the whole turn completes. The
    /// assistant text is threaded to the UI via `events` as it arrives,
    /// so there's no need to surface it through the join handle.
    handle: JoinHandle<(Agent, Result<()>)>,
    /// Receiver for live agent progress. Drained on every tick.
    events: mpsc::UnboundedReceiver<AgentEvent>,
    started_at: Instant,
    user_message: String,
    /// Index into `messages` of the streaming assistant placeholder for
    /// this job. We append tokens here as they arrive.
    assistant_index: usize,
}

impl InteractiveMode {
    pub fn new(agent: Agent) -> Result<Self> {
        let event_loop = EventLoop::new()?;
        Ok(Self {
            event_loop,
            agent: Some(agent),
            messages: Vec::new(),
            editor: InputEditor::new(),
            queued_messages: Vec::new(),
            status: "Ready. Type / for commands, Enter to send, Ctrl+C twice to quit.".to_string(),
            executing: false,
            show_thinking: false,
            show_tools: false,
            palette_active: false,
            palette_items: Vec::new(),
            palette_selected: 0,
            all_commands: get_all_slash_commands(),
            display_mode: DisplayMode::Chat,
            model_list: Vec::new(),
            model_filter: String::new(),
            model_selected: 0,
            settings_list: Vec::new(),
            settings_selected: 0,
            login_list: Vec::new(),
            login_selected: 0,
            login_provider_id: None,
            login_key_input: String::new(),
            tree_rows: Vec::new(),
            tree_selected: 0,
            thinking_options: vec!["off", "low", "medium", "high"],
            thinking_selected: 0,
            sessions_dir: default_sessions_dir(),
            last_escape_time: None,
            ctrl_c_count: 0,
            active_job: None,
            usage: UsageStats::default(),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        loop {
            // Drain keyboard/terminal events first so the UI stays responsive.
            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(50)) {
                use pi_tui::tui::AppEvent;
                let cont = match event {
                    AppEvent::Key(cmd) => self.handle_key(cmd).await?,
                    _ => true,
                };
                if !cont {
                    break;
                }
            }

            // Check whether a background prompt finished; if so, pull the
            // result into the UI. Done every tick so responses appear as
            // soon as the provider returns.
            self.poll_active_job().await;

            // Keep the "Executing... (Ns)" counter ticking while a job is
            // in flight so the user has visible feedback.
            if let Some(job) = &self.active_job {
                let elapsed = job.started_at.elapsed().as_secs();
                self.status = format!("Executing on {} ({}s)... Esc to cancel", self.current_model_label(), elapsed);
            }

            self.draw()?;
        }
        Ok(())
    }

    fn current_model_label(&self) -> String {
        self.agent
            .as_ref()
            .map(|a| a.model_id().to_string())
            .or_else(|| {
                self.active_job
                    .as_ref()
                    .map(|j| format!("(pending) {}", j.user_message.chars().take(20).collect::<String>()))
            })
            .unwrap_or_else(|| "(no agent)".to_string())
    }

    /// Drain any AgentEvents the background task pushed since the last
    /// tick, mutating the streaming assistant placeholder as tokens /
    /// tool calls / usage events arrive. When the task joins, restore
    /// `self.agent` and kick off any queued follow-up message.
    async fn poll_active_job(&mut self) {
        // Pull events out of the receiver into a local buffer so we can
        // drop the mutable borrow on `self.active_job` before calling
        // `self.handle_agent_event` (which also needs `&mut self`).
        let (assistant_index, events, finished, task_completed_events) = {
            let Some(job) = self.active_job.as_mut() else {
                return;
            };
            let mut buffered: Vec<AgentEvent> = Vec::new();
            loop {
                match job.events.try_recv() {
                    Ok(event) => buffered.push(event),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
            let finished = job.handle.is_finished();
            // We also drain once more AFTER the finished check below, to
            // catch any events the worker pushed between `try_recv` and
            // task completion. That drain happens in the post-join branch.
            (job.assistant_index, buffered, finished, Vec::<AgentEvent>::new())
        };
        let _ = task_completed_events;

        for event in events {
            self.handle_agent_event(assistant_index, event);
        }

        if !finished {
            return;
        }

        // Worker task finished. Drain any tail events, then join the
        // task so we can restore `self.agent` and finalize UI state.
        let mut job = self.active_job.take().expect("active_job just observed");
        let mut tail: Vec<AgentEvent> = Vec::new();
        while let Ok(event) = job.events.try_recv() {
            tail.push(event);
        }
        for event in tail {
            self.handle_agent_event(assistant_index, event);
        }

        match job.handle.await {
            Ok((agent, result)) => {
                self.agent = Some(agent);
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    msg.streaming = false;
                    if msg.content.trim().is_empty()
                        && msg.thinking.trim().is_empty()
                        && msg.tool_calls.is_empty()
                    {
                        msg.content = "(empty response)".to_string();
                    }
                }
                match result {
                    Ok(()) => {
                        self.status = format!(
                            "Ready ({:.1}s)",
                            job.started_at.elapsed().as_secs_f32()
                        );
                    }
                    Err(e) => {
                        self.status = format!("Error: {}", e);
                    }
                }
                self.executing = false;

                if let Some(next) = self.queued_messages.pop() {
                    self.spawn_prompt(next);
                }
            }
            Err(join_err) => {
                self.status = format!("Agent task panicked: {}", join_err);
                self.executing = false;
            }
        }
    }

    fn handle_agent_event(&mut self, assistant_index: usize, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta { delta } => {
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    msg.content.push_str(&delta);
                }
            }
            AgentEvent::ThinkingDelta { delta, .. } => {
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    msg.thinking.push_str(&delta);
                }
            }
            AgentEvent::ToolCallStart { id, name, input } => {
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    let preview = preview_tool_args(&input);
                    msg.tool_calls.push(ConversationToolCall {
                        id,
                        name,
                        input_preview: preview,
                        output: None,
                        is_error: false,
                    });
                }
            }
            AgentEvent::ToolCallResult { id, output, is_error } => {
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    if let Some(call) = msg.tool_calls.iter_mut().find(|c| c.id == id) {
                        call.output = Some(output);
                        call.is_error = is_error;
                    }
                }
            }
            AgentEvent::Usage(usage) => {
                // Accumulate totals for the session. Providers emit one
                // Usage event per turn, so summing input+output from each
                // is correct. Cache counters come in optional form.
                self.usage.input_tokens = self.usage.input_tokens.saturating_add(usage.input_tokens as u64);
                self.usage.output_tokens = self.usage.output_tokens.saturating_add(usage.output_tokens as u64);
                if let Some(cr) = usage.cache_read_tokens {
                    self.usage.cache_read_tokens =
                        self.usage.cache_read_tokens.saturating_add(cr as u64);
                }
                if let Some(cw) = usage.cache_write_tokens {
                    self.usage.cache_write_tokens =
                        self.usage.cache_write_tokens.saturating_add(cw as u64);
                }
                // total = input + output + cache_read (cache_read counts
                // toward the context window at a discount but for UI
                // purposes we surface the raw sum).
                self.usage.total_tokens = self.usage.total_tokens.saturating_add(
                    (usage.input_tokens as u64) + (usage.output_tokens as u64),
                );
            }
            AgentEvent::TurnComplete { .. } => {
                // No-op at this layer; Done carries the final_text summary.
            }
            AgentEvent::Done { final_text: _ } => {
                // Finalization happens in the join path where we also
                // have the agent handle. Nothing to do here; the last
                // TextDelta already populated content.
            }
            AgentEvent::Error { message } => {
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    msg.streaming = false;
                    if msg.content.is_empty() {
                        msg.content = format!("(error) {message}");
                    }
                }
                self.status = format!("Error: {message}");
            }
        }
    }

    fn spawn_prompt(&mut self, text: String) {
        if self.active_job.is_some() {
            self.queued_messages.insert(0, text);
            return;
        }
        let Some(mut agent) = self.agent.take() else {
            self.status = "No agent configured".to_string();
            return;
        };
        if !agent.has_api_key() {
            let registry = ModelRegistry::global();
            let hint = registry
                .find_model(agent.model_id())
                .map(|(p, _)| {
                    if p.env_vars.is_empty() {
                        format!("provider '{}' has no env vars registered", p.id)
                    } else {
                        format!(
                            "set {} for provider '{}'",
                            p.env_vars.join(" or "),
                            p.id
                        )
                    }
                })
                .unwrap_or_else(|| "configure an API key".to_string());
            self.status = format!("Cannot send: {}", hint);
            self.agent = Some(agent);
            return;
        }

        // Create the streaming assistant placeholder. Further deltas will
        // append to `content`/`thinking` and push into `tool_calls`.
        self.messages.push(ConversationMessage {
            role: "assistant".to_string(),
            content: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            streaming: true,
        });
        let assistant_index = self.messages.len() - 1;

        let (tx, rx) = mpsc::unbounded_channel::<AgentEvent>();
        let text_for_task = text.clone();
        let started_at = Instant::now();
        let handle = tokio::spawn(async move {
            let result = agent.prompt_stream(&text_for_task, tx).await;
            (agent, result)
        });
        self.active_job = Some(PromptJob {
            handle,
            events: rx,
            started_at,
            user_message: text,
            assistant_index,
        });
        self.executing = true;
        self.status = "Executing...".to_string();
    }

    async fn handle_key(&mut self, cmd: KeyCommand) -> Result<bool> {
        match cmd {
            // ENTER: select from palette, select model/setting, or send message
            KeyCommand::Enter => {
                if self.palette_active {
                    // Select command from palette
                    if !self.palette_items.is_empty() {
                        let cmd_name = self.palette_items[self.palette_selected].name.clone();
                        self.close_palette();
                        self.execute_command(&cmd_name).await?;
                        self.editor.clear();
                    }
                } else if self.display_mode == DisplayMode::ModelList {
                    // Select model: swap the agent's model + api key to match
                    // the new provider. Without this the previously-configured
                    // provider stays active and the user's selection is a no-op.
                    if !self.model_list.is_empty() {
                        let m = self.model_list[self.model_selected].clone();
                        let qualified = format!("{}/{}", m.provider_id, m.model_id);
                        let registry = ModelRegistry::global();
                        let provider_env = registry
                            .provider(&m.provider_id)
                            .and_then(|p| p.resolve_env_key());

                        let status = if self.active_job.is_some() {
                            format!(
                                "Wait for current request to finish before switching to {}",
                                m.model_name
                            )
                        } else if let Some(ref mut agent) = self.agent {
                            match agent.set_model(&qualified) {
                                Ok(()) => {
                                    match provider_env {
                                        Some((var, value)) => {
                                            agent.set_api_key(Some(value));
                                            format!(
                                                "Switched to {} ({}) using {}",
                                                m.model_name, m.provider_display, var
                                            )
                                        }
                                        None => {
                                            agent.set_api_key(None);
                                            let hint = registry
                                                .provider(&m.provider_id)
                                                .filter(|p| !p.env_vars.is_empty())
                                                .map(|p| {
                                                    format!("set {}", p.env_vars.join(" or "))
                                                })
                                                .unwrap_or_else(|| {
                                                    "no env var registered for this provider"
                                                        .to_string()
                                                });
                                            format!(
                                                "Switched to {} ({}) - auth missing: {}",
                                                m.model_name, m.provider_display, hint
                                            )
                                        }
                                    }
                                }
                                Err(e) => format!("Cannot switch to {}: {}", m.model_id, e),
                            }
                        } else {
                            format!("Selected {} (no agent initialized)", m.model_name)
                        };

                        self.status = status;
                        self.display_mode = DisplayMode::Chat;
                        self.model_filter.clear();
                    }
                } else if self.display_mode == DisplayMode::SettingsList {
                    // Select setting
                    if !self.settings_list.is_empty() {
                        let s = &self.settings_list[self.settings_selected];
                        self.status = format!("Setting: {} = {}", s.name, s.current);
                        self.display_mode = DisplayMode::Chat;
                    }
                } else if self.display_mode == DisplayMode::LoginList {
                    if !self.login_list.is_empty() {
                        let row = self.login_list[self.login_selected].clone();
                        if self.login_provider_id.as_deref() == Some("<logout>") {
                            // Logout flow — clear this provider's key.
                            match crate::auth::clear_key(&row.provider_id) {
                                Ok(true) => {
                                    self.status = format!(
                                        "Cleared auth.json entry for {}",
                                        row.display_name
                                    );
                                }
                                Ok(false) => {
                                    self.status = format!(
                                        "No auth.json entry for {} to clear",
                                        row.display_name
                                    );
                                }
                                Err(e) => {
                                    self.status = format!("Logout failed: {e}");
                                }
                            }
                            self.login_provider_id = None;
                            self.display_mode = DisplayMode::Chat;
                        } else {
                            // Login flow — transition to key-entry.
                            self.login_provider_id = Some(row.provider_id.clone());
                            self.login_key_input.clear();
                            self.display_mode = DisplayMode::LoginKeyEntry;
                            self.status = format!(
                                "Paste your {} API key. Enter to save, Esc to cancel.",
                                row.display_name
                            );
                        }
                    }
                } else if self.display_mode == DisplayMode::LoginKeyEntry {
                    self.finalize_login();
                    self.login_provider_id = None;
                    self.login_key_input.clear();
                    self.display_mode = DisplayMode::Chat;
                } else if self.display_mode == DisplayMode::ThinkingList {
                    if let Some(&opt) = self.thinking_options.get(self.thinking_selected) {
                        let level = thinking_level_from_label(opt);
                        if let Some(agent) = self.agent.as_mut() {
                            agent.set_thinking_level(level);
                        }
                        self.status = format!("Thinking level → {opt}");
                    }
                    self.display_mode = DisplayMode::Chat;
                } else if self.display_mode == DisplayMode::TreeView {
                    // Enter on a tree node is a no-op for now; Esc returns.
                    self.status = "Tree node selected (navigation only for now)".to_string();
                } else if !self.editor.is_empty() {
                    // Send message
                    let msg = self.editor.text().trim().to_string();
                    if msg.starts_with('/') {
                        let cmd_name = msg.trim_start_matches('/').to_string();
                        self.execute_command(&cmd_name).await?;
                    } else {
                        self.send_message(&msg);
                    }
                    self.editor.clear();
                }
                Ok(true)
            }

            // Alt+Enter: queue message
            KeyCommand::AltEnter => {
                if !self.editor.is_empty() {
                    self.queued_messages.push(self.editor.text().trim().to_string());
                    self.status = format!("Queued ({})", self.queued_messages.len());
                    self.editor.clear();
                }
                Ok(true)
            }

            // Ctrl+C: clear or quit
            KeyCommand::CtrlC => {
                self.ctrl_c_count += 1;
                if self.ctrl_c_count >= 2 {
                    return Ok(false);
                }
                if !self.editor.is_empty() {
                    self.editor.clear();
                    self.close_palette();
                }
                self.status = "Press Ctrl+C again to quit".to_string();
                Ok(true)
            }

            // Escape: cancel or go back
            KeyCommand::Escape => {
                self.ctrl_c_count = 0;
                if self.palette_active {
                    self.close_palette();
                    self.editor.clear();
                    self.status = "Palette closed".to_string();
                } else if self.display_mode != DisplayMode::Chat {
                    self.display_mode = DisplayMode::Chat;
                    self.model_filter.clear();
                    self.login_provider_id = None;
                    self.login_key_input.clear();
                    self.status = "Back to chat".to_string();
                } else if !self.editor.is_empty() {
                    self.editor.clear();
                    self.status = "Input cleared".to_string();
                }
                Ok(true)
            }

            // Character input
            KeyCommand::Char(c) => {
                self.ctrl_c_count = 0;

                // In model list, typing filters without putting chars in main input
                if self.display_mode == DisplayMode::ModelList {
                    self.model_filter.push(c);
                    let rows = build_model_rows(ModelRegistry::global());
                    self.model_list = filter_model_rows(&self.model_filter, &rows);
                    self.model_selected = 0;
                    return Ok(true);
                }

                // Login key entry — accumulate chars privately, don't
                // touch the editor buffer so the key never lands in
                // transcript text.
                if self.display_mode == DisplayMode::LoginKeyEntry {
                    self.login_key_input.push(c);
                    return Ok(true);
                }

                self.editor.insert_char(c);

                // Activate palette on first /
                if self.editor.text() == "/" {
                    self.palette_active = true;
                    self.palette_items = self.all_commands.clone();
                    self.palette_selected = 0;
                    self.status = "Type to filter commands, \u{2191}\u{2193} to navigate, Enter to select".to_string();
                } else if self.palette_active && self.editor.text().starts_with('/') {
                    // Filter as user types
                    let query = &self.editor.text()[1..];
                    self.palette_items = filter_commands(query, &self.all_commands);
                    self.palette_selected = 0;
                }
                Ok(true)
            }

            // Backspace
            KeyCommand::Backspace => {
                self.ctrl_c_count = 0;

                // Backspace in model list trims the filter
                if self.display_mode == DisplayMode::ModelList {
                    self.model_filter.pop();
                    let rows = build_model_rows(ModelRegistry::global());
                    self.model_list = filter_model_rows(&self.model_filter, &rows);
                    self.model_selected = 0;
                    return Ok(true);
                }
                if self.display_mode == DisplayMode::LoginKeyEntry {
                    self.login_key_input.pop();
                    return Ok(true);
                }

                self.editor.backspace();
                if self.palette_active {
                    if self.editor.is_empty() {
                        self.close_palette();
                    } else if self.editor.text().starts_with('/') {
                        let query = &self.editor.text()[1..];
                        self.palette_items = filter_commands(query, &self.all_commands);
                        self.palette_selected = 0;
                    } else {
                        self.close_palette();
                    }
                }
                Ok(true)
            }

            // Forward-delete (Del key)
            KeyCommand::Delete => {
                self.editor.delete_forward();
                Ok(true)
            }

            // Arrow Up: navigate palette/lists
            KeyCommand::ArrowUp => {
                if self.palette_active && !self.palette_items.is_empty() {
                    self.palette_selected = self.palette_selected.saturating_sub(1);
                } else if self.display_mode == DisplayMode::ModelList {
                    self.model_selected = self.model_selected.saturating_sub(1);
                } else if self.display_mode == DisplayMode::SettingsList {
                    self.settings_selected = self.settings_selected.saturating_sub(1);
                } else if self.display_mode == DisplayMode::LoginList {
                    self.login_selected = self.login_selected.saturating_sub(1);
                } else if self.display_mode == DisplayMode::TreeView {
                    self.tree_selected = self.tree_selected.saturating_sub(1);
                } else if self.display_mode == DisplayMode::ThinkingList {
                    self.thinking_selected = self.thinking_selected.saturating_sub(1);
                }
                Ok(true)
            }

            // Arrow Down: navigate palette/lists
            KeyCommand::ArrowDown => {
                if self.palette_active && !self.palette_items.is_empty() {
                    if self.palette_selected < self.palette_items.len() - 1 {
                        self.palette_selected += 1;
                    }
                } else if self.display_mode == DisplayMode::ModelList && !self.model_list.is_empty() {
                    if self.model_selected < self.model_list.len() - 1 {
                        self.model_selected += 1;
                    }
                } else if self.display_mode == DisplayMode::SettingsList && !self.settings_list.is_empty() {
                    if self.settings_selected < self.settings_list.len() - 1 {
                        self.settings_selected += 1;
                    }
                } else if self.display_mode == DisplayMode::LoginList && !self.login_list.is_empty() {
                    if self.login_selected < self.login_list.len() - 1 {
                        self.login_selected += 1;
                    }
                } else if self.display_mode == DisplayMode::TreeView && !self.tree_rows.is_empty() {
                    if self.tree_selected < self.tree_rows.len() - 1 {
                        self.tree_selected += 1;
                    }
                } else if self.display_mode == DisplayMode::ThinkingList {
                    if self.thinking_selected < self.thinking_options.len() - 1 {
                        self.thinking_selected += 1;
                    }
                }
                Ok(true)
            }

            // Tab: autocomplete command
            KeyCommand::Tab => {
                if self.palette_active && !self.palette_items.is_empty() {
                    let cmd_name = self.palette_items[self.palette_selected].name.clone();
                    self.editor.clear();
                    self.editor.insert_str(&format!("/{}", cmd_name));
                    self.close_palette();
                }
                Ok(true)
            }

            // Display toggles
            KeyCommand::CtrlT => {
                self.show_thinking = !self.show_thinking;
                self.status = format!("Thinking: {}", if self.show_thinking { "ON" } else { "OFF" });
                Ok(true)
            }
            KeyCommand::CtrlO => {
                self.show_tools = !self.show_tools;
                self.status = format!("Tool output: {}", if self.show_tools { "ON" } else { "OFF" });
                Ok(true)
            }

            // Model selector shortcut
            KeyCommand::CtrlL => {
                self.execute_command("model").await?;
                Ok(true)
            }

            // ---------- Editor cursor movement ----------
            KeyCommand::ArrowLeft => {
                self.editor.move_left();
                Ok(true)
            }
            KeyCommand::ArrowRight => {
                self.editor.move_right();
                Ok(true)
            }
            KeyCommand::Home | KeyCommand::CtrlA => {
                self.editor.move_line_start();
                Ok(true)
            }
            KeyCommand::End | KeyCommand::CtrlE => {
                self.editor.move_line_end();
                Ok(true)
            }
            KeyCommand::AltB => {
                self.editor.move_word_left();
                Ok(true)
            }
            KeyCommand::AltF => {
                self.editor.move_word_right();
                Ok(true)
            }

            // ---------- Editor kill/yank ----------
            KeyCommand::CtrlW => {
                self.editor.kill_word_back();
                Ok(true)
            }
            KeyCommand::AltD => {
                self.editor.kill_word_forward();
                Ok(true)
            }
            KeyCommand::CtrlK => {
                self.editor.kill_to_line_end();
                Ok(true)
            }
            KeyCommand::CtrlU => {
                self.editor.kill_to_line_start();
                Ok(true)
            }
            KeyCommand::CtrlY => {
                self.editor.yank();
                Ok(true)
            }

            // ---------- Undo / redo ----------
            KeyCommand::CtrlZ => {
                self.editor.undo();
                Ok(true)
            }
            KeyCommand::AltZ => {
                self.editor.redo();
                Ok(true)
            }

            _ => Ok(true),
        }
    }

    fn close_palette(&mut self) {
        self.palette_active = false;
        self.palette_items.clear();
        self.palette_selected = 0;
    }

    async fn execute_command(&mut self, name: &str) -> Result<()> {
        match name {
            "help" | "hotkeys" => {
                self.status = "Enter: send | Alt+Enter: queue | Esc: cancel | Ctrl+L: models | Ctrl+T: thinking | Ctrl+O: tools | Ctrl+C x2: quit".to_string();
            }
            "model" => {
                self.display_mode = DisplayMode::ModelList;
                let rows = build_model_rows(ModelRegistry::global());
                let total = rows.len();
                let providers = ModelRegistry::global().total_providers();
                self.model_list = rows;
                self.model_filter.clear();
                self.model_selected = 0;
                self.status = format!(
                    "{} models across {} providers - type to filter, \u{2191}\u{2193} nav, Enter to select, Esc to cancel",
                    total, providers
                );
            }
            "settings" => {
                self.display_mode = DisplayMode::SettingsList;
                self.settings_list = get_settings_list();
                self.settings_selected = 0;
                self.status = "Edit settings - ↑↓ navigate, Enter to edit, Esc to cancel".to_string();
            }
            "new" => {
                self.messages.clear();
                self.queued_messages.clear();
                self.usage = UsageStats::default();
                // Also clear the agent's own message history so the
                // next turn doesn't continue the prior conversation.
                if let Some(agent) = self.agent.as_mut() {
                    // There is no explicit reset; easiest safe path is to
                    // rebuild by dropping the old one into a fresh model.
                    // `/new` should be cheap, but if something is in
                    // flight we let it finish first.
                    if self.active_job.is_none() {
                        let id = agent.model_id().to_string();
                        let new_agent = Agent::new(&id);
                        *agent = new_agent;
                    }
                }
                self.status = "New session started".to_string();
            }
            "tree" => {
                self.tree_rows = build_tree_rows(&self.messages);
                self.tree_selected = 0;
                self.display_mode = DisplayMode::TreeView;
                self.status = format!(
                    "Session tree — {} node(s). ↑↓ nav, Esc back.",
                    self.tree_rows.len()
                );
            }
            "session" => {
                self.status = format!(
                    "Session: {} messages, {} queued",
                    self.messages.len(),
                    self.queued_messages.len()
                );
            }
            "login" => {
                let registry = ModelRegistry::global();
                let saved = crate::auth::load_auth().unwrap_or_default();
                self.login_list = build_login_rows(registry, &saved);
                self.login_selected = 0;
                self.display_mode = DisplayMode::LoginList;
                self.status = "Pick a provider to sign in to. ↑↓ nav, Enter to select, Esc to cancel.".to_string();
            }
            "logout" => {
                // For logout we reuse the login list UI but on Enter we'll
                // clear the key instead of prompting for one. The selection
                // model sets `login_provider_id` as a side-channel marker.
                let registry = ModelRegistry::global();
                let saved = crate::auth::load_auth().unwrap_or_default();
                self.login_list = build_login_rows(registry, &saved);
                self.login_selected = 0;
                self.login_provider_id = Some("<logout>".to_string());
                self.display_mode = DisplayMode::LoginList;
                self.status = "Pick a provider to clear from auth.json. Enter to clear, Esc to cancel.".to_string();
            }
            "thinking" => {
                let current = self.agent.as_ref().and_then(|a| a.thinking_level());
                // Preselect the current level.
                self.thinking_selected = self
                    .thinking_options
                    .iter()
                    .position(|o| thinking_level_from_label(o) == current)
                    .unwrap_or(0);
                self.display_mode = DisplayMode::ThinkingList;
                self.status = "Pick thinking level. Enter to apply, Esc to cancel.".to_string();
            }
            "export" => {
                let ts = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
                let path = std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(format!("pi-session-{ts}.html"));
                match export_conversation_html(&self.messages, &path) {
                    Ok(()) => {
                        self.status = format!("Exported → {}", path.display());
                    }
                    Err(e) => {
                        self.status = format!("Export failed: {e}");
                    }
                }
            }
            "copy" => match copy_last_assistant(&self.messages) {
                Ok(bytes) => {
                    self.status = format!("Copied {} chars to clipboard", bytes);
                }
                Err(e) => {
                    self.status = format!("Copy failed: {e}");
                }
            },
            "fork" => {
                let ts = chrono::Local::now().format("%Y%m%dT%H%M%S").to_string();
                let path = self.sessions_dir.join(format!("fork-{ts}.json"));
                let model = self
                    .agent
                    .as_ref()
                    .map(|a| a.model_id().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                match save_transcript_json(&self.messages, &path, &model) {
                    Ok(()) => {
                        self.status = format!("Forked → {}", path.display());
                    }
                    Err(e) => {
                        self.status = format!("Fork failed: {e}");
                    }
                }
            }
            "compact" => {
                // Render a best-effort inline summary into messages and
                // clear the agent's memory so future turns run on the
                // compacted context. If no agent is available (pre-start)
                // we only summarize locally.
                let summary = compact_summary(&self.messages);
                let removed = self.messages.len();
                self.messages.clear();
                self.messages.push(ConversationMessage {
                    role: "system".to_string(),
                    content: summary.clone(),
                    ..Default::default()
                });
                if let Some(agent) = self.agent.as_mut() {
                    if self.active_job.is_none() {
                        let id = agent.model_id().to_string();
                        *agent = Agent::new(&id);
                    }
                }
                self.status = format!(
                    "Compacted {removed} message(s) into a {} char summary",
                    summary.len()
                );
            }
            "reload" => {
                // Registry is a one-shot statically loaded global; we can
                // at least re-read the auth file and refresh the saved-key
                // column for the login UI.
                let _ = crate::auth::load_auth();
                self.status = "Reloaded auth cache. (Provider TOMLs are loaded at process start.)".to_string();
            }
            "share" => {
                self.status = "/share is not yet wired; use /export to produce an HTML transcript".to_string();
            }
            "quit" => {
                self.status = "Exiting...".to_string();
                self.ctrl_c_count = 2;
            }
            _ => {
                self.status = format!("Unknown command: /{}", name);
            }
        }
        Ok(())
    }

    /// Finalize the login key entry screen: write the provider + key
    /// into `~/.pi/auth.json` and push the key onto the live agent if
    /// we're currently on that provider.
    fn finalize_login(&mut self) {
        let provider_id = match self.login_provider_id.clone() {
            Some(id) if id != "<logout>" => id,
            _ => return,
        };
        let key = std::mem::take(&mut self.login_key_input);
        if key.is_empty() {
            self.status = "No key entered — cancelled".to_string();
            return;
        }
        match crate::auth::set_key(&provider_id, &key) {
            Ok(()) => {
                // If the live agent is routing to this provider already,
                // push the key in immediately so the next turn uses it.
                if let Some(agent) = self.agent.as_mut() {
                    let model = agent.model_id().to_string();
                    if model.starts_with(&format!("{provider_id}/")) {
                        agent.set_api_key(Some(key.clone()));
                    }
                }
                self.status = format!("Saved {provider_id} key to auth.json");
            }
            Err(e) => {
                self.status = format!("Could not save key: {e}");
            }
        }
    }

    /// Push a user message and spawn the agent turn in the background. The
    /// TUI continues rendering "Executing... (Ns)" while the task runs, and
    /// `poll_active_job` on each tick picks up the response as soon as the
    /// provider returns.
    /// Append the user's input to the conversation transcript and kick
    /// off a streaming turn via [`Self::spawn_prompt`]. The streaming
    /// assistant placeholder is created inside `spawn_prompt` so the
    /// indices line up for `poll_active_job`.
    fn send_message(&mut self, text: &str) {
        self.messages.push(ConversationMessage {
            role: "user".to_string(),
            content: text.to_string(),
            ..Default::default()
        });
        self.spawn_prompt(text.to_string());
    }

    /// Short summary of a tool-call's JSON input for inline display. We
    /// surface the most useful single field for the common tools (bash,
    /// read, write, edit, grep, find, ls) and fall back to a single-line
    /// JSON string otherwise.
    #[allow(dead_code)]
    fn _preview_tool_args_placeholder() {}

    /// Redraw the terminal in a style that mirrors the TypeScript pi TUI:
    /// no hard bordered boxes, messages flow vertically in plain text, a
    /// single dim hint row sits just above the editor, and the bottom two
    /// rows are the pwd/branch line and the tokens/model/thinking line.
    ///
    /// Overlay modes (command palette, model selector, settings) drop in
    /// above the editor as inline lists styled the same way — still no
    /// borders, just a titled section with a highlighted selected row.
    fn draw(&mut self) -> Result<()> {
        // Snapshot state for the ratatui closure.
        let messages = self.messages.clone();
        let input_lines = self.editor.visual_lines();
        let (cursor_row, cursor_col) = self.editor.cursor_line_col();
        let status_text = self.status.clone();
        let queued_count = self.queued_messages.len();
        let executing = self.executing;
        let palette_active = self.palette_active;
        let show_thinking = self.show_thinking;
        let show_tools = self.show_tools;
        let palette_items = self.palette_items.clone();
        let palette_selected = self.palette_selected;
        let display_mode = self.display_mode;
        let model_list = self.model_list.clone();
        let model_selected = self.model_selected;
        let model_filter = self.model_filter.clone();
        let settings_list = self.settings_list.clone();
        let settings_selected = self.settings_selected;
        let login_list = self.login_list.clone();
        let login_selected = self.login_selected;
        let login_provider_id = self.login_provider_id.clone();
        let login_key_len = self.login_key_input.chars().count();
        let tree_rows = self.tree_rows.clone();
        let tree_selected = self.tree_selected;
        let thinking_options = self.thinking_options.clone();
        let thinking_selected = self.thinking_selected;
        let thinking_level = self.agent.as_ref().and_then(|a| a.thinking_level());

        // Footer inputs: pwd + branch on one line, token stats + model on
        // the next. These are resolved via the registry so they stay truthy
        // when /model swaps the active provider.
        let pwd = pwd_with_tilde();
        let git_branch = git_branch_for_pwd();
        let active_model_label = self
            .agent
            .as_ref()
            .map(|a| a.model_id().to_string())
            .unwrap_or_else(|| "(no model)".to_string());
        let active_model_info = self
            .agent
            .as_ref()
            .and_then(|a| active_model_summary(a.model_id()));
        let usage = self.usage.clone();
        let context_window: Option<u32> = self
            .agent
            .as_ref()
            .and_then(|a| {
                let model_id = a.model_id();
                let reg = pi_core::model_registry::ModelRegistry::global();
                if let Some((pid, mid)) = model_id.split_once('/') {
                    reg.find_by_provider(pid, mid).or_else(|| reg.find_model(model_id))
                } else {
                    reg.find_model(model_id)
                }
            })
            .map(|(_, m)| m.context_window)
            .filter(|w| *w > 0);
        let first_turn = messages.is_empty() && !palette_active && display_mode == DisplayMode::Chat;

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();
            if size.height < 4 || size.width < 20 {
                return;
            }

            // -------- Bottom 2 rows: footer --------
            let footer_stats_row = size.height - 1;
            let footer_pwd_row = size.height - 2;
            let hint_row = size.height - 3;

            // -------- Editor: 1-N rows above the hint row --------
            let editor_lines_src: Vec<String> = input_lines.clone();
            let editor_height = (editor_lines_src.len() as u16).clamp(1, 6);
            let editor_top = hint_row.saturating_sub(editor_height);

            // -------- Overlay area above editor (command palette / model
            //          list / settings list). Takes up to ~60% of screen. --------
            let overlay_max_h = (size.height as f32 * 0.55) as u16;
            let overlay_active = palette_active
                || display_mode == DisplayMode::ModelList
                || display_mode == DisplayMode::SettingsList
                || display_mode == DisplayMode::LoginList
                || display_mode == DisplayMode::LoginKeyEntry
                || display_mode == DisplayMode::TreeView
                || display_mode == DisplayMode::ThinkingList;
            let overlay_height: u16 = if overlay_active {
                match display_mode {
                    DisplayMode::ModelList => (model_list.len() as u16 + 3).min(overlay_max_h),
                    DisplayMode::SettingsList => (settings_list.len() as u16 + 3).min(overlay_max_h),
                    DisplayMode::LoginList => (login_list.len() as u16 + 3).min(overlay_max_h),
                    DisplayMode::LoginKeyEntry => 6,
                    DisplayMode::TreeView => (tree_rows.len() as u16 + 3).min(overlay_max_h),
                    DisplayMode::ThinkingList => (thinking_options.len() as u16 + 3).min(overlay_max_h),
                    _ if palette_active => (palette_items.len() as u16 + 3).min(overlay_max_h),
                    _ => 0,
                }
            } else {
                0
            };
            let overlay_top = editor_top.saturating_sub(overlay_height);

            // -------- Messages area: top down to overlay --------
            let msg_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: overlay_top,
            };

            // ===== Messages / startup banner =====
            if first_turn {
                let lines = render_startup_banner();
                let para = Paragraph::new(lines).wrap(Wrap { trim: false });
                frame.render_widget(para, msg_area);
            } else {
                let lines = render_messages(&messages, show_thinking, show_tools);
                // Pin the most recent messages to the BOTTOM of the
                // messages area so the editor feels glued to the live end
                // of the conversation, matching the TS flow. Older lines
                // scroll off the top.
                let avail = msg_area.height as usize;
                let n = lines.len();
                let slice: Vec<Line> = if n >= avail {
                    lines.into_iter().skip(n - avail).collect()
                } else {
                    let pad = avail - n;
                    let mut out: Vec<Line> = (0..pad).map(|_| Line::from("")).collect();
                    out.extend(lines);
                    out
                };
                let para = Paragraph::new(slice).wrap(Wrap { trim: false });
                frame.render_widget(para, msg_area);
            }

            // ===== Overlay (command palette / model list / settings) =====
            if overlay_active {
                let overlay_area = Rect {
                    x: 0,
                    y: overlay_top,
                    width: size.width,
                    height: overlay_height,
                };
                let overlay_lines = if palette_active {
                    render_palette(&palette_items, palette_selected)
                } else if display_mode == DisplayMode::ModelList {
                    render_model_list(&model_list, model_selected, &model_filter)
                } else if display_mode == DisplayMode::SettingsList {
                    render_settings_list(&settings_list, settings_selected)
                } else if display_mode == DisplayMode::LoginList {
                    render_login_list(&login_list, login_selected)
                } else if display_mode == DisplayMode::LoginKeyEntry {
                    let provider_label = login_provider_id
                        .as_deref()
                        .unwrap_or("provider")
                        .to_string();
                    render_login_key_entry(&provider_label, login_key_len)
                } else if display_mode == DisplayMode::TreeView {
                    render_tree(&tree_rows, tree_selected)
                } else if display_mode == DisplayMode::ThinkingList {
                    render_thinking_list(&thinking_options, thinking_selected, thinking_level)
                } else {
                    Vec::new()
                };
                let para = Paragraph::new(overlay_lines).wrap(Wrap { trim: false });
                frame.render_widget(para, overlay_area);
            }

            // ===== Editor =====
            let editor_area = Rect {
                x: 0,
                y: editor_top,
                width: size.width,
                height: editor_height,
            };
            let editor_lines_rendered: Vec<Line> = editor_lines_src
                .iter()
                .enumerate()
                .map(|(i, line)| {
                    let prompt = if i == 0 {
                        if executing {
                            Span::styled("▌ ", Style::default().fg(Color::Yellow))
                        } else {
                            Span::styled("▌ ", Style::default().fg(Color::Cyan))
                        }
                    } else {
                        Span::styled("  ", Style::default())
                    };
                    // Slice the line at the cursor column and render the
                    // cell under the cursor with an inverted style so the
                    // user sees where edits will land. Only do this on
                    // the row that actually holds the cursor and only when
                    // overlays/lists aren't taking focus.
                    let cursor_here = i == cursor_row && !overlay_active;
                    if cursor_here {
                        let mut before = String::new();
                        let mut under = String::from(" ");
                        let mut after = String::new();
                        let mut seen = 0usize;
                        for (_, ch) in line.char_indices() {
                            if seen < cursor_col {
                                before.push(ch);
                            } else if seen == cursor_col {
                                under = ch.to_string();
                            } else {
                                after.push(ch);
                            }
                            seen += 1;
                        }
                        Line::from(vec![
                            prompt,
                            Span::raw(before),
                            Span::styled(
                                under,
                                Style::default().bg(Color::Gray).fg(Color::Black),
                            ),
                            Span::raw(after),
                        ])
                    } else {
                        Line::from(vec![prompt, Span::raw(line.clone())])
                    }
                })
                .collect();
            let editor = Paragraph::new(editor_lines_rendered).wrap(Wrap { trim: false });
            frame.render_widget(editor, editor_area);

            // ===== Hint row (just above editor) =====
            let hint_area = Rect { x: 0, y: hint_row, width: size.width, height: 1 };
            let hint = if executing {
                render_executing_hint(&status_text, queued_count)
            } else if palette_active {
                render_hint_line_palette()
            } else if display_mode == DisplayMode::ModelList {
                render_hint_line_model_list()
            } else if display_mode == DisplayMode::SettingsList {
                render_hint_line_settings_list()
            } else {
                render_hint_line_chat(&status_text, queued_count)
            };
            frame.render_widget(Paragraph::new(hint), hint_area);

            // ===== Footer line 1: pwd (git-branch) • session =====
            let mut pwd_display = pwd.clone();
            if let Some(branch) = git_branch.as_deref() {
                pwd_display = format!("{pwd_display} ({branch})");
            }
            let pwd_line = Line::from(Span::styled(pwd_display, dim_style()));
            frame.render_widget(
                Paragraph::new(pwd_line),
                Rect { x: 0, y: footer_pwd_row, width: size.width, height: 1 },
            );

            // ===== Footer line 2: tokens left, model right =====
            // Build the stats segment the same way the TS footer does:
            //   ↑<input>  ↓<output>  R<cache-read>  W<cache-write>  <ctx%>/<window>
            // Each counter omitted when zero. Percent colored at 70% /
            // 90% thresholds against the resolved context window.
            let mut stats_parts: Vec<Span<'static>> = Vec::new();
            let mut push_part = |parts: &mut Vec<Span<'static>>, prefix: &str, n: u64, style: Style| {
                if n == 0 {
                    return;
                }
                if !parts.is_empty() {
                    parts.push(Span::raw(" "));
                }
                parts.push(Span::styled(format!("{prefix}{}", format_tokens(n)), style));
            };
            push_part(&mut stats_parts, "↑", usage.input_tokens, dim_style());
            push_part(&mut stats_parts, "↓", usage.output_tokens, dim_style());
            push_part(&mut stats_parts, "R", usage.cache_read_tokens, dim_style());
            push_part(&mut stats_parts, "W", usage.cache_write_tokens, dim_style());

            // Context percent: compute only if we know the window. Color
            // matches TS: warning > 70%, error > 90%, otherwise dim.
            let (pct_value, pct_display) = match context_window {
                Some(window) if window > 0 => {
                    let pct = (usage.total_tokens as f64 / window as f64) * 100.0;
                    (Some(pct), format!("{:.1}%/{}", pct, format_tokens(window as u64)))
                }
                _ => (None, format!("?/{}", context_window.map(|w| format_tokens(w as u64)).unwrap_or_else(|| "?".to_string()))),
            };
            let pct_style = match pct_value {
                Some(v) if v > 90.0 => Style::default().fg(Color::Red),
                Some(v) if v > 70.0 => Style::default().fg(Color::Yellow),
                _ => dim_style(),
            };
            if !stats_parts.is_empty() {
                stats_parts.push(Span::raw(" "));
            }
            stats_parts.push(Span::styled(pct_display, pct_style));
            stats_parts.push(Span::styled(" (auto)".to_string(), dim_style()));

            let model_segment = match &active_model_info {
                Some((provider, display, thinking)) => {
                    let base = format!("({}) {}", provider, display);
                    match thinking {
                        Some(level) => format!("{base} \u{00b7} {level}"),
                        None => base,
                    }
                }
                None => format!("(unknown) {active_model_label}"),
            };

            let stats_left_w: usize = stats_parts
                .iter()
                .map(|s| s.content.chars().count())
                .sum();
            let right_w = model_segment.chars().count();
            let total_w = size.width as usize;
            let pad = total_w.saturating_sub(stats_left_w + right_w).max(2);
            stats_parts.push(Span::raw(" ".repeat(pad)));
            stats_parts.push(Span::styled(model_segment, dim_style()));
            let stats_line = Line::from(stats_parts);
            frame.render_widget(
                Paragraph::new(stats_line),
                Rect { x: 0, y: footer_stats_row, width: size.width, height: 1 },
            );
        })?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers — kept as free functions so draw() stays readable and so
// individual pieces can be unit-tested without spinning up a terminal.
// ---------------------------------------------------------------------------

fn dim_style() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn muted_style() -> Style {
    Style::default().fg(Color::Gray)
}

fn accent_style() -> Style {
    Style::default().fg(Color::Cyan)
}

/// Build the startup banner that the TS version shows on a fresh session.
/// Structure (each line a separate Line):
///   pi v<version>                                 ← accent + dim
///   escape interrupt · ctrl+c/ctrl+d clear/exit · …  ← key hints joined by muted ·
///   Press ctrl+o to show full startup help …       ← dim
///   (blank)
///   Pi can explain its own features and look up …  ← dim
fn render_startup_banner() -> Vec<Line<'static>> {
    fn key(name: &str) -> Span<'static> {
        Span::styled(name.to_string(), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    }
    fn desc(text: &str) -> Span<'static> {
        Span::styled(text.to_string(), Style::default())
    }
    fn sep() -> Span<'static> {
        Span::styled(" \u{00b7} ".to_string(), muted_style())
    }

    let version = env!("CARGO_PKG_VERSION");
    let title_line = Line::from(vec![
        Span::styled("pi".to_string(), accent_style().add_modifier(Modifier::BOLD)),
        Span::styled(format!(" v{version}"), dim_style()),
    ]);

    let hints_line = Line::from(vec![
        key("escape"),
        Span::raw(" "),
        desc("interrupt"),
        sep(),
        key("ctrl+c"),
        Span::raw("/"),
        key("ctrl+d"),
        Span::raw(" "),
        desc("clear/exit"),
        sep(),
        key("/"),
        Span::raw(" "),
        desc("commands"),
        sep(),
        key("ctrl+l"),
        Span::raw(" "),
        desc("model"),
        sep(),
        key("ctrl+o"),
        Span::raw(" "),
        desc("more"),
    ]);

    let press_more = Line::from(Span::styled(
        "Press ctrl+o to show full startup help and loaded resources.".to_string(),
        dim_style(),
    ));

    let onboarding = Line::from(Span::styled(
        "Pi can explain its own features and look up its docs. Ask it how to use or extend Pi."
            .to_string(),
        dim_style(),
    ));

    vec![
        Line::from(""),
        title_line,
        hints_line,
        press_more,
        Line::from(""),
        onboarding,
        Line::from(""),
    ]
}

/// Render conversation messages the same way the TS version does: user
/// messages in accent cyan prefixed with `You`, assistant text plain, and
/// spacer rows between turns.
/// Render conversation messages the same way the TS version does: user
/// messages in accent cyan prefixed with `You`, assistant text plain, and
/// spacer rows between turns. When `show_thinking` is true, thinking
/// blocks are rendered inline in dim italic. Tool calls always render
/// inline (matching TS behavior: tool-execution components always show).
fn render_messages(
    messages: &[ConversationMessage],
    show_thinking: bool,
    show_tools: bool,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for msg in messages {
        if msg.role == "user" {
            // Match the TS user-message visual: cyan bold prefix, text in
            // the default color with a subtle › divider. A full-width
            // background tint would need the frame width threaded into
            // the renderer; we keep it visually distinct via prefix
            // color alone, which also plays nicer with copy/paste out
            // of the terminal.
            let prefix_style = Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD);
            for (i, text_line) in msg.content.lines().enumerate() {
                if i == 0 {
                    lines.push(Line::from(vec![
                        Span::styled("You".to_string(), prefix_style),
                        Span::styled(" \u{203a} ".to_string(), muted_style()),
                        Span::raw(text_line.to_string()),
                    ]));
                } else {
                    lines.push(Line::from(vec![
                        Span::raw("    ".to_string()),
                        Span::raw(text_line.to_string()),
                    ]));
                }
            }
            lines.push(Line::from(""));
            continue;
        }

        // Assistant message: thinking (optional) → tool calls → text.
        if show_thinking && !msg.thinking.trim().is_empty() {
            lines.push(Line::from(Span::styled(
                "● thinking".to_string(),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )));
            for t_line in msg.thinking.lines() {
                lines.push(Line::from(vec![
                    Span::styled("  ".to_string(), dim_style()),
                    Span::styled(
                        t_line.to_string(),
                        Style::default()
                            .fg(Color::Gray)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
            lines.push(Line::from(""));
        }

        for call in &msg.tool_calls {
            // Background shades mirror the TS theme:
            //   pending -> `toolPendingBg` (#282832) → Indexed(236)
            //   success -> `toolSuccessBg` (#283228) → Indexed(22)
            //   error   -> `toolErrorBg`   (#3c2828) → Indexed(52)
            let (icon, icon_style, bg_color) = match (&call.output, call.is_error) {
                (None, _) => ("○", Style::default().fg(Color::Yellow), Color::Indexed(236)),
                (Some(_), false) => ("✓", Style::default().fg(Color::Green), Color::Indexed(22)),
                (Some(_), true) => ("✗", Style::default().fg(Color::Red), Color::Indexed(52)),
            };
            let bg = Style::default().bg(bg_color);
            let header = Line::from(vec![
                Span::styled(format!(" {icon} "), icon_style.bg(bg_color)),
                Span::styled(
                    call.name.clone(),
                    bg.fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ".to_string(), bg),
                Span::styled(call.input_preview.clone(), bg.fg(Color::Gray)),
            ]);
            lines.push(header);
            if show_tools {
                if let Some(output) = &call.output {
                    let preview_max_lines = 20usize;
                    let mut shown = 0;
                    for out_line in output.lines().take(preview_max_lines) {
                        lines.push(Line::from(vec![
                            Span::styled("  │ ".to_string(), bg.fg(Color::Gray)),
                            Span::styled(out_line.to_string(), bg.fg(Color::Gray)),
                        ]));
                        shown += 1;
                    }
                    let total = output.lines().count();
                    if total > preview_max_lines {
                        lines.push(Line::from(Span::styled(
                            format!("  │ … {} more line(s) (ctrl+o to toggle)", total - shown),
                            bg.fg(Color::DarkGray),
                        )));
                    }
                }
            } else if call.output.is_some() {
                let first_line = call
                    .output
                    .as_deref()
                    .and_then(|s| s.lines().next())
                    .unwrap_or("")
                    .chars()
                    .take(80)
                    .collect::<String>();
                if !first_line.is_empty() {
                    lines.push(Line::from(vec![
                        Span::styled("  │ ".to_string(), bg.fg(Color::Gray)),
                        Span::styled(first_line, bg.fg(Color::Gray)),
                    ]));
                    lines.push(Line::from(Span::styled(
                        "  │ (ctrl+o to expand)".to_string(),
                        bg.fg(Color::DarkGray),
                    )));
                }
            }
        }

        if !msg.content.is_empty() {
            // Assistant text is rendered as markdown so **bold**, `code`,
            // fenced blocks, links, and lists appear styled rather than
            // as raw punctuation.
            for rendered in crate::markdown::render_markdown(&msg.content) {
                lines.push(rendered);
            }
        } else if msg.streaming && msg.tool_calls.is_empty() {
            // Nothing visible yet but the turn is live. Show a pulsing
            // caret so the user sees the TUI hasn't frozen.
            lines.push(Line::from(Span::styled(
                "…".to_string(),
                dim_style(),
            )));
        }
        lines.push(Line::from(""));
    }
    lines
}

/// Keybinding hint line shown just above the editor.
fn render_hint_line_chat(status: &str, queued: usize) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = vec![
        Span::styled("enter".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" send "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("alt+enter".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" queue "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("/".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" commands "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("ctrl+l".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" model "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("ctrl+c".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" clear"),
    ];
    if queued > 0 {
        spans.push(Span::styled("   ".to_string(), muted_style()));
        spans.push(Span::styled(
            format!("{queued} queued"),
            Style::default().fg(Color::Yellow),
        ));
    }
    if !status.is_empty() && status != "Ready" {
        spans.push(Span::styled("   ".to_string(), muted_style()));
        spans.push(Span::styled(status.to_string(), muted_style()));
    }
    Line::from(spans).style(Style::default())
}

fn render_executing_hint(_status: &str, queued: usize) -> Line<'static> {
    let mut spans = vec![
        Span::styled("○ executing".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw("  "),
        Span::styled("esc".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" cancel"),
    ];
    if queued > 0 {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(
            format!("{queued} queued"),
            Style::default().fg(Color::Yellow),
        ));
    }
    Line::from(spans)
}

fn render_hint_line_palette() -> Line<'static> {
    Line::from(vec![
        Span::styled("↑↓".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" navigate "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("enter".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" run "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("tab".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" complete "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("esc".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" dismiss"),
    ])
}

fn render_hint_line_model_list() -> Line<'static> {
    Line::from(vec![
        Span::styled("↑↓".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" navigate "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("type".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" filter "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("enter".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" select "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("esc".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" cancel"),
    ])
}

fn render_hint_line_settings_list() -> Line<'static> {
    Line::from(vec![
        Span::styled("↑↓".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" navigate "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("enter".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" select "),
        Span::styled("\u{00b7}".to_string(), muted_style()),
        Span::raw(" "),
        Span::styled("esc".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" cancel"),
    ])
}

fn render_palette(items: &[SlashCommand], selected: usize) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!("Commands ({})", items.len()),
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    ))];
    for (i, cmd) in items.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "› " } else { "  " };
        let name_style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(format!("/{:<10}", cmd.name), name_style),
            Span::raw("  "),
            Span::styled(cmd.description.clone(), dim_style()),
        ]));
    }
    lines
}

fn render_model_list(
    rows: &[ModelRow],
    selected: usize,
    filter: &str,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("Models ({})", rows.len()),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::styled(
            if filter.is_empty() { String::new() } else { format!("filter: {filter}") },
            muted_style(),
        ),
    ])];
    let mut current_provider = String::new();
    for (i, m) in rows.iter().enumerate() {
        if m.provider_id != current_provider {
            current_provider = m.provider_id.clone();
            let env_tag = if m.env_configured { "auth ok" } else { "no auth" };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{} ", m.provider_display.to_uppercase()),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("[{env_tag}]"), dim_style()),
            ]));
        }
        let is_sel = i == selected;
        let prefix = if is_sel { "› " } else { "  " };
        let name_style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else if !m.env_configured {
            dim_style()
        } else {
            Style::default()
        };
        let ctx_w = format_context_window(m.context_window);
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(format!("{:<36}", m.model_name), name_style),
            Span::raw(" "),
            Span::styled(format!("{ctx_w:>6}"), muted_style()),
            Span::raw("  "),
            Span::styled(m.model_id.clone(), dim_style()),
        ]));
    }
    lines
}

fn render_settings_list(rows: &[SettingInfo], selected: usize) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!("Settings ({})", rows.len()),
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    ))];
    for (i, s) in rows.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "› " } else { "  " };
        let name_style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(format!("{:<22}", s.name), name_style),
            Span::raw("  "),
            Span::styled(format!("= {:<18}", s.current), Style::default()),
            Span::raw("  "),
            Span::styled(s.options.clone(), dim_style()),
        ]));
    }
    lines
}

// ----- LoginRow + build/render helpers + tree rows + thinking list + export

#[derive(Clone, Debug)]
pub struct LoginRow {
    pub provider_id: String,
    pub display_name: String,
    pub source: String,
    pub env_hint: String,
}

fn build_login_rows(
    registry: &ModelRegistry,
    saved: &std::collections::BTreeMap<String, String>,
) -> Vec<LoginRow> {
    let mut rows: Vec<LoginRow> = Vec::new();
    for (id, provider) in registry.providers_iter() {
        let env_hit = provider.resolve_env_key().map(|(var, _)| var);
        let source = if let Some(var) = env_hit.clone() {
            format!("env {var}")
        } else if saved.contains_key(id) {
            "auth.json".to_string()
        } else {
            "-".to_string()
        };
        let env_hint = if provider.env_vars.is_empty() {
            "(no env var)".to_string()
        } else {
            provider.env_vars.join(", ")
        };
        rows.push(LoginRow {
            provider_id: id.clone(),
            display_name: if provider.display_name.is_empty() {
                id.clone()
            } else {
                provider.display_name.clone()
            },
            source,
            env_hint,
        });
    }
    rows.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()));
    rows
}

fn render_login_list(rows: &[LoginRow], selected: usize) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!("Sign in ({} providers)", rows.len()),
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    ))];
    for (i, r) in rows.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "› " } else { "  " };
        let name_style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        let source_style = if r.source == "-" {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::Green)
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(format!("{:<24}", r.display_name), name_style),
            Span::raw("  "),
            Span::styled(format!("{:<14}", r.source), source_style),
            Span::raw("  "),
            Span::styled(r.env_hint.clone(), dim_style()),
        ]));
    }
    lines
}

fn render_login_key_entry(provider: &str, key_len: usize) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            format!("Sign in to {provider}"),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Paste your API key and press Enter. Esc to cancel.".to_string(),
            dim_style(),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("key › ".to_string(), Style::default().fg(Color::Cyan)),
            Span::styled(
                "*".repeat(key_len.min(48)),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                if key_len > 48 { format!(" ({} more)", key_len - 48) } else { String::new() },
                dim_style(),
            ),
        ]),
    ]
}

#[derive(Clone, Debug)]
pub struct TreeRow {
    pub depth: usize,
    pub icon: String,
    pub label: String,
    pub hint: String,
}

fn build_tree_rows(messages: &[ConversationMessage]) -> Vec<TreeRow> {
    let mut rows: Vec<TreeRow> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let (icon, label) = match m.role.as_str() {
            "user" => ("▶".to_string(), truncate_first_line(&m.content, 80)),
            "assistant" => ("◀".to_string(), truncate_first_line(&m.content, 80)),
            _ => ("·".to_string(), truncate_first_line(&m.content, 80)),
        };
        let hint = format!("#{i}");
        rows.push(TreeRow { depth: 0, icon, label, hint });
        for tc in &m.tool_calls {
            let ok = if tc.is_error { "error" } else if tc.output.is_some() { "ok" } else { "…" };
            rows.push(TreeRow {
                depth: 1,
                icon: "↪".to_string(),
                label: format!("{} — {}", tc.name, tc.input_preview),
                hint: ok.to_string(),
            });
        }
    }
    rows
}

fn truncate_first_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() <= max {
        line.to_string()
    } else {
        let truncated: String = line.chars().take(max).collect();
        format!("{truncated}…")
    }
}

fn render_tree(rows: &[TreeRow], selected: usize) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!("Session tree ({} nodes)", rows.len()),
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    ))];
    if rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (transcript is empty)".to_string(),
            dim_style(),
        )));
        return lines;
    }
    for (i, r) in rows.iter().enumerate() {
        let is_sel = i == selected;
        let indent = "  ".repeat(r.depth);
        let prefix = if is_sel { "› " } else { "  " };
        let label_style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::raw(indent),
            Span::styled(format!("{} ", r.icon), Style::default().fg(Color::Cyan)),
            Span::styled(r.label.clone(), label_style),
            Span::raw("  "),
            Span::styled(r.hint.clone(), dim_style()),
        ]));
    }
    lines
}

fn render_thinking_list(
    options: &[&'static str],
    selected: usize,
    current: Option<pi_ai::types::ThinkingLevel>,
) -> Vec<Line<'static>> {
    let current_str = match current {
        Some(pi_ai::types::ThinkingLevel::Minimal) => "minimal",
        Some(pi_ai::types::ThinkingLevel::Low) => "low",
        Some(pi_ai::types::ThinkingLevel::Medium) => "medium",
        Some(pi_ai::types::ThinkingLevel::High) => "high",
        Some(pi_ai::types::ThinkingLevel::Xhigh) => "xhigh",
        None => "off",
    };
    let mut lines = vec![Line::from(Span::styled(
        format!("Thinking level (current: {current_str})"),
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    ))];
    for (i, opt) in options.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "› " } else { "  " };
        let style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        let marker = if *opt == current_str { " (active)" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(opt.to_string(), style),
            Span::styled(marker.to_string(), dim_style()),
        ]));
    }
    lines
}

fn thinking_level_from_label(label: &str) -> Option<pi_ai::types::ThinkingLevel> {
    match label {
        "off" => None,
        "minimal" => Some(pi_ai::types::ThinkingLevel::Minimal),
        "low" => Some(pi_ai::types::ThinkingLevel::Low),
        "medium" => Some(pi_ai::types::ThinkingLevel::Medium),
        "high" => Some(pi_ai::types::ThinkingLevel::High),
        "xhigh" => Some(pi_ai::types::ThinkingLevel::Xhigh),
        _ => None,
    }
}

fn default_sessions_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".pi").join("sessions")
    } else {
        PathBuf::from(".pi/sessions")
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn export_conversation_html(messages: &[ConversationMessage], path: &Path) -> Result<()> {
    let mut body = String::new();
    for m in messages {
        let role_class = match m.role.as_str() {
            "user" => "user",
            "assistant" => "assistant",
            _ => "system",
        };
        body.push_str(&format!("<section class=\"msg {role_class}\">\n"));
        body.push_str(&format!(
            "<header class=\"role\">{}</header>\n",
            html_escape(&m.role)
        ));
        if !m.thinking.is_empty() {
            body.push_str(&format!(
                "<details class=\"thinking\"><summary>thinking</summary><pre>{}</pre></details>\n",
                html_escape(&m.thinking)
            ));
        }
        for tc in &m.tool_calls {
            body.push_str("<div class=\"tool\">\n");
            body.push_str(&format!(
                "<div class=\"tool-name\">{}</div>\n",
                html_escape(&tc.name)
            ));
            body.push_str(&format!(
                "<pre class=\"tool-args\">{}</pre>\n",
                html_escape(&tc.input_preview)
            ));
            if let Some(out) = &tc.output {
                body.push_str(&format!(
                    "<pre class=\"tool-out\">{}</pre>\n",
                    html_escape(out)
                ));
            }
            body.push_str("</div>\n");
        }
        if !m.content.is_empty() {
            body.push_str(&format!(
                "<div class=\"content\"><pre>{}</pre></div>\n",
                html_escape(&m.content)
            ));
        }
        body.push_str("</section>\n");
    }
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let html = format!(
        r#"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>pi session {ts}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0e1116; color: #e6edf3; padding: 1.5rem; max-width: 980px; margin: auto; }}
.msg {{ border-top: 1px solid #30363d; padding: 1rem 0; }}
.msg.user {{ background: rgba(128,200,255,0.05); }}
.msg.assistant {{ background: rgba(128,255,200,0.03); }}
.role {{ font-weight: 600; color: #79c0ff; margin-bottom: .5rem; text-transform: uppercase; font-size: .75rem; letter-spacing: .05em; }}
pre {{ background: #161b22; padding: .75rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }}
.tool-name {{ font-weight: 600; color: #7ee787; }}
.tool-args {{ color: #d2a8ff; }}
.tool-out {{ color: #c9d1d9; }}
.thinking {{ color: #8b949e; }}
header.page {{ color: #8b949e; font-size: .85rem; margin-bottom: 1rem; }}
</style>
</head><body>
<header class="page">pi session — {ts} — {count} message(s)</header>
{body}
</body></html>
"#,
        ts = ts,
        count = messages.len(),
        body = body,
    );
    fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))?;
    fs::write(path, html)?;
    Ok(())
}

fn copy_last_assistant(messages: &[ConversationMessage]) -> Result<usize> {
    let last = messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && !m.content.is_empty())
        .ok_or_else(|| anyhow::anyhow!("no assistant message to copy"))?;
    let text = last.content.clone();
    let bytes = text.len();
    let mut clip = arboard::Clipboard::new()
        .map_err(|e| anyhow::anyhow!("clipboard unavailable: {e}"))?;
    clip.set_text(text)
        .map_err(|e| anyhow::anyhow!("clipboard write failed: {e}"))?;
    Ok(bytes)
}

fn compact_summary(messages: &[ConversationMessage]) -> String {
    let mut lines = Vec::new();
    let mut tool_count = 0usize;
    for m in messages {
        if !m.tool_calls.is_empty() {
            tool_count += m.tool_calls.len();
        }
        let role = match m.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            "system" => "System",
            _ => "?",
        };
        let first = m.content.lines().next().unwrap_or("").trim();
        if !first.is_empty() {
            let clipped: String = first.chars().take(160).collect();
            lines.push(format!("{role}: {clipped}"));
        }
    }
    format!(
        "[compact] {n} message(s), {t} tool call(s)\n{body}",
        n = messages.len(),
        t = tool_count,
        body = lines.join("\n"),
    )
}

fn save_transcript_json(messages: &[ConversationMessage], path: &Path, model: &str) -> Result<()> {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content,
                "thinking": m.thinking,
                "tool_calls": m.tool_calls.iter().map(|tc| serde_json::json!({
                    "name": tc.name,
                    "input_preview": tc.input_preview,
                    "output": tc.output,
                    "is_error": tc.is_error,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "id": chrono::Local::now().format("%Y%m%dT%H%M%S").to_string(),
        "model": model,
        "created": chrono::Local::now().to_rfc3339(),
        "messages": msgs,
    });
    fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))?;
    fs::write(path, serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}


/// Short summary of a tool-call's JSON input for inline display. We
/// surface the most useful single field for the common tools (bash,
/// read, write, edit, grep, find, ls) and fall back to a single-line
/// JSON string otherwise.
fn preview_tool_args(input: &serde_json::Value) -> String {
    fn first_str<'a>(obj: &'a serde_json::Value, keys: &[&str]) -> Option<&'a str> {
        for k in keys {
            if let Some(v) = obj.get(*k).and_then(|v| v.as_str()) {
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
        None
    }

    if let Some(s) = first_str(
        input,
        &["command", "path", "file_path", "pattern", "query", "url"],
    ) {
        let one_line = s.replace(['\n', '\r'], " ");
        if one_line.chars().count() > 80 {
            let truncated: String = one_line.chars().take(77).collect();
            return format!("{truncated}...");
        }
        return one_line;
    }

    // Fallback: serialize as compact JSON, truncated.
    let s = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
    if s.chars().count() > 80 {
        let truncated: String = s.chars().take(77).collect();
        format!("{truncated}...")
    } else {
        s
    }
}

/// Format a token count with `k` / `M` suffix the same way the TS
/// footer does. Matches `formatTokens` in
/// `packages/coding-agent/src/modes/interactive/components/footer.ts`.
fn format_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else if n < 1_000_000 {
        format!("{}k", ((n as f64) / 1_000.0).round() as u64)
    } else if n < 10_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else {
        format!("{}M", ((n as f64) / 1_000_000.0).round() as u64)
    }
}

fn format_context_window(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else if n > 0 {
        format!("{n}")
    } else {
        "?".to_string()
    }
}

/// Replace $HOME in pwd with ~ for the footer line.
fn pwd_with_tilde() -> String {
    let cwd = std::env::current_dir().ok().unwrap_or_default();
    let cwd = cwd.to_string_lossy().to_string();
    if let Some(home) = std::env::var("HOME").ok() {
        if !home.is_empty() && cwd.starts_with(&home) {
            return format!("~{}", &cwd[home.len()..]);
        }
    }
    cwd
}

/// Best-effort current git branch for the pwd. Returns None if the
/// directory isn't in a git checkout or `HEAD` doesn't parse.
fn git_branch_for_pwd() -> Option<String> {
    let git_head = std::env::current_dir().ok()?.join(".git").join("HEAD");
    let contents = std::fs::read_to_string(&git_head).ok()?;
    let trimmed = contents.trim();
    // Typical contents: `ref: refs/heads/our.pi.v2` or a 40-char sha for detached HEAD.
    if let Some(rest) = trimmed.strip_prefix("ref: refs/heads/") {
        Some(rest.to_string())
    } else if trimmed.len() == 40 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(format!("detached@{}", &trimmed[..7]))
    } else {
        None
    }
}

/// Look up `(provider_display, model_display, thinking_level)` for the
/// footer line using the global registry. The third element is `None` if
/// the model doesn't declare `reasoning = true`.
fn active_model_summary(
    model_id: &str,
) -> Option<(String, String, Option<String>)> {
    let registry = pi_core::model_registry::ModelRegistry::global();
    let (provider, model) = if let Some((pid, mid)) = model_id.split_once('/') {
        registry.find_by_provider(pid, mid).or_else(|| registry.find_model(model_id))?
    } else {
        registry.find_model(model_id)?
    };
    let thinking = if model.reasoning {
        Some("medium".to_string())
    } else {
        None
    };
    Some((provider.id.clone(), model.id.clone(), thinking))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_tokens_matches_typescript_thresholds() {
        // Boundary conditions identical to formatTokens() in TS footer.
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_000), "1.0k");
        assert_eq!(format_tokens(1_234), "1.2k");
        assert_eq!(format_tokens(9_999), "10.0k");
        assert_eq!(format_tokens(10_000), "10k");
        assert_eq!(format_tokens(123_456), "123k");
        assert_eq!(format_tokens(999_999), "1000k");
        assert_eq!(format_tokens(1_000_000), "1.0M");
        assert_eq!(format_tokens(2_500_000), "2.5M");
        assert_eq!(format_tokens(10_000_000), "10M");
        assert_eq!(format_tokens(12_345_678), "12M");
    }

    #[test]
    fn preview_tool_args_prefers_common_fields() {
        let bash = serde_json::json!({"command": "ls -la"});
        assert_eq!(preview_tool_args(&bash), "ls -la");

        let read = serde_json::json!({"file_path": "/tmp/a.txt", "offset": 0});
        assert_eq!(preview_tool_args(&read), "/tmp/a.txt");

        // Falls back to compact JSON when none of the known keys match.
        let misc = serde_json::json!({"foo": "bar"});
        assert_eq!(preview_tool_args(&misc), r#"{"foo":"bar"}"#);

        // Long strings get truncated with an ellipsis.
        let long = serde_json::json!({"command": "x".repeat(200)});
        let rendered = preview_tool_args(&long);
        assert!(rendered.ends_with("..."), "{rendered}");
        assert!(rendered.chars().count() <= 80);
    }
}
