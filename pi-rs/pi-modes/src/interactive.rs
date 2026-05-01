//! Interactive mode - TUI matching TypeScript variant
//! - Enter to send message
//! - Alt+Enter to queue follow-up
//! - Escape to cancel
//! - Ctrl+L for model selector
//! - Ctrl+T for thinking toggle
//! - Ctrl+O for tool output toggle
//! - Ctrl+C to clear, Ctrl+C twice to quit
//! - / to open command palette

use anyhow::Result;
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
    input_text: String,
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
            input_text: String::new(),
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
                        self.input_text.clear();
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
                } else if !self.input_text.is_empty() {
                    // Send message
                    let msg = self.input_text.trim().to_string();
                    if msg.starts_with('/') {
                        let cmd_name = msg.trim_start_matches('/').to_string();
                        self.execute_command(&cmd_name).await?;
                    } else {
                        self.send_message(&msg);
                    }
                    self.input_text.clear();
                }
                Ok(true)
            }

            // Alt+Enter: queue message
            KeyCommand::AltEnter => {
                if !self.input_text.is_empty() {
                    self.queued_messages.push(self.input_text.trim().to_string());
                    self.status = format!("Queued ({})", self.queued_messages.len());
                    self.input_text.clear();
                }
                Ok(true)
            }

            // Ctrl+C: clear or quit
            KeyCommand::CtrlC => {
                self.ctrl_c_count += 1;
                if self.ctrl_c_count >= 2 {
                    return Ok(false);
                }
                if !self.input_text.is_empty() {
                    self.input_text.clear();
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
                    self.input_text.clear();
                    self.status = "Palette closed".to_string();
                } else if self.display_mode != DisplayMode::Chat {
                    self.display_mode = DisplayMode::Chat;
                    self.model_filter.clear();
                    self.status = "Back to chat".to_string();
                } else if !self.input_text.is_empty() {
                    self.input_text.clear();
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

                self.input_text.push(c);

                // Activate palette on first /
                if self.input_text == "/" {
                    self.palette_active = true;
                    self.palette_items = self.all_commands.clone();
                    self.palette_selected = 0;
                    self.status = "Type to filter commands, \u{2191}\u{2193} to navigate, Enter to select".to_string();
                } else if self.palette_active && self.input_text.starts_with('/') {
                    // Filter as user types
                    let query = &self.input_text[1..];
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

                self.input_text.pop();
                if self.palette_active {
                    if self.input_text.is_empty() {
                        self.close_palette();
                    } else if self.input_text.starts_with('/') {
                        let query = &self.input_text[1..];
                        self.palette_items = filter_commands(query, &self.all_commands);
                        self.palette_selected = 0;
                    } else {
                        self.close_palette();
                    }
                }
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
                }
                Ok(true)
            }

            // Tab: autocomplete command
            KeyCommand::Tab => {
                if self.palette_active && !self.palette_items.is_empty() {
                    let cmd_name = self.palette_items[self.palette_selected].name.clone();
                    self.input_text = format!("/{}", cmd_name);
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
                self.status = "New session started".to_string();
            }
            "tree" => {
                self.status = format!("Session tree: {} messages", self.messages.len());
            }
            "session" => {
                self.status = format!("Session: {} messages, {} queued", self.messages.len(), self.queued_messages.len());
            }
            "quit" => {
                // Set flag to exit on next iteration
                self.status = "Exiting...".to_string();
                // Force exit by triggering CtrlC twice logic
                self.ctrl_c_count = 2;
            }
            _ => {
                self.status = format!("Unknown command: /{}", name);
            }
        }
        Ok(())
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
        let input_text = self.input_text.clone();
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
            let editor_lines: Vec<&str> =
                if input_text.is_empty() { vec![""] } else { input_text.split('\n').collect() };
            let editor_height = (editor_lines.len() as u16).clamp(1, 6);
            let editor_top = hint_row.saturating_sub(editor_height);

            // -------- Overlay area above editor (command palette / model
            //          list / settings list). Takes up to ~60% of screen. --------
            let overlay_max_h = (size.height as f32 * 0.55) as u16;
            let overlay_active = palette_active
                || display_mode == DisplayMode::ModelList
                || display_mode == DisplayMode::SettingsList;
            let overlay_height: u16 = if overlay_active {
                match display_mode {
                    DisplayMode::ModelList => (model_list.len() as u16 + 3).min(overlay_max_h),
                    DisplayMode::SettingsList => (settings_list.len() as u16 + 3).min(overlay_max_h),
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
                } else {
                    render_settings_list(&settings_list, settings_selected)
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
            let editor_lines_rendered: Vec<Line> = editor_lines
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
                    Line::from(vec![prompt, Span::raw((*line).to_string())])
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
