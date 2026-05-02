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
    pub input_cost: f64,
    pub output_cost: f64,
}

fn build_model_rows(registry: &ModelRegistry) -> Vec<ModelRow> {
    let mut rows: Vec<ModelRow> = Vec::new();
    for provider in registry.providers() {
        let env_configured = provider.resolve_env_key().is_some();
        for model in &provider.models {
            let (input_cost, output_cost) = model
                .cost
                .as_ref()
                .map(|c| (c.input, c.output))
                .unwrap_or((0.0, 0.0));
            rows.push(ModelRow {
                provider_id: provider.id.clone(),
                provider_display: provider.display_name.clone(),
                model_id: model.id.clone(),
                model_name: model.name.clone(),
                context_window: model.context_window,
                reasoning: model.reasoning,
                env_configured,
                input_cost,
                output_cost,
            });
        }
    }
    // Stable sort: configured providers first, then alphabetical by
    // provider then model.
    rows.sort_by(|a, b| {
        b.env_configured
            .cmp(&a.env_configured)
            .then_with(|| a.provider_display.to_lowercase().cmp(&b.provider_display.to_lowercase()))
            .then_with(|| a.model_name.to_lowercase().cmp(&b.model_name.to_lowercase()))
    });
    rows
}

/// Subsequence match: does every character of `needle` appear in
/// `haystack` in order (case-insensitive)? Returns an optional score
/// (lower is better) for fuzzy ranking; `None` means no match.
fn fuzzy_score(needle: &str, haystack: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let needle = needle.to_lowercase();
    let haystack = haystack.to_lowercase();
    let mut needle_iter = needle.chars().peekable();
    let mut last_match: Option<usize> = None;
    let mut score: usize = 0;
    let mut first_match: Option<usize> = None;
    for (i, c) in haystack.char_indices() {
        if let Some(&needed) = needle_iter.peek() {
            if c == needed {
                if first_match.is_none() {
                    first_match = Some(i);
                }
                if let Some(last) = last_match {
                    // Penalize gaps between matched characters.
                    score += i - last - 1;
                }
                last_match = Some(i);
                needle_iter.next();
            }
        } else {
            break;
        }
    }
    if needle_iter.peek().is_some() {
        None
    } else {
        // Weight the prefix position less heavily than the middle-match
        // gaps so matches at the start of the string win ties.
        Some(score + first_match.unwrap_or(0) / 2)
    }
}

fn filter_model_rows(query: &str, all: &[ModelRow]) -> Vec<ModelRow> {
    if query.is_empty() {
        return all.to_vec();
    }
    let q = query.trim();
    let mut scored: Vec<(usize, ModelRow)> = Vec::new();
    for row in all {
        // Try each searchable field; keep the best (lowest) score.
        let candidates = [
            row.model_id.as_str(),
            row.model_name.as_str(),
            row.provider_id.as_str(),
            row.provider_display.as_str(),
        ];
        let best = candidates
            .iter()
            .filter_map(|field| fuzzy_score(q, field))
            .min();
        if let Some(s) = best {
            scored.push((s, row.clone()));
        }
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0));
    scored.into_iter().map(|(_, r)| r).collect()
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
    /// Raw JSON input (retained so the renderer can extract structured
    /// fields like `old_string`/`new_string` for edit diffs).
    pub input_raw: Option<serde_json::Value>,
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

/// Stats from the most recent completed turn. Shown in the row above
/// the input box so the user can see throughput at a glance without
/// having to parse the running session total.
#[derive(Clone, Debug)]
pub struct LastCallStats {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total: u64,
    pub duration_secs: f64,
}

impl LastCallStats {
    fn tps(&self) -> f64 {
        if self.duration_secs > 0.0 {
            self.output as f64 / self.duration_secs
        } else {
            0.0
        }
    }
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

    // Editor autocomplete (`@path` / `!bash`)
    autocomplete_active: bool,
    autocomplete_kind: Option<crate::autocomplete::TriggerKind>,
    autocomplete_fragment_start: usize,
    autocomplete_items: Vec<crate::autocomplete::Suggestion>,
    autocomplete_selected: usize,

    // Cached sessions directory for save/fork
    sessions_dir: PathBuf,

    // Escape/Ctrl+C tracking
    last_escape_time: Option<std::time::Instant>,
    ctrl_c_count: u32,
    /// Set to `true` by `/quit` (and any other shutdown path) so the
    /// main event loop can exit cleanly from outside the Ctrl+C
    /// handler. Checked after every key / tick.
    should_exit: bool,

    // Active prompt request. When this is `Some`, `agent` is `None` because
    // the agent has been moved into the background task; it returns via the
    // join handle along with the result.
    active_job: Option<PromptJob>,

    /// Cumulative usage across the current session. Updated on every
    /// `AgentEvent::Usage` so the footer's `0.0%/?` can show the real
    /// percentage of context used.
    usage: UsageStats,
    /// Stats from the most recent completed turn (output / input /
    /// cache rw / total / duration). Rendered in the row above the
    /// input box.
    last_call: Option<LastCallStats>,
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
    /// Usage snapshot taken at turn-start so the `LastCallStats` on
    /// completion reflects the delta for this turn (not the whole
    /// session).
    pre_turn_usage: UsageStats,
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
            autocomplete_active: false,
            autocomplete_kind: None,
            autocomplete_fragment_start: 0,
            autocomplete_items: Vec::new(),
            autocomplete_selected: 0,
            sessions_dir: default_sessions_dir(),
            last_escape_time: None,
            ctrl_c_count: 0,
            should_exit: false,
            active_job: None,
            usage: UsageStats::default(),
            last_call: None,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        loop {
            // Drain keyboard/terminal events first so the UI stays responsive.
            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(50)) {
                use pi_tui::tui::AppEvent;
                let cont = match event {
                    AppEvent::Key(cmd) => self.handle_key(cmd).await?,
                    AppEvent::Paste(text) => {
                        // Strip a single trailing newline (common when
                        // pasting a line from another terminal) so the
                        // paste does not accidentally submit the turn.
                        let trimmed = text.trim_end_matches('\n').to_string();
                        self.editor.insert_str(&trimmed);
                        self.refresh_autocomplete();
                        true
                    }
                    _ => true,
                };
                if !cont {
                    break;
                }
                // `/quit` (and any other slash command) sets
                // `should_exit`. Check here so the current frame still
                // draws the "Exiting..." status before we tear down.
                if self.should_exit {
                    let _ = self.draw();
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
                        let elapsed = job.started_at.elapsed().as_secs_f64();
                        // Snapshot the delta against the pre-turn usage
                        // so the last-call row reflects just this turn.
                        self.last_call = Some(LastCallStats {
                            input: self
                                .usage
                                .input_tokens
                                .saturating_sub(job.pre_turn_usage.input_tokens),
                            output: self
                                .usage
                                .output_tokens
                                .saturating_sub(job.pre_turn_usage.output_tokens),
                            cache_read: self
                                .usage
                                .cache_read_tokens
                                .saturating_sub(job.pre_turn_usage.cache_read_tokens),
                            cache_write: self
                                .usage
                                .cache_write_tokens
                                .saturating_sub(job.pre_turn_usage.cache_write_tokens),
                            total: self
                                .usage
                                .total_tokens
                                .saturating_sub(job.pre_turn_usage.total_tokens),
                            duration_secs: elapsed,
                        });
                        self.status = format!("Ready ({:.1}s)", elapsed);
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
                        input_raw: Some(input),
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
                // Drop the streaming placeholder entirely and push a
                // dedicated error row so the red-tinted visual kicks in.
                if let Some(msg) = self.messages.get_mut(assistant_index) {
                    msg.streaming = false;
                    if msg.content.is_empty() && msg.tool_calls.is_empty() {
                        // Swap the blank placeholder in-place so the
                        // message index tracked by the active job stays
                        // valid for any late-arriving events.
                        msg.role = "error".to_string();
                        msg.content = message.clone();
                    } else {
                        // There was partial output; keep it and append a
                        // separate error row.
                        self.messages.push(ConversationMessage {
                            role: "error".to_string(),
                            content: message.clone(),
                            ..Default::default()
                        });
                    }
                } else {
                    self.messages.push(ConversationMessage {
                        role: "error".to_string(),
                        content: message.clone(),
                        ..Default::default()
                    });
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
            pre_turn_usage: self.usage.clone(),
        });
        self.executing = true;
        self.status = "Executing...".to_string();
    }

    async fn handle_key(&mut self, cmd: KeyCommand) -> Result<bool> {
        let result = self.handle_key_inner(cmd).await?;
        // Any key that could mutate the editor text might invalidate
        // the autocomplete dropdown. Recompute after every key so the
        // suggestions always track the active fragment.
        self.refresh_autocomplete();
        Ok(result)
    }

    async fn handle_key_inner(&mut self, cmd: KeyCommand) -> Result<bool> {
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
                        // Map the chosen level to a budget preset so
                        // Anthropic extended-thinking picks a sensible
                        // token count automatically.
                        let budgets = budgets_for_level(opt);
                        if let Some(agent) = self.agent.as_mut() {
                            agent.set_thinking_level(level);
                            agent.set_thinking_budgets(budgets);
                        }
                        self.status = format!(
                            "Thinking level → {opt}{}",
                            if budget_hint_for_level(opt).is_empty() {
                                String::new()
                            } else {
                                format!(" ({})", budget_hint_for_level(opt))
                            },
                        );
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

            // Shift+Enter (and Ctrl+J as a universal fallback): insert
            // a literal newline into the editor buffer instead of
            // submitting. Gives users a way to compose multi-line
            // prompts even on terminals that don't forward Shift+Enter.
            KeyCommand::ShiftEnter | KeyCommand::CtrlJ => {
                self.editor.newline();
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

            // Ctrl+C: copy selection, clear, or quit
            KeyCommand::CtrlC => {
                // If there's an active selection, Ctrl+C copies it to
                // the system clipboard and does NOT count toward the
                // double-press quit counter. This matches TS / VSCode /
                // most terminals where Ctrl+C-on-selection is a copy.
                if self.editor.has_selection() {
                    if let Some(text) = self.editor.copy_selection() {
                        match arboard::Clipboard::new()
                            .and_then(|mut c| c.set_text(text.clone()))
                        {
                            Ok(()) => {
                                self.status =
                                    format!("Copied {} chars to clipboard", text.len());
                            }
                            Err(e) => {
                                self.status = format!("Copy failed: {e}");
                            }
                        }
                    }
                    self.editor.clear_selection();
                    return Ok(true);
                }
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

            // Ctrl+X: cut selection to kill ring + system clipboard.
            KeyCommand::CtrlX => {
                if let Some(text) = self.editor.copy_selection() {
                    let _ = arboard::Clipboard::new().and_then(|mut c| c.set_text(text.clone()));
                    self.editor.cut_selection();
                    self.status = format!("Cut {} chars", text.len());
                }
                Ok(true)
            }

            // Shift+Arrow / Shift+Home / Shift+End: extend the
            // selection. The editor tracks an anchor the first time
            // one of these fires and extends as more arrive.
            KeyCommand::ShiftArrowLeft => {
                self.editor.select_left();
                Ok(true)
            }
            KeyCommand::ShiftArrowRight => {
                self.editor.select_right();
                Ok(true)
            }
            KeyCommand::ShiftArrowUp => {
                self.editor.select_up();
                Ok(true)
            }
            KeyCommand::ShiftArrowDown => {
                self.editor.select_down();
                Ok(true)
            }
            KeyCommand::ShiftHome => {
                self.editor.select_line_start();
                Ok(true)
            }
            KeyCommand::ShiftEnd => {
                self.editor.select_line_end();
                Ok(true)
            }

            // Escape: cancel or go back
            KeyCommand::Escape => {
                self.ctrl_c_count = 0;
                if self.autocomplete_active {
                    self.autocomplete_active = false;
                    self.autocomplete_items.clear();
                    return Ok(true);
                }
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
                if self.autocomplete_active {
                    self.autocomplete_selected = self.autocomplete_selected.saturating_sub(1);
                } else if self.palette_active && !self.palette_items.is_empty() {
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
                if self.autocomplete_active {
                    if self.autocomplete_selected < self.autocomplete_items.len() - 1 {
                        self.autocomplete_selected += 1;
                    }
                } else if self.palette_active && !self.palette_items.is_empty() {
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
                // Editor autocomplete (file / bash) wins over the palette
                // when a fragment is active.
                if self.autocomplete_active {
                    self.accept_autocomplete();
                    return Ok(true);
                }
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

    /// Rescan autocomplete suggestions based on the current editor
    /// buffer. Called after every insertion / backspace / cursor
    /// move that could change the active fragment.
    fn refresh_autocomplete(&mut self) {
        // Don't autocomplete while we're inside a modal overlay — the
        // palette already owns the dropdown real-estate.
        if self.palette_active || self.display_mode != DisplayMode::Chat {
            self.autocomplete_active = false;
            return;
        }
        let text = self.editor.text();
        let cursor = self.editor.cursor_byte();
        let Some((kind, start, fragment)) =
            crate::autocomplete::detect_trigger(text, cursor)
        else {
            self.autocomplete_active = false;
            self.autocomplete_items.clear();
            return;
        };
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let items = match kind {
            crate::autocomplete::TriggerKind::File => {
                crate::autocomplete::scan_file_completions(fragment, &cwd, 12)
            }
            crate::autocomplete::TriggerKind::Bash => {
                crate::autocomplete::scan_bash_completions(fragment, 12)
            }
            crate::autocomplete::TriggerKind::Session => {
                crate::autocomplete::scan_session_completions(
                    fragment,
                    &self.sessions_dir,
                    12,
                )
            }
        };
        if items.is_empty() {
            self.autocomplete_active = false;
            self.autocomplete_items.clear();
            return;
        }
        self.autocomplete_active = true;
        self.autocomplete_kind = Some(kind);
        self.autocomplete_fragment_start = start;
        self.autocomplete_items = items;
        self.autocomplete_selected = 0;
    }

    /// Replace the current `@frag`/`!frag` in the editor with the
    /// selected suggestion and close the dropdown.
    fn accept_autocomplete(&mut self) {
        if !self.autocomplete_active {
            return;
        }
        let Some(sug) = self.autocomplete_items.get(self.autocomplete_selected).cloned()
        else {
            self.autocomplete_active = false;
            return;
        };
        let start = self.autocomplete_fragment_start;
        let cursor = self.editor.cursor_byte();
        // Build the new text: text[..start] + sug.insert + text[cursor..].
        let text = self.editor.text();
        if start > cursor || cursor > text.len() {
            self.autocomplete_active = false;
            return;
        }
        let mut new_text = String::with_capacity(text.len() + sug.insert.len());
        new_text.push_str(&text[..start]);
        new_text.push_str(&sug.insert);
        new_text.push_str(&text[cursor..]);
        self.editor.set_text(new_text);
        self.autocomplete_active = false;
        self.autocomplete_items.clear();
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
                self.last_call = None;
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
                self.status = "Pick thinking level + budget. Enter to apply, Esc to cancel.".to_string();
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
                match share_via_gh_gist(&self.messages) {
                    Ok(url) => {
                        // Try to drop the URL onto the clipboard too so
                        // the user can paste it directly.
                        let _ = arboard::Clipboard::new()
                            .and_then(|mut c| c.set_text(url.clone()));
                        self.status = format!("Shared → {url} (copied)");
                    }
                    Err(e) => {
                        self.status = format!(
                            "Share failed: {e}. Use /export to produce an HTML file instead."
                        );
                    }
                }
            }
            "quit" => {
                // Signal the main event loop to break cleanly. Unlike
                // Ctrl+C (which is handled inside `KeyCommand::CtrlC`
                // and can short-circuit the loop by returning Ok(false)),
                // a slash command runs inside `execute_command` and
                // has no return channel, so we need a shared flag.
                self.status = "Exiting...".to_string();
                self.should_exit = true;
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
        let editor_text = self.editor.text().to_string();
        let selection_byte_range = self.editor.selection_range();
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
        let autocomplete_active = self.autocomplete_active;
        let autocomplete_items = self.autocomplete_items.clone();
        let autocomplete_selected = self.autocomplete_selected;
        let autocomplete_kind = self.autocomplete_kind;
        let last_call = self.last_call.clone();

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

            // -------- Bottom 3 rows: footer --------
            let footer_stats_row = size.height - 1;  // tokens + model
            let footer_pwd_row = size.height - 2;    // pwd (branch)
            let hint_row = size.height - 3;          // key hints + status

            // -------- Input textbox (bracketed by horizontal rules) --------
            // Max 7 *visual* rows; beyond that the viewport scrolls so
            // the cursor stays in view. We hard-wrap long logical lines
            // against the available terminal width so text never runs
            // past the right edge, and the cursor position tracks into
            // the wrapped row.
            //
            // `content_width` is the column budget for text; 2 chars are
            // reserved on the left for the `▌ ` prompt / 2-space
            // continuation padding.
            let content_width = (size.width as usize).saturating_sub(2).max(1);
            let visuals = wrap_visual_lines(&input_lines, content_width);
            let total_visual_rows = visuals.len().max(1) as u16;
            let editor_height: u16 = total_visual_rows.clamp(1, 7);
            let lower_rule_row = hint_row.saturating_sub(1);
            let editor_top = lower_rule_row.saturating_sub(editor_height);
            let upper_rule_row = editor_top.saturating_sub(1);
            let last_call_row = upper_rule_row.saturating_sub(1);

            // Cursor position mapped into visual coordinates.
            let (cursor_vrow, cursor_vcol) =
                logical_to_visual(&visuals, cursor_row, cursor_col);

            // Selection span mapped into visual coordinates (if any).
            let selection_visual: Option<((usize, usize), (usize, usize))> =
                selection_byte_range.map(|(start, end)| {
                    let (sr, sc) = byte_to_row_col(&editor_text, start);
                    let (er, ec) = byte_to_row_col(&editor_text, end);
                    let s_vis = logical_to_visual(&visuals, sr, sc);
                    let e_vis = logical_to_visual(&visuals, er, ec);
                    (s_vis, e_vis)
                });

            // Bottom-anchor the viewport and shift up if the cursor would
            // fall outside it.
            let (first_visible_vrow, visible_slice): (usize, Vec<VisualLine>) =
                if visuals.len() <= editor_height as usize {
                    (0, visuals.clone())
                } else {
                    let max_first = visuals.len().saturating_sub(editor_height as usize);
                    let mut first = max_first;
                    if cursor_vrow < first {
                        first = cursor_vrow;
                    } else if cursor_vrow >= first + editor_height as usize {
                        first = cursor_vrow + 1 - editor_height as usize;
                    }
                    let slice = visuals[first..first + editor_height as usize].to_vec();
                    (first, slice)
                };
            let local_cursor_vrow = cursor_vrow.saturating_sub(first_visible_vrow);

            // -------- Overlay area (command palette / model list / ...) --------
            let overlay_max_h = (size.height as f32 * 0.55) as u16;
            let overlay_active = palette_active
                || autocomplete_active
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
                    _ if autocomplete_active => (autocomplete_items.len() as u16 + 2).min(overlay_max_h),
                    _ => 0,
                }
            } else {
                0
            };
            let overlay_top = last_call_row.saturating_sub(overlay_height);

            // -------- Messages area: top down to overlay / call-info row --------
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
                } else if autocomplete_active {
                    render_autocomplete_dropdown(
                        autocomplete_kind,
                        &autocomplete_items,
                        autocomplete_selected,
                    )
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
            let editor_lines_rendered: Vec<Line> = visible_slice
                .iter()
                .enumerate()
                .map(|(local_vi, vline)| {
                    // `local_vi` is the viewport row; `vrow` is the
                    // global visual row index (across the wrapped
                    // document). Cursor / selection highlighting works
                    // in visual coordinates.
                    let vrow = local_vi + first_visible_vrow;
                    // First visual row of the logical line 0 gets the
                    // ▌ prompt; everything else (including wrapped
                    // continuation rows) gets 2-space padding.
                    let prompt = if vline.logical_row == 0 && vline.logical_col_start == 0 {
                        if executing {
                            Span::styled("▌ ", Style::default().fg(Color::Yellow))
                        } else {
                            Span::styled("▌ ", Style::default().fg(Color::Cyan))
                        }
                    } else {
                        Span::styled("  ", Style::default())
                    };
                    let line_char_count = vline.text.chars().count();
                    // Selection span (visual-col start..end on this row).
                    let row_selection: Option<(usize, usize)> =
                        selection_visual.and_then(|((sr, sc), (er, ec))| {
                            if vrow < sr || vrow > er {
                                return None;
                            }
                            let start_col = if vrow == sr { sc } else { 0 };
                            let end_col = if vrow == er { ec } else { line_char_count };
                            if end_col <= start_col {
                                None
                            } else {
                                Some((start_col, end_col))
                            }
                        });
                    let cursor_here = vrow == cursor_vrow && !overlay_active;
                    let mut spans: Vec<Span<'static>> = vec![prompt];
                    let mut col_idx = 0usize;
                    let mut buf = String::new();
                    let mut buf_style = Style::default();
                    let selection_style =
                        Style::default().bg(Color::Indexed(60)).fg(Color::White);
                    let cursor_style =
                        Style::default().bg(Color::Gray).fg(Color::Black);
                    let flush = |spans: &mut Vec<Span<'static>>, buf: &mut String, style: Style| {
                        if !buf.is_empty() {
                            spans.push(Span::styled(std::mem::take(buf), style));
                        }
                    };
                    for ch in vline.text.chars() {
                        let in_sel = row_selection
                            .map(|(s, e)| col_idx >= s && col_idx < e)
                            .unwrap_or(false);
                        let at_cursor = cursor_here && col_idx == cursor_vcol;
                        let desired = if at_cursor {
                            cursor_style
                        } else if in_sel {
                            selection_style
                        } else {
                            Style::default()
                        };
                        if desired != buf_style {
                            flush(&mut spans, &mut buf, buf_style);
                            buf_style = desired;
                        }
                        buf.push(ch);
                        col_idx += 1;
                    }
                    flush(&mut spans, &mut buf, buf_style);
                    // Trailing cursor cell when the cursor sits past the
                    // last char on this visual row.
                    if cursor_here && col_idx == cursor_vcol {
                        spans.push(Span::styled(" ".to_string(), cursor_style));
                    }
                    let _ = local_vi;
                    Line::from(spans)
                })
                .collect();
            // We've already hard-wrapped the content to `content_width`,
            // so tell ratatui *not* to soft-wrap again (which would add a
            // second wrap at the Paragraph's own width and break the
            // cursor math). Each rendered line is pre-sized.
            let editor = Paragraph::new(editor_lines_rendered);
            frame.render_widget(editor, editor_area);

            // ===== Last-call stats row (dim grey, above upper boundary) =====
            // Rendered only when we have a completed turn to summarize;
            // otherwise the row stays blank so the boundary-line + editor
            // block sits closer to the transcript.
            if !overlay_active {
                let last_call_area = Rect {
                    x: 0,
                    y: last_call_row,
                    width: size.width,
                    height: 1,
                };
                let last_call_line = last_call.as_ref().map(render_last_call_line).unwrap_or_else(|| Line::from(""));
                frame.render_widget(Paragraph::new(last_call_line), last_call_area);
            }

            // ===== Upper + lower boundary rules =====
            if !overlay_active {
                let rule = "─".repeat(size.width as usize);
                let upper_area = Rect {
                    x: 0,
                    y: upper_rule_row,
                    width: size.width,
                    height: 1,
                };
                let lower_area = Rect {
                    x: 0,
                    y: lower_rule_row,
                    width: size.width,
                    height: 1,
                };
                let rule_style = Style::default().fg(Color::DarkGray);
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(rule.clone(), rule_style))),
                    upper_area,
                );
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(rule, rule_style))),
                    lower_area,
                );
            }

            // ===== Hint row (just below lower boundary) =====
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
        if msg.role == "error" {
            // Error rows get a dedicated red-tinted style so provider
            // / tool failures are visually distinct from normal
            // assistant output.
            let bg = Style::default().bg(Color::Indexed(52));
            let prefix_style = bg
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD);
            for (i, text_line) in msg.content.lines().enumerate() {
                if i == 0 {
                    lines.push(Line::from(vec![
                        Span::styled(" error ".to_string(), prefix_style),
                        Span::styled(" ".to_string(), bg),
                        Span::styled(text_line.to_string(), bg.fg(Color::Red)),
                    ]));
                } else {
                    lines.push(Line::from(vec![
                        Span::styled("       ".to_string(), bg),
                        Span::styled(text_line.to_string(), bg.fg(Color::Red)),
                    ]));
                }
            }
            lines.push(Line::from(""));
            continue;
        }

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
            // Tool-specific structured preview (edit -> mini-diff, write
            // -> content preview). Shown regardless of Ctrl+O because
            // it's the main signal that the tool call will do something
            // destructive.
            if let Some(raw) = call.input_raw.as_ref() {
                for extra in tool_structured_preview(&call.name, raw, bg) {
                    lines.push(extra);
                }
            }
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
        Span::styled("ctrl+j".to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(" newline "),
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
        Span::raw(" model"),
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

fn render_autocomplete_dropdown(
    kind: Option<crate::autocomplete::TriggerKind>,
    items: &[crate::autocomplete::Suggestion],
    selected: usize,
) -> Vec<Line<'static>> {
    let title = match kind {
        Some(crate::autocomplete::TriggerKind::File) => format!("Files ({})", items.len()),
        Some(crate::autocomplete::TriggerKind::Bash) => format!("Commands ({})", items.len()),
        Some(crate::autocomplete::TriggerKind::Session) => {
            format!("Sessions ({})", items.len())
        }
        None => format!("Suggestions ({})", items.len()),
    };
    let mut lines = vec![Line::from(Span::styled(
        title,
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    ))];
    for (i, s) in items.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "› " } else { "  " };
        let style = if is_sel {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(s.label.clone(), style),
        ]));
    }
    lines
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
            if filter.is_empty() {
                String::new()
            } else {
                format!("filter: {filter}")
            },
            muted_style(),
        ),
    ])];
    // Header row so the columns line up visually.
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{:<36}", "name"),
            dim_style().add_modifier(Modifier::UNDERLINED),
        ),
        Span::raw(" "),
        Span::styled(format!("{:>6}", "ctx"), dim_style().add_modifier(Modifier::UNDERLINED)),
        Span::raw("  "),
        Span::styled(
            format!("{:>16}", "$/M in/out"),
            dim_style().add_modifier(Modifier::UNDERLINED),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{:<3}", "rsn"),
            dim_style().add_modifier(Modifier::UNDERLINED),
        ),
        Span::raw("  "),
        Span::styled("id", dim_style().add_modifier(Modifier::UNDERLINED)),
    ]));
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
        let cost_cell = if m.input_cost == 0.0 && m.output_cost == 0.0 {
            "-".to_string()
        } else {
            format!("{:.2}/{:.2}", m.input_cost, m.output_cost)
        };
        let reasoning_cell = if m.reasoning { "yes" } else { "-" };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(format!("{:<36}", truncate_for_col(&m.model_name, 36)), name_style),
            Span::raw(" "),
            Span::styled(format!("{ctx_w:>6}"), muted_style()),
            Span::raw("  "),
            Span::styled(format!("{cost_cell:>16}"), muted_style()),
            Span::raw("  "),
            Span::styled(format!("{reasoning_cell:<3}"), dim_style()),
            Span::raw("  "),
            Span::styled(m.model_id.clone(), dim_style()),
        ]));
    }
    lines
}

fn truncate_for_col(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
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
        let hint = budget_hint_for_level(opt);
        let hint_part = if hint.is_empty() {
            String::new()
        } else {
            format!("  {hint}")
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(opt.to_string(), style),
            Span::styled(hint_part, dim_style()),
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

/// Preset token budgets per thinking level. Applied whenever the user
/// picks a level from the `/thinking` overlay so Anthropic-style
/// extended-thinking providers get a sensible budget without a separate
/// prompt. Providers that ignore `thinking_budgets` simply drop it.
fn budgets_for_level(label: &str) -> Option<pi_ai::types::ThinkingBudgets> {
    match label {
        "off" => None,
        "low" => Some(pi_ai::types::ThinkingBudgets {
            minimal: Some(512),
            low: Some(1024),
            medium: Some(4096),
            high: Some(16384),
        }),
        "medium" => Some(pi_ai::types::ThinkingBudgets {
            minimal: Some(1024),
            low: Some(2048),
            medium: Some(8192),
            high: Some(24576),
        }),
        "high" => Some(pi_ai::types::ThinkingBudgets {
            minimal: Some(2048),
            low: Some(4096),
            medium: Some(16384),
            high: Some(32768),
        }),
        _ => None,
    }
}

/// Short budget description surfaced next to each level option in the
/// `/thinking` overlay.
fn budget_hint_for_level(label: &str) -> &'static str {
    match label {
        "off" => "",
        "low" => "~1k tokens",
        "medium" => "~8k tokens",
        "high" => "~16k tokens",
        _ => "",
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

/// Build a single markdown transcript for sharing and hand it to
/// `gh gist create` as stdin. Returns the gist URL that `gh` prints
/// to stdout. Requires the `gh` CLI on PATH and the user logged in;
/// surfaces a clear error if either is missing.
fn share_via_gh_gist(messages: &[ConversationMessage]) -> Result<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let markdown = conversation_to_markdown(messages);
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();

    let mut child = Command::new("gh")
        .args([
            "gist",
            "create",
            "--public",
            "--filename",
            "pi-session.md",
            "--desc",
            &format!("pi session {ts}"),
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("gh not found on PATH: {e}"))?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(markdown.as_bytes())
            .map_err(|e| anyhow::anyhow!("write stdin: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| anyhow::anyhow!("wait: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(anyhow::anyhow!(if err.is_empty() {
            "gh gist create exited non-zero".to_string()
        } else {
            err
        }));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines().rev() {
        let t = line.trim();
        if t.starts_with("http://") || t.starts_with("https://") {
            return Ok(t.to_string());
        }
    }
    Err(anyhow::anyhow!(
        "gh gist returned 0 but no URL found in output"
    ))
}

fn conversation_to_markdown(messages: &[ConversationMessage]) -> String {
    let mut out = String::new();
    out.push_str("# pi session\n\n");
    out.push_str(&format!(
        "_{} \u{2022} {} message(s)_\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        messages.len(),
    ));
    for m in messages {
        let header = match m.role.as_str() {
            "user" => "## You".to_string(),
            "assistant" => "## Assistant".to_string(),
            "system" => "## System".to_string(),
            "error" => "## Error".to_string(),
            other => format!("## {other}"),
        };
        out.push_str(&header);
        out.push_str("\n\n");
        if !m.thinking.is_empty() {
            out.push_str("<details><summary>thinking</summary>\n\n```\n");
            out.push_str(&m.thinking);
            out.push_str("\n```\n\n</details>\n\n");
        }
        for tc in &m.tool_calls {
            out.push_str(&format!(
                "**tool → {}** `{}`\n\n",
                tc.name, tc.input_preview
            ));
            if let Some(o) = &tc.output {
                out.push_str("```\n");
                out.push_str(o);
                if !o.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("```\n\n");
            }
        }
        if !m.content.is_empty() {
            out.push_str(&m.content);
            out.push_str("\n\n");
        }
    }
    out
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
/// Render small, tool-specific preview lines below a tool-call
/// header. Used for `edit` (mini unified-diff) and `write` (first few
/// lines of content), returning an empty list for tools that don't
/// benefit from a structured preview.
fn tool_structured_preview(
    name: &str,
    input: &serde_json::Value,
    bg: Style,
) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    match name {
        "edit" => {
            // Try the batched form first (`edits: [{oldText, newText}]`)
            // and fall back to the flat form (`old_string`/`new_string`)
            // used by some provider-side tool schemas.
            let edits_list = input.get("edits").and_then(|v| v.as_array());
            let (old_text, new_text): (String, String) = if let Some(arr) = edits_list {
                let mut old_buf = String::new();
                let mut new_buf = String::new();
                for (i, e) in arr.iter().enumerate() {
                    if i > 0 {
                        old_buf.push_str("\n---\n");
                        new_buf.push_str("\n---\n");
                    }
                    if let Some(o) = e.get("oldText").and_then(|v| v.as_str()) {
                        old_buf.push_str(o);
                    }
                    if let Some(n) = e.get("newText").and_then(|v| v.as_str()) {
                        new_buf.push_str(n);
                    }
                }
                (old_buf, new_buf)
            } else {
                (
                    input.get("old_string").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    input.get("new_string").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                )
            };
            if old_text.is_empty() && new_text.is_empty() {
                return out;
            }
            let old_lines: Vec<&str> = old_text.split('\n').collect();
            let new_lines: Vec<&str> = new_text.split('\n').collect();
            let max_each = 5usize;
            for (i, line) in old_lines.iter().take(max_each).enumerate() {
                let marker = if i == 0 { "  − " } else { "    " };
                out.push(Line::from(vec![
                    Span::styled(marker.to_string(), bg.fg(Color::Red)),
                    Span::styled((*line).to_string(), bg.fg(Color::Red)),
                ]));
            }
            if old_lines.len() > max_each {
                out.push(Line::from(Span::styled(
                    format!("    … {} more removed line(s)", old_lines.len() - max_each),
                    bg.fg(Color::DarkGray),
                )));
            }
            for (i, line) in new_lines.iter().take(max_each).enumerate() {
                let marker = if i == 0 { "  + " } else { "    " };
                out.push(Line::from(vec![
                    Span::styled(marker.to_string(), bg.fg(Color::Green)),
                    Span::styled((*line).to_string(), bg.fg(Color::Green)),
                ]));
            }
            if new_lines.len() > max_each {
                out.push(Line::from(Span::styled(
                    format!("    … {} more added line(s)", new_lines.len() - max_each),
                    bg.fg(Color::DarkGray),
                )));
            }
        }
        "write" => {
            let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if content.is_empty() {
                return out;
            }
            let max = 6usize;
            for line in content.split('\n').take(max) {
                out.push(Line::from(vec![
                    Span::styled("  + ".to_string(), bg.fg(Color::Green)),
                    Span::styled(line.to_string(), bg.fg(Color::Green)),
                ]));
            }
            let total = content.split('\n').count();
            if total > max {
                out.push(Line::from(Span::styled(
                    format!("    … {} more line(s)", total - max),
                    bg.fg(Color::DarkGray),
                )));
            }
        }
        _ => {}
    }
    out
}

/// Format a non-negative integer with Indian-style digit grouping:
/// last 3 digits grouped, then pairs. E.g. `1234` -> `1,234`;
/// `12162440` -> `1,21,62,440`.
fn format_int_indian(n: u64) -> String {
    let s = n.to_string();
    if s.len() <= 3 {
        return s;
    }
    let (head, tail) = s.split_at(s.len() - 3);
    let mut groups: Vec<&str> = Vec::new();
    let mut i = head.len();
    while i > 0 {
        if i >= 2 {
            groups.push(&head[i - 2..i]);
            i -= 2;
        } else {
            groups.push(&head[..i]);
            i = 0;
        }
    }
    groups.reverse();
    format!("{},{}", groups.join(","), tail)
}

/// Render the last-call stats row shown above the input textbox.
/// All parts are dim grey so the row stays low-contrast; separators
/// are middle-dots.
fn render_last_call_line(stats: &LastCallStats) -> Line<'static> {
    let sep = Span::styled(" \u{00b7} ".to_string(), Style::default().fg(Color::DarkGray));
    let st = Style::default().fg(Color::DarkGray);
    let mut spans: Vec<Span<'static>> = Vec::new();
    spans.push(Span::styled(
        format!("TPS {:.1} tok/s", stats.tps()),
        st,
    ));
    spans.push(sep.clone());
    spans.push(Span::styled(
        format!("out {}", format_int_indian(stats.output)),
        st,
    ));
    spans.push(sep.clone());
    spans.push(Span::styled(
        format!("in {}", format_int_indian(stats.input)),
        st,
    ));
    if stats.cache_read > 0 || stats.cache_write > 0 {
        spans.push(sep.clone());
        spans.push(Span::styled(
            format!(
                "cache r/w {}/{}",
                format_int_indian(stats.cache_read),
                format_int_indian(stats.cache_write),
            ),
            st,
        ));
    }
    if stats.total > 0 {
        spans.push(sep.clone());
        spans.push(Span::styled(
            format!("total {}", format_int_indian(stats.total)),
            st,
        ));
    }
    spans.push(sep);
    spans.push(Span::styled(format!("{:.1}s", stats.duration_secs), st));
    Line::from(spans)
}

/// A hard-wrapped view of one piece of a logical editor line. Used by
/// the draw routine to show long inputs without running them past the
/// right edge of the textbox.
#[derive(Clone, Debug)]
struct VisualLine {
    /// Logical line index (into `input_lines`).
    logical_row: usize,
    /// Char column in the logical line where this chunk starts.
    logical_col_start: usize,
    /// The chunk text (<= `content_width` chars).
    text: String,
}

/// Hard-wrap each logical line into at most `content_width`-char
/// chunks. Preserves empty logical lines (so a trailing `\n` still
/// reserves a blank visual row). When `content_width` is 0 the input
/// is returned as-is to avoid a divide-by-zero loop.
fn wrap_visual_lines(lines: &[String], content_width: usize) -> Vec<VisualLine> {
    let mut out: Vec<VisualLine> = Vec::new();
    if content_width == 0 {
        for (row, line) in lines.iter().enumerate() {
            out.push(VisualLine {
                logical_row: row,
                logical_col_start: 0,
                text: line.clone(),
            });
        }
        return out;
    }
    for (row, line) in lines.iter().enumerate() {
        if line.is_empty() {
            out.push(VisualLine {
                logical_row: row,
                logical_col_start: 0,
                text: String::new(),
            });
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + content_width).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            out.push(VisualLine {
                logical_row: row,
                logical_col_start: start,
                text: chunk,
            });
            start = end;
        }
    }
    out
}

/// Map a (logical_row, char-col) pair to a (visual_row, visual_col)
/// pair given the wrapped view. If the cursor sits exactly at the
/// wrap boundary (col == content_width of its row), it's reported at
/// the *start* of the next visual row, matching how the text rolls
/// over when the user types past the boundary.
fn logical_to_visual(
    visuals: &[VisualLine],
    logical_row: usize,
    logical_col: usize,
) -> (usize, usize) {
    let mut last_for_row: Option<(usize, &VisualLine)> = None;
    for (vi, vl) in visuals.iter().enumerate() {
        if vl.logical_row != logical_row {
            continue;
        }
        let chunk_len = vl.text.chars().count();
        let chunk_start = vl.logical_col_start;
        let chunk_end_exclusive = chunk_start + chunk_len;
        if logical_col >= chunk_start && logical_col < chunk_end_exclusive {
            return (vi, logical_col - chunk_start);
        }
        last_for_row = Some((vi, vl));
    }
    if let Some((vi, vl)) = last_for_row {
        let chunk_len = vl.text.chars().count();
        let col = logical_col.saturating_sub(vl.logical_col_start).min(chunk_len);
        return (vi, col);
    }
    (0, 0)
}

/// Convert a byte offset into a `(row, char-column)` pair using the
/// same row/col semantics as `InputEditor::cursor_line_col`. Used by
/// the selection renderer to paint the highlighted range row by row.
fn byte_to_row_col(text: &str, byte: usize) -> (usize, usize) {
    let clamp = byte.min(text.len());
    let prefix = &text[..clamp];
    let row = prefix.bytes().filter(|b| *b == b'\n').count();
    let line_start = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let col = text[line_start..clamp].chars().count();
    (row, col)
}

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

    #[test]
    fn fuzzy_score_prefers_contiguous_matches() {
        let a = fuzzy_score("cld", "claude").unwrap();
        let b = fuzzy_score("cld", "cerebras-llama-data").unwrap();
        assert!(a < b, "claude {a} should score better than {b}");
    }

    #[test]
    fn fuzzy_score_returns_none_for_missing_chars() {
        assert!(fuzzy_score("xyz", "claude-haiku").is_none());
    }

    #[test]
    fn filter_model_rows_orders_by_match_quality() {
        let rows = vec![
            ModelRow {
                provider_id: "anthropic".into(),
                provider_display: "Anthropic".into(),
                model_id: "claude-opus-4".into(),
                model_name: "Claude Opus 4".into(),
                context_window: 200_000,
                reasoning: true,
                env_configured: true,
                input_cost: 15.0,
                output_cost: 75.0,
            },
            ModelRow {
                provider_id: "google".into(),
                provider_display: "Google".into(),
                model_id: "gemini-pro-1.5".into(),
                model_name: "Gemini Pro 1.5".into(),
                context_window: 2_000_000,
                reasoning: false,
                env_configured: false,
                input_cost: 0.0,
                output_cost: 0.0,
            },
        ];
        let filtered = filter_model_rows("claude", &rows);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_id, "claude-opus-4");
    }

    #[test]
    fn indian_digit_grouping_matches_user_preference() {
        assert_eq!(format_int_indian(0), "0");
        assert_eq!(format_int_indian(999), "999");
        assert_eq!(format_int_indian(1_000), "1,000");
        assert_eq!(format_int_indian(12_345), "12,345");
        assert_eq!(format_int_indian(1_00_000), "1,00,000");
        assert_eq!(format_int_indian(12_162_440), "1,21,62,440");
        assert_eq!(format_int_indian(1_25_11_376), "1,25,11,376");
    }

    #[test]
    fn wrap_visual_lines_splits_at_content_width() {
        let lines = vec!["hello world foo bar baz".to_string()];
        let out = wrap_visual_lines(&lines, 10);
        let texts: Vec<&str> = out.iter().map(|v| v.text.as_str()).collect();
        assert_eq!(texts, vec!["hello worl", "d foo bar ", "baz"]);
        assert!(out.iter().all(|v| v.logical_row == 0));
        assert_eq!(out[0].logical_col_start, 0);
        assert_eq!(out[1].logical_col_start, 10);
        assert_eq!(out[2].logical_col_start, 20);
    }

    #[test]
    fn wrap_preserves_empty_lines_and_row_indices() {
        let lines = vec![
            "a".to_string(),
            String::new(),
            "bbbbbbbbb".to_string(),
        ];
        let out = wrap_visual_lines(&lines, 4);
        let rows: Vec<usize> = out.iter().map(|v| v.logical_row).collect();
        // `a` -> 1 row, empty -> 1 row, `bbbbbbbbb` -> 3 rows (4 + 4 + 1).
        assert_eq!(rows, vec![0, 1, 2, 2, 2]);
    }

    #[test]
    fn logical_to_visual_wraps_across_chunks() {
        let lines = vec!["abcdefghij".to_string()]; // 10 chars
        let visuals = wrap_visual_lines(&lines, 4);
        // chunks: "abcd" (0..4), "efgh" (4..8), "ij" (8..10)
        assert_eq!(logical_to_visual(&visuals, 0, 0), (0, 0));
        assert_eq!(logical_to_visual(&visuals, 0, 3), (0, 3));
        // Col == 4 falls through first chunk (exclusive end) so reports
        // at the start of the second chunk.
        assert_eq!(logical_to_visual(&visuals, 0, 4), (1, 0));
        assert_eq!(logical_to_visual(&visuals, 0, 7), (1, 3));
        assert_eq!(logical_to_visual(&visuals, 0, 8), (2, 0));
        assert_eq!(logical_to_visual(&visuals, 0, 10), (2, 2));
    }
}
