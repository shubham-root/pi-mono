//! Interactive mode - TUI matching TypeScript variant exactly
//! Implements Phase 3 with proper UX:
//! - Enter to send message
//! - Alt+Enter to queue follow-up
//! - Escape to cancel
//! - Escape twice for tree
//! - Ctrl+L for model selector
//! - Ctrl+T for thinking toggle
//! - Ctrl+O for tool output toggle
//! - Ctrl+C to clear, Ctrl+C twice to quit

use anyhow::Result;
use pi_core::Agent;
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap, List, ListItem, Gauge};
use std::time::Duration;
use chrono::Local;

/// Slash command info
#[derive(Clone, Debug)]
pub struct SlashCommand {
    pub name: String,
    pub description: String,
}

/// Get all available slash commands
fn get_all_slash_commands() -> Vec<SlashCommand> {
    vec![
        SlashCommand { name: "help".to_string(), description: "Show keyboard shortcuts".to_string() },
        SlashCommand { name: "hotkeys".to_string(), description: "Show all keyboard shortcuts".to_string() },
        SlashCommand { name: "model".to_string(), description: "Switch models (Ctrl+L also works)".to_string() },
        SlashCommand { name: "settings".to_string(), description: "Edit settings (theme, thinking level)".to_string() },
        SlashCommand { name: "login".to_string(), description: "OAuth authentication".to_string() },
        SlashCommand { name: "logout".to_string(), description: "Clear authentication".to_string() },
        SlashCommand { name: "new".to_string(), description: "Start fresh session".to_string() },
        SlashCommand { name: "tree".to_string(), description: "Show session tree (Escape 2x also works)".to_string() },
        SlashCommand { name: "session".to_string(), description: "Show session info (tokens, cost, path)".to_string() },
        SlashCommand { name: "fork".to_string(), description: "Fork session from current point".to_string() },
        SlashCommand { name: "export".to_string(), description: "Export session to HTML file".to_string() },
        SlashCommand { name: "share".to_string(), description: "Share session as GitHub gist".to_string() },
        SlashCommand { name: "compact".to_string(), description: "Manual context compaction".to_string() },
        SlashCommand { name: "copy".to_string(), description: "Copy last assistant message".to_string() },
        SlashCommand { name: "reload".to_string(), description: "Reload config and extensions".to_string() },
        SlashCommand { name: "quit".to_string(), description: "Exit pi".to_string() },
    ]
}

/// Filter commands by query
fn filter_commands(query: &str, all_commands: &[SlashCommand]) -> Vec<SlashCommand> {
    let query_lower = query.to_lowercase();
    all_commands
        .iter()
        .filter(|cmd| cmd.name.contains(&query_lower))
        .cloned()
        .collect()
}


/// UI display mode
#[derive(Clone, Copy, Debug, PartialEq)]
enum DisplayMode {
    Chat,
    ModelList,
    SettingsList,
}

/// Model info for display
#[derive(Clone, Debug)]
struct ModelInfo {
    id: String,
    name: String,
    provider: String,
}

/// Setting info for display
#[derive(Clone, Debug)]
struct SettingInfo {
    name: String,
    description: String,
    current: String,
    options: String,
}

fn get_models_list() -> Vec<ModelInfo> {
    vec![
        ModelInfo { id: "claude-opus-4-7".to_string(), name: "Claude Opus 4.7".to_string(), provider: "anthropic".to_string() },
        ModelInfo { id: "claude-sonnet-4-6".to_string(), name: "Claude Sonnet 4.6".to_string(), provider: "anthropic".to_string() },
        ModelInfo { id: "claude-haiku-4-5".to_string(), name: "Claude Haiku 4.5".to_string(), provider: "anthropic".to_string() },
        ModelInfo { id: "gpt-4o".to_string(), name: "GPT-4o".to_string(), provider: "openai".to_string() },
        ModelInfo { id: "gpt-4-turbo".to_string(), name: "GPT-4 Turbo".to_string(), provider: "openai".to_string() },
        ModelInfo { id: "gpt-3.5-turbo".to_string(), name: "GPT-3.5 Turbo".to_string(), provider: "openai".to_string() },
        ModelInfo { id: "gemini-2-pro".to_string(), name: "Gemini 2 Pro".to_string(), provider: "google".to_string() },
        ModelInfo { id: "gemini-2-flash".to_string(), name: "Gemini 2 Flash".to_string(), provider: "google".to_string() },
        ModelInfo { id: "claude-opus-bedrock".to_string(), name: "Claude Opus (Bedrock)".to_string(), provider: "bedrock".to_string() },
        ModelInfo { id: "nova-pro".to_string(), name: "Nova Pro".to_string(), provider: "bedrock".to_string() },
    ]
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


/// Message display with metadata
#[derive(Clone, Debug)]
pub struct ConversationMessage {
    pub role: String, // "user" or "assistant"
    pub content: String,
    pub thinking: Option<String>,
    pub tool_output: Option<String>,
}

/// Context usage tracking
#[derive(Clone, Debug, Default)]
pub struct ContextStatus {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read: u32,
    pub cache_write: u32,
    pub total_tokens: u32,
    pub estimated_cost: f32,
}

pub struct InteractiveMode {
    event_loop: EventLoop,
    agent: Option<Agent>,
    messages: Vec<ConversationMessage>,
    input_text: String,
    queued_messages: Vec<String>,
    
    // UI state
    status: String,
    running: bool,
    executing: bool,
    
    // Display toggles
    show_thinking: bool,
    show_tools: bool,
    
    // Context tracking
    context: ContextStatus,
    
    // Tree navigation
    branch_points: Vec<BranchPoint>,
    selected_branch: Option<usize>,
    
    // Keybinding state
    last_escape_time: Option<std::time::Instant>,
    ctrl_c_count: u32,
    
    // Model selector state
    show_model_selector: bool,
    selected_model_idx: usize,    
    // Command palette
    command_palette_active: bool,
    command_palette_filtered: Vec<SlashCommand>,
    command_palette_selected: usize,
    all_slash_commands: Vec<SlashCommand>,    
    // Display mode
    display_mode: DisplayMode,
    model_list: Vec<ModelInfo>,
    model_selected: usize,
    settings_list: Vec<SettingInfo>,
    settings_selected: usize,
}

#[derive(Clone, Debug)]
struct BranchPoint {
    message_index: usize,
    timestamp: String,
    label: Option<String>,
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
            
            status: "Ready to chat. Type / for commands. Ctrl+L for model selector.".to_string(),
            running: true,
            executing: false,
            
            show_thinking: false,
            show_tools: false,
            
            context: ContextStatus::default(),
            
            branch_points: Vec::new(),
            selected_branch: None,
            
            last_escape_time: None,
            ctrl_c_count: 0,
            
            show_model_selector: false,
            selected_model_idx: 0,            
            command_palette_active: false,
            command_palette_filtered: vec![],
            command_palette_selected: 0,
            all_slash_commands: get_all_slash_commands(),            
            display_mode: DisplayMode::Chat,
            model_list: vec![],
            model_selected: 0,
            settings_list: vec![],
            settings_selected: 0,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        self.run_loop().await
    }

    async fn run_loop(&mut self) -> Result<()> {
        let mut last_tick = std::time::Instant::now();
        
        loop {
            // Poll for events with 50ms timeout
            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(50)) {
                use pi_tui::tui::AppEvent;
                
                let should_continue = match event {
                    AppEvent::Key(cmd) => self.handle_key_command(cmd).await?,
                    AppEvent::Resize(w, h) => {
                        self.status = format!("Terminal: {}x{}", w, h);
                        true
                    }
                    AppEvent::Tick => true,
                    _ => true,
                };

                if !should_continue {
                    self.running = false;
                    break;
                }
            }

            // Redraw UI every 50ms
            self.draw_ui()?;

            // Check for escape timeout (2 seconds)
            let now = std::time::Instant::now();
            if now.duration_since(last_tick) > Duration::from_secs(2) {
                self.ctrl_c_count = 0;
                last_tick = now;
            }

            if !self.running {
                break;
            }
        }

        Ok(())
    }

    async fn handle_key_command(&mut self, cmd: KeyCommand) -> Result<bool> {
        match cmd {
            // **SEND MESSAGE or SELECT COMMAND** - Enter key
            KeyCommand::Enter => {
                if self.display_mode == DisplayMode::ModelList {
                    if !self.model_list.is_empty() && self.model_selected < self.model_list.len() {
                        let selected = &self.model_list[self.model_selected];
                        self.status = format!("Selected model: {} ({})", selected.name, selected.provider);
                        self.display_mode = DisplayMode::Chat;
                    }
                } else if self.display_mode == DisplayMode::SettingsList {
                    if !self.settings_list.is_empty() && self.settings_selected < self.settings_list.len() {
                        let selected = &self.settings_list[self.settings_selected];
                        self.status = format!("Edit {}: {} (current: {})", selected.name, selected.description, selected.current);
                        self.display_mode = DisplayMode::Chat;
                    }
                } else if self.command_palette_active {
                    // Select from command palette
                    if !self.command_palette_filtered.is_empty() {
                        let cmd = self.command_palette_filtered[self.command_palette_selected].name.clone();
                        self.input_text = format!("/{}", cmd);
                        self.command_palette_active = false;
                        self.command_palette_filtered.clear();
                        self.command_palette_selected = 0;
                    }
                } else if self.show_model_selector {
                    // In model selector, Enter to select model
                    self.show_model_selector = false;
                    self.status = "Model selected".to_string();
                } else if !self.input_text.is_empty() {
                    let msg = self.input_text.trim().to_string();
                    
                    if msg.starts_with('/') {
                        self.handle_slash_command(&msg).await?;
                    } else {
                        self.send_user_message(&msg).await?;
                    }
                    
                    self.input_text.clear();
                }
                Ok(true)
            }

            // **QUEUE FOLLOW-UP** - Alt+Enter (queue after agent finishes)
            KeyCommand::AltEnter => {
                if !self.input_text.is_empty() {
                    self.queued_messages.push(self.input_text.trim().to_string());
                    self.status = format!("Queued message ({})", self.queued_messages.len());
                    self.input_text.clear();
                }
                Ok(true)
            }

            // **NEWLINE IN INPUT** - Shift+Enter or just for multiline (we'll treat Enter only)
            KeyCommand::ShiftEnter => {
                self.input_text.push('\n');
                Ok(true)
            }

            // **CLEAR EDITOR / QUIT** - Ctrl+C (clear once, quit on second press)
            KeyCommand::CtrlC => {
                self.ctrl_c_count += 1;
                
                if self.ctrl_c_count == 1 {
                    if !self.input_text.is_empty() {
                        self.input_text.clear();
                        self.status = "Input cleared (Ctrl+C again to quit)".to_string();
                    } else {
                        self.status = "Press Ctrl+C again to quit".to_string();
                    }
                    
                    // Reset counter after 2 seconds
                    std::thread::sleep(Duration::from_millis(50));
                } else if self.ctrl_c_count >= 2 {
                    return Ok(false); // Exit
                }
                
                Ok(true)
            }

            // **CANCEL EXECUTION** - Escape (single), Tree Navigation (double)
            KeyCommand::Escape => {
                // If in list mode, go back to chat
                if self.display_mode == DisplayMode::ModelList || self.display_mode == DisplayMode::SettingsList {
                    self.display_mode = DisplayMode::Chat;
                    self.status = "Back to chat".to_string();
                    return Ok(true);
                }
                
                let now = std::time::Instant::now();
                let time_since_last = if let Some(last) = self.last_escape_time { now.duration_since(last) } else { Duration::from_secs(10) };
                
                if time_since_last < Duration::from_millis(500) {
                    // Double escape - open tree selector
                    self.status = "Tree selector - not yet wired".to_string();
                } else {
                    // Single escape - cancel/abort
                    if self.executing {
                        self.executing = false;
                        self.status = "Execution cancelled".to_string();
                    } else if !self.input_text.is_empty() {
                        self.input_text.clear();
                        self.status = "Input cleared".to_string();
                    } else if !self.queued_messages.is_empty() {
                        // Restore queued messages to editor
                        self.input_text = self.queued_messages.pop().unwrap_or_default();
                        self.status = format!("Restored queued message. {} remaining", self.queued_messages.len());
                    }
                }
                
                self.last_escape_time = Some(std::time::Instant::now());
                Ok(true)
            }

            // **MODEL SELECTOR** - Ctrl+L
            KeyCommand::CtrlL => {
                self.show_model_selector = !self.show_model_selector;
                self.status = if self.show_model_selector {
                    "Model selector - use ↑↓ to navigate, Enter to select".to_string()
                } else {
                    "Model selector closed".to_string()
                };
                Ok(true)
            }

            // **CYCLE MODELS FORWARD** - Ctrl+P
            KeyCommand::CtrlP => {
                self.selected_model_idx += 1;
                self.status = format!("Model index: {}", self.selected_model_idx);
                Ok(true)
            }

            // **CYCLE MODELS BACKWARD** - Shift+Ctrl+P
            KeyCommand::ShiftCtrlP => {
                self.selected_model_idx = self.selected_model_idx.saturating_sub(1);
                self.status = format!("Model index: {}", self.selected_model_idx);
                Ok(true)
            }

            // **TOGGLE THINKING** - Ctrl+T
            KeyCommand::CtrlT => {
                self.show_thinking = !self.show_thinking;
                self.status = if self.show_thinking {
                    "Thinking blocks: ON".to_string()
                } else {
                    "Thinking blocks: OFF".to_string()
                };
                Ok(true)
            }

            // **TOGGLE TOOL OUTPUT** - Ctrl+O
            KeyCommand::CtrlO => {
                self.show_tools = !self.show_tools;
                self.status = if self.show_tools {
                    "Tool output: ON".to_string()
                } else {
                    "Tool output: OFF".to_string()
                };
                Ok(true)
            }

            // **THINKING LEVEL** - Shift+Tab
            KeyCommand::ShiftTab => {
                self.status = "Thinking level cycled (not yet implemented)".to_string();
                Ok(true)
            }

            // **REGULAR TEXT INPUT** or COMMAND PALETTE
            KeyCommand::Char(c) => {
                self.ctrl_c_count = 0;
                self.input_text.push(c);
                
                // Activate command palette when '/' is typed
                if self.input_text == "/" {
                    self.command_palette_active = true;
                    self.command_palette_filtered = self.all_slash_commands.clone();
                    self.command_palette_selected = 0;
                    self.status = "Type command name to filter, ↑↓ to navigate, Enter to select".to_string();
                } else if self.input_text.starts_with('/') && self.command_palette_active {
                    // Filter commands as user types
                    let query = self.input_text[1..].trim(); // Remove '/' prefix
                    self.command_palette_filtered = filter_commands(query, &self.all_slash_commands);
                    self.command_palette_selected = 0;
                }
                
                Ok(true)
            }

            // **BACKSPACE**
            KeyCommand::Backspace => {
                self.ctrl_c_count = 0;
                self.input_text.pop();
                
                // Keep palette active while typing command
                if self.input_text.starts_with('/') && self.input_text.len() > 1 {
                    let query = self.input_text[1..].trim();
                    self.command_palette_filtered = filter_commands(query, &self.all_slash_commands);
                    self.command_palette_selected = 0;
                } else if self.input_text == "/" {
                    self.command_palette_active = true;
                    self.command_palette_filtered = self.all_slash_commands.clone();
                    self.command_palette_selected = 0;
                } else {
                    self.command_palette_active = false;
                    self.command_palette_filtered.clear();
                }
                
                Ok(true)
            }

            // **NAVIGATION IN MODEL SELECTOR or COMMAND PALETTE**
            KeyCommand::ArrowUp => {
                if self.display_mode == DisplayMode::ModelList && !self.model_list.is_empty() {
                    self.model_selected = self.model_selected.saturating_sub(1);
                } else if self.display_mode == DisplayMode::SettingsList && !self.settings_list.is_empty() {
                    self.settings_selected = self.settings_selected.saturating_sub(1);
                } else if self.command_palette_active && !self.command_palette_filtered.is_empty() {
                    self.command_palette_selected = self.command_palette_selected.saturating_sub(1);
                }
                Ok(true)
            }
            KeyCommand::ArrowDown => {
                if self.display_mode == DisplayMode::ModelList && !self.model_list.is_empty() {
                    self.model_selected = (self.model_selected + 1).min(self.model_list.len() - 1);
                } else if self.display_mode == DisplayMode::SettingsList && !self.settings_list.is_empty() {
                    self.settings_selected = (self.settings_selected + 1).min(self.settings_list.len() - 1);
                } else if self.command_palette_active && !self.command_palette_filtered.is_empty() {
                    self.command_palette_selected = (self.command_palette_selected + 1).min(self.command_palette_filtered.len() - 1);
                }
                Ok(true)
            }

            // **TAB - AUTOCOMPLETE**
            KeyCommand::Tab => {
                if self.command_palette_active && !self.command_palette_filtered.is_empty() {
                    // Auto-complete selected command
                    let cmd = self.command_palette_filtered[self.command_palette_selected].name.clone();
                    self.input_text = format!("/{}", cmd);
                    self.command_palette_active = false;
                    self.command_palette_filtered.clear();
                }
                Ok(true)
            }

            _ => Ok(true),
        }
    }

    async fn send_user_message(&mut self, text: &str) -> Result<()> {
        // Add user message
        self.messages.push(ConversationMessage {
            role: "user".to_string(),
            content: text.to_string(),
            thinking: None,
            tool_output: None,
        });

        // Execute agent if available
        if let Some(ref mut agent) = self.agent {
            self.executing = true;
            self.status = "Executing...".to_string();

            match agent.prompt(text).await {
                Ok(response) => {
                    self.messages.push(ConversationMessage {
                        role: "assistant".to_string(),
                        content: response,
                        thinking: None,
                        tool_output: None,
                    });
                    self.status = "Ready".to_string();
                }
                Err(e) => {
                    self.status = format!("Error: {}", e);
                }
            }
            self.executing = false;
        } else {
            self.status = "No API key configured".to_string();
        }

        Ok(())
    }

    async fn handle_slash_command(&mut self, cmd: &str) -> Result<()> {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts.get(0).map(|s| *s) {
            Some("/help") | Some("/hotkeys") => {
                self.status = "Enter: send | Alt+Enter: queue | Escape: cancel | Ctrl+L: models | Ctrl+T: thinking | Ctrl+O: tools | Ctrl+C: quit".to_string();
            }
            Some("/model") => {
                self.display_mode = DisplayMode::ModelList;
                self.model_list = get_models_list();
                self.model_selected = 0;
                self.status = "Select model - ↑↓ to navigate, Enter to select, Escape to cancel".to_string();
            }
            Some("/settings") => {
                self.display_mode = DisplayMode::SettingsList;
                self.settings_list = get_settings_list();
                self.settings_selected = 0;
                self.status = "Select setting - ↑↓ to navigate, Enter to edit, Escape to cancel".to_string();
            }
            Some("/login") => {
                self.status = "Login dialog - not yet wired".to_string();
            }
            Some("/new") => {
                self.display_mode = DisplayMode::Chat;
                self.messages.clear();
                self.queued_messages.clear();
                self.input_text.clear();
                self.branch_points.clear();
                self.context = ContextStatus::default();
                self.status = "New session started".to_string();
            }
            Some("/tree") => {
                self.status = format!("Session tree - {} messages, {} branches", self.messages.len(), self.branch_points.len());
            }
            Some("/session") => {
                self.status = format!("Session: {} messages, {} tokens, ${:.4}", 
                    self.messages.len(), 
                    self.context.total_tokens,
                    self.context.estimated_cost
                );
            }
            Some("/fork") => {
                self.status = "Fork session - not yet implemented".to_string();
            }
            Some("/export") => {
                self.status = "Export session - not yet implemented".to_string();
            }
            Some("/quit") => {
                return Ok(());
            }
            _ => {
                self.status = format!("Unknown command: {}", parts.get(0).unwrap_or(&""));
            }
        }
        Ok(())
    }

    fn autocomplete_slash_command(&mut self) {
        let commands = [
            "/help", "/model", "/settings", "/login", "/new", "/tree", "/session",
            "/fork", "/export", "/quit", "/hotkeys"
        ];

        for cmd in &commands {
            if cmd.starts_with(&self.input_text) {
                self.input_text = cmd.to_string();
                break;
            }
        }
    }

    fn draw_ui(&mut self) -> Result<()> {
        let messages = self.messages.clone();
        let input_text = self.input_text.clone();
        let status = self.status.clone();
        let queued_count = self.queued_messages.len();
        let executing = self.executing;
        let show_thinking = self.show_thinking;
        let show_tools = self.show_tools;
        let show_model_selector = self.show_model_selector;

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();

            // Main layout: 70% messages, 20% input, 10% footer
            let messages_height = (size.height as f32 * 0.70) as u16;
            let input_height = (size.height as f32 * 0.20) as u16;
            let footer_height = size.height - messages_height - input_height;

            // Messages pane
            let messages_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: messages_height,
            };

            // Input pane
            let input_area = Rect {
                x: 0,
                y: messages_height,
                width: size.width,
                height: input_height,
            };

            // Footer
            let footer_area = Rect {
                x: 0,
                y: messages_height + input_height,
                width: size.width,
                height: footer_height,
            };

            
            // === MODEL LIST ===
            if self.display_mode == DisplayMode::ModelList && !self.model_list.is_empty() {
                let msg_items: Vec<ListItem> = self.model_list
                    .iter()
                    .enumerate()
                    .map(|(idx, model)| {
                        let style = if idx == self.model_selected {
                            Style::default().bg(Color::DarkGray).fg(Color::White)
                        } else {
                            Style::default()
                        };
                        let text = format!("{:<15} {:<40} ({})", model.name, model.id, model.provider);
                        ListItem::new(text).style(style)
                    })
                    .collect();

                let msg_list = List::new(msg_items)
                    .block(Block::default().borders(Borders::ALL).title("Models - Select with ↑↓, Enter to choose"));
                frame.render_widget(msg_list, messages_area);
            }
            // === SETTINGS LIST ===
            else if self.display_mode == DisplayMode::SettingsList && !self.settings_list.is_empty() {
                let settings_items: Vec<ListItem> = self.settings_list
                    .iter()
                    .enumerate()
                    .map(|(idx, setting)| {
                        let style = if idx == self.settings_selected {
                            Style::default().bg(Color::DarkGray).fg(Color::White)
                        } else {
                            Style::default()
                        };
                        let text = format!("{:<25} = {:<20} {}", setting.name, setting.current, setting.options);
                        ListItem::new(text).style(style)
                    })
                    .collect();

                let settings_list = List::new(settings_items)
                    .block(Block::default().borders(Borders::ALL).title("Settings - Select with ↑↓, Enter to edit"));
                frame.render_widget(settings_list, messages_area);
            }
            // === NORMAL CHAT MODE ===
            else if self.display_mode == DisplayMode::Chat {
                // === MESSAGES PANE ===
                let mut message_lines = vec![];
                for msg in &messages {
                    let prefix = if msg.role == "user" { "YOU " } else { "AI  " };
                    let color = if msg.role == "user" { Color::Cyan } else { Color::Green };
                    
                    for line in msg.content.lines() {
                        message_lines.push(line.to_string());
                    }
                }

                // Show thinking if toggled
                if show_thinking {
                    message_lines.push("".to_string());
                    message_lines.push("[THINKING BLOCKS]".to_string());
                }

                // Show tools if toggled
                if show_tools {
                    message_lines.push("".to_string());
                    message_lines.push("[TOOL OUTPUT]".to_string());
                }


                let msg_items: Vec<ListItem> = message_lines
                .iter()
                .map(|line| ListItem::new(line.clone()))
                .collect();

            let msg_list = List::new(msg_items)
                .block(Block::default().borders(Borders::ALL).title("Conversation"));
            frame.render_widget(msg_list, messages_area);

            // === INPUT PANE ===
            let input_title = if executing {
                format!("INPUT (executing...)")
            } else if queued_count > 0 {
                format!("INPUT ({} queued)", queued_count)
            } else if show_model_selector {
                "MODEL SELECTOR - ↑↓ navigate, Enter to select".to_string()
            } else {
                "INPUT - Enter to send, Alt+Enter to queue, / for commands".to_string()
            };

            let input_widget = Paragraph::new(input_text.clone())
                .block(Block::default().borders(Borders::ALL).title(input_title))
                .style(if executing { Style::default().fg(Color::Yellow) } else { Style::default() })
                .wrap(Wrap { trim: true });
            frame.render_widget(input_widget, input_area);

            } // End Chat mode
            // === FOOTER ===
            let footer_text = format!(
                "{} | Messages: {} | Tokens: {}/{} | Cost: ${:.4} | {}",
                status.clone(),
                messages.len(),
                0,  // TODO: actual token count
                5000,
                0.0,
                if executing { "⏳ EXECUTING" } else { "✓ READY" }
            );

            let footer_widget = Paragraph::new(footer_text)
                .style(Style::default().fg(if executing { Color::Yellow } else { Color::Gray }));
            frame.render_widget(footer_widget, footer_area);
        })?;

        Ok(())
    }
}
