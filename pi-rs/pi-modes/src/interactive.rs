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
use pi_core::Agent;
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use std::time::Duration;

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

/// Model info
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
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

/// Conversation message
#[derive(Clone, Debug)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

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
    model_list: Vec<ModelInfo>,
    model_selected: usize,
    settings_list: Vec<SettingInfo>,
    settings_selected: usize,

    // Escape/Ctrl+C tracking
    last_escape_time: Option<std::time::Instant>,
    ctrl_c_count: u32,
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
            model_selected: 0,
            settings_list: Vec::new(),
            settings_selected: 0,
            last_escape_time: None,
            ctrl_c_count: 0,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        loop {
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
            self.draw()?;
        }
        Ok(())
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
                    // Select model
                    if !self.model_list.is_empty() {
                        let m = &self.model_list[self.model_selected];
                        self.status = format!("Selected: {} ({})", m.name, m.provider);
                        self.display_mode = DisplayMode::Chat;
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
                        self.send_message(&msg).await?;
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
                self.input_text.push(c);

                // Activate palette on first /
                if self.input_text == "/" {
                    self.palette_active = true;
                    self.palette_items = self.all_commands.clone();
                    self.palette_selected = 0;
                    self.status = "Type to filter commands, ↑↓ to navigate, Enter to select".to_string();
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
                self.model_list = get_models_list();
                self.model_selected = 0;
                self.status = "Select a model - ↑↓ navigate, Enter to select, Esc to cancel".to_string();
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

    async fn send_message(&mut self, text: &str) -> Result<()> {
        self.messages.push(ConversationMessage {
            role: "user".to_string(),
            content: text.to_string(),
        });

        if let Some(ref mut agent) = self.agent {
            self.executing = true;
            self.status = "Executing...".to_string();

            match agent.prompt(text).await {
                Ok(response) => {
                    self.messages.push(ConversationMessage {
                        role: "assistant".to_string(),
                        content: response,
                    });
                    self.status = "Ready".to_string();
                }
                Err(e) => {
                    self.status = format!("Error: {}", e);
                }
            }
            self.executing = false;
        } else {
            self.status = "No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.".to_string();
        }
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        // Snapshot state for closure
        let messages = self.messages.clone();
        let input_text = self.input_text.clone();
        let status = self.status.clone();
        let queued_count = self.queued_messages.len();
        let executing = self.executing;
        let show_thinking = self.show_thinking;
        let show_tools = self.show_tools;
        let palette_active = self.palette_active;
        let palette_items = self.palette_items.clone();
        let palette_selected = self.palette_selected;
        let display_mode = self.display_mode;
        let model_list = self.model_list.clone();
        let model_selected = self.model_selected;
        let settings_list = self.settings_list.clone();
        let settings_selected = self.settings_selected;

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();
            let messages_h = (size.height as f32 * 0.70) as u16;
            let input_h = (size.height as f32 * 0.20) as u16;
            let footer_h = size.height.saturating_sub(messages_h + input_h);

            let msg_area = Rect { x: 0, y: 0, width: size.width, height: messages_h };
            let input_area = Rect { x: 0, y: messages_h, width: size.width, height: input_h };
            let footer_area = Rect { x: 0, y: messages_h + input_h, width: size.width, height: footer_h };

            // === MESSAGES / MODEL LIST / SETTINGS LIST ===
            if palette_active && !palette_items.is_empty() {
                // Show command palette in message area
                let items: Vec<ListItem> = palette_items
                    .iter()
                    .enumerate()
                    .map(|(i, cmd)| {
                        let style = if i == palette_selected {
                            Style::default().bg(Color::Cyan).fg(Color::Black)
                        } else {
                            Style::default()
                        };
                        let text = format!("  /{:<15}  {}", cmd.name, cmd.description);
                        ListItem::new(text).style(style)
                    })
                    .collect();
                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("Commands ({}) - ↑↓ navigate, Enter to select, Esc to cancel", palette_items.len())),
                );
                frame.render_widget(list, msg_area);
            } else if display_mode == DisplayMode::ModelList && !model_list.is_empty() {
                // Show model list grouped by provider
                let mut items: Vec<ListItem> = vec![];
                let mut current_provider = String::new();
                for (i, m) in model_list.iter().enumerate() {
                    if m.provider != current_provider {
                        current_provider = m.provider.clone();
                        items.push(
                            ListItem::new(format!("── {} ──", current_provider.to_uppercase()))
                                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        );
                    }
                    let style = if i == model_selected {
                        Style::default().bg(Color::Cyan).fg(Color::Black)
                    } else {
                        Style::default()
                    };
                    items.push(ListItem::new(format!("  {} ({})", m.name, m.id)).style(style));
                }
                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Models - ↑↓ navigate, Enter to select, Esc to cancel"),
                );
                frame.render_widget(list, msg_area);
            } else if display_mode == DisplayMode::SettingsList && !settings_list.is_empty() {
                // Show settings list
                let items: Vec<ListItem> = settings_list
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let style = if i == settings_selected {
                            Style::default().bg(Color::Cyan).fg(Color::Black)
                        } else {
                            Style::default()
                        };
                        let text = format!(
                            "  {:<25} = {:<15}  ({})",
                            s.name, s.current, s.options
                        );
                        ListItem::new(text).style(style)
                    })
                    .collect();
                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Settings - ↑↓ navigate, Enter to select, Esc to cancel"),
                );
                frame.render_widget(list, msg_area);
            } else {
                // Show conversation
                let mut lines: Vec<ListItem> = vec![];
                for msg in &messages {
                    let prefix = if msg.role == "user" { "YOU" } else { "AI " };
                    let color = if msg.role == "user" { Color::Cyan } else { Color::Green };
                    for line in msg.content.lines() {
                        lines.push(
                            ListItem::new(format!("{} > {}", prefix, line))
                                .style(Style::default().fg(color)),
                        );
                    }
                    lines.push(ListItem::new(""));
                }
                if show_thinking {
                    lines.push(ListItem::new("[Thinking blocks visible]").style(Style::default().fg(Color::Magenta)));
                }
                if show_tools {
                    lines.push(ListItem::new("[Tool output visible]").style(Style::default().fg(Color::Yellow)));
                }
                let list = List::new(lines).block(Block::default().borders(Borders::ALL).title("Conversation"));
                frame.render_widget(list, msg_area);
            }

            // === INPUT ===
            let input_title = if executing {
                "INPUT (executing...)".to_string()
            } else if queued_count > 0 {
                format!("INPUT ({} queued)", queued_count)
            } else if palette_active {
                format!("INPUT - {} commands available", palette_items.len())
            } else {
                "INPUT - Enter to send, Alt+Enter to queue, / for commands".to_string()
            };
            let input_style = if executing { Style::default().fg(Color::Yellow) } else { Style::default() };
            let input_widget = Paragraph::new(input_text.clone())
                .block(Block::default().borders(Borders::ALL).title(input_title))
                .style(input_style)
                .wrap(Wrap { trim: false });
            frame.render_widget(input_widget, input_area);

            // === FOOTER ===
            let mode_indicator = match display_mode {
                DisplayMode::Chat => if executing { "⏳" } else { "✓" },
                DisplayMode::ModelList => "📦",
                DisplayMode::SettingsList => "⚙",
            };
            let footer_text = format!(
                "{}  {}  Messages: {}  Queued: {}",
                mode_indicator,
                status,
                messages.len(),
                queued_count,
            );
            let footer_widget = Paragraph::new(footer_text).style(Style::default().fg(Color::Gray));
            frame.render_widget(footer_widget, footer_area);
        })?;

        Ok(())
    }
}
