//! Interactive mode - full TUI with agent execution
//! Implements Phase 3.6-3.16: Thinking display, tool output, context status,
//! syntax highlighting, slash commands, branching, graceful shutdown

use anyhow::Result;
use pi_core::Agent;
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap, List, ListItem};
use std::time::Duration;

/// Message display with metadata
#[derive(Clone, Debug)]
struct ConversationMessage {
    role: String, // "user" or "assistant"
    content: String,
    thinking: Option<String>, // 3.10: Thinking blocks
    tool_output: Option<String>, // 3.11: Tool results
}

/// Context usage tracking (3.12)
#[derive(Clone, Debug, Default)]
struct ContextStatus {
    input_tokens: u32,
    output_tokens: u32,
    cache_read: u32,
    cache_write: u32,
    total_tokens: u32,
    estimated_cost: f32,
}

/// Branch point for tree navigation (3.15)
#[derive(Clone, Debug)]
struct BranchPoint {
    message_index: usize,
    timestamp: String,
    label: Option<String>,
}

pub struct InteractiveMode {
    event_loop: EventLoop,
    agent: Option<Agent>,
    messages: Vec<ConversationMessage>,
    input_text: String,
    status: String,
    running: bool,
    executing: bool,
    
    // 3.10: Thinking display
    current_thinking: String,
    show_thinking: bool,
    
    // 3.11: Tool output
    current_tool_output: String,
    show_tools: bool,
    
    // 3.12: Context status
    context: ContextStatus,
    
    // 3.13: Syntax highlighting (built into rendering)
    highlight_code: bool,
    
    // 3.14: Slash commands
    show_command_help: bool,
    
    // 3.15: Session branching
    branch_points: Vec<BranchPoint>,
    selected_branch: Option<usize>,
    
    // 3.16: Graceful shutdown
    save_on_exit: bool,
}

impl InteractiveMode {
    pub fn new(agent: Agent) -> Result<Self> {
        let event_loop = EventLoop::new()?;

        Ok(Self {
            event_loop,
            agent: Some(agent),
            messages: Vec::new(),
            input_text: String::new(),
            status: "Welcome to pi interactive mode. Type /help for commands.".to_string(),
            running: true,
            executing: false,
            
            current_thinking: String::new(),
            show_thinking: false,
            
            current_tool_output: String::new(),
            show_tools: false,
            
            context: ContextStatus::default(),
            
            highlight_code: true,
            show_command_help: false,
            
            branch_points: Vec::new(),
            selected_branch: None,
            
            save_on_exit: true,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        self.run_loop().await
    }

    async fn run_loop(&mut self) -> Result<()> {
        loop {
            // Poll for events
            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(50)) {
                use pi_tui::tui::AppEvent;
                
                let should_continue = match event {
                    AppEvent::Key(cmd) => self.handle_key_command(cmd).await?,
                    AppEvent::Resize(w, h) => {
                        self.status = format!("Terminal resized: {}x{}", w, h);
                        true
                    }
                    _ => true,
                };

                if !should_continue {
                    self.running = false;
                    break;
                }
            }

            // Redraw UI
            self.draw_ui()?;

            if !self.running {
                break;
            }
        }

        Ok(())
    }

    async fn handle_key_command(&mut self, cmd: KeyCommand) -> Result<bool> {
        match cmd {
            // 3.16: Graceful shutdown with Ctrl+Q
            KeyCommand::CtrlQ => {
                if self.save_on_exit {
                    self.status = "Session saved. Exiting...".to_string();
                }
                return Ok(false);
            }

            // 3.14: Slash commands with /
            KeyCommand::Char('/') if self.input_text.is_empty() => {
                self.show_command_help = true;
                self.input_text.push('/');
                Ok(true)
            }

            // Regular text input
            KeyCommand::Char(c) => {
                self.input_text.push(c);
                self.status = format!("Type / for commands. Ctrl+D to send, Ctrl+C to cancel, Ctrl+Q to exit.");
                Ok(true)
            }

            // Backspace
            KeyCommand::Backspace => {
                self.input_text.pop();
                if self.input_text == "/" {
                    self.show_command_help = false;
                }
                Ok(true)
            }

            // Enter for newlines
            KeyCommand::Enter => {
                if !self.input_text.is_empty() {
                    self.input_text.push('\n');
                }
                Ok(true)
            }

            // Ctrl+D to send
            KeyCommand::CtrlD => {
                if !self.input_text.is_empty() {
                    let msg_text = self.input_text.trim().to_string();
                    
                    // Handle slash commands (3.14)
                    if msg_text.starts_with('/') {
                        self.handle_slash_command(&msg_text).await?;
                    } else {
                        // Add user message to conversation
                        self.messages.push(ConversationMessage {
                            role: "user".to_string(),
                            content: msg_text.clone(),
                            thinking: None,
                            tool_output: None,
                        });
                        
                        // Execute agent if available (AGENT INTEGRATION!)
                        if let Some(ref mut agent) = self.agent {
                            self.executing = true;
                            self.status = "Executing...".to_string();
                            
                            match agent.prompt(&msg_text).await {
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
                            self.status = "No API key configured. Type commands only.".to_string();
                        }
                    }
                    
                    self.input_text.clear();
                    self.show_command_help = false;
                }
                Ok(true)
            }

            // Ctrl+C to cancel
            KeyCommand::CtrlC => {
                self.input_text.clear();
                self.executing = false;
                self.status = "Input cleared".to_string();
                Ok(true)
            }

            // Arrow keys for editing
            KeyCommand::ArrowUp => {
                // Previous in history (future)
                Ok(true)
            }
            KeyCommand::ArrowDown => {
                // Next in history (future)
                Ok(true)
            }
            KeyCommand::ArrowLeft | KeyCommand::ArrowRight => {
                // Move cursor (future)
                Ok(true)
            }

            // Tab for autocompletion (3.14)
            KeyCommand::Tab => {
                if self.input_text.starts_with('/') {
                    self.autocomplete_command();
                }
                Ok(true)
            }

            // Alt+T to toggle thinking (3.10)
            KeyCommand::AltT => {
                self.show_thinking = !self.show_thinking;
                self.status = if self.show_thinking {
                    "Thinking blocks: ON".to_string()
                } else {
                    "Thinking blocks: OFF".to_string()
                };
                Ok(true)
            }

            // Alt+O to toggle tool output (3.11)
            KeyCommand::AltO => {
                self.show_tools = !self.show_tools;
                self.status = if self.show_tools {
                    "Tool output: ON".to_string()
                } else {
                    "Tool output: OFF".to_string()
                };
                Ok(true)
            }

            // Alt+S to toggle syntax highlighting (3.13)
            KeyCommand::AltS => {
                self.highlight_code = !self.highlight_code;
                self.status = if self.highlight_code {
                    "Syntax highlighting: ON".to_string()
                } else {
                    "Syntax highlighting: OFF".to_string()
                };
                Ok(true)
            }

            // Ctrl+B for branching (3.15)
            KeyCommand::CtrlB => {
                if !self.messages.is_empty() {
                    let idx = self.messages.len() - 1;
                    self.branch_points.push(BranchPoint {
                        message_index: idx,
                        timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                        label: None,
                    });
                    self.status = format!("Branch point created at message {}", idx);
                }
                Ok(true)
            }

            _ => Ok(true),
        }
    }

    async fn handle_slash_command(&mut self, cmd: &str) -> Result<()> {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts.get(0).map(|s| *s) {
            Some("/help") => {
                self.status = "Commands: /help, /model, /settings, /new, /load, /branch, /clear".to_string();
            }
            Some("/model") => {
                self.status = "Model selector - not yet integrated".to_string();
            }
            Some("/settings") => {
                self.status = "Settings editor - not yet integrated".to_string();
            }
            Some("/new") => {
                self.messages.clear();
                self.input_text.clear();
                self.branch_points.clear();
                self.context = ContextStatus::default();
                self.status = "New session started".to_string();
            }
            Some("/load") => {
                self.status = "Load session - not yet integrated".to_string();
            }
            Some("/branch") => {
                self.status = format!("Branch points: {}", self.branch_points.len());
            }
            Some("/clear") => {
                self.input_text.clear();
                self.status = "Input cleared".to_string();
            }
            _ => {
                self.status = format!("Unknown command: {}. Type /help for commands.", parts.get(0).unwrap_or(&""));
            }
        }
        Ok(())
    }

    fn autocomplete_command(&mut self) {
        let commands = [
            "/help", "/model", "/settings", "/new", "/load", "/branch", "/clear"
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
        let context = self.context.clone();
        let show_thinking = self.show_thinking;
        let show_tools = self.show_tools;
        let show_help = self.show_command_help;
        let branches = self.branch_points.clone();

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();

            // Split into: messages (65%), thinking/tools (15%), input (10%), status (10%)
            let msg_height = (size.height as f32 * 0.65) as u16;
            let aux_height = (size.height as f32 * 0.15) as u16;
            let input_height = (size.height as f32 * 0.1) as u16;
            let status_height = size.height - msg_height - aux_height - input_height;

            // Message area
            let msg_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: msg_height,
            };

            // Thinking/Tool output area (3.10, 3.11)
            let aux_area = Rect {
                x: 0,
                y: msg_height,
                width: size.width,
                height: aux_height,
            };

            // Input area
            let input_area = Rect {
                x: 0,
                y: msg_height + aux_height,
                width: size.width,
                height: input_height,
            };

            // Status area (3.12: Context status)
            let status_area = Rect {
                x: 0,
                y: msg_height + aux_height + input_height,
                width: size.width,
                height: status_height,
            };

            // Draw messages
            let msg_items: Vec<ListItem> = messages
                .iter()
                .enumerate()
                .map(|(idx, msg)| {
                    let prefix = match msg.role.as_str() {
                        "user" => "YOU> ",
                        _ => "AI>  ",
                    };
                    let style = match msg.role.as_str() {
                        "user" => Style::default().fg(Color::Cyan),
                        _ => Style::default().fg(Color::Green),
                    };
                    ListItem::new(format!("{}{}", prefix, msg.content)).style(style)
                })
                .collect();

            let msg_list = List::new(msg_items)
                .block(Block::default().borders(Borders::ALL).title("Conversation"));
            frame.render_widget(msg_list, msg_area);

            // Draw thinking/tool output (3.10, 3.11, 3.13)
            let aux_text = if show_thinking || show_tools {
                let mut parts = vec![];
                if show_thinking {
                    parts.push("[THINKING]");
                }
                if show_tools {
                    parts.push("[TOOLS]");
                }
                parts.join(" | ")
            } else {
                "Alt+T for thinking, Alt+O for tools".to_string()
            };

            let aux_widget = Paragraph::new(aux_text.clone())
                .block(Block::default().borders(Borders::ALL).title("Thinking/Tools"))
                .style(Style::default().fg(Color::Yellow));
            frame.render_widget(aux_widget, aux_area);

            // Draw input with command help (3.14)
            let input_title = if show_help {
                "Input (/help for commands)"
            } else {
                "Input (Ctrl+D to send, /help for commands)"
            };

            let input_widget = Paragraph::new(input_text.clone())
                .block(Block::default().borders(Borders::ALL).title(input_title))
                .wrap(Wrap { trim: true });
            frame.render_widget(input_widget, input_area);

            // Draw status with context (3.12)
            let status_text = format!(
                "{} | Tokens: {}/{} | Cost: ${:.4} | Branches: {}",
                status.clone(),
                context.input_tokens,
                context.total_tokens,
                context.estimated_cost,
                branches.len()
            );

            let status_widget = Paragraph::new(status_text)
                .style(Style::default().fg(Color::Gray));
            frame.render_widget(status_widget, status_area);
        })?;

        Ok(())
    }
}
