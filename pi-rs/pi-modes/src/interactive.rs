//! Interactive mode - full TUI with agent execution
//! Implements Phase 3.6

use anyhow::Result;
use pi_core::Agent;
use pi_tui::{input::KeyCommand, tui::AppEvent, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tokio::task;

pub struct InteractiveMode {
    event_loop: EventLoop,
    messages: Vec<(String, String)>,
    input_text: String,
    status: String,
    running: bool,
    executing: bool,
    agent_tx: mpsc::Sender<String>,
    agent_rx: mpsc::Receiver<String>,
}

impl InteractiveMode {
    pub fn new(agent: Agent) -> Result<Self> {
        let mut event_loop = EventLoop::new()?;
        event_loop.set_agent(agent);

        let (tx, rx) = mpsc::channel();

        Ok(Self {
            event_loop,
            messages: Vec::new(),
            input_text: String::new(),
            status: "Ready - Ctrl+D to send, Ctrl+C to exit, Ctrl+L to load session".to_string(),
            running: true,
            executing: false,
            agent_tx: tx,
            agent_rx: rx,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        self.run_loop().await
    }

    async fn run_loop(&mut self) -> Result<()> {
        while self.running {
            // Check for agent responses
            while let Ok(msg) = self.agent_rx.try_recv() {
                self.messages.push(("Agent".to_string(), msg));
                self.executing = false;
            }

            // Poll for events
            let should_continue = if let Some(event) = self.event_loop.poll_event(Duration::from_millis(50)) {
                self.handle_event(event).await?
            } else {
                true
            };

            if !should_continue {
                self.running = false;
                break;
            }

            // Redraw UI
            self.draw_ui()?;
        }

        Ok(())
    }

    fn draw_ui(&mut self) -> Result<()> {
        let messages = self.messages.clone();
        let input_text = self.input_text.clone();
        let status = self.status.clone();
        let executing = self.executing;

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();

            // Split layout: 80% messages, 20% input
            let msg_height = (size.height as f32 * 0.8) as u16;

            let msg_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: msg_height.saturating_sub(1),
            };

            let input_area = Rect {
                x: 0,
                y: msg_height,
                width: size.width,
                height: (size.height - msg_height).saturating_sub(1),
            };

            let status_area = Rect {
                x: 0,
                y: size.height.saturating_sub(1),
                width: size.width,
                height: 1,
            };

            // Draw messages with scroll
            let message_lines: Vec<Line> = messages
                .iter()
                .flat_map(|(role, msg)| {
                    let style = if role == "User" {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::Green)
                    };

                    let role_line = Line::from(Span::styled(
                        format!("{}: ", role),
                        style.bold(),
                    ));

                    let msg_lines: Vec<Line> = msg
                        .lines()
                        .map(|line| Line::from(Span::raw(format!("  {}", line))))
                        .collect();

                    let mut result = vec![role_line];
                    result.extend(msg_lines);
                    result.push(Line::from(""));
                    result
                })
                .collect();

            let messages_widget = Paragraph::new(message_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Conversation"),
                )
                .wrap(Wrap { trim: true });
            frame.render_widget(messages_widget, msg_area);

            // Draw input
            let input_title = if executing {
                "Input (Waiting for response...)"
            } else {
                "Input (Ctrl+D to send)"
            };

            let input_widget = Paragraph::new(input_text.clone())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(input_title),
                )
                .wrap(Wrap { trim: true });
            frame.render_widget(input_widget, input_area);

            // Draw status
            let status_style = if executing {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Gray)
            };

            let status_widget = Paragraph::new(status.clone())
                .style(status_style);
            frame.render_widget(status_widget, status_area);
        })?;

        Ok(())
    }

    async fn handle_event(&mut self, event: AppEvent) -> Result<bool> {
        match event {
            AppEvent::Key(cmd) => Ok(self.handle_key_command(cmd)),
            AppEvent::Resize(_, _) => Ok(true),
            AppEvent::Tick => Ok(true),
            AppEvent::AgentMessage(msg) => {
                self.messages.push(("Agent".to_string(), msg));
                self.executing = false;
                Ok(true)
            }
            _ => Ok(true),
        }
    }

    fn handle_key_command(&mut self, cmd: KeyCommand) -> bool {
        if self.executing {
            // Only allow Ctrl+C to cancel
            return cmd != KeyCommand::CtrlC;
        }

        match cmd {
            KeyCommand::CtrlC => false, // Exit
            KeyCommand::CtrlD => {
                // Submit message
                if !self.input_text.is_empty() {
                    self.messages.push(("User".to_string(), self.input_text.clone()));
                    let prompt = self.input_text.clone();
                    self.input_text.clear();
                    self.executing = true;
                    self.status = "Executing...".to_string();

                    // TODO: Execute agent.prompt(prompt) asynchronously
                    // For now, just simulate a response
                    let tx = self.agent_tx.clone();
                    thread::spawn(move || {
                        thread::sleep(Duration::from_secs(1));
                        let _ = tx.send(format!("Received: {}", prompt));
                    });
                }
                true
            }
            KeyCommand::Char(c) => {
                self.input_text.push(c);
                true
            }
            KeyCommand::Backspace => {
                self.input_text.pop();
                true
            }
            KeyCommand::Delete => {
                // Delete forward (not implemented yet)
                true
            }
            KeyCommand::Enter => {
                self.input_text.push('\n');
                true
            }
            KeyCommand::CtrlU => {
                // Clear line
                self.input_text.clear();
                true
            }
            KeyCommand::Home => {
                // TODO: Implement cursor movement
                true
            }
            KeyCommand::End => {
                // TODO: Implement cursor movement
                true
            }
            _ => true,
        }
    }
}
