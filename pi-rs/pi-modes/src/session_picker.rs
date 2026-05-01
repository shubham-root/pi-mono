//! Session picker - load and manage sessions
//! Implements Phase 3.7

use anyhow::Result;
use pi_core::session::{Session, SessionManager};
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use std::path::PathBuf;
use std::time::Duration;

pub struct SessionPicker {
    event_loop: EventLoop,
    sessions_dir: PathBuf,
    sessions: Vec<String>,
    selected: usize,
    status: String,
}

impl SessionPicker {
    pub fn new(sessions_dir: PathBuf) -> Result<Self> {
        let event_loop = EventLoop::new()?;

        let mut picker = Self {
            event_loop,
            sessions_dir,
            sessions: Vec::new(),
            selected: 0,
            status: "Loading sessions...".to_string(),
        };

        picker.refresh_sessions()?;
        Ok(picker)
    }

    fn refresh_sessions(&mut self) -> Result<()> {
        let manager = SessionManager::new(self.sessions_dir.clone())?;
        let sessions = manager.list()?;

        self.sessions = sessions
            .iter()
            .map(|s| format!("{} ({})", s.created_at, s.model))
            .collect();

        if self.sessions.is_empty() {
            self.status = "No sessions found. Press 'n' to create a new session.".to_string();
        } else {
            self.status = format!("Found {} session(s). Press Enter to load.", self.sessions.len());
        }

        Ok(())
    }

    pub async fn run(&mut self) -> Result<Option<String>> {
        loop {
            // Draw UI
            self.draw_ui()?;

            // Handle events
            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(100)) {
                use pi_tui::tui::AppEvent;

                match event {
                    AppEvent::Key(cmd) => match cmd {
                        KeyCommand::CtrlC => return Ok(None), // Cancel
                        KeyCommand::Enter => {
                            // Load selected session
                            if !self.sessions.is_empty() {
                                return Ok(Some(self.sessions[self.selected].clone()));
                            }
                        }
                        KeyCommand::ArrowUp => {
                            if self.selected > 0 {
                                self.selected -= 1;
                            }
                        }
                        KeyCommand::ArrowDown => {
                            if self.selected < self.sessions.len().saturating_sub(1) {
                                self.selected += 1;
                            }
                        }
                        KeyCommand::Char('n') => {
                            // Create new session - return special marker
                            return Ok(Some("NEW".to_string()));
                        }
                        KeyCommand::Char('r') => {
                            // Refresh
                            self.refresh_sessions()?;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    fn draw_ui(&mut self) -> Result<()> {
        let sessions = self.sessions.clone();
        let selected = self.selected;
        let status = self.status.clone();

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();

            // Header
            let header_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: 3,
            };

            let title_widget = Paragraph::new("Session Picker")
                .block(Block::default().borders(Borders::ALL))
                .style(Style::default().bold());
            frame.render_widget(title_widget, header_area);

            // Sessions list
            let list_area = Rect {
                x: 0,
                y: 3,
                width: size.width,
                height: size.height.saturating_sub(4),
            };

            let items: Vec<ListItem> = sessions
                .iter()
                .enumerate()
                .map(|(idx, session)| {
                    let style = if idx == selected {
                        Style::default().fg(Color::Cyan).bold()
                    } else {
                        Style::default()
                    };

                    ListItem::new(format!(
                        "{} {}",
                        if idx == selected { ">" } else { " " },
                        session
                    ))
                    .style(style)
                })
                .collect();

            let list = if items.is_empty() {
                List::new(vec![ListItem::new("No sessions - press 'n' to create")])
            } else {
                List::new(items)
            };

            let list_widget = list.block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Sessions (↑↓ navigate, Enter to load, 'n' for new, 'r' to refresh)"),
            );

            frame.render_widget(list_widget, list_area);

            // Status
            let status_area = Rect {
                x: 0,
                y: size.height.saturating_sub(1),
                width: size.width,
                height: 1,
            };

            let status_widget = Paragraph::new(status.clone())
                .style(Style::default().fg(Color::Gray));
            frame.render_widget(status_widget, status_area);
        })?;

        Ok(())
    }
}
