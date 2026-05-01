//! Settings UI - edit configuration in TUI
//! Implements Phase 3.8

use anyhow::Result;
use pi_core::settings::Settings;
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
enum SettingField {
    Model,
    Thinking,
    Temperature,
    MaxTokens,
}

pub struct SettingsUI {
    event_loop: EventLoop,
    settings: Settings,
    selected_field: SettingField,
    model_options: Vec<String>,
    thinking_options: Vec<String>,
    status: String,
    modified: bool,
}

impl SettingsUI {
    pub fn new(settings_dir: &Path) -> Result<Self> {
        let event_loop = EventLoop::new()?;
        let settings = Settings::load_merged(settings_dir)?;

        let model_options = vec![
            "claude-3-sonnet".to_string(),
            "claude-3-haiku".to_string(),
            "gpt-4o".to_string(),
            "gpt-4-turbo".to_string(),
            "gpt-3.5-turbo".to_string(),
        ];

        let thinking_options = vec![
            "minimal".to_string(),
            "low".to_string(),
            "medium".to_string(),
            "high".to_string(),
            "xhigh".to_string(),
        ];

        Ok(Self {
            event_loop,
            settings,
            selected_field: SettingField::Model,
            model_options,
            thinking_options,
            status: "Edit settings - Arrow keys to navigate, Enter to edit, Ctrl+S to save, Ctrl+C to cancel"
                .to_string(),
            modified: false,
        })
    }

    pub async fn run(&mut self) -> Result<bool> {
        loop {
            self.draw_ui()?;

            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(100)) {
                use pi_tui::tui::AppEvent;

                match event {
                    AppEvent::Key(cmd) => match cmd {
                        KeyCommand::CtrlC => return Ok(false), // Cancel
                        KeyCommand::CtrlS => {
                            // Save settings
                            self.settings.save_global()?;
                            self.status = "Settings saved!".to_string();
                            self.modified = false;
                            return Ok(true);
                        }
                        KeyCommand::ArrowUp => self.prev_field(),
                        KeyCommand::ArrowDown => self.next_field(),
                        KeyCommand::ArrowLeft => self.prev_option(),
                        KeyCommand::ArrowRight => self.next_option(),
                        KeyCommand::Enter => {
                            // Cycle through options for selected field
                            self.next_option();
                            self.modified = true;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    fn prev_field(&mut self) {
        self.selected_field = match self.selected_field {
            SettingField::Model => SettingField::MaxTokens,
            SettingField::Thinking => SettingField::Model,
            SettingField::Temperature => SettingField::Thinking,
            SettingField::MaxTokens => SettingField::Temperature,
        };
    }

    fn next_field(&mut self) {
        self.selected_field = match self.selected_field {
            SettingField::Model => SettingField::Thinking,
            SettingField::Thinking => SettingField::Temperature,
            SettingField::Temperature => SettingField::MaxTokens,
            SettingField::MaxTokens => SettingField::Model,
        };
    }

    fn prev_option(&mut self) {
        match self.selected_field {
            SettingField::Model => {
                if let Some(current) = &self.settings.model {
                    if let Some(idx) = self.model_options.iter().position(|m| m == current) {
                        if idx > 0 {
                            self.settings.model = Some(self.model_options[idx - 1].clone());
                        }
                    }
                }
            }
            SettingField::Thinking => {
                if let Some(current) = &self.settings.thinking {
                    if let Some(idx) = self.thinking_options.iter().position(|t| t == current) {
                        if idx > 0 {
                            self.settings.thinking = Some(self.thinking_options[idx - 1].clone());
                        }
                    }
                }
            }
            SettingField::Temperature => {
                if let Some(temp) = self.settings.temperature {
                    if temp > 0.1 {
                        self.settings.temperature = Some((temp - 0.1).max(0.0));
                    }
                }
            }
            SettingField::MaxTokens => {
                if let Some(tokens) = self.settings.max_tokens {
                    if tokens > 100 {
                        self.settings.max_tokens = Some(tokens - 100);
                    }
                }
            }
        }
        self.modified = true;
    }

    fn next_option(&mut self) {
        match self.selected_field {
            SettingField::Model => {
                if let Some(current) = &self.settings.model {
                    if let Some(idx) = self.model_options.iter().position(|m| m == current) {
                        if idx < self.model_options.len() - 1 {
                            self.settings.model = Some(self.model_options[idx + 1].clone());
                        }
                    }
                }
            }
            SettingField::Thinking => {
                if let Some(current) = &self.settings.thinking {
                    if let Some(idx) = self.thinking_options.iter().position(|t| t == current) {
                        if idx < self.thinking_options.len() - 1 {
                            self.settings.thinking = Some(self.thinking_options[idx + 1].clone());
                        }
                    }
                }
            }
            SettingField::Temperature => {
                if let Some(temp) = self.settings.temperature {
                    if temp < 2.0 {
                        self.settings.temperature = Some((temp + 0.1).min(2.0));
                    }
                }
            }
            SettingField::MaxTokens => {
                if let Some(tokens) = self.settings.max_tokens {
                    if tokens < 200000 {
                        self.settings.max_tokens = Some(tokens + 100);
                    }
                }
            }
        }
        self.modified = true;
    }

    fn draw_ui(&mut self) -> Result<()> {
        let settings = self.settings.clone();
        let selected_field = self.selected_field;
        let status = self.status.clone();
        let modified = self.modified;

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();

            // Title
            let title_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: 2,
            };

            let title = if modified {
                "Settings (MODIFIED)"
            } else {
                "Settings"
            };

            let title_widget = Paragraph::new(title)
                .block(Block::default().borders(Borders::ALL))
                .style(if modified {
                    Style::default().fg(Color::Yellow).bold()
                } else {
                    Style::default().bold()
                });
            frame.render_widget(title_widget, title_area);

            // Settings fields
            let fields_area = Rect {
                x: 0,
                y: 2,
                width: size.width,
                height: size.height.saturating_sub(3),
            };

            let items = vec![
                ListItem::new(format!(
                    "Model: {}",
                    settings.model.as_ref().unwrap_or(&"none".to_string())
                )),
                ListItem::new(format!(
                    "Thinking: {}",
                    settings.thinking.as_ref().unwrap_or(&"low".to_string())
                )),
                ListItem::new(format!(
                    "Temperature: {}",
                    settings.temperature.map(|t| format!("{:.1}", t)).unwrap_or_else(|| "0.7".to_string())
                )),
                ListItem::new(format!(
                    "Max Tokens: {}",
                    settings.max_tokens.unwrap_or(4096)
                )),
            ];

            let list = List::new(items)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Settings (← → to adjust, ↑ ↓ to move, Ctrl+S to save)"),
                )
                .style(match selected_field {
                    SettingField::Model => Style::default(),
                    _ => Style::default(),
                });

            frame.render_widget(list, fields_area);

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
