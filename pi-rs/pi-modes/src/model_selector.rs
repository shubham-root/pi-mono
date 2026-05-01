//! Model selector - browse and choose models
//! Implements Phase 3.9

use anyhow::Result;
use crate::models;
use pi_tui::{input::KeyCommand, EventLoop};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
enum FilterMode {
    All,
    ByProvider,
}

pub struct ModelSelector {
    event_loop: EventLoop,
    filtered_models: Vec<String>,
    selected: usize,
    filter_mode: FilterMode,
    selected_provider: Option<String>,
    status: String,
}

impl ModelSelector {
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new()?;

        // Load models from generated registry
        let all_models = models::get_all_models();
        let filtered_models: Vec<String> = all_models
            .iter()
            .map(|(id, name, provider)| format!("{} ({})", name, provider))
            .collect();

        Ok(Self {
            event_loop,
            filtered_models,
            selected: 0,
            filter_mode: FilterMode::All,
            selected_provider: None,
            status: format!("Select a model ({} available) - ↑↓ to navigate, Enter to select, 'f' to filter by provider, Ctrl+C to cancel", all_models.len()),
        })
    }

    pub async fn run(&mut self) -> Result<Option<String>> {
        loop {
            self.draw_ui()?;

            if let Some(event) = self.event_loop.poll_event(Duration::from_millis(100)) {
                use pi_tui::tui::AppEvent;

                match event {
                    AppEvent::Key(cmd) => match cmd {
                        KeyCommand::CtrlC => return Ok(None), // Cancel
                        KeyCommand::Enter => {
                            // Select current model
                            if !self.filtered_models.is_empty() {
                                return Ok(Some(self.filtered_models[self.selected].clone()));
                            }
                        }
                        KeyCommand::ArrowUp => {
                            if self.selected > 0 {
                                self.selected -= 1;
                            }
                        }
                        KeyCommand::ArrowDown => {
                            if self.selected < self.filtered_models.len().saturating_sub(1) {
                                self.selected += 1;
                            }
                        }
                        KeyCommand::Char('f') => {
                            // Cycle through filter modes
                            self.filter_mode = match self.filter_mode {
                                FilterMode::All => FilterMode::ByProvider,
                                FilterMode::ByProvider => FilterMode::All,
                            };
                            self.apply_filter();
                        }
                        KeyCommand::Char('p') if self.filter_mode == FilterMode::ByProvider => {
                            // Cycle provider
                            let providers = models::get_providers();
                            if let Some(current) = &self.selected_provider {
                                if let Some(idx) = providers.iter().position(|p| p == current) {
                                    if idx < providers.len() - 1 {
                                        self.selected_provider = Some(providers[idx + 1].to_string());
                                    } else {
                                        self.selected_provider = Some(providers[0].to_string());
                                    }
                                }
                            } else if !providers.is_empty() {
                                self.selected_provider = Some(providers[0].to_string());
                            }
                            self.apply_filter();
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    fn apply_filter(&mut self) {
        let all_models = models::get_all_models();
        self.selected = 0;

        match self.filter_mode {
            FilterMode::All => {
                self.filtered_models = all_models
                    .iter()
                    .map(|(id, name, provider)| format!("{} ({})", name, provider))
                    .collect();
            }
            FilterMode::ByProvider => {
                let providers = models::get_providers();
                if self.selected_provider.is_none() && !providers.is_empty() {
                    self.selected_provider = Some(providers[0].to_string());
                }

                if let Some(ref provider) = self.selected_provider {
                    let provider_models = models::get_models_for_provider(provider);
                    self.filtered_models = provider_models
                        .iter()
                        .map(|(id, name)| format!("{} ({})", name, provider))
                        .collect();
                }
            }
        }
    }

    fn draw_ui(&mut self) -> Result<()> {
        let models = self.filtered_models.clone();
        let selected = self.selected;
        let status = self.status.clone();
        let filter_mode = self.filter_mode;
        let provider = self.selected_provider.clone();

        self.event_loop.terminal().draw(|frame| {
            let size = frame.size();

            // Header with filter info
            let header_area = Rect {
                x: 0,
                y: 0,
                width: size.width,
                height: 3,
            };

            let filter_text = match filter_mode {
                FilterMode::All => format!("Showing: All {} Models", models.len()),
                FilterMode::ByProvider => {
                    if let Some(p) = &provider {
                        format!("Showing: {} ({} models)", p, models.len())
                    } else {
                        "Showing: Select Provider".to_string()
                    }
                }
            };

            let header_widget = Paragraph::new(filter_text.clone())
                .block(Block::default().borders(Borders::ALL).title("Model Selector"))
                .style(Style::default().bold());
            frame.render_widget(header_widget, header_area);

            // Models list
            let list_area = Rect {
                x: 0,
                y: 3,
                width: size.width,
                height: size.height.saturating_sub(4),
            };

            let items: Vec<ListItem> = models
                .iter()
                .enumerate()
                .map(|(idx, model)| {
                    let style = if idx == selected {
                        Style::default().fg(Color::Cyan).bold()
                    } else {
                        Style::default()
                    };

                    ListItem::new(format!(
                        "{} {}",
                        if idx == selected { ">" } else { " " },
                        model
                    ))
                    .style(style)
                })
                .collect();

            let list = if items.is_empty() {
                List::new(vec![ListItem::new("No models found")])
            } else {
                List::new(items)
            };

            let title = match filter_mode {
                FilterMode::All => "Models (↑↓ navigate, Enter to select, 'f' to filter)",
                FilterMode::ByProvider => "Models (↑↓ navigate, Enter to select, 'p' to change provider, 'f' to unfilter)",
            };

            let list_widget = list.block(Block::default().borders(Borders::ALL).title(title));

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
