//! Select list component - selectable item list
//! Implements Phase 3.3

use super::Component;
use crate::input::KeyCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

pub struct SelectList {
    title: String,
    items: Vec<String>,
    selected: usize,
}

impl SelectList {
    pub fn new(title: impl Into<String>, items: Vec<String>) -> Self {
        Self {
            title: title.into(),
            items,
            selected: 0,
        }
    }

    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn move_down(&mut self) {
        if self.selected < self.items.len().saturating_sub(1) {
            self.selected += 1;
        }
    }

    pub fn selected_item(&self) -> Option<&str> {
        self.items.get(self.selected).map(|s| s.as_str())
    }

    pub fn selected_index(&self) -> usize {
        self.selected
    }
}

impl Component for SelectList {
    fn render(&self, frame: &mut Frame, area: Rect) {
        let lines: Vec<Line> = self
            .items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                if idx == self.selected {
                    Line::from(Span::styled(
                        format!("> {}", item),
                        Style::default().fg(Color::Cyan).bold(),
                    ))
                } else {
                    Line::from(Span::raw(format!("  {}", item)))
                }
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .title(self.title.as_str());

        let paragraph = Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, area);
    }

    fn handle_input(&mut self, cmd: &KeyCommand) -> bool {
        match cmd {
            KeyCommand::ArrowUp => {
                self.move_up();
                true
            }
            KeyCommand::ArrowDown => {
                self.move_down();
                true
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "SelectList"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_list_new() {
        let list = SelectList::new("Models", vec!["gpt-4".to_string(), "claude-3".to_string()]);
        assert_eq!(list.selected, 0);
        assert_eq!(list.selected_item(), Some("gpt-4"));
    }

    #[test]
    fn test_select_list_navigation() {
        let mut list = SelectList::new("Models", vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        list.move_down();
        assert_eq!(list.selected, 1);
        list.move_down();
        assert_eq!(list.selected, 2);
        list.move_down();
        assert_eq!(list.selected, 2); // Stays at last
    }
}
