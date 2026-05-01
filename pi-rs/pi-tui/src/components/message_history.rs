//! Message history component - displays conversation
//! Implements Phase 3.3 - scrollable message display

use super::Component;
use crate::input::KeyCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

pub struct MessageHistory {
    messages: Vec<Message>,
    scroll: usize,
}

pub struct Message {
    pub role: String,
    pub content: String,
}

impl MessageHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            scroll: 0,
        }
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.messages.push(Message {
            role: role.into(),
            content: content.into(),
        });
        // Auto-scroll to bottom
        self.scroll_to_bottom();
    }

    pub fn scroll_up(&mut self) {
        if self.scroll > 0 {
            self.scroll -= 1;
        }
    }

    pub fn scroll_down(&mut self) {
        self.scroll += 1;
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll = self.messages.len().saturating_sub(1);
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.scroll = 0;
    }
}

impl Default for MessageHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for MessageHistory {
    fn render(&self, frame: &mut Frame, area: Rect) {
        if area.height < 3 {
            return;
        }

        let content_area = Rect {
            x: area.x + 1,
            y: area.y + 1,
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(2),
        };

        let mut lines: Vec<Line> = Vec::new();

        for msg in &self.messages {
            // Add role line
            lines.push(Line::from(Span::styled(
                format!("{}: ", msg.role),
                Style::default().bold(),
            )));

            // Add content lines, wrapped
            for content_line in msg.content.lines() {
                lines.push(Line::from(Span::raw(format!("  {}", content_line))));
            }

            // Add spacing
            lines.push(Line::from(""));
        }

        let start = self.scroll.min(lines.len().saturating_sub(1));
        let visible_lines: Vec<Line> = lines
            .iter()
            .skip(start)
            .take(content_area.height as usize)
            .cloned()
            .collect();

        let paragraph = Paragraph::new(visible_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Conversation"),
            )
            .scroll((0, 0));

        frame.render_widget(paragraph, area);
    }

    fn handle_input(&mut self, cmd: &KeyCommand) -> bool {
        match cmd {
            KeyCommand::PageUp => {
                self.scroll_up();
                true
            }
            KeyCommand::PageDown => {
                self.scroll_down();
                true
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "MessageHistory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_history_new() {
        let history = MessageHistory::new();
        assert_eq!(history.messages.len(), 0);
    }

    #[test]
    fn test_message_history_add() {
        let mut history = MessageHistory::new();
        history.add_message("User", "Hello");
        assert_eq!(history.messages.len(), 1);
        assert_eq!(history.messages[0].role, "User");
        assert_eq!(history.messages[0].content, "Hello");
    }

    #[test]
    fn test_message_history_scroll() {
        let mut history = MessageHistory::new();
        for i in 0..10 {
            history.add_message("User", format!("Message {}", i));
        }
        history.scroll = 0;
        history.scroll_down();
        assert_eq!(history.scroll, 1);
    }
}
