//! Markdown rendering component (simplified)
//! Implements Phase 3.3

use super::Component;
use crate::input::KeyCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Wrap};

pub struct Markdown {
    content: String,
}

impl Markdown {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    fn parse_lines(&self) -> Vec<Line> {
        let mut lines = Vec::new();

        for line in self.content.lines() {
            // Simple markdown parsing
            if line.starts_with("# ") {
                lines.push(Line::from(Span::styled(
                    &line[2..],
                    Style::default().fg(Color::Yellow).bold(),
                )));
            } else if line.starts_with("## ") {
                lines.push(Line::from(Span::styled(
                    &line[3..],
                    Style::default().fg(Color::Cyan).bold(),
                )));
            } else if line.starts_with("- ") {
                lines.push(Line::from(Span::raw(format!("  • {}", &line[2..]))));
            } else if line.starts_with("```") {
                // Code block marker - just style it differently
                lines.push(Line::from(Span::styled(
                    line,
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                lines.push(Line::from(Span::raw(line)));
            }
        }

        lines
    }
}

impl Component for Markdown {
    fn render(&self, frame: &mut Frame, area: Rect) {
        let lines = self.parse_lines();
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
    }

    fn name(&self) -> &str {
        "Markdown"
    }
}
