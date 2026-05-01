//! Text rendering component
//! Implements Phase 3.3 - static text display

use super::Component;
use crate::input::KeyCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Wrap};

pub struct Text {
    content: String,
    style: Style,
}

impl Text {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            style: Style::default(),
        }
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

impl Component for Text {
    fn render(&self, frame: &mut Frame, area: Rect) {
        let lines: Vec<&str> = self.content.lines().collect();
        let text = lines
            .iter()
            .map(|&line| Line::from(Span::styled(line, self.style)))
            .collect::<Vec<_>>();

        let paragraph = Paragraph::new(text).wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
    }

    fn name(&self) -> &str {
        "Text"
    }
}
