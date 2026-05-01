//! Box border component - bordered container
//! Implements Phase 3.3

use super::Component;
use crate::input::KeyCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders};

pub struct BoxBorder {
    title: String,
    inner: Option<Box<dyn Component>>,
}

impl BoxBorder {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            inner: None,
        }
    }

    pub fn with_child(mut self, child: Box<dyn Component>) -> Self {
        self.inner = Some(child);
        self
    }
}

impl Component for BoxBorder {
    fn render(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(self.title.as_str());

        frame.render_widget(block, area);

        if let Some(inner) = &self.inner {
            let inner_area = Rect {
                x: area.x + 1,
                y: area.y + 1,
                width: area.width.saturating_sub(2),
                height: area.height.saturating_sub(2),
            };
            inner.render(frame, inner_area);
        }
    }

    fn name(&self) -> &str {
        "BoxBorder"
    }
}
