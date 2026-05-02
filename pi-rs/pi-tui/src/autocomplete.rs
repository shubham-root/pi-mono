//! Autocomplete component - fuzzy completion dropdown
//! To be implemented in Phase 3.10

use std::sync::Arc;

pub struct Autocomplete {
    provider: Box<dyn CompletionProvider>,
    visible: bool,
    selected: usize,
    items: Vec<String>,
}

pub trait CompletionProvider: Send + Sync {
    fn completions(&self, prefix: &str) -> Vec<String>;
    fn accept(&self, item: &str);
}

impl Autocomplete {
    pub fn new(provider: Box<dyn CompletionProvider>) -> Self {
        Self {
            provider,
            visible: false,
            selected: 0,
            items: Vec::new(),
        }
    }

    pub fn show(&mut self, prefix: &str) {
        self.items = self.provider.completions(prefix);
        self.visible = !self.items.is_empty();
        self.selected = 0;
    }

    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn next(&mut self) {
        if self.visible && self.selected < self.items.len() - 1 {
            self.selected += 1;
        }
    }

    pub fn previous(&mut self) {
        if self.visible && self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn selected_item(&self) -> Option<&str> {
        self.items.get(self.selected).map(|s| s.as_str())
    }
}
