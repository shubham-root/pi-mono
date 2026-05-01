//! Component model for building terminal UIs using Ratatui.
//! Implements Phase 3.3 with flexible component system.

use crate::input::KeyCommand;
use ratatui::prelude::*;

/// A component that can render itself and handle input.
pub trait Component {
    /// Render the component to a frame
    fn render(&self, frame: &mut Frame, area: Rect);

    /// Handle a key command
    fn handle_input(&mut self, _cmd: &KeyCommand) -> bool {
        false // return true if handled
    }

    /// Get the component's name for debugging
    fn name(&self) -> &str {
        "Component"
    }
}

pub mod text;
pub mod box_border;
pub mod editor;
pub mod select_list;
pub mod message_history;

pub use text::Text;
pub use box_border::BoxBorder;
pub use editor::Editor;
pub use select_list::SelectList;
pub use message_history::MessageHistory;
