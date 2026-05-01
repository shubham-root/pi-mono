//! pi-tui: Terminal User Interface components for pi-rs.
//!
//! Provides cross-platform terminal handling with raw mode,
//! input parsing, and a component-based rendering system.

pub mod terminal;
pub mod input;
pub mod components;
pub mod tui;
pub mod autocomplete;
pub mod syntax_highlight;

pub use terminal::Terminal;
pub use input::{InputParser, KeyCommand};
pub use components::{Editor, MessageHistory};
pub use tui::EventLoop;
