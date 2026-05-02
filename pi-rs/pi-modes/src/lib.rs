//! pi-modes: Different execution modes for pi-rs.
//!
//! - **Print**: Single-turn non-interactive mode
//! - **Interactive**: Full TUI
//! - **RPC**: JSON-RPC over stdin/stdout

pub mod print;
pub mod interactive;
pub mod rpc;
pub mod session_picker;
pub mod settings_ui;
pub mod markdown;
pub mod editor;
pub mod auth;

pub use print::PrintMode;
pub use interactive::InteractiveMode;
pub use rpc::RpcMode;
pub use session_picker::SessionPicker;
pub use settings_ui::SettingsUI;
