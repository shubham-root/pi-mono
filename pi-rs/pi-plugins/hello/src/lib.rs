//! Example plugin — greets the user at the start of every agent
//! loop via a TUI notification, and logs whenever it loads / unloads.
//!
//! Build with `cargo build -p pi-plugin-hello --target wasm32-wasip2
//! --release`, then copy the resulting
//! `target/wasm32-wasip2/release/pi_plugin_hello.wasm` next to this
//! crate's `plugin.toml` and point `pi` at it.

use pi_plugin_sdk::*;

#[derive(Default)]
pub struct Hello {
    greetings_shown: u32,
}

impl Plugin for Hello {
    fn on_load(&mut self, _config: serde_json::Value, cwd: String) {
        host::log_info(format!("hello loaded (cwd={cwd})"));
        host::subscribe(EventKind::AgentStart);
    }

    fn on_event(&mut self, event: Event) -> EventReply {
        match event {
            Event::AgentStart => {
                self.greetings_shown = self.greetings_shown.saturating_add(1);
                host::ui_notify(
                    "info",
                    format!(
                        "\u{1F44B} hello plugin says hi ({} so far)",
                        self.greetings_shown
                    ),
                );
            }
            Event::Unload { reason } => {
                host::log_info(format!("hello unloading: {reason:?}"));
            }
            _ => {}
        }
        EventReply::None
    }
}

pi_plugin!(Hello::default());
