//! pi-plugin-sdk: Rust SDK for authoring pi plugins.
//!
//! The SDK provides four things:
//!
//! 1. The [`Plugin`] trait plugin authors implement.
//! 2. The [`pi_plugin!`] macro that wires a `Plugin` impl to the
//!    required WASM exports (`pi_load`, `pi_on_event`, `pi_alloc`,
//!    `pi_dealloc`). Plugin crates write exactly one line to use it.
//! 3. The [`host`] module with type-safe wrappers over the
//!    `pi_host_call` ABI (e.g. `host::log_info`, `host::subscribe`).
//! 4. Re-exports of every protocol type so a plugin crate only
//!    needs one `use pi_plugin_sdk::*;`.
//!
//! Example plugin (see `examples/hello` for the real thing):
//!
//! ```no_run
//! use pi_plugin_sdk::*;
//!
//! #[derive(Default)]
//! struct Hello;
//!
//! impl Plugin for Hello {
//!     fn on_load(&mut self, _config: serde_json::Value, _cwd: String) {
//!         host::log_info("hello plugin loaded");
//!         host::subscribe(EventKind::AgentStart);
//!     }
//!
//!     fn on_event(&mut self, event: Event) -> EventReply {
//!         if matches!(event, Event::AgentStart) {
//!             host::ui_notify("info", "agent starting");
//!         }
//!         EventReply::None
//!     }
//! }
//!
//! pi_plugin!(Hello::default());
//! ```
//!
//! `Hello::default()` needs an explicit `Default` impl (not shown) or
//! any expression that produces an instance of your plugin type.

#![deny(missing_docs)]
// Our panic / alloc helpers intentionally use `extern "C"` and raw
// pointers to satisfy the WASM ABI the host requires.

pub mod host;
pub mod runtime;

pub use pi_plugin_protocol::{
    Event, EventKind, EventReply, FaultPolicy, HostError, HostRequest, HostResponse, Manifest,
    Permission, PermissionGrant, SessionStartReason, ToolResult, ToolResultKind, ToolSchema,
    UnloadReason, PROTOCOL_VERSION,
};

/// The trait every plugin implements. All methods have default no-op
/// implementations so plugins only override what they need.
pub trait Plugin {
    /// Called once after the WASM module is instantiated. The plugin
    /// should register tools and subscribe to events here.
    ///
    /// - `config`: per-plugin config loaded from
    ///   `~/.config/pi/plugins/<name>.toml` (or `{}` if none).
    /// - `cwd`: the host's working directory at load time.
    fn on_load(&mut self, config: serde_json::Value, cwd: String) {
        let _ = (config, cwd);
    }

    /// Handle an event dispatched by the host. Return
    /// `EventReply::None` if the event doesn't need a reply;
    /// `EventReply::ToolResult(..)` for `Event::ToolCall` events;
    /// `EventReply::Cancel { .. }` to abort a cancellable flow.
    ///
    /// The default dispatches to `on_event_default` which just
    /// returns `EventReply::None` — override this method or
    /// `on_event_default` as needed.
    fn on_event(&mut self, event: Event) -> EventReply {
        self.on_event_default(event)
    }

    /// Catch-all used by the default `on_event`. Override this if you
    /// want pattern-matching on your side without losing the default
    /// `on_event` dispatch machinery.
    fn on_event_default(&mut self, event: Event) -> EventReply {
        let _ = event;
        EventReply::None
    }
}

/// Generate the required WASM exports for a plugin type. Must be
/// called exactly once at the crate root of a plugin crate.
///
/// The macro takes an expression that evaluates to the initial plugin
/// instance. That instance lives forever in a module-local
/// `OnceLock` and all events are dispatched to it.
///
/// ```ignore
/// use pi_plugin_sdk::*;
///
/// #[derive(Default)]
/// struct MyPlugin;
///
/// impl Plugin for MyPlugin { /* ... */ }
///
/// pi_plugin!(MyPlugin::default());
/// ```
#[macro_export]
macro_rules! pi_plugin {
    ($init:expr) => {
        // Module-local global guarded by a mutex. WASM is single-
        // threaded under WASI so the lock is never contended, but the
        // mutex satisfies the `Sync` requirement for `static`s.
        static __PI_PLUGIN: ::std::sync::Mutex<Option<Box<dyn $crate::Plugin + Send>>> =
            ::std::sync::Mutex::new(None);

        fn __pi_get_plugin() -> &'static ::std::sync::Mutex<Option<Box<dyn $crate::Plugin + Send>>>
        {
            $crate::runtime::install_panic_hook();
            let mut guard = __PI_PLUGIN.lock().expect("plugin mutex poisoned");
            if guard.is_none() {
                *guard = Some(Box::new($init));
            }
            drop(guard);
            &__PI_PLUGIN
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn pi_alloc(n: u32) -> u32 {
            $crate::runtime::alloc(n)
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn pi_dealloc(ptr: u32, len: u32) {
            $crate::runtime::dealloc(ptr, len)
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn pi_load() {
            let lock = __pi_get_plugin();
            let mut guard = lock.lock().expect("plugin mutex poisoned");
            let plugin = guard
                .as_mut()
                .expect("plugin initialised in pi_alloc/on_load pair");
            // The host also synthesizes a `Load` event right after
            // `pi_load`, so we don't call `on_load` here — we do it
            // from the `pi_on_event` handler when we see
            // `Event::Load`.
            let _ = plugin;
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn pi_on_event(ptr: u32, len: u32) -> u64 {
            let bytes = $crate::runtime::take_buffer(ptr, len);
            let event: $crate::Event = match ::serde_json::from_slice(&bytes) {
                Ok(e) => e,
                Err(_) => return 0,
            };
            let lock = __pi_get_plugin();
            let mut guard = lock.lock().expect("plugin mutex poisoned");
            let plugin = guard
                .as_mut()
                .expect("plugin initialised in pi_alloc/on_load pair");
            // Intercept the synthetic Load event so plugins can
            // override `on_load` in the trait and don't have to
            // match on `Event::Load` themselves.
            if let $crate::Event::Load { ref config, ref cwd } = event {
                plugin.on_load(config.clone(), cwd.clone());
                return 0;
            }
            let reply = plugin.on_event(event);
            match reply {
                $crate::EventReply::None => 0,
                other => $crate::runtime::write_reply(&other),
            }
        }
    };
}
