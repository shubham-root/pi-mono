//! pi-plugin-host: WASM plugin host runtime.
//!
//! The host loads `.wasm` plugins from `~/.config/pi/plugins/` or a
//! project-local `.pi/plugins/` directory, dispatches events, and
//! exposes a JSON-over-linear-memory ABI for plugins to call back
//! into pi.
//!
//! Current status: Stage 1 (protocol + manifest + permission store)
//! is wired up; the wasmtime runtime layer is in `engine` and grows
//! in follow-up stages.

#![deny(missing_docs)]

pub mod engine;
pub mod isolation;
pub mod manifest;
pub mod permissions;

pub use engine::{HostServices, Plugin, PluginHost, PluginId, PluginState};
pub use isolation::{Fault, FaultLedger, OwnershipRegistry};
pub use manifest::{load_manifest, resolve_wasm_path, ManifestError};
pub use permissions::{PermissionStore, PermissionStoreError};

// Re-export the protocol types so callers only need `pi-plugin-host`
// in their `use` statements.
pub use pi_plugin_protocol::{
    Event, EventKind, EventReply, FaultPolicy, HostError, HostRequest, HostResponse, Manifest,
    Permission, PermissionGrant, ToolResult, ToolResultKind, ToolSchema,
};
