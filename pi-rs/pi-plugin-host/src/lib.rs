//! pi-plugin-host: WASM plugin host runtime for pi-rs.
//!
//! Loads and executes WASM plugins with sandboxing, permission checks,
//! and inter-plugin communication.

pub mod host;
pub mod abi;
pub mod isolation;
pub mod permissions;
pub mod manifest;

pub use host::PluginHost;
pub use abi::*;
pub use isolation::*;
pub use permissions::*;
pub use manifest::*;
