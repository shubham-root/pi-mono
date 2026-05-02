//! pi-plugin-sdk: SDK for building pi-rs plugins.
//!
//! Provides the `#[pi_plugin]` attribute macro (via separate proc-macro crate),
//! plugin trait, and host function bindings for plugin authors.

pub mod plugin;

// Proc-macro will be defined in a separate crate
// #[macro_use] extern crate pi_plugin_sdk_derive;

pub use plugin::{Plugin, PluginEvent, ToolContext, ToolOutcome};
