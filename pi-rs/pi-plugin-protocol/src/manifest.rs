//! Plugin manifest — the `plugin.toml` file that ships alongside the
//! `.wasm` binary.
//!
//! Parsing lives on the host side (see
//! `pi-plugin-host/src/manifest.rs`). This crate only defines the
//! in-memory shape so SDK language bindings can construct manifests
//! programmatically if they wish.

use serde::{Deserialize, Serialize};

use crate::event::EventKind;
use crate::permission::Permission;

/// Contents of `plugin.toml`.
///
/// Example:
///
/// ```toml
/// name = "hello"
/// version = "0.1.0"
/// description = "Logs a greeting when any agent turn starts."
/// entry = "hello.wasm"
///
/// permissions = ["ui"]
/// events = ["agent_start"]
///
/// [limits]
/// memory_bytes = 67108864      # 64MB
/// fuel = 1_000_000_000         # 1B instructions per call
/// call_timeout_ms = 30000
///
/// [fault]
/// policy = "disable"
/// max_faults = 3
/// fault_window_secs = 60
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Plugin name (unique within a pi installation).
    pub name: String,
    /// Semver version string.
    pub version: String,
    /// One-line description surfaced in `/plugin list`.
    #[serde(default)]
    pub description: String,
    /// Relative path to the `.wasm` file (resolved against the
    /// manifest's parent directory).
    pub entry: String,

    /// Permissions the plugin declares it will request.
    /// The host still prompts the user on first use; this list
    /// drives the prompt UI and the pre-flight rejection of
    /// host calls that aren't covered.
    #[serde(default)]
    pub permissions: Vec<Permission>,

    /// Events the plugin wants delivered without an explicit
    /// `pi_subscribe` call. The plugin can add more at runtime
    /// via `pi_subscribe`.
    #[serde(default)]
    pub events: Vec<EventKind>,

    /// Hard resource limits enforced by wasmtime.
    #[serde(default)]
    pub limits: Limits,

    /// Fault policy controlling what happens when the plugin traps.
    #[serde(default)]
    pub fault: FaultConfig,
}

/// Per-plugin resource ceilings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Limits {
    /// Maximum linear memory size in bytes.
    pub memory_bytes: u64,
    /// Fuel budget per guest call (`pi_on_event`, `pi_load`, ...).
    /// One fuel unit ≈ one WASM instruction. `0` disables metering.
    pub fuel: u64,
    /// Wall-clock timeout per guest call.
    pub call_timeout_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            memory_bytes: 64 * 1024 * 1024,
            fuel: 1_000_000_000,
            call_timeout_ms: 30_000,
        }
    }
}

/// How the host reacts when a plugin call traps (out-of-fuel,
/// out-of-memory, panic, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultConfig {
    /// What to do on trap.
    pub policy: FaultPolicy,
    /// Max number of faults within `fault_window_secs` before the
    /// plugin is forcibly disabled regardless of `policy`.
    pub max_faults: u32,
    /// Rolling window (seconds) for the `max_faults` counter.
    pub fault_window_secs: u64,
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            policy: FaultPolicy::Disable,
            max_faults: 3,
            fault_window_secs: 60,
        }
    }
}

/// Per-plugin fault reaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultPolicy {
    /// Auto-reload the `.wasm` file after a backoff (1s, 5s, 30s,
    /// give up). Useful for plugins whose work is best-effort (like a
    /// status indicator).
    Restart,
    /// Mark the plugin disabled. User must `/plugin reload <name>`
    /// to bring it back. Default — safest for tool-registering
    /// plugins.
    Disable,
    /// Log the fault and continue. For non-critical plugins whose
    /// failure shouldn't be visible.
    Ignore,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_are_sane() {
        let l = Limits::default();
        assert_eq!(l.memory_bytes, 64 * 1024 * 1024);
        assert_eq!(l.fuel, 1_000_000_000);
        assert_eq!(l.call_timeout_ms, 30_000);
    }

    #[test]
    fn manifest_minimal_roundtrip_via_json() {
        let m = Manifest {
            name: "hello".into(),
            version: "0.1.0".into(),
            description: "".into(),
            entry: "hello.wasm".into(),
            permissions: vec![Permission::Ui],
            events: vec![EventKind::AgentStart],
            limits: Limits::default(),
            fault: FaultConfig::default(),
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }
}
