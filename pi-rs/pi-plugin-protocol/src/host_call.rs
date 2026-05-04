//! Host-call requests: what a plugin asks the host to do.
//!
//! All host calls are dispatched through a single `pi_host_call`
//! import, with the request JSON encoded in a buffer the plugin
//! passed by `(ptr, len)`. The host replies by allocating a
//! guest-side buffer via the plugin's `pi_alloc` export and returning
//! the `(ptr, len)` pair in a multi-value tuple.
//!
//! Each variant corresponds to one user-visible host function
//! name (`pi_log`, `pi_subscribe`, `pi_register_tool`, ...). Keeping
//! them in one tagged enum means:
//! - the ABI is stable as we add APIs (the enum variant list grows,
//!   but the entry-point signature never changes);
//! - every SDK language does ser/de exactly once;
//! - host-side permission gating lives in a single `match` arm.

use serde::{Deserialize, Serialize};

use crate::event::EventKind;
use crate::tool::{ToolResult, ToolSchema};

/// A single request from plugin → host. Each variant maps 1:1 to a
/// user-facing `pi_*` function the plugin may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum HostRequest {
    /// Write a structured log line for this plugin. Always permitted;
    /// the host routes these to `~/.local/share/pi/logs/plugins/<name>.log`.
    Log {
        /// Log level: `"error" | "warn" | "info" | "debug" | "trace"`.
        level: String,
        /// Free-form message text.
        message: String,
    },

    /// Subscribe to an event kind. Subscriptions are additive with
    /// whatever the manifest declared. Idempotent.
    Subscribe {
        /// Event kind to subscribe to.
        event: EventKind,
    },

    /// Remove a previously-added subscription. No-op if not
    /// subscribed. Manifest-declared subscriptions cannot be removed.
    Unsubscribe {
        /// Event kind to remove.
        event: EventKind,
    },

    /// Register a tool the model can call. The plugin will receive
    /// `Event::ToolCall` for invocations and must reply via
    /// `EventReply::ToolResult`.
    RegisterTool {
        /// The tool schema the model will see.
        tool: ToolSchema,
    },

    /// Remove a previously-registered tool.
    UnregisterTool {
        /// Tool name to remove.
        name: String,
    },

    /// Send a `ToolResult` asynchronously (for tools whose execution
    /// spans multiple `pi_on_event` invocations — e.g. long-running
    /// bash commands). The plugin replies synchronously from
    /// `ToolCall` with an empty placeholder result and then streams
    /// updates via this call.
    ToolResult {
        /// Tool call id to attach the result to.
        tool_call_id: String,
        /// The result payload.
        result: ToolResult,
    },

    /// Show a user notification (toast) in the TUI.
    UiNotify {
        /// Notification severity: `"info" | "warn" | "error"`.
        level: String,
        /// The message to show.
        message: String,
    },

    /// Read the plugin's config object (the same value delivered in
    /// `Event::Load.config`). Useful after a hot-reload.
    GetConfig,

    /// Return the host's current working directory.
    GetCwd,

    /// Return the qualified ID (`provider/model`) of the active model,
    /// or `null` if no model is selected.
    GetActiveModel,
}

/// Reply from host → plugin for a single `HostRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "ok", rename_all = "snake_case")]
pub enum HostResponse {
    /// Request succeeded with no payload.
    Ok,
    /// Request succeeded and returned a JSON value (for getters).
    Value(serde_json::Value),
    /// Request failed; see `HostError` for the reason.
    Err(HostError),
}

/// Structured failure reason for a `HostRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostError {
    /// Short machine-readable code, e.g. `"permission_denied"`,
    /// `"unknown_tool"`, `"invalid_argument"`, `"faulted"`.
    pub code: String,
    /// Human-readable explanation shown in logs / UI.
    pub message: String,
}

impl HostError {
    /// Convenience constructor.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_request_roundtrip() {
        let cases = vec![
            HostRequest::Log {
                level: "info".into(),
                message: "hi".into(),
            },
            HostRequest::Subscribe {
                event: EventKind::AgentStart,
            },
            HostRequest::UnregisterTool {
                name: "my_tool".into(),
            },
            HostRequest::GetCwd,
        ];
        for req in cases {
            let s = serde_json::to_string(&req).unwrap();
            let back: HostRequest = serde_json::from_str(&s).unwrap();
            assert_eq!(req, back);
        }
    }

    #[test]
    fn host_response_ok_empty() {
        let r = HostResponse::Ok;
        assert_eq!(serde_json::to_string(&r).unwrap(), r#"{"ok":"ok"}"#);
    }

    #[test]
    fn host_response_err_shape() {
        let r = HostResponse::Err(HostError::new("permission_denied", "nope"));
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""ok":"err""#));
        assert!(s.contains(r#""code":"permission_denied""#));
        assert!(s.contains(r#""message":"nope""#));
    }

    #[test]
    fn host_request_wire_tag_is_op() {
        let r = HostRequest::GetCwd;
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, r#"{"op":"get_cwd"}"#);
    }
}
