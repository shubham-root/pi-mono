//! Permissions declared by a plugin in its `plugin.toml` and granted
//! or denied by the user. Every side-effecting host call is gated
//! behind exactly one permission.

use serde::{Deserialize, Serialize};

/// One permission a plugin may declare / request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Register tools the LLM can call. Tools are still subject to
    /// the user's per-tool confirmation flow.
    RegisterTools,
    /// Show notifications / status / widgets in the TUI.
    Ui,
    /// Read files under the current working directory.
    ReadFiles,
    /// Write files under the current working directory.
    WriteFiles,
    /// Spawn processes. Future work.
    RunCommand,
    /// Make outbound HTTP requests. Future work.
    HttpRequest,
    /// Call the LLM via the host's provider registry. Future work.
    LlmCall,
    /// Pipe messages to other plugins. Future work.
    InterPlugin,
    /// Access other sessions via the Phase 5 cross-session bus.
    CrossSession,
}

impl Permission {
    /// Stable string id used in `plugin.toml` and on-disk grant files.
    /// Matches the `#[serde(rename_all = "snake_case")]` output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RegisterTools => "register_tools",
            Self::Ui => "ui",
            Self::ReadFiles => "read_files",
            Self::WriteFiles => "write_files",
            Self::RunCommand => "run_command",
            Self::HttpRequest => "http_request",
            Self::LlmCall => "llm_call",
            Self::InterPlugin => "inter_plugin",
            Self::CrossSession => "cross_session",
        }
    }
}

/// One persisted grant decision. The host stores the full set of
/// grants per-plugin in `~/.config/pi/plugin-permissions.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionGrant {
    /// Permission this grant covers.
    pub permission: Permission,
    /// Whether the user granted or denied.
    pub granted: bool,
    /// Unix timestamp (seconds) at which the decision was recorded.
    pub decided_at: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_wire_names_are_snake_case() {
        assert_eq!(Permission::RegisterTools.as_str(), "register_tools");
        assert_eq!(Permission::ReadFiles.as_str(), "read_files");
        // serde's rename_all matches as_str exactly
        let s = serde_json::to_string(&Permission::HttpRequest).unwrap();
        assert_eq!(s, "\"http_request\"");
    }

    #[test]
    fn grant_roundtrip() {
        let g = PermissionGrant {
            permission: Permission::Ui,
            granted: true,
            decided_at: 42,
        };
        let s = serde_json::to_string(&g).unwrap();
        let back: PermissionGrant = serde_json::from_str(&s).unwrap();
        assert_eq!(g, back);
    }
}
