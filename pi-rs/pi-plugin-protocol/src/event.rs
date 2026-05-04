//! Event types the host dispatches to subscribed plugins.
//!
//! Mirrors (a subset of) the TS `ExtensionEvent` union — see
//! `packages/coding-agent/src/core/extensions/types.ts`. We start
//! with the events that are useful for the first round of example
//! plugins (hello, pirate, git-checkpoint) and extend as needed.
//! Every variant is tagged via serde's `type`-tagged enum
//! representation so the wire format is self-describing.

use serde::{Deserialize, Serialize};

/// One event sent from host to plugin. Plugins subscribe to events
/// by `EventKind` in their manifest or via `pi_subscribe`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Fired once when the plugin is loaded. The plugin should do
    /// startup work here (read config, set up caches, register tools).
    Load {
        /// User-scoped config for this plugin (from `~/.config/pi/plugins/<name>.toml`).
        /// Empty object if no config file exists.
        config: serde_json::Value,
        /// Current working directory of the host.
        cwd: String,
    },

    /// Fired once when the plugin is about to be unloaded. The
    /// plugin should release its own resources. The host also
    /// automatically releases any tool registrations / event
    /// subscriptions / widgets the plugin owns.
    Unload {
        /// Why the plugin is being unloaded.
        reason: UnloadReason,
    },

    /// Fired when a new session becomes active.
    SessionStart {
        /// Absolute path to the session JSONL file.
        session_path: String,
        /// Why the session became active.
        reason: SessionStartReason,
    },

    /// Fired before the agent loop begins processing a user prompt.
    BeforeAgentStart {
        /// Raw user prompt text.
        prompt: String,
    },

    /// Fired when an agent loop starts. No payload — use
    /// `BeforeAgentStart` if you need the prompt text.
    AgentStart,

    /// Fired when an agent loop ends (all turns in the loop have
    /// finished executing).
    AgentEnd,

    /// Fired at the start of each individual turn within an
    /// agent loop.
    TurnStart {
        /// 0-based index of the turn within the current agent loop.
        turn_index: u32,
    },

    /// Fired at the end of each individual turn within an agent loop.
    TurnEnd {
        /// 0-based index of the turn within the current agent loop.
        turn_index: u32,
    },

    /// Fired when the model emits a tool call and the host is about
    /// to dispatch it. Plugins that own the tool receive this as a
    /// request and reply with a `ToolResult` via `EventReply`.
    ToolCall {
        /// Unique ID assigned by the model for this call.
        tool_call_id: String,
        /// Tool name (must match a registered tool).
        tool_name: String,
        /// Tool arguments, as a JSON object matching the tool's schema.
        input: serde_json::Value,
    },

    /// Fired after a tool call completes (built-in or plugin-owned).
    /// Read-only notification — reply is ignored.
    ToolResult {
        /// Unique ID assigned by the model for this call.
        tool_call_id: String,
        /// Tool name.
        tool_name: String,
        /// Whether the tool returned an error.
        is_error: bool,
    },

    /// Fired after the active model is switched (user, plugin, or
    /// slash-command driven).
    ModelSelect {
        /// Qualified provider/model id (e.g. `anthropic/claude-sonnet-4-5`).
        model_id: String,
    },

    /// Fired when the user submits a prompt via the TUI. Plugins
    /// receive this before the agent begins processing.
    UserInput {
        /// The raw prompt text typed by the user.
        text: String,
    },
}

/// Why a plugin was unloaded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnloadReason {
    /// pi is quitting.
    Quit,
    /// User ran `/plugin reload <name>`.
    Reload,
    /// User ran `/plugin unload <name>`.
    UserRequest,
    /// Plugin exceeded its fault budget and was force-disabled.
    Faulted,
}

/// Why a new session became active.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStartReason {
    /// Initial launch.
    Startup,
    /// `/new` created a fresh session.
    New,
    /// `/resume` loaded an existing session.
    Resume,
    /// `/fork` branched from an existing session.
    Fork,
}

/// Machine-readable discriminator for `Event` — lets the manifest
/// and `pi_subscribe` identify events by name without carrying a
/// payload. The names match `Event`'s `#[serde(rename_all)]` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum EventKind {
    Load,
    Unload,
    SessionStart,
    BeforeAgentStart,
    AgentStart,
    AgentEnd,
    TurnStart,
    TurnEnd,
    ToolCall,
    ToolResult,
    ModelSelect,
    UserInput,
}

impl EventKind {
    /// Return the `EventKind` matching the given `Event`.
    pub fn of(event: &Event) -> Self {
        match event {
            Event::Load { .. } => Self::Load,
            Event::Unload { .. } => Self::Unload,
            Event::SessionStart { .. } => Self::SessionStart,
            Event::BeforeAgentStart { .. } => Self::BeforeAgentStart,
            Event::AgentStart => Self::AgentStart,
            Event::AgentEnd => Self::AgentEnd,
            Event::TurnStart { .. } => Self::TurnStart,
            Event::TurnEnd { .. } => Self::TurnEnd,
            Event::ToolCall { .. } => Self::ToolCall,
            Event::ToolResult { .. } => Self::ToolResult,
            Event::ModelSelect { .. } => Self::ModelSelect,
            Event::UserInput { .. } => Self::UserInput,
        }
    }
}

/// Reply from a plugin's `pi_on_event` export. Most events don't
/// need a reply (the plugin returns `EventReply::None`); events that
/// do (like `ToolCall`) carry their result in the `data` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventReply {
    /// Plugin processed the event; nothing to return.
    None,
    /// Plugin handled a `ToolCall` event and produced a result.
    ToolResult(crate::tool::ToolResult),
    /// Plugin transformed the event's data (for events whose contract
    /// allows it, e.g. a future `Context` event). The shape is
    /// event-specific and documented per event.
    Transform(serde_json::Value),
    /// Plugin wants the default dispatch flow cancelled. The optional
    /// `reason` is shown to the user.
    Cancel {
        /// Optional human-readable reason shown in the UI.
        reason: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_event_roundtrip() {
        let cases = vec![
            Event::AgentStart,
            Event::TurnStart { turn_index: 0 },
            Event::UserInput {
                text: "hello".into(),
            },
            Event::ToolCall {
                tool_call_id: "abc".into(),
                tool_name: "read".into(),
                input: serde_json::json!({"path": "/tmp/x"}),
            },
            Event::SessionStart {
                session_path: "/home/me/.pi/sessions/x.jsonl".into(),
                reason: SessionStartReason::Resume,
            },
        ];
        for ev in cases {
            let s = serde_json::to_string(&ev).unwrap();
            let back: Event = serde_json::from_str(&s).unwrap();
            assert_eq!(ev, back, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn event_kind_tracks_event() {
        assert_eq!(EventKind::of(&Event::AgentStart), EventKind::AgentStart);
        assert_eq!(
            EventKind::of(&Event::TurnStart { turn_index: 0 }),
            EventKind::TurnStart
        );
    }

    #[test]
    fn event_kind_wire_name_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&EventKind::ToolCall).unwrap(),
            "\"tool_call\""
        );
        assert_eq!(
            serde_json::to_string(&EventKind::BeforeAgentStart).unwrap(),
            "\"before_agent_start\""
        );
    }

    #[test]
    fn event_reply_tagged() {
        let r = EventReply::None;
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, r#"{"type":"none"}"#);
        let r2 = EventReply::Cancel {
            reason: Some("not allowed".into()),
        };
        let s2 = serde_json::to_string(&r2).unwrap();
        assert!(s2.contains(r#""type":"cancel""#));
        assert!(s2.contains(r#""reason":"not allowed""#));
    }
}
