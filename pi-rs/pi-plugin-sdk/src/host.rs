//! Ergonomic wrappers over the raw `pi_host_call` ABI.
//!
//! Plugins typically call the functions in this module rather than
//! constructing `HostRequest` values by hand:
//!
//! ```ignore
//! host::log_info("starting");
//! host::subscribe(EventKind::AgentStart);
//! host::ui_notify("info", "hi there");
//! let cwd = host::get_cwd().unwrap_or_default();
//! ```
//!
//! Every function serializes a `HostRequest`, invokes the import
//! `pi_host_call(ptr, len) -> i64`, and decodes the resulting
//! `HostResponse`. Errors are surfaced as `Result<_, HostError>`.

use pi_plugin_protocol::{EventKind, HostError, HostRequest, HostResponse, ToolResult, ToolSchema};

#[allow(unused_imports)]
use crate::runtime::{pack_ptrlen, take_buffer, unpack_ptrlen};

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "pi")]
unsafe extern "C" {
    fn pi_host_call(ptr: u32, len: u32, out_ptr_addr: u32, out_len_addr: u32);
}

/// Call the single host entry point with a pre-serialized request
/// body. Returns the raw reply bytes or an error if the host refused
/// to allocate / write a reply.
///
/// On non-wasm targets (unit tests running on the host) this always
/// returns an error — the extern isn't defined there, so we stub it.
pub fn raw_host_call(bytes: &[u8]) -> Result<Vec<u8>, HostError> {
    #[cfg(target_arch = "wasm32")]
    {
        let mut out_ptr: u32 = 0;
        let mut out_len: u32 = 0;
        // SAFETY: `bytes` stays live until the extern returns; the
        // host reads it synchronously before writing the reply
        // through our `pi_alloc`. The two output slots live on our
        // stack for the duration of this call.
        unsafe {
            pi_host_call(
                bytes.as_ptr() as usize as u32,
                bytes.len() as u32,
                &mut out_ptr as *mut u32 as usize as u32,
                &mut out_len as *mut u32 as usize as u32,
            );
        }
        if out_len == 0 {
            return Err(HostError::new(
                "no_response",
                "host returned empty response",
            ));
        }
        Ok(take_buffer(out_ptr, out_len))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = bytes;
        let _ = take_buffer as fn(u32, u32) -> Vec<u8>;
        Err(HostError::new(
            "host_unavailable",
            "pi_host_call is only available on wasm32 targets",
        ))
    }
}

/// Send a `HostRequest` and decode the `HostResponse`.
pub fn send(request: &HostRequest) -> HostResponse {
    let bytes = match serde_json::to_vec(request) {
        Ok(b) => b,
        Err(e) => {
            return HostResponse::Err(HostError::new(
                "encode_failed",
                format!("encode request: {e}"),
            ));
        }
    };
    match raw_host_call(&bytes) {
        Ok(reply) => match serde_json::from_slice::<HostResponse>(&reply) {
            Ok(r) => r,
            Err(e) => HostResponse::Err(HostError::new(
                "decode_failed",
                format!("decode reply: {e}"),
            )),
        },
        Err(e) => HostResponse::Err(e),
    }
}

// ---------- log helpers ----------

/// Log at the given level. Level strings are `"error" | "warn" |
/// "info" | "debug" | "trace"`; anything else is forwarded verbatim
/// and left for the host to interpret.
pub fn log(level: impl Into<String>, message: impl Into<String>) {
    let _ = send(&HostRequest::Log {
        level: level.into(),
        message: message.into(),
    });
}

/// Shortcut: `log("info", ...)`.
pub fn log_info(message: impl Into<String>) {
    log("info", message);
}

/// Shortcut: `log("warn", ...)`.
pub fn log_warn(message: impl Into<String>) {
    log("warn", message);
}

/// Shortcut: `log("error", ...)`.
pub fn log_error(message: impl Into<String>) {
    log("error", message);
}

// ---------- subscription ----------

/// Subscribe to an `EventKind` at runtime. Idempotent.
pub fn subscribe(event: EventKind) {
    let _ = send(&HostRequest::Subscribe { event });
}

/// Remove a runtime subscription. No-op if not subscribed.
pub fn unsubscribe(event: EventKind) {
    let _ = send(&HostRequest::Unsubscribe { event });
}

// ---------- tools ----------

/// Register a tool the LLM can call. Plugins should reply to the
/// matching `Event::ToolCall` with `EventReply::ToolResult(..)`.
pub fn register_tool(tool: ToolSchema) {
    let _ = send(&HostRequest::RegisterTool { tool });
}

/// Remove a previously-registered tool.
pub fn unregister_tool(name: impl Into<String>) {
    let _ = send(&HostRequest::UnregisterTool { name: name.into() });
}

/// Send a tool result asynchronously (for tools that span multiple
/// `pi_on_event` calls).
pub fn tool_result(tool_call_id: impl Into<String>, result: ToolResult) {
    let _ = send(&HostRequest::ToolResult {
        tool_call_id: tool_call_id.into(),
        result,
    });
}

// ---------- ui ----------

/// Show a toast-style notification in the TUI. `level` is one of
/// `"info" | "warn" | "error"`.
pub fn ui_notify(level: impl Into<String>, message: impl Into<String>) {
    let _ = send(&HostRequest::UiNotify {
        level: level.into(),
        message: message.into(),
    });
}

// ---------- getters ----------

/// Get the host's current working directory, or `None` if unavailable.
pub fn get_cwd() -> Option<String> {
    match send(&HostRequest::GetCwd) {
        HostResponse::Value(serde_json::Value::String(s)) => Some(s),
        _ => None,
    }
}

/// Get the qualified (`provider/model`) id of the active model, or
/// `None` if no model is selected / the host rejected the request.
pub fn get_active_model() -> Option<String> {
    match send(&HostRequest::GetActiveModel) {
        HostResponse::Value(serde_json::Value::String(s)) => Some(s),
        _ => None,
    }
}

/// Get the plugin's own config object (from
/// `~/.config/pi/plugins/<name>.toml`).
pub fn get_config() -> serde_json::Value {
    match send(&HostRequest::GetConfig) {
        HostResponse::Value(v) => v,
        _ => serde_json::json!({}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_call_stubbed_on_non_wasm() {
        // On the host-side test target, raw_host_call must return a
        // well-shaped error rather than crash.
        let err = raw_host_call(b"whatever").unwrap_err();
        assert_eq!(err.code, "host_unavailable");
    }

    #[test]
    fn send_on_non_wasm_returns_host_error() {
        match send(&HostRequest::GetCwd) {
            HostResponse::Err(e) => assert_eq!(e.code, "host_unavailable"),
            other => panic!("expected Err, got {other:?}"),
        }
    }
}
