//! End-to-end smoke tests for the plugin host.
//!
//! Each test builds a tiny WebAssembly module in WAT, compiles it via
//! wasmtime, and loads it through `PluginHost::load_with_manifest`
//! paired with an in-memory `Manifest`. This gives us coverage of the
//! full memory / allocator / host-call ABI without needing a
//! prebuilt plugin binary in the tree.

use std::path::PathBuf;
use std::sync::Arc;

use pi_plugin_host::{
    HostRequest, HostServices, Manifest, Plugin, PluginHost,
};
use pi_plugin_protocol::{Event, EventKind, Permission};

/// A `HostServices` stub that records every dispatched request.
#[derive(Default)]
struct Recorder {
    inner: std::sync::Mutex<Vec<(String, HostRequest)>>,
}

impl HostServices for Recorder {
    fn dispatch(
        &self,
        plugin: &str,
        request: HostRequest,
    ) -> pi_plugin_host::HostResponse {
        self.inner
            .lock()
            .unwrap()
            .push((plugin.to_string(), request));
        pi_plugin_host::HostResponse::Ok
    }
}

fn test_manifest(name: &str, events: &[EventKind]) -> Manifest {
    Manifest {
        name: name.to_string(),
        version: "0.0.0".into(),
        description: "test".into(),
        entry: "test.wasm".into(),
        permissions: vec![Permission::Ui],
        events: events.to_vec(),
        limits: Default::default(),
        fault: Default::default(),
    }
}

fn write_wasm(dir: &std::path::Path, wat: &str) -> PathBuf {
    let bytes = wat::parse_str(wat).expect("parse wat");
    let path = dir.join("test.wasm");
    std::fs::write(&path, bytes).unwrap();
    path
}

/// WAT for a plugin that:
/// - exports memory, pi_alloc (bump allocator into page 0 starting at
///   64KB worth of space), pi_dealloc (noop), pi_load, pi_on_event;
/// - in pi_load: writes `{"op":"log","level":"info","message":"hi"}`
///   into memory, calls pi_host_call with out-param pointers 100/104;
/// - in pi_on_event: always returns 0 (no reply).
const LOG_ON_LOAD_WAT: &str = r#"
(module
  (import "pi" "pi_host_call"
    (func $pi_host_call (param i32 i32 i32 i32)))

  (memory (export "memory") 1)

  ;; Bump allocator. Head starts at 1024 to leave room for static
  ;; data + out-param slots we might stash below.
  (global $head (mut i32) (i32.const 1024))

  (func (export "pi_alloc") (param $n i32) (result i32)
    (local $out i32)
    (local.set $out (global.get $head))
    (global.set $head (i32.add (global.get $head) (local.get $n)))
    (local.get $out))

  (func (export "pi_dealloc") (param $p i32) (param $n i32))

  ;; Static UTF-8 bytes for the request body.
  ;; {"op":"log","level":"info","message":"hi"}
  (data (i32.const 1) "{\22op\22:\22log\22,\22level\22:\22info\22,\22message\22:\22hi\22}")

  (func (export "pi_load")
    ;; Call pi_host_call(ptr=1, len=42, out_ptr_addr=100, out_len_addr=104).
    (call $pi_host_call (i32.const 1) (i32.const 42) (i32.const 100) (i32.const 104)))

  (func (export "pi_on_event") (param $ptr i32) (param $len i32) (result i64)
    (i64.const 0))
)
"#;

#[test]
fn plugin_load_triggers_host_log_call() {
    let dir = tempfile::tempdir().unwrap();
    let wasm_path = write_wasm(dir.path(), LOG_ON_LOAD_WAT);
    let services = Arc::new(Recorder::default());
    let mut host = PluginHost::new(services.clone()).unwrap();
    let manifest = test_manifest("hello", &[]);
    let id = host.load_with_manifest(manifest, &wasm_path).unwrap();
    assert_eq!(id, "hello");
    assert!(host.get("hello").unwrap().is_dispatchable());

    let calls = services.inner.lock().unwrap();
    assert!(
        calls.iter().any(|(p, r)| p == "hello"
            && matches!(
                r,
                HostRequest::Log { level, message }
                    if level == "info" && message == "hi"
            )),
        "expected a Log call recorded, got {calls:?}",
    );
}

/// WAT for a plugin that echoes back every event it receives.
/// `pi_on_event` returns a non-zero reply so we can exercise the
/// full reply-ptr/len decode path.
const ECHO_EVENT_WAT: &str = r#"
(module
  (import "pi" "pi_host_call"
    (func $pi_host_call (param i32 i32 i32 i32)))

  (memory (export "memory") 1)

  (global $head (mut i32) (i32.const 1024))

  (func (export "pi_alloc") (param $n i32) (result i32)
    (local $out i32)
    (local.set $out (global.get $head))
    (global.set $head (i32.add (global.get $head) (local.get $n)))
    (local.get $out))

  (func (export "pi_dealloc") (param $p i32) (param $n i32))

  ;; {"type":"none"}
  (data (i32.const 1) "{\22type\22:\22none\22}")

  (func (export "pi_on_event") (param $ptr i32) (param $len i32) (result i64)
    ;; pack (ptr=1, len=15) into i64
    (i64.or
      (i64.shl (i64.const 1) (i64.const 32))
      (i64.const 15)))
)
"#;

#[test]
fn plugin_receives_dispatched_event() {
    let dir = tempfile::tempdir().unwrap();
    let wasm_path = write_wasm(dir.path(), ECHO_EVENT_WAT);
    let services = Arc::new(Recorder::default());
    let mut host = PluginHost::new(services).unwrap();
    let manifest = test_manifest("echo", &[EventKind::AgentStart]);
    host.load_with_manifest(manifest, &wasm_path).unwrap();

    let replies = host.dispatch(&Event::AgentStart);
    assert_eq!(replies.len(), 1);
    assert_eq!(replies[0].0, "echo");
    match &replies[0].1 {
        pi_plugin_protocol::EventReply::None => {}
        other => panic!("expected None reply, got {other:?}"),
    }
}

#[test]
fn unsubscribed_plugin_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let wasm_path = write_wasm(dir.path(), ECHO_EVENT_WAT);
    let services = Arc::new(Recorder::default());
    let mut host = PluginHost::new(services).unwrap();
    // Subscribed to AgentEnd, not AgentStart.
    let manifest = test_manifest("echo", &[EventKind::AgentEnd]);
    host.load_with_manifest(manifest, &wasm_path).unwrap();

    let replies = host.dispatch(&Event::AgentStart);
    assert!(replies.is_empty());
}

#[test]
fn unload_removes_plugin() {
    let dir = tempfile::tempdir().unwrap();
    let wasm_path = write_wasm(dir.path(), ECHO_EVENT_WAT);
    let services = Arc::new(Recorder::default());
    let mut host = PluginHost::new(services).unwrap();
    let manifest = test_manifest("echo", &[EventKind::AgentStart]);
    host.load_with_manifest(manifest, &wasm_path).unwrap();
    assert_eq!(host.len(), 1);
    host.unload("echo").unwrap();
    assert_eq!(host.len(), 0);

    let replies = host.dispatch(&Event::AgentStart);
    assert!(replies.is_empty());
}

#[test]
fn duplicate_load_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let wasm_path = write_wasm(dir.path(), ECHO_EVENT_WAT);
    let services = Arc::new(Recorder::default());
    let mut host = PluginHost::new(services).unwrap();
    let manifest = test_manifest("echo", &[EventKind::AgentStart]);
    host.load_with_manifest(manifest.clone(), &wasm_path)
        .unwrap();
    let err = host
        .load_with_manifest(manifest, &wasm_path)
        .unwrap_err()
        .to_string();
    assert!(err.contains("already loaded"), "got: {err}");
}

#[test]
fn plugin_ref_exposes_state() {
    let dir = tempfile::tempdir().unwrap();
    let wasm_path = write_wasm(dir.path(), ECHO_EVENT_WAT);
    let services = Arc::new(Recorder::default());
    let mut host = PluginHost::new(services).unwrap();
    let manifest = test_manifest("echo", &[EventKind::AgentStart]);
    host.load_with_manifest(manifest, &wasm_path).unwrap();
    let plugin: &Plugin = host.get("echo").unwrap();
    assert_eq!(plugin.id, "echo");
    assert!(plugin.subscriptions.contains(&EventKind::AgentStart));
}
