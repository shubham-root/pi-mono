//! End-to-end test that builds the `hello` example plugin from
//! source (via `cargo build --target wasm32-unknown-unknown`), loads
//! it into `PluginHost`, dispatches `Event::AgentStart`, and asserts
//! the plugin produced the expected host calls through the real
//! Rust SDK.
//!
//! This exercises:
//! - the `pi_plugin!` macro expansion (pi_alloc / pi_dealloc /
//!   pi_load / pi_on_event)
//! - the SDK allocator and `pi_host_call` wrapper
//! - JSON round-trip over WASM linear memory
//! - subscription filtering in the host
//! - the `Event::Load` synthetic delivery path

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use pi_plugin_host::{HostRequest, HostResponse, HostServices, PluginHost};
use pi_plugin_protocol::Event;

/// Backend that remembers every `HostRequest` the plugin issued so
/// the test body can assert on them.
#[derive(Default)]
struct Recorder {
    calls: Mutex<Vec<(String, HostRequest)>>,
}

impl HostServices for Recorder {
    fn dispatch(&self, plugin: &str, request: HostRequest) -> HostResponse {
        self.calls
            .lock()
            .unwrap()
            .push((plugin.to_string(), request));
        HostResponse::Ok
    }
}

/// Build the hello plugin via cargo and return the path to the
/// resulting `.wasm` file. Uses `--offline` so the test doesn't hit
/// the network when the lockfile is already populated.
fn build_hello_wasm() -> PathBuf {
    let pi_rs_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let status = Command::new(env!("CARGO"))
        .arg("build")
        .arg("-p")
        .arg("pi-plugin-hello")
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .arg("--release")
        .current_dir(&pi_rs_root)
        .status()
        .expect("spawn cargo");
    assert!(status.success(), "cargo build -p pi-plugin-hello failed");

    let wasm = pi_rs_root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("pi_plugin_hello.wasm");
    assert!(wasm.is_file(), "expected {}", wasm.display());
    wasm
}

fn copy_hello_tree(dest_dir: &Path, wasm_src: &Path) -> PathBuf {
    let pi_rs_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    // Copy plugin.toml so `load_from_manifest` can find it next to
    // the wasm file.
    let manifest_src = pi_rs_root.join("pi-plugins").join("hello").join("plugin.toml");
    let manifest_dst = dest_dir.join("plugin.toml");
    std::fs::copy(&manifest_src, &manifest_dst).unwrap();
    let wasm_dst = dest_dir.join("pi_plugin_hello.wasm");
    std::fs::copy(wasm_src, &wasm_dst).unwrap();
    manifest_dst
}

#[test]
fn hello_plugin_greets_on_agent_start() {
    let wasm = build_hello_wasm();
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = copy_hello_tree(tmp.path(), &wasm);

    let recorder = Arc::new(Recorder::default());
    let mut host = PluginHost::new(recorder.clone()).unwrap();

    let id = host.load_from_manifest(&manifest_path).unwrap();
    assert_eq!(id, "hello");

    // The synthetic Load event should have delivered a log line.
    {
        let calls = recorder.calls.lock().unwrap();
        let logged_cwd_line = calls.iter().any(|(p, r)| {
            p == "hello"
                && matches!(r, HostRequest::Log { message, .. } if message.starts_with("hello loaded"))
        });
        assert!(
            logged_cwd_line,
            "expected a `hello loaded` log call after load, got {:?}",
            *calls
        );
    }

    // Dispatch an AgentStart and verify the plugin emitted exactly
    // one ui_notify with the expected shape.
    let before = recorder.calls.lock().unwrap().len();
    host.dispatch(&Event::AgentStart);
    let after: Vec<_> = recorder.calls.lock().unwrap()[before..].to_vec();
    let notifies: Vec<_> = after
        .iter()
        .filter(|(_, r)| matches!(r, HostRequest::UiNotify { .. }))
        .collect();
    assert_eq!(notifies.len(), 1, "expected 1 UiNotify, got {after:?}");
    if let (_, HostRequest::UiNotify { level, message }) = notifies[0] {
        assert_eq!(level, "info");
        assert!(
            message.contains("hello plugin says hi"),
            "unexpected notify: {message:?}"
        );
        assert!(
            message.contains("(1 so far)"),
            "counter should start at 1: {message:?}"
        );
    }

    // A second AgentStart should increment the counter.
    let before = recorder.calls.lock().unwrap().len();
    host.dispatch(&Event::AgentStart);
    let after: Vec<_> = recorder.calls.lock().unwrap()[before..].to_vec();
    let message = after
        .iter()
        .find_map(|(_, r)| match r {
            HostRequest::UiNotify { message, .. } => Some(message.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        message.contains("(2 so far)"),
        "counter should be 2: {message:?}"
    );
}

#[test]
fn hello_plugin_ignores_unsubscribed_events() {
    let wasm = build_hello_wasm();
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = copy_hello_tree(tmp.path(), &wasm);
    let recorder = Arc::new(Recorder::default());
    let mut host = PluginHost::new(recorder.clone()).unwrap();
    host.load_from_manifest(&manifest_path).unwrap();

    let before = recorder.calls.lock().unwrap().len();
    // hello subscribes to agent_start only. AgentEnd must be
    // filtered out entirely.
    host.dispatch(&Event::AgentEnd);
    let after = recorder.calls.lock().unwrap().len();
    assert_eq!(after, before, "plugin must not receive unsubscribed events");
}
