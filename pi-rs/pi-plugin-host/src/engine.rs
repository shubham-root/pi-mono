//! wasmtime-backed plugin engine.
//!
//! Responsibilities:
//! - Compile and instantiate `.wasm` modules with fuel metering and
//!   a linear-memory ceiling.
//! - Expose a single host import `pi_host_call(ptr, len) -> (ptr, len)`
//!   that the plugin SDK wraps in ergonomic `pi_log` / `pi_subscribe`
//!   helpers.
//! - Expose a small set of required guest exports (`pi_alloc`,
//!   `pi_dealloc`, `pi_load`, `pi_on_event`).
//! - Dispatch events (serialized `pi_plugin_protocol::Event`) into
//!   `pi_on_event` and decode the returned `EventReply`.
//! - Plug `HostServices` in as the backend that actually executes
//!   `HostRequest`s (writing logs, showing UI, etc.). The default
//!   `NullHostServices` is a no-op so unit tests don't need a real
//!   pi runtime.
//!
//! Phase 4 Stage 2 deliberately limits the API surface to:
//!   - loading a plugin;
//!   - dispatching `Event::Load` and `Event::AgentStart`;
//!   - handling `HostRequest::Log` and `HostRequest::UiNotify`.
//!
//! Subscription gating, tool registration, and richer host requests
//! land in follow-up stages of this phase.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use pi_plugin_protocol::{
    Event, EventKind, EventReply, FaultPolicy, HostError, HostRequest, HostResponse, Manifest,
    PROTOCOL_VERSION,
};
use wasmtime::{Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

use crate::isolation::{FaultRegistry, OwnershipRegistry};
use crate::manifest::resolve_wasm_path;

/// Opaque plugin id — for v1 we key plugins by their manifest `name`.
pub type PluginId = String;

/// Backend the host uses to execute `HostRequest`s issued by plugins.
///
/// The coding-agent implements this to route log lines to disk, pop
/// notifications into the TUI, update the tool registry, and so on.
/// Unit tests use `NullHostServices` which records calls for asserting.
pub trait HostServices: Send + Sync + 'static {
    /// Execute one host request. The returned response goes back to
    /// the plugin over the ABI.
    fn dispatch(&self, plugin: &str, request: HostRequest) -> HostResponse;
}

/// No-op `HostServices` used in tests. Captures every call so tests
/// can assert what the plugin did.
#[derive(Default, Debug)]
pub struct NullHostServices {
    /// Every `(plugin_id, request)` pair observed, in order.
    pub calls: Mutex<Vec<(String, HostRequest)>>,
}

impl HostServices for NullHostServices {
    fn dispatch(&self, plugin: &str, request: HostRequest) -> HostResponse {
        self.calls
            .lock()
            .expect("NullHostServices mutex poisoned")
            .push((plugin.to_string(), request.clone()));
        match request {
            HostRequest::GetCwd => HostResponse::Value(serde_json::Value::String(
                std::env::current_dir()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| String::new()),
            )),
            HostRequest::GetConfig => HostResponse::Value(serde_json::json!({})),
            HostRequest::GetActiveModel => HostResponse::Value(serde_json::Value::Null),
            _ => HostResponse::Ok,
        }
    }
}

/// Lifecycle state of a loaded plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Loaded and ready to receive events.
    Ready,
    /// Trapped at least once; further dispatches are skipped until
    /// the `FaultPolicy` says otherwise.
    Faulted,
    /// Explicitly disabled (user ran `/plugin unload`, or the fault
    /// budget was exceeded).
    Disabled,
}

/// One loaded plugin and the wasmtime state that backs it.
pub struct Plugin {
    /// Plugin id (= manifest name).
    pub id: PluginId,
    /// Parsed manifest.
    pub manifest: Manifest,
    /// Current lifecycle state.
    pub state: PluginState,
    /// Events this plugin is subscribed to (manifest + runtime).
    pub subscriptions: std::collections::HashSet<EventKind>,
    /// Ownership registry: what this plugin registered on the host.
    pub ownership: OwnershipRegistry,
    /// Backing wasmtime state. Held in an `Option` so we can
    /// dispose it on fault/unload without dropping the whole
    /// `Plugin` entry.
    runtime: Option<PluginRuntime>,
}

impl Plugin {
    /// True if the plugin is in a state that can receive events.
    pub fn is_dispatchable(&self) -> bool {
        matches!(self.state, PluginState::Ready) && self.runtime.is_some()
    }
}

/// wasmtime-side state for one plugin: Store + Instance + cached
/// typed function handles for the required exports.
struct PluginRuntime {
    store: Store<StoreCtx>,
    #[allow(dead_code)]
    instance: Instance,
    memory: Memory,
    pi_alloc: TypedFunc<u32, u32>,
    pi_dealloc: TypedFunc<(u32, u32), ()>,
    pi_load: Option<TypedFunc<(), ()>>,
    pi_on_event: Option<TypedFunc<(u32, u32), u64>>,
}

/// Per-Store context available to host functions.
#[derive(Clone)]
struct StoreCtx {
    plugin_id: PluginId,
    services: Arc<dyn HostServices>,
    /// Memory handle copied in after instantiation. `None` during
    /// the narrow window between `Store::new` and the linker step
    /// that stores the handle.
    memory: Option<Memory>,
    /// Same for `pi_alloc`, needed when the host has to allocate a
    /// buffer inside the guest to reply to a `pi_host_call`.
    pi_alloc: Option<TypedFunc<u32, u32>>,
}

/// Top-level engine: shared `wasmtime::Engine`, HostServices handle,
/// and the map of loaded plugins.
pub struct PluginHost {
    engine: Engine,
    services: Arc<dyn HostServices>,
    plugins: HashMap<PluginId, Plugin>,
    faults: FaultRegistry,
}

impl PluginHost {
    /// Construct a new host with the given backend. The backend is
    /// cloned (via `Arc`) into every plugin's `Store`.
    pub fn new(services: Arc<dyn HostServices>) -> Result<Self> {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config).context("build wasmtime Engine")?;
        Ok(Self {
            engine,
            services,
            plugins: HashMap::new(),
            faults: FaultRegistry::default(),
        })
    }

    /// How many plugins are currently loaded.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// True if no plugins are loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Iterate over loaded plugins.
    pub fn plugins(&self) -> impl Iterator<Item = &Plugin> {
        self.plugins.values()
    }

    /// Look up a plugin by id.
    pub fn get(&self, id: &str) -> Option<&Plugin> {
        self.plugins.get(id)
    }

    /// Parse `plugin.toml` at `manifest_path`, compile its `.wasm`,
    /// instantiate it, call `pi_load`, and register it under
    /// `manifest.name`.
    pub fn load_from_manifest(&mut self, manifest_path: impl AsRef<Path>) -> Result<PluginId> {
        let manifest_path = manifest_path.as_ref();
        let manifest = crate::manifest::load_manifest(manifest_path)
            .with_context(|| format!("parse manifest at {manifest_path:?}"))?;
        let wasm_path = resolve_wasm_path(manifest_path, &manifest)
            .with_context(|| format!("resolve wasm for {}", manifest.name))?;
        self.load_with_manifest(manifest, &wasm_path)
    }

    /// Load a plugin whose manifest you already have in memory. Used
    /// by tests that build a `.wasm` inline and pair it with an
    /// ad-hoc manifest.
    pub fn load_with_manifest(
        &mut self,
        manifest: Manifest,
        wasm_path: impl AsRef<Path>,
    ) -> Result<PluginId> {
        let id = manifest.name.clone();
        if self.plugins.contains_key(&id) {
            return Err(anyhow!("plugin {id:?} already loaded"));
        }

        let module = Module::from_file(&self.engine, wasm_path.as_ref())
            .with_context(|| format!("compile wasm for plugin {id:?}"))?;

        // Build the Store *before* the linker so we can stash the
        // context with the plugin id in it.
        let mut store = Store::new(
            &self.engine,
            StoreCtx {
                plugin_id: id.clone(),
                services: self.services.clone(),
                memory: None,
                pi_alloc: None,
            },
        );
        store
            .set_fuel(manifest.limits.fuel.max(1))
            .context("set fuel")?;

        let mut linker: Linker<StoreCtx> = Linker::new(&self.engine);
        self.install_host_imports(&mut linker)?;

        let instance = linker
            .instantiate(&mut store, &module)
            .with_context(|| format!("instantiate plugin {id:?}"))?;

        // Required guest exports — missing any is a hard error.
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("plugin {id:?} did not export `memory`"))?;
        let pi_alloc = instance
            .get_typed_func::<u32, u32>(&mut store, "pi_alloc")
            .context("plugin must export pi_alloc(u32)->u32")?;
        let pi_dealloc = instance
            .get_typed_func::<(u32, u32), ()>(&mut store, "pi_dealloc")
            .context("plugin must export pi_dealloc(u32,u32)->()")?;

        // Optional lifecycle exports.
        let pi_load = instance
            .get_typed_func::<(), ()>(&mut store, "pi_load")
            .ok();
        let pi_on_event = instance
            .get_typed_func::<(u32, u32), u64>(&mut store, "pi_on_event")
            .ok();

        // Back-fill the store context so host imports can reach
        // memory and the allocator during dispatch.
        store.data_mut().memory = Some(memory);
        store.data_mut().pi_alloc = Some(pi_alloc);

        let mut runtime = PluginRuntime {
            store,
            instance,
            memory,
            pi_alloc,
            pi_dealloc,
            pi_load,
            pi_on_event,
        };

        if let Some(f) = runtime.pi_load {
            f.call(&mut runtime.store, ())
                .with_context(|| format!("plugin {id:?} trapped in pi_load"))?;
        }

        let mut subscriptions = std::collections::HashSet::new();
        for ev in &manifest.events {
            subscriptions.insert(*ev);
        }

        let plugin = Plugin {
            id: id.clone(),
            manifest,
            state: PluginState::Ready,
            subscriptions,
            ownership: OwnershipRegistry::default(),
            runtime: Some(runtime),
        };
        self.plugins.insert(id.clone(), plugin);

        // Deliver a synthetic `Load` event so plugins get their
        // config + cwd even if they don't export pi_load.
        let load_event = Event::Load {
            config: serde_json::json!({}),
            cwd: std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        // Ignore errors here — if the plugin trapped on Load we
        // already recorded it.
        let _ = self.dispatch_single(&id, &load_event);
        Ok(id)
    }

    /// Unload a plugin: drop its wasmtime state, clear subscriptions
    /// and ownership. Idempotent.
    pub fn unload(&mut self, id: &str) -> Result<()> {
        if let Some(plugin) = self.plugins.get_mut(id) {
            plugin.state = PluginState::Disabled;
            plugin.runtime = None;
            plugin.subscriptions.clear();
            plugin.ownership = OwnershipRegistry::default();
            self.plugins.remove(id);
            self.faults.remove(id);
        }
        Ok(())
    }

    /// Dispatch an event to every loaded, subscribed, dispatchable
    /// plugin.
    ///
    /// Returns the list of `EventReply`s from plugins that responded,
    /// in the order they were dispatched. Plugins that trap are
    /// marked faulted and skipped.
    pub fn dispatch(&mut self, event: &Event) -> Vec<(PluginId, EventReply)> {
        let kind = EventKind::of(event);
        // Snapshot the candidate list so the dispatch loop can take
        // `&mut self.plugins`.
        let ids: Vec<PluginId> = self
            .plugins
            .iter()
            .filter_map(|(id, p)| {
                if p.is_dispatchable() && p.subscriptions.contains(&kind) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut replies = Vec::new();
        for id in ids {
            match self.dispatch_single(&id, event) {
                Ok(Some(reply)) => replies.push((id, reply)),
                Ok(None) => {}
                Err(e) => {
                    self.record_fault(&id, "trap", format!("{e:?}"));
                }
            }
        }
        replies
    }

    fn dispatch_single(&mut self, id: &str, event: &Event) -> Result<Option<EventReply>> {
        let Some(plugin) = self.plugins.get_mut(id) else {
            return Ok(None);
        };
        let Some(runtime) = plugin.runtime.as_mut() else {
            return Ok(None);
        };
        let Some(pi_on_event) = runtime.pi_on_event else {
            // Plugin didn't export pi_on_event; nothing to do.
            return Ok(None);
        };

        let payload = serde_json::to_vec(event).context("encode event json")?;
        let (ptr, len) = write_to_guest(runtime, &payload)?;
        let packed = pi_on_event
            .call(&mut runtime.store, (ptr, len))
            .context("pi_on_event trap")?;
        runtime
            .pi_dealloc
            .call(&mut runtime.store, (ptr, len))
            .context("pi_dealloc trap (event buffer)")?;
        let (reply_ptr, reply_len) = unpack_ptrlen(packed);
        if reply_len == 0 {
            return Ok(None);
        }
        let bytes = read_from_guest(runtime, reply_ptr, reply_len)?;
        runtime
            .pi_dealloc
            .call(&mut runtime.store, (reply_ptr, reply_len))
            .context("pi_dealloc trap (reply buffer)")?;
        let reply: EventReply = serde_json::from_slice(&bytes).context("decode event reply")?;
        Ok(Some(reply))
    }

    fn record_fault(&mut self, id: &str, category: &str, detail: String) {
        tracing::warn!(plugin = id, category, detail = %detail, "plugin fault");
        self.faults.ledger_mut(id).record(category, detail);
        if let Some(plugin) = self.plugins.get_mut(id) {
            plugin.state = PluginState::Faulted;
            match plugin.manifest.fault.policy {
                FaultPolicy::Disable => {
                    plugin.state = PluginState::Disabled;
                    plugin.runtime = None;
                }
                FaultPolicy::Ignore => {
                    // Keep runtime — may work for the next call.
                }
                FaultPolicy::Restart => {
                    // Stage 1: restart policy records the fault but
                    // doesn't actually re-instantiate yet. Backoff &
                    // reload land with the tool-execution stage.
                    plugin.runtime = None;
                }
            }
        }
    }

    /// Install the `pi` host imports on the linker. For Stage 2 we
    /// expose exactly one entry point: `pi_host_call(ptr, len) -> (ptr, len)`.
    fn install_host_imports(&self, linker: &mut Linker<StoreCtx>) -> Result<()> {
        linker
            .func_new(
                "pi",
                "pi_host_call",
                wasmtime::FuncType::new(
                    [
                        wasmtime::ValType::I32,
                        wasmtime::ValType::I32,
                        wasmtime::ValType::I32,
                        wasmtime::ValType::I32,
                    ],
                    [],
                ),
                |mut caller, params, _results| {
                    // Signature: pi_host_call(req_ptr, req_len,
                    //                         out_ptr_addr, out_len_addr) -> ()
                    //
                    // Using a single-return-type signature avoids
                    // wasm32-unknown-unknown's ABI split for u64
                    // returns, which otherwise forces us into the
                    // multivalue proposal (off by default in stable
                    // rustc). Returning nothing and writing the
                    // output `(ptr, len)` to two guest-provided
                    // address slots is portable across every SDK
                    // language (Rust, Go, Zig, AssemblyScript).
                    let req_ptr = params[0].unwrap_i32() as u32;
                    let req_len = params[1].unwrap_i32() as u32;
                    let out_ptr_addr = params[2].unwrap_i32() as u32;
                    let out_len_addr = params[3].unwrap_i32() as u32;

                    let (memory, pi_alloc, plugin_id, services) = {
                        let ctx = caller.data().clone();
                        let memory = ctx.memory.ok_or_else(|| anyhow!("memory not bound"))?;
                        let pi_alloc = ctx.pi_alloc.ok_or_else(|| anyhow!("pi_alloc not bound"))?;
                        (memory, pi_alloc, ctx.plugin_id, ctx.services)
                    };

                    let request_bytes =
                        read_memory(&mut caller, &memory, req_ptr, req_len)?;
                    let response = match serde_json::from_slice::<HostRequest>(&request_bytes) {
                        Ok(r) => services.dispatch(&plugin_id, r),
                        Err(e) => HostResponse::Err(HostError::new(
                            "invalid_request",
                            format!("decode: {e}"),
                        )),
                    };
                    write_host_response_via_outparams(
                        &mut caller,
                        &memory,
                        &pi_alloc,
                        &response,
                        out_ptr_addr,
                        out_len_addr,
                    )
                },
            )
            .context("link pi.pi_host_call")?;
        Ok(())
    }

    /// Protocol version the host advertises. Useful for a future
    /// `pi_host_info` negotiation call.
    pub fn protocol_version(&self) -> (u16, u16) {
        PROTOCOL_VERSION
    }
}

// ---- memory helpers ----

fn write_to_guest(runtime: &mut PluginRuntime, bytes: &[u8]) -> Result<(u32, u32)> {
    let len = u32::try_from(bytes.len()).context("payload too large for u32 length")?;
    let ptr = runtime
        .pi_alloc
        .call(&mut runtime.store, len)
        .context("pi_alloc trap")?;
    runtime
        .memory
        .write(&mut runtime.store, ptr as usize, bytes)
        .context("memory.write failed")?;
    Ok((ptr, len))
}

fn read_from_guest(runtime: &mut PluginRuntime, ptr: u32, len: u32) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    runtime
        .memory
        .read(&runtime.store, ptr as usize, &mut buf)
        .context("memory.read failed")?;
    Ok(buf)
}

fn read_memory(
    caller: &mut wasmtime::Caller<'_, StoreCtx>,
    memory: &Memory,
    ptr: u32,
    len: u32,
) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    memory
        .read(&mut *caller, ptr as usize, &mut buf)
        .context("read guest memory")?;
    Ok(buf)
}

/// Write the serialized `response` into a guest-allocated buffer and
/// publish its `(ptr, len)` into the two out-slot addresses the
/// guest passed in.
fn write_host_response_via_outparams(
    caller: &mut wasmtime::Caller<'_, StoreCtx>,
    memory: &Memory,
    pi_alloc: &TypedFunc<u32, u32>,
    response: &HostResponse,
    out_ptr_addr: u32,
    out_len_addr: u32,
) -> Result<()> {
    let bytes = serde_json::to_vec(response).context("encode host response")?;
    let len = u32::try_from(bytes.len()).context("response too large")?;
    let ptr = if len == 0 {
        0
    } else {
        pi_alloc
            .call(&mut *caller, len)
            .context("pi_alloc (host reply)")?
    };
    if len > 0 {
        memory
            .write(&mut *caller, ptr as usize, &bytes)
            .context("write host reply")?;
    }
    memory
        .write(&mut *caller, out_ptr_addr as usize, &ptr.to_le_bytes())
        .context("publish out_ptr")?;
    memory
        .write(&mut *caller, out_len_addr as usize, &len.to_le_bytes())
        .context("publish out_len")?;
    Ok(())
}

fn unpack_ptrlen(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32)
}

#[allow(dead_code)]
fn pack_ptrlen(ptr: u32, len: u32) -> u64 {
    ((ptr as u64) << 32) | (len as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptrlen_pack_roundtrip() {
        for (p, l) in [(0u32, 0u32), (1, 2), (0xFFFF_FFFF, 0), (12345, 67890)] {
            let packed = pack_ptrlen(p, l);
            assert_eq!(unpack_ptrlen(packed), (p, l));
        }
    }

    #[test]
    fn null_services_records_calls() {
        let svc = Arc::new(NullHostServices::default());
        let _ = svc.dispatch(
            "hello",
            HostRequest::Log {
                level: "info".into(),
                message: "hi".into(),
            },
        );
        assert_eq!(svc.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn new_host_starts_empty() {
        let host = PluginHost::new(Arc::new(NullHostServices::default())).unwrap();
        assert!(host.is_empty());
        assert_eq!(host.protocol_version(), PROTOCOL_VERSION);
    }
}
