//! pi-plugin-protocol: wire types shared between the pi plugin host and
//! plugin SDKs (Rust, Go, Zig, AssemblyScript, ...).
//!
//! The wire format is JSON. Host and guest exchange frames through a
//! single host-call entry point plus a small number of guest exports
//! (`pi_load`, `pi_on_event`, `pi_alloc` / `pi_dealloc`).
//!
//! Why JSON and not protobuf? Every SDK language already has a built-in
//! JSON codec; debugging is trivial (`println!("{}", frame)`); plugin
//! event rates are measured in hundreds per session, not millions per
//! second, so the CPU difference is irrelevant. If a future profile
//! shows otherwise we can add a `format: "pbf"` header and swap
//! codecs in one place.
//!
//! ## Frame layout
//!
//! Every host call is a single JSON object written to WASM linear
//! memory from the guest, pointed at by `(ptr, len)` passed to
//! `pi_host_call`. The host replies by allocating a guest-side
//! buffer via the guest's `pi_alloc` export and returning the
//! `(ptr, len)` pair in multi-value returns.
//!
//! The host → guest event path mirrors the shape: the host allocates
//! a guest-side buffer, writes the `Event` JSON there, and invokes
//! the guest's `pi_on_event(ptr, len) -> (response_ptr, response_len)`
//! export. The guest's reply is an optional `EventReply` JSON object
//! used for events that have a return value (e.g. `ToolCall` →
//! tool result).

#![deny(missing_docs)]

pub mod event;
pub mod host_call;
pub mod manifest;
pub mod permission;
pub mod tool;

pub use event::{Event, EventKind, EventReply, SessionStartReason, UnloadReason};
pub use host_call::{HostRequest, HostResponse, HostError};
pub use manifest::{Manifest, FaultPolicy};
pub use permission::{Permission, PermissionGrant};
pub use tool::{ToolSchema, ToolResult, ToolResultKind};

/// Current protocol version. Host refuses to load plugins built
/// against a different major version.
pub const PROTOCOL_VERSION: (u16, u16) = (0, 1);
