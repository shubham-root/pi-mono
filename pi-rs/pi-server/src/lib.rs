//! pi-server: Multi-session daemon and cross-session communication bus.
//!
//! Manages multiple concurrent agent sessions, handles IPC for
//! attach/detach, cross-session messaging, and plugin coordination.

pub mod daemon;
pub mod ipc;
pub mod session_lifecycle;
pub mod session_bus;

pub use daemon::PiServer;
pub use ipc::{IpcServer, IpcClient};
pub use session_lifecycle::SessionManager;
pub use session_bus::CrossSessionBus;
