//! Low-level runtime glue used by the `pi_plugin!` macro.
//!
//! Plugin authors should not call these functions directly; they are
//! here so the macro can expand to `$crate::runtime::alloc(..)`
//! without pulling the implementation into every plugin crate.
//!
//! ## Allocator design
//!
//! The allocator is a bump arena at a fixed high address inside the
//! guest's linear memory. We deliberately avoid Rust's default heap
//! allocator for ABI buffers for two reasons:
//!
//! 1. **Layout symmetry**: the host handing us `(ptr, len)` pairs
//!    has no `Layout` to round-trip. Returning those pointers to
//!    dlmalloc via `Layout::from_size_align(len, 1)` only works when
//!    the allocator honours alignment=1 exactly — in practice
//!    wasm32-unknown-unknown's dlmalloc sometimes corrupts its
//!    free list under that pattern and manifests as bizarre
//!    out-of-bounds reads on subsequent allocations.
//! 2. **Portability**: Go / Zig / AssemblyScript SDKs can implement
//!    this same bump arena in ~15 lines; mirroring Rust's dlmalloc
//!    across four languages would be enormous.
//!
//! The trade-off is unbounded memory growth: plugins that do heavy
//! host I/O will eventually exhaust the arena and `abort`. Once we
//! start exercising that in practice we'll switch to a free-list.
//! The host's per-plugin linear-memory cap (default 64 MiB) is the
//! hard upper bound.

use std::cell::Cell;

use pi_plugin_protocol::EventReply;

/// Install a panic hook that writes the panic message through
/// `host::log_error` before the WASM module traps.
///
/// Without this, a Rust panic inside a plugin shows up on the host
/// side as a bare `wasm trap: unreachable` with no message, making
/// plugin debugging painful. Call this once (idempotent) — the
/// `pi_plugin!` macro calls it automatically on first entry.
pub fn install_panic_hook() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let payload = info.payload();
            let message = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                String::from("plugin panicked")
            };
            let location = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| String::from("<unknown>"));
            crate::host::log_error(format!("panic at {location}: {message}"));
        }));
    });
}

/// Pack a `(ptr, len)` pair into a single `u64` the way the host
/// expects to receive it from `pi_on_event`.
pub fn pack_ptrlen(ptr: u32, len: u32) -> u64 {
    ((ptr as u64) << 32) | (len as u64)
}

/// Split a `u64` packed by `pack_ptrlen` back into `(ptr, len)`.
pub fn unpack_ptrlen(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32)
}

// ---------- bump arena ----------

/// Base address of the plugin ABI arena, in the guest's linear
/// memory. Chosen to sit above any static data and Rust's shadow
/// stack (both typically end before 1 MiB on wasm32-unknown-unknown).
#[allow(dead_code)]
const ARENA_BASE: u32 = 0x0080_0000;
/// Arena size: 8 MiB. Bounded by the per-plugin linear-memory cap
/// enforced by the host (default 64 MiB).
#[allow(dead_code)]
const ARENA_SIZE: u32 = 8 * 1024 * 1024;

#[cfg(target_arch = "wasm32")]
std::thread_local! {
    // `Cell` is enough because wasm32 is single-threaded under WASI.
    static ARENA_HEAD: Cell<u32> = const { Cell::new(ARENA_BASE) };
}

/// Ensure the guest's linear memory has at least `end` bytes. The
/// default Rust wasm module only brings up one page (64 KiB); we
/// need to grow to cover the arena the first time anyone touches it.
#[cfg(target_arch = "wasm32")]
fn ensure_memory_for(end: u32) {
    // `memory.size` is in 64 KiB pages; grow in whole-page chunks.
    const PAGE: u32 = 65_536;
    let need_pages = end.div_ceil(PAGE);
    // `core::arch::wasm32::memory_{size,grow}` are safe on
    // wasm32; the generated WASM just emits `memory.size` /
    // `memory.grow` against memory index 0. No `unsafe` needed.
    let current = core::arch::wasm32::memory_size(0) as u32;
    if current < need_pages {
        let delta = need_pages - current;
        let prev = core::arch::wasm32::memory_grow(0, delta as usize);
        if prev == usize::MAX {
            std::process::abort();
        }
    }
}

/// Allocate `n` bytes from the ABI arena and return a guest pointer.
/// The alignment is always 1 — plugin payloads are byte buffers.
pub fn alloc(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    #[cfg(target_arch = "wasm32")]
    {
        ARENA_HEAD.with(|head| {
            let base = head.get();
            let next = base.saturating_add(n);
            if next > ARENA_BASE.saturating_add(ARENA_SIZE) {
                // Arena full: abort deterministically so the host
                // records a trap rather than silently returning a
                // duplicate pointer.
                std::process::abort();
            }
            ensure_memory_for(next);
            head.set(next);
            base
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = Cell::<u32>::new;
        0
    }
}

/// Free a buffer previously returned by `alloc`. Currently a no-op
/// in the bump allocator; kept so `pi_dealloc` retains a stable ABI
/// for when we switch to a free-list.
pub fn dealloc(ptr: u32, len: u32) {
    let _ = (ptr, len);
}

// ---------- buffer transfer helpers ----------

/// Read a host-allocated buffer into a Vec and free the original.
pub fn take_buffer(ptr: u32, len: u32) -> Vec<u8> {
    if ptr == 0 || len == 0 {
        return Vec::new();
    }
    // SAFETY: `ptr`/`len` point at a buffer the host just wrote via
    // our `alloc`. Copying the bytes out of linear memory is sound.
    let slice = unsafe { std::slice::from_raw_parts(ptr as usize as *const u8, len as usize) };
    let out = slice.to_vec();
    dealloc(ptr, len);
    out
}

/// Serialize an `EventReply`, copy it into a freshly allocated guest
/// buffer, and return the packed `(ptr, len)` the host expects.
pub fn write_reply(reply: &EventReply) -> u64 {
    let bytes = match serde_json::to_vec(reply) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    write_bytes(&bytes)
}

/// Copy `bytes` into guest memory via `alloc` and return the packed
/// `(ptr, len)` for the host to consume.
pub fn write_bytes(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let len = bytes.len() as u32;
    let ptr = alloc(len);
    // SAFETY: `alloc` returned a valid, live buffer of exactly `len`
    // bytes; writing `bytes` into it stays in-bounds.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            ptr as usize as *mut u8,
            len as usize,
        );
    }
    pack_ptrlen(ptr, len)
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
}
