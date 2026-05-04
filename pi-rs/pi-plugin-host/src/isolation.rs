//! Fault tracking and ownership-registry types.
//!
//! Two concerns live here:
//!
//! 1. `FaultLedger` — rolling-window counter of recent plugin faults
//!    used by the host to decide whether `FaultPolicy::Restart` should
//!    keep restarting or give up.
//! 2. `OwnershipRegistry` — the per-plugin book of every side effect
//!    the plugin has registered (tools, subscriptions, widgets, ...).
//!    `unload()` walks this registry to remove the plugin's
//!    footprint cleanly.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};

use pi_plugin_protocol::EventKind;

/// One recorded fault. Timestamps let us answer "how many faults in
/// the last N seconds?" without growing unbounded.
#[derive(Debug, Clone)]
pub struct Fault {
    /// When the fault happened (wall clock).
    pub at: SystemTime,
    /// Short category for logs / UI: `"trap"`, `"timeout"`,
    /// `"memory"`, `"panic"`.
    pub category: String,
    /// Free-form detail, e.g. the wasmtime trap message.
    pub detail: String,
}

/// Rolling-window counter of recent faults for a plugin.
#[derive(Debug, Default)]
pub struct FaultLedger {
    faults: Vec<Fault>,
}

impl FaultLedger {
    /// Record a new fault.
    pub fn record(&mut self, category: impl Into<String>, detail: impl Into<String>) {
        self.faults.push(Fault {
            at: SystemTime::now(),
            category: category.into(),
            detail: detail.into(),
        });
    }

    /// Number of faults recorded within the last `window`.
    pub fn count_within(&mut self, window: Duration) -> usize {
        let cutoff = SystemTime::now() - window;
        self.faults.retain(|f| f.at >= cutoff);
        self.faults.len()
    }

    /// Every recorded fault (no trimming). Useful for `/plugin status`.
    pub fn all(&self) -> &[Fault] {
        &self.faults
    }

    /// Forget every recorded fault. Called on `unload` / `reload`.
    pub fn clear(&mut self) {
        self.faults.clear();
    }
}

/// Everything a single plugin has registered on the host side.
///
/// On `unload`/`reload` the host walks this struct and rips each
/// registration out of the corresponding live state (tool list,
/// subscription map, widget stack, etc.). Nothing here requires a
/// `&mut PluginHost` because the registry is plain data.
#[derive(Debug, Default)]
pub struct OwnershipRegistry {
    /// Tools this plugin registered, keyed by tool name.
    pub tools: HashSet<String>,
    /// Event kinds this plugin subscribed to at runtime (manifest
    /// subscriptions are not tracked here since they're declared by
    /// the manifest and recomputed on reload).
    pub runtime_subscriptions: HashSet<EventKind>,
    /// Widget keys owned by the plugin (reserved for a future UI API).
    pub widgets: HashSet<String>,
    /// Status keys owned by the plugin (footer status text).
    pub statuses: HashSet<String>,
}

impl OwnershipRegistry {
    /// Merge `other` into `self`. Used when a plugin is re-registered
    /// after an allowed pattern (e.g. second `RegisterTool` for the
    /// same name is a no-op).
    pub fn extend(&mut self, other: OwnershipRegistry) {
        self.tools.extend(other.tools);
        self.runtime_subscriptions
            .extend(other.runtime_subscriptions);
        self.widgets.extend(other.widgets);
        self.statuses.extend(other.statuses);
    }
}

/// Collection of per-plugin ledgers, keyed by plugin id.
#[derive(Debug, Default)]
pub struct FaultRegistry {
    inner: HashMap<String, FaultLedger>,
}

impl FaultRegistry {
    /// Mutable handle to the ledger for `plugin`, creating one if
    /// missing.
    pub fn ledger_mut(&mut self, plugin: &str) -> &mut FaultLedger {
        self.inner.entry(plugin.to_string()).or_default()
    }

    /// Read-only handle, or `None` if no faults have been recorded.
    pub fn ledger(&self, plugin: &str) -> Option<&FaultLedger> {
        self.inner.get(plugin)
    }

    /// Drop every ledger for `plugin`.
    pub fn remove(&mut self, plugin: &str) {
        self.inner.remove(plugin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn ledger_counts_within_window() {
        let mut l = FaultLedger::default();
        l.record("trap", "a");
        l.record("trap", "b");
        assert_eq!(l.count_within(Duration::from_secs(60)), 2);
    }

    #[test]
    fn ledger_drops_expired() {
        let mut l = FaultLedger::default();
        // Manually backdate to simulate an old fault.
        l.faults.push(Fault {
            at: SystemTime::now() - Duration::from_secs(120),
            category: "old".into(),
            detail: "".into(),
        });
        l.record("trap", "fresh");
        assert_eq!(l.count_within(Duration::from_secs(60)), 1);
    }

    #[test]
    fn registry_extends_merges_fields() {
        let mut a = OwnershipRegistry::default();
        a.tools.insert("x".into());
        let mut b = OwnershipRegistry::default();
        b.tools.insert("y".into());
        b.runtime_subscriptions.insert(EventKind::AgentStart);
        a.extend(b);
        assert!(a.tools.contains("x"));
        assert!(a.tools.contains("y"));
        assert!(a.runtime_subscriptions.contains(&EventKind::AgentStart));
    }

    #[test]
    fn fault_registry_round_trip() {
        let mut reg = FaultRegistry::default();
        reg.ledger_mut("hello").record("trap", "bad");
        assert_eq!(reg.ledger("hello").unwrap().all().len(), 1);
        reg.remove("hello");
        assert!(reg.ledger("hello").is_none());
    }
}
