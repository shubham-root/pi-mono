//! Plugin manifest parsing and validation.
//! To be implemented in Phase 4.2.

use anyhow::Result;
use crate::permissions::Permission;

#[derive(Debug, Clone)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub permissions: Vec<Permission>,
    pub events: Vec<String>,
    pub fault_policy: FaultPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultPolicy {
    Restart,
    Disable,
    Ignore,
}

impl PluginManifest {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        // Stub - parse TOML manifest
        Ok(Self {
            name: "example".into(),
            version: "0.1.0".into(),
            permissions: Vec::new(),
            events: Vec::new(),
            fault_policy: FaultPolicy::Disable,
        })
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}
