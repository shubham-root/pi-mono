//! Permission system for plugins.
//! To be implemented in Phase 4.8.

use anyhow::Result;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    FileRead,
    FileWrite,
    RunCommand,
    Network,
    Llm,
    CrossSession,
}

pub struct PermissionManager {
    granted: std::collections::HashMap<String, Vec<Permission>>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self {
            granted: HashMap::new(),
        }
    }

    pub fn has_permission(&self, plugin: &str, perm: Permission) -> bool {
        self.granted
            .get(plugin)
            .map(|perms| perms.contains(&perm))
            .unwrap_or(false)
    }

    pub async fn request(&mut self, _plugin: &str, _permission: Permission) -> bool {
        // Stub - will prompt user in TUI
        true
    }
}
