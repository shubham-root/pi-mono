//! Permission store: persisted user decisions per plugin.
//!
//! Grants live at `~/.config/pi/plugin-permissions.json`. The store is
//! an in-memory cache that can read / write the on-disk JSON.
//!
//! Schema:
//!
//! ```json
//! {
//!   "version": 1,
//!   "plugins": {
//!     "hello": {
//!       "ui":          {"granted": true,  "decided_at": 1700000000},
//!       "read_files":  {"granted": false, "decided_at": 1700000005}
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use pi_plugin_protocol::{Permission, PermissionGrant};

/// Errors returned by `PermissionStore`.
#[derive(Debug, Error)]
pub enum PermissionStoreError {
    /// Could not read the backing file.
    #[error("read {path}: {source}")]
    Read {
        /// File path involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// Could not write the backing file.
    #[error("write {path}: {source}")]
    Write {
        /// File path involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// JSON parse failure.
    #[error("parse {path}: {source}")]
    Parse {
        /// File path involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct OnDisk {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    plugins: HashMap<String, HashMap<String, StoredGrant>>,
}

fn default_version() -> u32 {
    1
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredGrant {
    granted: bool,
    decided_at: u64,
}

/// In-memory permission cache backed by a JSON file on disk.
#[derive(Debug)]
pub struct PermissionStore {
    path: PathBuf,
    data: OnDisk,
}

impl PermissionStore {
    /// Open (or create) the permission store at `path`. Returns an
    /// empty store if the file doesn't exist yet.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, PermissionStoreError> {
        let path = path.into();
        if !path.exists() {
            return Ok(Self {
                path,
                data: OnDisk::default(),
            });
        }
        let text =
            std::fs::read_to_string(&path).map_err(|source| PermissionStoreError::Read {
                path: path.clone(),
                source,
            })?;
        let data: OnDisk =
            serde_json::from_str(&text).map_err(|source| PermissionStoreError::Parse {
                path: path.clone(),
                source,
            })?;
        Ok(Self { path, data })
    }

    /// Path this store is backed by.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Query whether `plugin` has been granted `perm`. Returns:
    /// - `Some(true)` if the user granted the permission;
    /// - `Some(false)` if the user explicitly denied it;
    /// - `None` if no decision has been recorded yet.
    pub fn get(&self, plugin: &str, perm: Permission) -> Option<bool> {
        self.data
            .plugins
            .get(plugin)?
            .get(perm.as_str())
            .map(|g| g.granted)
    }

    /// Record a grant decision in memory. Does not persist — call
    /// `save()` to write to disk.
    pub fn set(&mut self, plugin: &str, perm: Permission, granted: bool) {
        let stored = StoredGrant {
            granted,
            decided_at: now_secs(),
        };
        self.data
            .plugins
            .entry(plugin.to_string())
            .or_default()
            .insert(perm.as_str().to_string(), stored);
    }

    /// Return every grant recorded for `plugin`. Useful for UI.
    pub fn list(&self, plugin: &str) -> Vec<PermissionGrant> {
        let Some(map) = self.data.plugins.get(plugin) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(map.len());
        for (k, v) in map {
            if let Some(p) = parse_permission(k) {
                out.push(PermissionGrant {
                    permission: p,
                    granted: v.granted,
                    decided_at: v.decided_at,
                });
            }
        }
        out
    }

    /// Persist the in-memory state back to the backing file. Creates
    /// parent directories as needed.
    pub fn save(&self) -> Result<(), PermissionStoreError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| PermissionStoreError::Write {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let text = serde_json::to_string_pretty(&self.data).map_err(|source| {
            PermissionStoreError::Parse {
                path: self.path.clone(),
                source,
            }
        })?;
        std::fs::write(&self.path, text).map_err(|source| PermissionStoreError::Write {
            path: self.path.clone(),
            source,
        })
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

fn parse_permission(s: &str) -> Option<Permission> {
    // Reuse serde's case handling so this table never diverges from
    // `Permission::as_str`.
    serde_json::from_value(serde_json::Value::String(s.to_string())).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("pi-plugin-host-perm-test")
            .tempdir()
            .unwrap()
    }

    #[test]
    fn empty_store_returns_none() {
        let dir = tmp();
        let store = PermissionStore::open(dir.path().join("perms.json")).unwrap();
        assert_eq!(store.get("hello", Permission::Ui), None);
    }

    #[test]
    fn set_then_get_roundtrip() {
        let dir = tmp();
        let mut store = PermissionStore::open(dir.path().join("perms.json")).unwrap();
        store.set("hello", Permission::Ui, true);
        store.set("hello", Permission::ReadFiles, false);
        assert_eq!(store.get("hello", Permission::Ui), Some(true));
        assert_eq!(store.get("hello", Permission::ReadFiles), Some(false));
        assert_eq!(store.get("hello", Permission::WriteFiles), None);
    }

    #[test]
    fn save_and_reload_preserves_grants() {
        let dir = tmp();
        let path = dir.path().join("perms.json");
        {
            let mut store = PermissionStore::open(&path).unwrap();
            store.set("hello", Permission::Ui, true);
            store.set("bye", Permission::HttpRequest, false);
            store.save().unwrap();
        }
        let store = PermissionStore::open(&path).unwrap();
        assert_eq!(store.get("hello", Permission::Ui), Some(true));
        assert_eq!(store.get("bye", Permission::HttpRequest), Some(false));
        assert_eq!(store.get("hello", Permission::HttpRequest), None);
    }

    #[test]
    fn list_returns_known_grants() {
        let dir = tmp();
        let mut store = PermissionStore::open(dir.path().join("perms.json")).unwrap();
        store.set("hello", Permission::Ui, true);
        store.set("hello", Permission::ReadFiles, false);
        let mut grants = store.list("hello");
        grants.sort_by_key(|g| g.permission.as_str());
        assert_eq!(grants.len(), 2);
        assert!(grants
            .iter()
            .any(|g| g.permission == Permission::Ui && g.granted));
        assert!(grants
            .iter()
            .any(|g| g.permission == Permission::ReadFiles && !g.granted));
    }

    #[test]
    fn save_creates_missing_parent_dirs() {
        let dir = tmp();
        let nested = dir.path().join("does/not/exist/perms.json");
        let mut store = PermissionStore::open(&nested).unwrap();
        store.set("hello", Permission::Ui, true);
        store.save().unwrap();
        assert!(nested.is_file());
    }
}
