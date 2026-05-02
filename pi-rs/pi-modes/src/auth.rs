//! Persistent auth-key storage at `~/.pi/auth.json`.
//!
//! Matches the TypeScript pi on-disk format: a single flat JSON object
//! mapping `provider_id` → API-key string. Missing file is treated as
//! an empty map; malformed file surfaces as an error so the user can
//! fix it rather than having keys silently drop.

use anyhow::{Context, Result};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

pub fn auth_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not resolve $HOME")?;
    Ok(home.join(".pi").join("auth.json"))
}

/// Load the auth file into a provider→key map. Returns empty if the
/// file doesn't exist.
pub fn load_auth() -> Result<BTreeMap<String, String>> {
    let path = auth_path()?;
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    if raw.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    let v: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse {}", path.display()))?;
    let mut out = BTreeMap::new();
    if let Value::Object(map) = v {
        for (k, val) in map {
            if let Value::String(s) = val {
                out.insert(k, s);
            }
        }
    }
    Ok(out)
}

/// Write the auth map back. Creates `~/.pi/` if missing. File is
/// written with mode 0600 on unix so keys can't be read by other
/// users on a shared machine.
pub fn save_auth(map: &BTreeMap<String, String>) -> Result<()> {
    let path = auth_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let mut obj = Map::new();
    for (k, v) in map {
        obj.insert(k.clone(), Value::String(v.clone()));
    }
    let v = Value::Object(obj);
    let pretty = serde_json::to_string_pretty(&v)?;
    fs::write(&path, pretty)
        .with_context(|| format!("write {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)?.permissions();
        perms.set_mode(0o600);
        let _ = fs::set_permissions(&path, perms);
    }
    Ok(())
}

/// Add or update a single provider's key and persist the file.
pub fn set_key(provider_id: &str, key: &str) -> Result<()> {
    let mut auth = load_auth().unwrap_or_default();
    auth.insert(provider_id.to_string(), key.to_string());
    save_auth(&auth)
}

/// Clear a provider's key.
pub fn clear_key(provider_id: &str) -> Result<bool> {
    let mut auth = load_auth().unwrap_or_default();
    if auth.remove(provider_id).is_some() {
        save_auth(&auth)?;
        Ok(true)
    } else {
        Ok(false)
    }
}
