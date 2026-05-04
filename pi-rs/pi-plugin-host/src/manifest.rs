//! Manifest loading: parse `plugin.toml` from disk and resolve the
//! sibling `.wasm` file path.

use std::path::{Path, PathBuf};

use thiserror::Error;

pub use pi_plugin_protocol::Manifest;

/// Errors produced by `load_manifest` / `resolve_wasm_path`.
#[derive(Debug, Error)]
pub enum ManifestError {
    /// The manifest file is missing or unreadable.
    #[error("could not read manifest at {path}: {source}")]
    Io {
        /// File path that failed.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// The manifest file is not valid TOML or has the wrong shape.
    #[error("could not parse manifest at {path}: {source}")]
    Parse {
        /// File path that failed.
        path: PathBuf,
        /// Underlying TOML error.
        #[source]
        source: toml::de::Error,
    },
    /// The `entry` field pointed at a non-existent `.wasm` file.
    #[error("entry {entry} (resolved to {resolved}) does not exist")]
    EntryMissing {
        /// Raw value from the manifest.
        entry: String,
        /// Absolute path we tried to read.
        resolved: PathBuf,
    },
}

/// Parse `plugin.toml` at `path`. Does not check that the `entry`
/// WASM file exists — use `resolve_wasm_path` for that.
pub fn load_manifest(path: impl AsRef<Path>) -> Result<Manifest, ManifestError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&text).map_err(|source| ManifestError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

/// Resolve the `entry` field of a manifest against the manifest's
/// parent directory and check the resulting `.wasm` file exists.
pub fn resolve_wasm_path(
    manifest_path: impl AsRef<Path>,
    manifest: &Manifest,
) -> Result<PathBuf, ManifestError> {
    let parent = manifest_path
        .as_ref()
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let resolved = parent.join(&manifest.entry);
    if !resolved.is_file() {
        return Err(ManifestError::EntryMissing {
            entry: manifest.entry.clone(),
            resolved,
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_plugin_protocol::{EventKind, Permission};
    use std::io::Write;

    fn tmpdir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("pi-plugin-host-manifest-test")
            .tempdir()
            .unwrap()
    }

    #[test]
    fn parses_minimal_manifest() {
        let dir = tmpdir();
        let path = dir.path().join("plugin.toml");
        let wasm = dir.path().join("hello.wasm");
        std::fs::File::create(&wasm).unwrap();
        std::fs::write(
            &path,
            r#"
name = "hello"
version = "0.1.0"
description = "say hi"
entry = "hello.wasm"
permissions = ["ui"]
events = ["agent_start"]
"#,
        )
        .unwrap();

        let m = load_manifest(&path).unwrap();
        assert_eq!(m.name, "hello");
        assert_eq!(m.version, "0.1.0");
        assert_eq!(m.permissions, vec![Permission::Ui]);
        assert_eq!(m.events, vec![EventKind::AgentStart]);

        let resolved = resolve_wasm_path(&path, &m).unwrap();
        assert_eq!(resolved, wasm);
    }

    #[test]
    fn accepts_custom_limits_and_fault_policy() {
        let dir = tmpdir();
        let path = dir.path().join("plugin.toml");
        std::fs::File::create(dir.path().join("p.wasm")).unwrap();
        std::fs::write(
            &path,
            r#"
name = "big"
version = "1.2.3"
entry = "p.wasm"

[limits]
memory_bytes = 134217728
fuel = 5_000_000_000
call_timeout_ms = 60000

[fault]
policy = "restart"
max_faults = 5
fault_window_secs = 120
"#,
        )
        .unwrap();
        let m = load_manifest(&path).unwrap();
        assert_eq!(m.limits.memory_bytes, 128 * 1024 * 1024);
        assert_eq!(m.limits.fuel, 5_000_000_000);
        assert_eq!(m.fault.policy, pi_plugin_protocol::FaultPolicy::Restart);
        assert_eq!(m.fault.max_faults, 5);
    }

    #[test]
    fn reports_parse_errors() {
        let dir = tmpdir();
        let path = dir.path().join("plugin.toml");
        std::fs::write(&path, "not = valid = toml\n").unwrap();
        match load_manifest(&path) {
            Err(ManifestError::Parse { .. }) => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn reports_missing_wasm() {
        let dir = tmpdir();
        let path = dir.path().join("plugin.toml");
        std::fs::write(
            &path,
            "name = \"x\"\nversion = \"0.0.0\"\nentry = \"missing.wasm\"\n",
        )
        .unwrap();
        let m = load_manifest(&path).unwrap();
        match resolve_wasm_path(&path, &m) {
            Err(ManifestError::EntryMissing { entry, .. }) => assert_eq!(entry, "missing.wasm"),
            other => panic!("expected EntryMissing, got {other:?}"),
        }
    }

    #[test]
    fn reports_io_errors() {
        let r = load_manifest("/nonexistent/plugin.toml");
        match r {
            Err(ManifestError::Io { .. }) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    // Keep clippy happy about unused import on non-test builds.
    #[allow(dead_code)]
    fn _keep_write_import(_w: &mut impl Write) {}
}
