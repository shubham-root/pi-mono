//! Resource loader — aggregates skills (and future prompt templates,
//! themes) from every configured source so the TUI, agent, and
//! plugin host can all consume a single snapshot.
//!
//! Mirrors the shape of TS's `resource-loader.ts` but narrower: in
//! Phase 4 we only need skills. Prompt templates and themes get
//! hung off the same loader when they land.

use std::path::PathBuf;

use crate::session::default_agent_dir;
use crate::settings::Settings;
use crate::skills::{load_skills, LoadSkillsOptions, LoadSkillsResult};

/// Configuration for a single `ResourceLoader` reload. `cwd` and
/// `agent_dir` mirror the TS constructor; everything else is
/// optional.
#[derive(Debug, Clone)]
pub struct ResourceLoaderOptions {
    /// Project root — used to discover `.pi/skills/` and walk up to
    /// find `.agents/skills/` dirs.
    pub cwd: PathBuf,
    /// Global agent directory. Defaults to `~/.pi` via
    /// `default_agent_dir()` when `None`.
    pub agent_dir: Option<PathBuf>,
    /// Extra skill paths from CLI (`--skill <path>`) and settings
    /// (`skills: [..]`). Each entry can be a directory or a single
    /// `.md` file. Duplicates are fine — the loader dedupes.
    pub skill_paths: Vec<PathBuf>,
    /// If true, skip the default global / project scan. Explicit
    /// `skill_paths` still load. Matches the TS `--no-skills` flag.
    pub no_skills: bool,
}

impl Default for ResourceLoaderOptions {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            agent_dir: None,
            skill_paths: Vec::new(),
            no_skills: false,
        }
    }
}

/// Snapshot of every resource Pi has loaded. Immutable — rebuild by
/// calling `ResourceLoader::reload`.
#[derive(Debug, Clone, Default)]
pub struct ResourceSnapshot {
    /// Skills available this turn.
    pub skills: LoadSkillsResult,
}

/// Handle to reload resources on demand. Cheap to construct; the
/// real work happens in `reload`.
#[derive(Debug)]
pub struct ResourceLoader {
    opts: ResourceLoaderOptions,
    snapshot: ResourceSnapshot,
}

impl ResourceLoader {
    /// Create a fresh loader with no resources yet. Call `reload` to
    /// populate.
    pub fn new(opts: ResourceLoaderOptions) -> Self {
        Self {
            opts,
            snapshot: ResourceSnapshot::default(),
        }
    }

    /// Create a loader and immediately populate it from the given
    /// options. Convenience for CLI paths.
    pub fn open(opts: ResourceLoaderOptions) -> Self {
        let mut loader = Self::new(opts);
        loader.reload();
        loader
    }

    /// Build a loader from merged settings + CLI input. CLI wins;
    /// settings append. Paths coming from settings.skills use the
    /// per-location source (user / project) that `load_skills`
    /// infers from the path.
    pub fn from_settings_and_cli(
        cwd: PathBuf,
        agent_dir: Option<PathBuf>,
        settings: &Settings,
        cli_skills: Vec<PathBuf>,
        no_skills: bool,
    ) -> Self {
        let mut skill_paths = Vec::new();
        for path in settings.skills.iter().flatten() {
            skill_paths.push(expand_user(path));
        }
        skill_paths.extend(cli_skills);
        Self::open(ResourceLoaderOptions {
            cwd,
            agent_dir,
            skill_paths,
            no_skills,
        })
    }

    /// Rescan every configured source and replace the snapshot.
    pub fn reload(&mut self) {
        let agent_dir = self
            .opts
            .agent_dir
            .clone()
            .unwrap_or_else(default_agent_dir);
        let skills = load_skills(LoadSkillsOptions {
            cwd: self.opts.cwd.clone(),
            agent_dir,
            skill_paths: self.opts.skill_paths.clone(),
            include_defaults: !self.opts.no_skills
                || !self.opts.skill_paths.is_empty()
                    && !self.opts.no_skills,
        });
        self.snapshot = ResourceSnapshot { skills };
    }

    /// Accessor for the current snapshot.
    pub fn snapshot(&self) -> &ResourceSnapshot {
        &self.snapshot
    }

    /// Shortcut — current skill set.
    pub fn skills(&self) -> &LoadSkillsResult {
        &self.snapshot.skills
    }
}

/// Expand a leading `~` or `~/` to the user's home directory.
/// Everything else is returned as-is, letting callers decide whether
/// to treat relative paths as cwd-relative.
pub fn expand_user(path: &str) -> PathBuf {
    let trimmed = path.trim();
    if trimmed == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::SkillSource;
    use std::fs;
    use std::io::Write;

    fn tmp() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("pi-resource-loader-test-")
            .tempdir()
            .unwrap()
    }

    fn write_file(path: &std::path::Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    #[test]
    fn loads_nothing_when_no_skills_and_no_paths() {
        let loader = ResourceLoader::open(ResourceLoaderOptions {
            cwd: PathBuf::from("/nonexistent/cwd"),
            agent_dir: Some(PathBuf::from("/nonexistent/agent")),
            skill_paths: vec![],
            no_skills: true,
        });
        assert!(loader.skills().skills.is_empty());
    }

    #[test]
    fn loads_explicit_paths_even_when_no_skills() {
        let dir = tmp();
        let skill = dir.path().join("greet/SKILL.md");
        write_file(
            &skill,
            "---\nname: greet\ndescription: say hi.\n---\nbody",
        );
        let loader = ResourceLoader::open(ResourceLoaderOptions {
            cwd: dir.path().to_path_buf(),
            agent_dir: Some(dir.path().to_path_buf()),
            skill_paths: vec![dir.path().join("greet")],
            no_skills: true,
        });
        let skills = &loader.skills().skills;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "greet");
        assert_eq!(skills[0].source, SkillSource::Path);
    }

    #[test]
    fn expand_user_handles_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_user("~"), home);
        assert_eq!(expand_user("~/foo"), home.join("foo"));
        assert_eq!(expand_user("/abs/path"), PathBuf::from("/abs/path"));
    }
}
