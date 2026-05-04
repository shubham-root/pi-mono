//! Skills: self-contained capability packages the agent loads on
//! demand.
//!
//! Mirrors the TS implementation in
//! `packages/coding-agent/src/core/skills.ts` and the [Agent Skills
//! standard](https://agentskills.io/specification). Each skill lives
//! in its own directory with a `SKILL.md` that has YAML frontmatter
//! (`name`, `description`, optionally
//! `disable-model-invocation`). Optional helper scripts and
//! reference docs live alongside.
//!
//! Pi offers skills via progressive disclosure: only the
//! `description` of each skill is added to the system prompt at
//! turn-start; the full `SKILL.md` body is loaded by the model using
//! the `read` tool when a task matches, or by the user via the
//! `/skill:<name>` slash command.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::frontmatter::parse_frontmatter;

/// Max name length per the spec.
pub const MAX_NAME_LENGTH: usize = 64;
/// Max description length per the spec.
pub const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// One loaded skill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    /// Skill name (must match parent directory, see spec).
    pub name: String,
    /// Description surfaced to the model in the system prompt.
    pub description: String,
    /// Absolute path to the `SKILL.md` file.
    pub file_path: PathBuf,
    /// Directory that contains `SKILL.md`. Relative references
    /// inside the skill body (scripts, assets) are resolved here.
    pub base_dir: PathBuf,
    /// Where this skill came from, for diagnostics and UI.
    pub source: SkillSource,
    /// If true, the skill is hidden from the system prompt and can
    /// only be loaded by the user via `/skill:<name>`.
    pub disable_model_invocation: bool,
}

/// Origin of a discovered skill. Mirrors the TS `source` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillSource {
    /// Global user scope (`~/.pi/agent/skills/` or `~/.agents/skills/`).
    User,
    /// Project scope (`.pi/skills/` or `.agents/skills/`).
    Project,
    /// Loaded via `--skill <path>` or `settings.skills`.
    Path,
}

/// Diagnostic produced while loading skills. Non-fatal issues are
/// warnings; the skill still loads unless the message says otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillDiagnostic {
    /// Severity.
    pub kind: DiagnosticKind,
    /// Human-readable message.
    pub message: String,
    /// File the diagnostic relates to.
    pub path: PathBuf,
}

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticKind {
    /// Validation warning; skill still loads.
    Warning,
    /// Skill was skipped because of a hard error (e.g. missing
    /// description) or a name collision with an earlier skill.
    Collision,
}

/// Result of loading skills from one or more locations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadSkillsResult {
    /// Deduplicated list of skills, in discovery order.
    pub skills: Vec<Skill>,
    /// Warnings and collisions.
    pub diagnostics: Vec<SkillDiagnostic>,
}

/// Inputs for the top-level `load_skills` entry point.
#[derive(Debug, Clone)]
pub struct LoadSkillsOptions {
    /// Project root used to resolve project-local skill dirs.
    pub cwd: PathBuf,
    /// Global agent directory (usually `~/.pi`).
    pub agent_dir: PathBuf,
    /// Additional `--skill <path>` inputs from CLI and settings.
    /// Each path can be a directory to scan or an individual
    /// markdown file.
    pub skill_paths: Vec<PathBuf>,
    /// If false, the default global / project skill locations are
    /// not scanned. Explicit `skill_paths` still load.
    pub include_defaults: bool,
}

/// Discover skills from every configured location and merge them
/// with name collisions surfaced as diagnostics.
pub fn load_skills(opts: LoadSkillsOptions) -> LoadSkillsResult {
    let mut skill_map: HashMap<String, Skill> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut real_paths: std::collections::HashSet<PathBuf> =
        std::collections::HashSet::new();
    let mut diagnostics: Vec<SkillDiagnostic> = Vec::new();

    let mut add = |result: LoadSkillsResult,
                   skill_map: &mut HashMap<String, Skill>,
                   order: &mut Vec<String>,
                   real_paths: &mut std::collections::HashSet<PathBuf>,
                   diagnostics: &mut Vec<SkillDiagnostic>| {
        diagnostics.extend(result.diagnostics);
        for skill in result.skills {
            let real = fs::canonicalize(&skill.file_path).unwrap_or_else(|_| skill.file_path.clone());
            if real_paths.contains(&real) {
                continue;
            }
            match skill_map.get(&skill.name) {
                Some(existing) => {
                    diagnostics.push(SkillDiagnostic {
                        kind: DiagnosticKind::Collision,
                        message: format!(
                            "name \"{}\" collision (kept {})",
                            skill.name,
                            existing.file_path.display()
                        ),
                        path: skill.file_path.clone(),
                    });
                }
                None => {
                    order.push(skill.name.clone());
                    real_paths.insert(real);
                    skill_map.insert(skill.name.clone(), skill);
                }
            }
        }
    };

    if opts.include_defaults {
        // Global: ~/.pi/agent/skills/
        let global_pi = opts.agent_dir.join("agent").join("skills");
        add(
            load_skills_from_dir(&global_pi, SkillSource::User, true),
            &mut skill_map,
            &mut order,
            &mut real_paths,
            &mut diagnostics,
        );
        // Global: ~/.agents/skills/ (shared cross-harness convention).
        if let Some(home) = dirs::home_dir() {
            let global_agents = home.join(".agents").join("skills");
            if global_agents != global_pi {
                add(
                    load_skills_from_dir(&global_agents, SkillSource::User, false),
                    &mut skill_map,
                    &mut order,
                    &mut real_paths,
                    &mut diagnostics,
                );
            }
        }
        // Project: .pi/skills
        let project_pi = opts.cwd.join(".pi").join("skills");
        add(
            load_skills_from_dir(&project_pi, SkillSource::Project, true),
            &mut skill_map,
            &mut order,
            &mut real_paths,
            &mut diagnostics,
        );
        // Project: .agents/skills walked up to the git root (or
        // filesystem root). We stop at either the first `.git` dir
        // we hit or when `parent()` returns None.
        for ancestor in opts.cwd.ancestors() {
            let candidate = ancestor.join(".agents").join("skills");
            if candidate.is_dir() {
                add(
                    load_skills_from_dir(&candidate, SkillSource::Project, false),
                    &mut skill_map,
                    &mut order,
                    &mut real_paths,
                    &mut diagnostics,
                );
            }
            if ancestor.join(".git").exists() {
                break;
            }
        }
    }

    for raw in &opts.skill_paths {
        let resolved = if raw.is_absolute() {
            raw.clone()
        } else {
            opts.cwd.join(raw)
        };
        if !resolved.exists() {
            diagnostics.push(SkillDiagnostic {
                kind: DiagnosticKind::Warning,
                message: "skill path does not exist".into(),
                path: resolved,
            });
            continue;
        }
        let source = infer_source(&resolved, &opts);
        if resolved.is_dir() {
            add(
                load_skills_from_dir(&resolved, source, true),
                &mut skill_map,
                &mut order,
                &mut real_paths,
                &mut diagnostics,
            );
        } else if resolved.extension().and_then(|s| s.to_str()) == Some("md") {
            let result = load_skill_from_file(&resolved, source);
            add(
                LoadSkillsResult {
                    skills: result.skill.into_iter().collect(),
                    diagnostics: result.diagnostics,
                },
                &mut skill_map,
                &mut order,
                &mut real_paths,
                &mut diagnostics,
            );
        } else {
            diagnostics.push(SkillDiagnostic {
                kind: DiagnosticKind::Warning,
                message: "skill path is not a markdown file".into(),
                path: resolved,
            });
        }
    }

    let skills = order
        .into_iter()
        .filter_map(|name| skill_map.remove(&name))
        .collect();
    LoadSkillsResult { skills, diagnostics }
}

/// Assign a `SkillSource` to a CLI-provided path based on whether it
/// lives under the global or project skill directory tree. Paths
/// outside both get `SkillSource::Path`.
fn infer_source(path: &Path, opts: &LoadSkillsOptions) -> SkillSource {
    let user_roots = [
        opts.agent_dir.join("agent").join("skills"),
        dirs::home_dir()
            .map(|h| h.join(".agents").join("skills"))
            .unwrap_or_else(|| opts.agent_dir.clone()),
    ];
    for root in &user_roots {
        if is_under(path, root) {
            return SkillSource::User;
        }
    }
    let project_roots = [opts.cwd.join(".pi").join("skills")];
    for root in &project_roots {
        if is_under(path, root) {
            return SkillSource::Project;
        }
    }
    // Walk up for .agents/skills
    for ancestor in opts.cwd.ancestors() {
        let candidate = ancestor.join(".agents").join("skills");
        if is_under(path, &candidate) {
            return SkillSource::Project;
        }
    }
    SkillSource::Path
}

fn is_under(path: &Path, root: &Path) -> bool {
    let p = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let r = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    p == r || p.starts_with(&r)
}

/// Scan a directory for skills.
///
/// Discovery rules (match TS):
/// - If `dir/SKILL.md` exists, `dir` *is* the skill — recurse no
///   further.
/// - Otherwise, when `include_root_md_files` is true, each direct
///   `.md` child at the root of `dir` is treated as an individual
///   skill (matches the TS behaviour for `~/.pi/agent/skills/` and
///   `.pi/skills/` where loose markdown files are allowed).
/// - In every case, subdirectories are recursively scanned for
///   `SKILL.md`. Dotfiles and `node_modules/` are skipped.
pub fn load_skills_from_dir(
    dir: &Path,
    source: SkillSource,
    include_root_md_files: bool,
) -> LoadSkillsResult {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    if !dir.is_dir() {
        return LoadSkillsResult {
            skills,
            diagnostics,
        };
    }

    // First: is this directory itself a skill root?
    let skill_md = dir.join("SKILL.md");
    if skill_md.is_file() {
        let result = load_skill_from_file(&skill_md, source);
        if let Some(skill) = result.skill {
            skills.push(skill);
        }
        diagnostics.extend(result.diagnostics);
        return LoadSkillsResult {
            skills,
            diagnostics,
        };
    }

    // Otherwise walk entries.
    let Ok(entries) = fs::read_dir(dir) else {
        return LoadSkillsResult {
            skills,
            diagnostics,
        };
    };

    let mut entry_paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entry_paths.sort();

    for path in &entry_paths {
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let meta = match path.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        if meta.is_dir() {
            // Recurse, but never pick up loose root .md files in
            // sub-levels — only in the top-level entry that this
            // function was called on.
            let sub = load_skills_from_dir(path, source, false);
            skills.extend(sub.skills);
            diagnostics.extend(sub.diagnostics);
        } else if meta.is_file()
            && include_root_md_files
            && path.extension().and_then(|s| s.to_str()) == Some("md")
        {
            let result = load_skill_from_file(path, source);
            if let Some(skill) = result.skill {
                skills.push(skill);
            }
            diagnostics.extend(result.diagnostics);
        }
    }

    LoadSkillsResult {
        skills,
        diagnostics,
    }
}

/// Outcome of reading a single `SKILL.md`-style file.
struct LoadOne {
    skill: Option<Skill>,
    diagnostics: Vec<SkillDiagnostic>,
}

/// Read a single `.md` file, parse its frontmatter, run validation,
/// and return a `Skill` if the description is present. Validation
/// violations become warnings; only a missing description drops the
/// skill.
fn load_skill_from_file(path: &Path, source: SkillSource) -> LoadOne {
    let mut diagnostics = Vec::new();
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            diagnostics.push(SkillDiagnostic {
                kind: DiagnosticKind::Warning,
                message: format!("could not read: {e}"),
                path: path.to_path_buf(),
            });
            return LoadOne {
                skill: None,
                diagnostics,
            };
        }
    };
    let parsed = parse_frontmatter(&raw);
    let skill_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let parent_name = skill_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let description = parsed.string("description").unwrap_or("").trim().to_string();
    for err in validate_description(&description) {
        diagnostics.push(SkillDiagnostic {
            kind: DiagnosticKind::Warning,
            message: err,
            path: path.to_path_buf(),
        });
    }

    // Name: from frontmatter, fall back to parent directory name.
    let name = parsed
        .string("name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| parent_name.clone());

    for err in validate_name(&name, &parent_name) {
        diagnostics.push(SkillDiagnostic {
            kind: DiagnosticKind::Warning,
            message: err,
            path: path.to_path_buf(),
        });
    }

    // Hard error: missing description drops the skill.
    if description.is_empty() {
        return LoadOne {
            skill: None,
            diagnostics,
        };
    }

    let disable_model_invocation = parsed
        .bool("disable-model-invocation")
        .unwrap_or(false);

    let skill = Skill {
        name,
        description,
        file_path: path.to_path_buf(),
        base_dir: skill_dir,
        source,
        disable_model_invocation,
    };
    LoadOne {
        skill: Some(skill),
        diagnostics,
    }
}

/// Validate a skill name against the Agent Skills standard.
fn validate_name(name: &str, parent_dir_name: &str) -> Vec<String> {
    let mut errors = Vec::new();

    if !parent_dir_name.is_empty() && name != parent_dir_name {
        errors.push(format!(
            "name \"{name}\" does not match parent directory \"{parent_dir_name}\""
        ));
    }
    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.chars().count()
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .into(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".into());
    }
    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".into());
    }
    errors
}

/// Validate a skill description.
fn validate_description(description: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if description.is_empty() {
        errors.push("description is required".into());
    } else if description.chars().count() > MAX_DESCRIPTION_LENGTH {
        errors.push(format!(
            "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
            description.chars().count()
        ));
    }
    errors
}

/// Format the visible skills as an `<available_skills>` XML block
/// to append to the system prompt.
///
/// Skills with `disable_model_invocation = true` are excluded — the
/// user can still invoke them via `/skill:<name>`, but the model
/// does not see them.
pub fn format_skills_for_prompt(skills: &[Skill]) -> String {
    let visible: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.disable_model_invocation)
        .collect();
    if visible.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\n\nThe following skills provide specialized instructions for specific tasks.\n");
    out.push_str("Use the read tool to load a skill's file when the task matches its description.\n");
    out.push_str("When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n");
    out.push_str("<available_skills>\n");
    for s in visible {
        out.push_str("  <skill>\n");
        out.push_str(&format!("    <name>{}</name>\n", escape_xml(&s.name)));
        out.push_str(&format!(
            "    <description>{}</description>\n",
            escape_xml(&s.description)
        ));
        out.push_str(&format!(
            "    <location>{}</location>\n",
            escape_xml(&s.file_path.display().to_string())
        ));
        out.push_str("  </skill>\n");
    }
    out.push_str("</available_skills>");
    out
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Expand `/skill:<name> <args>` into the full skill body wrapped in
/// a `<skill>` tag, with relative-path guidance and the user's args
/// appended. Returns `None` if `text` isn't a skill invocation or
/// the named skill isn't in `skills`.
pub fn expand_skill_command(text: &str, skills: &[Skill]) -> Option<String> {
    let rest = text.strip_prefix("/skill:")?;
    let (name, args) = match rest.find(|c: char| c.is_whitespace()) {
        Some(i) => (&rest[..i], rest[i + 1..].trim()),
        None => (rest, ""),
    };
    let skill = skills.iter().find(|s| s.name == name)?;
    let body = crate::frontmatter::strip_frontmatter(&fs::read_to_string(&skill.file_path).ok()?);
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path.display(),
        skill.base_dir.display(),
        body.trim()
    );
    if args.is_empty() {
        Some(block)
    } else {
        Some(format!("{block}\n\n{args}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    fn tmp() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("pi-skills-test-")
            .tempdir()
            .unwrap()
    }

    #[test]
    fn loads_single_skill_with_frontmatter() {
        let dir = tmp();
        let skill_dir = dir.path().join("greet");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: greet\ndescription: Say hi in any language.\n---\n\n# Greet\nSay hi.\n",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert_eq!(r.skills.len(), 1, "skills: {:?}", r.skills);
        let s = &r.skills[0];
        assert_eq!(s.name, "greet");
        assert_eq!(s.description, "Say hi in any language.");
        assert_eq!(s.base_dir, skill_dir);
        assert!(!s.disable_model_invocation);
    }

    #[test]
    fn missing_description_drops_skill_but_warns() {
        let dir = tmp();
        let skill_dir = dir.path().join("noisy");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: noisy\n---\n\n# Body\n",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert!(r.skills.is_empty());
        assert!(r
            .diagnostics
            .iter()
            .any(|d| d.message.contains("description is required")));
    }

    #[test]
    fn name_mismatch_logs_warning_but_loads() {
        let dir = tmp();
        let skill_dir = dir.path().join("actual-name");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: not-the-dir\ndescription: whatever.\n---\nbody",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert_eq!(r.skills.len(), 1);
        assert_eq!(r.skills[0].name, "not-the-dir");
        assert!(r
            .diagnostics
            .iter()
            .any(|d| d.message.contains("does not match parent directory")));
    }

    #[test]
    fn invalid_name_chars_warn() {
        let dir = tmp();
        let skill_dir = dir.path().join("UPPER");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: UPPER\ndescription: x.\n---\n",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert_eq!(r.skills.len(), 1);
        assert!(r
            .diagnostics
            .iter()
            .any(|d| d.message.contains("invalid characters")));
    }

    #[test]
    fn loose_root_markdown_is_a_skill() {
        let dir = tmp();
        // Two top-level loose .md files.
        write_file(
            &dir.path().join("hi.md"),
            "---\nname: hi\ndescription: greet.\n---\nbody",
        );
        write_file(
            &dir.path().join("bye.md"),
            "---\nname: bye\ndescription: farewell.\n---\nbody",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, true);
        let names: Vec<&str> = r.skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"hi"));
        assert!(names.contains(&"bye"));
    }

    #[test]
    fn root_markdown_ignored_when_opted_out() {
        let dir = tmp();
        write_file(
            &dir.path().join("hi.md"),
            "---\nname: hi\ndescription: greet.\n---\nbody",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert!(r.skills.is_empty());
    }

    #[test]
    fn disable_model_invocation_is_parsed() {
        let dir = tmp();
        let skill_dir = dir.path().join("secret");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: secret\ndescription: hidden.\ndisable-model-invocation: true\n---\n",
        );
        let r = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert_eq!(r.skills.len(), 1);
        assert!(r.skills[0].disable_model_invocation);
    }

    #[test]
    fn format_skills_for_prompt_filters_disabled() {
        let s_visible = Skill {
            name: "visible".into(),
            description: "visible skill".into(),
            file_path: PathBuf::from("/s/visible/SKILL.md"),
            base_dir: PathBuf::from("/s/visible"),
            source: SkillSource::User,
            disable_model_invocation: false,
        };
        let s_hidden = Skill {
            name: "hidden".into(),
            description: "hidden skill".into(),
            file_path: PathBuf::from("/s/hidden/SKILL.md"),
            base_dir: PathBuf::from("/s/hidden"),
            source: SkillSource::User,
            disable_model_invocation: true,
        };
        let out = format_skills_for_prompt(&[s_visible, s_hidden]);
        assert!(out.contains("<available_skills>"));
        assert!(out.contains("<name>visible</name>"));
        assert!(!out.contains("<name>hidden</name>"));
    }

    #[test]
    fn format_skills_for_prompt_empty_when_all_hidden() {
        let s = Skill {
            name: "a".into(),
            description: "b".into(),
            file_path: PathBuf::from("/x"),
            base_dir: PathBuf::from("/"),
            source: SkillSource::User,
            disable_model_invocation: true,
        };
        assert_eq!(format_skills_for_prompt(&[s]), "");
    }

    #[test]
    fn format_skills_for_prompt_escapes_xml() {
        let s = Skill {
            name: "ok".into(),
            description: "Uses <tags> & \"quotes\".".into(),
            file_path: PathBuf::from("/x"),
            base_dir: PathBuf::from("/"),
            source: SkillSource::User,
            disable_model_invocation: false,
        };
        let out = format_skills_for_prompt(&[s]);
        assert!(out.contains("&lt;tags&gt;"));
        assert!(out.contains("&amp;"));
        assert!(out.contains("&quot;quotes&quot;"));
    }

    #[test]
    fn expand_skill_command_wraps_body_and_args() {
        let dir = tmp();
        let skill_dir = dir.path().join("greet");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: greet\ndescription: say hi.\n---\nHello there.\n",
        );
        let loaded = load_skills_from_dir(dir.path(), SkillSource::User, false);
        assert_eq!(loaded.skills.len(), 1);
        let expanded = expand_skill_command("/skill:greet to Alice", &loaded.skills).unwrap();
        assert!(expanded.contains("<skill name=\"greet\""));
        assert!(expanded.contains("Hello there."));
        assert!(expanded.ends_with("to Alice"));
    }

    #[test]
    fn expand_skill_command_without_args_has_no_trailing_user_text() {
        let dir = tmp();
        let skill_dir = dir.path().join("greet");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: greet\ndescription: say hi.\n---\nHi.\n",
        );
        let loaded = load_skills_from_dir(dir.path(), SkillSource::User, false);
        let expanded = expand_skill_command("/skill:greet", &loaded.skills).unwrap();
        assert!(expanded.ends_with("</skill>"));
    }

    #[test]
    fn expand_skill_command_unknown_returns_none() {
        let out = expand_skill_command("/skill:missing hi", &[]);
        assert!(out.is_none());
    }

    #[test]
    fn top_level_load_skills_merges_and_dedupes() {
        let root = tmp();
        let cwd = root.path().join("proj");
        let agent = root.path().join("home_pi");
        // Project skill.
        let proj_skill = cwd.join(".pi/skills/greet/SKILL.md");
        write_file(
            &proj_skill,
            "---\nname: greet\ndescription: project version.\n---\n",
        );
        // Global skill with same name.
        let user_skill = agent.join("agent/skills/greet/SKILL.md");
        write_file(
            &user_skill,
            "---\nname: greet\ndescription: user version.\n---\n",
        );
        let r = load_skills(LoadSkillsOptions {
            cwd: cwd.clone(),
            agent_dir: agent.clone(),
            skill_paths: Vec::new(),
            include_defaults: true,
        });
        assert_eq!(r.skills.len(), 1);
        // User is walked first, so the project collision is what
        // gets logged.
        assert_eq!(r.skills[0].description, "user version.");
        assert!(r
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::Collision));
    }

    #[test]
    fn explicit_skill_path_overrides_off_default_scan() {
        let root = tmp();
        let skill_dir = root.path().join("somewhere/greet");
        write_file(
            &skill_dir.join("SKILL.md"),
            "---\nname: greet\ndescription: explicit.\n---\n",
        );
        let r = load_skills(LoadSkillsOptions {
            cwd: root.path().to_path_buf(),
            agent_dir: root.path().to_path_buf(),
            skill_paths: vec![skill_dir.clone()],
            include_defaults: false,
        });
        assert_eq!(r.skills.len(), 1);
        assert_eq!(r.skills[0].name, "greet");
    }

    #[test]
    fn nonexistent_skill_path_warns() {
        let root = tmp();
        let r = load_skills(LoadSkillsOptions {
            cwd: root.path().to_path_buf(),
            agent_dir: root.path().to_path_buf(),
            skill_paths: vec![root.path().join("nope")],
            include_defaults: false,
        });
        assert!(r.skills.is_empty());
        assert!(r
            .diagnostics
            .iter()
            .any(|d| d.message.contains("does not exist")));
    }
}
