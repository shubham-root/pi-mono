//! Session persistence - JSONL format for conversation history.
//! Implements Phase 2.13 with full save/load functionality.

use anyhow::{anyhow, Result};
use pi_ai::types::Message;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Unique session identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Generate a new session ID (UUID v4)
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create from string
    pub fn from_str(s: &str) -> Self {
        Self(s.to_string())
    }

    /// Get as string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Session header entry (first line of JSONL)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub entry_type: String, // "session"
    pub version: u32,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

impl SessionHeader {
    /// Create new session header
    pub fn new(id: &SessionId, cwd: &str, model: &str) -> Self {
        Self {
            entry_type: "session".to_string(),
            version: 1,
            id: id.as_str().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            cwd: cwd.to_string(),
            model: model.to_string(),
            parent_session: None,
        }
    }
}

/// A session entry in JSONL format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    /// Message entry
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        message: Message,
    },
    /// Thinking level change
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        thinking_level: String,
    },
    /// Model change
    #[serde(rename = "model_change")]
    ModelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        model_id: String,
    },
    /// Compaction entry (summarized old messages)
    #[serde(rename = "compaction")]
    Compaction {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        first_kept_entry_id: String,
        tokens_before: u32,
    },
    /// Branch summary
    #[serde(rename = "branch_summary")]
    BranchSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        from_id: String,
        summary: String,
    },
    /// Custom extension entry
    #[serde(rename = "custom")]
    Custom {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        custom_type: String,
        #[serde(flatten)]
        data: serde_json::Value,
    },
}

impl SessionEntry {
    /// Get entry ID
    pub fn id(&self) -> &str {
        match self {
            SessionEntry::Message { id, .. } => id,
            SessionEntry::ThinkingLevelChange { id, .. } => id,
            SessionEntry::ModelChange { id, .. } => id,
            SessionEntry::Compaction { id, .. } => id,
            SessionEntry::BranchSummary { id, .. } => id,
            SessionEntry::Custom { id, .. } => id,
        }
    }

    /// Get entry timestamp
    pub fn timestamp(&self) -> &str {
        match self {
            SessionEntry::Message { timestamp, .. } => timestamp,
            SessionEntry::ThinkingLevelChange { timestamp, .. } => timestamp,
            SessionEntry::ModelChange { timestamp, .. } => timestamp,
            SessionEntry::Compaction { timestamp, .. } => timestamp,
            SessionEntry::BranchSummary { timestamp, .. } => timestamp,
            SessionEntry::Custom { timestamp, .. } => timestamp,
        }
    }

    /// Create a message entry
    pub fn message(message: Message) -> Self {
        Self::Message {
            id: Uuid::new_v4().to_string(),
            parent_id: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
            message,
        }
    }
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
    pub cwd: String,
    pub message_count: usize,
}

/// A complete session with messages and metadata
#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub messages: Vec<Message>,
    pub metadata: SessionMetadata,
    entries: Vec<SessionEntry>,
    file_path: Option<PathBuf>,
}

impl Session {
    /// Create a new session
    pub fn new(model: &str, cwd: &str) -> Self {
        let id = SessionId::new();
        let now = chrono::Utc::now().to_rfc3339();

        Self {
            id,
            messages: Vec::new(),
            metadata: SessionMetadata {
                model: model.to_string(),
                created_at: now.clone(),
                updated_at: now,
                cwd: cwd.to_string(),
                message_count: 0,
            },
            entries: Vec::new(),
            file_path: None,
        }
    }

    /// Get session ID
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// Get all messages
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Add a message to the session
    pub fn add_message(&mut self, msg: Message) {
        self.messages.push(msg.clone());
        let entry = SessionEntry::message(msg);
        self.entries.push(entry);
        self.metadata.message_count += 1;
        self.metadata.updated_at = chrono::Utc::now().to_rfc3339();
    }

    /// Full path to the JSONL file backing this session (None until
    /// `save()` or `append_entry()` is first called with a backing
    /// path in place).
    pub fn file_path(&self) -> Option<&Path> {
        self.file_path.as_deref()
    }

    /// Metadata snapshot (message count, model, cwd, updated_at).
    pub fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    /// All raw entries (messages, model changes, compactions, etc.)
    /// in file order. Lets the TUI replay events verbatim when
    /// resuming.
    pub fn entries(&self) -> &[SessionEntry] {
        &self.entries
    }

    /// Attach a file path without rewriting. Used when hydrating an
    /// existing session on `/resume` so subsequent `append_entry()`
    /// calls land in the same file.
    pub fn set_file_path(&mut self, path: PathBuf) {
        self.file_path = Some(path);
    }

    /// Save session to JSONL file
    pub fn save(&mut self, path: &Path) -> Result<()> {
        // Create parent directories
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut content = String::new();

        // Write header
        let header = SessionHeader::new(&self.id, &self.metadata.cwd, &self.metadata.model);
        content.push_str(&serde_json::to_string(&header)?);
        content.push('\n');

        // Write all entries
        for entry in &self.entries {
            content.push_str(&serde_json::to_string(entry)?);
            content.push('\n');
        }

        fs::write(path, content)?;
        self.file_path = Some(path.to_path_buf());
        Ok(())
    }

    /// Load session from JSONL file
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(anyhow!("Session file not found: {:?}", path));
        }

        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return Err(anyhow!("Empty session file"));
        }

        // Parse header
        let header: SessionHeader = serde_json::from_str(lines[0])?;
        if header.entry_type != "session" {
            return Err(anyhow!("Invalid session header"));
        }

        let mut session = Self {
            id: SessionId::from_str(&header.id),
            messages: Vec::new(),
            metadata: SessionMetadata {
                model: header.model.clone(),
                created_at: header.timestamp.clone(),
                updated_at: header.timestamp,
                cwd: header.cwd,
                message_count: 0,
            },
            entries: Vec::new(),
            file_path: Some(path.to_path_buf()),
        };

        // Parse entries
        for line in &lines[1..] {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<SessionEntry>(line) {
                Ok(entry) => {
                    // Extract messages
                    if let SessionEntry::Message { message, .. } = &entry {
                        session.messages.push(message.clone());
                        session.metadata.message_count += 1;
                    }

                    // Extract model changes
                    if let SessionEntry::ModelChange { model_id, .. } = &entry {
                        session.metadata.model = model_id.clone();
                    }

                    session.entries.push(entry);
                }
                Err(e) => {
                    // Skip malformed lines with warning
                    eprintln!("Skipping malformed session entry: {}", e);
                }
            }
        }

        Ok(session)
    }

    /// Append an entry to the session file (atomic append)
    pub fn append_entry(&mut self, entry: SessionEntry) -> Result<()> {
        self.entries.push(entry.clone());
        self.metadata.updated_at = chrono::Utc::now().to_rfc3339();

        // If we have a file path, append to it
        if let Some(path) = &self.file_path {
            let line = serde_json::to_string(&entry)?;
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?
                .write_all((line + "\n").as_bytes())?;
        }

        Ok(())
    }
}

/// Session manager for creating, listing, and loading sessions
pub struct SessionManager {
    sessions_dir: PathBuf,
}

impl SessionManager {
    /// Create a new session manager rooted at an arbitrary directory.
    pub fn new(sessions_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&sessions_dir)?;
        Ok(Self { sessions_dir })
    }

    /// Create a session manager rooted at the default per-cwd location:
    ///
    ///   ~/.pi/sessions/--<encoded-cwd>--/
    ///
    /// Matches the TypeScript pi layout so each project's sessions are
    /// isolated by working directory. The encoded form replaces path
    /// separators + colons with `-` and wraps the result in `--...--`
    /// for easy recognition in `ls` output.
    pub fn for_cwd(cwd: &Path) -> Result<Self> {
        let base = default_agent_dir();
        let encoded = encode_cwd(cwd);
        let dir = base.join("sessions").join(encoded);
        Self::new(dir)
    }

    /// Find the most recently-modified `.jsonl` session file in the
    /// manager's directory. `None` if there are no sessions yet.
    pub fn most_recent(&self) -> Result<Option<PathBuf>> {
        if !self.sessions_dir.exists() {
            return Ok(None);
        }
        let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
        for entry in fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let mtime = entry
                .metadata()?
                .modified()
                .unwrap_or(std::time::UNIX_EPOCH);
            match &best {
                Some((t, _)) if *t >= mtime => {}
                _ => best = Some((mtime, path)),
            }
        }
        Ok(best.map(|(_, p)| p))
    }

    /// Open a session by file path. Preserves any existing entries and
    /// wires `set_file_path` so subsequent `append_entry` calls go to
    /// the same file.
    pub fn open_path(&self, path: &Path) -> Result<Session> {
        let mut session = Session::load(path)?;
        session.set_file_path(path.to_path_buf());
        Ok(session)
    }

    /// Continue the most recent session in this directory; if there
    /// is none, create a fresh session. The returned session already
    /// has its file path set so `append_entry` persists changes.
    pub fn continue_recent(&self, model: &str, cwd: &str) -> Result<Session> {
        match self.most_recent()? {
            Some(path) => self.open_path(&path),
            None => {
                let mut session = Session::new(model, cwd);
                let filename = format!("{}.jsonl", session.id.as_str());
                let path = self.sessions_dir.join(filename);
                session.save(&path)?;
                Ok(session)
            }
        }
    }

    /// Create a new session, save its header to disk immediately, and
    /// return it. Used by the interactive TUI so the session file
    /// exists as soon as the user opens pi — append-on-turn then
    /// flows through the same file.
    pub fn create_on_disk(&self, model: &str, cwd: &str) -> Result<Session> {
        let mut session = Session::new(model, cwd);
        let filename = format!("{}.jsonl", session.id.as_str());
        let path = self.sessions_dir.join(filename);
        session.save(&path)?;
        Ok(session)
    }

    /// Create a new session and optionally save it
    pub fn create(&self, model: &str, cwd: &str) -> Result<Session> {
        Ok(Session::new(model, cwd))
    }

    /// Create and save a session
    pub fn create_and_save(&self, model: &str, cwd: &str, name: Option<&str>) -> Result<Session> {
        let mut session = Session::new(model, cwd);
        let filename = name
            .map(|n| format!("{}.jsonl", n))
            .unwrap_or_else(|| format!("{}.jsonl", session.id.as_str()));
        let path = self.sessions_dir.join(&filename);
        session.save(&path)?;
        Ok(session)
    }

    /// Load a session from file
    pub fn load(&self, name: &str) -> Result<Session> {
        let path = self.sessions_dir.join(format!("{}.jsonl", name));
        self.open_path(&path)
    }

    /// Load a session by ID
    pub fn load_by_id(&self, id: &SessionId) -> Result<Session> {
        self.load(id.as_str())
    }

    /// List all available sessions as `(metadata, file_path)` pairs,
    /// newest first by last-modified time. Malformed files are
    /// silently skipped so one bad session never blocks the picker.
    pub fn list_with_paths(&self) -> Result<Vec<(SessionMetadata, PathBuf)>> {
        let mut sessions: Vec<(SessionMetadata, PathBuf, std::time::SystemTime)> = Vec::new();
        if !self.sessions_dir.exists() {
            return Ok(Vec::new());
        }
        for entry in fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let mtime = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            match Session::load(&path) {
                Ok(session) => sessions.push((session.metadata, path, mtime)),
                Err(e) => eprintln!("Failed to load session {:?}: {}", path, e),
            }
        }
        sessions.sort_by(|a, b| b.2.cmp(&a.2));
        Ok(sessions.into_iter().map(|(m, p, _)| (m, p)).collect())
    }

    /// List all available sessions (metadata only, newest first).
    pub fn list(&self) -> Result<Vec<SessionMetadata>> {
        Ok(self.list_with_paths()?.into_iter().map(|(m, _)| m).collect())
    }

    /// Delete a session
    pub fn delete(&self, name: &str) -> Result<()> {
        let path = self.sessions_dir.join(format!("{}.jsonl", name));
        fs::remove_file(path)?;
        Ok(())
    }

    /// Get sessions directory
    pub fn dir(&self) -> &Path {
        &self.sessions_dir
    }
}

/// Default root for pi's agent data: `~/.pi` (unix) or
/// `%LOCALAPPDATA%\pi` (windows). Mirrors TS's `getDefaultAgentDir`.
pub fn default_agent_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        return home.join(".pi");
    }
    PathBuf::from(".pi")
}

/// Encode a working directory to a filesystem-safe segment. Matches
/// the TypeScript encoder (`--<path-with-separators-replaced>--`) so
/// sessions stay comparable across implementations.
pub fn encode_cwd(cwd: &Path) -> String {
    let raw = cwd.to_string_lossy();
    let trimmed = raw.trim_start_matches('/').trim_start_matches('\\');
    let safe: String = trimmed
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '-',
            _ => c,
        })
        .collect();
    format!("--{safe}--")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_session_id_creation() {
        let id1 = SessionId::new();
        let id2 = SessionId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_session_new() {
        let session = Session::new("claude-3", "/home/user");
        assert_eq!(session.metadata.model, "claude-3");
        assert_eq!(session.metadata.cwd, "/home/user");
        assert_eq!(session.messages.len(), 0);
    }

    #[test]
    fn test_session_add_message() {
        let mut session = Session::new("gpt-4", "/tmp");
        let msg = Message::User(vec![pi_ai::types::Content::Text {
            text: "Hello".to_string(),
            cache_control: None,
        }]);

        session.add_message(msg);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.metadata.message_count, 1);
    }

    #[test]
    fn test_session_save_and_load() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_file = temp_dir.path().join("test.jsonl");

        // Create and save
        let mut session1 = Session::new("claude-3", "/home/user");
        let msg = Message::User(vec![pi_ai::types::Content::Text {
            text: "Test message".to_string(),
            cache_control: None,
        }]);
        session1.add_message(msg);
        session1.save(&session_file)?;

        // Load
        let session2 = Session::load(&session_file)?;
        assert_eq!(session1.id, session2.id);
        assert_eq!(session1.metadata.model, session2.metadata.model);
        assert_eq!(session1.messages.len(), session2.messages.len());

        Ok(())
    }

    #[test]
    fn test_session_manager_create_and_list() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = SessionManager::new(temp_dir.path().to_path_buf())?;

        // Create sessions
        manager.create_and_save("gpt-4", "/proj1", Some("proj1_session"))?;
        manager.create_and_save("claude-3", "/proj2", Some("proj2_session"))?;

        // List
        let sessions = manager.list()?;
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].model, "claude-3"); // Newest first
        assert_eq!(sessions[1].model, "gpt-4");

        Ok(())
    }

    #[test]
    fn test_session_manager_load_by_name() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = SessionManager::new(temp_dir.path().to_path_buf())?;

        manager.create_and_save("gpt-4", "/home/user", Some("my_session"))?;
        let session = manager.load("my_session")?;

        assert_eq!(session.metadata.model, "gpt-4");
        assert_eq!(session.metadata.cwd, "/home/user");

        Ok(())
    }

    #[test]
    fn test_session_manager_delete() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = SessionManager::new(temp_dir.path().to_path_buf())?;

        manager.create_and_save("gpt-4", "/tmp", Some("to_delete"))?;
        assert_eq!(manager.list()?.len(), 1);

        manager.delete("to_delete")?;
        assert_eq!(manager.list()?.len(), 0);

        Ok(())
    }

    #[test]
    fn test_session_jsonl_format() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let session_file = temp_dir.path().join("format_test.jsonl");

        let mut session = Session::new("gpt-4", "/tmp");
        let msg = Message::User(vec![pi_ai::types::Content::Text {
            text: "Format test".to_string(),
            cache_control: None,
        }]);
        session.add_message(msg);
        session.save(&session_file)?;

        // Read and verify JSONL format
        let content = fs::read_to_string(&session_file)?;
        let lines: Vec<&str> = content.lines().collect();

        // First line should be header
        let header: SessionHeader = serde_json::from_str(lines[0])?;
        assert_eq!(header.entry_type, "session");

        // Second line should be message
        let entry: SessionEntry = serde_json::from_str(lines[1])?;
        match entry {
            SessionEntry::Message { .. } => {}
            _ => panic!("Expected message entry"),
        }

        Ok(())
    }

    #[test]
    fn encode_cwd_matches_expected_shape() {
        let p = PathBuf::from("/Users/me/projects/pi");
        assert_eq!(encode_cwd(&p), "--Users-me-projects-pi--");
    }

    #[test]
    fn continue_recent_falls_back_to_new_when_dir_is_empty() {
        let dir = TempDir::new().unwrap();
        let mgr = SessionManager::new(dir.path().to_path_buf()).unwrap();
        let sess = mgr.continue_recent("m", "/tmp").unwrap();
        assert!(sess.file_path().is_some());
        assert_eq!(sess.messages().len(), 0);
    }

    #[test]
    fn continue_recent_picks_newest_file() {
        let dir = TempDir::new().unwrap();
        let mgr = SessionManager::new(dir.path().to_path_buf()).unwrap();
        let s_old = mgr.create_on_disk("m1", "/a").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let s_new = mgr.create_on_disk("m2", "/a").unwrap();
        let resumed = mgr.continue_recent("fallback-model", "/a").unwrap();
        assert_eq!(resumed.id(), s_new.id());
        assert_ne!(resumed.id(), s_old.id());
    }

    #[test]
    fn list_with_paths_orders_newest_first() {
        let dir = TempDir::new().unwrap();
        let mgr = SessionManager::new(dir.path().to_path_buf()).unwrap();
        let _a = mgr.create_on_disk("m", "/a").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let b = mgr.create_on_disk("m", "/a").unwrap();
        let items = mgr.list_with_paths().unwrap();
        assert_eq!(items.len(), 2);
        assert!(items[0].1.file_name().unwrap().to_string_lossy().contains(b.id().as_str()));
    }
}
