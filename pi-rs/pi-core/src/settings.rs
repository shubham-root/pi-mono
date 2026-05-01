//! Settings manager - load and merge global and project settings.
//! Implements Phase 2.10 with TOML config support.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Settings structure - all optional, allowing partial overrides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Default model ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Thinking level (minimal, low, medium, high, xhigh)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    /// Custom system prompt override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Enable/disable tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,

    /// List of plugins to load
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugins: Option<Vec<String>>,

    /// Temperature for sampling (0.0 - 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Max tokens per response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Custom settings (extension-specific)
    #[serde(default)]
    pub custom: HashMap<String, toml::Value>,
}

impl Settings {
    /// Create empty settings
    pub fn new() -> Self {
        Self {
            model: None,
            thinking: None,
            system_prompt: None,
            tools: None,
            plugins: None,
            temperature: None,
            max_tokens: None,
            custom: HashMap::new(),
        }
    }

    /// Load global settings from ~/.pi/config.toml
    pub fn load_global() -> Result<Self> {
        let config_path = Self::global_config_path()?;
        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            Ok(Self::new())
        }
    }

    /// Load project settings from .pi/config.toml
    pub fn load_project(cwd: &Path) -> Result<Self> {
        let project_path = cwd.join(".pi/config.toml");
        if project_path.exists() {
            Self::load_from_file(&project_path)
        } else {
            Ok(Self::new())
        }
    }

    /// Load settings from a specific file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let content = fs::read_to_string(path)?;
        let settings: Settings = toml::from_str(&content)?;
        Ok(settings)
    }

    /// Deep merge - other overrides self
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            model: other.model.clone().or_else(|| self.model.clone()),
            thinking: other.thinking.clone().or_else(|| self.thinking.clone()),
            system_prompt: other.system_prompt.clone().or_else(|| self.system_prompt.clone()),
            tools: other.tools.or(self.tools),
            plugins: other.plugins.clone().or_else(|| self.plugins.clone()),
            temperature: other.temperature.or(self.temperature),
            max_tokens: other.max_tokens.or(self.max_tokens),
            custom: {
                let mut merged = self.custom.clone();
                merged.extend(other.custom.clone());
                merged
            },
        }
    }

    /// Save settings to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directories
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let toml_string = toml::to_string_pretty(self)?;
        fs::write(path, toml_string)?;
        Ok(())
    }

    /// Save to global config location
    pub fn save_global(&self) -> Result<()> {
        let config_path = Self::global_config_path()?;
        self.save(&config_path)
    }

    /// Save to project config location
    pub fn save_project(&self, cwd: &Path) -> Result<()> {
        let project_path = cwd.join(".pi/config.toml");
        self.save(&project_path)
    }

    /// Get global config path (~/.config/pi/config.toml on Linux/macOS, or platform equivalent)
    pub fn global_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow!("Could not determine config directory"))?;
        Ok(config_dir.join("pi").join("config.toml"))
    }

    /// Get all settings with defaults applied
    pub fn with_defaults(&self) -> Self {
        Self {
            model: self.model.clone().or_else(|| Some("claude-3-sonnet".to_string())),
            thinking: self.thinking.clone().or_else(|| Some("low".to_string())),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.or(Some(true)),
            plugins: self.plugins.clone(),
            temperature: self.temperature.or(Some(0.7)),
            max_tokens: self.max_tokens.or(Some(4096)),
            custom: self.custom.clone(),
        }
    }

    /// Load and merge: global + project settings
    pub fn load_merged(cwd: &Path) -> Result<Self> {
        let global = Self::load_global()?;
        let project = Self::load_project(cwd)?;
        Ok(global.merge(&project))
    }

    /// Load and merge with defaults
    pub fn load_merged_with_defaults(cwd: &Path) -> Result<Self> {
        Ok(Self::load_merged(cwd)?.with_defaults())
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_settings_new() {
        let settings = Settings::new();
        assert_eq!(settings.model, None);
        assert_eq!(settings.thinking, None);
        assert_eq!(settings.tools, None);
    }

    #[test]
    fn test_settings_with_defaults() {
        let settings = Settings::new();
        let with_defaults = settings.with_defaults();
        assert_eq!(with_defaults.model, Some("claude-3-sonnet".to_string()));
        assert_eq!(with_defaults.thinking, Some("low".to_string()));
        assert_eq!(with_defaults.tools, Some(true));
        assert_eq!(with_defaults.temperature, Some(0.7));
    }

    #[test]
    fn test_settings_merge() {
        let settings1 = Settings {
            model: Some("gpt-4".to_string()),
            thinking: Some("low".to_string()),
            temperature: Some(0.5),
            ..Settings::new()
        };

        let settings2 = Settings {
            model: Some("claude-3".to_string()),
            temperature: Some(0.8),
            ..Settings::new()
        };

        let merged = settings1.merge(&settings2);
        assert_eq!(merged.model, Some("claude-3".to_string())); // settings2 overrides
        assert_eq!(merged.thinking, Some("low".to_string())); // from settings1
        assert_eq!(merged.temperature, Some(0.8)); // settings2 overrides
    }

    #[test]
    fn test_settings_save_and_load() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let settings_file = temp_dir.path().join("config.toml");

        let settings = Settings {
            model: Some("gpt-4".to_string()),
            thinking: Some("high".to_string()),
            tools: Some(true),
            temperature: Some(0.7),
            ..Settings::new()
        };

        settings.save(&settings_file)?;
        let loaded = Settings::load_from_file(&settings_file)?;

        assert_eq!(loaded.model, settings.model);
        assert_eq!(loaded.thinking, settings.thinking);
        assert_eq!(loaded.tools, settings.tools);
        assert_eq!(loaded.temperature, settings.temperature);

        Ok(())
    }

    #[test]
    fn test_settings_load_nonexistent() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let settings_file = temp_dir.path().join("nonexistent.toml");

        let settings = Settings::load_from_file(&settings_file)?;
        assert_eq!(settings.model, None);

        Ok(())
    }

    #[test]
    fn test_settings_toml_format() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let settings_file = temp_dir.path().join("config.toml");

        let settings = Settings {
            model: Some("claude-3".to_string()),
            thinking: Some("medium".to_string()),
            tools: Some(true),
            plugins: Some(vec!["plugin1".to_string(), "plugin2".to_string()]),
            temperature: Some(0.75),
            ..Settings::new()
        };

        settings.save(&settings_file)?;

        // Verify TOML format
        let content = fs::read_to_string(&settings_file)?;
        assert!(content.contains("model = \"claude-3\""));
        assert!(content.contains("thinking = \"medium\""));
        assert!(content.contains("tools = true"));
        assert!(content.contains("temperature = 0.75"));

        Ok(())
    }

    #[test]
    fn test_settings_merge_with_empty() {
        let settings = Settings {
            model: Some("gpt-4".to_string()),
            temperature: Some(0.5),
            ..Settings::new()
        };

        let empty = Settings::new();
        let merged = settings.merge(&empty);

        // Empty doesn't override
        assert_eq!(merged.model, Some("gpt-4".to_string()));
        assert_eq!(merged.temperature, Some(0.5));
    }
}
