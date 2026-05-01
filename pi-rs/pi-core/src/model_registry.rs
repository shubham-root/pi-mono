//! Model registry: discovers providers and models from TOML files.
//!
//! # Sources (merged in order, later wins)
//!
//! 1. Embedded TOML files from `pi-core/providers/*.toml` (baked in at compile time)
//! 2. Global user TOMLs from `~/.pi/providers/*.toml`
//! 3. Project-local TOMLs from `.pi/providers/*.toml` (relative to cwd)
//!
//! # Extending at runtime
//!
//! To add a new provider or override built-ins without recompiling, drop a
//! `.toml` file into `~/.pi/providers/` or `.pi/providers/` with the same
//! schema as the bundled files. Matching `id` replaces the built-in provider
//! entirely; different `id` adds a new provider.
//!
//! # TOML schema
//!
//! ```toml
//! id = "my-provider"
//! display_name = "My Provider"
//! env_vars = ["MY_PROVIDER_API_KEY"]
//!
//! [[models]]
//! id = "my-model-1"
//! name = "My Model 1"
//! api = "openai-completions"
//! base_url = "https://api.example.com"
//! reasoning = false
//! input = ["text", "image"]
//! context_window = 128000
//! max_tokens = 4096
//! cost = { input = 1.0, output = 3.0, cache_read = 0.1, cache_write = 1.25 }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

include!(concat!(env!("OUT_DIR"), "/providers_index.rs"));

/// Cost per 1M tokens for a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
}

/// A single model entry within a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    /// API identifier (e.g., "anthropic-messages", "openai-completions",
    /// "bedrock-converse-stream"). Consumers map this to a provider
    /// implementation in `pi-ai`.
    pub api: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default = "default_input_modalities")]
    pub input: Vec<String>,
    #[serde(default)]
    pub context_window: u32,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub cost: Option<ModelCost>,
}

fn default_input_modalities() -> Vec<String> {
    vec!["text".to_string()]
}

/// A provider definition parsed from a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub env_vars: Vec<String>,
    #[serde(default)]
    pub models: Vec<Model>,
}

impl Provider {
    /// Return the first env var that has a non-empty value, if any.
    /// Useful for resolving API keys at runtime.
    pub fn resolve_env_key(&self) -> Option<(String, String)> {
        self.resolve_env_key_with(|name| std::env::var(name).ok())
    }

    /// Resolver variant that takes an env lookup closure. Used by tests and
    /// by consumers that want to resolve against a custom environment
    /// (for example `.env` files loaded at startup).
    pub fn resolve_env_key_with<F>(&self, lookup: F) -> Option<(String, String)>
    where
        F: Fn(&str) -> Option<String>,
    {
        for var in &self.env_vars {
            if let Some(value) = lookup(var) {
                if !value.is_empty() {
                    return Some((var.clone(), value));
                }
            }
        }
        None
    }

    pub fn find_model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// Sources a provider was loaded from, for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderSource {
    /// Embedded in the binary via `pi-core/providers/*.toml`.
    Embedded,
    /// Loaded from `~/.pi/providers/*.toml`.
    UserGlobal,
    /// Loaded from `.pi/providers/*.toml` in the current directory.
    Project,
}

#[derive(Debug, Clone)]
struct ProviderEntry {
    provider: Provider,
    source: ProviderSource,
}

/// Registry of all known providers and their models.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    /// Ordered by provider id for stable display.
    providers: BTreeMap<String, ProviderEntry>,
}

#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    pub embedded_count: usize,
    pub user_global_count: usize,
    pub project_count: usize,
    pub errors: Vec<String>,
}

impl ModelRegistry {
    /// Load from embedded TOMLs + user globals (`~/.pi/providers/`) +
    /// project-local (`.pi/providers/`). Errors are collected rather than
    /// fatal so a broken user file cannot disable the registry.
    pub fn load() -> (Self, LoadReport) {
        let mut report = LoadReport::default();
        let mut providers: BTreeMap<String, ProviderEntry> = BTreeMap::new();

        // 1. Embedded (lowest precedence)
        for (filename, content) in EMBEDDED_PROVIDERS {
            match toml::from_str::<Provider>(content) {
                Ok(p) => {
                    providers.insert(
                        p.id.clone(),
                        ProviderEntry {
                            provider: p,
                            source: ProviderSource::Embedded,
                        },
                    );
                    report.embedded_count += 1;
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("embedded {}: {}", filename, e));
                }
            }
        }

        // 2. User global (`~/.pi/providers/`)
        if let Some(home) = dirs::home_dir() {
            let dir = home.join(".pi").join("providers");
            report.user_global_count += load_dir(&dir, ProviderSource::UserGlobal, &mut providers, &mut report);
        }

        // 3. Project-local (`.pi/providers/` relative to cwd)
        if let Ok(cwd) = std::env::current_dir() {
            let dir = cwd.join(".pi").join("providers");
            report.project_count += load_dir(&dir, ProviderSource::Project, &mut providers, &mut report);
        }

        (Self { providers }, report)
    }

    /// Return a reference to the process-wide registry (loaded once).
    pub fn global() -> &'static ModelRegistry {
        static INSTANCE: OnceLock<ModelRegistry> = OnceLock::new();
        INSTANCE.get_or_init(|| Self::load().0)
    }

    pub fn providers(&self) -> impl Iterator<Item = &Provider> {
        self.providers.values().map(|e| &e.provider)
    }

    pub fn provider(&self, id: &str) -> Option<&Provider> {
        self.providers.get(id).map(|e| &e.provider)
    }

    pub fn provider_source(&self, id: &str) -> Option<ProviderSource> {
        self.providers.get(id).map(|e| e.source)
    }

    /// Find a model by id. If multiple providers share an id the first (by
    /// provider id alphabetical order) wins; callers that care should pass
    /// the provider id explicitly via [`Self::find_by_provider`].
    pub fn find_model(&self, id: &str) -> Option<(&Provider, &Model)> {
        for entry in self.providers.values() {
            if let Some(model) = entry.provider.find_model(id) {
                return Some((&entry.provider, model));
            }
        }
        None
    }

    pub fn find_by_provider(&self, provider_id: &str, model_id: &str) -> Option<(&Provider, &Model)> {
        let entry = self.providers.get(provider_id)?;
        let model = entry.provider.find_model(model_id)?;
        Some((&entry.provider, model))
    }

    /// All models across providers, each paired with its owning provider.
    pub fn all_models(&self) -> impl Iterator<Item = (&Provider, &Model)> {
        self.providers
            .values()
            .flat_map(|e| e.provider.models.iter().map(move |m| (&e.provider, m)))
    }

    pub fn total_models(&self) -> usize {
        self.providers.values().map(|e| e.provider.models.len()).sum()
    }

    pub fn total_providers(&self) -> usize {
        self.providers.len()
    }

    /// Resolve API key for a provider from env, returning `(env_var, value)`.
    pub fn resolve_env_key(&self, provider_id: &str) -> Option<(String, String)> {
        self.provider(provider_id)?.resolve_env_key()
    }
}

fn load_dir(
    dir: &Path,
    source: ProviderSource,
    providers: &mut BTreeMap<String, ProviderEntry>,
    report: &mut LoadReport,
) -> usize {
    if !dir.is_dir() {
        return 0;
    }
    let mut count = 0;
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) => {
            report.errors.push(format!("read_dir {:?}: {}", dir, e));
            return 0;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                report.errors.push(format!("entry {:?}: {}", dir, e));
                continue;
            }
        };
        let path: PathBuf = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                report.errors.push(format!("read {:?}: {}", path, e));
                continue;
            }
        };
        match toml::from_str::<Provider>(&content) {
            Ok(p) => {
                providers.insert(
                    p.id.clone(),
                    ProviderEntry {
                        provider: p,
                        source,
                    },
                );
                count += 1;
            }
            Err(e) => {
                report.errors.push(format!("parse {:?}: {}", path, e));
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_loads() {
        let (registry, report) = ModelRegistry::load();
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        assert!(registry.total_providers() >= 20, "providers: {}", registry.total_providers());
        assert!(registry.total_models() >= 500, "models: {}", registry.total_models());
    }

    #[test]
    fn anthropic_provider_present() {
        let (registry, _) = ModelRegistry::load();
        let p = registry.provider("anthropic").expect("anthropic");
        assert_eq!(p.display_name, "Anthropic");
        assert!(p.env_vars.iter().any(|v| v == "ANTHROPIC_API_KEY"));
        assert!(!p.models.is_empty());
    }

    #[test]
    fn find_model_claude() {
        let (registry, _) = ModelRegistry::load();
        let (provider, model) = registry
            .find_by_provider("anthropic", "claude-opus-4-7")
            .expect("claude-opus-4-7");
        assert_eq!(provider.id, "anthropic");
        assert_eq!(model.id, "claude-opus-4-7");
        assert_eq!(model.api, "anthropic-messages");
    }

    #[test]
    fn env_vars_configured_for_known_providers() {
        let (registry, _) = ModelRegistry::load();
        let checks = [
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("google", "GEMINI_API_KEY"),
            ("mistral", "MISTRAL_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("xai", "XAI_API_KEY"),
        ];
        for (provider, expected_env) in checks {
            let p = registry.provider(provider).unwrap_or_else(|| panic!("{provider} missing"));
            assert!(
                p.env_vars.iter().any(|v| v == expected_env),
                "{provider} should list {expected_env}, got {:?}",
                p.env_vars
            );
        }
    }

    #[test]
    fn provider_source_is_embedded_by_default() {
        let (registry, _) = ModelRegistry::load();
        assert_eq!(registry.provider_source("anthropic"), Some(ProviderSource::Embedded));
    }

    #[test]
    fn custom_provider_toml_parses() {
        let toml = r#"
id = "acme-inference"
display_name = "ACME Inference"
env_vars = ["ACME_API_KEY"]

[[models]]
id = "acme-fast-v1"
name = "ACME Fast v1"
api = "openai-completions"
base_url = "https://api.acme.example"
reasoning = false
input = ["text", "image"]
context_window = 128000
max_tokens = 8192
cost = { input = 1.0, output = 3.0, cache_read = 0.1, cache_write = 1.25 }

[[models]]
id = "acme-reason-v1"
name = "ACME Reason v1"
api = "openai-completions"
reasoning = true
context_window = 200000
max_tokens = 32000
"#;
        let p: Provider = toml::from_str(toml).expect("valid toml");
        assert_eq!(p.id, "acme-inference");
        assert_eq!(p.display_name, "ACME Inference");
        assert_eq!(p.env_vars, vec!["ACME_API_KEY".to_string()]);
        assert_eq!(p.models.len(), 2);

        let fast = p.find_model("acme-fast-v1").expect("fast model");
        assert_eq!(fast.context_window, 128000);
        assert!(!fast.reasoning);
        assert_eq!(fast.input, vec!["text".to_string(), "image".to_string()]);
        let cost = fast.cost.as_ref().expect("cost");
        assert_eq!(cost.input, 1.0);
        assert_eq!(cost.cache_write, 1.25);

        let reason = p.find_model("acme-reason-v1").expect("reason model");
        assert!(reason.reasoning);
        assert!(reason.cost.is_none());
        // default modalities applied
        assert_eq!(reason.input, vec!["text".to_string()]);
    }

    #[test]
    fn model_entry_rejects_missing_required_fields() {
        // missing id
        let bad = r#"
display_name = "Bad"
env_vars = []

[[models]]
name = "no id"
api = "openai-completions"
"#;
        assert!(toml::from_str::<Provider>(bad).is_err());
    }

    #[test]
    fn resolve_env_key_with_picks_first_populated_var() {
        let provider = Provider {
            id: "test".into(),
            display_name: "Test".into(),
            env_vars: vec![
                "PRIMARY".to_string(),
                "FALLBACK".to_string(),
            ],
            models: vec![],
        };

        // Nothing set -> None.
        assert!(provider.resolve_env_key_with(|_| None).is_none());

        // Empty string is ignored (matches real env semantics where unset
        // and empty are effectively the same for credential purposes).
        let empty_primary = |name: &str| match name {
            "PRIMARY" => Some(String::new()),
            "FALLBACK" => Some("fallback-value".to_string()),
            _ => None,
        };
        let (which, val) = provider
            .resolve_env_key_with(empty_primary)
            .expect("fallback resolves when primary is empty");
        assert_eq!(which, "FALLBACK");
        assert_eq!(val, "fallback-value");

        // Both set -> first wins.
        let both = |name: &str| match name {
            "PRIMARY" => Some("p".to_string()),
            "FALLBACK" => Some("f".to_string()),
            _ => None,
        };
        let (which, val) = provider.resolve_env_key_with(both).expect("primary wins");
        assert_eq!(which, "PRIMARY");
        assert_eq!(val, "p");

        // Only fallback populated.
        let only_fallback = |name: &str| (name == "FALLBACK").then(|| "f".to_string());
        let (which, val) = provider
            .resolve_env_key_with(only_fallback)
            .expect("fallback alone resolves");
        assert_eq!(which, "FALLBACK");
        assert_eq!(val, "f");
    }
}
