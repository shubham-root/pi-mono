//! pi-core: Core agent state, settings, model registry, and session management

pub mod agent;
pub mod model_registry;
pub mod settings;
pub mod session;
pub mod system_prompt;
pub mod compaction;

#[cfg(test)]
mod agent_test;

#[cfg(test)]
mod integration_test;

// Re-exports
pub use agent::Agent;
pub use settings::Settings;
pub use session::{Session, SessionManager, SessionId};
pub use system_prompt::SystemPromptBuilder;
