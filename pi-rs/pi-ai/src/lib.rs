//! pi-ai: Multi-provider AI client library
//!
//! This crate provides unified interfaces for streaming and non-streaming
//! completions from various LLM providers (OpenAI, Anthropic, Google, etc.)
//! with tool calling support.

pub mod providers;
pub mod stream;
pub mod transform;
pub mod types;
pub mod generation;

// Re-export core types
pub use types::{
    Message, Content, ToolCall, ToolResult, Model, Context, StreamOptions,
    StreamEvent, ToolSchema, Usage, CacheRetention, Transport, ThinkingLevel,
    Api, Provider, KnownApi, KnownProvider,
};
