//! Core types for the pi AI client library.
//!
//! This is a stub implementation matching the TypeScript version's type signatures.
//! Full implementation will be done in Phase 1.1.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// API identifier for each provider implementation.
pub type KnownApi = &'static [&'static str];

/// Provider identifier.
pub type KnownProvider = &'static [&'static str];

/// Thinking level for models that support reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

/// Token budgets for each thinking level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    pub minimal: Option<u32>,
    pub low: Option<u32>,
    pub medium: Option<u32>,
    pub high: Option<u32>,
}

/// Cache retention preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

/// Transport preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    Sse,
    WebSocket,
    Auto,
}

/// Provider response metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

/// Base options shared by all providers.
#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    /// Sampling temperature (0-2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Abort signal for cancellation.
    #[serde(skip)]
    pub signal: Option<CancellationToken>,

    /// API key for authentication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Preferred transport.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,

    /// Prompt cache retention preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,

    /// Session identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Custom HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,

    /// Reasoning effort level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ThinkingLevel>,

    /// Thinking budget override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<ThinkingBudgets>,
}

/// A user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub role: String,
    pub content: Vec<Content>,
    pub id: Option<String>,
}

/// An assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Vec<Content>,
    pub id: Option<String>,
    pub usage: Option<Usage>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// A tool result message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMessage {
    pub role: String,
    pub content: Vec<Content>,
    #[serde(rename = "tool_use_id")]
    pub tool_use_id: String,
    pub is_error: Option<bool>,
}

/// Content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Audio {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Video {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Pdf {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Media source - can be URL or base64-encoded data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    Url {
        url: String,
    },
    Base64 {
        #[serde(rename = "media_type")]
        media_type: String,
        #[serde(rename = "data")]
        data: String,
    },
}

/// Image source for image content blocks (legacy, use MediaSource).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub type_: String,
    pub media_type: String,
    pub data: String,
}

/// Cache control for content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<u32>,
}

/// Tool call representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool result representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Model information and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: Provider,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub reasoning: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multimodal: Option<MultimodalCapabilities>,
}

/// Cost information for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

/// Compatibility flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_choice: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cache_control: Option<bool>,
}

/// Multimodal capabilities for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,        // Image input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<bool>,         // Audio input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<bool>,         // Video input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pdf: Option<bool>,           // PDF input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_generation: Option<bool>, // Image output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_generation: Option<bool>, // Video output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_generation: Option<bool>, // Audio/TTS output
}

/// API identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Api {
    OpenAiCompletions,
    MistralConversations,
    OpenAiResponses,
    AzureOpenAiResponses,
    OpenAiCodexResponses,
    AnthropicMessages,
    BedrockConverseStream,
    GoogleGenerativeAi,
    GoogleGeminiCli,
    GoogleVertex,
    Faux,
    OpenRouterMessages,
    VercelAiGateway,
}

/// Provider identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    AmazonBedrock,
    Anthropic,
    Google,
    GoogleGeminiCli,
    GoogleAntigravity,
    GoogleVertex,
    OpenAi,
    AzureOpenAiResponses,
    OpenAiCodex,
    DeepSeek,
    GitHubCopilot,
    Xai,
    Groq,
    Cerebras,
    OpenRouter,
    VercelAiGateway,
    Zai,
    Mistral,
    Minimax,
    MinimaxCn,
    HuggingFace,
    Fireworks,
    OpenCode,
    OpenCodeGo,
    KimiCoding,
    CloudflareWorkersAi,
}

/// Context for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ToolSchema>>,
}

/// Token usage and cost tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

/// Tool schema for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Unified message enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum Message {
    User(Vec<Content>),
    Assistant(Vec<Content>),
    Tool {
        tool_use_id: String,
        content: Vec<Content>,
        is_error: Option<bool>,
    },
}

/// Streaming event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Start {
        model: String,
        usage: Option<Usage>,
    },
    TextDelta {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
    },
    ThinkingDelta {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolCallDelta {
        delta: ToolCallDelta,
    },
    Usage {
        usage: Usage,
    },
    Stop {
        stop_reason: String,
        stop_sequence: Option<String>,
    },
    Error {
        error: String,
        code: Option<u16>,
    },
}

/// Partial tool call update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<serde_json::Value>,
}

/// Cancellation token.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: tokio_util::sync::CancellationToken,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self { inner: tokio_util::sync::CancellationToken::new() }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn inner(&self) -> &tokio_util::sync::CancellationToken {
        &self.inner
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Stub for message types - full implementation in Phase 1.1
pub mod stream {
    use super::*;

    /// Async stream of stream events.
    pub type AssistantMessageEventStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, anyhow::Error>> + Send>>;

    /// Create a new event stream.
    pub async fn stream(
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
    ) -> Result<AssistantMessageEventStream, anyhow::Error> {
        unimplemented!("streaming will be implemented in Phase 1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_message_serialization() {
        let user_msg = Message::User(vec![Content::Text {
            text: "Hello".into(),
            cache_control: None,
        }]);
        let json = serde_json::to_string(&user_msg).unwrap();
        let _: Message = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_stream_options_serialization() {
        let opts = StreamOptions {
            temperature: Some(0.7),
            max_tokens: Some(1024),
            signal: None,
            api_key: None,
            transport: None,
            cache_retention: None,
            session_id: None,
            headers: None,
            reasoning_effort: None,
            thinking_budgets: None,
        };
        let json = serde_json::to_string(&opts).unwrap();
        // Note: Can't deserialize StreamOptions due to non-serializable signal field
        assert!(!json.is_empty());
    }
}
