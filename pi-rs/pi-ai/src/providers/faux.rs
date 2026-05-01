//! Faux mock provider for testing and development.
//!
//! This provider allows queuing pre-defined responses without hitting real APIs.
//! Useful for testing streaming, tool calls, thinking blocks, and abort handling.

use crate::types::*;
use anyhow::{anyhow, Result};
use futures::stream::{self, Stream};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Helper to create text content
pub fn faux_text(text: impl Into<String>) -> Content {
    Content::Text {
        text: text.into(),
        cache_control: None,
    }
}

/// Helper to create thinking content
pub fn faux_thinking(thinking: impl Into<String>) -> Content {
    Content::Thinking {
        thinking: thinking.into(),
        signature: None,
        cache_control: None,
    }
}

/// Helper to create tool call content
pub fn faux_tool_call(
    name: impl Into<String>,
    input: serde_json::Value,
    id: Option<String>,
) -> Content {
    Content::ToolUse {
        id: id.unwrap_or_else(|| format!("call_{}", uuid())),
        name: name.into(),
        input,
        cache_control: None,
    }
}

/// A pre-defined assistant message response for the faux provider
#[derive(Debug, Clone)]
pub struct FauxResponse {
    pub content: Vec<Content>,
    pub stop_reason: String,
}

impl FauxResponse {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![faux_text(text)],
            stop_reason: "stop".to_string(),
        }
    }

    pub fn thinking(thinking: impl Into<String>) -> Self {
        Self {
            content: vec![faux_thinking(thinking)],
            stop_reason: "stop".to_string(),
        }
    }

    pub fn tool_call(
        name: impl Into<String>,
        input: serde_json::Value,
        id: Option<String>,
    ) -> Self {
        Self {
            content: vec![faux_tool_call(name, input, id)],
            stop_reason: "tool_use".to_string(),
        }
    }

    pub fn with_content(mut self, content: impl Into<Content>) -> Self {
        self.content.push(content.into());
        self
    }

    pub fn with_stop_reason(mut self, reason: impl Into<String>) -> Self {
        self.stop_reason = reason.into();
        self
    }
}

/// Configuration for the Faux provider
#[derive(Debug, Clone)]
pub struct FauxConfig {
    /// Tokens per second for simulated streaming (None = no delay)
    pub tokens_per_second: Option<u32>,
    /// Minimum token chunk size
    pub min_chunk_size: usize,
    /// Maximum token chunk size
    pub max_chunk_size: usize,
}

impl Default for FauxConfig {
    fn default() -> Self {
        Self {
            tokens_per_second: None,
            min_chunk_size: 3,
            max_chunk_size: 5,
        }
    }
}

/// Faux provider state
pub struct FauxProvider {
    config: FauxConfig,
    responses: Arc<Mutex<Vec<FauxResponse>>>,
    call_count: Arc<Mutex<usize>>,
}

impl FauxProvider {
    pub fn new(config: FauxConfig) -> Self {
        Self {
            config,
            responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn add_response(&self, response: FauxResponse) {
        self.responses.lock().await.push(response);
    }

    pub async fn add_responses(&self, responses: Vec<FauxResponse>) {
        self.responses.lock().await.extend(responses);
    }

    pub async fn call_count(&self) -> usize {
        *self.call_count.lock().await
    }

    pub async fn pending_count(&self) -> usize {
        self.responses.lock().await.len()
    }

    pub async fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: &StreamOptions,
    ) -> Result<crate::types::stream::AssistantMessageEventStream> {

/// Stream from the Faux provider (for testing).
/// Returns queued responses in FIFO order from the lazy static provider.
pub async fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<crate::types::stream::AssistantMessageEventStream> {
    // For now, return an error - users should use FauxProvider directly
    Err(anyhow!("Use FauxProvider::new() directly for testing"))
}

        let mut responses = self.responses.lock().await;
        let response = responses.pop().ok_or_else(|| {
            anyhow!("No more faux responses queued. Use faux_provider.add_response() first.")
        })?;

        let mut count = self.call_count.lock().await;
        *count += 1;

        // Convert FauxResponse to stream events
        let events = vec![
            Ok(StreamEvent::Start {
                model: model.id.clone(),
                usage: None,
            }),
        ];

        // Add text/thinking deltas
        let mut events: Vec<Result<StreamEvent>> = events;
        for (idx, content) in response.content.iter().enumerate() {
            match content {
                Content::Text { text, .. } => {
                    for chunk in split_text(text, self.config.min_chunk_size) {
                        events.push(Ok(StreamEvent::TextDelta {
                            delta: chunk,
                            thinking: None,
                        }));
                    }
                }
                Content::Thinking { thinking, .. } => {
                    for chunk in split_text(thinking, self.config.min_chunk_size) {
                        events.push(Ok(StreamEvent::ThinkingDelta {
                            delta: chunk,
                            signature: None,
                        }));
                    }
                }
                Content::ToolUse { id, name, input, .. } => {
                    events.push(Ok(StreamEvent::ToolCallDelta {
                        delta: ToolCallDelta {
                            id: Some(id.clone()),
                            name: Some(name.clone()),
                            input: Some(input.clone()),
                        },
                    }));
                }
                _ => {}
            }
        }

        // Add stop event
        events.push(Ok(StreamEvent::Stop {
            stop_reason: response.stop_reason.clone(),
            stop_sequence: None,
        }));

        let stream = Box::pin(stream::iter(events));
        Ok(stream)
    }
}

/// Split text into chunks for streaming simulation
fn split_text(text: &str, min_size: usize) -> Vec<String> {
    if text.is_empty() {
        return vec!["".to_string()];
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < chars.len() {
        let chunk_size = min_size.min(chars.len() - pos).max(1);
        let end = (pos + chunk_size).min(chars.len());
        let chunk: String = chars[pos..end].iter().collect();
        chunks.push(chunk);
        pos = end;
    }

    if chunks.is_empty() {
        chunks.push(text.to_string());
    }

    chunks
}

fn uuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{:x}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faux_text_response() {
        let response = FauxResponse::text("Hello, world!");
        assert_eq!(response.stop_reason, "stop");
        assert_eq!(response.content.len(), 1);
    }

    #[test]
    fn test_faux_tool_call_response() {
        let response = FauxResponse::tool_call(
            "bash",
            serde_json::json!({"command": "ls -la"}),
            Some("call_123".to_string()),
        );
        assert_eq!(response.stop_reason, "tool_use");
        assert_eq!(response.content.len(), 1);
    }

    #[test]
    fn test_faux_thinking_response() {
        let response = FauxResponse::thinking("Let me think about this...");
        assert_eq!(response.content.len(), 1);
    }

    #[test]
    fn test_split_text() {
        let text = "Hello, world!";
        let chunks = split_text(text, 3);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_split_text_utf8() {
        let text = "你好世界"; // Chinese characters (UTF-8)
        let chunks = split_text(text, 1);
        assert!(!chunks.is_empty());
    }

    #[tokio::test]
    async fn test_faux_provider_creation() {
        let provider = FauxProvider::new(FauxConfig::default());
        assert_eq!(provider.pending_count().await, 0);
        assert_eq!(provider.call_count().await, 0);
    }

    #[tokio::test]
    async fn test_faux_provider_add_response() {
        let provider = FauxProvider::new(FauxConfig::default());
        provider.add_response(FauxResponse::text("Test")).await;
        assert_eq!(provider.pending_count().await, 1);
    }

    #[tokio::test]
    async fn test_faux_provider_stream() {
        let provider = FauxProvider::new(FauxConfig::default());
        provider.add_response(FauxResponse::text("Hello")).await;

        let model = Model {
            id: "faux".to_string(),
            name: "Faux".to_string(),
            api: Api::Faux,
            provider: Provider::Anthropic, // Just for testing
            base_url: None,
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };

        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };

        let options = StreamOptions {
            temperature: None,
            max_tokens: None,
            signal: None,
            api_key: None,
            transport: None,
            cache_retention: None,
            session_id: None,
            headers: None,
            reasoning_effort: None,
            thinking_budgets: None,
        };

        let result = provider.stream(&model, &context, &options).await;
        assert!(result.is_ok());
    }
}
