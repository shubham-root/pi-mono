//! Anthropic Messages API provider (streaming).
//! Phase 1.3

use crate::types::{
    Model, StreamOptions, StreamEvent, Context, Content,
    Usage, ToolCallDelta,
};
use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures::StreamExt;
use tokio_util::io::StreamReader;
use std::io::ErrorKind;

/// Anthropic request
#[derive(Debug, Serialize)]
struct AnthropicMessageRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(rename = "messages")]
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: AnthropicRole,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum AnthropicRole {
    User,
    Assistant,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text { text: String },
    Image {
        source: AnthropicImageSource,
    },
    Thinking {
        thinking: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    type_: &'static str,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    type_: &'static str,
    budget_tokens: u32,
}

/// SSE event data from Anthropic
#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    #[serde(rename = "type")]
    type_: String,
    message: AnthropicMessageStartInner,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStartInner {
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlockDelta {
    #[serde(rename = "type")]
    type_: String,
    index: u32,
    delta: AnthropicDelta,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    /// Cryptographic signature for an extended-thinking block. Sent after
    /// the last `thinking_delta` for a block; used by the provider to
    /// authenticate the reasoning trace on subsequent requests. We capture
    /// it so callers can pass the signature back unchanged on follow-up
    /// turns (required by Anthropic's extended thinking protocol).
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
    /// Catch-all for future variants. Without this, a new delta type from
    /// the provider breaks the entire stream with an `unknown variant` error
    /// rather than being gracefully ignored.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDelta {
    #[serde(rename = "type")]
    type_: String,
    delta: AnthropicMessageDeltaInner,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDeltaInner {
    /// Optional because some proxies (OpenCode) omit the field; we synthesize
    /// zero in that case rather than hard-failing the whole stream.
    #[serde(default)]
    output_tokens: u32,
    #[serde(rename = "stop_reason")]
    stop_reason: Option<String>,
}

/// Convert internal Content blocks to Anthropic content blocks.
fn convert_content(blocks: &[Content]) -> Vec<AnthropicContentBlock> {
    blocks.iter().map(|b| {
        match b {
            Content::Text { text, .. } => {
                AnthropicContentBlock::Text { text: text.clone() }
            }
            Content::Image { source, .. } => {
                match source {
                    crate::types::MediaSource::Url { url } => {
                        // For URL images, return text representation (Anthropic will fetch)
                        AnthropicContentBlock::Text {
                            text: format!("[Image URL: {}]", url),
                        }
                    }
                    crate::types::MediaSource::Base64 { media_type, data } => {
                        AnthropicContentBlock::Image {
                            source: AnthropicImageSource {
                                type_: "base64",
                                media_type: media_type.clone(),
                                data: data.clone(),
                            },
                        }
                    }
                }
            }
            Content::Audio { source, .. } => {
                match source {
                    crate::types::MediaSource::Url { url } => {
                        AnthropicContentBlock::Text {
                            text: format!("[Audio URL: {}]", url),
                        }
                    }
                    crate::types::MediaSource::Base64 { .. } => {
                        AnthropicContentBlock::Text {
                            text: "[Audio content - not yet supported by Anthropic]".to_string(),
                        }
                    }
                }
            }
            Content::Video { source, .. } => {
                match source {
                    crate::types::MediaSource::Url { url } => {
                        AnthropicContentBlock::Text {
                            text: format!("[Video URL: {}]", url),
                        }
                    }
                    crate::types::MediaSource::Base64 { .. } => {
                        AnthropicContentBlock::Text {
                            text: "[Video content - not yet supported by Anthropic]".to_string(),
                        }
                    }
                }
            }
            Content::Pdf { source, .. } => {
                match source {
                    crate::types::MediaSource::Url { url } => {
                        AnthropicContentBlock::Text {
                            text: format!("[PDF URL: {}]", url),
                        }
                    }
                    crate::types::MediaSource::Base64 { .. } => {
                        AnthropicContentBlock::Text {
                            text: "[PDF content - not yet supported by Anthropic]".to_string(),
                        }
                    }
                }
            }
            Content::Thinking { thinking, .. } => {
                AnthropicContentBlock::Thinking { thinking: thinking.clone() }
            }
            Content::ToolUse { id, name, input, .. } => {
                AnthropicContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }
            }
        }
    }).collect()
}

/// Build request body.
fn build_request(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<AnthropicMessageRequest> {
    // Convert each message
    let mut messages = Vec::new();
    for msg in &context.messages {
        let anthropic_msg = match msg {
            crate::types::Message::User(blocks) => AnthropicMessage {
                role: AnthropicRole::User,
                content: convert_content(blocks),
            },
            crate::types::Message::Assistant(blocks) => AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: convert_content(blocks),
            },
            crate::types::Message::Tool { tool_use_id, content, .. } => {
                // Tool results are represented as user message with tool_result content
                // Collapse tool result content into text representation
                let result_text = content.iter()
                    .filter_map(|c| match c {
                        Content::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![AnthropicContentBlock::Text {
                        text: format!("Tool result for {}: {}", tool_use_id, result_text),
                    }],
                }
            }
        };
        messages.push(anthropic_msg);
    }

    // Convert tools
    let tools = context.tools.as_ref().map(|t| {
        t.iter().map(|schema| AnthropicTool {
            name: schema.name.clone(),
            description: schema.description.clone(),
            input_schema: schema.input_schema.clone(),
        }).collect()
    });

    // Thinking configuration
    let thinking = if model.reasoning {
        let budget = match options.reasoning_effort {
            Some(crate::types::ThinkingLevel::High) => 4096,
            Some(crate::types::ThinkingLevel::Medium) => 2048,
            Some(crate::types::ThinkingLevel::Low) => 1024,
            Some(crate::types::ThinkingLevel::Minimal) => 512,
            Some(crate::types::ThinkingLevel::Xhigh) => 8192,
            None => 2048,
        };
        Some(AnthropicThinking {
            type_: "enabled",
            budget_tokens: budget,
        })
    } else {
        None
    };

    Ok(AnthropicMessageRequest {
        model: model.id.clone(),
        max_tokens: options.max_tokens.unwrap_or(4096),
        system: context.system_prompt.clone(),
        messages,
        tools,
        thinking,
        temperature: options.temperature,
        stream: true,
    })
}

/// Convert a single SSE event to our StreamEvent.
fn convert_event(
    event: &str,
    data: &str,
    model_id: &str,
) -> Result<StreamEvent, anyhow::Error> {
    match event {
        "message_start" => {
            let parsed: AnthropicMessageStart = serde_json::from_str(data)?;
            let usage = Usage {
                input_tokens: parsed.message.usage.input_tokens,
                output_tokens: 0,
                cache_read_tokens: None,
                cache_write_tokens: None,
                total_tokens: Some(parsed.message.usage.input_tokens),
                cost: None,
            };
            Ok(StreamEvent::Start {
                model: model_id.to_string(),
                usage: Some(usage),
            })
        }
        "content_block_delta" => {
            let parsed: AnthropicContentBlockDelta = serde_json::from_str(data)?;
            match parsed.delta {
                AnthropicDelta::TextDelta { text } => {
                    Ok(StreamEvent::TextDelta { delta: text, thinking: None })
                }
                AnthropicDelta::ThinkingDelta { thinking } => {
                    Ok(StreamEvent::ThinkingDelta { delta: thinking, signature: None })
                }
                AnthropicDelta::SignatureDelta { signature } => {
                    // Signature attached to the preceding thinking block.
                    // Surface it via ThinkingDelta so callers can round-trip
                    // the signature back to the provider on follow-up turns.
                    Ok(StreamEvent::ThinkingDelta {
                        delta: String::new(),
                        signature: Some(signature),
                    })
                }
                AnthropicDelta::InputJsonDelta { .. } => {
                    Ok(StreamEvent::ToolCallDelta {
                        delta: ToolCallDelta {
                            id: None,
                            name: None,
                            input: None,
                        },
                    })
                }
                AnthropicDelta::Unknown => {
                    // Forward-compat no-op: unrecognized delta type. Emit an
                    // empty text delta so the stream keeps flowing instead
                    // of erroring out on a newly-added Anthropic variant.
                    Ok(StreamEvent::TextDelta {
                        delta: String::new(),
                        thinking: None,
                    })
                }
            }
        }
        "message_delta" => {
            let parsed: AnthropicMessageDelta = serde_json::from_str(data)?;
            let usage = Usage {
                input_tokens: 0,
                output_tokens: parsed.delta.output_tokens,
                cache_read_tokens: None,
                cache_write_tokens: None,
                total_tokens: None,
                cost: None,
            };
            Ok(StreamEvent::Usage { usage })
        }
        "message_stop" => {
            Ok(StreamEvent::Stop {
                stop_reason: "stop".to_string(),
                stop_sequence: None,
            })
        }
        "error" => {
            Ok(StreamEvent::Error {
                error: data.to_string(),
                code: None,
            })
        }
        _ => {
            // Ignore other events (ping, content_block_start, etc.)
            Ok(StreamEvent::TextDelta {
                delta: String::new(),
                thinking: None,
            })
        }
    }
}

/// Build the `/v1/messages` endpoint URL for an Anthropic-wire-compatible
/// provider. Matches the TypeScript Anthropic SDK's `baseURL` semantics:
/// `base_url` is the API root, SDK appends `/v1/messages`.
///
/// Valid inputs by registry convention:
///   - `https://api.anthropic.com` → `.../v1/messages`
///   - `https://ai-gateway.vercel.sh` → `.../v1/messages`
///   - `https://api.fireworks.ai/inference` → `.../inference/v1/messages`
///   - `https://api.minimax.io/anthropic` → `.../anthropic/v1/messages`
fn build_endpoint_url(base_url: Option<&str>) -> String {
    let base = base_url
        .unwrap_or("https://api.anthropic.com")
        .trim_end_matches('/');
    format!("{}/v1/messages", base)
}

/// Execute streaming request to Anthropic.
pub async fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<crate::types::stream::AssistantMessageEventStream, anyhow::Error> {
    // Build request
    let request = build_request(model, context, options)?;

    // Determine endpoint. Registry convention: `base_url` is the API root
    // that the TypeScript Anthropic SDK treats as `baseURL`; the SDK appends
    // `/v1/messages`. We mirror that here so every anthropic-messages-compat
    // provider (Anthropic, Vercel AI Gateway, Fireworks, GitHub Copilot,
    // MiniMax, Kimi, OpenCode) routes through its own base URL correctly.
    let endpoint = build_endpoint_url(model.base_url.as_deref());

    // API key
    let api_key = options.api_key.as_ref()
        .ok_or_else(|| anyhow!("Anthropic API key required"))?;

    // HTTP client
    let client = Client::new();
    let response = client
        .post(&endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await?;
        return Err(anyhow!("Anthropic API error {}: {}", status, text));
    }

    // Content-Type safety net: see openai.rs for rationale. Providers that
    // proxy Anthropic (Vercel AI Gateway, Fireworks, GitHub Copilot, etc.)
    // sometimes return HTML error pages with 200 OK when an upstream auth
    // token is missing or invalid. Without this guard the SSE decoder sees
    // no data: lines and the caller silently gets an empty response.
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if !content_type.contains("text/event-stream") {
        let body_sample = response
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(512)
            .collect::<String>();
        return Err(anyhow!(
            "Anthropic provider: expected text/event-stream, got '{}'. Body starts with: {}",
            content_type,
            body_sample
        ));
    }

    // Byte stream with error conversion for StreamReader
    let byte_stream = response.bytes_stream().map(|result| {
        result.map_err(|e| std::io::Error::new(ErrorKind::Other, e))
    });
    let stream_reader = StreamReader::new(byte_stream);
    let buf_reader = tokio::io::BufReader::new(stream_reader);

    // Decode SSE
    let sse_stream = crate::stream::decode_sse(buf_reader);

    // Convert to StreamEvent stream
    let model_id = model.id.clone();
    let mapped = sse_stream.map(move |result| {
        match result {
            Ok(sse) => {
                let event_name = sse.event.unwrap_or_default();
                convert_event(&event_name, &sse.data, &model_id)
            }
            Err(e) => Err(anyhow!("SSE decode error: {}", e)),
        }
    }).boxed();

    Ok(Box::pin(mapped) as crate::types::stream::AssistantMessageEventStream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_build_request() {
        let model = Model {
            id: "claude-3".into(),
            name: "Claude 3".into(),
            api: crate::types::Api::AnthropicMessages,
            provider: crate::types::Provider::Anthropic,
            base_url: None,
            reasoning: true,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let context = Context {
            system_prompt: Some("You are helpful".into()),
            messages: Vec::new(),
            tools: None,
        };
        let options = StreamOptions {
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
        let req = build_request(&model, &context, &options).unwrap();
        assert_eq!(req.model, "claude-3");
        assert!(req.thinking.is_some());
    }

    /// Regression: URL construction must match every real registry base_url
    /// shape for providers that speak the Anthropic Messages protocol. All
    /// of these resolve against our single `build_endpoint_url` helper; a
    /// change to it must not break any of them.
    #[test]
    fn endpoint_url_matches_every_anthropic_compat_provider() {
        let cases = [
            (
                "anthropic",
                "https://api.anthropic.com",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "vercel-ai-gateway",
                "https://ai-gateway.vercel.sh",
                "https://ai-gateway.vercel.sh/v1/messages",
            ),
            (
                "fireworks",
                "https://api.fireworks.ai/inference",
                "https://api.fireworks.ai/inference/v1/messages",
            ),
            (
                "github-copilot",
                "https://api.individual.githubcopilot.com",
                "https://api.individual.githubcopilot.com/v1/messages",
            ),
            (
                "minimax",
                "https://api.minimax.io/anthropic",
                "https://api.minimax.io/anthropic/v1/messages",
            ),
            (
                "minimax-cn",
                "https://api.minimaxi.com/anthropic",
                "https://api.minimaxi.com/anthropic/v1/messages",
            ),
            (
                "kimi-coding",
                "https://api.kimi.com/coding",
                "https://api.kimi.com/coding/v1/messages",
            ),
            (
                "opencode",
                "https://opencode.ai/zen",
                "https://opencode.ai/zen/v1/messages",
            ),
        ];
        for (pid, base, expected) in cases {
            let got = build_endpoint_url(Some(base));
            assert_eq!(
                got, expected,
                "provider {pid}: base {base} → expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn endpoint_url_defaults_to_anthropic_when_unset() {
        assert_eq!(
            build_endpoint_url(None),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn endpoint_url_tolerates_trailing_slash() {
        assert_eq!(
            build_endpoint_url(Some("https://api.anthropic.com/")),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            build_endpoint_url(Some("https://ai-gateway.vercel.sh//")),
            "https://ai-gateway.vercel.sh/v1/messages"
        );
    }

    /// Regression: `signature_delta` is emitted by providers proxying
    /// Anthropic extended thinking (notably OpenCode). Previously we
    /// deserialized strictly against a 3-variant enum and hard-failed the
    /// whole stream on any unknown discriminator, so one signature chunk
    /// mid-stream meant the user saw only an error, never the response.
    #[test]
    fn content_block_delta_parses_all_known_variants_and_tolerates_unknown() {
        let text_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
        let thinking_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"because"}}"#;
        let signature_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123"}}"#;
        let input_json_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#;
        let unknown_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"future_variant_that_does_not_exist_yet","payload":"x"}}"#;

        let t = convert_event("content_block_delta", text_delta, "m").unwrap();
        assert!(matches!(t, StreamEvent::TextDelta { .. }));

        let t = convert_event("content_block_delta", thinking_delta, "m").unwrap();
        match t {
            StreamEvent::ThinkingDelta { delta, signature } => {
                assert_eq!(delta, "because");
                assert!(signature.is_none());
            }
            other => panic!("expected ThinkingDelta, got {other:?}"),
        }

        let t = convert_event("content_block_delta", signature_delta, "m").unwrap();
        match t {
            StreamEvent::ThinkingDelta { delta, signature } => {
                assert!(delta.is_empty());
                assert_eq!(signature.as_deref(), Some("abc123"));
            }
            other => panic!("expected ThinkingDelta w/ signature, got {other:?}"),
        }

        let t = convert_event("content_block_delta", input_json_delta, "m").unwrap();
        assert!(matches!(t, StreamEvent::ToolCallDelta { .. }));

        // Unknown variant must NOT return Err — the whole turn would die.
        let t = convert_event("content_block_delta", unknown_delta, "m").unwrap();
        assert!(
            matches!(t, StreamEvent::TextDelta { .. }),
            "unknown delta should degrade to empty TextDelta, got {t:?}"
        );
    }
}
