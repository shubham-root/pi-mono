//! OpenAI Chat Completions provider (streaming).
//! Phase 1.4

use crate::types::{
    Model, StreamOptions, StreamEvent, Context, Content, ToolSchema,
    Usage, ToolCallDelta,
};
use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures::StreamExt;
use tokio_util::io::StreamReader;
use std::io::ErrorKind;
use std::collections::HashMap;

/// OpenAI request
#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(rename = "stream_options")]
    stream_options: Option<OpenAIStreamOptions>,
}

/// OpenAI-wire chat message. The variants cover the full spec instead of
/// the toy text-only shape the first version shipped with.
///
/// * `system` / `user` / `assistant` with text content.
/// * `assistant` that invoked tools: `content` is `null`, `tool_calls`
///   carries id + name + JSON-stringified arguments.
/// * `tool`: the result of a tool invocation, linked back to the original
///   call by `tool_call_id`.
///
/// Previous implementation dropped `ToolUse` content blocks from the
/// assistant message and replaced every tool result with the literal
/// string `"(tool result)"`. That made the model blind to its own tool
/// calls and forced it to retry indefinitely, hitting `Max turns exceeded`.
#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum OpenAIMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        /// Serialized as JSON `null` when the assistant only emitted tool
        /// calls with no accompanying text. OpenAI rejects assistant
        /// messages with an empty string content when tool_calls is
        /// present on some compat backends, and accepts `null` universally.
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<OpenAIAssistantToolCall>>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
struct OpenAIAssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAIAssistantFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAIAssistantFunctionCall {
    name: String,
    /// OpenAI wire expects a JSON *string* here, not a parsed object.
    /// Example: `"{\"command\":\"pwd\"}"`.
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    r#type: &'static str,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OpenAIStreamOptions {
    include_usage: bool,
}

/// OpenAI SSE chunk
#[derive(Debug, Deserialize)]
struct OpenAIChatChunk {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    choices: Option<Vec<OpenAIChoice>>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    index: u32,
    delta: Option<OpenAIDelta>,
    finish_reason: Option<String>,
    #[serde(rename = "logprobs")]
    logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
    role: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
    /// Chain-of-thought content. The field name is not standardized across
    /// OpenAI-compatible providers:
    ///   - OpenAI / Azure: `reasoning_content`
    ///   - OpenRouter / Groq / xAI (some): `reasoning`
    /// Accept both aliases so reasoning traces surface regardless of route.
    #[serde(alias = "reasoning")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCallDelta {
    index: u32,
    id: Option<String>,
    r#type: Option<String>,
    function: Option<OpenAIFunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Bring a tool-call id into the character set / length range
/// every OpenAI-compatible backend accepts. Rules:
///
/// - OpenAI Responses API (which OpenRouter sometimes routes to for
///   Azure-hosted models) caps `call_id` at 64 chars and enforces
///   `[a-zA-Z0-9_-]`. Anything longer triggers `string_above_max_length`;
///   anything with `|`, `+`, `/`, `=` (as the OpenAI Codex / Responses
///   pipe format often does) can be silently dropped during
///   translation and surface as "No tool call found for function call
///   output with call_id X".
/// - Pipe-separated `call_abc|<long base64>` ids collapse to the
///   `call_abc` portion, matching the TS `normalizeToolCallId`
///   helper in `packages/ai/src/providers/openai-completions.ts`.
/// - Everything else is truncated to 40 chars to stay inside even
///   the strictest Chat Completions implementations.
///
/// Sanitization always swaps disallowed characters for `_` so two
/// different upstream ids can't accidentally collapse onto the same
/// normalized id (it keeps the original run length).
fn normalize_tool_call_id(id: &str) -> String {
    let base = if let Some((head, _)) = id.split_once('|') {
        head
    } else {
        id
    };
    let sanitized: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.chars().count() > 40 {
        sanitized.chars().take(40).collect()
    } else {
        sanitized
    }
}

fn build_request(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<OpenAIChatRequest> {
    let mut openai_messages = Vec::new();

    // System message first if present.
    if let Some(sys) = &context.system_prompt {
        openai_messages.push(OpenAIMessage::System {
            content: sys.clone(),
        });
    }

    // Conversation messages. Keep tool_use / tool_result blocks intact so
    // the model sees its own previous tool calls and their results; without
    // this the agent loops forever until `Max turns exceeded`.
    // Normalize tool-call ids once, up-front. OpenAI-compatible
    // providers (and crucially some of the backends OpenRouter
    // reaches, like Azure Responses API) enforce a 64-char cap on
    // `call_id` and a `[a-zA-Z0-9_-]` charset. Upstream tool-call
    // ids sometimes blow past both — e.g. OpenAI Responses emits
    // pipe-separated `call_abc|<400+ base64 chars>` ids, Anthropic
    // uses `toolu_...`, some aggregator-proxied models send
    // `tooluse_...`. If we echo them back verbatim, Azure rejects
    // the request with either "string too long" or "no tool call
    // found for function call output with call_id X" (when the
    // translation layer drops the malformed id).
    //
    // Build the mapping across assistant tool_uses first so the
    // matching tool_results get rewritten to the same normalized
    // id — otherwise the pairing breaks. Mirrors the
    // `normalizeToolCallId` pass in `openai-completions.ts`.
    let mut id_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in &context.messages {
        if let crate::types::Message::Assistant(blocks) = msg {
            for c in blocks {
                if let Content::ToolUse { id, .. } = c {
                    let normalized = normalize_tool_call_id(id);
                    if normalized != *id {
                        id_map.insert(id.clone(), normalized);
                    }
                }
            }
        }
    }

    for msg in &context.messages {
        match msg {
            crate::types::Message::User(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                openai_messages.push(OpenAIMessage::User { content: text });
            }
            crate::types::Message::Assistant(blocks) => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<OpenAIAssistantToolCall> = Vec::new();
                for c in blocks {
                    match c {
                        Content::Text { text, .. } => text_parts.push(text.clone()),
                        Content::ToolUse {
                            id, name, input, ..
                        } => {
                            let resolved_id = id_map
                                .get(id)
                                .cloned()
                                .unwrap_or_else(|| id.clone());
                            tool_calls.push(OpenAIAssistantToolCall {
                                id: resolved_id,
                                kind: "function",
                                function: OpenAIAssistantFunctionCall {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                        // Thinking blocks are not part of the OpenAI wire
                        // format for assistant turns; omit them. They are
                        // used only for display at the caller side.
                        _ => {}
                    }
                }
                let joined = text_parts.join("\n");
                let content = if joined.is_empty() {
                    None
                } else {
                    Some(joined)
                };
                let tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };
                openai_messages.push(OpenAIMessage::Assistant {
                    content,
                    tool_calls,
                });
            }
            crate::types::Message::Tool {
                tool_use_id,
                content,
                ..
            } => {
                // OpenAI `tool` messages carry the raw tool output linked
                // back to the assistant's `tool_calls[*].id` via
                // `tool_call_id`. Joining Text blocks keeps multi-part
                // results readable without inventing a JSON envelope.
                let text = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                openai_messages.push(OpenAIMessage::Tool {
                    tool_call_id: id_map
                        .get(tool_use_id)
                        .cloned()
                        .unwrap_or_else(|| tool_use_id.clone()),
                    content: text,
                });
            }
        }
    }

    // Tools
    let tools = context.tools.as_ref().map(|t| {
        t.iter().map(|schema| OpenAITool {
            r#type: "function",
            function: OpenAIFunction {
                name: schema.name.clone(),
                description: schema.description.clone(),
                parameters: schema.input_schema.clone(),
            },
        }).collect()
    });

    Ok(OpenAIChatRequest {
        model: model.id.clone(),
        messages: openai_messages,
        tools,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        stream: true,
        stream_options: Some(OpenAIStreamOptions { include_usage: true }),
    })
}

/// Convert OpenAI chunk to StreamEvent.
/// Accumulator for partial tool calls streamed across multiple chunks.
///
/// OpenAI-compatible providers send `tool_calls` as deltas keyed by a stable
/// `index`. The first delta for a given index typically carries the `id`
/// and `function.name`; subsequent deltas append to `function.arguments`
/// one token at a time until the final chunk (finish_reason == "tool_calls"
/// or "stop") arrives with nothing more to add. Only then can we parse the
/// accumulated args JSON and emit a complete [`ToolCallDelta`].
///
/// We keep insertion order (not just index) so downstream consumers see
/// parallel tool calls in the order the model emitted them. That matters
/// for providers that don't use monotonically increasing `index` values
/// (some OpenRouter-proxied backends reuse indices across turns).
#[derive(Default)]
pub(crate) struct ToolCallAcc {
    /// index -> (insertion_order, id, name, args_partial)
    by_index: HashMap<u32, (usize, String, String, String)>,
    next_seq: usize,
}

impl ToolCallAcc {
    fn record(&mut self, tc: &OpenAIToolCallDelta) {
        let seq = self.next_seq;
        let entry = self.by_index.entry(tc.index).or_insert_with(|| {
            self.next_seq += 1;
            (seq, String::new(), String::new(), String::new())
        });
        if let Some(id) = &tc.id {
            // Providers that include id on every delta: keep the first
            // non-empty value; later frames may be empty strings.
            if entry.1.is_empty() && !id.is_empty() {
                entry.1 = id.clone();
            }
        }
        if let Some(f) = &tc.function {
            if let Some(name) = &f.name {
                if entry.2.is_empty() && !name.is_empty() {
                    entry.2 = name.clone();
                }
            }
            if let Some(args) = &f.arguments {
                entry.3.push_str(args);
            }
        }
    }

    fn drain_sorted(&mut self) -> Vec<(String, String, String)> {
        let mut rows: Vec<_> = std::mem::take(&mut self.by_index)
            .into_iter()
            .map(|(_idx, (seq, id, name, args))| (seq, id, name, args))
            .collect();
        rows.sort_by_key(|(seq, _, _, _)| *seq);
        rows.into_iter()
            .map(|(_, id, name, args)| (id, name, args))
            .collect()
    }
}

fn convert_chunk(
    chunk: &OpenAIChatChunk,
    model_id: &str,
    tool_call_accumulator: &mut ToolCallAcc,
) -> Result<Vec<StreamEvent>, anyhow::Error> {
    let mut events = Vec::new();

    if let Some(choices) = &chunk.choices {
        for choice in choices {
            if let Some(delta) = &choice.delta {
                // Reasoning content (if supported)
                if let Some(reasoning) = &delta.reasoning_content {
                    events.push(StreamEvent::ThinkingDelta {
                        delta: reasoning.clone(),
                        signature: None,
                    });
                }

                // Text content
                if let Some(text) = &delta.content {
                    events.push(StreamEvent::TextDelta {
                        delta: text.clone(),
                        thinking: None,
                    });
                }

                // Tool calls (partial). Each delta contributes to the
                // accumulator; final ToolCallDelta events are emitted when
                // finish_reason arrives.
                if let Some(tool_calls) = &delta.tool_calls {
                    for tc in tool_calls {
                        tool_call_accumulator.record(tc);
                    }
                }

                // Finish reason
                if let Some(reason) = &choice.finish_reason {
                    // Emit assembled tool calls regardless of reason: some
                    // providers finalize with "stop" even when tool_calls
                    // are present (observed on OpenRouter-proxied models).
                    let rows = tool_call_accumulator.drain_sorted();
                    for (id, name, args) in rows {
                        if name.is_empty() {
                            // No function name → provider emitted an empty
                            // tool_calls entry. Skip rather than feeding
                            // garbage into the agent.
                            continue;
                        }
                        let input = if args.trim().is_empty() {
                            serde_json::Value::Object(serde_json::Map::new())
                        } else {
                            match serde_json::from_str::<serde_json::Value>(&args) {
                                Ok(v) => v,
                                Err(e) => {
                                    return Err(anyhow!(
                                        "tool-call arguments were not valid JSON for tool '{name}' (id={id}): {e}; raw: {args}"
                                    ));
                                }
                            }
                        };
                        events.push(StreamEvent::ToolCallDelta {
                            delta: ToolCallDelta {
                                id: (!id.is_empty()).then_some(id),
                                name: Some(name),
                                input: Some(input),
                            },
                        });
                    }

                    events.push(StreamEvent::Stop {
                        stop_reason: reason.clone(),
                        stop_sequence: None,
                    });
                }
            }
        }
    }

    // Usage (typically last chunk)
    if let Some(usage) = &chunk.usage {
        let usage_event = Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            total_tokens: Some(usage.total_tokens),
            cost: None,
        };
        events.push(StreamEvent::Usage { usage: usage_event });
    }

    Ok(events)
}

/// Build the `/chat/completions` endpoint URL for an OpenAI-wire-compatible
/// provider. Semantics of `model.base_url` match the TypeScript OpenAI SDK's
/// `baseURL` option: it is the full API root *including* any version
/// segment, and the caller appends the resource path. Trailing `/` is
/// tolerated.
///
/// Valid inputs by registry convention:
///   - `https://api.openai.com/v1` → `.../v1/chat/completions`
///   - `https://openrouter.ai/api/v1` → `.../api/v1/chat/completions`
///   - `https://api.groq.com/openai/v1` → `.../openai/v1/chat/completions`
///   - `https://api.deepseek.com` → `.../chat/completions` (DeepSeek's real endpoint)
///   - `https://api.z.ai/api/coding/paas/v4` → `.../paas/v4/chat/completions`
fn build_endpoint_url(base_url: Option<&str>) -> String {
    let base = base_url
        .unwrap_or("https://api.openai.com/v1")
        .trim_end_matches('/');
    format!("{}/chat/completions", base)
}

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<crate::types::stream::AssistantMessageEventStream, anyhow::Error> {
    // Build request
    let request = build_request(model, context, options)?;

    // Endpoint. See [`build_endpoint_url`] for semantics. Appending
    // `/v1/chat/completions` here (the original bug) produced doubled
    // `/v1/v1/...` URLs that OpenRouter silently 404'd on, leaving the
    // stream empty.
    let endpoint = build_endpoint_url(model.base_url.as_deref());

    // API key
    let api_key = options.api_key.as_ref()
        .ok_or_else(|| anyhow!("OpenAI API key required"))?;

    // HTTP client. `reqwest::RequestBuilder::json()` sets Content-Type, so
    // we don't set it manually (setting it twice produces duplicate headers
    // that OpenAI's server rejects — documented quirk in the reference
    // Rust port).
    let client = Client::new();
    let request_builder = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Accept", "text/event-stream");

    // OpenRouter requires attribution headers for its directory listing and
    // rate-limit accounting. Without these, requests still work but the
    // model doesn't appear correctly in the dashboard and some models fall
    // back to stricter limits. We detect OpenRouter by base_url rather than
    // provider enum so any future OpenRouter-proxying registry entry works.
    let is_openrouter = model
        .base_url
        .as_deref()
        .map(|u| u.contains("openrouter.ai"))
        .unwrap_or(false);
    let request_builder = if is_openrouter {
        request_builder
            .header("HTTP-Referer", "https://github.com/mariozechner/pi")
            .header("X-Title", "pi")
    } else {
        request_builder
    };

    let response = request_builder.json(&request).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await?;
        return Err(anyhow!("OpenAI API error {}: {}", status, text));
    }

    // Content-Type safety net. If a provider returns 200 OK with an HTML or
    // JSON body instead of SSE (common for CDN auth pages, some
    // mis-routed proxies), the raw bytes parse as empty SSE events and the
    // caller sees "empty response". Detect that and surface a clear error.
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
            "OpenAI provider: expected text/event-stream, got '{}'. Body starts with: {}",
            content_type,
            body_sample
        ));
    }

    // Byte stream
    let byte_stream = response.bytes_stream().map(|result| {
        result.map_err(|e| std::io::Error::new(ErrorKind::Other, e))
    });
    let stream_reader = StreamReader::new(byte_stream);
    let buf_reader = tokio::io::BufReader::new(stream_reader);

    // Decode SSE
    let sse_stream = crate::stream::decode_sse(buf_reader);

    // Stateful transformation: persistent accumulator for tool call deltas
    // that span multiple SSE chunks (OpenAI streams tool arguments one token
    // at a time). `flat_map` + `iter` flattens the `Vec<StreamEvent>` from
    // `convert_chunk` so we never silently drop events — the previous
    // `events[0]` selector lost text deltas, tool-call assembly, and Stop
    // markers whenever a chunk contained more than one field, which for
    // some providers (OpenRouter / StepFun) happened on the final chunk
    // and prevented the agent loop from terminating.
    use std::sync::{Arc, Mutex};
    let model_id = model.id.clone();
    let accumulator: Arc<Mutex<ToolCallAcc>> = Arc::new(Mutex::new(ToolCallAcc::default()));

    let mapped = sse_stream
        .flat_map(move |result| {
            let events: Vec<Result<StreamEvent, anyhow::Error>> = match result {
                Ok(sse) => {
                    let data = &sse.data;
                    if data == "[DONE]" {
                        vec![Ok(StreamEvent::Stop {
                            stop_reason: "stop".to_string(),
                            stop_sequence: None,
                        })]
                    } else {
                        match serde_json::from_str::<OpenAIChatChunk>(data) {
                            Ok(chunk) => {
                                let mut acc = accumulator.lock().expect("mutex");
                                match convert_chunk(&chunk, &model_id, &mut acc) {
                                    Ok(events) => events.into_iter().map(Ok).collect(),
                                    Err(e) => vec![Err(anyhow!("convert error: {}", e))],
                                }
                            }
                            Err(e) => vec![Err(anyhow!(
                                "JSON parse error: {} on data: {}",
                                e,
                                data
                            ))],
                        }
                    }
                }
                Err(e) => vec![Err(e)],
            };
            futures::stream::iter(events)
        })
        .boxed();

    Ok(Box::pin(mapped) as crate::types::stream::AssistantMessageEventStream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_tool_call_id_passes_short_clean_ids_through() {
        assert_eq!(normalize_tool_call_id("call_abc123"), "call_abc123");
        assert_eq!(
            normalize_tool_call_id("tooluse_d8WN6xC4dw54X9Mig4xHlA"),
            "tooluse_d8WN6xC4dw54X9Mig4xHlA"
        );
    }

    #[test]
    fn normalize_tool_call_id_truncates_to_40_chars() {
        let long_id = "call_".to_string() + &"x".repeat(80);
        let normalized = normalize_tool_call_id(&long_id);
        assert_eq!(normalized.len(), 40);
        assert!(normalized.starts_with("call_"));
    }

    #[test]
    fn normalize_tool_call_id_splits_pipe_separated_responses_ids() {
        // OpenAI Responses / Codex / opencode emit ids like
        // `call_abc123|fc_longbase64+/=stuff`. Azure via
        // OpenRouter only accepts the pre-pipe part.
        let input = "call_Bv123|fc_LongBase64Chars+=xyz/abc_ABCdef";
        let normalized = normalize_tool_call_id(input);
        assert_eq!(normalized, "call_Bv123");
    }

    #[test]
    fn normalize_tool_call_id_sanitizes_disallowed_chars() {
        let input = "call+slash/eq=";
        assert_eq!(normalize_tool_call_id(input), "call_slash_eq_");
    }

    #[test]
    fn normalize_tool_call_id_handles_over_limit_pipe_tail() {
        // 71-char id that Azure explicitly rejects (reproduced
        // against the live OpenRouter endpoint):
        //   `Invalid 'input[1].call_id': string too long.
        //    Expected a string with maximum length 64, but got
        //    a string with length 71 instead.`
        let input = "call_abc123|fc_x6LsomethingVeryLongWithBase64chars+=abc/xyz123456789abc";
        let normalized = normalize_tool_call_id(input);
        assert_eq!(normalized, "call_abc123");
    }

    #[test]
    fn build_request_rewrites_long_tool_call_ids_on_the_wire() {
        // End-to-end guard: a 71-char pipe-separated id on the
        // assistant's tool_use AND the matching tool result must
        // both render as the same short normalized id in the
        // serialized request body. Without this, OpenRouter /
        // Azure return 400 with either
        // `string_above_max_length` or
        // `No tool call found for function call output with call_id X`.
        let long = "call_abc123|fc_x6LsomethingVeryLongWithBase64chars+=abc/xyz123456789abc";
        let model = Model {
            id: "openai/gpt-5".into(),
            name: "gpt-5".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::OpenRouter,
            base_url: Some("https://openrouter.ai/api/v1".into()),
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let ctx = crate::types::Context {
            system_prompt: None,
            messages: vec![
                crate::types::Message::User(vec![crate::types::Content::Text {
                    text: "!ls".into(),
                    cache_control: None,
                }]),
                crate::types::Message::Assistant(vec![crate::types::Content::ToolUse {
                    id: long.to_string(),
                    name: "ls".into(),
                    input: serde_json::json!({}),
                    cache_control: None,
                }]),
                crate::types::Message::Tool {
                    tool_use_id: long.to_string(),
                    content: vec![crate::types::Content::Text {
                        text: "file.txt".into(),
                        cache_control: None,
                    }],
                    is_error: Some(false),
                },
            ],
            tools: None,
        };
        let options = crate::types::StreamOptions {
            temperature: None,
            max_tokens: None,
            signal: None,
            api_key: Some("test".into()),
            transport: None,
            cache_retention: None,
            session_id: None,
            headers: None,
            reasoning_effort: None,
            thinking_budgets: None,
        };
        let req = build_request(&model, &ctx, &options).unwrap();
        let json = serde_json::to_string(&req).unwrap();
        let normalized = normalize_tool_call_id(long);
        assert_eq!(normalized, "call_abc123");
        assert!(
            !json.contains(long),
            "expected long id to be stripped, body: {json}"
        );
        // Both the assistant's tool_calls[].id and the tool
        // message's tool_call_id must use the same normalized id.
        let target = format!("\"{normalized}\"");
        let occurrences = json.matches(&target).count();
        assert!(
            occurrences >= 2,
            "expected 2 occurrences of normalized id {target}, got {occurrences} in body: {json}"
        );
    }

    #[tokio::test]
    async fn test_build_request() {
        let model = Model {
            id: "gpt-4".into(),
            name: "GPT-4".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::OpenAi,
            base_url: None,
            reasoning: false,
            cost: None,
            context_window: None,
            max_tokens: None,
            compat: None,
            multimodal: None,
        };
        let context = Context {
            system_prompt: Some("You are GPT".into()),
            messages: Vec::new(),
            tools: None,
        };
        let options = StreamOptions {
            temperature: Some(0.5),
            max_tokens: Some(512),
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
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.stream, true);
        assert_eq!(req.stream_options.as_ref().unwrap().include_usage, true);
    }

    /// Regression: a chunk with both `delta.content` and `finish_reason` must
    /// yield BOTH a TextDelta and a Stop event. Previous implementation
    /// returned only `events[0]` which silently dropped the Stop, stranding
    /// callers that wait for end-of-turn.
    #[test]
    fn convert_chunk_emits_all_events_for_final_chunk() {
        let json = r#"{
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "stepfun/step-3.5-flash",
            "choices": [{
                "index": 0,
                "delta": {"content": "!"},
                "finish_reason": "stop"
            }]
        }"#;
        let chunk: OpenAIChatChunk = serde_json::from_str(json).unwrap();
        let mut acc = ToolCallAcc::default();
        let events = convert_chunk(&chunk, "stepfun/step-3.5-flash", &mut acc).unwrap();

        let text_events = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::TextDelta { .. }))
            .count();
        let stop_events = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Stop { .. }))
            .count();
        assert_eq!(text_events, 1, "expected text delta, got events: {:?}", events);
        assert_eq!(stop_events, 1, "expected stop, got events: {:?}", events);
    }

    /// Regression: tool-call arguments spanning multiple chunks must be
    /// assembled using the accumulator shared across calls. Previous
    /// implementation created a fresh HashMap per chunk, so partial
    /// arguments were overwritten.
    #[test]
    fn convert_chunk_accumulates_tool_call_args_across_chunks() {
        let chunk1_json = r#"{
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":"}
                }]}
            }]
        }"#;
        let chunk2_json = r#"{
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "\"hello\"}"}
                }]},
                "finish_reason": "tool_calls"
            }]
        }"#;

        let chunk1: OpenAIChatChunk = serde_json::from_str(chunk1_json).unwrap();
        let chunk2: OpenAIChatChunk = serde_json::from_str(chunk2_json).unwrap();

        let mut acc = ToolCallAcc::default();
        let events1 = convert_chunk(&chunk1, "m", &mut acc).unwrap();
        let events2 = convert_chunk(&chunk2, "m", &mut acc).unwrap();

        // No ToolCallDelta until finish_reason arrives.
        assert!(events1.iter().all(|e| !matches!(e, StreamEvent::ToolCallDelta { .. })));

        let tool_call = events2
            .iter()
            .find_map(|e| match e {
                StreamEvent::ToolCallDelta { delta } => Some(delta),
                _ => None,
            })
            .expect("expected assembled tool call in final chunk");
        assert_eq!(tool_call.name.as_deref(), Some("lookup"));
        // Regression: previously the accumulator stored only (name, args)
        // so the call id was dropped. The agent loop filters out tool calls
        // with empty id, meaning StepFun / OpenRouter tool calls were
        // silently dropped and the turn ended with "(empty response)".
        assert_eq!(
            tool_call.id.as_deref(),
            Some("call_abc123"),
            "tool call id must survive streaming accumulation"
        );
        let input = tool_call.input.as_ref().expect("input");
        assert_eq!(input.get("q").and_then(|v| v.as_str()), Some("hello"));
    }

    /// Parallel tool calls must all reach the agent in insertion order with
    /// their ids preserved.
    #[test]
    fn convert_chunk_emits_parallel_tool_calls_in_order() {
        let chunk_json = r#"{
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [
                    {"index": 0, "id": "call_a", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                    {"index": 1, "id": "call_b", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
                ]},
                "finish_reason": "tool_calls"
            }]
        }"#;
        let chunk: OpenAIChatChunk = serde_json::from_str(chunk_json).unwrap();
        let mut acc = ToolCallAcc::default();
        let events = convert_chunk(&chunk, "m", &mut acc).unwrap();

        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallDelta { delta } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("call_a"));
        assert_eq!(calls[0].name.as_deref(), Some("read"));
        assert_eq!(calls[1].id.as_deref(), Some("call_b"));
        assert_eq!(calls[1].name.as_deref(), Some("ls"));
    }

    /// Regression: URL construction must match every real registry base_url
    /// shape. Previously we appended `/v1/chat/completions` which doubled
    /// the version segment for openrouter / openai / groq / xai / cerebras
    /// / huggingface / cloudflare and produced 404s that surfaced as empty
    /// streams. These assertions cover every openai-compat provider we ship.
    #[test]
    fn endpoint_url_matches_every_openai_compat_provider() {
        // (provider_id, registry_base_url, expected_full_endpoint)
        let cases = [
            (
                "openai",
                "https://api.openai.com/v1",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "openrouter",
                "https://openrouter.ai/api/v1",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
            (
                "groq",
                "https://api.groq.com/openai/v1",
                "https://api.groq.com/openai/v1/chat/completions",
            ),
            (
                "xai",
                "https://api.x.ai/v1",
                "https://api.x.ai/v1/chat/completions",
            ),
            (
                "cerebras",
                "https://api.cerebras.ai/v1",
                "https://api.cerebras.ai/v1/chat/completions",
            ),
            (
                "deepseek",
                "https://api.deepseek.com",
                "https://api.deepseek.com/chat/completions",
            ),
            (
                "huggingface",
                "https://router.huggingface.co/v1",
                "https://router.huggingface.co/v1/chat/completions",
            ),
            (
                "zai",
                "https://api.z.ai/api/coding/paas/v4",
                "https://api.z.ai/api/coding/paas/v4/chat/completions",
            ),
            (
                "opencode-go",
                "https://opencode.ai/zen/go/v1",
                "https://opencode.ai/zen/go/v1/chat/completions",
            ),
            (
                "cloudflare-workers-ai",
                "https://api.cloudflare.com/client/v4/accounts/ACCT/ai/v1",
                "https://api.cloudflare.com/client/v4/accounts/ACCT/ai/v1/chat/completions",
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
    fn endpoint_url_defaults_to_openai_v1_when_unset() {
        assert_eq!(
            build_endpoint_url(None),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn endpoint_url_tolerates_trailing_slash() {
        assert_eq!(
            build_endpoint_url(Some("https://api.openai.com/v1/")),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            build_endpoint_url(Some("https://openrouter.ai/api/v1//")),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }
}
