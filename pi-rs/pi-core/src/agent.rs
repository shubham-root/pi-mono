//! Agent state and execution loop.
//! Phase 1.6 + tool dispatch (2.15)

use anyhow::{Result, anyhow};
use pi_ai::providers::stream as provider_stream;
use pi_ai::types::{
    CancellationToken, Message, Content, Context, StreamOptions, StreamEvent,
    Model, ToolSchema, ToolCall, ToolCallDelta, Api, Provider, Usage,
};
use pi_tools::{Tool, ToolContext, ToolResult};
use futures::{StreamExt, pin_mut};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::mpsc;

use crate::model_registry::{ModelRegistry, Model as RegistryModel, Provider as RegistryProvider};

/// Incremental agent progress events emitted over a channel by
/// [`Agent::prompt_stream`]. Consumers (TUI, tests, rpc mode) can stream
/// these into their own UI without waiting for the full turn to finish.
///
/// Every `prompt_stream` call emits exactly one `Done` event (or an
/// `Error` event that replaces it) as its last message. All other events
/// describe work in progress:
///
///   TextDelta / ThinkingDelta    → one per incoming token chunk
///   ToolCallStart                 → right before a tool runs
///   ToolCallResult                → right after a tool finishes
///   Usage                         → end of each provider turn
///   TurnComplete                  → assistant turn ended (may loop again if tools ran)
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TextDelta {
        delta: String,
    },
    ThinkingDelta {
        delta: String,
        signature: Option<String>,
    },
    ToolCallStart {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolCallResult {
        id: String,
        output: String,
        is_error: bool,
    },
    Usage(Usage),
    /// The provider completed an assistant turn. If `has_tool_calls` is
    /// true, the agent will loop and emit another set of deltas as the
    /// model responds to the tool results.
    TurnComplete {
        stop_reason: String,
        has_tool_calls: bool,
    },
    /// Final event. `final_text` is the concatenation of every TextDelta
    /// from the last assistant turn (what the blocking `prompt` returns).
    Done {
        final_text: String,
    },
    Error {
        message: String,
    },
}

/// Agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of tool-turn iterations (default: 10).
    pub max_turns: u32,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Whether to allow tool use.
    pub allow_tools: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            temperature: Some(0.7),
            allow_tools: true,
        }
    }
}

/// Agent state.
#[derive(Debug)]
pub struct Agent {
    messages: Vec<Message>,
    model_id: String,
    system_prompt: Option<String>,
    tools: Vec<Box<dyn pi_tools::Tool>>,
    abort: Option<CancellationToken>,
    api_key: Option<String>,
    config: AgentConfig,
    /// Reasoning effort forwarded to providers that support it
    /// (openai o-series, claude extended thinking, deepseek, ...).
    /// `None` means "provider default"; the TUI cycles between
    /// `None`, `Low`, `Medium`, `High`.
    thinking_level: Option<pi_ai::types::ThinkingLevel>,
}

impl Agent {
    pub fn new(model_id: &str) -> Self {
        Self {
            messages: Vec::new(),
            model_id: model_id.to_string(),
            system_prompt: None,
            tools: Vec::new(),
            abort: None,
            api_key: None,
            config: AgentConfig::default(),
            thinking_level: None,
        }
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    pub fn with_tool(mut self, tool: Box<dyn pi_tools::Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Swap the active model. Accepts the same id formats as [`resolve_model`]:
    /// a bare model id (first match in the registry wins) or a qualified
    /// `provider_id/model_id` pair for disambiguation. Returns an error if
    /// the id is not in the registry so callers can surface a clear message
    /// instead of silently falling back.
    pub fn set_model(&mut self, model_id: &str) -> Result<()> {
        resolve_model(model_id)?;
        self.model_id = model_id.to_string();
        Ok(())
    }

    /// Update the API key used for the active provider. Pass `None` to clear.
    pub fn set_api_key(&mut self, key: Option<String>) {
        self.api_key = key.filter(|s| !s.is_empty());
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key.as_deref().map(|s| !s.is_empty()).unwrap_or(false)
    }

    pub fn thinking_level(&self) -> Option<pi_ai::types::ThinkingLevel> {
        self.thinking_level
    }

    pub fn set_thinking_level(&mut self, level: Option<pi_ai::types::ThinkingLevel>) {
        self.thinking_level = level;
    }

    /// Convert tool registry into provider-agnostic ToolSchema list.
    pub fn build_tool_schemas(&self) -> Vec<ToolSchema> {
        self.tools
            .iter()
            .map(|tool| ToolSchema {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.schema().clone(),
                cache_control: None,
            })
            .collect()
    }

    /// Execute a single tool call and return the result string.
    async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        // Find tool by name
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_call.name)
            .ok_or_else(|| anyhow!("Tool not found: {}", tool_call.name))?;

        let ctx = ToolContext {
            cwd: std::env::current_dir()?,
            abort: self.abort.as_ref().map(|t| Arc::new(t.inner().clone())),
            env: std::env::vars().collect(),
        };

        let result = tool.execute(&ctx, tool_call.input.clone()).await?;
        match result {
            pi_tools::ToolResult::Success(output) => Ok(output),
            pi_tools::ToolResult::Error(msg) => Ok(format!("[ERROR] {}", msg)),
        }
    }

    /// Core agent loop: turn-based with tool execution.
    /// Returns the final text response from the assistant (concatenated text deltas).
    /// Run a prompt turn (possibly looping for tool use) and return the
    /// final assistant text. Keeps backward compatibility with callers
    /// that don't want streaming; the TUI uses [`Self::prompt_stream`]
    /// directly for incremental rendering.
    pub async fn prompt(&mut self, content: &str) -> Result<String> {
        let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
        let run = self.prompt_stream(content, tx);
        tokio::pin!(run);
        // Drain the channel in parallel with driving the prompt so the
        // unbounded sender doesn't hold onto events we never consume.
        let mut final_text = String::new();
        loop {
            tokio::select! {
                res = &mut run => {
                    // Consume any tail-end events the worker pushed after
                    // `Done` (usually none; belt and suspenders).
                    while let Ok(event) = rx.try_recv() {
                        if let AgentEvent::Done { final_text: t } = event {
                            final_text = t;
                        }
                    }
                    return res.map(|_| final_text);
                }
                event = rx.recv() => {
                    match event {
                        Some(AgentEvent::Done { final_text: t }) => final_text = t,
                        Some(AgentEvent::Error { message }) => {
                            return Err(anyhow!("{message}"));
                        }
                        Some(_) => {}
                        None => return Err(anyhow!("agent event channel closed prematurely")),
                    }
                }
            }
        }
    }

    /// Run a prompt turn and forward every agent event to `tx`. The
    /// channel is never closed by this method; the caller owns the
    /// receiver lifetime. Always emits exactly one terminal event
    /// (`Done` on success, `Error` on failure) before returning.
    pub async fn prompt_stream(
        &mut self,
        content: &str,
        tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<()> {
        // Append user message. A bail-out below (e.g. max-turns) still
        // leaves this in history so the session transcript stays
        // consistent with what the user typed.
        self.messages.push(Message::User(vec![Content::Text {
            text: content.to_string(),
            cache_control: None,
        }]));

        match self.prompt_stream_inner(&tx).await {
            Ok(final_text) => {
                let _ = tx.send(AgentEvent::Done { final_text });
                Ok(())
            }
            Err(e) => {
                let _ = tx.send(AgentEvent::Error {
                    message: e.to_string(),
                });
                Err(e)
            }
        }
    }

    async fn prompt_stream_inner(
        &mut self,
        tx: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<String> {
        let mut final_response_text = String::new();
        let mut turn_count = 0;

        loop {
            if turn_count >= self.config.max_turns {
                return Err(anyhow!("Max turns ({}) exceeded", self.config.max_turns));
            }
            turn_count += 1;

            let tool_schemas = if self.config.allow_tools && !self.tools.is_empty() {
                Some(self.build_tool_schemas())
            } else {
                None
            };

            let context = Context {
                system_prompt: self.system_prompt.clone(),
                messages: self.messages.clone(),
                tools: tool_schemas,
            };

            let options = StreamOptions {
                max_tokens: Some(4096),
                signal: self.abort.clone(),
                temperature: self.config.temperature,
                api_key: self.api_key.clone(),
                transport: None,
                cache_retention: None,
                session_id: None,
                headers: None,
                reasoning_effort: self.thinking_level,
                thinking_budgets: None,
            };

            let model = resolve_model(&self.model_id)?;
            let mut stream = provider_stream(&model, &context, &options).await?;

            let mut content_blocks: Vec<Content> = Vec::new();
            let mut current_text = String::new();
            let mut current_thinking: Option<String> = None;
            let mut thinking_signature: Option<String> = None;
            let mut collected_tool_calls: Vec<ToolCall> = Vec::new();
            let mut stop_reason = "end_turn".to_string();

            pin_mut!(stream);
            while let Some(event_res) = stream.as_mut().next().await {
                match event_res? {
                    StreamEvent::Start { .. } => {}
                    StreamEvent::TextDelta { delta, .. } => {
                        if !delta.is_empty() {
                            current_text.push_str(&delta);
                            // Forward each delta to the caller so the UI
                            // can render tokens as they arrive. Ignore
                            // send errors — if the receiver is gone the
                            // agent should still finish the turn cleanly
                            // so conversation state stays consistent.
                            let _ = tx.send(AgentEvent::TextDelta { delta });
                        }
                    }
                    StreamEvent::ThinkingDelta { delta, signature, .. } => {
                        if !delta.is_empty() {
                            current_thinking
                                .get_or_insert_with(String::new)
                                .push_str(&delta);
                        }
                        if signature.is_some() {
                            thinking_signature = signature.clone();
                        }
                        let _ = tx.send(AgentEvent::ThinkingDelta { delta, signature });
                    }
                    StreamEvent::ToolCallDelta { delta } => {
                        let name = delta.name.unwrap_or_default();
                        if name.is_empty() {
                            continue;
                        }
                        let id = delta
                            .id
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
                        let input = delta.input.unwrap_or_else(|| serde_json::json!({}));
                        collected_tool_calls.push(ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                        let _ = tx.send(AgentEvent::ToolCallStart { id, name, input });
                    }
                    StreamEvent::Usage { usage } => {
                        let _ = tx.send(AgentEvent::Usage(usage));
                    }
                    StreamEvent::Stop { stop_reason: reason, .. } => {
                        stop_reason = reason;
                        break;
                    }
                    StreamEvent::Error { error, .. } => {
                        return Err(anyhow::anyhow!("Stream error: {}", error));
                    }
                }
            }

            // Build assistant content blocks for the transcript. Ordering
            // mirrors Anthropic's wire shape: thinking → text → tool uses.
            if let Some(thinking) = current_thinking {
                content_blocks.push(Content::Thinking {
                    thinking,
                    signature: thinking_signature.clone(),
                    cache_control: None,
                });
            }
            if !current_text.is_empty() {
                content_blocks.push(Content::Text {
                    text: current_text.clone(),
                    cache_control: None,
                });
            }
            for tc in &collected_tool_calls {
                content_blocks.push(Content::ToolUse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input: tc.input.clone(),
                    cache_control: None,
                });
            }
            self.messages.push(Message::Assistant(content_blocks));

            let has_tools = !collected_tool_calls.is_empty();
            let _ = tx.send(AgentEvent::TurnComplete {
                stop_reason: stop_reason.clone(),
                has_tool_calls: has_tools,
            });

            if !has_tools {
                final_response_text = current_text;
                break;
            }

            // Execute each tool call sequentially, emitting results as
            // they arrive so the UI can render them before the next turn
            // kicks off.
            for tc in collected_tool_calls {
                let result_str = self.execute_tool(&tc).await?;
                let is_error = result_str.starts_with("[ERROR]");
                let _ = tx.send(AgentEvent::ToolCallResult {
                    id: tc.id.clone(),
                    output: result_str.clone(),
                    is_error,
                });
                self.messages.push(Message::Tool {
                    tool_use_id: tc.id,
                    content: vec![Content::Text {
                        text: result_str,
                        cache_control: None,
                    }],
                    is_error: Some(is_error),
                });
            }
        }

        Ok(final_response_text)
    }

    /// Abort current operation.
    pub fn abort(&self) {
        if let Some(token) = &self.abort {
            token.cancel();
        }
    }

    /// Get conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get tool count for testing.
    pub fn tools_count(&self) -> usize {
        self.tools.len()
    }

    /// Get config for testing.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get mutable config for testing.
    pub fn config_mut(&mut self) -> &mut AgentConfig {
        &mut self.config
    }

    /// Clear conversation history.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

/// Map the TOML `api` string stored in the registry to the pi-ai `Api` enum.
/// Returns `None` for APIs that don't yet have a provider implementation in
/// pi-ai; callers should treat that as a hard error with a clear message.
fn api_from_str(api: &str) -> Option<Api> {
    match api {
        "anthropic-messages" => Some(Api::AnthropicMessages),
        "openai-completions" => Some(Api::OpenAiCompletions),
        "openai-responses" => Some(Api::OpenAiResponses),
        "azure-openai-responses" => Some(Api::AzureOpenAiResponses),
        "openai-codex-responses" => Some(Api::OpenAiCodexResponses),
        "mistral-conversations" => Some(Api::MistralConversations),
        "bedrock-converse-stream" => Some(Api::BedrockConverseStream),
        "google-generative-ai" => Some(Api::GoogleGenerativeAi),
        "google-gemini-cli" => Some(Api::GoogleGeminiCli),
        "google-vertex" => Some(Api::GoogleVertex),
        "openrouter-messages" => Some(Api::OpenRouterMessages),
        "vercel-ai-gateway" => Some(Api::VercelAiGateway),
        "faux" => Some(Api::Faux),
        _ => None,
    }
}

/// Map a registry provider id to the pi-ai `Provider` enum. New providers can
/// be added to the TOML files without touching this table; unmapped ids fall
/// back to `Provider::OpenAi` purely as a placeholder since the `Api` enum is
/// what actually drives dispatch. The real routing decision is `Api`.
fn provider_from_id(id: &str) -> Provider {
    match id {
        "amazon-bedrock" => Provider::AmazonBedrock,
        "anthropic" => Provider::Anthropic,
        "google" => Provider::Google,
        "google-gemini-cli" => Provider::GoogleGeminiCli,
        "google-antigravity" => Provider::GoogleAntigravity,
        "google-vertex" => Provider::GoogleVertex,
        "openai" => Provider::OpenAi,
        "openai-codex" => Provider::OpenAiCodex,
        "azure-openai-responses" => Provider::AzureOpenAiResponses,
        "openrouter" => Provider::OpenRouter,
        "vercel-ai-gateway" => Provider::VercelAiGateway,
        "mistral" => Provider::Mistral,
        "groq" => Provider::Groq,
        "xai" => Provider::Xai,
        "deepseek" => Provider::DeepSeek,
        "cerebras" => Provider::Cerebras,
        "fireworks" => Provider::Fireworks,
        "huggingface" => Provider::HuggingFace,
        "cloudflare-workers-ai" => Provider::CloudflareWorkersAi,
        "github-copilot" => Provider::GitHubCopilot,
        "zai" => Provider::Zai,
        "minimax" => Provider::Minimax,
        "minimax-cn" => Provider::MinimaxCn,
        "kimi-coding" => Provider::KimiCoding,
        "opencode" => Provider::OpenCode,
        "opencode-go" => Provider::OpenCodeGo,
        // Unknown provider id: keep compiling but the provider enum doesn't
        // drive dispatch in pi-ai, so this is cosmetic-only.
        _ => Provider::OpenAi,
    }
}

/// Resolve a model id to the pi-ai `Model` carrying the correct API,
/// provider, and base URL. Supports two id formats:
///
///   1. Bare model id (e.g. `gpt-4o`): searched across all providers in the
///      registry. First match wins, so prefer the explicit form when two
///      providers share an id (e.g. `anthropic.claude-opus-4-6-v1` exists on
///      both `anthropic` and `amazon-bedrock`).
///   2. Qualified `provider_id/model_id` (e.g. `amazon-bedrock/amazon.nova-pro-v1:0`):
///      bound to the named provider directly.
///
/// Returns a descriptive error if the model isn't in the registry or its
/// `api` field doesn't map to a pi-ai provider implementation.
fn resolve_model(model_id: &str) -> Result<Model> {
    let registry = ModelRegistry::global();

    let (provider, model): (&RegistryProvider, &RegistryModel) =
        if let Some((pid, mid)) = model_id.split_once('/') {
            registry
                .find_by_provider(pid, mid)
                .ok_or_else(|| anyhow!("unknown model '{}' under provider '{}'", mid, pid))?
        } else {
            registry
                .find_model(model_id)
                .ok_or_else(|| anyhow!(
                    "unknown model id '{}'. Use `provider_id/model_id` to disambiguate, or drop a TOML into ~/.pi/providers/ to register it.",
                    model_id
                ))?
        };

    let api = api_from_str(&model.api).ok_or_else(|| anyhow!(
        "model '{}' declares api='{}' which has no pi-ai provider implementation yet",
        model.id, model.api
    ))?;

    Ok(Model {
        id: model.id.clone(),
        name: model.name.clone(),
        api,
        provider: provider_from_id(&provider.id),
        base_url: model.base_url.clone(),
        reasoning: model.reasoning,
        cost: None,
        context_window: if model.context_window > 0 {
            Some(model.context_window)
        } else {
            None
        },
        max_tokens: if model.max_tokens > 0 {
            Some(model.max_tokens)
        } else {
            None
        },
        compat: None,
        multimodal: None,
    })
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    /// Regression: every model resolves to an API that matches its provider,
    /// not the old Anthropic-by-default heuristic.
    #[test]
    fn openai_model_routes_to_openai_api() {
        let m = resolve_model("openai/gpt-4o").expect("gpt-4o should resolve");
        // OpenAI may use either completions or responses API depending on the
        // model; both must dispatch via an OpenAI-compatible client.
        assert!(
            matches!(
                m.api,
                Api::OpenAiCompletions | Api::OpenAiResponses
            ),
            "expected OpenAI api, got {:?}",
            m.api
        );
        assert_eq!(m.provider, Provider::OpenAi);
    }

    #[test]
    fn anthropic_model_routes_to_anthropic_api() {
        let m = resolve_model("claude-opus-4-7").expect("claude should resolve");
        assert_eq!(m.api, Api::AnthropicMessages);
        assert_eq!(m.provider, Provider::Anthropic);
    }

    #[test]
    fn bedrock_model_routes_to_bedrock_api() {
        let m = resolve_model("amazon-bedrock/amazon.nova-pro-v1:0")
            .expect("nova pro under bedrock should resolve");
        assert_eq!(m.api, Api::BedrockConverseStream);
        assert_eq!(m.provider, Provider::AmazonBedrock);
        assert!(
            m.base_url
                .as_deref()
                .map(|u| u.contains("bedrock-runtime"))
                .unwrap_or(false),
            "base_url should come from registry, got {:?}",
            m.base_url
        );
    }

    #[test]
    fn openrouter_model_uses_openai_compatible_api() {
        // OpenRouter speaks OpenAI's wire protocol with its own base URL,
        // so its models carry api="openai-completions" in the registry.
        // This regression test guards against routing them to Anthropic or
        // a non-existent OpenRouter-specific dispatcher.
        let registry = ModelRegistry::global();
        let id = registry
            .provider("openrouter")
            .and_then(|p| {
                p.models
                    .iter()
                    .find(|m| m.api == "openai-completions")
                    .map(|m| m.id.clone())
            })
            .expect("openrouter should have at least one openai-compatible model");

        let m = resolve_model(&format!("openrouter/{}", id))
            .expect("openrouter model should resolve via qualified id");
        assert_eq!(m.api, Api::OpenAiCompletions);
        assert_eq!(m.provider, Provider::OpenRouter);
        assert!(
            m.base_url
                .as_deref()
                .map(|u| u.contains("openrouter.ai"))
                .unwrap_or(false),
            "openrouter base_url must be preserved so requests don't go to api.openai.com, got {:?}",
            m.base_url
        );
    }

    #[test]
    fn vercel_gateway_model_dispatches_correctly() {
        // TS source declares all Vercel AI Gateway models with
        // `api = "anthropic-messages"`. We don't force a specific api here;
        // we just assert it resolves and preserves the vercel base_url so
        // requests don't leak to api.anthropic.com.
        let registry = ModelRegistry::global();
        let Some(provider) = registry.provider("vercel-ai-gateway") else {
            return; // registry without vercel is fine
        };
        let Some(first) = provider.models.first() else {
            return;
        };

        let m = resolve_model(&format!("vercel-ai-gateway/{}", first.id))
            .expect("vercel gateway model should resolve");
        assert_eq!(m.provider, Provider::VercelAiGateway);
        assert!(
            m.base_url
                .as_deref()
                .map(|u| u.contains("ai-gateway.vercel.sh"))
                .unwrap_or(false),
            "vercel gateway base_url must route to ai-gateway.vercel.sh, got {:?}",
            m.base_url
        );
    }

    /// Regression: StepFun / OpenRouter models must NOT end up on Bedrock.
    /// This is the specific failure the user hit before `resolve_model` was
    /// moved to the registry.
    #[test]
    fn openrouter_stepfun_does_not_route_to_bedrock() {
        let registry = ModelRegistry::global();
        let stepfun_id = registry
            .provider("openrouter")
            .and_then(|p| {
                p.models
                    .iter()
                    .find(|m| m.id.to_lowercase().contains("stepfun") || m.name.to_lowercase().contains("stepfun"))
                    .map(|m| m.id.clone())
            });

        if let Some(id) = stepfun_id {
            let m = resolve_model(&format!("openrouter/{}", id)).expect("stepfun via openrouter");
            assert_ne!(
                m.api,
                Api::BedrockConverseStream,
                "StepFun via OpenRouter must not dispatch to Bedrock"
            );
            assert_ne!(
                m.provider,
                Provider::AmazonBedrock,
                "StepFun via OpenRouter must not be labelled as Bedrock"
            );
        }
        // If the upstream OpenRouter catalog drops StepFun, the test is a
        // no-op rather than a false failure.
    }

    #[test]
    fn mistral_model_routes_to_mistral_api() {
        let registry = ModelRegistry::global();
        let id = registry
            .provider("mistral")
            .and_then(|p| p.models.first())
            .map(|m| m.id.clone())
            .expect("mistral should have at least one model");

        let m = resolve_model(&format!("mistral/{}", id)).expect("mistral resolve");
        assert_eq!(m.api, Api::MistralConversations);
        assert_eq!(m.provider, Provider::Mistral);
    }

    #[test]
    fn google_gemini_model_routes_to_google_api() {
        let registry = ModelRegistry::global();
        let id = registry
            .provider("google")
            .and_then(|p| p.models.first())
            .map(|m| m.id.clone())
            .expect("google should have at least one model");

        let m = resolve_model(&format!("google/{}", id)).expect("google resolve");
        assert_eq!(m.api, Api::GoogleGenerativeAi);
        assert_eq!(m.provider, Provider::Google);
    }

    #[test]
    fn unknown_model_returns_error_not_anthropic_fallback() {
        let err = resolve_model("definitely-not-a-real-model-xyz").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown model"),
            "expected unknown-model error, got: {msg}"
        );
    }

    #[test]
    fn qualified_id_beats_bare_id_for_disambiguation() {
        // `gpt-oss-120b` exists under several providers (groq, cerebras,
        // fireworks...). The qualified form must bind to the exact provider.
        let registry = ModelRegistry::global();
        let providers_with_gpt_oss: Vec<String> = registry
            .providers()
            .filter(|p| p.models.iter().any(|m| m.id.contains("gpt-oss")))
            .map(|p| p.id.clone())
            .collect();

        if providers_with_gpt_oss.len() >= 2 {
            for pid in &providers_with_gpt_oss {
                let mid = registry
                    .provider(pid)
                    .unwrap()
                    .models
                    .iter()
                    .find(|m| m.id.contains("gpt-oss"))
                    .map(|m| m.id.clone())
                    .unwrap();
                let resolved = resolve_model(&format!("{}/{}", pid, mid))
                    .expect("qualified id should resolve");
                assert_eq!(resolved.provider, provider_from_id(pid));
            }
        }
    }

    #[test]
    fn every_api_in_registry_has_pi_ai_implementation_mapping() {
        // Guard against a future drift where we add a new `api = "..."` in
        // TOML but forget to update `api_from_str`. We don't require every
        // *model* to resolve (some providers like gemini-cli are not wired
        // yet), but every distinct api string must be either supported or
        // explicitly listed as known-unsupported here.
        let registry = ModelRegistry::global();
        let mut unsupported_apis: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for (_, model) in registry.all_models() {
            if api_from_str(&model.api).is_none() {
                unsupported_apis.insert(model.api.clone());
            }
        }

        // The ones we know pi-ai doesn't implement yet. If this list shrinks,
        // great — remove entries. If it grows, the test fails loudly.
        let known_unsupported: std::collections::BTreeSet<String> = [].into_iter().collect();

        assert_eq!(
            unsupported_apis, known_unsupported,
            "registry declares api strings with no pi-ai mapping: {:?}",
            unsupported_apis
        );
    }
}
