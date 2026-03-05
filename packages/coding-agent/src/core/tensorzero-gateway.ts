/**
 * TensorZero gateway integration.
 *
 * When TENSORZERO_GATEWAY_URL is set, all LLM requests are routed through the
 * TensorZero gateway's OpenAI-compatible endpoint. The gateway handles provider
 * routing, observability, and fallbacks.
 *
 * Environment variables:
 *   TENSORZERO_GATEWAY_URL    - Gateway base URL (e.g. https://13-232-113-182.sslip.io/gateway)
 *   TENSORZERO_GATEWAY_API_KEY - Gateway API key for authentication (optional)
 */

import {
	type Api,
	type AssistantMessageEventStream,
	type CacheRetention,
	type Context,
	type Message,
	type Model,
	type SimpleStreamOptions,
	streamSimple,
	type UserMessage,
} from "@mariozechner/pi-ai";
import { randomBytes } from "crypto";

/**
 * Generate a UUID v7 (RFC 9562) – time-ordered, random.
 * TensorZero requires v7 for episode_id and rejects v4.
 */
function uuidv7(): string {
	const now = Date.now();
	const bytes = randomBytes(16);

	// Bytes 0-5: 48-bit big-endian millisecond timestamp
	bytes[0] = (now / 2 ** 40) & 0xff;
	bytes[1] = (now / 2 ** 32) & 0xff;
	bytes[2] = (now / 2 ** 24) & 0xff;
	bytes[3] = (now / 2 ** 16) & 0xff;
	bytes[4] = (now / 2 ** 8) & 0xff;
	bytes[5] = now & 0xff;

	// Version 7
	bytes[6] = (bytes[6] & 0x0f) | 0x70;
	// Variant 10xx
	bytes[8] = (bytes[8] & 0x3f) | 0x80;

	const hex = bytes.toString("hex");
	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export type TensorZeroCacheMode = "on" | "off" | "write_only" | "read_only";

export interface TensorZeroConfig {
	gatewayUrl: string;
	apiKey?: string;
	/** Episode ID for grouping inferences within a session. Auto-generated if not provided. */
	episodeId?: string;
}

export function getTensorZeroConfig(): TensorZeroConfig | undefined {
	const gatewayUrl = process.env.TENSORZERO_GATEWAY_URL;
	if (!gatewayUrl) return undefined;

	return {
		gatewayUrl: gatewayUrl.replace(/\/+$/, ""),
		apiKey: process.env.TENSORZERO_GATEWAY_API_KEY,
	};
}

/**
 * Providers that support TensorZero's implicit model resolution format:
 *   tensorzero::model_name::<provider_type>::<model_id>
 * Other providers must be pre-defined in tensorzero.toml and referenced by
 * their config name using the {provider}--{model_id} convention.
 */
const IMPLICIT_PROVIDERS: Record<string, string> = {
	anthropic: "anthropic",
	openai: "openai",
	google: "google_ai_studio_gemini",
	"google-vertex": "gcp_vertex_gemini",
	xai: "xai",
	mistral: "mistral",
	groq: "groq",
};

/**
 * Rewrite a model to route through TensorZero's OpenAI-compatible endpoint.
 *
 * For providers that support implicit model resolution (anthropic, openai, etc.):
 *   tensorzero::model_name::<provider_type>::<model_id>
 *
 * For providers whose models must be pre-defined in tensorzero.toml (bedrock,
 * openrouter, azure, etc.), use the config name convention: {provider}--{model_id}
 *   tensorzero::model_name::{provider}--{model_id}
 */
function rewriteModelForGateway(model: Model<Api>, config: TensorZeroConfig): Model<"openai-completions"> {
	let tzModelName: string;
	const tzProviderType = IMPLICIT_PROVIDERS[model.provider];
	if (tzProviderType) {
		tzModelName = `tensorzero::model_name::${tzProviderType}::${model.id}`;
	} else {
		// Must be pre-defined in tensorzero.toml as "{provider}--{model_id}"
		tzModelName = `tensorzero::model_name::${model.provider}--${model.id}`;
	}
	return {
		...model,
		id: tzModelName,
		api: "openai-completions" as const,
		baseUrl: `${config.gatewayUrl}/openai/v1`,
	};
}

/**
 * Check if a model supports Anthropic-style prompt caching (cache_control).
 * Only Anthropic Claude models support this. Non-Claude models routed through
 * Anthropic-compatible proxies (e.g. opencode.ai/zen) do not.
 */
function supportsAnthropicCaching(model: Model<Api>): boolean {
	if (model.cost.cacheRead || model.cost.cacheWrite) {
		return true;
	}
	const id = model.id.toLowerCase();
	if (id.includes("claude")) return true;
	if (model.provider === "anthropic") return true;
	return false;
}

/**
 * Check if a Bedrock model supports prompt caching.
 * Only certain Anthropic Claude models support cachePoint on Bedrock.
 * Other models (Moonshot Kimi, Qwen, etc.) reject cache requests with 403.
 */
function supportsBedrockPromptCaching(model: Model<Api>): boolean {
	if (model.cost.cacheRead || model.cost.cacheWrite) {
		return true;
	}

	const id = model.id.toLowerCase();
	// Claude 4.x models (opus-4, sonnet-4, haiku-4)
	if (id.includes("claude") && (id.includes("-4-") || id.includes("-4."))) return true;
	// Claude 3.7 Sonnet
	if (id.includes("claude-3-7-sonnet")) return true;
	// Claude 3.5 Haiku
	if (id.includes("claude-3-5-haiku")) return true;
	return false;
}

// ---------------------------------------------------------------------------
// Provider-level prompt caching via tensorzero::extra_body
//
// TensorZero's `tensorzero::extra_body` accepts an array of JSON Pointer
// patches applied to the provider's native request body right before it is
// sent.  This lets pi inject provider-specific cache-control markers without
// needing a custom TensorZero plugin.
//
// Supported providers:
//   anthropic     – cache_control: { type: "ephemeral" } on system block and
//                   last content block of last message.
//   amazon-bedrock – cachePoint: { type: "default" } appended to system array
//                   and to the last message's content array.
//   openrouter    – same cache_control shape as anthropic, but the system
//                   prompt lives at messages[0] and content is always sent as
//                   an array by TZ's OpenRouter provider, so we replace the
//                   entire content field of the last user message.
// ---------------------------------------------------------------------------

interface ExtraBodyPatch {
	pointer: string;
	value: unknown;
}

/**
 * Resolve cacheRetention from stream options, falling back to the
 * PI_CACHE_RETENTION environment variable (same logic as anthropic.ts).
 */
function resolveCacheRetention(cacheRetention?: CacheRetention): CacheRetention {
	if (cacheRetention) return cacheRetention;
	if (typeof process !== "undefined" && process.env.PI_CACHE_RETENTION === "long") return "long";
	return "short";
}

/**
 * Build the Anthropic cache_control object.
 * 1h TTL is only honoured by api.anthropic.com (extended caching feature).
 */
function buildAnthropicCacheControl(retention: CacheRetention, gatewayUrl: string): { type: "ephemeral"; ttl?: "1h" } {
	const ttl = retention === "long" && gatewayUrl.includes("api.anthropic.com") ? "1h" : undefined;
	return ttl ? { type: "ephemeral", ttl } : { type: "ephemeral" };
}

/**
 * Compute the index of the last message and its last content block as they
 * will appear in TensorZero's Anthropic-format request.
 *
 * TensorZero coalesces consecutive tool-result messages into a single
 * user message (to satisfy Anthropic's parallel tool-call requirement).
 * We mirror that logic here so the pointer targets the correct block.
 */
function computeAnthropicLastMessageInfo(messages: Message[]): { messageIndex: number; contentIndex: number } | null {
	if (messages.length === 0) return null;

	// Count TZ-format messages, coalescing consecutive toolResult runs.
	let tzCount = 0;
	let i = 0;
	while (i < messages.length) {
		if (messages[i].role === "toolResult") {
			while (i < messages.length && messages[i].role === "toolResult") i++;
			tzCount++;
		} else {
			i++;
			tzCount++;
		}
	}

	const messageIndex = tzCount - 1;

	// Determine contentIndex: how many content blocks does the last TZ message have?
	const lastMsg = messages[messages.length - 1];

	if (lastMsg.role === "toolResult") {
		// Count how many consecutive toolResult messages form the last TZ user message.
		let toolCount = 0;
		let j = messages.length - 1;
		while (j >= 0 && messages[j].role === "toolResult") {
			toolCount++;
			j--;
		}
		return { messageIndex, contentIndex: toolCount - 1 };
	}

	if (lastMsg.role === "user") {
		const content = lastMsg.content;
		if (typeof content === "string") return { messageIndex, contentIndex: 0 };
		return { messageIndex, contentIndex: Math.max(0, content.length - 1) };
	}

	// Last message is an assistant message – shouldn't happen in normal pi flow.
	return null;
}

/**
 * Build tensorzero::extra_body patches for Anthropic's native provider.
 *
 * TZ sends:
 *   { "system": [{ "type": "text", "text": "..." }], "messages": [...] }
 *
 * Patches add cache_control to:
 *   - /system/0/cache_control
 *   - /messages/{N}/content/{M}/cache_control  (last content block of last message)
 */
function buildAnthropicCachePatches(
	context: Context,
	cacheControl: ReturnType<typeof buildAnthropicCacheControl>,
): ExtraBodyPatch[] {
	const patches: ExtraBodyPatch[] = [];

	if (context.systemPrompt) {
		patches.push({ pointer: "/system/0/cache_control", value: cacheControl });
	}

	const lastInfo = computeAnthropicLastMessageInfo(context.messages);
	if (lastInfo) {
		patches.push({
			pointer: `/messages/${lastInfo.messageIndex}/content/${lastInfo.contentIndex}/cache_control`,
			value: cacheControl,
		});
	}

	return patches;
}

/**
 * Build tensorzero::extra_body patches for AWS Bedrock's native provider.
 *
 * Bedrock uses a cachePoint block appended to arrays rather than a
 * cache_control field on existing blocks.  TZ's `-` pointer appends.
 *
 * TZ sends:
 *   { "system": [...], "messages": [...] }
 *
 * Patches append cachePoint to:
 *   - /system/-
 *   - /messages/{N}/content/-
 */
function buildBedrockCachePatches(context: Context): ExtraBodyPatch[] {
	const patches: ExtraBodyPatch[] = [];
	const cachePoint = { cachePoint: { type: "default" } };

	if (context.systemPrompt) {
		patches.push({ pointer: "/system/-", value: cachePoint });
	}

	const lastInfo = computeAnthropicLastMessageInfo(context.messages);
	if (lastInfo) {
		patches.push({ pointer: `/messages/${lastInfo.messageIndex}/content/-`, value: cachePoint });
	}

	return patches;
}

/**
 * Build tensorzero::extra_body patches for Anthropic models routed via
 * OpenRouter through TensorZero.
 *
 * TZ's OpenRouter provider:
 *   - Inserts a system message at messages[0] when a system prompt is present.
 *   - Serialises single-text content blocks as a plain string, making a
 *     field-level patch impossible.
 *
 * Strategy: replace the entire content field of the last user message with an
 * array that includes cache_control on its last text block.  We can build that
 * array because we have the original message content in context.
 *
 * Tool-result messages are kept as role:"tool" in OpenAI format (no coalescing
 * at the OpenRouter level), so the relevant cache point is on the last explicit
 * user message, not the last tool result.
 */
function buildOpenRouterAnthropicCachePatches(context: Context): ExtraBodyPatch[] {
	const patches: ExtraBodyPatch[] = [];
	const cacheControl: { type: "ephemeral" } = { type: "ephemeral" };

	// Find the last UserMessage and its index in context.messages (= OpenAI message index).
	let lastUserIdx = -1;
	let lastUserMsg: UserMessage | null = null;
	for (let i = context.messages.length - 1; i >= 0; i--) {
		if (context.messages[i].role === "user") {
			lastUserIdx = i;
			lastUserMsg = context.messages[i] as UserMessage;
			break;
		}
	}

	if (lastUserIdx === -1 || !lastUserMsg) return patches;

	// TZ's OpenRouter provider inserts the system message at messages[0],
	// shifting all conversation messages by 1.
	const openRouterIdx = context.systemPrompt ? lastUserIdx + 1 : lastUserIdx;

	// Build the replacement content array with cache_control on the last text block.
	const newContent = buildContentArrayWithCacheControl(lastUserMsg.content, cacheControl);
	patches.push({ pointer: `/messages/${openRouterIdx}/content`, value: newContent });

	return patches;
}

/**
 * Convert a UserMessage content value to an array of content-block objects
 * with cache_control added to the last text block.
 */
function buildContentArrayWithCacheControl(
	content: UserMessage["content"],
	cacheControl: { type: "ephemeral" },
): unknown[] {
	if (typeof content === "string") {
		return [{ type: "text", text: content, cache_control: cacheControl }];
	}

	// Array content: find and annotate the last text block.
	const blocks: unknown[] = content.map((block) => ({ ...block }));
	for (let i = blocks.length - 1; i >= 0; i--) {
		const block = blocks[i] as { type: string };
		if (block.type === "text") {
			blocks[i] = { ...block, cache_control: cacheControl };
			break;
		}
	}
	return blocks;
}

/**
 * Compute the tensorzero::extra_body patches needed to enable provider-level
 * prompt caching for the given model, context, and cache-retention setting.
 *
 * Returns an empty array when caching is disabled or the provider is not
 * supported.
 */
function buildProviderCachePatches(
	model: Model<Api>,
	context: Context,
	cacheRetention: CacheRetention,
	gatewayUrl: string,
): ExtraBodyPatch[] {
	if (cacheRetention === "none") return [];

	const provider = model.provider;

	// opencode routes its anthropic-messages models to opencode.ai/zen using the
	// Anthropic Messages format, so the same cache_control patches apply.
	// Only apply to models that actually support Anthropic caching (not minimax, big-pickle, etc.).
	if (provider === "anthropic" || (provider === "opencode" && model.api === "anthropic-messages")) {
		if (!supportsAnthropicCaching(model)) return [];
		const cacheControl = buildAnthropicCacheControl(cacheRetention, gatewayUrl);
		return buildAnthropicCachePatches(context, cacheControl);
	}

	if (provider === "amazon-bedrock") {
		if (!supportsBedrockPromptCaching(model)) return [];
		return buildBedrockCachePatches(context);
	}

	if (provider === "openrouter" && model.id.startsWith("anthropic/")) {
		return buildOpenRouterAnthropicCachePatches(context);
	}

	return [];
}

/**
 * Create a stream function that routes all requests through TensorZero.
 * All inferences share the same episode_id (one per session) so TensorZero
 * can group them for analytics and caching.
 */
export function createTensorZeroStreamFn(
	config: TensorZeroConfig,
	options: { cacheMode?: TensorZeroCacheMode } = {},
): (model: Model<Api>, context: Context, streamOptions?: SimpleStreamOptions) => AssistantMessageEventStream {
	const episodeId = config.episodeId ?? uuidv7();
	const desiredCacheMode = options.cacheMode ?? "on";

	return (model: Model<Api>, context: Context, streamOptions?: SimpleStreamOptions): AssistantMessageEventStream => {
		const gatewayModel = rewriteModelForGateway(model, config);

		const headers = {
			...streamOptions?.headers,
		};

		const extraBody: Record<string, unknown> = {
			...streamOptions?.extraBody,
			"tensorzero::episode_id": episodeId,
			"tensorzero::include_raw_usage": true,
		};

		const existingCacheOptions = extraBody["tensorzero::cache_options"];
		const cacheOptions =
			typeof existingCacheOptions === "object" && existingCacheOptions !== null
				? { ...(existingCacheOptions as Record<string, unknown>) }
				: {};
		cacheOptions.enabled = desiredCacheMode;
		extraBody["tensorzero::cache_options"] = cacheOptions;

		// Inject provider-level prompt-cache patches via tensorzero::extra_body.
		// This runs before rewriteModelForGateway changes the provider, so we
		// still have the original provider name available via `model.provider`.
		const retention = resolveCacheRetention(streamOptions?.cacheRetention);
		const cachePatches = buildProviderCachePatches(model, context, retention, config.gatewayUrl);
		if (cachePatches.length > 0) {
			const existing = extraBody["tensorzero::extra_body"];
			const existingPatches: ExtraBodyPatch[] = Array.isArray(existing) ? existing : [];
			// Caller-supplied patches take precedence; provider cache patches come first.
			extraBody["tensorzero::extra_body"] = [...cachePatches, ...existingPatches];
		}

		const mergedOptions: SimpleStreamOptions = {
			...streamOptions,
			apiKey: config.apiKey || streamOptions?.apiKey || "not-used",
			headers,
			extraBody,
		};

		return streamSimple(gatewayModel, context, mergedOptions);
	};
}
