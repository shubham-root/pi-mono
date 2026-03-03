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
	type Context,
	type Model,
	type SimpleStreamOptions,
	streamSimple,
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
		};

		const existingCacheOptions = extraBody.cache_options;
		const cacheOptions =
			typeof existingCacheOptions === "object" && existingCacheOptions !== null
				? { ...(existingCacheOptions as Record<string, unknown>) }
				: {};
		cacheOptions.enabled = desiredCacheMode;
		extraBody.cache_options = cacheOptions;

		const mergedOptions: SimpleStreamOptions = {
			...streamOptions,
			apiKey: config.apiKey || streamOptions?.apiKey || "not-used",
			headers,
			extraBody,
		};

		return streamSimple(gatewayModel, context, mergedOptions);
	};
}
