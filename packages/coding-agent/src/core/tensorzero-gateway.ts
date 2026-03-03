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

export interface TensorZeroConfig {
	gatewayUrl: string;
	apiKey?: string;
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
 * Map pi-mono provider names to TensorZero provider names.
 * TensorZero uses its own provider identifiers in the model name format:
 *   tensorzero::model_name::<tz_provider>::<model_id>
 */
function mapProviderToTensorZero(provider: string): string {
	const providerMap: Record<string, string> = {
		"amazon-bedrock": "aws_bedrock",
		anthropic: "anthropic",
		openai: "openai",
		"azure-openai-responses": "azure",
		google: "google_ai_studio_gemini",
		"google-vertex": "gcp_vertex_gemini",
		xai: "xai",
		mistral: "mistral",
		groq: "groq",
		openrouter: "openai", // OpenRouter is OpenAI-compatible
	};
	return providerMap[provider] ?? provider;
}

/**
 * Rewrite a model to route through TensorZero's OpenAI-compatible endpoint.
 * Uses the tensorzero::model_name::<provider>::<model_id> format.
 */
function rewriteModelForGateway(model: Model<Api>, config: TensorZeroConfig): Model<"openai-completions"> {
	const tzProvider = mapProviderToTensorZero(model.provider);
	return {
		...model,
		id: `tensorzero::model_name::${tzProvider}::${model.id}`,
		api: "openai-completions" as const,
		baseUrl: `${config.gatewayUrl}/openai/v1`,
	};
}

/**
 * Create a stream function that routes all requests through TensorZero.
 */
export function createTensorZeroStreamFn(
	config: TensorZeroConfig,
): (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => AssistantMessageEventStream {
	return (model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream => {
		const gatewayModel = rewriteModelForGateway(model, config);

		const mergedOptions: SimpleStreamOptions = {
			...options,
			apiKey: config.apiKey || options?.apiKey || "not-used",
			headers: {
				...options?.headers,
			},
		};

		return streamSimple(gatewayModel, context, mergedOptions);
	};
}
