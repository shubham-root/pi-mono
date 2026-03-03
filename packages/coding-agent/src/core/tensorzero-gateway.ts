/**
 * TensorZero gateway integration.
 *
 * When TENSORZERO_GATEWAY_URL is set, all LLM requests are routed through the
 * TensorZero gateway's OpenAI-compatible endpoint. The gateway handles provider
 * routing, observability, and fallbacks.
 *
 * Environment variables:
 *   TENSORZERO_GATEWAY_URL  - Gateway base URL (e.g. https://13-232-113-182.sslip.io)
 *   TENSORZERO_GATEWAY_USER - Basic auth username (optional)
 *   TENSORZERO_GATEWAY_PASS - Basic auth password (optional)
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
	username?: string;
	password?: string;
}

export function getTensorZeroConfig(): TensorZeroConfig | undefined {
	const gatewayUrl = process.env.TENSORZERO_GATEWAY_URL;
	if (!gatewayUrl) return undefined;

	return {
		gatewayUrl: gatewayUrl.replace(/\/+$/, ""),
		username: process.env.TENSORZERO_GATEWAY_USER,
		password: process.env.TENSORZERO_GATEWAY_PASS,
	};
}

/**
 * Rewrite a model to route through TensorZero's OpenAI-compatible endpoint.
 * The original model ID is preserved so TensorZero can route to the correct backend.
 */
function rewriteModelForGateway(model: Model<Api>, config: TensorZeroConfig): Model<"openai-completions"> {
	return {
		...model,
		api: "openai-completions" as const,
		baseUrl: `${config.gatewayUrl}/v1`,
	};
}

/**
 * Build auth headers for the gateway (nginx basic auth).
 */
function buildGatewayHeaders(config: TensorZeroConfig): Record<string, string> {
	const headers: Record<string, string> = {};
	if (config.username && config.password) {
		const credentials = Buffer.from(`${config.username}:${config.password}`).toString("base64");
		headers.Authorization = `Basic ${credentials}`;
	}
	return headers;
}

/**
 * Create a stream function that routes all requests through TensorZero.
 */
export function createTensorZeroStreamFn(
	config: TensorZeroConfig,
): (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => AssistantMessageEventStream {
	return (model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream => {
		const gatewayModel = rewriteModelForGateway(model, config);
		const gatewayHeaders = buildGatewayHeaders(config);

		const mergedOptions: SimpleStreamOptions = {
			...options,
			headers: {
				...gatewayHeaders,
				...options?.headers,
			},
		};

		return streamSimple(gatewayModel, context, mergedOptions);
	};
}
