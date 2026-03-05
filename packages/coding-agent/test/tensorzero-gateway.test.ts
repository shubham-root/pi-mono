import { type Context, getModel, type Message, type Model } from "@mariozechner/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsManager } from "../src/core/settings-manager.js";
import {
	createTensorZeroStreamFn,
	getTensorZeroConfig,
	type TensorZeroConfig,
} from "../src/core/tensorzero-gateway.js";

// Capture the last call arguments so tests can inspect them
let capturedModel: unknown;
let _capturedContext: unknown;
let capturedOptions: unknown;

vi.mock("@mariozechner/pi-ai", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@mariozechner/pi-ai")>();
	return {
		...actual,
		streamSimple: vi.fn((...args: unknown[]) => {
			[capturedModel, _capturedContext, capturedOptions] = args;
			// Return a minimal async generator that immediately completes
			return (async function* () {})();
		}),
	};
});

const fakeConfig: TensorZeroConfig = {
	gatewayUrl: "http://localhost:3000",
	apiKey: "test-api-key",
	episodeId: "00000000-0000-0000-0000-000000000001",
};

const fakeModel = getModel("anthropic", "claude-opus-4-5")!;
const fakeContext: Context = { messages: [] };

describe("createTensorZeroStreamFn – cache options", () => {
	beforeEach(() => {
		capturedModel = undefined;
		_capturedContext = undefined;
		capturedOptions = undefined;
	});

	it('sets tensorzero::cache_options.enabled to "on" by default', () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		streamFn(fakeModel, fakeContext);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const cacheOptions = opts.extraBody["tensorzero::cache_options"] as Record<string, unknown>;
		expect(cacheOptions).toBeDefined();
		expect(cacheOptions.enabled).toBe("on");
	});

	it.each(["on", "off", "write_only", "read_only"] as const)(
		'sets tensorzero::cache_options.enabled to "%s" when cacheMode is "%s"',
		(mode) => {
			const streamFn = createTensorZeroStreamFn(fakeConfig, { cacheMode: mode });
			streamFn(fakeModel, fakeContext);

			const opts = capturedOptions as { extraBody: Record<string, unknown> };
			const cacheOptions = opts.extraBody["tensorzero::cache_options"] as Record<string, unknown>;
			expect(cacheOptions).toBeDefined();
			expect(cacheOptions.enabled).toBe(mode);
		},
	);

	it("preserves existing cache_options fields when overriding enabled", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig, { cacheMode: "read_only" });
		streamFn(fakeModel, fakeContext, {
			extraBody: { "tensorzero::cache_options": { max_age_s: 3600 } },
		});

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const cacheOptions = opts.extraBody["tensorzero::cache_options"] as Record<string, unknown>;
		expect(cacheOptions.enabled).toBe("read_only");
		expect(cacheOptions.max_age_s).toBe(3600);
	});

	it("sets tensorzero::episode_id in extraBody", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		streamFn(fakeModel, fakeContext);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::episode_id"]).toBe(fakeConfig.episodeId);
	});

	it("rewrites model to route through TensorZero OpenAI-compatible endpoint", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		streamFn(fakeModel, fakeContext);

		const model = capturedModel as Model<"openai-completions">;
		expect(model.api).toBe("openai-completions");
		expect(model.id).toMatch(/^tensorzero::model_name::/);
		expect(model.baseUrl).toBe("http://localhost:3000/openai/v1");
	});

	it("uses provided apiKey in merged options", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		streamFn(fakeModel, fakeContext);

		const opts = capturedOptions as { apiKey: string };
		expect(opts.apiKey).toBe("test-api-key");
	});
});

// ---------------------------------------------------------------------------
// Helper builders for test messages
// ---------------------------------------------------------------------------

function userMsg(text: string): Message {
	return { role: "user", content: text, timestamp: 0 };
}

function assistantMsg(): Message {
	return {
		role: "assistant",
		content: [],
		api: "anthropic",
		provider: "anthropic",
		model: "claude-opus-4-5",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 0,
	};
}

function toolResultMsg(id = "tool_1"): Message {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: "bash",
		content: [{ type: "text", text: "result" }],
		isError: false,
		timestamp: 0,
	};
}

// ---------------------------------------------------------------------------
// Anthropic provider-level prompt caching
// ---------------------------------------------------------------------------

describe("createTensorZeroStreamFn – Anthropic provider-level caching", () => {
	const anthropicModel = getModel("anthropic", "claude-opus-4-5")!;

	beforeEach(() => {
		capturedModel = undefined;
		_capturedContext = undefined;
		capturedOptions = undefined;
		delete process.env.PI_CACHE_RETENTION;
	});

	afterEach(() => {
		delete process.env.PI_CACHE_RETENTION;
	});

	it("adds no cache patches when cacheRetention is none", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "You are helpful.", messages: [userMsg("hi")] };
		streamFn(anthropicModel, ctx, { cacheRetention: "none" });

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::extra_body"]).toBeUndefined();
	});

	it("adds system and last-message cache patches for anthropic (short retention)", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "You are helpful.", messages: [userMsg("hello")] };
		streamFn(anthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		expect(patches).toBeDefined();

		const systemPatch = patches.find((p) => p.pointer === "/system/0/cache_control");
		expect(systemPatch?.value).toEqual({ type: "ephemeral" });

		const msgPatch = patches.find((p) => p.pointer === "/messages/0/content/0/cache_control");
		expect(msgPatch?.value).toEqual({ type: "ephemeral" });
	});

	it("omits system patch when no system prompt", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { messages: [userMsg("hello")] };
		streamFn(anthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string }>;
		expect(patches.some((p) => p.pointer.startsWith("/system"))).toBe(false);
	});

	it("adds 1h TTL when cacheRetention is long and gatewayUrl includes api.anthropic.com", () => {
		const anthropicDirectConfig: TensorZeroConfig = {
			...fakeConfig,
			gatewayUrl: "https://api.anthropic.com",
		};
		const streamFn = createTensorZeroStreamFn(anthropicDirectConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hi")] };
		streamFn(anthropicModel, ctx, { cacheRetention: "long" });

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		const systemPatch = patches.find((p) => p.pointer === "/system/0/cache_control");
		expect(systemPatch?.value).toEqual({ type: "ephemeral", ttl: "1h" });
	});

	it("does not add 1h TTL when gatewayUrl is not api.anthropic.com", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig); // localhost
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hi")] };
		streamFn(anthropicModel, ctx, { cacheRetention: "long" });

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		const systemPatch = patches.find((p) => p.pointer === "/system/0/cache_control");
		expect(systemPatch?.value).toEqual({ type: "ephemeral" });
	});

	it("computes correct message index after multi-turn with single tool result", () => {
		// [user, assistant, toolResult, assistant, user] → 5 TZ messages, last at index 4
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = {
			messages: [userMsg("q"), assistantMsg(), toolResultMsg(), assistantMsg(), userMsg("q2")],
		};
		streamFn(anthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string }>;
		expect(patches.some((p) => p.pointer === "/messages/4/content/0/cache_control")).toBe(true);
	});

	it("coalesces parallel tool results – message index accounts for grouping", () => {
		// [user, assistant, toolResult1, toolResult2, user] → 4 TZ messages (tool results coalesced)
		// last user = index 3, contentIndex = 0
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = {
			messages: [userMsg("q"), assistantMsg(), toolResultMsg("t1"), toolResultMsg("t2"), userMsg("q2")],
		};
		streamFn(anthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string }>;
		expect(patches.some((p) => p.pointer === "/messages/3/content/0/cache_control")).toBe(true);
	});

	it("sets contentIndex to last tool result in a trailing tool-result group", () => {
		// [user, assistant, toolResult1, toolResult2] → last TZ message is user with 2 tool_result blocks
		// messageIndex = 2, contentIndex = 1
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = {
			messages: [userMsg("q"), assistantMsg(), toolResultMsg("t1"), toolResultMsg("t2")],
		};
		streamFn(anthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string }>;
		expect(patches.some((p) => p.pointer === "/messages/2/content/1/cache_control")).toBe(true);
	});

	it("caller-supplied extra_body patches are preserved after cache patches", () => {
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hi")] };
		const callerPatch = { pointer: "/temperature", value: 0.5 };
		streamFn(anthropicModel, ctx, {
			extraBody: { "tensorzero::extra_body": [callerPatch] },
		});

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string }>;
		// Caller patch should be present
		expect(patches.some((p) => p.pointer === "/temperature")).toBe(true);
		// Cache patches should also be present
		expect(patches.some((p) => p.pointer === "/system/0/cache_control")).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// opencode provider – Anthropic-messages models
// ---------------------------------------------------------------------------

describe("createTensorZeroStreamFn – opencode Anthropic-messages caching", () => {
	// claude-sonnet-4-6 uses api: "anthropic-messages" on opencode
	const opencodeAnthropicModel = getModel("opencode", "claude-sonnet-4-6")!;
	// kimi-k2.5 uses api: "openai-completions" on opencode – should NOT get patches
	const opencodeOpenAIModel = getModel("opencode", "kimi-k2.5")!;

	beforeEach(() => {
		capturedModel = undefined;
		_capturedContext = undefined;
		capturedOptions = undefined;
	});

	it("adds system and last-message cache patches for anthropic-messages opencode models", () => {
		if (!opencodeAnthropicModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hello")] };
		streamFn(opencodeAnthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		expect(patches).toBeDefined();
		expect(patches.some((p) => p.pointer === "/system/0/cache_control")).toBe(true);
		expect(patches.some((p) => p.pointer === "/messages/0/content/0/cache_control")).toBe(true);
	});

	it("adds no cache patches for openai-completions opencode models", () => {
		if (!opencodeOpenAIModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hello")] };
		streamFn(opencodeOpenAIModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::extra_body"]).toBeUndefined();
	});

	it("adds no patches when cacheRetention is none", () => {
		if (!opencodeAnthropicModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hello")] };
		streamFn(opencodeAnthropicModel, ctx, { cacheRetention: "none" });

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::extra_body"]).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// AWS Bedrock provider-level prompt caching
// ---------------------------------------------------------------------------

describe("createTensorZeroStreamFn – Bedrock provider-level caching", () => {
	const bedrockModel = getModel("amazon-bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0")!;

	beforeEach(() => {
		capturedModel = undefined;
		_capturedContext = undefined;
		capturedOptions = undefined;
	});

	it("adds cachePoint patches for bedrock with system and user message", () => {
		if (!bedrockModel) return; // skip if model not in registry
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hi")] };
		streamFn(bedrockModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		expect(patches).toBeDefined();

		const systemPatch = patches.find((p) => p.pointer === "/system/-");
		expect(systemPatch?.value).toEqual({ cachePoint: { type: "default" } });

		const msgPatch = patches.find((p) => p.pointer === "/messages/0/content/-");
		expect(msgPatch?.value).toEqual({ cachePoint: { type: "default" } });
	});

	it("adds no patches when cacheRetention is none", () => {
		if (!bedrockModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hi")] };
		streamFn(bedrockModel, ctx, { cacheRetention: "none" });

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::extra_body"]).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// OpenRouter (Anthropic models) provider-level prompt caching
// ---------------------------------------------------------------------------

describe("createTensorZeroStreamFn – OpenRouter Anthropic caching", () => {
	const openRouterAnthropicModel = getModel("openrouter", "anthropic/claude-3.5-haiku")!;
	const openRouterOtherModel = getModel("openrouter", "openai/gpt-4")!;

	beforeEach(() => {
		capturedModel = undefined;
		_capturedContext = undefined;
		capturedOptions = undefined;
	});

	it("adds no patches for non-Anthropic OpenRouter models", () => {
		if (!openRouterOtherModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hi")] };
		streamFn(openRouterOtherModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::extra_body"]).toBeUndefined();
	});

	it("replaces last user message content with cache_control array (with system prompt)", () => {
		if (!openRouterAnthropicModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		// With system prompt: TZ inserts system at messages[0], so lastUser shifts to index 1.
		const ctx: Context = { systemPrompt: "sys", messages: [userMsg("hello world")] };
		streamFn(openRouterAnthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		expect(patches).toBeDefined();

		const contentPatch = patches.find((p) => p.pointer === "/messages/1/content");
		expect(contentPatch?.value).toEqual([
			{ type: "text", text: "hello world", cache_control: { type: "ephemeral" } },
		]);
	});

	it("targets index 0 when there is no system prompt", () => {
		if (!openRouterAnthropicModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { messages: [userMsg("hello")] };
		streamFn(openRouterAnthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string; value: unknown }>;
		const contentPatch = patches.find((p) => p.pointer === "/messages/0/content");
		expect(contentPatch).toBeDefined();
	});

	it("targets last UserMessage even when conversation ends with tool results", () => {
		if (!openRouterAnthropicModel) return;
		// [user@0, assistant@1, toolResult@2] — last user is at pi index 0 → OpenRouter index 1 (system at 0)
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = {
			systemPrompt: "sys",
			messages: [userMsg("q"), assistantMsg(), toolResultMsg()],
		};
		streamFn(openRouterAnthropicModel, ctx);

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		const patches = opts.extraBody["tensorzero::extra_body"] as Array<{ pointer: string }>;
		// user is at pi index 0 → OpenRouter index 0+1=1
		expect(patches.some((p) => p.pointer === "/messages/1/content")).toBe(true);
	});

	it("adds no patches when cacheRetention is none", () => {
		if (!openRouterAnthropicModel) return;
		const streamFn = createTensorZeroStreamFn(fakeConfig);
		const ctx: Context = { messages: [userMsg("hi")] };
		streamFn(openRouterAnthropicModel, ctx, { cacheRetention: "none" });

		const opts = capturedOptions as { extraBody: Record<string, unknown> };
		expect(opts.extraBody["tensorzero::extra_body"]).toBeUndefined();
	});
});

describe("TensorZero gateway gating", () => {
	const originalEnv = process.env.TENSORZERO_GATEWAY_URL;

	afterEach(() => {
		if (originalEnv === undefined) {
			delete process.env.TENSORZERO_GATEWAY_URL;
		} else {
			process.env.TENSORZERO_GATEWAY_URL = originalEnv;
		}
	});

	describe("getTensorZeroConfig", () => {
		it("returns undefined when TENSORZERO_GATEWAY_URL is not set", () => {
			delete process.env.TENSORZERO_GATEWAY_URL;
			expect(getTensorZeroConfig()).toBeUndefined();
		});

		it("returns a config when TENSORZERO_GATEWAY_URL is set", () => {
			process.env.TENSORZERO_GATEWAY_URL = "http://localhost:3000";
			const config = getTensorZeroConfig();
			expect(config).toBeDefined();
			expect(config?.gatewayUrl).toBe("http://localhost:3000");
		});

		it("strips trailing slashes from gatewayUrl", () => {
			process.env.TENSORZERO_GATEWAY_URL = "http://localhost:3000///";
			const config = getTensorZeroConfig();
			expect(config?.gatewayUrl).toBe("http://localhost:3000");
		});
	});

	describe("settings gate (tensorZeroGateway)", () => {
		it("defaults to true when TENSORZERO_GATEWAY_URL is set and setting is not explicitly configured", () => {
			process.env.TENSORZERO_GATEWAY_URL = "http://localhost:3000";
			const settings = SettingsManager.inMemory({});
			expect(settings.getTensorZeroGateway()).toBe(true);
		});

		it("returns false when tensorZeroGateway is explicitly false, even if URL is set", () => {
			process.env.TENSORZERO_GATEWAY_URL = "http://localhost:3000";
			const settings = SettingsManager.inMemory({ tensorZeroGateway: false });
			expect(settings.getTensorZeroGateway()).toBe(false);
		});

		it("returns false when TENSORZERO_GATEWAY_URL is not set and setting is not configured", () => {
			delete process.env.TENSORZERO_GATEWAY_URL;
			const settings = SettingsManager.inMemory({});
			expect(settings.getTensorZeroGateway()).toBe(false);
		});
	});

	describe("useTensorZero combined gate", () => {
		it("is false when URL is missing even if setting is true", () => {
			delete process.env.TENSORZERO_GATEWAY_URL;
			const settings = SettingsManager.inMemory({ tensorZeroGateway: true });
			const config = getTensorZeroConfig();
			const useTensorZero = config && settings.getTensorZeroGateway();
			expect(useTensorZero).toBeFalsy();
		});

		it("is false when setting is false even if URL is set", () => {
			process.env.TENSORZERO_GATEWAY_URL = "http://localhost:3000";
			const settings = SettingsManager.inMemory({ tensorZeroGateway: false });
			const config = getTensorZeroConfig();
			const useTensorZero = config && settings.getTensorZeroGateway();
			expect(useTensorZero).toBeFalsy();
		});

		it("is truthy only when both URL is set and setting is true", () => {
			process.env.TENSORZERO_GATEWAY_URL = "http://localhost:3000";
			const settings = SettingsManager.inMemory({ tensorZeroGateway: true });
			const config = getTensorZeroConfig();
			const useTensorZero = config && settings.getTensorZeroGateway();
			expect(useTensorZero).toBeTruthy();
		});
	});
});
