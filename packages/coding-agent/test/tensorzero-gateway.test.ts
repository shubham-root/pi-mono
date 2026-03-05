import { type Context, getModel, type Model } from "@mariozechner/pi-ai";
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
