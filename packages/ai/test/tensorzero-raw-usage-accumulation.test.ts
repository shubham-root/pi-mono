/**
 * Tests that TensorZero raw_usage entries are accumulated across streaming chunks.
 *
 * TensorZero streams tensorzero_raw_usage on early chunks (e.g., Anthropic
 * message_start with cache token counts) but sends aggregated usage on the
 * final chunk without tensorzero_raw_usage. The openai-completions provider
 * must accumulate raw_usage from all chunks so cache stats are available when
 * computing the final usage.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AssistantMessage, Model } from "../src/types.js";

// We need to mock the OpenAI client to simulate TZ streaming behavior
const mockCreate = vi.fn();

vi.mock("openai", () => {
	return {
		default: class OpenAI {
			chat = {
				completions: {
					create: mockCreate,
				},
			};
		},
	};
});

// Import after mocking
const { streamOpenAICompletions } = await import("../src/providers/openai-completions.js");

function makeTzModel(): Model<"openai-completions"> {
	return {
		id: "tensorzero::model_name::anthropic::claude-sonnet-4-20250514",
		name: "Claude Sonnet 4 (via TZ)",
		api: "openai-completions",
		provider: "anthropic",
		baseUrl: "http://localhost:3000/openai/v1",
		reasoning: false,
		input: ["text", "image"],
		cost: {
			input: 3.0,
			output: 15.0,
			cacheRead: 0.3,
			cacheWrite: 3.75,
		},
		contextWindow: 200000,
		maxTokens: 16384,
	};
}

/**
 * Simulate TZ streaming: raw_usage on early chunk, aggregated usage on final chunk.
 *
 * This mirrors TZ's `create_stream` behavior:
 * 1. Early chunks have content + tensorzero_raw_usage (but no usage)
 * 2. Final chunk has aggregated usage (but no tensorzero_raw_usage)
 */
async function* simulateTzStream(opts: {
	cacheRead: number;
	cacheWrite: number;
	inputTokens: number;
	outputTokens: number;
}) {
	// Chunk 1: Anthropic message_start — has raw_usage with cache stats, no usage
	yield {
		id: "chatcmpl-tz1",
		object: "chat.completion.chunk",
		choices: [
			{
				index: 0,
				delta: { role: "assistant", content: "" },
				finish_reason: null,
			},
		],
		tensorzero_raw_usage: [
			{
				model_inference_id: "00000000-0000-7000-8000-000000000001",
				provider_type: "anthropic",
				api_type: "chat_completions",
				data: {
					input_tokens: opts.inputTokens - opts.cacheRead - opts.cacheWrite,
					output_tokens: 0,
					cache_read_input_tokens: opts.cacheRead,
					cache_creation_input_tokens: opts.cacheWrite,
				},
			},
		],
	};

	// Chunk 2: Content delta — no usage, no raw_usage
	yield {
		id: "chatcmpl-tz1",
		object: "chat.completion.chunk",
		choices: [
			{
				index: 0,
				delta: { content: "Hello, world!" },
				finish_reason: null,
			},
		],
	};

	// Chunk 3: Finish + raw_usage from message_delta (output_tokens only)
	yield {
		id: "chatcmpl-tz1",
		object: "chat.completion.chunk",
		choices: [
			{
				index: 0,
				delta: {},
				finish_reason: "stop",
			},
		],
		tensorzero_raw_usage: [
			{
				model_inference_id: "00000000-0000-7000-8000-000000000001",
				provider_type: "anthropic",
				api_type: "chat_completions",
				data: {
					output_tokens: opts.outputTokens,
				},
			},
		],
	};

	// Chunk 4: Final chunk — aggregated usage, no raw_usage
	yield {
		id: "chatcmpl-tz1",
		object: "chat.completion.chunk",
		choices: [],
		usage: {
			// TZ aggregates: prompt_tokens = input + cache_read + cache_write
			prompt_tokens: opts.inputTokens,
			completion_tokens: opts.outputTokens,
			total_tokens: opts.inputTokens + opts.outputTokens,
		},
	};
}

/**
 * Simulate TZ streaming for Bedrock with cache stats.
 */
async function* simulateTzBedrockStream(opts: {
	cacheRead: number;
	cacheWrite: number;
	inputTokens: number;
	outputTokens: number;
}) {
	// Chunk 1: Raw usage with Bedrock cache stats
	yield {
		id: "chatcmpl-tz2",
		object: "chat.completion.chunk",
		choices: [
			{
				index: 0,
				delta: { role: "assistant", content: "Hi" },
				finish_reason: null,
			},
		],
		tensorzero_raw_usage: [
			{
				model_inference_id: "00000000-0000-7000-8000-000000000002",
				provider_type: "aws_bedrock",
				api_type: "chat_completions",
				data: {
					inputTokens: opts.inputTokens - opts.cacheRead - opts.cacheWrite,
					outputTokens: 0,
					cacheReadInputTokens: opts.cacheRead,
					cacheWriteInputTokens: opts.cacheWrite,
				},
			},
		],
	};

	// Chunk 2: Final with aggregated usage, no raw_usage
	yield {
		id: "chatcmpl-tz2",
		object: "chat.completion.chunk",
		choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
		usage: {
			prompt_tokens: opts.inputTokens,
			completion_tokens: opts.outputTokens,
			total_tokens: opts.inputTokens + opts.outputTokens,
		},
	};
}

/**
 * Simulate a standard OpenAI stream (no TZ). Cache info comes via
 * prompt_tokens_details.cached_tokens on the final usage chunk.
 */
async function* simulateStandardOpenAIStream(opts: {
	cachedTokens: number;
	inputTokens: number;
	outputTokens: number;
}) {
	yield {
		id: "chatcmpl-oai",
		object: "chat.completion.chunk",
		choices: [
			{
				index: 0,
				delta: { role: "assistant", content: "Hello" },
				finish_reason: null,
			},
		],
	};

	yield {
		id: "chatcmpl-oai",
		object: "chat.completion.chunk",
		choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
		usage: {
			prompt_tokens: opts.inputTokens,
			completion_tokens: opts.outputTokens,
			total_tokens: opts.inputTokens + opts.outputTokens,
			prompt_tokens_details: { cached_tokens: opts.cachedTokens },
		},
	};
}

describe("TensorZero raw_usage accumulation across streaming chunks", () => {
	beforeEach(() => {
		mockCreate.mockReset();
	});

	it("accumulates Anthropic cache stats from early chunks for final usage", async () => {
		const model = makeTzModel();
		mockCreate.mockResolvedValue(
			simulateTzStream({
				cacheRead: 50000,
				cacheWrite: 10000,
				inputTokens: 65000, // 5000 non-cached + 50000 cache_read + 10000 cache_write
				outputTokens: 1000,
			}),
		);

		const stream = streamOpenAICompletions(model, { messages: [] }, { apiKey: "test" });

		let result: AssistantMessage | undefined;
		for await (const event of stream) {
			if (event.type === "done") {
				result = event.message;
			}
		}

		expect(result).toBeDefined();
		expect(result!.usage.cacheRead).toBe(50000);
		expect(result!.usage.cacheWrite).toBe(10000);
		// input = prompt_tokens - cacheRead - cacheWrite = 65000 - 50000 - 10000 = 5000
		expect(result!.usage.input).toBe(5000);
		expect(result!.usage.output).toBe(1000);
		expect(result!.usage.totalTokens).toBe(5000 + 1000 + 50000 + 10000);
	});

	it("accumulates Bedrock cache stats from early chunks for final usage", async () => {
		const model: Model<"openai-completions"> = {
			...makeTzModel(),
			id: "tensorzero::model_name::aws_bedrock::us.anthropic.claude-haiku-4-5-20251001-v1:0",
			provider: "amazon-bedrock",
			cost: { input: 1.0, output: 5.0, cacheRead: 0.1, cacheWrite: 1.25 },
		};
		mockCreate.mockResolvedValue(
			simulateTzBedrockStream({
				cacheRead: 30000,
				cacheWrite: 5000,
				inputTokens: 40000, // 5000 non-cached + 30000 read + 5000 write
				outputTokens: 500,
			}),
		);

		const stream = streamOpenAICompletions(model, { messages: [] }, { apiKey: "test" });

		let result: AssistantMessage | undefined;
		for await (const event of stream) {
			if (event.type === "done") {
				result = event.message;
			}
		}

		expect(result).toBeDefined();
		expect(result!.usage.cacheRead).toBe(30000);
		expect(result!.usage.cacheWrite).toBe(5000);
		expect(result!.usage.input).toBe(5000);
		expect(result!.usage.output).toBe(500);
	});

	it("falls back to prompt_tokens_details.cached_tokens when no TZ raw_usage", async () => {
		const model: Model<"openai-completions"> = {
			...makeTzModel(),
			id: "gpt-4o",
			provider: "openai",
			baseUrl: "https://api.openai.com/v1",
			cost: { input: 2.5, output: 10.0, cacheRead: 1.25, cacheWrite: 0 },
		};
		mockCreate.mockResolvedValue(
			simulateStandardOpenAIStream({
				cachedTokens: 20000,
				inputTokens: 25000, // OpenAI includes cached in prompt_tokens
				outputTokens: 800,
			}),
		);

		const stream = streamOpenAICompletions(model, { messages: [] }, { apiKey: "test" });

		let result: AssistantMessage | undefined;
		for await (const event of stream) {
			if (event.type === "done") {
				result = event.message;
			}
		}

		expect(result).toBeDefined();
		expect(result!.usage.cacheRead).toBe(20000);
		expect(result!.usage.cacheWrite).toBe(0);
		// input = prompt_tokens - cached = 25000 - 20000 = 5000
		expect(result!.usage.input).toBe(5000);
		expect(result!.usage.output).toBe(800);
	});

	it("handles empty TZ raw_usage array (no recognized provider)", async () => {
		const model = makeTzModel();

		async function* emptyRawUsageStream() {
			yield {
				id: "chatcmpl-tz3",
				object: "chat.completion.chunk",
				choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }],
				tensorzero_raw_usage: [], // empty
			};
			yield {
				id: "chatcmpl-tz3",
				object: "chat.completion.chunk",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
			};
		}

		mockCreate.mockResolvedValue(emptyRawUsageStream());

		const stream = streamOpenAICompletions(model, { messages: [] }, { apiKey: "test" });

		let result: AssistantMessage | undefined;
		for await (const event of stream) {
			if (event.type === "done") {
				result = event.message;
			}
		}

		expect(result).toBeDefined();
		expect(result!.usage.cacheRead).toBe(0);
		expect(result!.usage.cacheWrite).toBe(0);
		expect(result!.usage.input).toBe(100);
		expect(result!.usage.output).toBe(50);
	});
});
