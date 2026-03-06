# Branch Changelog (our.pi)

Changes in this branch on top of upstream `main`. Kept here to avoid merge conflicts with upstream `packages/*/CHANGELOG.md`.

---

## Unreleased

### Fixed

- Fixed TensorZero abort+resume causing `502` errors from Bedrock and other providers: `computeAnthropicLastMessageInfo` now skips aborted/error assistant messages when computing the `tensorzero::extra_body` JSON Pointer index, matching what `transform-messages.ts` does when building the provider request. Previously, an aborted session left a trailing assistant message in the conversation history; when the user resumed and sent a new prompt, the pointer targeted a message index that didn't exist in the provider's request body (e.g. `/messages/2/content/-` on a 2-element array).

---

## v0.56.3 (our.pi)

### Fixed

- Fixed TensorZero gateway session resuming: wrap the output stream to restore the original model `id` and `api` in every `AssistantMessage` before it is persisted. Previously the TZ-internal rewritten values (`tensorzero::model_name::...` and `openai-completions`) were stored, causing model restoration to silently fail on resume.
- Fixed auto-retry leaving a trailing error `AssistantMessage` in agent state when max retries are exhausted.
- Fixed TensorZero Bedrock cache-patching off-by-one when a trailing error/aborted assistant message was present.
- Fixed auto-retry incorrectly retrying permanent `400 Bad Request` errors wrapped as `502` by TensorZero.
- Fixed TensorZero gateway sending `strict: false` in tool definitions to providers that reject the field (e.g. opencode.ai/zen Kimi K2.5). Emits `tensorzero::extra_body` delete patches at `/tools/${i}/strict` (the correct TensorZero serialization path) for models with `compat.supportsStrictMode: false`.
- Fixed TensorZero cache token extraction for AWS Bedrock (camelCase `cacheReadInputTokens`/`cacheWriteInputTokens` raw usage fields).
- Fixed TensorZero raw usage accumulation: raw_usage entries are now collected across all streaming chunks, not just the final one.

### Added

- Added TensorZero footer indicator showing gateway status.
- Added TensorZero cache mode setting (`on`/`off`/`write_only`/`read_only`) with UI in `/settings`.
- Added provider-level prompt caching via `tensorzero::extra_body` patches for Anthropic, AWS Bedrock, and OpenRouter (Anthropic models).
- Added `supportsStrictMode: false` compat flag to all opencode and opencode-go `openai-completions` models.

---

## v0.56.2 (our.pi)

### Fixed

- Fixed TensorZero gateway cache options using wrong key: changed `cache_options` to `tensorzero::cache_options`.

### Added

- Added TensorZero gateway toggle in `/settings` menu.
- Added TensorZero episode ID tied to session ID.
- Added Bedrock routing support through TensorZero.
