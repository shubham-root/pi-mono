# Execution Plan: pi-rs (Rust Port)

This document is the task-level breakdown of [RFC-rust-port.md](./RFC-rust-port.md).
Each task has dependencies, estimated effort, and acceptance criteria.

---

## Phase 0: Project Setup (Week 0)

### 0.1 Repository & Toolchain
- [ ] Create `pi-rs/` directory at repo root
- [ ] Initialize Cargo workspace with `[workspace]` in root `Cargo.toml`
- [ ] Add `.cargo/config.toml` with linker settings for cross-compilation
- [ ] Add `rust-toolchain.toml` pinning stable channel + `wasm32-wasip2` target
- [ ] Add `deny.toml` for `cargo-deny` (license + advisory audit)
- [ ] Add `clippy.toml` with strict lints
- [ ] Add CI workflow: `cargo check`, `cargo clippy`, `cargo test`, `cargo deny check`
- [ ] Create all crate directories with minimal `Cargo.toml` + `src/lib.rs` stubs

**Acceptance**: `cargo check` passes on empty workspace. CI green.

---

## Phase 1: AI Client & Streaming (Weeks 1-4)

The foundation. Everything else depends on being able to talk to an LLM.

### 1.1 Core Types (`pi-ai/src/types.rs`)
**Deps**: None | **Effort**: 2 days | **Lines ref**: `packages/ai/src/types.ts` (453 lines)

- [ ] Define `Message` enum: `User`, `Assistant`, `ToolResult`
- [ ] Define `Content` enum: `Text`, `Image`, `Thinking`, `ToolCall`
- [ ] Define `Model` struct: id, name, api, provider, baseUrl, reasoning, cost, contextWindow, maxTokens, compat
- [ ] Define `Context` struct: systemPrompt, messages, tools
- [ ] Define `Usage` struct: input, output, cacheRead, cacheWrite, totalTokens, cost
- [ ] Define `StreamEvent` enum: Start, TextDelta, ThinkingDelta, ToolCallDelta, Usage, Stop, Error
- [ ] Define `ToolSchema` (JSON Schema representation)
- [ ] Define `StreamOptions`: apiKey, headers, signal, reasoningEffort, cacheRetention
- [ ] Implement `serde::Serialize` / `Deserialize` for all types
- [ ] Unit tests for serialization round-trips

**Acceptance**: All message types serialize to JSON matching the TS version's format.

### 1.2 SSE Parser (`pi-ai/src/stream.rs`)
**Deps**: 1.1 | **Effort**: 2 days | **Lines ref**: `packages/ai/src/utils/event-stream.ts` (87 lines) + inline in providers

- [ ] Implement SSE line parser (handles `data:`, `event:`, multiline data, `[DONE]`)
- [ ] Create `EventStream` type wrapping `tokio::sync::mpsc` channel
- [ ] Implement `Stream` trait for async iteration
- [ ] Handle connection errors, partial chunks, reconnection
- [ ] Unit tests with recorded SSE fixtures

**Acceptance**: Can parse real Anthropic and OpenAI SSE streams from fixture files.

### 1.3 Anthropic Provider (`pi-ai/src/providers/anthropic.rs`)
**Deps**: 1.1, 1.2 | **Effort**: 5 days | **Lines ref**: `anthropic.ts` (1179 lines)

- [ ] Implement message conversion: internal `Message` → Anthropic Messages API format
- [ ] Implement system prompt with cache_control
- [ ] Implement tool definitions conversion
- [ ] Implement streaming: POST to `/v1/messages` with `stream: true`
- [ ] Parse SSE events: `message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`
- [ ] Handle thinking blocks (extended thinking)
- [ ] Handle tool_use blocks
- [ ] Extract usage from `message_start` and `message_delta`
- [ ] Calculate cost from model pricing
- [ ] Handle cache_control (ephemeral, TTL)
- [ ] Handle errors: rate limiting, context overflow, auth failure
- [ ] Integration test with real API (gated behind env var)

**Acceptance**: `ANTHROPIC_API_KEY=... cargo test anthropic_stream` produces valid assistant messages.

### 1.4 OpenAI Completions Provider (`pi-ai/src/providers/openai.rs`)
**Deps**: 1.1, 1.2 | **Effort**: 4 days | **Lines ref**: `openai-completions.ts` (1127 lines)

- [ ] Implement message conversion: internal → OpenAI Chat format
- [ ] Implement tool definitions conversion (function calling)
- [ ] Implement streaming: POST to `/v1/chat/completions` with `stream: true`
- [ ] Parse SSE chunks: `choices[0].delta` (content, tool_calls, reasoning_content)
- [ ] Handle `stream_options: { include_usage: true }`
- [ ] Handle reasoning fields (reasoning_content, reasoning_text)
- [ ] Handle finish_reason mapping (stop, tool_calls, length)
- [ ] Handle strict mode toggle for tool schemas
- [ ] Extract usage from final chunk
- [ ] Integration test with real API

**Acceptance**: Works with OpenAI, OpenRouter, and any OpenAI-compatible endpoint.

### 1.5 Message Transform (`pi-ai/src/transform.rs`)
**Deps**: 1.1 | **Effort**: 2 days | **Lines ref**: `transform-messages.ts` (220 lines)

- [ ] Implement thinking block management across model switches
- [ ] Implement thinking signature detection and stripping
- [ ] Handle assistant message filtering (drop aborted/error messages)
- [ ] Handle tool result coalescing for Anthropic format
- [ ] Unit tests for edge cases (model switch mid-conversation, thinking redaction)

**Acceptance**: Same transform behavior as TS version, verified by snapshot tests.

### 1.6 Basic Agent Loop (`pi-core/src/agent.rs`)
**Deps**: 1.3, 1.4, 1.5 | **Effort**: 3 days | **Lines ref**: `packages/agent/src/agent.ts` (818 lines)

- [ ] Define `AgentState`: messages, model, systemPrompt, thinkingLevel, tools, isStreaming
- [ ] Implement `prompt()`: append user message, call LLM, collect response
- [ ] Implement tool dispatch: detect tool_calls in response, execute, append results, loop
- [ ] Implement abort via `CancellationToken` (tokio)
- [ ] Implement `subscribe()` for event listeners
- [ ] Emit events: `agent_start`, `stream_start`, `stream_event`, `stream_end`, `tool_call`, `tool_result`, `agent_end`
- [ ] Handle max turns limit
- [ ] Unit tests with faux/mock provider

**Acceptance**: Agent can multi-turn with tool calls using mock tools, exits cleanly.

### 1.7 Print Mode (`pi-cli/src/main.rs` + `pi-modes/src/print.rs`)
**Deps**: 1.6 | **Effort**: 2 days

- [ ] Parse `--print` / `-p` flag with message content
- [ ] Parse `--model` / `-m` flag
- [ ] Load API key from env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
- [ ] Instantiate agent with no tools, print mode
- [ ] Stream assistant response tokens to stdout as they arrive
- [ ] Print final response, exit with code 0

**Acceptance**: `pi-rs --print "What is 2+2?" --model claude-sonnet-4-20250514` prints "4" (or equivalent) to stdout.

---

## Phase 2: Tools & Non-Interactive Mode (Weeks 5-8)

### 2.1 Tool Framework (`pi-tools/src/lib.rs`)
**Deps**: 1.1 | **Effort**: 1 day

- [ ] Define `Tool` trait: name, description, schema, execute
- [ ] Define `ToolResult` enum: Success(String), Error(String)
- [ ] Define `ToolContext`: cwd, abort signal
- [ ] Implement JSON Schema generation from Rust types (via `schemars` or manual)

**Acceptance**: Tool trait is ergonomic, schema generates valid JSON Schema.

### 2.2 Bash Tool (`pi-tools/src/bash.rs`)
**Deps**: 2.1 | **Effort**: 3 days | **Lines ref**: `bash.ts` (448 lines)

- [ ] Execute commands via `tokio::process::Command`
- [ ] Implement timeout (configurable, default 120s)
- [ ] Capture stdout + stderr, interleave in order
- [ ] Truncate output to max length
- [ ] Handle working directory (relative to session CWD)
- [ ] Handle abort (kill process group on cancellation)
- [ ] Sandbox: configurable allowed/denied patterns
- [ ] Unit tests: exit codes, timeout, large output

**Acceptance**: `bash(command: "echo hello")` returns "hello\n".

### 2.3 Read Tool (`pi-tools/src/read.rs`)
**Deps**: 2.1 | **Effort**: 1 day | **Lines ref**: `read.ts` (273 lines)

- [ ] Read file with offset/limit (line-based)
- [ ] Detect binary files, return error
- [ ] Handle image files: return base64 with mime type
- [ ] Truncate large files with message
- [ ] Path validation (relative to CWD, no escape)
- [ ] Unit tests

**Acceptance**: Reads text files with pagination, detects images.

### 2.4 Write Tool (`pi-tools/src/write.rs`)
**Deps**: 2.1 | **Effort**: 1 day | **Lines ref**: `write.ts` (281 lines)

- [ ] Write content to file, creating parent dirs
- [ ] Return diff preview (new file vs overwrite)
- [ ] Handle file mutation queue (serialize concurrent writes)
- [ ] Unit tests

**Acceptance**: Creates files, shows diffs, handles concurrent writes.

### 2.5 Edit Tool (`pi-tools/src/edit.rs`)
**Deps**: 2.1 | **Effort**: 3 days | **Lines ref**: `edit.ts` (487 lines) + `edit-diff.ts` (445 lines)

- [ ] Multi-edit: multiple replacements in one call
- [ ] Match `oldText` exactly (unique, non-overlapping)
- [ ] Apply all replacements against original file (not sequential)
- [ ] Generate unified diff for display
- [ ] Handle file mutation queue
- [ ] Error: oldText not found, overlapping edits, ambiguous match
- [ ] Unit tests: single edit, multi-edit, overlap detection, whitespace sensitivity

**Acceptance**: Matches TS edit tool behavior exactly, including error messages.

### 2.6 Grep Tool (`pi-tools/src/grep.rs`)
**Deps**: 2.1 | **Effort**: 2 days | **Lines ref**: `grep.ts` (384 lines)

- [ ] Regex search across files (using `grep` crate or `ripgrep` as library)
- [ ] Respect .gitignore
- [ ] Include/exclude patterns
- [ ] Context lines
- [ ] Truncate results
- [ ] Unit tests

**Acceptance**: Finds matches with context, respects gitignore.

### 2.7 Find Tool (`pi-tools/src/find.rs`)
**Deps**: 2.1 | **Effort**: 1 day | **Lines ref**: `find.ts` (370 lines)

- [ ] Find files by glob pattern
- [ ] Respect .gitignore
- [ ] Limit results
- [ ] Unit tests

**Acceptance**: Finds files matching patterns.

### 2.8 Ls Tool (`pi-tools/src/ls.rs`)
**Deps**: 2.1 | **Effort**: 1 day | **Lines ref**: `ls.ts` (229 lines)

- [ ] List directory contents
- [ ] Show file sizes, types
- [ ] Truncate long listings
- [ ] Unit tests

**Acceptance**: Lists directories with metadata.

### 2.9 File Mutation Queue (`pi-tools/src/file_queue.rs`)
**Deps**: 2.4, 2.5 | **Effort**: 1 day | **Lines ref**: `file-mutation-queue.ts` (39 lines)

- [ ] Serialize concurrent writes/edits to same file
- [ ] Per-file `tokio::sync::Mutex` keyed by canonical path
- [ ] Unit tests for concurrent access

**Acceptance**: Two simultaneous edits to same file execute sequentially.

### 2.10 Settings Manager (`pi-core/src/settings.rs`)
**Deps**: None | **Effort**: 2 days | **Lines ref**: `settings-manager.ts` (955 lines)

- [ ] Load `~/.config/pi/settings.json` (global)
- [ ] Load `.pi/settings.json` (project)
- [ ] Deep merge: project overrides global
- [ ] All setting fields with defaults
- [ ] Save modified settings back
- [ ] Unit tests for merge logic

**Acceptance**: Settings load and merge correctly, missing fields get defaults.

### 2.11 Model Registry (`pi-core/src/model_registry.rs`)
**Deps**: 1.1, 2.10 | **Effort**: 3 days | **Lines ref**: `model-registry.ts` (936 lines)

- [ ] Load built-in model catalog (embedded in binary as static data)
- [ ] Load `~/.config/pi/models.json` (user custom models)
- [ ] API key resolution: env vars, `auth.json`, OAuth tokens
- [ ] Provider detection from env vars
- [ ] Model lookup by ID, fuzzy match
- [ ] `getApiKeyAndHeaders()` equivalent
- [ ] Unit tests

**Acceptance**: Resolves models and API keys from env + config files.

### 2.12 System Prompt Assembly (`pi-core/src/system_prompt.rs`)
**Deps**: 2.10 | **Effort**: 1 day | **Lines ref**: `system-prompt.ts` (172 lines)

- [ ] Build system prompt from template
- [ ] Include CWD, OS info, date
- [ ] Include tool descriptions
- [ ] Include custom instructions from settings/AGENTS.md
- [ ] Unit tests

**Acceptance**: System prompt matches TS version's structure.

### 2.13 Session Persistence (`pi-core/src/session.rs`)
**Deps**: 1.1 | **Effort**: 3 days | **Lines ref**: `session-manager.ts` (1425 lines)

- [ ] JSONL format: one entry per line (UserMessage, AssistantMessage, ToolResult, Summary, BranchPoint)
- [ ] Session file creation with metadata header
- [ ] Append entries atomically
- [ ] Load session from file (replay)
- [ ] Session directory management (`~/.local/share/pi/sessions/`)
- [ ] List sessions with metadata
- [ ] Generate session IDs (UUID v7)
- [ ] Unit tests for write/read round-trip

**Acceptance**: Sessions persist across restarts, replay produces same state.

### 2.14 Context Compaction (`pi-core/src/compaction.rs`)
**Deps**: 1.6, 2.13 | **Effort**: 3 days | **Lines ref**: `compaction.ts` (839 lines) + `branch-summarization.ts` (355 lines)

- [ ] Detect context overflow (token count vs window)
- [ ] Build compaction prompt (summarize old messages, keep recent)
- [ ] Call LLM for summary
- [ ] Replace old messages with summary entry
- [ ] Persist compaction entry to session
- [ ] Branch summarization for tree navigation
- [ ] Unit tests with mock LLM

**Acceptance**: Compaction reduces token count while preserving key context.

### 2.15 Full Print Mode with Tools (`pi-modes/src/print.rs`)
**Deps**: 2.2-2.9, 2.10-2.14 | **Effort**: 2 days

- [ ] Wire tools into agent loop
- [ ] Load settings, model registry
- [ ] Build system prompt
- [ ] Handle multi-turn (tool calls loop)
- [ ] Handle auto-compaction on overflow
- [ ] Session persistence for print mode
- [ ] Output final response to stdout
- [ ] Handle `--no-tools`, `--tools` flags

**Acceptance**: `pi-rs --print "Create a file called hello.txt with 'hi' in it"` creates the file and reports success.

---

## Phase 3: Terminal UI (Weeks 9-14)

### 3.1 Terminal Backend (`pi-tui/src/terminal.rs`)
**Deps**: None | **Effort**: 2 days | **Lines ref**: `terminal.ts` (395 lines)

- [ ] Enter/exit raw mode (via `crossterm`)
- [ ] Alternate screen buffer
- [ ] Mouse capture (optional)
- [ ] Terminal size detection + resize events
- [ ] ANSI escape output: cursor move, colors, clear, scroll
- [ ] Cell-size query (for terminal images)

**Acceptance**: Raw mode enters/exits cleanly, resize events fire.

### 3.2 Input Parser (`pi-tui/src/input.rs`)
**Deps**: 3.1 | **Effort**: 4 days | **Lines ref**: `keys.ts` (1400 lines) + `stdin-buffer.ts` (411 lines)

- [ ] Parse Kitty keyboard protocol (CSI u sequences)
- [ ] Parse xterm modifyOtherKeys
- [ ] Parse legacy escape sequences
- [ ] Handle paste bracketing
- [ ] Handle key release events (Kitty)
- [ ] Debounce/buffer stdin reads
- [ ] Map to logical `Key` enum (with modifiers)
- [ ] Unit tests with raw byte fixtures

**Acceptance**: All key combinations from the TS test suite parse correctly.

### 3.3 Component Model (`pi-tui/src/components/mod.rs`)
**Deps**: 3.1 | **Effort**: 2 days

- [ ] Define `Component` trait: `render(width, height) -> Vec<Line>`, `handle_input(key)`, `min_height()`, `max_height()`
- [ ] Define `Container`: children list, vertical stacking layout
- [ ] Define `Focusable` trait: `is_focusable()`, focus chain navigation
- [ ] Implement layout algorithm: measure → allocate → render

**Acceptance**: Container with children renders correctly at given dimensions.

### 3.4 Text Component (`pi-tui/src/components/text.rs`)
**Deps**: 3.3 | **Effort**: 1 day | **Lines ref**: `text.ts` (106 lines)

- [ ] Render styled text with word wrapping
- [ ] Support ANSI color codes in content
- [ ] Truncation with ellipsis
- [ ] Unit tests

### 3.5 Box/Border Component (`pi-tui/src/components/box_border.rs`)
**Deps**: 3.3 | **Effort**: 1 day | **Lines ref**: `box.ts` (137 lines)

- [ ] Render box border around child
- [ ] Title in border
- [ ] Corner characters (rounded, square)

### 3.6 Editor Component (`pi-tui/src/components/editor.rs`)
**Deps**: 3.2, 3.3 | **Effort**: 5 days | **Lines ref**: `editor.ts` (2292 lines)

- [ ] Multi-line text editing
- [ ] Cursor movement (char, word, line start/end, page up/down)
- [ ] Text insertion, deletion (char, word, line)
- [ ] Undo/redo stack
- [ ] Kill ring (yank/yank-pop)
- [ ] Selection (shift+arrows) - if supported
- [ ] Clipboard integration (via `arboard` or pipe to pbcopy/xclip)
- [ ] Word wrapping display
- [ ] Scrolling for content exceeding height
- [ ] Configurable keybindings
- [ ] Unit tests for all edit operations

**Acceptance**: Full text editing with undo works identically to TS version.

### 3.7 SelectList Component (`pi-tui/src/components/select_list.rs`)
**Deps**: 3.3, 3.2 | **Effort**: 2 days | **Lines ref**: `select-list.ts` (229 lines)

- [ ] Scrollable list of items
- [ ] Keyboard navigation (up/down/page/home/end)
- [ ] Selection highlighting
- [ ] Confirm/cancel callbacks
- [ ] Optional secondary column
- [ ] Fuzzy filter

### 3.8 Markdown Renderer (`pi-tui/src/components/markdown.rs`)
**Deps**: 3.3 | **Effort**: 4 days | **Lines ref**: `markdown.ts` (852 lines)

- [ ] Parse markdown (via `pulldown-cmark`)
- [ ] Render headings with style
- [ ] Render code blocks with syntax highlighting (via `syntect`)
- [ ] Render inline code, bold, italic, links
- [ ] Render lists (ordered, unordered)
- [ ] Render blockquotes
- [ ] Render tables
- [ ] Word wrap within blocks
- [ ] Incremental/streaming render (append new content)

**Acceptance**: Markdown output visually matches TS version.

### 3.9 TUI Engine (`pi-tui/src/tui.rs`)
**Deps**: 3.1-3.8 | **Effort**: 3 days | **Lines ref**: `tui.ts` (1243 lines)

- [ ] Event loop: read input → dispatch to focused component → re-render
- [ ] Diff-based rendering (only redraw changed lines)
- [ ] Component tree management
- [ ] Focus management
- [ ] Overlay system (floating components)
- [ ] Resize handling (re-layout + full redraw)
- [ ] `requestRender()` for deferred renders

**Acceptance**: TUI renders, responds to input, redraws efficiently.

### 3.10 Autocomplete (`pi-tui/src/autocomplete.rs`)
**Deps**: 3.6, 3.7 | **Effort**: 2 days | **Lines ref**: `autocomplete.ts` (783 lines)

- [ ] Provider-based completion (files, commands, custom)
- [ ] Dropdown overlay below cursor
- [ ] Tab to accept, Escape to dismiss
- [ ] Debounced async provider calls

### 3.11 Interactive Mode: Layout (`pi-modes/src/interactive/layout.rs`)
**Deps**: 3.9 | **Effort**: 2 days

- [ ] Header component (logo, version, onboarding hint)
- [ ] Chat container (scrollable message list)
- [ ] Footer component (model, branch, tokens, cost)
- [ ] Input editor at bottom
- [ ] Working/loading indicator during streaming

### 3.12 Interactive Mode: Chat Display (`pi-modes/src/interactive/chat.rs`)
**Deps**: 3.11, 3.8 | **Effort**: 3 days

- [ ] Render user messages (with attachments indicator)
- [ ] Render assistant messages (streaming markdown)
- [ ] Render tool calls (collapsed/expanded, with diff preview)
- [ ] Render thinking blocks (collapsible)
- [ ] Render error messages
- [ ] Auto-scroll during streaming

### 3.13 Interactive Mode: Slash Commands (`pi-modes/src/interactive/commands.rs`)
**Deps**: 3.11, 3.7 | **Effort**: 3 days

- [ ] `/model` - model selector with fuzzy search
- [ ] `/settings` - settings list
- [ ] `/tree` - session tree navigation
- [ ] `/export` - HTML export
- [ ] `/clear` - new session
- [ ] `/compact` - manual compaction
- [ ] `/help` - command list
- [ ] `/login` - provider authentication
- [ ] `/thinking` - thinking level selector
- [ ] Tab completion for command names

### 3.14 Interactive Mode: Model Selector (`pi-modes/src/interactive/model_selector.rs`)
**Deps**: 2.11, 3.7 | **Effort**: 2 days | **Lines ref**: `model-selector.ts` (338 lines)

- [ ] List available models grouped by provider
- [ ] Fuzzy filter
- [ ] Show model details (cost, context window)
- [ ] Persist selection

### 3.15 Interactive Mode: Session Tree (`pi-modes/src/interactive/tree.rs`)
**Deps**: 2.13, 3.7 | **Effort**: 3 days | **Lines ref**: `tree-selector.ts` (1246 lines)

- [ ] Render session history as tree
- [ ] Branch points, labels, timestamps
- [ ] Navigation (jump to point, fork)
- [ ] Filter modes (user-only, no-tools, etc.)

### 3.16 Interactive Mode: Full Integration (`pi-modes/src/interactive/mod.rs`)
**Deps**: 3.11-3.15, 1.6, 2.2-2.9 | **Effort**: 5 days | **Lines ref**: `interactive-mode.ts` (5420 lines)

- [ ] Wire agent to TUI (events → display updates)
- [ ] Handle user input → send to agent
- [ ] Handle streaming display updates
- [ ] Handle tool execution display
- [ ] Keybindings (Escape to abort, Ctrl+D to exit, etc.)
- [ ] Settings changes (live model switch, thinking level)
- [ ] Session operations (new, fork, switch)
- [ ] Auto-compaction with UI feedback
- [ ] Error display and retry

**Acceptance**: Full interactive session works: prompt → stream → tool calls → display → next prompt.

---

## Phase 4: Plugin System (Weeks 15-21)

### 4.1 Protobuf Schema (`proto/pi_plugin.proto`)
**Deps**: 1.1 | **Effort**: 2 days

- [ ] Define all message types: Event, Message, ToolCall, ToolResult, Model, Usage, etc.
- [ ] Define all host function request/response messages
- [ ] Define Permission enum
- [ ] Define PluginConfig message
- [ ] Generate Rust code via `prost-build`

**Acceptance**: All types round-trip through protobuf correctly.

### 4.2 Plugin Manifest (`pi-plugin-host/src/manifest.rs`)
**Deps**: None | **Effort**: 1 day

- [ ] Parse `plugin.toml` manifest
- [ ] Validate required fields
- [ ] Extract permission declarations
- [ ] Extract event subscriptions

### 4.3 WASM Engine Setup (`pi-plugin-host/src/host.rs`)
**Deps**: 4.1 | **Effort**: 3 days

- [ ] Initialize wasmtime `Engine` with fuel metering (prevent infinite loops)
- [ ] Configure memory limits per plugin (default 64MB linear memory cap)
- [ ] Load `.wasm` file, validate module
- [ ] Instantiate with host function linker
- [ ] Call plugin `load()` export
- [ ] Maintain plugin instance pool (multiple plugins loaded simultaneously)
- [ ] Each plugin gets its own `Store` (no shared state between plugin WASM instances)

**Acceptance**: Can load and instantiate a minimal .wasm plugin.

### 4.4 Host Functions: Core (`pi-plugin-host/src/abi.rs`)
**Deps**: 4.3, 4.1 | **Effort**: 4 days

- [ ] Implement memory transfer protocol (allocate in guest, write from host)
- [ ] `pi_subscribe` / `pi_unsubscribe`
- [ ] `pi_request_permission`
- [ ] `pi_get_plugin_config`
- [ ] `pi_register_tool` (tool becomes available to agent)
- [ ] `pi_tool_result` / `pi_tool_error`
- [ ] Implement tool execution bridge: agent calls tool → host calls plugin `on_event(ToolCall)` → plugin calls `pi_tool_result`

### 4.5 Host Functions: Messages & Session (`pi-plugin-host/src/abi.rs`)
**Deps**: 4.4 | **Effort**: 2 days

- [ ] `pi_send_user_message` / `pi_send_steer_message` / `pi_queue_message`
- [ ] `pi_get_messages`
- [ ] `pi_get_cwd`, `pi_get_model`, `pi_get_context_usage`
- [ ] `pi_abort`, `pi_is_idle`
- [ ] `pi_compact`, `pi_get_system_prompt`

### 4.6 Host Functions: UI (`pi-plugin-host/src/abi.rs`)
**Deps**: 4.4, 3.9 | **Effort**: 3 days

- [ ] `pi_ui_notify` → show notification in TUI
- [ ] `pi_ui_set_status` → update footer status text
- [ ] `pi_ui_set_working_message` → update streaming indicator
- [ ] `pi_ui_select` → show selector, block plugin until user chooses (suspend WASM, resume on answer)
- [ ] `pi_ui_confirm` → show yes/no dialog
- [ ] `pi_ui_input` → show text input dialog
- [ ] `pi_ui_set_widget` → render text lines above/below editor

### 4.7 Host Functions: Filesystem (`pi-plugin-host/src/abi.rs`)
**Deps**: 4.4, 4.8 | **Effort**: 2 days

- [ ] `pi_read_file` → read file (scoped to CWD)
- [ ] `pi_write_file` → write file (scoped to CWD)
- [ ] `pi_list_dir` → list directory
- [ ] Path validation: no escape above CWD (canonical path check)

### 4.8 Permission System (`pi-plugin-host/src/permissions.rs`)
**Deps**: 4.2 | **Effort**: 2 days

- [ ] Store granted permissions per plugin (persistent in `~/.config/pi/plugin-permissions.json`)
- [ ] Prompt user on first use (via TUI confirm dialog)
- [ ] Gate each host function behind its required permission
- [ ] `pi plugin audit <file.wasm>` CLI command: inspect imported functions, infer needed permissions
- [ ] "Always allow" / "Allow once" / "Deny" options

**Acceptance**: Unpermitted host calls return error without executing.

### 4.9 Host Functions: Process & Network (`pi-plugin-host/src/abi.rs`)
**Deps**: 4.4, 4.8 | **Effort**: 2 days

- [ ] `pi_run_command` → run process, wait, return stdout/stderr/exit code
- [ ] `pi_spawn_command` → start process, return handle (for streaming output)
- [ ] `pi_http_request` → make HTTP request, return response

### 4.10 Host Functions: LLM & Inter-plugin (`pi-plugin-host/src/abi.rs`)
**Deps**: 4.4, 4.8, 1.3 | **Effort**: 2 days

- [ ] `pi_llm_complete` → call LLM, return full response
- [ ] `pi_llm_stream` → start LLM stream, return handle
- [ ] `pi_pipe_message` → send to specific plugin
- [ ] `pi_pipe_broadcast` → send to all plugins
- [ ] `pi_set_timer` / `pi_cancel_timer` → schedule callbacks

### 4.11 Event Dispatch (`pi-plugin-host/src/host.rs`)
**Deps**: 4.4 | **Effort**: 2 days

- [ ] When agent emits event, serialize to protobuf
- [ ] Dispatch to all plugins subscribed to that event type
- [ ] Each plugin dispatch is independent (failure in one does not block others)
- [ ] Dispatch order: deterministic (load order), but never guaranteed to plugins

### 4.17 Fault Isolation, Reporting & Recovery (`pi-plugin-host/src/isolation.rs`)
**Deps**: 4.3, 4.4, 4.11 | **Effort**: 4 days

Ensure a buggy or malicious plugin can never crash the host, corrupt other plugins, or degrade the user experience silently.

#### Memory Isolation
- [ ] Each plugin instance has its own wasmtime `Store` with separate linear memory
- [ ] Plugins cannot reference, read, or write another plugin's memory (enforced by WASM spec)
- [ ] Configure per-plugin memory ceiling (`max_memory_bytes` in manifest, default 64MB)
- [ ] If a plugin exceeds memory limit, wasmtime traps immediately → host catches the trap
- [ ] Host-side data passed to plugins is copied (serialized protobuf), never shared references

#### CPU / Execution Isolation
- [ ] Fuel metering: each plugin call gets a fuel budget (configurable, default 1B instructions)
- [ ] If fuel exhausted, wasmtime traps → host catches and reports timeout
- [ ] Wall-clock timeout per plugin call (default 30s for event handlers, 120s for tool execution)
- [ ] Implemented via `tokio::time::timeout` wrapping the synchronous WASM call on a blocking task
- [ ] Infinite loops in plugins are impossible: fuel runs out deterministically

#### Fault Handling & Recovery
- [ ] All plugin calls (`on_event`, `pipe`, tool execution) wrapped in `catch_unwind` + wasmtime trap handling
- [ ] On trap/panic: log error with plugin name, event type, and trap message
- [ ] Plugin enters `Faulted` state after a trap (not immediately unloaded)
- [ ] Faulted plugins are skipped for subsequent event dispatch
- [ ] Configurable fault policy per plugin (in manifest or global settings):
  - `fault_policy = "restart"` → auto-reload .wasm after fault (with backoff: 1s, 5s, 30s, give up)
  - `fault_policy = "disable"` → mark disabled, require manual `/plugin reload`
  - `fault_policy = "ignore"` → log and continue dispatching (for non-critical plugins)
- [ ] Fault counter: after N faults in M seconds, force-disable regardless of policy
- [ ] User notification on fault: `[plugin:git-checkpoint] crashed: fuel exhausted in on_event(MessageEnd). Disabled.`

#### Inter-Plugin Isolation
- [ ] Plugins communicate only via pipes (serialized byte messages) — no shared memory
- [ ] Pipe messages are delivered asynchronously (sender does not block on receiver)
- [ ] A faulted plugin's pipe listeners are removed (senders get a `PipeError::RecipientUnavailable`)
- [ ] Plugin A cannot unload/reload Plugin B (only the host/user can manage lifecycle)
- [ ] Tool name conflicts: if two plugins register same tool name, second registration fails with error logged
- [ ] Event ordering: plugins receive events independently, cannot observe or interfere with another plugin's handler

#### Host Protection
- [ ] Host functions validate all arguments from plugin before acting (untrusted input)
- [ ] Path traversal checks on `pi_read_file`/`pi_write_file` (canonical path must be under allowed roots)
- [ ] `pi_run_command` argument sanitization: no shell injection (args are passed as array, never interpolated)
- [ ] `pi_http_request` URL validation: optional allowlist/blocklist in manifest
- [ ] Host function call rate limiting (optional): prevent a plugin from spamming `pi_ui_notify` or `pi_http_request`
- [ ] All host function calls from a faulted plugin are rejected with error

#### Reporting & Observability
- [ ] Plugin log output: plugins can call `pi_log(level, message)` → written to `~/.local/share/pi/logs/plugins/<name>.log`
- [ ] Host-side event log: all plugin faults, permission denials, and lifecycle events logged
- [ ] `/plugin status` command: show per-plugin state (loaded, faulted, disabled), fault count, memory usage, last error
- [ ] `pi --plugin-debug` flag: verbose plugin dispatch logging to stderr
- [ ] Structured error context: on fault, capture {plugin_name, event_type, fuel_remaining, memory_used, timestamp}

#### Resource Cleanup on Fault
- [ ] On fault: cancel any in-flight timers owned by the plugin
- [ ] On fault: remove any widgets/status text owned by the plugin (prevent stale UI)
- [ ] On fault with `restart` policy: tools remain registered but return error until plugin restarts
- [ ] On fault with `disable` policy: full ownership cleanup (same as unload)

**Acceptance**:
- A plugin with `loop {}` is terminated after fuel budget, host continues normally, user is notified.
- A plugin that panics during `on_event(ToolCall)` does not prevent other plugins from receiving subsequent events.
- A plugin allocating memory in a loop is killed at 64MB, host memory is unaffected.
- Two plugins running simultaneously cannot observe or corrupt each other's state.
- `/plugin status` shows fault history and current state for all plugins.

### 4.12 Plugin SDK: Core (`pi-plugin-sdk/src/lib.rs`)
**Deps**: 4.1 | **Effort**: 3 days

- [ ] `#[pi_plugin]` proc macro: generates WASM exports (`load`, `on_event`, `pipe`)
- [ ] Implement `Plugin` trait with default methods
- [ ] Memory allocation helpers (for host↔guest data transfer)
- [ ] Convenience functions wrapping raw host function calls
- [ ] `register_tool!` macro for ergonomic tool registration
- [ ] Publish to crates.io as `pi-plugin-sdk`

### 4.13 Plugin SDK: Types (`pi-plugin-sdk/src/types.rs`)
**Deps**: 4.1, 4.12 | **Effort**: 1 day

- [ ] Re-export relevant types (Event, Message, Model, Permission, etc.)
- [ ] Builder patterns for ToolSchema
- [ ] Helper traits for serialization

### 4.14 Example Plugins
**Deps**: 4.12 | **Effort**: 3 days

- [ ] `hello` plugin: subscribes to session_start, sends greeting
- [ ] `pirate` plugin: transforms system prompt to pirate speak
- [ ] `todo` plugin: registers a tool, stores state
- [ ] `git-checkpoint` plugin: uses `pi_run_command` on session events
- [ ] `permission-gate` plugin: blocks tool calls with confirmation

**Acceptance**: All example plugins load, receive events, tools work.

### 4.15 Plugin Loading & Discovery
**Deps**: 4.3, 4.2 | **Effort**: 2 days

- [ ] Load plugins from `~/.config/pi/plugins/` directory
- [ ] Load plugins from `.pi/plugins/` (project-local)
- [ ] Load from CLI flag `--plugin <path.wasm>`
- [ ] Plugin cache (avoid re-compilation)
- [ ] `pi plugin install <url>` → download .wasm to plugins dir
- [ ] `pi plugin list` → show loaded plugins with permissions
- [ ] `pi plugin remove <name>`

### 4.16 Runtime Load/Unload & Ownership Tracking
**Deps**: 4.4, 4.11 | **Effort**: 3 days

Enable Neovim-style runtime plugin management: load, unload, and reload plugins mid-session without restart.

- [ ] `PluginInstance` ownership registry: track all host-side effects per plugin (tools, subscriptions, widgets, status keys, timers, keybindings, slash commands, pipe listeners)
- [ ] Every host function that creates a side effect appends to the calling plugin's ownership list
- [ ] `unload()` method: remove all owned side effects from host state, drop WASM instance, free memory
- [ ] `reload()` method: unload + fresh instantiation from .wasm file (preserves no state by default)
- [ ] Optional state serialization: plugin can export `save_state() -> Vec<u8>` and accept it in `load()` for state-preserving reload
- [ ] `/plugin load <path_or_name>` slash command: instantiate mid-session
- [ ] `/plugin unload <name>` slash command: clean unload with side effect removal
- [ ] `/plugin reload <name>` slash command: unload + load
- [ ] `/plugin list` enhanced: show per-plugin registrations (tools, events, permissions)
- [ ] Lazy loading support: register stubs for commands/tools/events declared in manifest, instantiate .wasm on first trigger
- [ ] Lazy trigger replay: after on-demand load, deliver the triggering event/command that caused the load
- [ ] Guard against unload of plugins with in-flight tool executions (wait or force with warning)
- [ ] Unit tests: load → register tools → unload → verify tools removed from agent
- [ ] Unit tests: two plugins subscribe to same event → unload one → other still receives events
- [ ] Unit tests: lazy load on slash command trigger → plugin activates and handles command
- [ ] Unit tests: reload preserves state via save_state/load cycle

**Acceptance**: `pi /plugin load ./my.wasm` mid-session adds tools; `/plugin unload my` removes them cleanly with no leaks or stale references.

---

## Phase 5: Multi-Session Daemon & Cross-Session Communication (Weeks 21-26)

### Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        pi-server (daemon process)                       │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                      Session Manager                             │  │
│  │  sessions: HashMap<SessionId, SessionHandle>                     │  │
│  │  bus: CrossSessionBus (broadcast + directed messages)            │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │  Session A   │  │  Session B   │  │  Session C   │  │  ...    │  │
│  │  agent loop  │  │  agent loop  │  │  agent loop  │  │         │  │
│  │  plugins     │  │  plugins     │  │  plugins     │  │         │  │
│  │  tools       │  │  tools       │  │  tools       │  │         │  │
│  │  cwd: /proj  │  │  cwd: /lib   │  │  cwd: /proj  │  │         │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘  │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│  IPC Server (Unix socket: ~/.local/share/pi/daemon.sock)               │
└────────────────────┬───────────────────────────┬──────────────────────┘
                     │                           │
              ┌──────┴───────┐            ┌──────┴───────┐
              │  pi attach A │            │  pi attach B │
              │  (terminal)  │            │  (terminal)  │
              └──────────────┘            └──────────────┘
```

### 5.1 Daemon Process (`pi-server/src/main.rs`)
**Deps**: Phase 2, Phase 3 complete | **Effort**: 4 days

- [ ] `pi server start` — start daemon in background (daemonize, write PID file)
- [ ] `pi server stop` — graceful shutdown (persist all sessions, then exit)
- [ ] `pi server status` — check if daemon is running
- [ ] Auto-start: if no daemon running when `pi` is invoked, start one implicitly
- [ ] Single-instance enforcement: PID file + socket liveness check
- [ ] Graceful shutdown on SIGTERM/SIGINT: persist all session state, unload plugins, exit
- [ ] Crash recovery: on restart, detect persisted sessions and offer resurrection
- [ ] Logging: daemon logs to `~/.local/share/pi/logs/daemon.log`

**Acceptance**: `pi server start` spawns a background process; `pi server status` reports it running; `pi server stop` shuts it down cleanly.

### 5.2 IPC Protocol (`pi-server/src/ipc.rs`)
**Deps**: 5.1 | **Effort**: 3 days

- [ ] Unix domain socket server (Linux/macOS) + named pipe (Windows)
- [ ] Protocol: length-prefixed protobuf frames over the socket
- [ ] Client commands:
  - `CreateSession { name, cwd, model }` → `SessionId`
  - `ListSessions` → `Vec<SessionInfo>`
  - `AttachSession { id }` → start streaming TUI frames
  - `DetachSession { id }` → stop streaming, session keeps running
  - `KillSession { id }` → abort + unload + persist + remove
  - `SessionPrompt { id, content }` → send user message without attaching
  - `SessionAbort { id }` → abort current turn
  - `CrossSessionMessage { from, to, payload }` → deliver to target session
  - `BroadcastMessage { from, payload }` → deliver to all sessions
- [ ] TUI streaming: daemon renders to virtual buffer, streams diff frames to attached client
- [ ] Multiple clients can attach to same session (read-only observers + one active writer)
- [ ] Heartbeat: client sends keepalive, daemon detaches stale clients after 30s

**Acceptance**: Two terminals can attach to same session; messages typed in one appear in both.

### 5.3 Client: Attach/Detach (`pi-cli/src/client.rs`)
**Deps**: 5.2 | **Effort**: 3 days

- [ ] `pi new [name] [--cwd path] [--model id]` — create + attach to new session
- [ ] `pi attach <name_or_id>` — connect terminal to existing session
- [ ] `pi detach` (or keybinding Ctrl+Shift+D) — disconnect, session continues running
- [ ] `pi list` / `pi ls` — show all sessions (name, status, model, cwd, age, attached?)
- [ ] `pi kill <name_or_id>` — terminate a session
- [ ] `pi rename <old> <new>` — rename a session
- [ ] When attaching: client enters raw mode, proxies input to daemon, renders received frames
- [ ] When detaching: restore terminal, print "detached from session X"
- [ ] If only one session exists, `pi` with no args attaches to it
- [ ] If no sessions exist, `pi` creates a default session and attaches
- [ ] Session picker: if multiple sessions exist and no name given, show selector

**Acceptance**: `pi new research`, switch terminal, `pi attach research` — same session, seamless.

### 5.4 Session Lifecycle & Persistence (`pi-server/src/session_lifecycle.rs`)
**Deps**: 5.1, 2.13 | **Effort**: 3 days

- [ ] Each session has independent: agent state, tool set, plugins, CWD, model, settings
- [ ] Sessions run their agent loop on separate tokio tasks (true concurrency)
- [ ] Headless sessions: agent continues working even when no terminal is attached
- [ ] Session serialization on shutdown: save full JSONL + metadata (model, plugins, cwd)
- [ ] Session resurrection on daemon restart: reload from persisted state
- [ ] Session auto-naming: use LLM-generated name from first user message (async, non-blocking)
- [ ] Session limits: configurable max concurrent sessions (default: 16)
- [ ] Idle session eviction: optionally persist and unload sessions idle for >N minutes

**Acceptance**: Kill daemon mid-session; restart; `pi attach` resumes where it left off.

### 5.5 Cross-Session Communication Bus (`pi-server/src/session_bus.rs`)
**Deps**: 5.1, 5.4 | **Effort**: 4 days

- [ ] `CrossSessionBus`: in-process message bus between sessions
- [ ] Directed messages: Session A sends message to Session B by name/ID
- [ ] Broadcast: Session A sends to all other sessions
- [ ] Message types:
  - `TextMessage { from, content }` — appears as a system message in the target session
  - `TaskRequest { from, task, context }` — asks target session to perform work
  - `TaskResult { from, request_id, result }` — response to a TaskRequest
  - `DataShare { from, key, value }` — publish data other sessions can read
  - `Signal { from, signal_type }` — lightweight notification (e.g., "I'm done")
- [ ] Message delivery: async, queued if target session is busy (delivered between turns)
- [ ] Message persistence: cross-session messages are recorded in both sessions' JSONL logs
- [ ] Dead letter handling: if target session doesn't exist, return error to sender

**Acceptance**: Session "coordinator" sends TaskRequest to session "worker"; worker completes and sends TaskResult back.

### 5.6 Cross-Session Host Functions (Plugin API)
**Deps**: 5.5, 4.4 | **Effort**: 2 days

- [ ] `pi_list_sessions() -> Vec<SessionInfo>` — plugin can see other sessions
- [ ] `pi_send_to_session(target_id, message)` — send cross-session message
- [ ] `pi_broadcast_sessions(message)` — broadcast to all sessions
- [ ] `pi_create_session(name, cwd, model) -> SessionId` — spawn a new session
- [ ] `pi_kill_session(id)` — terminate another session
- [ ] `pi_wait_for_session(id) -> SessionResult` — block until target session completes/idles
- [ ] New permission: `CrossSessionAccess` — required for all cross-session host functions
- [ ] New event: `CrossSessionMessage { from_session, payload }` — plugins can subscribe to incoming messages

**Acceptance**: A plugin in session A can spawn session B, send it a task, and receive the result.

### 5.7 Slash Commands for Multi-Session (`pi-modes/src/interactive/commands.rs`)
**Deps**: 5.3, 5.5 | **Effort**: 2 days

- [ ] `/sessions` — list all sessions (name, status, model, cwd)
- [ ] `/session new [name] [--cwd path]` — create a new session (stays attached to current)
- [ ] `/session switch <name>` — detach from current, attach to target
- [ ] `/session send <name> <message>` — send text to another session
- [ ] `/session kill <name>` — terminate another session
- [ ] `/session spawn <task>` — create a new session with a pre-filled prompt, run headless
- [ ] Keyboard shortcut (Ctrl+Shift+S): quick session switcher overlay

### 5.8 Coordinator Pattern (Built-in) (`pi-core/src/coordinator.rs`)
**Deps**: 5.5, 5.6 | **Effort**: 3 days

A first-class pattern for parallel agent work:

- [ ] `/delegate <task>` command: creates a sub-session, sends task, waits for result, injects result into current session
- [ ] `/parallel <task1> | <task2> | <task3>` command: spawn N sub-sessions, run in parallel, collect results
- [ ] Sub-sessions inherit parent's CWD and model (unless overridden)
- [ ] Sub-sessions are headless (no terminal attached)
- [ ] Results from sub-sessions appear as tool results in the parent session
- [ ] Configurable concurrency limit for parallel work
- [ ] Timeout per sub-session (default: 5 minutes)
- [ ] Progress indicator in parent session: "[delegate] 2/3 tasks complete"

**Acceptance**: `/parallel "write tests for auth" | "write tests for payments"` spawns two sessions, both work simultaneously, results appear in the parent when done.

### 5.9 Backward Compatibility: Single-Session Mode
**Deps**: 5.1-5.3 | **Effort**: 1 day

- [ ] `pi --no-daemon` flag: run in legacy single-process mode (no daemon, no IPC)
- [ ] Useful for: CI/CD, scripting, environments where daemon is undesirable
- [ ] Print mode (`--print`) always runs in single-process mode
- [ ] RPC mode can optionally connect to daemon or run standalone

---

## Phase 6: Remaining Providers & Feature Parity (Weeks 27-32)

### 5.1 Google/Vertex Provider (`pi-ai/src/providers/google.rs`)
**Deps**: 1.1, 1.2 | **Effort**: 4 days | **Lines ref**: `google.ts` (500) + `google-vertex.ts` (567) + `google-shared.ts` (354)

- [ ] Implement Gemini API message format
- [ ] Handle function calling
- [ ] Handle thinking (thinkingConfig)
- [ ] Handle image inputs
- [ ] Vertex AI endpoint (with OAuth/ADC auth)
- [ ] Integration tests

### 5.2 Bedrock Provider (`pi-ai/src/providers/bedrock.rs`)
**Deps**: 1.1, 1.2 | **Effort**: 5 days | **Lines ref**: `amazon-bedrock.ts` (956 lines)

- [ ] AWS SigV4 request signing (via `aws-sigv4`)
- [ ] Converse API streaming format
- [ ] Handle tool use blocks
- [ ] Handle thinking
- [ ] Handle prompt caching (cachePoint)
- [ ] Handle bedrock-mantle OpenAI-compatible fallback for Kimi models
- [ ] Region resolution, profile support
- [ ] Integration tests

### 5.3 Mistral Provider (`pi-ai/src/providers/mistral.rs`)
**Deps**: 1.1, 1.2 | **Effort**: 2 days | **Lines ref**: `mistral.ts` (629 lines)

- [ ] Mistral API message format
- [ ] Tool calling
- [ ] Integration tests

### 5.4 OpenAI Responses Provider (`pi-ai/src/providers/openai_responses.rs`)
**Deps**: 1.4 | **Effort**: 3 days | **Lines ref**: `openai-responses.ts` (279) + `openai-responses-shared.ts` (539)

- [ ] OpenAI Responses API format (items, function calls)
- [ ] Handle conversation items replay
- [ ] Handle reasoning
- [ ] Integration tests

### 5.5 Cloudflare Workers AI (`pi-ai/src/providers/cloudflare.rs`)
**Deps**: 1.4 | **Effort**: 1 day | **Lines ref**: `cloudflare.ts` (22 lines)

- [ ] Thin wrapper over OpenAI-compatible endpoint

### 5.6 OAuth Flows (`pi-core/src/oauth.rs`)
**Deps**: 2.11 | **Effort**: 3 days

- [ ] OAuth 2.0 PKCE flow (for Anthropic, Google, Codex)
- [ ] Local HTTP callback server
- [ ] Token storage in `auth.json`
- [ ] Token refresh
- [ ] `/login` command integration

### 5.7 RPC Mode (`pi-modes/src/rpc.rs`)
**Deps**: 1.6, 2.2-2.9 | **Effort**: 3 days | **Lines ref**: `rpc-mode.ts` (754) + `rpc-types.ts` (264)

- [ ] JSON-RPC 2.0 over stdin/stdout
- [ ] Methods: `prompt`, `get_status`, `get_session_stats`, `abort`, `set_model`, etc.
- [ ] Event notifications (streaming events as JSON-RPC notifications)
- [ ] Session management via RPC

### 5.8 Session Tree Navigation (`pi-core/src/session_tree.rs`)
**Deps**: 2.13 | **Effort**: 3 days

- [ ] Fork from any point (creates new session file)
- [ ] Branch point tracking
- [ ] Navigate back to branch point
- [ ] Label management
- [ ] Import from JSONL

### 5.9 HTML Export (`pi-core/src/export.rs`)
**Deps**: 2.13 | **Effort**: 2 days

- [ ] Render session to HTML with embedded CSS
- [ ] Syntax highlighting in code blocks
- [ ] Collapsible tool calls
- [ ] Theme support

### 5.10 Full CLI Parity (`pi-cli/src/main.rs`)
**Deps**: All above | **Effort**: 2 days | **Lines ref**: `args.ts` (342 lines)

- [ ] All flags: `--model`, `--print`, `--json`, `--rpc`, `--cwd`, `--session-dir`, `--tools`, `--no-tools`, `--system-prompt`, `--thinking`, `--plugin`, `--version`, `--help`
- [ ] Flag parsing via `clap` derive macro
- [ ] Env var documentation in `--help`

### 5.11 Cross-Compilation & Release
**Deps**: All above | **Effort**: 2 days

- [ ] CI matrix: linux-x64-musl, linux-arm64-musl, macos-x64, macos-arm64, windows-x64-msvc
- [ ] GitHub release artifacts
- [ ] Install script (`curl | sh` style)
- [ ] Homebrew formula
- [ ] Binary size optimization (`strip`, `lto = "thin"`)
- [ ] Startup time benchmark vs TS version

---

## Phase 7: Plugin Ecosystem & Multi-Language SDKs (Weeks 33-36)

### 6.1 Go SDK (`pi-plugin-sdk-go/`)
**Deps**: 4.1 | **Effort**: 3 days

- [ ] Go types matching protobuf schema
- [ ] TinyGo compilation target (`wasm32-wasip2`)
- [ ] Host function bindings via `//go:wasmimport`
- [ ] Plugin trait equivalent (interface)
- [ ] Example plugin in Go

### 6.2 Zig SDK (`pi-plugin-sdk-zig/`)
**Deps**: 4.1 | **Effort**: 3 days

- [ ] Zig types matching protobuf schema
- [ ] `wasm32-wasi` target compilation
- [ ] Host function imports via `extern`
- [ ] Example plugin in Zig

### 6.3 AssemblyScript SDK (`pi-plugin-sdk-as/`)
**Deps**: 4.1 | **Effort**: 3 days

- [ ] AssemblyScript types (familiar to JS developers)
- [ ] `asc` compiler → WASM
- [ ] Host function bindings
- [ ] Example plugin in AssemblyScript
- [ ] This is the bridge for JS developers transitioning from TS extensions

### 6.4 Plugin Template Generator
**Deps**: 6.1-6.3 | **Effort**: 2 days

- [ ] `pi plugin new --lang rust my-plugin` → scaffold with Cargo.toml, manifest, example code
- [ ] `pi plugin new --lang go my-plugin`
- [ ] `pi plugin new --lang zig my-plugin`
- [ ] `pi plugin new --lang assemblyscript my-plugin`

### 6.5 Plugin Testing Harness
**Deps**: 4.12 | **Effort**: 3 days

- [ ] Mock host functions for unit testing plugins without the full runtime
- [ ] `pi-plugin-test` crate: inject events, assert host function calls
- [ ] Snapshot testing for tool results
- [ ] CI integration for plugin repos

### 6.6 Plugin Hot Reload (Development)
**Deps**: 4.3, 4.15 | **Effort**: 2 days

- [ ] Watch `.wasm` file for changes
- [ ] Unload old instance, load new instance
- [ ] Preserve plugin state across reloads (optional, via serialization)
- [ ] `pi --plugin ./target/wasm32-wasip2/debug/my_plugin.wasm --watch`

### 6.7 Marketplace Infrastructure
**Deps**: 4.15 | **Effort**: 5 days

- [ ] Registry server: stores plugin metadata + .wasm binaries
- [ ] `pi plugin publish` → upload to registry
- [ ] `pi plugin search <query>` → search registry
- [ ] `pi plugin install <name>` → download from registry
- [ ] `pi plugin update` → update all installed plugins
- [ ] SHA256 verification on download
- [ ] Signature verification (optional, for verified publishers)

---

## Dependency Graph (Critical Path)

```
Phase 0 (setup)
    │
    ▼
Phase 1.1 (types) ──────────────────────────────────────────────────────┐
    │                                                                     │
    ├──► 1.2 (SSE parser)                                                │
    │        │                                                            │
    │        ├──► 1.3 (Anthropic) ──┐                                    │
    │        │                       │                                    │
    │        └──► 1.4 (OpenAI) ─────┤                                    │
    │                                │                                    │
    ├──► 1.5 (transform) ───────────┤                                    │
    │                                ▼                                    │
    │                           1.6 (agent loop) ◄────────────────────────┤
    │                                │                                    │
    │                                ▼                                    │
    │                           1.7 (print mode v1)                       │
    │                                                                     │
    ├──► 2.1 (tool framework) ──► 2.2-2.9 (all tools) ──┐               │
    │                                                      │              │
    ├──► 2.10 (settings) ──► 2.11 (model registry) ──────┤              │
    │                                                      │              │
    ├──► 2.12 (system prompt) ────────────────────────────┤              │
    │                                                      │              │
    ├──► 2.13 (session) ──► 2.14 (compaction) ───────────┤              │
    │                                                      ▼              │
    │                                                 2.15 (print mode full)
    │                                                                     │
    ├──► 3.1 (terminal) ──► 3.2 (input) ──┐                             │
    │                                       │                             │
    │    3.3 (components) ──► 3.4-3.8 ─────┤                             │
    │                                       ▼                             │
    │                                  3.9 (TUI engine)                   │
    │                                       │                             │
    │                                       ▼                             │
    │                              3.11-3.16 (interactive mode)           │
    │                                                                     │
    └──► 4.1 (protobuf) ──► 4.3 (wasmtime) ──► 4.4-4.11 (host fns)     │
                                                      │                   │
                                                      ▼                   │
                                                4.12-4.17 (SDK + plugins + load/unload + isolation) │
                                                                          │
                                                5.1-5.9 (multi-session daemon)
                                                          │
                                                          ▼
                                                6.1-6.11 (providers, parity)
                                                          │
                                                          ▼
                                                    7.1-7.7 (ecosystem)
```

## Testing Strategy

| Level | Tool | What |
|-------|------|------|
| Unit | `cargo test` | Each crate in isolation, mock dependencies |
| Integration | `cargo test --features integration` | Real API calls (gated by env vars) |
| E2E | Custom harness | Spawn `pi-rs` binary, feed input, assert output |
| Plugin | `pi-plugin-test` | Load .wasm, inject events, assert behavior |
| Snapshot | `insta` crate | TUI render output, JSON serialization |
| Benchmark | `criterion` | Streaming parse speed, TUI render FPS, tool execution |
| Fuzz | `cargo-fuzz` | SSE parser, JSON parser, input parser |

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Binary size | < 25 MB (stripped) | `ls -la target/release/pi` |
| Startup time | < 50ms to first prompt | `hyperfine ./pi --print "hi"` |
| Streaming latency | < 5ms per token render | Custom benchmark |
| Memory usage (idle) | < 20 MB RSS | `/usr/bin/time -v` |
| Memory usage (streaming) | < 50 MB RSS | During active stream |
| Plugin load time | < 10ms per plugin | Benchmark with 10 plugins |
| Test coverage | > 80% line coverage | `cargo-llvm-cov` |
| All provider tests pass | 100% | CI with API keys |
