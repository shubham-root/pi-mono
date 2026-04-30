# RFC: Porting pi-coder to Rust with WASM Plugin System

## Motivation

1. **Security** - npm supply chain attacks and library name squatting are an existential risk for a tool that runs with full system access. A Rust binary with WASM-sandboxed plugins eliminates transitive dependency risk entirely.
2. **Performance** - Native Rust for the hot path (TUI rendering, streaming SSE parsing, tool execution) eliminates GC pauses and V8 overhead.
3. **Distribution** - Single static binary, no runtime dependencies, instant startup.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         pi (Rust binary)                         │
├─────────────────────────────────────────────────────────────────┤
│  CLI / Args       │  Config / Settings  │  Session Manager      │
├───────────────────┼─────────────────────┼───────────────────────┤
│                     Core Agent Loop                              │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  AI Client  │  │  Tool Executor   │  │  Extension Host   │  │
│  │  (providers)│  │  (bash,edit,etc) │  │  (wasmtime)       │  │
│  └─────────────┘  └──────────────────┘  └───────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      TUI Renderer                                │
│  ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌───────────┐  │
│  │  Terminal  │  │Components │  │  Markdown  │  │   Input   │  │
│  │  Backend   │  │  Tree     │  │  Renderer  │  │   Parser  │  │
│  └───────────┘  └───────────┘  └────────────┘  └───────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Modes                                       │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Interactive   │  │     RPC      │  │      Print         │  │
│  └────────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   WASM Plugins      │
                    │  ┌───────────────┐  │
                    │  │ plugin.wasm   │  │
                    │  │ (Rust/Go/C/   │  │
                    │  │  Zig/etc)     │  │
                    │  └───────┬───────┘  │
                    │          │           │
                    │  Host Functions:     │
                    │  • pi_subscribe      │
                    │  • pi_register_tool  │
                    │  • pi_ui_select      │
                    │  • pi_send_message   │
                    │  • pi_read_file      │
                    │  • pi_run_command    │
                    │  • pi_http_request   │
                    └─────────────────────┘
```

## Cargo Workspace Layout

```
pi-rs/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── pi-cli/                   # Binary entry point, arg parsing
│   │   └── src/main.rs
│   ├── pi-core/                  # Agent loop, session, settings
│   │   └── src/
│   │       ├── agent.rs          # State machine, streaming loop
│   │       ├── session.rs        # JSONL persistence, tree navigation
│   │       ├── settings.rs       # Config loading, merge logic
│   │       ├── compaction.rs     # Context compaction
│   │       ├── system_prompt.rs  # Prompt assembly
│   │       └── model_registry.rs # Model discovery, auth
│   ├── pi-ai/                    # LLM provider clients
│   │   └── src/
│   │       ├── types.rs          # Message, Model, Context, Usage
│   │       ├── stream.rs         # SSE parsing, event stream
│   │       ├── providers/
│   │       │   ├── anthropic.rs
│   │       │   ├── openai.rs     # completions + responses
│   │       │   ├── google.rs     # gemini + vertex
│   │       │   ├── bedrock.rs
│   │       │   ├── mistral.rs
│   │       │   └── cloudflare.rs
│   │       └── transform.rs     # Message format conversion
│   ├── pi-tui/                   # Terminal UI framework
│   │   └── src/
│   │       ├── terminal.rs       # Raw mode, ANSI output
│   │       ├── input.rs          # Key parsing (Kitty, xterm, legacy)
│   │       ├── render.rs         # Diff-based rendering
│   │       ├── components/
│   │       │   ├── container.rs
│   │       │   ├── text.rs
│   │       │   ├── editor.rs     # Line editor with undo
│   │       │   ├── select_list.rs
│   │       │   ├── markdown.rs
│   │       │   └── box_border.rs
│   │       └── layout.rs        # Component tree measurement
│   ├── pi-tools/                 # Built-in tools
│   │   └── src/
│   │       ├── bash.rs
│   │       ├── read.rs
│   │       ├── write.rs
│   │       ├── edit.rs
│   │       ├── grep.rs
│   │       ├── find.rs
│   │       ├── ls.rs
│   │       └── file_queue.rs    # Mutation serialization
│   ├── pi-plugin-host/           # WASM plugin runtime
│   │   └── src/
│   │       ├── host.rs           # wasmtime engine, instance mgmt
│   │       ├── abi.rs            # Host function definitions
│   │       ├── permissions.rs    # Capability gating
│   │       ├── manifest.rs       # Plugin manifest parsing
│   │       └── pipe.rs           # Inter-plugin messaging
│   ├── pi-plugin-sdk/            # SDK crate for plugin authors (Rust)
│   │   └── src/
│   │       ├── lib.rs            # #[pi_plugin] macro, trait Plugin
│   │       ├── events.rs         # Event types
│   │       ├── tools.rs          # Tool registration helpers
│   │       ├── ui.rs             # UI request types
│   │       └── types.rs          # Shared types (Message, Model, etc.)
│   └── pi-modes/                 # Interactive, RPC, Print modes
│       └── src/
│           ├── interactive.rs
│           ├── rpc.rs
│           └── print.rs
├── plugins/                      # Example/built-in plugins
│   ├── hello/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── git-checkpoint/
│   ├── permission-gate/
│   └── custom-provider/
└── proto/                        # Protobuf definitions for plugin ABI
    └── pi_plugin.proto
```

## Plugin System Design (Inspired by Zellij)

### Lifecycle (mirrors Zellij's model)

```rust
// pi-plugin-sdk/src/lib.rs

/// Trait that all plugins implement.
pub trait Plugin: Default {
    /// Called once when plugin is loaded. Declare subscriptions and permissions.
    fn load(&mut self, config: PluginConfig);

    /// Called when a subscribed event fires.
    fn on_event(&mut self, event: Event);

    /// Called when the plugin receives a pipe message from another plugin or CLI.
    fn pipe(&mut self, message: PipeMessage) -> Option<String>;
}
```

### Events (pi-specific, analogous to Zellij events)

```rust
pub enum Event {
    // Session lifecycle
    SessionStart { reason: StartReason, cwd: String },
    SessionShutdown,

    // Agent events
    MessageStart { role: Role },
    MessageEnd { message: Message },
    ToolCall { name: String, args: Value, tool_call_id: String },
    ToolResult { name: String, result: String, tool_call_id: String },
    AgentError { error: String },

    // User interaction
    UserMessage { content: String },
    SlashCommand { name: String, args: String },
    KeyPress { key: Key },

    // System
    Timer { id: u32 },
    PipeMessage { source: String, payload: Vec<u8> },
    PermissionGranted { permissions: Vec<Permission> },
}
```

### Host Functions (what plugins can call into the host)

```rust
// Exposed to WASM via wasmtime host functions
// Serialization: protobuf (like Zellij) for structured data

// --- Core ---
fn pi_subscribe(events: &[EventType]);
fn pi_unsubscribe(events: &[EventType]);
fn pi_request_permission(permissions: &[Permission]);
fn pi_get_plugin_config() -> PluginConfig;

// --- Tools ---
fn pi_register_tool(name: &str, description: &str, schema_json: &str);
fn pi_tool_result(tool_call_id: &str, result: &str);
fn pi_tool_error(tool_call_id: &str, error: &str);

// --- Messages ---
fn pi_send_user_message(content: &str);
fn pi_send_steer_message(content: &str);
fn pi_queue_message(content: &str, deliver_as: DeliverMode);
fn pi_get_messages() -> Vec<Message>;

// --- UI ---
fn pi_ui_notify(message: &str, level: NotifyLevel);
fn pi_ui_set_status(key: &str, text: &str);
fn pi_ui_set_working_message(message: &str);
fn pi_ui_select(title: &str, options: &[&str]) -> Option<usize>;
fn pi_ui_confirm(title: &str, message: &str) -> bool;
fn pi_ui_input(title: &str, placeholder: &str) -> Option<String>;
fn pi_ui_set_widget(key: &str, lines: &[&str], placement: Placement);

// --- Session ---
fn pi_get_cwd() -> String;
fn pi_get_model() -> Model;
fn pi_get_context_usage() -> ContextUsage;
fn pi_abort();
fn pi_is_idle() -> bool;
fn pi_compact(instructions: &str);
fn pi_get_system_prompt() -> String;

// --- Filesystem (permission-gated) ---
fn pi_read_file(path: &str) -> Result<Vec<u8>, Error>;
fn pi_write_file(path: &str, content: &[u8]) -> Result<(), Error>;
fn pi_list_dir(path: &str) -> Result<Vec<DirEntry>, Error>;

// --- Process (permission-gated) ---
fn pi_run_command(cmd: &str, args: &[&str]) -> Result<CommandOutput, Error>;
fn pi_spawn_command(cmd: &str, args: &[&str]) -> Result<ProcessHandle, Error>;

// --- Network (permission-gated) ---
fn pi_http_request(request: HttpRequest) -> Result<HttpResponse, Error>;

// --- Inter-plugin ---
fn pi_pipe_message(plugin_name: &str, payload: &[u8]);
fn pi_pipe_broadcast(name: &str, payload: &[u8]);

// --- LLM (permission-gated) ---
fn pi_llm_complete(model: &str, messages: &[Message]) -> Result<String, Error>;
fn pi_llm_stream(model: &str, messages: &[Message]) -> StreamHandle;

// --- Timer ---
fn pi_set_timer(seconds: f64) -> u32;
fn pi_cancel_timer(id: u32);
```

### Permissions (inspired by Zellij)

```rust
pub enum Permission {
    /// Read agent state (messages, model, context usage)
    ReadAgentState,
    /// Modify agent state (send messages, change model, abort)
    ChangeAgentState,
    /// Read files from the filesystem
    ReadFilesystem,
    /// Write files to the filesystem
    WriteFilesystem,
    /// Run commands / spawn processes
    RunCommands,
    /// Make network requests
    NetworkAccess,
    /// Access LLM directly (costs money)
    LlmAccess,
    /// Send messages to other plugins
    InterPluginComms,
}
```

### Plugin Manifest

```toml
# plugin.toml
[plugin]
name = "git-checkpoint"
version = "0.1.0"
description = "Auto-commit on session changes"
authors = ["developer@example.com"]
license = "MIT"

[permissions]
required = ["RunCommands", "ReadAgentState"]
optional = ["NetworkAccess"]

[events]
subscribe = ["SessionStart", "MessageEnd", "SessionShutdown"]

[tools]
# Tools this plugin registers
[[tools.definitions]]
name = "git_checkpoint"
description = "Create a git checkpoint commit"

[build]
# How the plugin is compiled
target = "wasm32-wasip2"
```

### What Breaks vs Current Extensions

| Current Extension Pattern | WASM Equivalent | Migration |
|--------------------------|-----------------|-----------|
| `import { spawn } from "child_process"` | `pi_run_command("git", &["status"])` | Use host function |
| `import { readFile } from "fs"` | `pi_read_file("/path")` | Use host function |
| `fetch("https://...")` | `pi_http_request(req)` | Use host function |
| `extends Container` / custom TUI | **Not supported** | Use `pi_ui_set_widget` with text lines |
| `ctx.ui.custom(factory)` | **Not supported** | Use built-in UI primitives only |
| `import Anthropic from "@anthropic-ai/sdk"` | `pi_llm_stream(model, msgs)` | Use host LLM API |
| `ctx.ui.select(...)` | `pi_ui_select(...)` | Direct mapping |
| `pi.on("tool_call", handler)` | `fn on_event(Event::ToolCall{..})` | Direct mapping |
| `registerTool(...)` | `pi_register_tool(...)` | Direct mapping |

### Extensions That Cannot Be Ported to WASM

These require custom TUI component rendering (bidirectional callbacks):

| Extension | Reason | Alternative |
|-----------|--------|-------------|
| `doom-overlay` | Full game rendering in custom component | Not portable |
| `snake`, `space-invaders` | Games with custom rendering | Not portable |
| `tic-tac-toe` | Custom interactive component | Not portable |
| `modal-editor`, `rainbow-editor` | Custom editor replacement | Not portable |
| `custom-footer`, `custom-header` | Custom component factories | Use widget text API |
| `built-in-tool-renderer` | Custom render functions | Not portable |

These are all demo/novelty extensions. No production extension requires custom TUI rendering.

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Rust binary that can stream a single LLM response to stdout.

- [ ] Cargo workspace scaffold
- [ ] `pi-ai` crate: types (Message, Model, Context, Usage)
- [ ] `pi-ai` crate: Anthropic provider (SSE streaming via `reqwest` + `eventsource-stream`)
- [ ] `pi-ai` crate: OpenAI completions provider
- [ ] `pi-core` crate: basic agent loop (prompt → stream → collect)
- [ ] `pi-cli` crate: `pi --print "hello"` works with Anthropic

### Phase 2: Tools & Sessions (Weeks 5-8)

**Goal**: Non-interactive mode works end-to-end.

- [ ] `pi-tools` crate: bash, read, write, edit, grep, find, ls
- [ ] `pi-core`: tool dispatch loop (multi-turn)
- [ ] `pi-core`: JSONL session persistence
- [ ] `pi-core`: settings loading (settings.json)
- [ ] `pi-core`: model registry (models.json, env API keys)
- [ ] `pi-core`: system prompt assembly
- [ ] `pi-core`: context compaction (LLM-based summarization)
- [ ] `pi-cli`: print mode works fully

### Phase 3: TUI (Weeks 9-14)

**Goal**: Interactive mode works.

- [ ] `pi-tui`: terminal backend (raw mode, ANSI, alternate screen)
- [ ] `pi-tui`: input parser (Kitty protocol, xterm, CSI-u)
- [ ] `pi-tui`: component tree (Container, Text, Spacer, Box)
- [ ] `pi-tui`: diff-based renderer
- [ ] `pi-tui`: Editor component (line editing, undo, yank ring)
- [ ] `pi-tui`: SelectList, SettingsList
- [ ] `pi-tui`: Markdown renderer (code blocks, headings, lists)
- [ ] `pi-modes/interactive`: chat layout, footer, header
- [ ] `pi-modes/interactive`: streaming display
- [ ] `pi-modes/interactive`: slash commands (/model, /settings, /tree, /export)

### Phase 4: Plugin System (Weeks 15-20)

**Goal**: WASM plugins work with permission model.

- [ ] `pi-plugin-host`: wasmtime integration, instance lifecycle
- [ ] `pi-plugin-host`: host function ABI (protobuf serialization)
- [ ] `pi-plugin-host`: permission prompting and enforcement
- [ ] `pi-plugin-host`: event dispatch to subscribed plugins
- [ ] `pi-plugin-host`: tool registration from plugins
- [ ] `pi-plugin-host`: pipe messaging between plugins
- [ ] `pi-plugin-host`: plugin manifest parsing
- [ ] `pi-plugin-sdk`: Rust SDK crate with `#[pi_plugin]` proc macro
- [ ] `pi-plugin-sdk`: helper types and builder patterns
- [ ] Port example plugins: hello, git-checkpoint, permission-gate, pirate, todo

### Phase 5: Remaining Providers & Polish (Weeks 21-26)

**Goal**: Feature parity with TypeScript version.

- [ ] `pi-ai`: Google/Vertex provider
- [ ] `pi-ai`: Bedrock provider (AWS SigV4 signing)
- [ ] `pi-ai`: Mistral, Cloudflare, OpenAI Codex Responses
- [ ] `pi-ai`: OpenRouter, Azure OpenAI
- [ ] `pi-core`: OAuth flows (Anthropic, Google, OpenAI Codex)
- [ ] `pi-modes/rpc`: JSON-RPC mode
- [ ] `pi-core`: session tree navigation, forking
- [ ] `pi-cli`: full arg parsing parity
- [ ] Marketplace: plugin registry, `pi plugin install <url>`
- [ ] Binary releases: cross-compile for linux-x64, linux-arm64, macos-x64, macos-arm64, windows-x64

### Phase 6: Plugin Ecosystem (Weeks 27-30)

**Goal**: Mature plugin system with SDK for multiple languages.

- [ ] `pi-plugin-sdk-go`: Go SDK using TinyGo → WASM
- [ ] `pi-plugin-sdk-zig`: Zig SDK → WASM
- [ ] `pi-plugin-sdk-js`: AssemblyScript SDK for JS developers
- [ ] Plugin template generators: `pi plugin new --lang rust`
- [ ] Plugin testing harness (mock host functions)
- [ ] Plugin hot-reload during development

## Key Dependencies (Rust Crates)

| Crate | Purpose | Replaces |
|-------|---------|----------|
| `tokio` | Async runtime | Node.js event loop |
| `reqwest` | HTTP client | Node.js fetch/http |
| `wasmtime` | WASM runtime | - |
| `serde` / `serde_json` | Serialization | JSON.parse/stringify |
| `prost` | Protobuf (plugin ABI) | - |
| `crossterm` | Terminal raw mode, events | Custom terminal code |
| `unicode-width` | Character width calculation | Custom width utils |
| `syntect` | Syntax highlighting | - |
| `similar` | Diff computation (edit tool) | Custom diff |
| `clap` | CLI argument parsing | Custom arg parser |
| `toml` | Config/manifest parsing | JSON config |
| `aws-sigv4` | Bedrock auth | @aws-sdk |
| `jsonwebtoken` | OAuth token handling | - |
| `notify` | File watching (git HEAD) | fs.watch |
| `globset` | Glob matching (gitignore) | - |

## Security Model

### Binary Supply Chain
- All dependencies are Rust crates, audited via `cargo-audit`
- No JavaScript in the critical path
- Reproducible builds via `cargo --locked`
- Binary is statically linked (musl on Linux)

### Plugin Sandbox
- WASM linear memory: plugins cannot access host memory
- No filesystem access without explicit `ReadFilesystem` / `WriteFilesystem` permission
- No network without `NetworkAccess` permission
- No process spawning without `RunCommands` permission
- No LLM calls without `LlmAccess` permission
- Permissions are prompted to user on first use (like Zellij)
- Filesystem access is scoped to CWD by default

### Plugin Distribution
- Plugins are `.wasm` files (single file, inspectable)
- Marketplace publishes SHA256 hashes
- `pi plugin audit <file.wasm>` shows imported host functions → reveals actual capabilities
- No transitive dependencies at runtime (everything compiled into single .wasm)

## Migration Path for Existing Extensions

For the 32 "pure SDK" extensions (no system calls):

1. Rewrite in Rust using `pi-plugin-sdk`
2. Compile to `wasm32-wasip2`
3. Publish as `.wasm` to marketplace

For extensions needing system access (spawn, fs, net):

1. Rewrite in Rust using `pi-plugin-sdk` + request permissions
2. `pi_run_command()` replaces `child_process.spawn()`
3. `pi_read_file()` / `pi_write_file()` replaces `fs.*`
4. `pi_http_request()` replaces `fetch()`

For extensions needing custom TUI components (14 demo extensions):

- **Not supported** in WASM tier
- These are games/demos, not production extensions
- Alternative: use `pi_ui_set_widget()` for simple text-based widgets

## Build & Distribution

```bash
# Development
cargo build                          # debug build
cargo build --release                # optimized release

# Cross-compilation
cross build --target x86_64-unknown-linux-musl --release
cross build --target aarch64-unknown-linux-musl --release
cross build --target x86_64-apple-darwin --release
cross build --target aarch64-apple-darwin --release
cross build --target x86_64-pc-windows-msvc --release

# Plugin development
cd plugins/my-plugin
cargo build --target wasm32-wasip2 --release
# Output: target/wasm32-wasip2/release/my_plugin.wasm

# Install plugin
pi plugin install ./my_plugin.wasm
pi plugin install https://marketplace.example.com/plugins/git-checkpoint.wasm
```

## Open Questions

1. **Protobuf vs MessagePack for plugin ABI?** Zellij uses protobuf. MessagePack is simpler but less type-safe.
2. **WASI Preview 2 vs Preview 1?** P2 has better async support and component model, but tooling is newer.
3. **Plugin UI rendering**: Should plugins get a "render zone" they can print ANSI to (like Zellij), or only structured widget APIs?
4. **Background workers**: Should plugins get background threads (like Zellij workers) for long-running tasks?
5. **Plugin auto-update**: Should `pi` auto-update marketplace plugins, or require explicit `pi plugin update`?

## Comparison with Zellij's Approach

| Aspect | Zellij | pi (proposed) |
|--------|--------|---------------|
| Plugin renders | Prints to stdout (terminal pane) | Structured widget API |
| Serialization | Protobuf | Protobuf |
| Runtime | wasmtime | wasmtime |
| Permissions | Prompted on first use | Prompted on first use |
| Workers | Yes (background threads) | Yes (via async host functions) |
| Pipes | Plugin-to-plugin + CLI-to-plugin | Same |
| Languages | Rust (primary), any → WASM | Rust (primary), Go, Zig, AssemblyScript |
| Hot reload | `--skip-plugin-cache` | Dev mode hot reload |
| Distribution | URL or file path | Marketplace + URL + file path |
