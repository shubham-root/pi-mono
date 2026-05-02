# pi-rs

Rust port of pi, the AI coding agent.

## Structure

This is a Cargo workspace containing:

- `pi-ai` - Multi-provider AI client library
- `pi-core` - Core agent logic, settings, model registry, sessions
- `pi-tools` - Tool framework and built-in tools
- `pi-tui` - Terminal UI components
- `pi-modes` - Execution modes (print, interactive, RPC)
- `pi-cli` - CLI entry point
- `pi-plugin-host` - WASM plugin runtime
- `pi-plugin-sdk` - SDK for plugin authors
- `pi-server` - Multi-session daemon

## Development

```bash
# Check all crates
cargo check --workspace

# Run clippy
cargo clippy --workspace --all-targets

# Run tests
cargo test --workspace

# Build release binary
cargo build --release

# Run the CLI
cargo run --bin pi -- --help
```

## CI

The `.github/workflows/ci.yml` runs `cargo check`, `cargo clippy`, `cargo test`, and `cargo deny verify` on PRs and pushes.

## Building

Requires Rust 1.85+ with the `wasm32-wasip2` target for plugin development:

```bash
rustup target add wasm32-wasip2
```

## License

MIT - See the root LICENSE file.
