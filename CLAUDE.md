# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo fmt --all                                              # Format all crates
cargo clippy --all-targets --all-features --workspace -- -D warnings  # Lint (mirrors CI)
cargo test --all-features --workspace                        # Run all tests
cargo doc --all-features --no-deps --workspace               # Build docs
```

Run examples:
```bash
OPENAI_API_KEY=sk-... cargo run --example tool_macro -p aither-openai
GEMINI_API_KEY=... cargo run --example chatbot_gemini -p aither-gemini
```

Run a single test:
```bash
cargo test --package aither-core test_name
```

## Architecture

**aither** is a Rust workspace providing unified traits for AI model providers. The design separates abstract traits from provider implementations.

### Core Layer (`core/`)
`aither-core` defines provider-agnostic traits that all implementations must satisfy:
- `LanguageModel` / `LLMResponse` – streaming chat with reasoning and tool calling
- `EmbeddingModel` – text vectorization for RAG
- `ImageGenerator` – progressive image generation
- `AudioGenerator` / `AudioTranscriber` – TTS and speech recognition
- `Moderation` – content policy scoring
- `Tool` trait + `#[tool]` derive macro for type-safe function calling

The core is no-std capable (`alloc` only).

### Provider Layer
Thin bindings mapping traits to vendor APIs:
- `openai/` – OpenAI-compatible endpoints
- `gemini/` – Google Gemini with thinking budgets
- `claude/` – Anthropic Claude
- `ort/` – Local ONNX inference

### Agent Layer (`agent/`)
High-level orchestration framework:
- `Agent` builder with tool registration and conversation loop
- Subagent spawning for specialized tasks (explore, plan, etc.)
- Streaming response handling with `AgentStream`

### Tools (`tools/`)
Standalone tool crates that integrate with the agent:
- `websearch/`, `webfetch/` – web access
- `fs/` – filesystem operations
- `command/` – shell execution

### MCP (`mcp/`)
Model Context Protocol client/server implementation for tool discovery and execution.

### Other Crates
- `derive/` – proc-macros (`#[tool]`)
- `rag/` – retrieval-augmented generation with in-memory vector DB
- `mem0/` – conversation memory
- `models/` – model registry
- `sandbox/` – sandboxed execution

## Code Patterns

**Streaming-first**: Every `LanguageModel::respond` returns a stream. Use `futures_lite::StreamExt` for iteration.

**Builders over constructors**: Prefer `Request::new([...]).with_tool(...)` pattern.

**Tool definition**: Implement the `Tool` trait or use `#[derive(Tool)]` with `schemars::JsonSchema` for argument schemas.

**Feature flags**: Provider crates are optional (`openai`, `gemini`, `claude`). Use `full` feature for everything.

## Rust Version

Requires Rust 1.87+ (edition 2024).

## Linting

Workspace-wide pedantic clippy is enforced. All public APIs require `///` documentation.
