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
cargo run -p aither-cli                                      # Interactive CLI
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

### CLI (`cli/`)
Interactive terminal interface for testing agents. Binary name: `aither`.

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

## Agent development guide
- Appending is cheap, since it utilizes input cache of LLM.
- When testing agents, simulate a realistic scenario
- Inspire model's creativity by providing open-ended prompts
- Use XML labels to structure tool calls responses and your prompts
- Utilize subagent to parallelize complex tasks and save context window

## Tool development guide
- You are not allowed to be special in architecture. You must implement `Tool` trait.
- Your tool will be converted into bash-based automatically. You should never write workaround to expose CLI commands.

## Bash-based agent
Our agent only have a single tool: `bash`. All capabilities are exposed via CLI commands,
including web search, web fetch, task management, and LLM queries.

Technically, these command is a wrapper script that calls `aither-ipc <command> "$@"`.
Through from the agent's perspective, it is just calling a bash tool,
you are stilled require to develop a tool using `Tool` trait. We convert these tools 
to CLI commands under the hood.

For each tool, please write one sentence description in rustdoc, and then detailed prompt engineering.
The detailed prompt engineering would be provided to the agent when it run the command with `--help` flag.

## When subagent?

Subagent is useful when:
- The task can be decomposed into smaller subtasks performed independently
- You want to isolate context for better focus
- The task doesn't require interactive user input or feedback. For instance, research a topic or explore a codebase.

## Bad Smells

Avoid the following bad smells in code:
- Static variables that hold state across invocations
- Long monolithic functions that do multiple things
- Any patch or workaround that hides a problem instead of fixing it
- Duplicated code instead of shared functions or modules
- Legacy code left for fallback instead of removal
- Type erasure instead of using structs, traits, and generics
- Manual implementation of functionality that can be achieved with third-party crates

If you found yourself writing patch, stub or workaround, you are doing something wrong. Stop and rethink your design.

## Linting

Workspace-wide pedantic clippy is enforced. All public APIs require `///` documentation.
