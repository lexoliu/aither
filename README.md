<div align="center">
<img src="logo.svg" alt="aither logo" width="150" height="150">

# aither

Unified Rust traits for building AI applications across providers


[![Crates.io](https://img.shields.io/crates/v/aither.svg)](https://crates.io/crates/aither)
[![Documentation](https://docs.rs/aither/badge.svg)](https://docs.rs/aither)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)

</div>


**Write AI applications that work with any provider** 🚀

`aither` is a workspace of crates that gives you portable traits (`LanguageModel`, `EmbeddingModel`, `ImageGenerator`, …) plus thin provider bindings (`aither-openai`, `aither-gemini`, etc.). Build flows once and pick any backend that satisfies the traits—OpenAI, Gemini, local inference, or custom vendor endpoints.

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your App      │───▶│    aither        │◀───│   Providers     │
│                 │    │   (this crate)   │    │                 │
│ - Chat bots     │    │                  │    │ - openai        │
│ - Search        │    │ - LanguageModel  │    │ - anthropic     │
│ - Content gen   │    │ - EmbeddingModel │    │ - llama.cpp     │
│ - Voice apps    │    │ - ImageGenerator │    │ - whisper       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Highlights

- 🎯 **Provider-agnostic traits** – swap between OpenAI, Gemini, local adapters, or your own.
- ⚡ **Streaming-first** – every `LanguageModel::respond` returns a stream of [`Event`]s: visible text deltas, reasoning updates, and tool calls.
- 🧠 **Reasoning controls** – request chain-of-thought summaries, budgets, or effort tiers without macros.
- 🛠️ **Tooling & structured output** – JSON-schema tools, builders, and derive macros keep function calling type-safe.
- 🧱 **No-std capable** – `aither-core` runs in embedded/WASM targets and re-exports only `alloc`.
- 📦 **Batteries included** – provider crates (`openai`, `gemini`) plus runnable examples (`cargo run --example tool_macro`).

## Supported Capabilities

| Capability | Trait | Description |
|------------|-------|-------------|
| Language Models | `LanguageModel` / `Event` | Streaming chat, reasoning summaries, tool calling |
| Embeddings | `EmbeddingModel` | Vectorize text for search, clustering, and RAG |
| Images | `ImageGenerator` | Progressive generation + editing pipelines |
| Audio | `AudioGenerator` / `AudioTranscriber` | TTS + speech recognition |
| Moderation | `Moderation` | Policy scoring across multiple providers |

## Quick Start

1. Choose a provider crate (`aither-openai`, `aither-gemini`, …) alongside `aither` for the shared traits:

```toml
[dependencies]
aither = { version = "0.1", features = ["serde", "derive"] }
aither-openai = "0.1"
```

2. Instantiate the provider, then drive everything through the trait:

```rust
use aither::{LanguageModel, llm::{Event, LLMRequest, Message}};
use aither_openai::OpenAI;
use futures_lite::StreamExt;

async fn basic_chat(api_key: &str) -> anyhow::Result<String> {
    let model = OpenAI::new(api_key);
    let request = LLMRequest::new([
        Message::system("You are a multilingual assistant."),
        Message::user("What is the capital of France?")
    ]);

    let mut stream = model.respond(request);
    let mut transcript = String::new();
    while let Some(event) = stream.next().await {
        if let Event::Text(chunk) = event? {
            transcript.push_str(&chunk);
        }
    }
    Ok(transcript)
}
```

### Streaming Reasoning & Thinking Budgets

Reasoning-focused models (OpenAI O-series, Gemini Flash Thinking, etc.) expose chain-of-thought summaries as `Event::Reasoning` items in the same stream. You can request a thinking budget or reasoning effort via `Parameters`.

```rust
use aither::llm::{LanguageModel, Message, Request, model::Parameters};
use aither::{LanguageModel, llm::{Event, LLMRequest, Message, model::Parameters}};
use futures_lite::StreamExt;

async fn inspect_reasoning(model: impl LanguageModel) -> anyhow::Result<()> {
    let request = LLMRequest::new([
        Message::user("Solve 24 using numbers 4,4,4,4."),
    ])
    .with_parameters(
        Parameters::default()
            .include_reasoning(true)
            .reasoning_budget_tokens(256)
    );

    let mut stream = model.respond(request);
    let mut final_text = String::new();

    // Reasoning and visible text arrive interleaved in the same stream.
    while let Some(event) = stream.next().await {
        match event? {
            Event::Reasoning(thought) => println!("🤔 {thought}"),
            Event::Text(chunk) => final_text.push_str(&chunk),
            _ => {}
        }
    }

    println!("Answer: {final_text}");
    Ok(())
}
```

### Function Calling

```rust
use aither::llm::{LLMRequest, Message, Tool, ToolContext, ToolResult};
use schemars::JsonSchema;
use serde::Deserialize;
use std::borrow::Cow;

/// Get current weather for a location.
#[derive(JsonSchema, Deserialize)]
struct WeatherQuery {
    /// City to report on, e.g. "Tokyo".
    location: String,
}

struct WeatherTool;

impl Tool for WeatherTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("get_weather")
    }

    type Arguments = WeatherQuery;
    type Res = ToolResult;

    async fn call(&self, args: Self::Arguments, _cx: ToolContext) -> aither::Result<ToolResult> {
        Ok(ToolResult::text(format!("Weather in {}: 22°C, sunny", args.location)))
    }
}

// Advertise the tool on the request. The model answers with an
// `Event::ToolCall`; running it and feeding the result back is the caller's
// job, or `aither-agent`'s.
let request = LLMRequest::new([Message::user("What is the weather in Tokyo?")])
    .with_tool(&WeatherTool);
```

The tool's description defaults to the rustdoc on its `Arguments` type. Override
`Tool::description()` to set it explicitly — a tool with neither is rejected at
registration rather than reaching the model unexplained.

Every call also receives its own `ToolContext`. A long-running tool reports how
far it has got with `cx.report_progress(Progress::new(done).with_total(total))`;
served over MCP, each report becomes a `notifications/progress` for the
request's progress token, and when the caller asked for no progress the report
goes nowhere.

### Semantic Search & Multimodal

See `examples/chatbot_gemini.rs`, `examples/chatbot_openrouter.rs`, and `examples/tool_macro.rs` for end-to-end demos that combine embeddings, multimodal prompts, and structured outputs. Each example can be run with:

```bash
cargo run --example tool_macro --features derive
```

### Progressive Image Generation

```rust
use aither::{ImageGenerator, image::{Prompt, Size}};
use futures_lite::StreamExt;

async fn generate_image(generator: impl ImageGenerator) -> aither::Result<Vec<u8>> {
    let prompt = Prompt::new("A beautiful sunset over mountains");
    let size = Size::square(1024);
    
    let mut image_stream = generator.create(prompt, size);
    let mut final_image = Vec::new();
    
    while let Some(image_result) = image_stream.next().await {
        final_image = image_result?;
        println!("Received image update, {} bytes", final_image.len());
    }
    
    Ok(final_image)
}
```

## Workspace Layout

| Crate | Description |
|-------|-------------|
| `aither` | Entry crate re-exporting everything from `aither-core` + derive macros |
| `aither-core` | No-std traits (`LanguageModel`, `Event`, `LLMRequest`, embedders, moderation, …) |
| `aither-openai` | Provider bindings for OpenAI-compatible chat, images, audio, and moderation |
| `aither-gemini` | Google Gemini bindings with tool looping and thinking budgets |
| `aither-apple` | On-device Apple Intelligence, native tools and guided output on macOS/iOS 26+; [build and examples](apple/README.md) |
| `aither-rag` | Retrieval-Augmented Generation helper with a parallel in-memory vector DB |
| `aither-llama` | Local llama.cpp wrapper that statically links llama.cpp |
| `derive/` | Proc-macro helpers for tool schemas (`#[tool]`) |
| `examples/` | Runnable flows for chat, research, and tool macros |

## Development

Use the same commands as CI:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features --workspace -- -D warnings
cargo test --all-features --workspace
```

To try reasoning/tooling flows locally:

```bash
# Stream reasoning with tools enabled
OPENAI_API_KEY=sk-... cargo run --example tool_macro -p aither-openai

# Gemini thinking-budget demo
GEMINI_API_KEY=... cargo run --example chatbot_gemini -p aither-gemini
```

## License

MIT License - see [LICENSE](LICENSE) for details.

[`Event`]: core/src/llm/event.rs
