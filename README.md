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
- ⚡ **Streaming-first** – every `LanguageModel::respond` returns an [`LLMResponse`] stream with visible deltas plus reasoning updates.
- 🧠 **Reasoning controls** – request chain-of-thought summaries, budgets, or effort tiers without macros.
- 🛠️ **Tooling & structured output** – JSON-schema tools, builders, and derive macros keep function calling type-safe.
- 🧱 **No-std capable** – `aither-core` runs in embedded/WASM targets and re-exports only `alloc`.
- 📦 **Batteries included** – provider crates (`openai`, `gemini`) plus runnable examples (`cargo run --example tool_macro`).

## Supported Capabilities

| Capability | Trait | Description |
|------------|-------|-------------|
| Language Models | `LanguageModel` / `LLMResponse` | Streaming chat, reasoning summaries, tool calling |
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
use aither::{LanguageModel, llm::{Message, Request}};
use aither_openai::OpenAI;
use futures_lite::StreamExt;

async fn basic_chat(api_key: &str) -> aither::Result<String> {
    let model = OpenAI::new(api_key);
    let request = Request::new([
        Message::system("You are a multilingual assistant."),
        Message::user("What is the capital of France?")
    ]);

    let mut stream = model.respond(request);
    let mut transcript = String::new();
    while let Some(chunk) = stream.next().await {
        transcript.push_str(&chunk?);
    }
    Ok(transcript)
}
```

### Streaming Reasoning & Thinking Budgets

Reasoning-focused models (OpenAI O-series, Gemini 2.0 Flash Thinking, etc.) expose chain-of-thought summaries through [`LLMResponse::poll_reasoning_next`]. You can also request a thinking budget or reasoning effort via `Parameters`.

```rust
use aither::llm::{LanguageModel, Message, Request, model::Parameters};
use futures_lite::{StreamExt, future::poll_fn};
use core::pin::Pin;

async fn inspect_reasoning(model: impl LanguageModel) -> aither::Result<()> {
    let request = Request::new([
        Message::user("Solve 24 using numbers 4,4,4,4."),
    ])
    .with_parameters(
        Parameters::default()
            .include_reasoning(true)
            .reasoning_budget_tokens(256)
    );

    let mut response = model.respond(request);

    while let Some(thought) = poll_fn(|cx| Pin::new(&mut response).poll_reasoning_next(cx)).await {
        println!("🤔 {}", thought?);
    }

    let mut final_text = String::new();
    while let Some(chunk) = response.next().await {
        final_text.push_str(&chunk?);
    }
    println!("Answer: {final_text}");
    Ok(())
}
```

*(The helper `reasoning()` is just `StreamExt::next` over `poll_reasoning_next`; see `examples/tool_macro.rs` for a complete version.)*

### Function Calling

```rust
use aither::{LanguageModel, llm::{Message, Request, Tool}};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

#[derive(JsonSchema, Deserialize, Serialize)]
struct WeatherQuery {
    location: String,
    units: Option<String>,
}

struct WeatherTool;

impl Tool for WeatherTool {
    const NAME: &str = "get_weather";
    const DESCRIPTION: &str = "Get current weather for a location";
    type Arguments = WeatherQuery;
    
    async fn call(&mut self, args: Self::Arguments) -> aither::Result {
        Ok(format!("Weather in {}: 22°C, sunny", args.location))
    }
}

async fn weather_bot(model: impl LanguageModel) -> aither::Result {
    let request = Request::new([
        Message::user("What's the weather like in Tokyo?")
    ]).with_tool(WeatherTool);
    
    let response: String = model.generate(request).await?;
    Ok(response)
}
```

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
| `aither-core` | No-std traits (`LanguageModel`, `LLMResponse`, `LLMRequest`, embedders, moderation, …) |
| `aither-openai` | Provider bindings for OpenAI-compatible chat, images, audio, and moderation |
| `aither-gemini` | Google Gemini bindings with tool looping and thinking budgets |
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

[`LLMResponse`]: core/src/llm/mod.rs
