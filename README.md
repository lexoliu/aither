<div align="center">
<img src="logo.svg" alt="aither logo" width="150" height="150">

# aither

Providing unified trait abstractions for AI models


[![Crates.io](https://img.shields.io/crates/v/aither.svg)](https://crates.io/crates/aither)
[![Documentation](https://docs.rs/aither/badge.svg)](https://docs.rs/aither)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)

</div>


**Write AI applications that work with any provider** 🚀

Unified trait abstractions for AI models in Rust. Switch between OpenAI, Anthropic, local models, and more without changing your application logic.

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

## Features

- 🎯 **Provider Agnostic** - One interface, multiple providers
- ⚡ **Async Native** - Built with `async`/`await` and streaming
- 😊 **No-std Compatible** - Works in embedded and WASM environments
- 🛠️ **Function Calling** - Structured tool integration with JSON schemas
- 📸 **Multimodal** - Text, images, embeddings, and audio support
- 🔒 **Type Safe** - Leverage Rust's type system for AI applications

## Supported Capabilities

| Capability | Trait | Description |
|------------|-------|-------------|
| **Language Models** | `LanguageModel` | Text generation, conversations, streaming |
| **Embeddings** | `EmbeddingModel` | Convert text to vectors for semantic search |
| **Image Generation** | `ImageGenerator` | Create images with progressive quality |
| **Text-to-Speech** | `AudioGenerator` | Generate speech audio from text |
| **Speech-to-Text** | `AudioTranscriber` | Transcribe audio to text |
| **Content Moderation** | `Moderation` | Detect policy violations |

## Quick Start

```toml
[dependencies]
aither = "0.1.0"
```

### Basic Chat Bot

```rust
use aither::{LanguageModel, llm::{Message, Request}};
use futures_lite::StreamExt;

async fn chat_example(model: impl LanguageModel) -> aither::Result {
    let messages = [
        Message::system("You are a helpful assistant"),
        Message::user("What's the capital of France?")
    ];
    
    let request = Request::new(messages);
    let mut response = model.respond(request);
    
    let mut full_response = String::new();
    while let Some(chunk) = response.next().await {
        full_response.push_str(&chunk?);
    }
    
    Ok(full_response)
}
```

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

### Semantic Search

```rust
use aither::EmbeddingModel;

async fn find_similar_docs(
    model: impl EmbeddingModel,
    query: &str,
) -> aither::Result<Vec<f32>> {
    let query_embedding = model.embed(query).await?;
    println!("Embedding dimension: {}", query_embedding.len());
    Ok(query_embedding)
}
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

## License

MIT License - see [LICENSE](LICENSE) for details.