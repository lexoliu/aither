//! # aither
//!
//! **Write AI applications that work with any provider** 🚀
//!
//! `aither-core` hosts the no-std trait APIs that power the rest of the workspace. Use it directly
//! (or through the top-level [`aither`](https://crates.io/crates/aither) crate) to describe portable
//! language models, embeddings, moderation, image/audio generators, and more.
//! Every provider crate simply implements these traits.
//!
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Your App      │───▶│    aither        │◀───│   Providers     │
//! │                 │    │   (this crate)   │    │                 │
//! │ - Chat bots     │    │                  │    │ - openai        │
//! │ - Search        │    │ - LanguageModel  │    │ - anthropic     │
//! │ - Content gen   │    │ - EmbeddingModel │    │ - llama.cpp     │
//! │ - Voice apps    │    │ - ImageGenerator │    │ - whisper       │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

//!
//! ## Supported AI Capabilities
//!
//! | Capability | Trait | Description |
//! |------------|-------|-------------|
//! | **Language Models** | [`LanguageModel`] | Streaming events (text, reasoning, tool calls) |
//! | **Embeddings** | [`EmbeddingModel`] | Convert text to vectors for semantic search |
//! | **Image Generation** | [`ImageGenerator`] | Create images with progressive quality improvement |
//! | **Text-to-Speech** | [`AudioGenerator`] | Generate speech audio from text |
//! | **Speech-to-Text** | [`AudioTranscriber`] | Transcribe audio to text |
//! | **Content Moderation** | [`Moderation`] | Detect policy violations with confidence scores |
//!
//! ## Examples
//!
//! ### Streaming Responses with Events
//!
//! ```rust,ignore
//! use aither_core::llm::{LanguageModel, Event, Message, LLMRequest, model::Parameters};
//! use futures_lite::StreamExt;
//!
//! async fn event_demo(model: impl LanguageModel) -> aither_core::Result {
//!     let request = LLMRequest::new([
//!         Message::user("Explain how rainbows form like I'm five."),
//!     ])
//!     .with_parameters(Parameters::default().include_reasoning(true));
//!
//!     let mut stream = model.respond(request);
//!     let mut answer = String::new();
//!
//!     while let Some(event) = stream.next().await {
//!         match event? {
//!             Event::Text(text) => answer.push_str(&text),
//!             Event::Reasoning(thought) => println!("thinking: {}", thought),
//!             Event::ToolCall(call) => println!("tool requested: {}", call.name),
//!             _ => {}
//!         }
//!     }
//!     Ok(answer)
//! }
//! ```
//!
//! ### Structured Output with Tools
//!
//! ```rust
//! use aither_core::llm::{LLMRequest, Message, Tool, ToolContext, ToolResult};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//! use std::borrow::Cow;
//!
//! /// Get current weather for a location.
//! #[derive(JsonSchema, Deserialize)]
//! struct WeatherQuery {
//!     /// City to report on, e.g. "Tokyo".
//!     location: String,
//! }
//!
//! struct WeatherTool;
//!
//! impl Tool for WeatherTool {
//!     fn name(&self) -> Cow<'static, str> {
//!         Cow::Borrowed("get_weather")
//!     }
//!
//!     type Arguments = WeatherQuery;
//!     type Res = ToolResult;
//!
//!     async fn call(
//!         &self,
//!         args: Self::Arguments,
//!         _cx: ToolContext,
//!     ) -> aither_core::Result<Self::Res> {
//!         Ok(ToolResult::text(format!("Weather in {}: 22°C, sunny", args.location)))
//!     }
//! }
//!
//! // Advertise the tool on a request. The model replies with a ToolCall event;
//! // executing it is up to the caller (see `aither-agent`).
//! let request = LLMRequest::new([Message::user("What is the weather in Tokyo?")])
//!     .with_tool(&WeatherTool);
//! ```
//!
//! See [`llm::tool`] for more details on using tools with language models.
//!
//! ### Semantic Search with Embeddings
//!
//! ```rust
//! use aither_core::EmbeddingModel;
//!
//! async fn embed_query(
//!     model: impl EmbeddingModel,
//!     query: &str,
//! ) -> aither_core::Result<Vec<f32>> {
//!     // Compare this against your stored document embeddings with cosine
//!     // similarity, or hand it to `aither-rag`.
//!     model.embed(query).await
//! }
//! ```
//!
//! ### Progressive Image Generation
//!
//! ```rust,ignore
//! use aither_core::{ImageGenerator, image::{Prompt, Size}};
//! use futures_lite::StreamExt;
//!
//! async fn generate_image(generator: impl ImageGenerator) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let prompt = Prompt::new("A beautiful sunset over mountains");
//!     let size = Size::square(1024);
//!     
//!     let mut image_stream = generator.create(prompt, size);
//!     let mut final_image = Vec::new();
//!     
//!     // Each iteration gives us a complete image with progressively better quality
//!     while let Some(image_result) = image_stream.next().await {
//!         let current_image = image_result?;
//!         final_image = current_image; // Keep the latest (highest quality) version
//!         
//!         // Optional: Display preview of current quality level
//!         println!("Received image update, {} bytes", final_image.len());
//!     }
//!     
//!     Ok(final_image) // Return the final highest-quality image
//! }
//! ```
//!
//! ## Modules
//!
//! - [`audio`] — text-to-speech and transcription traits.
//! - [`embedding`] — turn text into dense vectors.
//! - [`image`] — image generation + editing APIs.
//! - [`llm`] — request builders, messages, provider traits, reasoning streams.
//! - [`moderation`] — moderation scoring traits.
//!
//!

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/lexoliu/aither/main/logo.svg",
    html_favicon_url = "https://raw.githubusercontent.com/lexoliu/aither/main/logo.svg"
)]
#![no_std]
extern crate alloc;

/// Audio generation and transcription.
///
/// Contains [`AudioGenerator`] and [`AudioTranscriber`] traits.
pub mod audio;
/// Text embeddings.
pub mod embedding;
/// Text-to-image generation.
///
/// Contains [`ImageGenerator`] trait for creating images from text.
pub mod image;
pub mod llm;

/// Content moderation utilities.
///
/// Contains traits and types for detecting and handling unsafe or inappropriate content.
pub mod moderation;

use alloc::string::String;

#[doc(inline)]
pub use audio::{AudioGenerator, AudioTranscriber};
#[doc(inline)]
pub use embedding::EmbeddingModel;
#[doc(inline)]
pub use image::ImageGenerator;
#[doc(inline)]
pub use llm::LanguageModel;
#[doc(inline)]
pub use moderation::Moderation;

/// Result type used throughout the crate.
///
/// Type alias for [`anyhow::Result<T>`](anyhow::Result) with [`String`] as default success type.
pub type Result<T = String> = anyhow::Result<T>;

pub use anyhow::Error;

// Re-export procedural macros
#[cfg(feature = "derive")]
pub use crate::llm::tool::tool;
