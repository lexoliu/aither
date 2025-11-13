//! # aither
//!
//! **Write AI applications that work with any provider** 🚀
//!
//! This crate provides unified trait abstractions for AI models, letting you write code once
//! and switch between providers (`OpenAI`, `Anthropic`, local models, etc.) without changing your application logic.
//!
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Your App      │───▶│    aither      │◀───│   Providers     │
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
//! | **Language Models** | [`LanguageModel`] | Text generation, conversations, structured output |
//! | **Text Streaming** | [`TextStream`] | Unified interface for streaming text responses |
//! | **Embeddings** | [`EmbeddingModel`] | Convert text to vectors for semantic search |
//! | **Image Generation** | [`ImageGenerator`] | Create images with progressive quality improvement |
//! | **Text-to-Speech** | [`AudioGenerator`] | Generate speech audio from text |
//! | **Speech-to-Text** | [`AudioTranscriber`] | Transcribe audio to text |
//! | **Content Moderation** | [`Moderation`] | Detect policy violations with confidence scores |
//!
//! ## Examples
//!
//! ### Basic Chat Bot
//!
//! ```rust
//! use aither::{LanguageModel, llm::{Message, Request}};
//! use futures_lite::StreamExt;
//!
//! async fn chat_example(model: impl LanguageModel) -> aither::Result {
//!     let messages = [
//!         Message::system("You are a helpful assistant"),
//!         Message::user("What's the capital of France?")
//!     ];
//!     
//!     let request = Request::new(messages);
//!     let mut response = model.respond(request);
//!     
//!     Ok(response.await?)
//! }
//! ```
//!
//! ### Structured Output with Tools
//!
//! ```rust
//! use aither::{LanguageModel, llm::{Message, Request, Tool}};
//! use serde::{Deserialize, Serialize};
//! use schemars::JsonSchema;
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! struct WeatherQuery {
//!     location: String,
//!     units: Option<String>,
//! }
//!
//! struct WeatherTool;
//!
//! impl Tool for WeatherTool {
//!     const NAME: &str = "get_weather";
//!     const DESCRIPTION: &str = "Get current weather for a location";
//!     type Arguments = WeatherQuery;
//!     
//!     async fn call(&mut self, args: Self::Arguments) -> aither::Result {
//!         Ok(format!("Weather in {}: 22°C, sunny", args.location))
//!     }
//! }
//!
//! async fn weather_bot(model: impl LanguageModel) -> aither::Result {
//!     let request = Request::new(vec![
//!         Message::user("What's the weather like in Tokyo?")
//!     ]).with_tool(WeatherTool);
//!     
//!     // Model can now call the weather tool automatically
//!     let response: String = model.generate(request).await?;
//!     Ok(response)
//! }
//! ```
//!
//! See [`llm::tool`] for more details on using tools with language models.
//!
//! ### Semantic Search with Embeddings
//!
//! ```rust
//! use aither::EmbeddingModel;
//!
//! async fn find_similar_docs(
//!     model: impl EmbeddingModel,
//!     query: &str,
//!     documents: &[&str]
//! ) -> aither::Result<Vec<f32>> {
//!     // Convert query to vector
//!     let query_embedding = model.embed(query).await?;
//!     
//!     // In a real app, you'd compare with document embeddings
//!     // and find the most similar ones using cosine similarity
//!     
//!     Ok(query_embedding)
//! }
//! ```
//!
//! ### Progressive Image Generation
//!
//! ```rust
//! use aither::{ImageGenerator, image::{Prompt, Size}};
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
