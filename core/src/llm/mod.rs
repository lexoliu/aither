//! # Language Models and Conversation Management
//!
//! This module provides everything you need to work with language models in a provider-agnostic way.
//! Build chat applications, generate structured output, and integrate tools without being tied to any specific AI service.
//!
//! ## Core Components
//!
//! - **[`LanguageModel`]** - The main trait for text generation and conversation
//! - **[`Request`]** - Encapsulates messages, tools, and parameters for model calls
//! - **[`Message`]** - Represents individual messages in a conversation
//! - **[`Tool`]** - Function calling interface for extending model capabilities
//!
//! ## Quick Start
//!
//! ### Basic Conversation
//!
//! ```rust
//! use aither::llm::{LanguageModel, Request, Message};
//! use futures_lite::StreamExt;
//!
//! async fn chat_with_model(model: impl LanguageModel) -> Result<String, Box<dyn std::error::Error>> {
//!     // Create a simple conversation
//!     let request = Request::oneshot(
//!         "You are a helpful assistant",
//!         "What's the capital of Japan?"
//!     );
//!
//!     // Stream the response
//!     let mut response = model.respond(request);
//!     let mut full_text = String::new();
//!     
//!     while let Some(chunk) = response.next().await {
//!         full_text.push_str(&chunk?);
//!     }
//!     
//!     Ok(full_text)
//! }
//! ```
//!
//! ### Multi-turn Conversation
//!
//! ```rust
//! use aither::llm::{Request, Message};
//!
//! let messages = [
//!     Message::system("You are a helpful coding assistant"),
//!     Message::user("How do I create a vector in Rust?"),
//!     Message::assistant("You can create a vector using `Vec::new()` or the `vec!` macro..."),
//!     Message::user("Can you show me an example?"),
//! ];
//!
//! let request = Request::new(messages);
//! ```
//!
//! ### Structured Output Generation
//!
//! ```rust
//! use aither::llm::{LanguageModel, Request, Message};
//! use serde::{Deserialize, Serialize};
//! use schemars::JsonSchema;
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! struct WeatherResponse {
//!     temperature: f32,
//!     condition: String,
//!     humidity: i32,
//! }
//!
//! async fn get_weather_data(model: impl LanguageModel) -> aither::Result<WeatherResponse> {
//!     let request = Request::oneshot(
//!         "Extract weather information from the following text",
//!         "It's 22°C and sunny with 65% humidity today"
//!     );
//!
//!     model.generate::<WeatherResponse>(request).await
//! }
//! ```
//!
//! ### Function Calling with Tools
//!
//! ```rust
//! use aither::llm::{Request, Message, Tool};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(JsonSchema, Deserialize)]
//! struct CalculatorArgs {
//!     operation: String,  // "add", "subtract", "multiply", "divide"
//!     x: f64,
//!     y: f64,
//! }
//!
//! struct Calculator;
//!
//! impl Tool for Calculator {
//!     const NAME: &str = "calculator";
//!     const DESCRIPTION: &str = "Performs basic arithmetic operations";
//!     type Arguments = CalculatorArgs;
//!
//!     async fn call(&mut self, args: Self::Arguments) -> aither::Result {
//!         let result = match args.operation.as_str() {
//!             "add" => args.x + args.y,
//!             "subtract" => args.x - args.y,
//!             "multiply" => args.x * args.y,
//!             "divide" => args.x / args.y,
//!             _ => return Err(anyhow::anyhow!("Unknown operation")),
//!         };
//!         Ok(result.to_string())
//!     }
//! }
//!
//! // Usage
//! let request = Request::new([
//!     Message::user("What's 15 multiplied by 23?")
//! ]).with_tool(Calculator);
//! ```
//!
//! ### Model Configuration
//!
//! ```rust
//! use aither::llm::{Request, Message, model::Parameters};
//!
//! let request = Request::new([
//!     Message::user("Write a creative story")
//! ]).with_parameters(
//!     Parameters::default()
//!         .temperature(0.8)        // More creative
//!         .top_p(0.9)             // Nucleus sampling
//!         .frequency_penalty(0.5)  // Reduce repetition
//! );
//! ```
//!
//!
//! ### Text Summarization
//!
//! ```rust
//! use aither::llm::LanguageModel;
//! use futures_lite::StreamExt;
//!
//! async fn summarize_text(model: impl LanguageModel, text: &str) -> Result<String, Box<dyn std::error::Error>> {
//!     let mut summary_stream = model.summarize(text);
//!     let mut summary = String::new();
//!     
//!     while let Some(chunk) = summary_stream.next().await {
//!         summary.push_str(&chunk?);
//!     }
//!     
//!     Ok(summary)
//! }
//! ```
//!
//! ### Text Categorization
//!
//! ```rust
//! use aither::llm::LanguageModel;
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! enum DocumentCategory {
//!     Technical,
//!     Marketing,
//!     Legal,
//!     Support,
//!     Internal,
//! }
//!
//! #[derive(JsonSchema, Deserialize, Serialize)]
//! struct ClassificationResult {
//!     category: DocumentCategory,
//!     confidence: f32,
//!     reasoning: String,
//! }
//!
//! async fn categorize_document(model: impl LanguageModel, text: &str) -> aither::Result<ClassificationResult> {
//!     model.categorize::<ClassificationResult>(text).await
//! }
//! ```
//!
//! ## Message Types and Annotations
//!
//! Messages support rich content including file attachments and URL annotations:
//!
//! ```rust
//! use aither::llm::{Message, UrlAnnotation, Annotation};
//! use url::Url;
//!
//! let message = Message::user("Check this documentation")
//!     .with_attachment("file:///path/to/doc.pdf")
//!     .with_annotation(
//!         Annotation::url(
//!             "https://docs.rs/aither",
//!             "AI Types Documentation",
//!             "Rust crate for AI model abstractions",
//!             0,
//!             25,
//!         )
//!    );
//! ```
/// Assistant module for managing assistant-related functionality.
pub mod assistant;
/// Message types and conversation handling.
pub mod message;
/// Model profiles and capabilities.
pub mod model;
/// Provider module for managing language model providers and their configurations.
pub mod provider;
/// Tool system for function calling.
pub mod tool;
use crate::llm::{model::Parameters, tool::Tools};
use alloc::{boxed::Box, string::String, sync::Arc};
use async_stream::try_stream;
use core::future::Future;
use futures_core::Stream;
use futures_lite::{StreamExt, pin};
pub use message::{Annotation, Message, Role, UrlAnnotation};
pub use provider::LanguageModelProvider;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
pub use tool::Tool;

use crate::llm::{model::Profile, tool::json};

/// Creates a two-message conversation with system and user prompts.
///
/// Returns an array containing a [`Message`] with [`Role::System`] and a [`Message`] with [`Role::User`].
fn oneshot(system: impl Into<String>, user: impl Into<String>) -> [Message; 2] {
    [Message::system(system.into()), Message::user(user.into())]
}

/// Language models for text generation and conversation.
///
/// See the [module documentation](crate::llm) for examples and usage patterns.
pub trait LanguageModel: Sized + Send + Sync + 'static {
    /// The error type returned by this language model.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Generates streaming response to conversation.
    fn respond(
        &self,
        messages: &[Message],
        tools: &mut Tools,
        parameters: &Parameters,
    ) -> impl Stream<Item = Result<String, Self::Error>> + Send;

    /// Generates structured output conforming to JSON schema.
    fn generate<T: JsonSchema + DeserializeOwned>(
        &self,
        messages: &[Message],
        tools: &mut Tools,
        parameters: &Parameters,
    ) -> impl Future<Output = crate::Result<T>> + Send {
        generate(self, messages, tools, parameters)
    }

    /// Completes given text prefix.
    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send;

    /// Summarizes text.
    fn summarize(&self, text: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        summarize(self, text)
    }

    /// Categorizes text.
    fn categorize<T: JsonSchema + DeserializeOwned>(
        &self,
        text: &str,
    ) -> impl Future<Output = crate::Result<T>> + Send {
        categorize(self, text)
    }

    /// Returns model profile and capabilities.
    ///
    /// See [`Profile`] for details on model metadata.
    fn profile(&self) -> impl Future<Output = Profile> + Send;
}

macro_rules! impl_language_model {
    ($($name:ident),*) => {
        $(
            impl<T: LanguageModel> LanguageModel for $name<T> {
                type Error = T::Error;

                fn respond(
                    &self,
                    messages: &[Message],
                    tools: &mut Tools,
                    parameters: &Parameters,
                ) -> impl Stream<Item = Result<String, Self::Error>> + Send {
                    T::respond(self, messages, tools, parameters)
                }

                fn generate<U: JsonSchema + DeserializeOwned>(
                    &self,
                    messages: &[Message],
                    tools: &mut Tools,
                    parameters: &Parameters,
                ) -> impl Future<Output = crate::Result<U>> + Send {
                    T::generate(self, messages, tools, parameters)
                }

                fn complete(
                    &self,
                    prefix: &str,
                ) -> impl Stream<Item = Result<String, Self::Error>> + Send {
                    T::complete(self, prefix)
                }

                fn summarize(
                    &self,
                    text: &str,
                ) -> impl Stream<Item = Result<String, Self::Error>> + Send {
                    T::summarize(self, text)
                }

                fn categorize<U: JsonSchema + DeserializeOwned>(
                    &self,
                    text: &str,
                ) -> impl Future<Output = crate::Result<U>> + Send {
                    T::categorize(self, text)
                }

                fn profile(&self) -> impl Future<Output = Profile> + Send {
                    T::profile(self)
                }
            }
        )*
    };
}

mod prompts;

impl_language_model!(Arc, Box);

/// Collects all chunks from a stream of `Result<String, Err>` into a single `String`.
///
/// # Errors
///
/// Returns an error if any chunk in the stream is an `Err`.
pub async fn try_collect<S, Err>(stream: S) -> Result<String, Err>
where
    S: Stream<Item = Result<String, Err>>,
{
    pin!(stream);

    stream
        .try_fold(String::new(), |acc, chunk| Ok(acc + &chunk))
        .await
}
async fn generate<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    messages: &[Message],
    tools: &mut Tools,
    parameters: &Parameters,
) -> crate::Result<T> {
    let schema = json(&schema_for!(T));

    let prompt = prompts::generate(&schema);
    let mut messages = messages.to_vec();
    messages.push(Message::system(prompt));
    let stream = model.respond(&messages, tools, parameters);
    let response = try_collect(stream).await?;

    let value: T = serde_json::from_str(&response)?;

    Ok(value)
}

fn summarize<M: LanguageModel>(
    model: &M,
    text: &str,
) -> impl Stream<Item = Result<String, M::Error>> + Send {
    try_stream! {
        let messages = oneshot("Summarize text:", text);
        let mut tools = Tools::new();
        let parameters = Parameters::default();
        let stream=model.respond(&messages, &mut tools, &parameters);
        pin!(stream);
        while let Some(chunk) = stream.try_next().await? {
            yield chunk;
        }
    }
}

async fn categorize<T: JsonSchema + DeserializeOwned, M: LanguageModel>(
    model: &M,
    text: &str,
) -> crate::Result<T> {
    model
        .generate(
            &oneshot("Categorize text by provided schema", text),
            &mut Tools::new(),
            &Parameters::default(),
        )
        .await
}
