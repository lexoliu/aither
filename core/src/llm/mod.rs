//! # Language Models and Conversation Management
//!
//! This module provides everything you need to work with language models in a provider-agnostic way.
//! Build chat applications, generate structured output, and integrate tools without being tied to any specific AI service.
//!
//! ## Core Components
//!
//! - **[`LanguageModel`]** - The main trait for text generation and conversation
//! - **[`LLMRequest`]** - Encapsulates messages, tools, and parameters for model calls
//! - **[`LLMResponse`]** - Trait for streaming responses with reasoning steps
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
//!     let request = oneshot(
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
/// Deep research workflows and agent capabilities.
pub mod researcher;
mod response;
/// Tool system for function calling.
pub mod tool;

use crate::llm::{model::Parameters, tool::Tools};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use anyhow::{Context, anyhow, bail};
use core::{any::TypeId, future::Future, pin::Pin, task::Poll};
use futures_core::Stream;
use futures_lite::{StreamExt, pin};
pub use message::{Annotation, Message, Role, UrlAnnotation};
pub use provider::LanguageModelProvider;
pub use researcher::{
    ResearchCitation, ResearchEvent, ResearchFinding, ResearchOptions, ResearchReport,
    ResearchRequest, ResearchSource, ResearchStage, Researcher, ResearcherProfile,
};
pub use response::{ReasoningStream, ResponseChunk};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::Value;
pub use tool::Tool;

use crate::llm::{model::Profile, tool::json};

/// Response stream from LLM.
///
/// Modern LLM can be divided into two categories:
/// 1. Instant models - these models generate final answers directly, such as gpt-4-turbo.
/// 2. Reasoning models - these models generate intermediate reasoning steps before arriving at a final answer, such as o4-mini.
///
/// For reasoning models, it would take some time to complete the reasoning steps, and the final answer is only available after all reasoning steps are done.
///
/// `poll_reasoning_next` allows polling the next reasoning step, if applicable.
///
/// For instant models, `poll_reasoning_next` would always return `Poll::Ready(None)`.
/// For reasoning models, `poll_reasoning_next` would return `Poll::Ready(Some(_))` when the next reasoning step is available, or `Poll:Pending` until thinking is done.
///
/// The reasoning steps that you get from `poll_reasoning_next` - is not always the actual thinking process of the model, some closed-source models may hide the internal reasoning steps and only provide high-level summaries.
/// So that, the reasoning steps are mainly for user experience, rather than debugging the model's internal process. It would never be a part of the chat messages.
///
/// Implementors must also implement `IntoFuture` to allow collecting the full response.
pub trait LLMResponse:
    Stream<Item = Result<String, Self::Error>>
    + IntoFuture<Output = Result<String, Self::Error>, IntoFuture: Send>
    + Send
{
    /// The error type returned by this response stream.
    type Error: core::error::Error + Send + Sync + 'static;
    /// Polls the next reasoning step, if applicable.
    fn poll_reasoning_next(self: Pin<&mut Self>) -> Poll<Option<crate::Result>>;
}

/// Builder-style request passed into [`LanguageModel::respond`].
///
/// Wraps the full conversation, model parameters, and optional tool registry a provider
/// needs in order to execute a call. Requests start with owned messages/parameters and
/// can be paired with borrowed tool registries via [`LLMRequest::with_tools`].
#[derive(Debug)]
pub struct LLMRequest<'tools> {
    messages: Vec<Message>,
    parameters: Parameters,
    tools: Option<&'tools mut Tools>,
}

impl LLMRequest<'static> {
    /// Creates a request from the provided messages using default parameters.
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            parameters: Parameters::default(),
            tools: None,
        }
    }

    /// Attaches a mutable tool registry to the request.
    pub fn with_tools(self, tools: &mut Tools) -> LLMRequest<'_> {
        LLMRequest {
            messages: self.messages,
            parameters: self.parameters,
            tools: Some(tools),
        }
    }
}

impl<'tools> LLMRequest<'tools> {
    /// Overrides the sampling parameters used for this call.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Returns the current conversation messages.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns the current parameter snapshot.
    #[must_use]
    pub const fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Returns the attached tool registry if one was provided.
    #[must_use]
    pub fn tools(&mut self) -> Option<&mut Tools> {
        self.tools.as_deref_mut()
    }

    /// Breaks the request into owned components for providers that want to take ownership.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Message>, Parameters, Option<&'tools mut Tools>) {
        (self.messages, self.parameters, self.tools)
    }

    /// Invokes a registered tool by name.
    ///
    /// # Errors
    /// Returns an error if tool is not found or the tool call fails.
    pub async fn call_tool(&mut self, name: &str, args_json: &str) -> crate::Result<String> {
        if let Some(tools) = &mut self.tools {
            tools.call(name, args_json).await
        } else {
            bail!("Tool {name} not found, please check your spelling or may it is not existing")
        }
    }
}

/// Language models for text generation and conversation.
///
/// See the [module documentation](crate::llm) for examples and usage patterns.
pub trait LanguageModel: Sized + Send + Sync {
    /// The error type returned by this language model.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Generates streaming response to conversation.
    fn respond(&self, request: LLMRequest) -> impl LLMResponse<Error = Self::Error>;

    /// Generates structured output conforming to JSON schema.
    ///
    /// # Note for Implementors
    /// By default, we use a system prompt to instruct the model to generate structured output based on the provided JSON schema.
    /// However, that is not efficient enough, if supported, provider should override this method to provide native structured generation support.
    /// Native structured generation can apply decode rules in token-level, ensuring the output is always valid JSON, and reducing parsing errors.
    fn generate<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        request: LLMRequest,
    ) -> impl Future<Output = crate::Result<T>> + Send {
        async { structured_generate(self, request).await }
    }

    /// Completes given text prefix.
    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        // Let's give a default implementation using oneshot and respond.
        self.respond(oneshot("Please complete the following text:", prefix))
    }

    /// Summarizes text.
    ///
    /// # Note for Implementors
    /// By default, we provide a generic summarization prompt. However, some model provider, like Apple Intelligence, may have native summarization support.
    /// It would load a summarization-specific lora at runtime, providing better quality.
    fn summarize(&self, text: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        summarize(self, text)
    }

    /// Categorizes text.
    ///
    /// # Note for Implementors
    /// By default, we use a system prompt to instruct the model to categorize text based on the provided JSON schema.
    /// However, that is not efficient enough, if supported, provider should override this method to provide native categorization support.
    /// Native categorization can apply decode rules in token-level, ensuring the output is always valid JSON, and reducing parsing errors.
    /// Moreover, it can leverage specialized model capabilities for classification tasks if available.
    fn categorize<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        text: &str,
    ) -> impl Future<Output = crate::Result<T>> + Send {
        async { categorize_text(self, text).await }
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
                        request: LLMRequest,
                    ) -> impl LLMResponse<Error = Self::Error> {
                    T::respond(self, request)
                }

                fn generate<U: JsonSchema + DeserializeOwned + 'static>(
                    &self,
                    request: LLMRequest,
                ) -> impl Future<Output = crate::Result<U>> + Send {
                    T::generate(self, request)
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

                fn categorize<U: JsonSchema + DeserializeOwned + 'static>(
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

impl<T: LanguageModel> LanguageModel for &T {
    type Error = T::Error;

    fn respond(&self, request: LLMRequest) -> impl LLMResponse<Error = Self::Error> {
        T::respond(self, request)
    }

    fn generate<U: JsonSchema + DeserializeOwned + 'static>(
        &self,
        request: LLMRequest,
    ) -> impl Future<Output = crate::Result<U>> + Send {
        T::generate(self, request)
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        T::complete(self, prefix)
    }

    fn summarize(&self, text: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        T::summarize(self, text)
    }

    fn categorize<U: JsonSchema + DeserializeOwned + 'static>(
        &self,
        text: &str,
    ) -> impl Future<Output = crate::Result<U>> + Send {
        T::categorize(self, text)
    }

    fn profile(&self) -> impl Future<Output = Profile> + Send {
        T::profile(self)
    }
}

mod prompts;

impl_language_model!(Arc, Box);

// Internal helper, please use `LLMResponse::into_future` directly, it is more ergonomic.
// For instance, you can use `.await` on the response to collect the full text.
pub(crate) async fn try_collect<S, Err>(stream: S) -> Result<String, Err>
where
    S: Stream<Item = Result<String, Err>>,
{
    pin!(stream);

    stream
        .try_fold(String::new(), |acc, chunk| Ok(acc + &chunk))
        .await
}

async fn structured_generate<T: JsonSchema + DeserializeOwned + 'static, M: LanguageModel>(
    model: &M,
    mut request: LLMRequest<'_>,
) -> crate::Result<T> {
    let schema = schema_for!(T);

    // If it is a string, we are not required to set up structured generation and JSON schema.
    let json = if schema.as_value().is_string() {
        let stream = model.respond(request);
        let response = try_collect(stream).await?; // Note: this string do not have double quotes around it, so it is not a JSON string.
        // Let's encode it as JSON string.
        serde_json::to_string(&response)?
    } else {
        let schema = json(&schema);

        let prompt = prompts::generate(&schema);
        request.messages.push(Message::system(prompt));
        request.parameters.structured_outputs = true;

        let stream = model.respond(request);
        try_collect(stream).await?
    };

    parse_json_with_recovery(&json)
}

/// Convenience helper that creates a single system + user [`LLMRequest`].
pub fn oneshot(system: impl Into<String>, user: impl Into<String>) -> LLMRequest<'static> {
    let messages = vec![Message::system(system.into()), Message::user(user.into())];
    LLMRequest::new(messages)
}

fn summarize<M: LanguageModel>(
    model: &M,
    text: &str,
) -> impl Stream<Item = Result<String, M::Error>> + Send {
    let messages = oneshot("Summarize text:", text);
    model.respond(messages)
}

async fn categorize_text<T: JsonSchema + DeserializeOwned + 'static, M: LanguageModel>(
    model: &M,
    text: &str,
) -> crate::Result<T> {
    let request = oneshot("Categorize text by provided schema", text);
    model.generate(request).await
}

fn parse_json_with_recovery<T: DeserializeOwned + 'static>(json: &str) -> crate::Result<T> {
    let trimmed = json.trim();
    let mut last_error: Option<serde_json::Error> = None;
    let mut last_candidate: Option<String> = None;

    for candidate in build_json_candidates(trimmed) {
        match serde_json::from_str::<T>(&candidate) {
            Ok(value) => return Ok(value),
            Err(err) => {
                last_error = Some(err);
                last_candidate = Some(candidate);
            }
        }
    }

    if is_string_type::<T>() {
        if let Some(candidate) = last_candidate.clone() {
            if let Ok(value) = serde_json::from_str::<Value>(&candidate) {
                let text = match value {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                let encoded = serde_json::to_string(&text)
                    .map_err(|err| anyhow!(err))
                    .context("failed to encode fallback string while parsing structured output")?;
                if let Ok(value) = serde_json::from_str::<T>(&encoded) {
                    return Ok(value);
                }
            }
        }
    }

    let primary = last_error
        .map(anyhow::Error::new)
        .unwrap_or_else(|| anyhow!("structured output was empty or missing JSON block"));
    let snippet = last_candidate
        .as_deref()
        .unwrap_or(trimmed)
        .chars()
        .take(500)
        .collect::<String>();
    Err(primary.context(format!(
        "failed to parse structured output; sample: {snippet}"
    )))
}

fn strip_code_fences(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let fence_start = trimmed.find("```")?;
    let after_fence = &trimmed[fence_start + 3..];
    let mut lines = after_fence.lines();
    let _maybe_lang = lines.next(); // optional language tag
    let body = lines.collect::<Vec<_>>().join("\n");
    let content = if let Some(end) = body.rfind("```") {
        &body[..end]
    } else {
        &body
    };

    let cleaned = content.trim();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

fn extract_json_block(raw: &str) -> Option<String> {
    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        if end >= start {
            let candidate = &raw[start..=end];
            if !candidate.trim().is_empty() {
                return Some(candidate.trim().to_string());
            }
        }
    }
    if let (Some(start), Some(end)) = (raw.find('['), raw.rfind(']')) {
        if end >= start {
            let candidate = &raw[start..=end];
            if !candidate.trim().is_empty() {
                return Some(candidate.trim().to_string());
            }
        }
    }
    None
}

fn build_json_candidates(raw: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // 1) Raw
    if !raw.is_empty() {
        candidates.push(raw.to_string());
    }

    // 2) Code fences
    if let Some(fenced) = strip_code_fences(raw) {
        candidates.push(fenced);
    }

    // 3) Extract first JSON object/array
    if let Some(block) = extract_json_block(raw) {
        candidates.push(block);
    }

    // 4) Unquote JSON encoded as a single string
    if let Some(dequoted) = dequote_json_string(raw) {
        candidates.push(dequoted);
    }

    // 5) Drop leading "json"/"JSON" labels
    if let Some(stripped) = strip_leading_label(raw, "json") {
        candidates.push(stripped);
    }

    // Remove duplicates while preserving order.
    let mut deduped = Vec::new();
    for candidate in candidates {
        if deduped.iter().all(|seen| seen != &candidate) {
            deduped.push(candidate);
        }
    }
    deduped
}

fn dequote_json_string(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if !(trimmed.starts_with('"') && trimmed.ends_with('"')) {
        return None;
    }
    let inner: String = serde_json::from_str(trimmed).ok()?;
    if inner.trim().is_empty() {
        None
    } else {
        Some(inner)
    }
}

fn strip_leading_label(raw: &str, label: &str) -> Option<String> {
    let trimmed = raw.trim_start();
    if !trimmed.to_ascii_lowercase().starts_with(label) {
        return None;
    }
    let stripped = trimmed[label.len()..]
        .trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '-')
        .trim();
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

fn is_string_type<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<String>()
}

#[cfg(test)]
mod tests {
    use super::parse_json_with_recovery;
    use alloc::string::String;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Foo {
        a: u8,
    }

    #[test]
    fn parses_plain_json() {
        let foo: Foo = parse_json_with_recovery(r#"{"a":1}"#).unwrap();
        assert_eq!(foo, Foo { a: 1 });
    }

    #[test]
    fn parses_code_fence_json() {
        let foo: Foo = parse_json_with_recovery("```json\n{\"a\":2}\n```").unwrap();
        assert_eq!(foo, Foo { a: 2 });
    }

    #[test]
    fn parses_embedded_block() {
        let foo: Foo = parse_json_with_recovery("noise {\"a\":3} trailing").unwrap();
        assert_eq!(foo, Foo { a: 3 });
    }

    #[test]
    fn parses_quoted_json_string() {
        let foo: Foo = parse_json_with_recovery(r#""{\"a\":4}""#).unwrap();
        assert_eq!(foo, Foo { a: 4 });
    }

    #[test]
    fn parses_labeled_json() {
        let foo: Foo = parse_json_with_recovery("json {\"a\":5}").unwrap();
        assert_eq!(foo, Foo { a: 5 });
    }

    #[test]
    fn coerces_object_to_string() {
        let value: String =
            parse_json_with_recovery(r#"{"title":"summary","type":"content"}"#).unwrap();
        assert!(
            value.contains("\"title\":\"summary\"") && value.contains("\"type\":\"content\""),
            "unexpected value: {value}"
        );
    }
}
