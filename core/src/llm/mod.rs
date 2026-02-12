//! # Language Models and Conversation Management
//!
//! This module provides everything you need to work with language models in a provider-agnostic way.
//! Build chat applications, generate structured output, and integrate tools without being tied to any specific AI service.
//!
//! ## Core Components
//!
//! - **[`LanguageModel`]** - The main trait for text generation and conversation
//! - **[`LLMRequest`]** - Encapsulates messages, tools, and parameters for model calls
//! - **[`Event`]** - Stream events from the model (text, reasoning, tool calls)
//! - **[`Message`]** - Represents individual messages in a conversation
//! - **[`Tool`]** - Function calling interface for extending model capabilities
//!
//! ## Design Philosophy
//!
//! The core crate provides a **low-level API** that emits events without executing tools.
//! Tool execution is the responsibility of higher-level abstractions like `aither-agent`.
//!
//! This design allows:
//! - Full control over tool execution flow
//! - Hooks for intercepting and modifying tool calls
//! - Proper context management between turns
//! - Clean separation between LLM communication and agent logic
//!
//! ## Quick Start
//!
//! ### Basic Conversation
//!
//! ```rust,ignore
//! use aither::llm::{LanguageModel, Event, oneshot};
//! use futures_lite::StreamExt;
//!
//! async fn chat_with_model(model: impl LanguageModel) -> Result<String, Box<dyn std::error::Error>> {
//!     let request = oneshot("You are a helpful assistant", "What's the capital of Japan?");
//!     let mut stream = model.respond(request);
//!     let mut full_text = String::new();
//!
//!     while let Some(event) = stream.next().await {
//!         match event? {
//!             Event::Text(chunk) => full_text.push_str(&chunk),
//!             Event::Reasoning(thought) => eprintln!("[thinking] {}", thought),
//!             Event::ToolCall(call) => {
//!                 // Handle tool call (typically done by agent crate)
//!                 println!("Tool requested: {}", call.name);
//!             }
//!             _ => {}
//!         }
//!     }
//!
//!     Ok(full_text)
//! }
//! ```
//!
//! ### With Tools (Agent-Controlled)
//!
//! ```rust,ignore
//! use aither::llm::{LanguageModel, Event, LLMRequest, Message};
//!
//! // The core crate does NOT execute tools - it emits ToolCall events.
//! // Tool execution should be handled by the agent crate.
//! let request = LLMRequest::new([Message::user("What's the weather?")])
//!     .with_tool_definitions(vec![weather_tool_definition()]);
//!
//! let mut stream = model.respond(request);
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         Event::ToolCall(call) => {
//!             // Execute tool and continue conversation
//!             let result = my_tool_executor.execute(&call).await;
//!             // Add result to messages and send another request...
//!         }
//!         _ => {}
//!     }
//! }
//! ```

/// Assistant module for managing assistant-related functionality.
pub mod assistant;
/// Event types for streaming responses.
pub mod event;
/// Message types and conversation handling.
pub mod message;
/// Model profiles and capabilities.
pub mod model;
/// Provider module for managing language model providers and their configurations.
pub mod provider;
/// Deep research workflows and agent capabilities.
pub mod researcher;
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
use anyhow::{Context, anyhow};
use core::{any::TypeId, future::Future};
pub use event::{Event, ToolCall, Usage};
use futures_core::Stream;
use futures_lite::{StreamExt, pin};
pub use message::{Message, Role};
pub use provider::LanguageModelProvider;
pub use researcher::{
    ResearchCitation, ResearchEvent, ResearchFinding, ResearchOptions, ResearchReport,
    ResearchRequest, ResearchSource, ResearchStage, Researcher, ResearcherProfile,
};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
pub use tool::{Tool, ToolOutput};

use crate::llm::{model::Profile, tool::json};

/// Builder-style request passed into [`LanguageModel::respond`].
///
/// Wraps the full conversation, model parameters, and tool definitions a provider
/// needs in order to execute a call.
#[derive(Debug, Clone)]
pub struct LLMRequest {
    messages: Vec<Message>,
    parameters: Parameters,
    tool_definitions: Vec<tool::ToolDefinition>,
}

impl LLMRequest {
    /// Creates a request from the provided messages using default parameters.
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            parameters: Parameters::default(),
            tool_definitions: Vec::new(),
        }
    }

    /// Adds tool definitions to the request.
    ///
    /// These define what tools the model can request. The model will emit
    /// [`Event::ToolCall`] events when it wants to use a tool.
    #[must_use]
    pub fn with_tool_definitions(mut self, definitions: Vec<tool::ToolDefinition>) -> Self {
        self.tool_definitions = definitions;
        self
    }

    /// Adds a single tool definition.
    #[must_use]
    pub fn with_tool<T: Tool>(mut self, tool: &T) -> Self {
        self.tool_definitions.push(tool::ToolDefinition::new(tool));
        self
    }

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

    /// Returns a mutable reference to messages for modification.
    pub const fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    /// Returns the current parameter snapshot.
    #[must_use]
    pub const fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Returns the tool definitions.
    #[must_use]
    pub fn tool_definitions(&self) -> &[tool::ToolDefinition] {
        &self.tool_definitions
    }

    /// Breaks the request into owned components.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Message>, Parameters, Vec<tool::ToolDefinition>) {
        (self.messages, self.parameters, self.tool_definitions)
    }
}

/// Legacy request builder that supports mutable tool registry.
///
/// This is provided for backwards compatibility with providers that
/// handle tool execution internally (built-in tools).
#[derive(Debug)]
pub struct LLMRequestWithTools<'tools> {
    inner: LLMRequest,
    tools: &'tools mut Tools,
}

impl LLMRequest {
    /// Attaches a mutable tool registry to the request.
    ///
    /// This is for providers that execute tools internally (e.g., built-in tools).
    /// For agent-controlled tool execution, use `with_tool_definitions` instead.
    pub fn with_tools(self, tools: &mut Tools) -> LLMRequestWithTools<'_> {
        let definitions = tools.definitions();
        LLMRequestWithTools {
            inner: self.with_tool_definitions(definitions),
            tools,
        }
    }
}

impl<'tools> LLMRequestWithTools<'tools> {
    /// Returns the inner request.
    #[must_use]
    pub const fn request(&self) -> &LLMRequest {
        &self.inner
    }

    /// Returns the tool registry.
    #[must_use]
    pub const fn tools(&mut self) -> &mut Tools {
        self.tools
    }

    /// Breaks into components.
    #[must_use]
    pub fn into_parts(self) -> (LLMRequest, &'tools mut Tools) {
        (self.inner, self.tools)
    }

    /// Invokes a registered tool by name.
    ///
    /// # Errors
    /// Returns an error if tool is not found or the tool call fails.
    pub async fn call_tool(&mut self, name: &str, args_json: &str) -> crate::Result<ToolOutput> {
        self.tools.call(name, args_json).await
    }
}

/// Language models for text generation and conversation.
///
/// The `respond` method returns a stream of [`Event`]s. Tool calls are emitted
/// as events and are NOT automatically executed - this allows higher-level
/// abstractions (like agents) to control tool execution.
///
/// See the [module documentation](crate::llm) for examples and usage patterns.
pub trait LanguageModel: Sized + Send + Sync {
    /// The error type returned by this language model.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Generates a streaming response to a conversation.
    ///
    /// Returns a stream of [`Event`]s including:
    /// - `Event::Text` - Visible text chunks
    /// - `Event::Reasoning` - Internal reasoning (for reasoning models)
    /// - `Event::ToolCall` - Requests to execute tools (NOT auto-executed)
    /// - `Event::BuiltInToolResult` - Results from provider's built-in tools
    fn respond(&self, request: LLMRequest)
    -> impl Stream<Item = Result<Event, Self::Error>> + Send;

    /// Generates a streaming response with a mutable tool registry.
    ///
    /// This is for providers that support built-in tool execution.
    /// The default implementation ignores the tools and delegates to `respond`.
    fn respond_with_tools(
        &self,
        request: LLMRequestWithTools<'_>,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let (inner, _tools) = request.into_parts();
        self.respond(inner)
    }

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
    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.respond(oneshot("Please complete the following text:", prefix))
    }

    /// Summarizes text.
    ///
    /// # Note for Implementors
    /// By default, we provide a generic summarization prompt. However, some model provider, like Apple Intelligence, may have native summarization support.
    /// It would load a summarization-specific lora at runtime, providing better quality.
    fn summarize(&self, text: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        summarize(self, text)
    }

    /// Categorizes text.
    ///
    /// # Note for Implementors
    /// By default, we use a system prompt to instruct the model to categorize text based on the provided JSON schema.
    /// However, that is not efficient enough, if supported, provider should override this method to provide native categorization support.
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
                ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
                    T::respond(self, request)
                }

                fn respond_with_tools(
                    &self,
                    request: LLMRequestWithTools<'_>,
                ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
                    T::respond_with_tools(self, request)
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
                ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
                    T::complete(self, prefix)
                }

                fn summarize(
                    &self,
                    text: &str,
                ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
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

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        T::respond(self, request)
    }

    fn respond_with_tools(
        &self,
        request: LLMRequestWithTools<'_>,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        T::respond_with_tools(self, request)
    }

    fn generate<U: JsonSchema + DeserializeOwned + 'static>(
        &self,
        request: LLMRequest,
    ) -> impl Future<Output = crate::Result<U>> + Send {
        T::generate(self, request)
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        T::complete(self, prefix)
    }

    fn summarize(&self, text: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
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

/// Collects text from an event stream.
///
/// # Errors
///
/// Returns the first stream error encountered while collecting text chunks.
pub async fn collect_text<S, E>(stream: S) -> Result<String, E>
where
    S: Stream<Item = Result<Event, E>>,
{
    pin!(stream);
    let mut result = String::new();
    while let Some(event) = stream.next().await {
        if let Event::Text(text) = event? {
            result.push_str(&text);
        }
    }
    Ok(result)
}

async fn structured_generate<T: JsonSchema + DeserializeOwned + 'static, M: LanguageModel>(
    model: &M,
    mut request: LLMRequest,
) -> crate::Result<T> {
    let schema = schema_for!(T);

    // If it is a string, we are not required to set up structured generation and JSON schema.
    let json = if schema.as_value().is_string() {
        let stream = model.respond(request);
        let response = collect_text(stream).await?;
        // Let's encode it as JSON string.
        serde_json::to_string(&response)?
    } else {
        let schema = json(&schema);
        let prompt = prompts::generate(&schema);
        request.messages.push(Message::system(prompt));
        request.parameters.structured_outputs = true;

        let stream = model.respond(request);
        collect_text(stream).await?
    };

    parse_json_with_recovery(&json)
}

/// Convenience helper that creates a single system + user [`LLMRequest`].
pub fn oneshot(system: impl Into<String>, user: impl Into<String>) -> LLMRequest {
    let messages = vec![Message::system(system.into()), Message::user(user.into())];
    LLMRequest::new(messages)
}

fn summarize<M: LanguageModel>(
    model: &M,
    text: &str,
) -> impl Stream<Item = Result<Event, M::Error>> + Send {
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
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) {
                let text = match value {
                    serde_json::Value::String(s) => s,
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

    let primary = last_error.map_or_else(
        || anyhow!("structured output was empty or missing JSON block"),
        anyhow::Error::new,
    );
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
    let _maybe_lang = lines.next();
    let body = lines.collect::<Vec<_>>().join("\n");
    let content = body.rfind("```").map_or(body.as_str(), |end| &body[..end]);

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

    if !raw.is_empty() {
        candidates.push(raw.to_string());
    }

    if let Some(fenced) = strip_code_fences(raw) {
        candidates.push(fenced);
    }

    if let Some(block) = extract_json_block(raw) {
        candidates.push(block);
    }

    if let Some(dequoted) = dequote_json_string(raw) {
        candidates.push(dequoted);
    }

    if let Some(stripped) = strip_leading_label(raw, "json") {
        candidates.push(stripped);
    }

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
