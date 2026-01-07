//! LLM response events.
//!
//! The [`Event`] enum represents all possible events emitted by a language model
//! during response generation. This provides a low-level, provider-agnostic interface
//! for streaming LLM responses.
//!
//! # Event Types
//!
//! - [`Event::Text`] - Visible text output
//! - [`Event::Reasoning`] - Internal reasoning/thinking (for reasoning models)
//! - [`Event::ToolCall`] - Request to execute a tool (NOT auto-executed)
//! - [`Event::BuiltInToolResult`] - Result from provider's built-in tool (e.g., Google Search)
//! - [`Event::Usage`] - Token usage and cost information
//!
//! # Design
//!
//! The core crate only emits events - it does NOT execute tool calls.
//! Tool execution is the responsibility of higher-level abstractions like `aither-agent`.
//! This separation allows:
//! - Full control over tool execution (hooks, compression, error handling)
//! - Clean separation between LLM communication and agent logic
//! - Proper context management between tool calls

use alloc::string::{String, ToString};
use serde_json::Value;

/// Token usage information from a model response.
///
/// Providers should emit this at the end of each response stream.
/// Token counts and costs are optional since not all providers report them.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Usage {
    /// Number of tokens in the prompt/input.
    pub prompt_tokens: Option<u32>,
    /// Number of tokens in the completion/output.
    pub completion_tokens: Option<u32>,
    /// Total tokens (prompt + completion).
    pub total_tokens: Option<u32>,
    /// Tokens used for reasoning/thinking (for reasoning models).
    pub reasoning_tokens: Option<u32>,
    /// Tokens read from cache (for providers with prompt caching).
    pub cache_read_tokens: Option<u32>,
    /// Tokens written to cache.
    pub cache_write_tokens: Option<u32>,
    /// Estimated cost in USD for this request.
    pub cost_usd: Option<f64>,
}

impl Usage {
    /// Creates a new usage with basic token counts.
    #[must_use]
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens: Some(prompt_tokens),
            completion_tokens: Some(completion_tokens),
            total_tokens: Some(prompt_tokens + completion_tokens),
            reasoning_tokens: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
            cost_usd: None,
        }
    }

    /// Adds reasoning token count.
    #[must_use]
    pub const fn with_reasoning_tokens(mut self, tokens: u32) -> Self {
        self.reasoning_tokens = Some(tokens);
        self
    }

    /// Adds cache token counts.
    #[must_use]
    pub const fn with_cache_tokens(mut self, read: u32, write: u32) -> Self {
        self.cache_read_tokens = Some(read);
        self.cache_write_tokens = Some(write);
        self
    }

    /// Adds estimated cost.
    #[must_use]
    pub const fn with_cost(mut self, cost_usd: f64) -> Self {
        self.cost_usd = Some(cost_usd);
        self
    }

    /// Accumulates usage from another instance.
    pub fn accumulate(&mut self, other: &Self) {
        if let Some(v) = other.prompt_tokens {
            *self.prompt_tokens.get_or_insert(0) += v;
        }
        if let Some(v) = other.completion_tokens {
            *self.completion_tokens.get_or_insert(0) += v;
        }
        if let Some(v) = other.total_tokens {
            *self.total_tokens.get_or_insert(0) += v;
        }
        if let Some(v) = other.reasoning_tokens {
            *self.reasoning_tokens.get_or_insert(0) += v;
        }
        if let Some(v) = other.cache_read_tokens {
            *self.cache_read_tokens.get_or_insert(0) += v;
        }
        if let Some(v) = other.cache_write_tokens {
            *self.cache_write_tokens.get_or_insert(0) += v;
        }
        if let Some(v) = other.cost_usd {
            *self.cost_usd.get_or_insert(0.0) += v;
        }
    }
}

/// Events emitted by a language model during response generation.
///
/// This is the primary output type from [`LanguageModel::respond`].
/// Consumers should handle each event type appropriately.
///
/// # Example
///
/// ```rust,ignore
/// use futures_lite::StreamExt;
///
/// let mut stream = model.respond(request);
/// while let Some(event) = stream.next().await {
///     match event? {
///         Event::Text(text) => print!("{}", text),
///         Event::Reasoning(thought) => eprintln!("[thinking] {}", thought),
///         Event::ToolCall(call) => {
///             // Execute tool and continue conversation
///             let result = execute_tool(&call).await;
///             // ... add result to messages and continue
///         }
///         Event::BuiltInToolResult { tool, result } => {
///             println!("[{}] {}", tool, result);
///         }
///         Event::Usage(usage) => {
///             println!("Tokens used: {:?}", usage.total_tokens);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub enum Event {
    /// Visible text chunk from the model.
    ///
    /// These chunks should be concatenated to form the complete response.
    Text(String),

    /// Internal reasoning or thinking from reasoning models.
    ///
    /// Not all models emit reasoning. For models like Claude with extended thinking
    /// or OpenAI's o1, this contains the model's internal thought process.
    /// This is for observability only - it's not part of the conversation.
    Reasoning(String),

    /// Request to execute a tool.
    ///
    /// **Important**: The core crate does NOT execute tool calls.
    /// This event indicates the model wants to use a tool. The consumer
    /// (typically an agent) should:
    /// 1. Execute the tool
    /// 2. Add the result to the conversation
    /// 3. Continue the conversation with the model
    ToolCall(ToolCall),

    /// Result from a provider's built-in tool.
    ///
    /// Some providers have native tools that are executed server-side:
    /// - Gemini: Google Search grounding
    /// - OpenAI: Code interpreter, file search
    /// - Claude: (future built-in tools)
    ///
    /// These are already executed - this event contains the result.
    BuiltInToolResult {
        /// Name of the built-in tool that was executed.
        tool: String,
        /// Result from the tool execution.
        result: String,
    },

    /// Token usage and cost information.
    ///
    /// Emitted at the end of a response stream with usage statistics.
    /// Use this to track token consumption and costs across requests.
    Usage(Usage),
}

impl Event {
    /// Creates a text event.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Creates a reasoning event.
    #[must_use]
    pub fn reasoning(thought: impl Into<String>) -> Self {
        Self::Reasoning(thought.into())
    }

    /// Creates a tool call event.
    #[must_use]
    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self::ToolCall(ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        })
    }

    /// Creates a built-in tool result event.
    #[must_use]
    pub fn builtin_result(tool: impl Into<String>, result: impl Into<String>) -> Self {
        Self::BuiltInToolResult {
            tool: tool.into(),
            result: result.into(),
        }
    }

    /// Creates a usage event.
    #[must_use]
    pub const fn usage(usage: Usage) -> Self {
        Self::Usage(usage)
    }

    /// Returns the text content if this is a Text event.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the reasoning content if this is a Reasoning event.
    #[must_use]
    pub fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::Reasoning(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the tool call if this is a ToolCall event.
    #[must_use]
    pub const fn as_tool_call(&self) -> Option<&ToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// Returns true if this is a text event.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns true if this is a tool call event.
    #[must_use]
    pub const fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall(_))
    }

    /// Returns the usage info if this is a Usage event.
    #[must_use]
    pub const fn as_usage(&self) -> Option<&Usage> {
        match self {
            Self::Usage(u) => Some(u),
            _ => None,
        }
    }

    /// Returns true if this is a usage event.
    #[must_use]
    pub const fn is_usage(&self) -> bool {
        matches!(self, Self::Usage(_))
    }
}

/// A request from the model to execute a tool.
///
/// This represents an "intent" to call a tool - the tool has NOT been executed.
/// The consumer is responsible for:
/// 1. Looking up the tool by name
/// 2. Parsing and validating arguments
/// 3. Executing the tool
/// 4. Returning results to the model
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    ///
    /// Used to correlate tool results with their requests when
    /// continuing the conversation.
    pub id: String,

    /// Name of the tool to execute.
    pub name: String,

    /// Arguments to pass to the tool, as a JSON value.
    ///
    /// The structure depends on the tool's schema.
    pub arguments: Value,
}

impl ToolCall {
    /// Creates a new tool call.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Returns the arguments as a JSON string.
    #[must_use]
    pub fn arguments_json(&self) -> String {
        self.arguments.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_constructors() {
        let text = Event::text("hello");
        assert!(text.is_text());
        assert_eq!(text.as_text(), Some("hello"));

        let reasoning = Event::reasoning("thinking...");
        assert_eq!(reasoning.as_reasoning(), Some("thinking..."));

        let tool = Event::tool_call("call_1", "search", serde_json::json!({"query": "rust"}));
        assert!(tool.is_tool_call());
        let call = tool.as_tool_call().unwrap();
        assert_eq!(call.name, "search");
        assert_eq!(call.id, "call_1");
    }

    #[test]
    fn test_tool_call_arguments() {
        let call = ToolCall::new("id", "test", serde_json::json!({"key": "value"}));
        let json = call.arguments_json();
        assert!(json.contains("key"));
        assert!(json.contains("value"));
    }
}
