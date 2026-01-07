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
