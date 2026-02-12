//! Agent events for streaming execution.

use crate::error::AgentError;

/// Events emitted during agent execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Text chunk from the LLM response.
    Text(String),

    /// Reasoning step (for models with extended thinking).
    Reasoning(String),

    /// Tool is about to be called.
    ToolCallStart {
        /// Unique identifier for this tool call.
        id: String,
        /// Name of the tool being called.
        name: String,
        /// JSON-encoded arguments.
        arguments: String,
    },

    /// Tool execution completed.
    ToolCallEnd {
        /// Unique identifier matching the start event.
        id: String,
        /// Name of the tool that was called.
        name: String,
        /// Result of the tool execution (Ok = success text, Err = error text).
        result: Result<String, String>,
    },

    /// Agent turn completed (may have more turns if tools were called).
    TurnComplete {
        /// The turn number (0-indexed).
        turn: usize,
        /// Whether this turn included tool calls.
        has_tool_calls: bool,
    },

    /// Agent finished processing successfully.
    Complete {
        /// The final response text.
        final_text: String,
        /// Total number of turns taken.
        turns: usize,
    },

    /// Token usage information from LLM.
    Usage(aither_core::llm::Usage),

    /// Error occurred during execution.
    Error(AgentError),
}

impl AgentEvent {
    /// Creates a new text event.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Creates a new reasoning event.
    #[must_use]
    pub fn reasoning(content: impl Into<String>) -> Self {
        Self::Reasoning(content.into())
    }

    /// Creates a new tool call start event.
    #[must_use]
    pub fn tool_start(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Creates a new tool call end event with success.
    #[must_use]
    pub fn tool_success(
        id: impl Into<String>,
        name: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self::ToolCallEnd {
            id: id.into(),
            name: name.into(),
            result: Ok(result.into()),
        }
    }

    /// Creates a new tool call end event with failure.
    #[must_use]
    pub fn tool_failure(
        id: impl Into<String>,
        name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::ToolCallEnd {
            id: id.into(),
            name: name.into(),
            result: Err(error.into()),
        }
    }

    /// Creates a new turn complete event.
    #[must_use]
    pub const fn turn_complete(turn: usize, has_tool_calls: bool) -> Self {
        Self::TurnComplete {
            turn,
            has_tool_calls,
        }
    }

    /// Creates a new completion event.
    #[must_use]
    pub fn complete(final_text: impl Into<String>, turns: usize) -> Self {
        Self::Complete {
            final_text: final_text.into(),
            turns,
        }
    }

    /// Creates a new error event.
    #[must_use]
    pub const fn error(error: AgentError) -> Self {
        Self::Error(error)
    }

    /// Returns `true` if this is a completion event.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    /// Returns `true` if this is an error event.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Returns `true` if this is a terminal event (complete or error).
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.is_complete() || self.is_error()
    }
}
