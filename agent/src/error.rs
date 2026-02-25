//! Agent error types.

use core::fmt;

/// Errors that can occur during agent execution.
#[derive(Debug, Clone)]
pub enum AgentError {
    /// LLM returned an error.
    Llm(String),

    /// Tool execution failed.
    ToolExecution {
        /// Name of the tool that failed.
        name: String,
        /// The underlying error message.
        error: String,
    },

    /// Maximum iterations exceeded without completing the task.
    MaxIterations {
        /// The iteration limit that was exceeded.
        limit: usize,
    },

    /// A hook rejected the operation.
    HookRejected {
        /// Name of the hook that rejected.
        hook: &'static str,
        /// Reason for rejection.
        reason: String,
    },

    /// Tool not found.
    ToolNotFound {
        /// Name of the missing tool.
        name: String,
    },

    /// Configuration error.
    Config(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm(e) => write!(f, "LLM error: {e}"),
            Self::ToolExecution { name, error } => {
                write!(f, "tool '{name}' failed: {error}")
            }
            Self::MaxIterations { limit } => {
                write!(f, "exceeded maximum iterations ({limit})")
            }
            Self::HookRejected { hook, reason } => {
                write!(f, "hook '{hook}' rejected: {reason}")
            }
            Self::ToolNotFound { name } => {
                write!(f, "tool '{name}' not found")
            }
            Self::Config(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for AgentError {}

impl AgentError {
    /// Returns `true` if this is a provider-level failure worth retrying
    /// (e.g. with a different provider in a multi-provider setup).
    ///
    /// Only `Llm` errors are considered retryable — they indicate the
    /// LLM provider itself failed (rate limit, timeout, auth, server error, etc.).
    /// Other variants (`MaxIterations`, `HookRejected`, `ToolNotFound`, etc.)
    /// are not provider-related and retrying would not help.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Llm(_))
    }
}

impl From<anyhow::Error> for AgentError {
    fn from(error: anyhow::Error) -> Self {
        Self::Llm(error.to_string())
    }
}

impl From<String> for AgentError {
    fn from(error: String) -> Self {
        Self::Llm(error)
    }
}
