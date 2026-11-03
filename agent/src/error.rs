//! Agent error types.

use core::fmt;

use aither_core::llm::{Classify, Failure};

/// Errors that can occur during agent execution.
#[derive(Debug, Clone)]
pub enum AgentError {
    /// LLM returned an error.
    Llm {
        /// The provider's error message.
        message: String,
        /// How the provider error classifies: network loss, transient fault,
        /// quota exhaustion or permanent failure, and the requested retry
        /// delay.
        failure: Failure,
    },

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
            Self::Llm { message, .. } => write!(f, "LLM error: {message}"),
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
    /// Captures a provider error with its classification.
    pub fn from_model_error<E: std::error::Error + Classify>(error: &E) -> Self {
        Self::Llm {
            message: error.to_string(),
            failure: error.failure(),
        }
    }

    /// An LLM-side error that originates in this process, never from a
    /// provider, and so is permanent.
    pub fn llm_local(message: impl Into<String>) -> Self {
        Self::Llm {
            message: message.into(),
            failure: Failure::permanent(),
        }
    }

    /// How the provider failure classifies, or `None` when this error is not
    /// a provider failure.
    ///
    /// Retry policy belongs to the caller: this says whether a retry can help
    /// ([`Failure::is_retryable`]) and what delay the upstream asked for.
    #[must_use]
    pub const fn failure(&self) -> Option<&Failure> {
        match self {
            Self::Llm { failure, .. } => Some(failure),
            _ => None,
        }
    }
}

impl From<anyhow::Error> for AgentError {
    fn from(error: anyhow::Error) -> Self {
        Self::llm_local(error.to_string())
    }
}

impl From<String> for AgentError {
    fn from(error: String) -> Self {
        Self::llm_local(error)
    }
}

impl From<crate::context::CompactionError> for AgentError {
    fn from(error: crate::context::CompactionError) -> Self {
        use crate::context::CompactionError;
        let failure = match &error {
            CompactionError::Provider { failure, .. } => *failure,
            CompactionError::Invalid { .. } | CompactionError::Empty => Failure::permanent(),
        };
        Self::Llm {
            message: error.to_string(),
            failure,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aither_core::llm::{Classify, Failure, FailureKind};

    use super::AgentError;

    #[derive(Debug)]
    struct Overloaded;

    impl std::fmt::Display for Overloaded {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("overloaded")
        }
    }

    impl std::error::Error for Overloaded {}

    impl Classify for Overloaded {
        fn failure(&self) -> Failure {
            Failure::transient().with_retry_after(Some(Duration::from_secs(5)))
        }
    }

    #[test]
    fn provider_failure_is_carried() {
        let error = AgentError::from_model_error(&Overloaded);
        assert_eq!(error.to_string(), "LLM error: overloaded");
        let failure = error.failure().expect("an LLM error carries a failure");
        assert_eq!(failure.kind, FailureKind::Transient);
        assert_eq!(failure.retry_after, Some(Duration::from_secs(5)));
    }

    #[test]
    fn non_llm_errors_have_no_failure() {
        assert!(AgentError::Config("x".into()).failure().is_none());
        assert_eq!(
            AgentError::llm_local("x").failure().map(|f| f.kind),
            Some(FailureKind::Permanent)
        );
    }
}
