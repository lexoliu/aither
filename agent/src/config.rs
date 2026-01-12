//! Agent configuration.

use crate::compression::ContextStrategy;

/// Configuration for agent behavior.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of agent loop iterations.
    pub max_iterations: usize,

    /// Context management strategy.
    pub context: ContextStrategy,

    /// System prompt to prepend to conversations.
    pub system_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            // Very high default - should effectively never hit this limit
            // Individual use cases can set lower limits if needed
            max_iterations: 10_000,
            context: ContextStrategy::default(),
            system_prompt: None,
        }
    }
}

impl AgentConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, limit: usize) -> Self {
        self.max_iterations = limit;
        self
    }

    /// Sets the context strategy.
    #[must_use]
    pub fn with_context(mut self, strategy: ContextStrategy) -> Self {
        self.context = strategy;
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}
