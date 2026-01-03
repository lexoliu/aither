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

    /// Tool search configuration.
    pub tool_search: ToolSearchConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            context: ContextStrategy::default(),
            system_prompt: None,
            tool_search: ToolSearchConfig::default(),
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

    /// Sets the tool search configuration.
    #[must_use]
    pub fn with_tool_search(mut self, config: ToolSearchConfig) -> Self {
        self.tool_search = config;
        self
    }
}

/// Configuration for tool searching behavior.
#[derive(Debug, Clone)]
pub struct ToolSearchConfig {
    /// Auto-enable search when tool count exceeds this threshold.
    /// Set to `None` to disable auto-enable.
    pub auto_threshold: Option<usize>,

    /// Explicit override for search behavior.
    /// - `None`: Use auto-detection based on threshold
    /// - `Some(true)`: Force enable search
    /// - `Some(false)`: Force disable search
    pub enabled: Option<bool>,

    /// Maximum number of tools to load per search query.
    pub top_k: usize,

    /// Search strategy to use.
    pub strategy: SearchStrategy,
}

impl Default for ToolSearchConfig {
    fn default() -> Self {
        Self {
            auto_threshold: Some(10),
            enabled: None,
            top_k: 5,
            strategy: SearchStrategy::Bm25,
        }
    }
}

impl ToolSearchConfig {
    /// Creates a new configuration with tool search disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: Some(false),
            ..Self::default()
        }
    }

    /// Creates a new configuration with tool search always enabled.
    #[must_use]
    pub fn always_enabled() -> Self {
        Self {
            enabled: Some(true),
            ..Self::default()
        }
    }

    /// Determines if tool search should be enabled given the tool count.
    #[must_use]
    pub fn should_enable(&self, tool_count: usize) -> bool {
        match self.enabled {
            Some(explicitly_enabled) => explicitly_enabled,
            None => self
                .auto_threshold
                .is_some_and(|threshold| tool_count > threshold),
        }
    }

    /// Sets the auto-enable threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: usize) -> Self {
        self.auto_threshold = Some(threshold);
        self
    }

    /// Sets the top-k results to return.
    #[must_use]
    pub const fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Sets the search strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }
}

/// Strategy for searching deferred tools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchStrategy {
    /// BM25 keyword ranking (default).
    #[default]
    Bm25,

    /// Regex pattern matching.
    Regex,
}
