//! Agent configuration.

use crate::compression::ContextStrategy;

/// Agent specialization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentKind {
    /// Coding-focused agent (loads workspace facts like AGENT.md/CLAUDE.md).
    #[default]
    Coding,
    /// Generic chat assistant (no coding workspace-fact loading).
    Chatbot,
}

/// Priority for custom context blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextBlockPriority {
    /// Must be retained first and appears before all other blocks.
    Critical,
    /// High-value block that should be preserved ahead of normal context.
    High,
    /// Default priority for regular contextual information.
    Normal,
    /// Optional/background context that can be dropped first.
    Low,
}

impl ContextBlockPriority {
    /// Stable numeric rank used for deterministic ordering.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::High => 1,
            Self::Normal => 2,
            Self::Low => 3,
        }
    }
}

/// A structured context block rendered into XML in the system prompt.
#[derive(Debug, Clone)]
pub struct ContextBlock {
    /// XML tag name, e.g. `workspace` or `repo_state`.
    pub tag: String,
    /// Block content.
    pub content: String,
    /// Relative priority used for deterministic ordering.
    pub priority: ContextBlockPriority,
}

impl ContextBlock {
    /// Creates a context block with `Normal` priority.
    #[must_use]
    pub fn new(tag: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            content: content.into(),
            priority: ContextBlockPriority::Normal,
        }
    }

    /// Sets block priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: ContextBlockPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Prompt-level context assembly settings.
#[derive(Debug, Clone)]
pub struct ContextAssemblerConfig {
    /// Fraction of context reserved for static/system blocks.
    pub static_budget_fraction: f32,
    /// Usage threshold to request handoff summary.
    pub handoff_threshold: f32,
    /// Instruction injected near context exhaustion.
    pub handoff_instruction: String,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self {
            static_budget_fraction: 0.2,
            handoff_threshold: 0.9,
            handoff_instruction: "Your context window is nearly exhausted. Generate a concise handoff summary now, preserving current goals, constraints, file paths, pending tasks, and immediate next actions.".to_string(),
        }
    }
}

/// Configuration for agent behavior.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of agent loop iterations.
    pub max_iterations: usize,

    /// Context management strategy.
    pub context: ContextStrategy,

    /// Generic system prompt content.
    pub system_prompt: Option<String>,

    /// Optional persona overlay.
    pub persona_prompt: Option<String>,

    /// Agent specialization (coding vs chatbot).
    pub agent_kind: AgentKind,

    /// Optional transcript path for long-memory recovery.
    pub transcript_path: Option<String>,

    /// Additional structured context blocks.
    pub context_blocks: Vec<ContextBlock>,

    /// Context assembly behavior.
    pub context_assembler: ContextAssemblerConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            // Very high default - should effectively never hit this limit
            // Individual use cases can set lower limits if needed
            max_iterations: 10_000,
            context: ContextStrategy::default(),
            system_prompt: None,
            persona_prompt: None,
            agent_kind: AgentKind::default(),
            transcript_path: None,
            context_blocks: Vec::new(),
            context_assembler: ContextAssemblerConfig::default(),
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
    pub const fn with_context(mut self, strategy: ContextStrategy) -> Self {
        self.context = strategy;
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the persona prompt.
    #[must_use]
    pub fn with_persona_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.persona_prompt = Some(prompt.into());
        self
    }

    /// Sets the agent kind.
    #[must_use]
    pub const fn with_agent_kind(mut self, kind: AgentKind) -> Self {
        self.agent_kind = kind;
        self
    }

    /// Sets a transcript path for memory recovery.
    #[must_use]
    pub fn with_transcript_path(mut self, path: impl Into<String>) -> Self {
        self.transcript_path = Some(path.into());
        self
    }

    /// Adds a custom structured context block.
    #[must_use]
    pub fn with_context_block(mut self, block: ContextBlock) -> Self {
        self.context_blocks.push(block);
        self
    }
}
