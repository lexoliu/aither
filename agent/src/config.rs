//! Agent configuration.

use std::time::Duration;

use aither_core::llm::model::Parameters;

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

/// Prompt-level context assembly settings.
#[derive(Debug, Clone)]
pub struct ContextAssemblerConfig {
    /// Fraction of context reserved for static/system blocks.
    pub static_budget_fraction: f32,
    /// Usage threshold to request handoff summary.
    pub handoff_threshold: f32,
    /// Reassemble ephemeral context when no LLM request has been sent for this long.
    pub idle_reassemble_after: Duration,
    /// Instruction injected near context exhaustion.
    pub handoff_instruction: String,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self {
            static_budget_fraction: 0.2,
            handoff_threshold: 0.9,
            idle_reassemble_after: Duration::from_secs(300),
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

    /// Context assembly behavior.
    pub context_assembler: ContextAssemblerConfig,

    /// Tool turns tolerated before an unmaintained `tasks.md` earns a
    /// staleness alert.
    ///
    /// A turn counts toward the limit when it ran tool calls but left
    /// `tasks.md` byte-identical — either still holding unchecked items or
    /// missing entirely. Set to `usize::MAX` to disable the alert.
    pub tasks_stale_after_turns: usize,

    /// Request-level model parameters applied to each LLM call.
    pub request_parameters: Parameters,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            context: ContextStrategy::default(),
            system_prompt: None,
            persona_prompt: None,
            agent_kind: AgentKind::default(),
            transcript_path: None,
            context_assembler: ContextAssemblerConfig::default(),
            tasks_stale_after_turns: 4,
            request_parameters: Parameters::default(),
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

    /// Sets the idle gap after which ephemeral context should be reassembled.
    #[must_use]
    pub const fn with_idle_reassemble_after(mut self, idle_reassemble_after: Duration) -> Self {
        self.context_assembler.idle_reassemble_after = idle_reassemble_after;
        self
    }

    /// Sets default request-level model parameters for each LLM call.
    #[must_use]
    pub fn with_request_parameters(mut self, parameters: Parameters) -> Self {
        self.request_parameters = parameters;
        self
    }
}
