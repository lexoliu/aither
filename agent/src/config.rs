//! Agent configuration.

use std::time::Duration;

use aither_core::llm::model::Parameters;

use crate::context::ContextPolicy;

/// Agent specialization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentKind {
    /// Coding-focused agent (loads workspace facts like AGENT.md/CLAUDE.md).
    #[default]
    Coding,
    /// Generic chat assistant (no coding workspace-fact loading).
    Chatbot,
}

/// Configuration for agent behavior.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of agent loop iterations.
    pub max_iterations: usize,

    /// Thresholds governing compaction, pausing and reassembly.
    pub context_policy: ContextPolicy,

    /// Generic system prompt content.
    pub system_prompt: Option<String>,

    /// Optional persona overlay.
    pub persona_prompt: Option<String>,

    /// Agent specialization (coding vs chatbot).
    pub agent_kind: AgentKind,

    /// Tool turns tolerated before an unmaintained `tasks.md` earns a
    /// staleness alert.
    ///
    /// A turn counts toward the limit when it ran tool calls but left
    /// `tasks.md` byte-identical — either still holding unchecked items or
    /// missing entirely. Set to `usize::MAX` to disable the alert.
    pub tasks_stale_after_turns: usize,

    /// Request-level model parameters applied to each LLM call.
    ///
    /// This is the only path for request parameters: the agent fills in the
    /// prompt cache controls it derives from the window layout and sends
    /// everything else exactly as configured here.
    pub request_parameters: Parameters,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            context_policy: ContextPolicy::default(),
            system_prompt: None,
            persona_prompt: None,
            agent_kind: AgentKind::default(),
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

    /// Sets the context management policy.
    #[must_use]
    pub const fn with_context_policy(mut self, policy: ContextPolicy) -> Self {
        self.context_policy = policy;
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

    /// Sets the idle gap after which the window is reassembled.
    #[must_use]
    pub const fn with_idle_reassemble_after(mut self, idle_reassemble_after: Duration) -> Self {
        self.context_policy.idle_reassemble_after = idle_reassemble_after;
        self
    }

    /// Sets default request-level model parameters for each LLM call.
    #[must_use]
    pub fn with_request_parameters(mut self, parameters: Parameters) -> Self {
        self.request_parameters = parameters;
        self
    }
}
