//! Agent configuration.

use std::time::Duration;

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

/// Prompt-cache keep-alive settings for an agent that waits on its own
/// background commands.
///
/// Enabled on subagent runs by `SubagentTool`: while the run parks waiting
/// for background terminal jobs, the conversation's prompt cache entry is
/// refreshed by re-sending the last request with `max_tokens: 0`. A `None`
/// value disables keep-alive entirely (the default for ordinary agents,
/// including the parent agent itself).
///
/// How long an entry stays warm is the agent's
/// [`PromptCache`](crate::context::PromptCache) lifetime — a
/// [`CacheLifetime::Permanent`](crate::context::CacheLifetime) cache gets no
/// keep-alives.
#[derive(Debug, Clone)]
pub struct CacheKeepAlive {
    /// How long before the cache entry's lifetime expires the keep-alive is
    /// sent.
    pub send_before_expiry: Duration,
    /// Number of keep-alives in one run that counts as idling frequently;
    /// crossing it produces a single parent alert.
    pub frequent_idle_threshold: usize,
}

impl Default for CacheKeepAlive {
    fn default() -> Self {
        Self {
            send_before_expiry: Duration::from_secs(30),
            frequent_idle_threshold: 3,
        }
    }
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

    /// Prompt-cache keep-alive while the run waits on its own background
    /// commands. `None` disables keep-alive.
    pub cache_keep_alive: Option<CacheKeepAlive>,

    /// What a run does with background commands still running when the
    /// model ends its turn.
    pub background_jobs: BackgroundJobPolicy,

    /// The communication and working languages, enforced on questions,
    /// replies and records. `None` enforces nothing.
    #[cfg(feature = "languages")]
    pub languages: Option<crate::LanguageGuard>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            context_policy: ContextPolicy::default(),
            system_prompt: None,
            persona_prompt: None,
            agent_kind: AgentKind::default(),
            cache_keep_alive: None,
            background_jobs: BackgroundJobPolicy::default(),
            #[cfg(feature = "languages")]
            languages: None,
        }
    }
}

/// What a run does with background commands still running when the model
/// ends its turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundJobPolicy {
    /// The run waits until they finish, waking the model for each completion
    /// and for a command blocked on stdin. Each command's own `timeout`
    /// bounds the wait. For an agent nothing resumes once its run ends, such
    /// as a subagent.
    #[default]
    Await,
    /// The run ends with the model's turn. Completions and alerts that
    /// arrive afterwards are taken by [`Agent::wait_background`], which
    /// tells the host whether the model has something to act on; the host
    /// then continues the conversation with [`Agent::resume`].
    ///
    /// [`Agent::wait_background`]: crate::Agent::wait_background
    /// [`Agent::resume`]: crate::Agent::resume
    Detach,
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

    /// Enforces the languages `guard` holds.
    #[cfg(feature = "languages")]
    #[must_use]
    pub fn with_languages(mut self, guard: crate::LanguageGuard) -> Self {
        self.languages = Some(guard);
        self
    }

    /// Enables prompt-cache keep-alive while the run waits on its own
    /// background commands.
    #[must_use]
    pub const fn with_cache_keep_alive(mut self, keep_alive: CacheKeepAlive) -> Self {
        self.cache_keep_alive = Some(keep_alive);
        self
    }
}
