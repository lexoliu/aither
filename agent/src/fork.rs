//! What a fork subagent inherits from the main agent.
//!
//! A main agent publishes the request it is about to send to a
//! [`ForkSource`]; the `fork` subagent kind reads the latest one and starts
//! from it, so its first request is the main agent's request followed by the
//! fork prompt and reads the main agent's warm prompt cache.

use std::sync::{Arc, Mutex};

use aither_core::llm::model::{Parameters, Profile};
use aither_core::llm::tool::ToolDefinition;

use crate::context::{Context, ContextBudget, ContextPolicy, PromptCache};
use crate::model_group::ModelTier;

/// The main agent's request, as the fork subagent starts from it.
///
/// Everything a request is made of is kept as the main agent assembled it:
/// the window (system blocks, handoff, conversation and live tail, thinking
/// blocks included), the tool definitions, the request parameters and the
/// prompt-cache identity.
#[derive(Debug, Clone)]
pub struct ForkSnapshot {
    /// The tier of the model the main agent ran the request on.
    pub(crate) tier: ModelTier,
    /// The window the request was built from, with its live tail as the
    /// last message.
    pub(crate) context: Context,
    pub(crate) prompt_cache: PromptCache,
    pub(crate) budget: ContextBudget,
    /// The main agent's model, whose name identifies it.
    pub(crate) profile: Profile,
    pub(crate) request_parameters: Parameters,
    /// The request-level effort the main agent sends, so the fork's first
    /// request carries the same and reads the main agent's cache.
    pub(crate) effort: crate::effort::EffortAnchor,
    pub(crate) context_policy: ContextPolicy,
    pub(crate) definitions: Vec<ToolDefinition>,
    /// Window usage the request was sent at.
    pub(crate) usage: f32,
}

impl ForkSnapshot {
    /// The tier of the model the main agent runs on; a fork runs on it too.
    #[must_use]
    pub const fn tier(&self) -> ModelTier {
        self.tier
    }

    /// The name of the model the main agent ran the request on.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.profile.name
    }

    /// The fraction of the context window the main agent's request used.
    #[must_use]
    pub const fn usage(&self) -> f32 {
        self.usage
    }

    /// The fraction of the context window the main agent compacts at.
    #[must_use]
    pub fn compaction_usage(&self) -> f32 {
        let window = f64::from(self.budget.window());
        if window == 0.0 {
            return 0.0;
        }
        let point = f64::from(
            u32::try_from(self.context_policy.compact_point(self.budget.window()))
                .unwrap_or(u32::MAX),
        );
        num_traits::ToPrimitive::to_f32(&(point / window)).unwrap_or(1.0)
    }

    /// Whether a fork may start from the request the main agent sent: it is
    /// refused within a tenth of the main agent's auto-compact point.
    #[must_use]
    pub fn allows_fork(&self) -> bool {
        self.context_policy
            .allows_fork(self.budget.estimated_tokens(), self.budget.window())
    }

    /// The main agent's context policy.
    #[must_use]
    pub const fn policy(&self) -> &ContextPolicy {
        &self.context_policy
    }
}

/// The latest request of a main agent, shared with the tool that forks it.
///
/// Create one with [`SubagentTool::fork_source`](crate::specialized::SubagentTool::fork_source)
/// and hand it to the main agent's builder with
/// [`AgentOptions::fork_source`](crate::AgentOptions::fork_source); the agent
/// publishes every request it sends, the tool reads the newest.
#[derive(Debug, Clone, Default)]
pub struct ForkSource {
    latest: Arc<Mutex<Option<Arc<ForkSnapshot>>>>,
}

impl ForkSource {
    /// Replaces the published request.
    pub(crate) fn publish(&self, snapshot: ForkSnapshot) {
        *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(snapshot));
    }

    /// The request the main agent sent last; `None` before its first.
    #[must_use]
    pub fn latest(&self) -> Option<Arc<ForkSnapshot>> {
        self.latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}
