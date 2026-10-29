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
/// [`AgentBuilder::fork_source`](crate::AgentBuilder::fork_source); the agent
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
