//! SubAgent - Wrap an agent as a callable tool.
//!
//! Allows composing agents by delegating tasks to specialized sub-agents.

use std::borrow::Cow;

use aither_core::{LanguageModel, llm::Tool};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{Agent, AgentError, hook::Hook};

/// A sub-agent that can be used as a tool by a parent agent.
///
/// This allows building hierarchies of agents, where a main agent
/// can delegate specific tasks to specialized sub-agents.
///
/// # Example
///
/// ```rust,ignore
/// let reviewer = Agent::builder(claude)
///     .system_prompt("You are a code reviewer...")
///     .build();
///
/// let reviewer_tool = SubAgentTool::new(
///     reviewer,
///     "code-reviewer",
///     "Reviews code for security and quality issues"
/// );
///
/// let main_agent = Agent::builder(claude)
///     .tool(reviewer_tool)
///     .build();
///
/// // Main agent can now delegate review tasks
/// main_agent.query("Write and review a new function").await?;
/// ```
pub struct SubAgentTool<LLM, H = ()> {
    agent: Agent<LLM, H>,
    name: String,
    description: String,
}

impl<LLM, H> SubAgentTool<LLM, H>
where
    LLM: LanguageModel,
    H: Hook,
{
    /// Creates a new sub-agent tool.
    pub fn new(
        agent: Agent<LLM, H>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            agent,
            name: name.into(),
            description: description.into(),
        }
    }

    /// Returns the sub-agent's name.
    #[must_use]
    pub fn agent_name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the underlying agent.
    #[must_use]
    pub const fn agent(&self) -> &Agent<LLM, H> {
        &self.agent
    }

    /// Returns a mutable reference to the underlying agent.
    pub fn agent_mut(&mut self) -> &mut Agent<LLM, H> {
        &mut self.agent
    }
}

/// Arguments for calling a sub-agent.
#[derive(Debug, Clone, JsonSchema, Deserialize)]
pub struct SubAgentQuery {
    /// The task to delegate to this sub-agent.
    /// Be specific about what you want the sub-agent to do.
    pub task: String,
}

impl<LLM, H> Tool for SubAgentTool<LLM, H>
where
    LLM: LanguageModel + 'static,
    H: Hook + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = SubAgentQuery;

    async fn call(&mut self, args: Self::Arguments) -> aither_core::Result {
        self.agent
            .query(&args.task)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Extension trait for converting an agent into a sub-agent tool.
pub trait IntoSubAgent {
    /// The language model type.
    type LLM: LanguageModel;
    /// The hook type.
    type Hooks: Hook;

    /// Converts this agent into a sub-agent tool.
    fn into_subagent(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> SubAgentTool<Self::LLM, Self::Hooks>;
}

impl<LLM, H> IntoSubAgent for Agent<LLM, H>
where
    LLM: LanguageModel,
    H: Hook,
{
    type LLM = LLM;
    type Hooks = H;

    fn into_subagent(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> SubAgentTool<LLM, H> {
        SubAgentTool::new(self, name, description)
    }
}
