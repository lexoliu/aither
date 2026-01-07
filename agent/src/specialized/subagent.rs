//! SubAgent - Wrap an agent configuration as a callable tool.
//!
//! Allows composing agents by delegating tasks to specialized sub-agents.
//! Each call spawns a fresh agent instance.

use std::borrow::Cow;

use aither_core::{LanguageModel, llm::Tool};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::Agent;

/// A sub-agent tool that spawns a fresh agent for each call.
///
/// This allows building hierarchies of agents, where a main agent
/// can delegate specific tasks to specialized sub-agents.
///
/// # Example
///
/// ```rust,ignore
/// let reviewer_tool = SubAgentTool::new(claude.clone())
///     .name("code-reviewer")
///     .description("Reviews code for security and quality issues")
///     .system_prompt("You are a code reviewer...");
///
/// let main_agent = Agent::builder(claude)
///     .tool(reviewer_tool)
///     .build();
///
/// // Main agent can now delegate review tasks
/// main_agent.query("Write and review a new function").await?;
/// ```
pub struct SubAgentTool<LLM> {
    llm: LLM,
    name: String,
    description: String,
    system_prompt: Option<String>,
}

impl<LLM: Clone> SubAgentTool<LLM> {
    /// Creates a new sub-agent tool with the given LLM.
    pub fn new(llm: LLM) -> Self {
        Self {
            llm,
            name: "subagent".to_string(),
            description: "A sub-agent that can handle delegated tasks".to_string(),
            system_prompt: None,
        }
    }

    /// Sets the tool name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the tool description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the system prompt for the sub-agent.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

/// Arguments for calling a sub-agent.
#[derive(Debug, Clone, JsonSchema, Deserialize)]
pub struct SubAgentQuery {
    /// The task to delegate to this sub-agent.
    /// Be specific about what you want the sub-agent to do.
    pub task: String,
}

impl<LLM> Tool for SubAgentTool<LLM>
where
    LLM: LanguageModel + Clone + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = SubAgentQuery;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result {
        // Create a fresh agent for this call
        let mut builder = Agent::builder(self.llm.clone());

        if let Some(ref prompt) = self.system_prompt {
            builder = builder.system_prompt(prompt);
        }

        let mut agent = builder.build();

        agent
            .query(&args.task)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Extension trait for creating sub-agent tools from an LLM.
pub trait IntoSubAgent: LanguageModel + Clone + Sized {
    /// Creates a sub-agent tool from this LLM.
    fn into_subagent(self) -> SubAgentTool<Self> {
        SubAgentTool::new(self)
    }
}

impl<LLM: LanguageModel + Clone> IntoSubAgent for LLM {}
