use aither_core::{LanguageModel, llm::Tool};

use crate::Agent;

/// A sub-agent that can be used as a tool within a larger agent.
///
/// # What is a Sub-Agent?
/// A Sub-Agent is a specialized agent designed to handle specific tasks or domains
/// within a larger agent framework. It encapsulates its own planning and execution
/// logic, allowing it to operate semi-autonomously while still being callable by
/// the parent agent.
#[derive(Debug)]
pub struct SubAgent<LLM, Planner, Executor> {
    agent: Agent<LLM, Planner, Executor>,
    name: String,
    description: String,
}

/// Implementations for sub-agent
impl<LLM, Planner, Executor> SubAgent<LLM, Planner, Executor> {
    /// Creates a new sub-agent wrapping the given Agent.
    pub fn new(
        agent: Agent<LLM, Planner, Executor>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            agent,
            name: name.into(),
            description: description.into(),
        }
    }
}

impl<LLM, Planner, Executor> Tool for SubAgent<LLM, Planner, Executor>
where
    LLM: LanguageModel,
    Planner: crate::plan::Planner + 'static,
    Executor: crate::execute::Executor + 'static,
{
    fn name(&self) -> std::borrow::Cow<'static, str> {
        self.name.clone().into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        self.description.clone().into()
    }
    type Arguments = String; // Task description
    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        self.agent.run(&arguments).await
    }
}
