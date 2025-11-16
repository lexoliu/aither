//! Opinionated `Agent` wrapper tailored for coding workflows.

use aither_core::LanguageModel;

use crate::{
    Agent, AgentConfig, ToolingConfig,
    execute::{self, Executor},
    memory::ContextStrategy,
    plan::{self, Planner},
};

/// Agent wrapper that mounts filesystem + shell tooling suitable for code editing tasks.
#[derive(Debug)]
pub struct Coder<LLM, PlannerImpl = plan::DefaultPlanner, ExecutorImpl = execute::DefaultExecutor> {
    agent: Agent<LLM, PlannerImpl, ExecutorImpl>,
}

impl<LLM> Coder<LLM, plan::DefaultPlanner, execute::DefaultExecutor>
where
    LLM: LanguageModel,
{
    /// Creates a coder agent with the default planner/executor pair.
    #[must_use]
    pub fn new(llm: LLM) -> Self {
        Self::with_components(llm, plan::DefaultPlanner, execute::DefaultExecutor)
    }
}

impl<LLM, PlannerImpl, ExecutorImpl> Coder<LLM, PlannerImpl, ExecutorImpl>
where
    LLM: LanguageModel,
    PlannerImpl: Planner,
    ExecutorImpl: Executor,
{
    /// Builds a coder agent with custom planner and executor implementations.
    #[must_use]
    pub fn with_components(llm: LLM, planner: PlannerImpl, executor: ExecutorImpl) -> Self {
        Self::with_config(llm, planner, executor, coder_config())
    }

    /// Builds a coder agent with a custom configuration.
    #[must_use]
    pub fn with_config(
        llm: LLM,
        planner: PlannerImpl,
        executor: ExecutorImpl,
        config: AgentConfig,
    ) -> Self {
        Self {
            agent: Agent::custom(llm, planner, executor, config),
        }
    }

    /// Returns the default configuration used by [`Coder::new`].
    #[must_use]
    pub fn default_config() -> AgentConfig {
        coder_config()
    }

    /// Executes a goal using the underlying agent.
    ///
    /// # Errors
    ///
    /// Propagates any planner or executor failures encountered by [`Agent::run`].
    pub async fn run(&mut self, goal: &str) -> aither_core::Result<String> {
        self.agent.run(goal).await
    }

    /// Accesses the inner agent.
    #[must_use]
    pub const fn agent(&self) -> &Agent<LLM, PlannerImpl, ExecutorImpl> {
        &self.agent
    }

    /// Mutably accesses the inner agent.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn agent_mut(&mut self) -> &mut Agent<LLM, PlannerImpl, ExecutorImpl> {
        &mut self.agent
    }

    /// Consumes the wrapper and returns the inner agent.
    pub fn into_inner(self) -> Agent<LLM, PlannerImpl, ExecutorImpl> {
        self.agent
    }
}

fn coder_config() -> AgentConfig {
    AgentConfig::new(
        ContextStrategy::Summarize {
            max_messages: 48,
            retain_recent: 12,
            instructions: "Keep file paths, commands, and compiler diagnostics verbatim.".into(),
        },
        64,
        ToolingConfig::coder(),
    )
}
