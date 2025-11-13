//! Agent framework for building autonomous AI agents with planning and execution capabilities.

use aither_core::{
    LanguageModel,
    llm::{Message, tool::Tools},
};

pub mod execute;
pub mod memory;
pub mod plan;
pub mod sub_agent;
pub mod todo;

use crate::{sub_agent::SubAgent, todo::TodoList};

#[derive(Debug)]
pub struct Agent<LLM, Planner, Executor> {
    llm: LLM,
    planner: Planner,
    executor: Executor,
    state: AgentState,
}

#[derive(Debug, Default)]
pub struct AgentState {
    tools: Tools,
    todo: Option<TodoList>,
    history: Vec<Message>,
}

impl<LLM> Agent<LLM, plan::DefaultPlanner, execute::DefaultExecutor>
where
    LLM: LanguageModel,
{
    /// Creates a new Agent with default planner and executor.
    pub fn new(llm: LLM) -> Self {
        Self::custom(llm, plan::DefaultPlanner, execute::DefaultExecutor)
    }
}

impl<LLM, Planner, Executor> Agent<LLM, Planner, Executor>
where
    LLM: LanguageModel,
    Planner: plan::Planner,
    Executor: execute::Executor,
{
    /// Creates a new Agent with the specified components.
    pub fn custom(llm: LLM, planner: Planner, executor: Executor) -> Self {
        Self {
            llm,
            planner,
            executor,
            state: AgentState::default(),
        }
    }

    /// Runs the agent to achieve the specified goal.
    ///
    /// # Parameters
    /// - `goal`: The goal the agent should work towards.
    ///
    /// # Errors
    /// Returns an error if the agent fails to achieve the goal.
    pub async fn run(&mut self, goal: &str) -> aither_core::Result<String> {
        loop {
            let plan_outcome = self.planner.plan(&self.llm, &mut self.state, goal).await?;

            match plan_outcome {
                plan::PlanOutcome::Completed(result) => return Ok(result),
                plan::PlanOutcome::NeedsMoreSteps(mut tasks) => {
                    todo!()
                }
            }
        }
    }

    /// Converts the agent into a sub-agent that can be used as a tool.
    ///
    /// Learn more about sub-agent in the [`SubAgent`] struct documentation.
    ///
    /// # Parameters
    /// - `name`: The name of the sub-agent tool.
    /// - `description`: A description of the sub-agent tool.
    pub fn into_subagent(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> SubAgent<LLM, Planner, Executor> {
        SubAgent::new(self, name, description)
    }
}
