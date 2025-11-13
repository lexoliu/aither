use aither_core::{
    LanguageModel, Result,
    llm::{Message, model::Parameters},
};
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentState, todo::TodoList};

/// Planner trait for generating plans to achieve goals.
pub trait Planner: Send + Sync {
    /// Plans the next steps to achieve the given goal.
    fn plan(
        &self,
        llm: impl LanguageModel,
        state: &mut AgentState,
        goal: &str,
    ) -> impl Future<Output = Result<PlanOutcome>> + Send;
}

pub struct DefaultPlanner;

impl Planner for DefaultPlanner {
    async fn plan(
        &self,
        llm: impl LanguageModel,
        state: &mut AgentState,
        goal: &str,
    ) -> Result<PlanOutcome> {
        // Build a planning prompt that incorporates the goal and current context
        let system_prompt = "You are a strategic planner. Break down the given goal into concrete, actionable steps. Or ";

        let mut messages = vec![Message::system(system_prompt), Message::user(goal)];

        // Include relevant history if available
        if !state.history.is_empty() {
            messages.insert(
                1,
                Message::user(format!("Previous context: {:?}", state.history.last())),
            );
        }

        // Request a plan from the LLM
        let tasks: Vec<String> = llm
            .generate(&messages, &mut state.tools, &Parameters::default())
            .await
            .context("Fail to create task todo list, try again please.")?;

        let tasks = TodoList::new(tasks);

        Ok(PlanOutcome::NeedsMoreSteps(tasks))
    }
}

/// Outcome of a planning operation.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub enum PlanOutcome {
    /// The plan has been completed with the given result.
    Completed(String),
    /// More steps are needed, represented by a TodoList.
    NeedsMoreSteps(TodoList),
}
