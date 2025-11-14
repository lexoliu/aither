use core::future::Future;

use aither_core::{
    LanguageModel, Result,
    llm::{LLMRequest, Message},
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

/// Default planner implementation that uses the LLM to generate plans.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultPlanner;

impl Planner for DefaultPlanner {
    async fn plan(
        &self,
        llm: impl LanguageModel,
        state: &mut AgentState,
        goal: &str,
    ) -> Result<PlanOutcome> {
        let mut system_prompt = String::from(
            "You are an expert planner. Always reason first, then return JSON that matches the target schema.",
        );
        if let Some(summary) = state.capability_summary() {
            system_prompt.push_str("\n\nAvailable capabilities:\n");
            system_prompt.push_str(&summary);
        }
        let mut messages = vec![Message::system(system_prompt)];
        messages.extend(state.messages());
        messages.push(Message::system(format!(
            "Plan steps to achieve: {goal}. If the goal is already complete, mark status as Completed."
        )));

        let response: PlanResponse = llm
            .generate(LLMRequest::new(messages).with_tools(&mut state.tools))
            .await
            .context("failed to build task list")?;

        match response.status {
            PlanStatus::Completed { result } => Ok(PlanOutcome::Completed(result)),
            PlanStatus::Steps { tasks } => Ok(PlanOutcome::NeedsMoreSteps(TodoList::new(tasks))),
        }
    }
}

/// Outcome of a planning operation.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub enum PlanOutcome {
    /// The plan has been completed with the given result.
    Completed(String),
    /// More steps are needed, represented by a `TodoList`.
    NeedsMoreSteps(TodoList),
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct PlanResponse {
    status: PlanStatus,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum PlanStatus {
    Completed { result: String },
    Steps { tasks: Vec<String> },
}
