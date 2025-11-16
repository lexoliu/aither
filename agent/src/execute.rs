use aither_core::Result;
use core::future::Future;

use crate::AgentState;

/// Executes concrete plan steps produced by the planner.
pub trait Executor: Send + Sync {
    /// Runs a single step and returns a textual description of the outcome.
    ///
    /// # Errors
    ///
    /// Returns an error if the step execution fails.
    fn execute(
        &self,
        step: &str,
        state: &AgentState,
    ) -> impl Future<Output = Result<String>> + Send;
}

/// Default executor that only simulates execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultExecutor;

impl Executor for DefaultExecutor {
    async fn execute(&self, _step: &str, _state: &AgentState) -> Result<String> {
        todo!()
    }
}
