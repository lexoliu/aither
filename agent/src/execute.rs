use crate::AgentState;

pub trait Executor: Send + Sync {
    fn execute(&self, step: &str, state: &AgentState) -> String;
}

pub struct DefaultExecutor;

impl Executor for DefaultExecutor {
    fn execute(&self, step: &str, _state: &AgentState) -> String {
        format!("Executed step: {}", step)
    }
}
