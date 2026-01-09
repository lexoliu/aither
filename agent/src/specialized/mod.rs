//! Specialized subagent implementations.
//!
//! These are subagent types that can be spawned by the main agent
//! for specific tasks like exploration or deep research.

mod subagent;
pub mod task;

pub use subagent::{IntoSubAgent, SubAgentQuery, SubAgentTool};
pub use task::{SubagentType, TaskArgs, TaskTool};
