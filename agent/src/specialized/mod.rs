//! Specialized subagent implementations.
//!
//! These are subagent types that can be spawned by the main agent
//! for specific tasks like exploration or deep research.

mod external;
mod subagent;
pub mod task;

pub use external::{ExternalSession, ExternalStart, ExternalSubagent};
pub use subagent::{IntoSubAgent, SubAgentQuery, SubAgentTool};
pub use task::{
    FORK_DESCRIPTION, SubagentArgs, SubagentBuilder, SubagentFileMount, SubagentRunInfo,
    SubagentRuns, SubagentTool,
};
