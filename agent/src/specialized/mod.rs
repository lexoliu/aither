//! Specialized subagent implementations.
//!
//! These are subagent types that can be spawned by the main agent
//! for specific tasks like exploration or deep research.

mod external;
mod router;
mod subagent;
pub mod task;

pub use external::{ExternalSession, ExternalStart, ExternalSubagent};
pub use router::{
    AnySubagentRouter, JudgmentRouter, RouteRequest, SubagentRouteError, SubagentRouter, TierOffer,
};
pub use subagent::{IntoSubAgent, SubAgentQuery, SubAgentTool};
pub(crate) use task::default_fork_refusal;
pub use task::{
    FORK_DESCRIPTION, ForkRefusal, SubagentArgs, SubagentFileMount, SubagentRunInfo, SubagentRuns,
    SubagentSetup, SubagentTool,
};
