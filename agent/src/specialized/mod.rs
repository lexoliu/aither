//! Specialized agent implementations.
//!
//! These are pre-configured agents for common use cases.

mod coder;
mod subagent;

pub use coder::{Coder, CoderBuilder};
pub use subagent::{SubAgentQuery, SubAgentTool};
