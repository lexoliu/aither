//! Who made a tool call: the call's id and the agent behind it.
//!
//! The agent loop attaches a [`CallIdentity`] to the [`ToolContext`](super::ToolContext)
//! of every call it dispatches, so a tool, and any command the tool runs on its
//! behalf, can attribute what it did to the exact call and agent.

use alloc::boxed::Box;
use alloc::string::String;
use core::fmt::{self, Display};

/// The id a model provider gave one tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolCallId(String);

impl ToolCallId {
    /// Wraps a provider-assigned tool call id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as the provider spelled it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A subagent: its type and the call that started it.
///
/// The starting call's identity makes the subagent unique across nested
/// subagents and ties it to the call that launched it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubagentIdentity {
    name: String,
    started_by: Option<Box<CallIdentity>>,
}

impl SubagentIdentity {
    /// A subagent of type `name` started by `started_by`; `None` when the
    /// dispatcher of the starting call attached no identity.
    #[must_use]
    pub fn new(name: impl Into<String>, started_by: Option<CallIdentity>) -> Self {
        Self {
            name: name.into(),
            started_by: started_by.map(Box::new),
        }
    }

    /// The subagent type it runs as.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The call that started this subagent, if its dispatcher identified it.
    #[must_use]
    pub fn started_by(&self) -> Option<&CallIdentity> {
        self.started_by.as_deref()
    }
}

/// The agent making a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentIdentity {
    /// The main agent.
    Main,
    /// A subagent, at any nesting depth.
    Subagent(SubagentIdentity),
}

/// What identifies one tool call: its id and the agent that made it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CallIdentity {
    call: ToolCallId,
    agent: AgentIdentity,
}

impl CallIdentity {
    /// The identity of call `call` made by `agent`.
    #[must_use]
    pub const fn new(call: ToolCallId, agent: AgentIdentity) -> Self {
        Self { call, agent }
    }

    /// The call's id.
    #[must_use]
    pub const fn call(&self) -> &ToolCallId {
        &self.call
    }

    /// The agent that made the call.
    #[must_use]
    pub const fn agent(&self) -> &AgentIdentity {
        &self.agent
    }
}
