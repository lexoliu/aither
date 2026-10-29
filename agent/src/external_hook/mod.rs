//! External hooks: commands from host configuration that observe a session.
//!
//! A hook is a command that reads one typed JSON input on stdin and may print
//! one typed JSON answer on stdout. [`HookRunner`] runs it under a deadline
//! in a process group of its own and kills the whole group when the deadline
//! passes; each kind of hook defines its own input and answer types on top of
//! it. [`ExternalHooks`] adapts the lifecycle kind into the agent's [`Hook`]
//! chain, and the command kind into the command gate of a terminal's
//! registry.
//!
//! [`Hook`]: crate::Hook

mod command;
mod lifecycle;
mod reviewer;
mod runner;

pub use command::{CommandHookConfig, CommandHookEvent, CommandHookInput, CommandHookOutput};
pub use lifecycle::{
    AfterCompaction, AskUser, BeforeCompaction, CommandCalled, ExternalHooks, HookInput,
    InjectedContext, LifecycleHookConfig, LifecycleKind, LifecycleOutput, LifecyclePayload,
    SessionStart, Stop, SubagentEnd, SubagentStart, SubagentStop, UserMessage,
};
pub use reviewer::{
    ReviewAnswer, ReviewRequest, ReviewedPermission, ReviewerChain, ReviewerConfig,
};
pub use runner::{HookCommand, HookRole, HookRunError, HookRunner};

// The hook fixtures are `sh` scripts made executable with Unix permissions.
#[cfg(all(test, unix))]
mod command_tests;
#[cfg(all(test, unix))]
pub(crate) mod tests;
