//! External hooks: commands from host configuration that observe a session.
//!
//! A hook is a command that reads one typed JSON input on stdin and may print
//! one typed JSON answer on stdout. [`HookRunner`] runs it under a deadline
//! in a process group of its own and kills the whole group when the deadline
//! passes; each kind of hook defines its own input and answer types on top of
//! it. [`LiveHooks`] holds the configured hooks of a host and follows their
//! configuration and scripts live; [`ExternalHooks`] adapts them into the
//! agent's [`Hook`] chain, the command gate of a terminal's registry and the
//! reviewers of its permission path, and [`HookTestTool`] is the `hook_test`
//! command that runs a hook on samples.
//!
//! [`Hook`]: crate::Hook

mod command;
mod hook_test;
mod lifecycle;
mod live;
mod reviewer;
mod runner;
mod script;

pub use command::{CommandHookConfig, CommandHookEvent, CommandHookInput, CommandHookOutput};
pub use hook_test::{EventSample, HookTestArgs, HookTestError, HookTestTool};
pub use lifecycle::{
    AfterCompaction, AskUser, BeforeCompaction, CommandCalled, ExternalHooks, HookInput,
    InjectedContext, LifecycleHookConfig, LifecycleKind, LifecycleOutput, LifecyclePayload,
    SessionStart, Stop, SubagentEnd, SubagentStart, SubagentStop, UserMessage,
};
pub use live::{HookKind, HookWarning, HooksConfig, HooksConfigError, LiveHooks, LiveHooksError};
pub use reviewer::{ReviewAnswer, ReviewRequest, ReviewedPermission, ReviewerConfig};
pub use runner::{HookCommand, HookRole, HookRunError, HookRunner};
pub use script::HookCheckError;

// The hook fixtures are `sh` scripts made executable with Unix permissions.
#[cfg(all(test, unix))]
mod command_tests;
#[cfg(all(test, unix))]
mod live_tests;
#[cfg(all(test, unix))]
pub(crate) mod tests;
