//! Command hooks: external hooks that gate every shell command.

use aither_sandbox::{AnyCommandGate, CommandGate, GateDecision};
use askama::Template;
use serde::{Deserialize, Serialize};

use super::lifecycle::ExternalHooks;
use super::runner::{HookCommand, HookRole, HookRunError};

/// A command hook as the host configures it.
///
/// Every shell command of the main agent and of every subagent is put to each
/// configured command hook before it runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandHookConfig {
    /// Names the hook in the refusals it causes and in diagnostics.
    pub name: String,
    /// What to run and for how long.
    #[serde(flatten)]
    pub command: HookCommand,
}

/// The one event a command hook runs for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandHookEvent {
    /// A shell command is about to run.
    BeforeCommand,
}

/// What a command hook reads on stdin: the role of the agent that wants to
/// run the command, and the command's full text.
#[derive(Debug, Serialize)]
pub struct CommandHookInput<'a> {
    /// The event, always `before_command`.
    pub event: CommandHookEvent,
    /// The agent that wants to run the command: the main agent or a subagent.
    pub role: HookRole,
    /// The script exactly as the agent wrote it.
    pub command: &'a str,
}

/// What a command hook prints on stdout: `{"decision":"allow"}` or
/// `{"decision":"block","reason":"..."}`.
///
/// There is nothing to grant: an answer carrying any other field is invalid,
/// and an allowed command still goes through the permission path unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum CommandHookOutput {
    /// The command may go on. A struct variant, so that an extra field is
    /// refused: serde ignores them on unit variants.
    Allow {},
    /// The command must not run.
    Block {
        /// Why; the model reads it as the command's result.
        reason: String,
    },
}

#[derive(Template)]
#[template(path = "command_hook_blocked.txt", escape = "none")]
struct BlockedTemplate<'a> {
    hook: &'a str,
    reason: &'a str,
}

#[derive(Template)]
#[template(path = "command_hook_failed.txt", escape = "none")]
struct FailedTemplate<'a> {
    hook: &'a str,
    failure: &'a str,
}

impl CommandHookConfig {
    /// The gate's decision for this hook's answer; a hook that gave no valid
    /// answer blocks, with a reason naming the failure.
    fn decide(&self, answer: Result<Option<CommandHookOutput>, HookRunError>) -> GateDecision {
        let blocked = |reason: askama::Result<String>| GateDecision::Block {
            reason: reason.expect("command hook templates must render"),
        };
        let failed = |failure: &str| {
            tracing::warn!(hook = %self.name, %failure, "command hook failed; the command is blocked");
            blocked(
                FailedTemplate {
                    hook: &self.name,
                    failure,
                }
                .render(),
            )
        };
        match answer {
            Ok(Some(CommandHookOutput::Allow {})) => GateDecision::Allow,
            Ok(Some(CommandHookOutput::Block { reason })) => blocked(
                BlockedTemplate {
                    hook: &self.name,
                    reason: &reason,
                }
                .render(),
            ),
            Ok(None) => failed("it printed no answer"),
            Err(error) => failed(&error.to_string()),
        }
    }
}

impl ExternalHooks {
    /// Puts `command` to every command hook, concurrently, and blocks it if
    /// any hook blocks it or fails.
    ///
    /// A hook that cannot run, runs past its deadline, exits unsuccessfully or
    /// prints nothing or an invalid answer blocks the command with a reason
    /// naming the failure: a gate that cannot answer lets nothing through.
    /// Of several blocking hooks the first configured one is reported.
    pub async fn check_command(&self, command: &str) -> GateDecision {
        let input = CommandHookInput {
            event: CommandHookEvent::BeforeCommand,
            role: self.role(),
            command,
        };
        let runs = self.command_hooks().iter().map(|hook| {
            let input = &input;
            async move {
                let answer = self
                    .runner()
                    .run::<_, CommandHookOutput>(&hook.command, input)
                    .await;
                hook.decide(answer)
            }
        });
        futures::future::join_all(runs)
            .await
            .into_iter()
            .find(|decision| matches!(decision, GateDecision::Block { .. }))
            .unwrap_or(GateDecision::Allow)
    }

    /// This scope as the command gate of a terminal's registry.
    #[must_use]
    pub fn command_gate(&self) -> AnyCommandGate {
        AnyCommandGate::new(self.clone())
    }
}

impl CommandGate for ExternalHooks {
    async fn check(&self, command: &str) -> GateDecision {
        self.check_command(command).await
    }
}
