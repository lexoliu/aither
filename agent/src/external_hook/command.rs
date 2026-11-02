//! Command hooks: external hooks that gate every shell command.

use aither_sandbox::{AnyCommandGate, CommandGate, GateDecision};
use askama::Template;
use serde::{Deserialize, Serialize};

use super::lifecycle::ExternalHooks;
use super::live::Run;
use super::runner::{HookCommand, HookRole};

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
    /// The gate's decision for this hook's answer.
    fn decide(&self, answer: Run<CommandHookOutput>) -> GateDecision {
        let blocked = |reason: askama::Result<String>| GateDecision::Block {
            reason: reason.expect("command hook templates must render"),
        };
        match answer {
            Run::Answered(CommandHookOutput::Allow {}) | Run::Gone => GateDecision::Allow,
            Run::Answered(CommandHookOutput::Block { reason }) => blocked(
                BlockedTemplate {
                    hook: &self.name,
                    reason: &reason,
                }
                .render(),
            ),
            Run::Failed(failure) => {
                tracing::warn!(hook = %self.name, %failure, "command hook failed; the command is blocked");
                blocked(
                    FailedTemplate {
                        hook: &self.name,
                        failure: &failure,
                    }
                    .render(),
                )
            }
        }
    }
}

impl ExternalHooks {
    /// Puts `command` to every command hook, concurrently, and blocks it if
    /// any hook blocks it or fails.
    ///
    /// A hook that cannot run, runs past its deadline, exits unsuccessfully or
    /// prints nothing or an invalid answer is replaced by its previous version
    /// when it has one (see [`LiveHooks`](super::LiveHooks)); when no working
    /// version is left, the user is asked to fix it and the command waits, or
    /// is blocked with a reason naming the failure if the user cannot be
    /// asked: a gate that cannot answer lets nothing through. Of several
    /// blocking hooks the first configured one is reported.
    pub async fn check_command(&self, command: &str) -> GateDecision {
        let input = CommandHookInput {
            event: CommandHookEvent::BeforeCommand,
            role: self.role(),
            command,
        };
        let set = self.live().set();
        let runs = set.command.iter().map(|hook| {
            let input = &input;
            async move {
                let answer = self
                    .live()
                    .run_hook(
                        self.user(),
                        hook,
                        input,
                        |answer: Option<CommandHookOutput>| {
                            answer.ok_or_else(|| "it printed no answer".to_string())
                        },
                    )
                    .await;
                hook.config.decide(answer)
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
