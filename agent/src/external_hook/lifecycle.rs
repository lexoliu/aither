//! Lifecycle hooks: event-driven external hooks that can only add context.

use aither_sandbox::{AnyCommandObserver, CommandCall, CommandObserver};
use askama::Template;
use executor_core::Executor;
use serde::{Deserialize, Serialize};

use super::live::{LiveHooks, UserPause};
use super::runner::{HookCommand, HookRole};
use crate::StopReason;
use crate::ask_user::{AskUserBroker, Question, QuestionAnswer};
use crate::hook::Hook;

/// A moment in a session a lifecycle hook can subscribe to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleKind {
    /// The session's first run begins.
    SessionStart,
    /// A compaction is about to start.
    BeforeCompaction,
    /// A compaction was folded into the window.
    AfterCompaction,
    /// A run ended.
    Stop,
    /// The user sent a message.
    UserMessage,
    /// A subagent is about to run.
    SubagentStart,
    /// A subagent's run ended.
    SubagentStop,
    /// The user answered an `ask_user` question.
    AskUser,
    /// A command was called through the IPC bridge. Its hooks run in the
    /// background, so they never delay the call; every other event runs its
    /// hooks before the agent goes on, so their context reaches its very next
    /// request.
    CommandCall,
}

impl LifecycleKind {
    /// What happened, as the context alert says it.
    pub(super) const fn occasion(self) -> &'static str {
        match self {
            Self::SessionStart => "the session started",
            Self::BeforeCompaction => "a compaction was about to start",
            Self::AfterCompaction => "a compaction finished",
            Self::Stop => "a run stopped",
            Self::UserMessage => "the user sent a message",
            Self::SubagentStart => "a subagent was about to run",
            Self::SubagentStop => "a subagent finished running",
            Self::AskUser => "the user answered a question",
            Self::CommandCall => "a command was called",
        }
    }
}

/// The input of one lifecycle event; the hook receives it with the event name
/// and the caller's role (see [`HookInput`]).
pub trait LifecyclePayload: Serialize + Send + Sync {
    /// The event this payload describes.
    const KIND: LifecycleKind;
}

/// What a lifecycle hook reads on stdin: the event, the role of the agent
/// whose session it happened in, and the event's own fields.
#[derive(Debug, Serialize)]
pub struct HookInput<'a, Payload> {
    /// The event.
    pub event: LifecycleKind,
    /// The agent the event happened to: the main agent or a subagent.
    pub role: HookRole,
    /// The event's fields.
    #[serde(flatten)]
    pub payload: &'a Payload,
}

/// What a lifecycle hook prints on stdout.
///
/// The only thing a lifecycle hook can say is context to add; there is no
/// field to block or approve anything, and an answer carrying any other field
/// is invalid. Printing nothing, or `{}`, adds nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleOutput {
    /// Context for the model, delivered once as a system alert.
    #[serde(default)]
    pub context: Option<String>,
}

/// Input of [`LifecycleKind::SessionStart`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SessionStart {}

/// Input of [`LifecycleKind::BeforeCompaction`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BeforeCompaction {
    /// Messages in the conversation about to be compacted.
    pub messages: usize,
    /// What the compaction was asked to focus on, if anything.
    pub focus: Option<String>,
}

/// Input of [`LifecycleKind::AfterCompaction`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AfterCompaction {
    /// Messages left in the conversation after the compaction.
    pub messages_remaining: usize,
}

/// Input of [`LifecycleKind::Stop`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Stop {
    /// The run's final response text.
    pub final_text: String,
    /// Turns the run took.
    pub turns: usize,
    /// Why the run stopped.
    pub reason: StopReason,
}

/// Input of [`LifecycleKind::UserMessage`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UserMessage {
    /// The message text.
    pub message: String,
    /// How many attachments came with it.
    pub attachments: usize,
}

/// Input of [`LifecycleKind::SubagentStart`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SubagentStart {
    /// The subagent type.
    pub subagent: String,
    /// The task the subagent was given.
    pub prompt: String,
}

/// How a subagent's run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SubagentEnd {
    /// The subagent answered.
    Completed,
    /// The subagent failed.
    Failed,
    /// The user cancelled the subagent.
    Cancelled,
}

/// Input of [`LifecycleKind::SubagentStop`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SubagentStop {
    /// The subagent type.
    pub subagent: String,
    /// How the run ended.
    pub end: SubagentEnd,
    /// The subagent's answer, or the failure.
    pub result: String,
}

/// Input of [`LifecycleKind::AskUser`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AskUser {
    /// The questions asked.
    pub questions: Vec<Question>,
    /// The user's answers.
    pub answers: Vec<QuestionAnswer>,
}

/// Input of [`LifecycleKind::CommandCall`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CommandCalled {
    /// The resolved command name.
    pub name: String,
    /// The arguments it was called with.
    pub args: Vec<String>,
    /// Whether the command succeeded.
    pub ok: bool,
    /// What the command printed on success.
    pub output: Option<String>,
    /// Why the command failed.
    pub error: Option<String>,
}

macro_rules! payloads {
    ($($payload:ident => $kind:ident),* $(,)?) => {
        $(impl LifecyclePayload for $payload {
            const KIND: LifecycleKind = LifecycleKind::$kind;
        })*
    };
}

payloads! {
    SessionStart => SessionStart,
    BeforeCompaction => BeforeCompaction,
    AfterCompaction => AfterCompaction,
    Stop => Stop,
    UserMessage => UserMessage,
    SubagentStart => SubagentStart,
    SubagentStop => SubagentStop,
    AskUser => AskUser,
    CommandCalled => CommandCall,
}

/// A lifecycle hook as the host configures it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleHookConfig {
    /// Names the hook in the alerts it raises and in diagnostics.
    pub name: String,
    /// What to run and for how long.
    #[serde(flatten)]
    pub command: HookCommand,
    /// The events the hook is run for.
    pub events: Vec<LifecycleKind>,
}

/// Context a lifecycle hook injected, waiting to be delivered as a system
/// alert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectedContext {
    hook: String,
    event: LifecycleKind,
    context: String,
}

#[derive(Template)]
#[template(path = "lifecycle_context.txt", escape = "none")]
struct LifecycleContextTemplate<'a> {
    hook: &'a str,
    event: &'a str,
    context: &'a str,
}

impl InjectedContext {
    /// The alert the model reads.
    ///
    /// # Panics
    ///
    /// Panics if the alert template fails to render, which it cannot for
    /// text input.
    #[must_use]
    pub fn alert_text(&self) -> String {
        LifecycleContextTemplate {
            hook: &self.hook,
            event: self.event.occasion(),
            context: &self.context,
        }
        .render()
        .expect("lifecycle context template must render")
    }
}

/// External hooks for one agent: the host's live hooks, the role of the
/// agent, and the inbox its injected context waits in.
///
/// Add it to an agent's hook chain with `hook(...)`, to the command registry
/// with [`ToolRegistryBuilder::observe_commands`], and to a subagent tool or
/// the `ask_user` tool that fires events of its own. Clones share the inbox;
/// [`for_subagent`](Self::for_subagent) makes the scope of a subagent, with
/// an inbox of its own, so context raised in a subagent reaches that
/// subagent and not its parent.
///
/// Lifecycle hooks cannot block: a hook that fails, runs past its deadline or
/// prints an invalid answer is reported through `tracing` and its result is
/// ignored.
///
/// [`ToolRegistryBuilder::observe_commands`]: aither_sandbox::ToolRegistryBuilder::observe_commands
#[derive(Debug, Clone)]
pub struct ExternalHooks {
    live: LiveHooks,
    role: HookRole,
    /// Where a failing command hook or reviewer asks the user to fix it.
    user: Option<UserPause>,
    inbox: async_channel::Sender<InjectedContext>,
    outbox: async_channel::Receiver<InjectedContext>,
}

impl ExternalHooks {
    /// The scope of the main agent, running the hooks of `live`.
    #[must_use]
    pub fn new(live: &LiveHooks) -> Self {
        Self::scope(live.clone(), HookRole::Main, None)
    }

    /// A scope with no hooks, for an agent whose host configured none.
    #[must_use]
    pub fn none() -> Self {
        Self::new(&LiveHooks::none())
    }

    /// Asks the user through `broker`, as `session_id`, to fix a command hook
    /// or reviewer that fails when no working version of it is left. The
    /// session waits for the answer; without this, such a hook blocks its
    /// command, and a reviewer passes its request on. One question is open at
    /// a time per call of this method: the scopes made from the result for
    /// subagents share it, other sessions' calls do not.
    #[must_use]
    pub fn with_user_pause(mut self, broker: AskUserBroker, session_id: impl Into<String>) -> Self {
        self.user = Some(UserPause::new(broker, session_id.into()));
        self
    }

    pub(super) const fn live(&self) -> &LiveHooks {
        &self.live
    }

    pub(super) const fn user(&self) -> Option<&UserPause> {
        self.user.as_ref()
    }

    fn scope(live: LiveHooks, role: HookRole, user: Option<UserPause>) -> Self {
        let (inbox, outbox) = async_channel::unbounded();
        Self {
            live,
            role,
            user,
            inbox,
            outbox,
        }
    }

    /// The scope of a subagent: the same hooks, the subagent role and an
    /// inbox of its own.
    #[must_use]
    pub fn for_subagent(&self) -> Self {
        Self::scope(self.live.clone(), HookRole::Subagent, self.user.clone())
    }

    /// The role of the agent this scope belongs to.
    #[must_use]
    pub const fn role(&self) -> HookRole {
        self.role
    }

    /// Runs the hooks subscribed to `P`'s event with `payload`, concurrently
    /// (at most as many as the host configured), and queues the context they
    /// inject for the scope's agent.
    ///
    /// Never fails and never blocks the caller beyond the hooks' deadlines: a
    /// hook that fails is reported and ignored.
    ///
    /// # Panics
    ///
    /// Panics if the scope's own inbox is closed, which cannot happen while
    /// the scope exists.
    pub async fn fire<P: LifecyclePayload>(&self, payload: &P) {
        let input = HookInput {
            event: P::KIND,
            role: self.role,
            payload,
        };
        let set = self.live.set();
        let runs = set
            .lifecycle
            .iter()
            .filter(|hook| hook.config.events.contains(&P::KIND))
            .filter_map(|hook| Some((hook, hook.current.as_ref()?)))
            .map(|(hook, version)| {
                let input = &input;
                async move {
                    let answer = self
                        .live
                        .runner()
                        .run::<_, LifecycleOutput>(&version.command, input)
                        .await;
                    (&hook.config, answer)
                }
            });
        for (hook, answer) in futures::future::join_all(runs).await {
            match answer {
                Ok(Some(LifecycleOutput {
                    context: Some(context),
                })) if !context.trim().is_empty() => {
                    self.inbox
                        .try_send(InjectedContext {
                            hook: hook.name.clone(),
                            event: P::KIND,
                            context,
                        })
                        .expect("the scope holds its own inbox open");
                }
                Ok(_) => {}
                Err(error) => tracing::warn!(
                    hook = %hook.name,
                    event = ?P::KIND,
                    %error,
                    "lifecycle hook failed; its result is ignored"
                ),
            }
        }
    }

    /// How many pieces of context wait to be delivered.
    #[cfg(test)]
    pub(super) fn queued(&self) -> usize {
        self.outbox.len()
    }

    /// Takes the context injected since the last call; each piece is
    /// delivered once.
    fn drain(&self) -> Vec<InjectedContext> {
        std::iter::from_fn(|| self.outbox.try_recv().ok()).collect()
    }
}

impl Hook for ExternalHooks {
    fn on_lifecycle<P: LifecyclePayload>(
        &self,
        payload: &P,
    ) -> impl std::future::Future<Output = ()> + Send {
        self.fire(payload)
    }

    fn take_injected_context(&self) -> Vec<InjectedContext> {
        self.drain()
    }
}

impl CommandObserver for ExternalHooks {
    /// Starts the hooks of the call on the scope's executor and returns at
    /// once: they cannot change the call's answer, so the call never waits
    /// for them. Their context is queued when they finish and reaches the
    /// agent at its next request, once.
    async fn command_called(&self, call: &CommandCall<'_>) {
        if !self
            .live
            .set()
            .lifecycle
            .iter()
            .any(|hook| hook.config.events.contains(&LifecycleKind::CommandCall))
        {
            return;
        }
        let payload = CommandCalled {
            name: call.name.to_string(),
            args: call.args.to_vec(),
            ok: call.outcome.ok,
            output: call
                .outcome
                .payload
                .as_ref()
                .map(aither_sandbox::CommandPayload::render_for_cli),
            error: call.outcome.error.clone(),
        };
        let hooks = self.clone();
        self.live
            .executor()
            .expect("a scope with hooks has an executor")
            .spawn(async move { hooks.fire(&payload).await })
            .detach();
    }
}

impl ExternalHooks {
    /// This scope as the observer of a command registry.
    #[must_use]
    pub fn command_observer(&self) -> AnyCommandObserver {
        AnyCommandObserver::new(self.clone())
    }

    /// The scope a terminal's registry reports to, when the host installed
    /// one there.
    #[must_use]
    pub fn of_observer(observer: Option<&AnyCommandObserver>) -> Option<Self> {
        observer?.downcast_ref::<Self>().cloned()
    }
}
