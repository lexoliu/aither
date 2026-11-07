//! The text the model reads for every alert, notice, reminder and refusal
//! the agent raises.
//!
//! The agent raises each as typed data: a [`HostLoadEvent`], a [`JobAlert`],
//! a [`CompactRequest`], a [`ForkRefusal`] and so on. The [`Notices`] set on
//! the builder ([`AgentOptions::notices`](crate::AgentOptions::notices))
//! turns each into the natural language the model reads; [`DefaultNotices`]
//! renders aither's own wording. An embedder rewords a family by overriding
//! its one method and keeps the default for the rest.
//!
//! [`Notices`] extends the sandbox's [`SandboxNotices`], so the value given
//! to the builder also renders the terminal's refusals and notices.

use std::fmt;
use std::sync::Arc;

use aither_sandbox::{
    AnySandboxNotices, CommandKilled, CompletedTask, JobAlert, SandboxNotices,
    StartedBackgroundTask,
};
use askama::Template;

use crate::compact_tool::CompactRequest;
use crate::external_hook::InjectedContext;
use crate::host_load::HostLoadEvent;
use crate::secret_guard::{SecretFinding, SecretRefusal};
use crate::specialized::ForkRefusal;
use crate::working_document::DocumentAlertItem;

/// The context is close to automatic compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreCompaction {
    /// About how many tokens remain before compaction starts on its own.
    pub remaining_tokens: usize,
}

/// A tool call whose run was cancelled before it finished, given a result
/// so the conversation stays well-formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelledToolCall<'a> {
    /// The tool that was called.
    pub tool: &'a str,
}

/// The working document entering the conversation: whole, as the diff of a
/// change, or with the issue trees of roots it newly lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocumentShown<'a> {
    /// The document's name, its file as the embedder named it.
    pub document: &'a str,
    /// The whole document, at session start and after a reassembly.
    pub content: Option<&'a str>,
    /// The changed lines since the document was last shown, each prefixed
    /// `- ` or `+ `; empty when only blank lines changed.
    pub changes: Option<&'a str>,
    /// The issue trees of the roots the conversation has not seen yet,
    /// rendered inside their external-content boundary.
    pub trees: Option<&'a str>,
}

/// What the turn's checks of the working document found.
#[derive(Debug, Clone, Copy)]
pub struct DocumentAlert<'a> {
    /// The document's name, its file as the embedder named it.
    pub document: &'a str,
    /// Everything the alert says, in order; never empty.
    pub items: &'a [DocumentAlertItem],
    /// Whether it is raised as the model ends its turn and the run goes on
    /// after it, so the model must bring the document up to date before
    /// ending the turn.
    pub held: bool,
}

/// A shell command a command hook refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandHookRefusal<'a> {
    /// The hook answered with a block.
    Blocked {
        /// The hook's name.
        hook: &'a str,
        /// The reason the hook gave.
        reason: &'a str,
    },
    /// The hook did not answer, so the command is refused until it works.
    Failed {
        /// The hook's name.
        hook: &'a str,
        /// How it failed.
        failure: &'a str,
    },
}

/// A permission request a reviewer hook denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReviewerDenial<'a> {
    /// The reviewer's name.
    pub reviewer: &'a str,
    /// The reason it gave.
    pub reason: &'a str,
}

/// A main-agent tool called in a fork, where it is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForkToolRefusal<'a> {
    /// The tool's name.
    pub tool: &'a str,
}

#[derive(Template)]
#[template(path = "precompaction_alert.txt", escape = "none")]
struct PreCompactionTemplate {
    remaining_tokens: usize,
}

#[derive(Template)]
#[template(path = "tool_call_cancelled.txt", escape = "none")]
struct CancelledTemplate<'a> {
    tool: &'a str,
}

#[derive(Template)]
#[template(path = "compact_started.txt", escape = "none")]
struct CompactStarted;

#[derive(Template)]
#[template(path = "compact_in_flight.txt", escape = "none")]
struct CompactInFlight;

#[derive(Template)]
#[template(path = "compact_settled.txt", escape = "none")]
struct CompactSettled;

#[derive(Template)]
#[template(path = "compact_below_floor.txt", escape = "none")]
struct CompactBelowFloor {
    floor: usize,
}

#[derive(Template)]
#[template(path = "lifecycle_context.txt", escape = "none")]
struct LifecycleContext<'a> {
    hook: &'a str,
    event: &'a str,
    context: &'a str,
}

#[derive(Template)]
#[template(path = "command_hook_blocked.txt", escape = "none")]
struct HookBlocked<'a> {
    hook: &'a str,
    reason: &'a str,
}

#[derive(Template)]
#[template(path = "command_hook_failed.txt", escape = "none")]
struct HookFailed<'a> {
    hook: &'a str,
    failure: &'a str,
}

#[derive(Template)]
#[template(path = "reviewer_denial.txt", escape = "none")]
struct Denial<'a> {
    reviewer: &'a str,
    reason: &'a str,
}

#[derive(Template)]
#[template(path = "fork_tool_refused.txt", escape = "none")]
struct ForkToolRefused<'a> {
    name: &'a str,
}

#[derive(Template)]
#[template(path = "secret_refusal.txt", escape = "none")]
struct SecretRefused<'a> {
    findings: &'a [SecretFinding],
}

pub(crate) fn rendered(template: &impl Template) -> String {
    template
        .render()
        .expect("a notice template renders typed input")
}

/// Renders the text the model reads for each alert, notice, reminder and
/// refusal the agent raises, one method per family.
///
/// Every method has a default, aither's own wording, so an implementation
/// overrides only the families it rewords, and a family added later arrives
/// with its default. An override that wants to extend the default calls the
/// same method on [`DefaultNotices`].
///
/// The supertrait [`SandboxNotices`] holds the families the terminal renders
/// itself; an implementation writes `impl SandboxNotices for … {}` to keep
/// all of them.
pub trait Notices: SandboxNotices {
    /// A host load or memory or disk pressure condition starting, changing
    /// or ending, or a job killed for memory.
    fn host_load(&self, event: &HostLoadEvent) -> String {
        event.alert_text()
    }

    /// An event of a background job or of a monitor: processes killed with
    /// a foreground command, a long-running or stdin-blocked job, a reset
    /// working directory, or a monitor's output, stall or end.
    fn job_alert(&self, alert: &JobAlert) -> String {
        crate::agent::default_job_alert(alert)
    }

    /// The reminder pushed after a command moved to the background.
    fn background_started(&self, task: &StartedBackgroundTask) -> String {
        crate::agent::default_background_started(task)
    }

    /// A background command that finished, was killed or failed. The
    /// default states a timeout kill through
    /// [`command_killed`](SandboxNotices::command_killed).
    fn background_finished(&self, task: &CompletedTask) -> String {
        let timeout = task
            .timeout_kill
            .as_ref()
            .map(|kill| self.command_killed(&CommandKilled::TimedOut(kill)));
        crate::agent::default_background_finished(task, timeout.as_deref())
    }

    /// The one alert before automatic compaction.
    fn precompaction(&self, alert: &PreCompaction) -> String {
        rendered(&PreCompactionTemplate {
            remaining_tokens: alert.remaining_tokens,
        })
    }

    /// The result of a tool call whose run was cancelled before it finished.
    fn tool_call_cancelled(&self, call: &CancelledToolCall<'_>) -> String {
        rendered(&CancelledTemplate { tool: call.tool })
    }

    /// The result of a `compact` call.
    fn compact(&self, request: &CompactRequest) -> String {
        match *request {
            CompactRequest::Started => rendered(&CompactStarted),
            CompactRequest::InFlight => rendered(&CompactInFlight),
            CompactRequest::Settled => rendered(&CompactSettled),
            CompactRequest::BelowFloor { floor } => rendered(&CompactBelowFloor { floor }),
        }
    }

    /// The working document entering the conversation.
    fn working_document(&self, shown: &DocumentShown<'_>) -> String {
        crate::working_document::default_shown(shown)
    }

    /// What the turn's checks of the working document found.
    fn document_alert(&self, alert: &DocumentAlert<'_>) -> String {
        crate::working_document::default_alert(alert)
    }

    /// Context a lifecycle hook added.
    fn lifecycle_context(&self, injected: &InjectedContext) -> String {
        rendered(&LifecycleContext {
            hook: injected.hook(),
            event: injected.event().occasion(),
            context: injected.context(),
        })
    }

    /// A command a command hook refused. The terminal then renders the
    /// result through [`command_blocked`](SandboxNotices::command_blocked).
    fn command_hook(&self, refusal: &CommandHookRefusal<'_>) -> String {
        match *refusal {
            CommandHookRefusal::Blocked { hook, reason } => rendered(&HookBlocked { hook, reason }),
            CommandHookRefusal::Failed { hook, failure } => rendered(&HookFailed { hook, failure }),
        }
    }

    /// A permission request a reviewer hook denied.
    fn reviewer_denial(&self, denial: &ReviewerDenial<'_>) -> String {
        rendered(&Denial {
            reviewer: denial.reviewer,
            reason: denial.reason,
        })
    }

    /// A fork that was not started.
    fn fork_refusal(&self, refusal: &ForkRefusal) -> String {
        crate::specialized::default_fork_refusal(refusal)
    }

    /// A main-agent tool called in a fork.
    fn fork_tool_refusal(&self, refusal: &ForkToolRefusal<'_>) -> String {
        rendered(&ForkToolRefused { name: refusal.tool })
    }

    /// Text refused because it carries secrets.
    fn secret_refusal(&self, refusal: &SecretRefusal) -> String {
        rendered(&SecretRefused {
            findings: &refusal.findings,
        })
    }

    /// The language alerts raised since the last request, as one alert.
    #[cfg(feature = "languages")]
    fn language_alerts(&self, alerts: &[crate::language::LanguageAlert]) -> String {
        crate::language::default_alerts(alerts)
    }

    /// Text refused because it is not in the language it must be written in.
    #[cfg(feature = "languages")]
    fn language_refusal(&self, refusal: &crate::LanguageRefusal) -> String {
        crate::language::default_refusal(refusal)
    }
}

/// aither's own wording for every family.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultNotices;

impl SandboxNotices for DefaultNotices {}

impl Notices for DefaultNotices {}

/// A shared [`Notices`], cloned into the agent, its subagents and every
/// tool and hook that renders a notice.
///
/// One value renders both crates' families: [`sandbox`](Self::sandbox)
/// shares the same object as the terminal's [`AnySandboxNotices`].
#[derive(Clone)]
pub struct AnyNotices(Arc<dyn Notices>);

impl AnyNotices {
    /// Shares `notices`.
    #[must_use]
    pub fn new(notices: impl Notices) -> Self {
        Self(Arc::new(notices))
    }

    /// The same object, as the sandbox's notices: what a
    /// [`ToolRegistryBuilder`](aither_sandbox::ToolRegistryBuilder) built by
    /// hand is given through its `notices`.
    #[must_use]
    pub fn sandbox(&self) -> AnySandboxNotices {
        let shared: Arc<dyn SandboxNotices> = self.0.clone();
        AnySandboxNotices::from(shared)
    }
}

impl Default for AnyNotices {
    fn default() -> Self {
        Self::new(DefaultNotices)
    }
}

impl std::ops::Deref for AnyNotices {
    type Target = dyn Notices;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl fmt::Debug for AnyNotices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyNotices").finish_non_exhaustive()
    }
}
