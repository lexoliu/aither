//! The text the model reads for what the sandbox refuses and announces.
//!
//! The sandbox raises every refusal and notice as typed data: an
//! [`UnchangedTaskRead`], a [`CommandBlocked`], a [`MonitorStarted`] and so
//! on. The [`SandboxNotices`] set on the [`ToolRegistryBuilder`] turns each
//! into the natural language the model reads; [`DefaultSandboxNotices`]
//! renders the sandbox's own wording. An embedder rewords a family by
//! overriding its one method and keeps the default for the rest. A
//! [`CommandPolicy`](crate::CommandPolicy) refusal carries its own text and
//! has no family here.
//!
//! [`ToolRegistryBuilder`]: crate::ToolRegistryBuilder

use std::fmt;
use std::sync::Arc;

use askama::Template;

use crate::command::CommandBlocked;
use crate::job_registry::UnchangedTaskRead;
use crate::monitor::{MAX_TIMEOUT_SECS, MonitorRefusal};
use crate::terminal::TimeoutKill;

/// A `monitor` call that started watching a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorStarted<'a> {
    /// The task the monitor watches.
    pub task_id: &'a str,
    /// How often its new output is delivered, in seconds.
    pub interval_secs: u64,
}

/// A command invoked by a subagent that only the main agent may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedForMain<'a> {
    /// The command's name.
    pub command: &'a str,
}

/// Why a foreground command the terminal killed was killed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKilled<'a> {
    /// Its `timeout` bound passed.
    TimedOut(&'a TimeoutKill),
    /// It was killed before its bound passed, by `terminal_kill` or a
    /// watchdog.
    BeforeTimeout,
}

#[derive(Template)]
#[template(path = "unchanged_task_read.txt")]
struct UnchangedTaskReadTemplate<'a> {
    task_id: &'a str,
}

#[derive(Template)]
#[template(path = "command_blocked.txt")]
struct Blocked<'a> {
    reason: &'a str,
}

#[derive(Template)]
#[template(path = "monitor_refused.txt", escape = "none")]
struct MonitorRefused<'a> {
    refusal: &'a MonitorRefusal,
    max_timeout_secs: u64,
}

#[derive(Template)]
#[template(path = "monitor_started.txt", escape = "none")]
struct MonitorStartedTemplate<'a> {
    task_id: &'a str,
    interval_secs: u64,
}

#[derive(Template)]
#[template(path = "reserved_for_main.txt", escape = "none")]
struct Reserved<'a> {
    name: &'a str,
}

#[derive(Template)]
#[template(path = "command_timed_out.txt", escape = "none")]
struct Killed<'a> {
    killed: CommandKilled<'a>,
}

fn rendered(template: &impl Template) -> String {
    template
        .render()
        .expect("a sandbox notice template renders typed input")
}

/// Renders the text the model reads for each refusal and notice the sandbox
/// raises, one method per family.
///
/// Every method has a default, the sandbox's own wording, so an
/// implementation overrides only the families it rewords, and a family added
/// later arrives with its default. An override that wants to extend the
/// default calls the same method on [`DefaultSandboxNotices`].
///
/// aither-agent's `Notices` extends this trait, so one value an embedder
/// gives the agent builder renders the sandbox's families too.
pub trait SandboxNotices: Send + Sync + 'static {
    /// A `terminal_read` refused because the task's output has not changed
    /// since its last read.
    fn unchanged_task_read(&self, read: &UnchangedTaskRead) -> String {
        rendered(&UnchangedTaskReadTemplate {
            task_id: &read.task_id,
        })
    }

    /// A command the terminal's command gate blocked, wrapping the reason
    /// the gate gave.
    fn command_blocked(&self, blocked: &CommandBlocked) -> String {
        rendered(&Blocked {
            reason: &blocked.reason,
        })
    }

    /// A `monitor` call refused before anything started.
    fn monitor_refusal(&self, refusal: &MonitorRefusal) -> String {
        rendered(&MonitorRefused {
            refusal,
            max_timeout_secs: MAX_TIMEOUT_SECS,
        })
    }

    /// The result of a `monitor` call that started watching.
    fn monitor_started(&self, started: &MonitorStarted<'_>) -> String {
        rendered(&MonitorStartedTemplate {
            task_id: started.task_id,
            interval_secs: started.interval_secs,
        })
    }

    /// A subagent's call of a command reserved for the main agent.
    fn reserved_for_main(&self, reserved: &ReservedForMain<'_>) -> String {
        rendered(&Reserved {
            name: reserved.command,
        })
    }

    /// Why a command was killed, appended to its result.
    fn command_killed(&self, killed: &CommandKilled<'_>) -> String {
        rendered(&Killed { killed: *killed })
    }
}

/// The sandbox's own wording for every family.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultSandboxNotices;

impl SandboxNotices for DefaultSandboxNotices {}

/// A shared [`SandboxNotices`], cloned into every registry and tool that
/// renders a notice.
#[derive(Clone)]
pub struct AnySandboxNotices(Arc<dyn SandboxNotices>);

impl AnySandboxNotices {
    /// Shares `notices`.
    #[must_use]
    pub fn new(notices: impl SandboxNotices) -> Self {
        Self(Arc::new(notices))
    }
}

/// Shares a value another wrapper already holds, such as aither-agent's
/// `AnyNotices`, so both render through the same object.
impl From<Arc<dyn SandboxNotices>> for AnySandboxNotices {
    fn from(notices: Arc<dyn SandboxNotices>) -> Self {
        Self(notices)
    }
}

impl Default for AnySandboxNotices {
    fn default() -> Self {
        Self::new(DefaultSandboxNotices)
    }
}

impl std::ops::Deref for AnySandboxNotices {
    type Target = dyn SandboxNotices;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl fmt::Debug for AnySandboxNotices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnySandboxNotices").finish_non_exhaustive()
    }
}

/// A call refused before it ran: the typed reason, and the text its
/// notices rendered for the model.
///
/// Tools return it as their error, so an agent loop tells one refusal from
/// another, and from any other failure, by downcasting to
/// `Refused<UnchangedTaskRead>`, `Refused<CommandBlocked>` and so on.
#[derive(Debug)]
pub struct Refused<R> {
    reason: R,
    text: String,
}

impl<R> Refused<R> {
    /// The refusal of `reason`, read by the model as what `render` returns.
    pub fn rendered(reason: R, render: impl FnOnce(&R) -> String) -> Self {
        let text = render(&reason);
        Self { reason, text }
    }

    /// Why the call was refused.
    pub const fn reason(&self) -> &R {
        &self.reason
    }

    /// The text the model reads.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Takes the reason, dropping the text.
    pub fn into_reason(self) -> R {
        self.reason
    }
}

impl<R> fmt::Display for Refused<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl<R: fmt::Debug> std::error::Error for Refused<R> {}

#[cfg(test)]
mod tests {
    use super::*;

    struct Terse;

    impl SandboxNotices for Terse {
        fn unchanged_task_read(&self, read: &UnchangedTaskRead) -> String {
            format!("no news from {}", read.task_id)
        }
    }

    fn unchanged() -> UnchangedTaskRead {
        UnchangedTaskRead {
            task_id: "amber-forest-thunder-pearl".into(),
        }
    }

    #[test]
    fn an_override_replaces_its_family_and_keeps_the_rest() {
        let notices = AnySandboxNotices::new(Terse);
        assert_eq!(
            notices.unchanged_task_read(&unchanged()),
            "no news from amber-forest-thunder-pearl"
        );
        let blocked = CommandBlocked {
            reason: "not today".into(),
        };
        assert_eq!(
            notices.command_blocked(&blocked),
            DefaultSandboxNotices.command_blocked(&blocked),
        );
    }

    #[test]
    fn a_refusal_carries_its_reason_and_rendered_text() {
        let notices = AnySandboxNotices::default();
        let refused = Refused::rendered(unchanged(), |read| notices.unchanged_task_read(read));
        assert_eq!(refused.reason(), &unchanged());
        assert!(
            refused
                .to_string()
                .contains("'amber-forest-thunder-pearl' produced no new output")
        );
    }

    #[test]
    fn default_monitor_refusal_names_the_cap() {
        let text = DefaultSandboxNotices.monitor_refusal(&MonitorRefusal::TimeoutOutOfRange);
        assert_eq!(
            text,
            "The monitor call was refused: timeout_secs must be between 1 and 7200."
        );
    }

    #[test]
    fn default_kill_notice_without_a_bound_passing() {
        assert_eq!(
            DefaultSandboxNotices.command_killed(&CommandKilled::BeforeTimeout),
            "Killed before its timeout bound passed."
        );
    }
}
