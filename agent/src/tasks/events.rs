//! The state changes a `tasks.md` row tracks.
//!
//! Whoever causes one records it on the agent's [`TaskEvents`]: the agent
//! itself for its background jobs, the `subagent` command for the subagents
//! it runs, the host for issues, pull requests and questions. The tasks
//! document takes them at the next request and, when a turn leaves
//! `tasks.md` as it was, alerts once per event, naming the row it affects.

use super::reference::WorkRef;

/// A state change a `tasks.md` row may have to follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskEvent {
    /// An issue was created.
    IssueCreated(WorkRef),
    /// An issue was closed.
    IssueClosed(WorkRef),
    /// A pull request was opened.
    PullRequestOpened(WorkRef),
    /// A pull request was merged.
    PullRequestMerged(WorkRef),
    /// A pull request was rejected and needs work before it can land.
    PullRequestRejected(WorkRef),
    /// A pull request was closed without landing.
    PullRequestClosed(WorkRef),
    /// An `ask_user` question was asked; the payload is its request id.
    QuestionAsked(String),
    /// An `ask_user` question was answered.
    QuestionAnswered(String),
    /// An `ask_user` question was withdrawn.
    QuestionWithdrawn(String),
    /// A subagent started; the payload is its name.
    SubagentStarted(String),
    /// A subagent finished.
    SubagentFinished(String),
    /// A background job finished; the payload is its task id.
    JobFinished(String),
}

/// What an event is about, as a row would name it.
pub enum Subject<'a> {
    /// An issue or pull request.
    Work(&'a WorkRef),
    /// A question, subagent or job id.
    Id(&'a str),
}

impl TaskEvent {
    pub(crate) fn subject(&self) -> Subject<'_> {
        match self {
            Self::IssueCreated(reference)
            | Self::IssueClosed(reference)
            | Self::PullRequestOpened(reference)
            | Self::PullRequestMerged(reference)
            | Self::PullRequestRejected(reference)
            | Self::PullRequestClosed(reference) => Subject::Work(reference),
            Self::QuestionAsked(id)
            | Self::QuestionAnswered(id)
            | Self::QuestionWithdrawn(id)
            | Self::SubagentStarted(id)
            | Self::SubagentFinished(id)
            | Self::JobFinished(id) => Subject::Id(id),
        }
    }

    /// Whether the event starts something `tasks.md` must carry: a row naming
    /// its subject records it, and without one it alerts. Every other event
    /// concerns only the rows that already name its subject.
    pub(crate) const fn opens_work(&self) -> bool {
        matches!(
            self,
            Self::IssueCreated(_)
                | Self::PullRequestOpened(_)
                | Self::QuestionAsked(_)
                | Self::SubagentStarted(_)
        )
    }

    /// Whether this event ends what `start` opened: a subagent that finished
    /// or a question answered or withdrawn leaves nothing for a row to carry.
    pub(crate) fn ends(&self, start: &Self) -> bool {
        match (start, self) {
            (Self::SubagentStarted(started), Self::SubagentFinished(finished)) => {
                started == finished
            }
            (
                Self::QuestionAsked(asked),
                Self::QuestionAnswered(closed) | Self::QuestionWithdrawn(closed),
            ) => asked == closed,
            _ => false,
        }
    }
}

/// The inbox of [`TaskEvent`]s for one agent's tasks document.
///
/// Cloning shares the inbox. Recording never blocks and never fails: the
/// inbox is unbounded and lives as long as any handle does.
#[derive(Debug, Clone)]
pub struct TaskEvents {
    sender: async_channel::Sender<TaskEvent>,
    receiver: async_channel::Receiver<TaskEvent>,
}

impl Default for TaskEvents {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskEvents {
    /// An empty inbox.
    #[must_use]
    pub fn new() -> Self {
        let (sender, receiver) = async_channel::unbounded();
        Self { sender, receiver }
    }

    /// Records `event`.
    ///
    /// # Panics
    ///
    /// Never: this handle keeps the inbox open, and it is unbounded.
    pub fn record(&self, event: TaskEvent) {
        self.sender
            .try_send(event)
            .expect("an unbounded inbox this handle keeps open accepts every event");
    }

    /// Takes every event recorded so far, oldest first.
    pub(crate) fn take(&self) -> Vec<TaskEvent> {
        std::iter::from_fn(|| self.receiver.try_recv().ok()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_taken_once_in_order_from_any_clone() {
        let events = TaskEvents::new();
        let host = events.clone();
        host.record(TaskEvent::PullRequestMerged(WorkRef::pull_request(12)));
        host.record(TaskEvent::JobFinished("bg-1".to_string()));
        assert_eq!(
            events.take(),
            vec![
                TaskEvent::PullRequestMerged(WorkRef::pull_request(12)),
                TaskEvent::JobFinished("bg-1".to_string()),
            ]
        );
        assert_eq!(events.take(), Vec::new());
    }

    #[test]
    fn a_finish_ends_only_its_own_start() {
        let started = TaskEvent::SubagentStarted("reviewer".into());
        assert!(TaskEvent::SubagentFinished("reviewer".into()).ends(&started));
        assert!(!TaskEvent::SubagentFinished("other".into()).ends(&started));
        let asked = TaskEvent::QuestionAsked("req-1".into());
        assert!(TaskEvent::QuestionAnswered("req-1".into()).ends(&asked));
        assert!(TaskEvent::QuestionWithdrawn("req-1".into()).ends(&asked));
        assert!(!TaskEvent::QuestionAnswered("req-1".into()).ends(&started));
    }
}
