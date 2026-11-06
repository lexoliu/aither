//! `tasks.md`: the session's durable record of its work, and the harness
//! that keeps it current.
//!
//! The [`format`] fixes what the document holds; [`TaskEvents`] carry the
//! state changes its rows track; a host's [`WorkTracker`] says what state
//! the issues and pull requests it names are in and what hangs under each
//! root; [`TasksDocument`] puts all of it into the conversation and alerts,
//! once per event, when the document falls behind.

mod document;
mod events;
pub mod format;
mod reference;
mod tracker;

pub use document::TasksDocument;
pub use events::{TaskEvent, TaskEvents};
pub use format::{LineViolation, Row, Section, TasksFile, Violation};
pub use reference::{WorkKind, WorkRef};
pub use tracker::{
    AnyWorkTracker, RelatedWork, WorkComment, WorkItem, WorkState, WorkTracker, WorkTrackerError,
};

use crate::working_docs::AnyTaskBlockerResolver;

/// What a host gives an agent's tasks document: how marks resolve, where
/// items are read, and the inbox events arrive in.
#[derive(Debug, Clone, Default)]
pub(crate) struct TasksWiring {
    pub(crate) blocker_resolver: Option<AnyTaskBlockerResolver>,
    pub(crate) tracker: Option<AnyWorkTracker>,
    pub(crate) events: TaskEvents,
}

impl TasksWiring {
    /// The document watching `tasks.md` in `sandbox_dir`.
    pub(crate) fn document(
        &self,
        sandbox_dir: std::path::PathBuf,
        line_limit: usize,
    ) -> TasksDocument {
        let mut document = TasksDocument::new(sandbox_dir, line_limit, self.events.clone());
        if let Some(resolver) = self.blocker_resolver.clone() {
            document = document.with_blocker_resolver(resolver);
        }
        if let Some(tracker) = self.tracker.clone() {
            document = document.with_work_tracker(tracker);
        }
        document
    }
}
