//! The working document: a file in the agent's sandbox that records the
//! session's work, and the harness that keeps it in the conversation and
//! current.
//!
//! aither knows no document format. The embedder names the file and reads
//! it through its [`DocumentSchema`], into rows, the items they name, what
//! they wait on, the roots and the format's problems
//! ([`DocumentSpec`], set with
//! [`AgentOptions::working_document`](crate::AgentOptions::working_document)).
//! [`WorkEvents`] carry the state changes the rows track; a host's
//! [`WorkTracker`] says what state the issues and pull requests they name
//! are in and what hangs under each root, and its [`BlockerResolver`] what
//! a blocker still waits on. [`WorkingDocument`] puts all of it into the
//! conversation and alerts, once per event, when the document falls behind.
//!
//! Whether the run may end while the document holds open work is the
//! embedder's [`StopHook`](crate::StopHook), which reads the
//! [`DocumentReview`] taken as the model ends its turn.

mod blocker;
mod document;
mod events;
pub(crate) mod file;
mod reference;
mod review;
mod schema;
#[cfg(test)]
pub(crate) mod test_schema;
mod tracker;

pub use blocker::{AnyBlockerResolver, BlockerError, BlockerResolver};
pub use document::{DocumentCheckpoint, WorkingDocument};
pub(crate) use document::{default_alert, default_shown};
pub use events::{WorkEvent, WorkEvents};
pub use reference::{WorkKind, WorkRef, words};
pub use review::{DocumentAlertItem, DocumentReview};
pub use schema::{AnyDocumentSchema, DocumentRow, DocumentSchema, DocumentSpec, ParsedDocument};
pub use tracker::{
    AnyWorkTracker, RelatedWork, WorkComment, WorkItem, WorkState, WorkStatus, WorkTracker,
    WorkTrackerError,
};

/// What a host gives an agent's working document: the document itself,
/// how blockers resolve, where items are read, and the inbox events arrive
/// in.
#[derive(Debug, Clone, Default)]
pub(crate) struct DocumentWiring {
    pub(crate) spec: Option<DocumentSpec>,
    pub(crate) blocker_resolver: Option<AnyBlockerResolver>,
    pub(crate) tracker: Option<AnyWorkTracker>,
    pub(crate) events: WorkEvents,
}

impl DocumentWiring {
    /// The document watched in `sandbox_dir`; `None` when the host keeps
    /// none or the agent has no sandbox.
    pub(crate) fn document(
        &self,
        sandbox_dir: Option<&std::path::Path>,
        notices: crate::AnyNotices,
    ) -> Option<WorkingDocument> {
        let spec = self.spec.as_ref()?;
        let sandbox_dir = sandbox_dir?;
        let mut document =
            WorkingDocument::new(spec, sandbox_dir, self.events.clone()).with_notices(notices);
        if let Some(resolver) = self.blocker_resolver.clone() {
            document = document.with_blocker_resolver(resolver);
        }
        if let Some(tracker) = self.tracker.clone() {
            document = document.with_work_tracker(tracker);
        }
        Some(document)
    }
}
