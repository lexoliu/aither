//! The state of the working document as the model ends its turn.

use std::collections::HashMap;

use super::blocker::BlockerError;
use super::events::WorkEvent;
use super::reference::{WorkKind, WorkRef};
use super::schema::{DocumentRow, ParsedDocument};
use super::tracker::{WorkState, WorkStatus, WorkTrackerError};

/// What every blocker of the document resolved to, each once.
pub type Blockers = HashMap<String, Result<bool, BlockerError>>;

/// The state of every item the document's rows name, each read once.
pub type Statuses = HashMap<WorkRef, Result<WorkStatus, WorkTrackerError>>;

/// One thing a working-document alert says, rendered by
/// [`Notices::document_alert`](crate::Notices::document_alert).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentAlertItem {
    /// An event the turn left the document unchanged after, with the rows
    /// naming its subject; none for an event that opens work no row carries.
    Event {
        /// The event.
        event: WorkEvent,
        /// The summaries of the rows naming its subject.
        rows: Vec<String>,
    },
    /// A row naming a finished issue or pull request.
    Stale {
        /// The row's summary.
        row: String,
        /// The issue or pull request it names.
        reference: WorkRef,
        /// The state that item is in.
        state: WorkState,
    },
    /// A row whose blocker no longer waits on anything.
    DeadBlocker {
        /// The row's summary.
        row: String,
        /// Its blocker.
        blocker: String,
    },
    /// A way the document breaks its format, as the schema rendered it.
    Problem(String),
}

/// The working document as the model ends its turn: its rows, what the
/// items they name and the blockers they wait on resolved to, and what the
/// alert raised at that point says.
///
/// A [`StopHook`](crate::StopHook) reads it from
/// [`TurnBoundaryContext::document`](crate::TurnBoundaryContext::document)
/// to decide whether the run ends.
#[derive(Debug, Clone)]
pub struct DocumentReview {
    document: ParsedDocument,
    statuses: Statuses,
    blockers: Blockers,
    findings: Vec<DocumentAlertItem>,
}

impl DocumentReview {
    pub(crate) const fn new(
        document: ParsedDocument,
        statuses: Statuses,
        blockers: Blockers,
        findings: Vec<DocumentAlertItem>,
    ) -> Self {
        Self {
            document,
            statuses,
            blockers,
            findings,
        }
    }

    /// The document as its schema read it; empty when it is absent.
    #[must_use]
    pub const fn document(&self) -> &ParsedDocument {
        &self.document
    }

    /// What the document fell behind on: events it left unrecorded, rows
    /// naming finished items or dead blockers, and the problems its schema
    /// found. The alert raised as the turn ends says exactly these.
    #[must_use]
    pub fn findings(&self) -> &[DocumentAlertItem] {
        &self.findings
    }

    /// What the tracker said of `reference`; `None` without a tracker or
    /// for an item no row names.
    #[must_use]
    pub fn status(&self, reference: &WorkRef) -> Option<&Result<WorkStatus, WorkTrackerError>> {
        self.statuses.get(reference)
    }

    /// Whether `blocker` still holds its row: anything but the resolver's
    /// answer that it no longer waits, an unresolvable blocker included.
    /// Without a resolver every blocker holds.
    #[must_use]
    pub fn blocker_holds(&self, blocker: &str) -> bool {
        !matches!(self.blockers.get(blocker), Some(Ok(false)))
    }

    /// Whether `row` waits on something: a blocker that holds, or an issue
    /// it names first that is blocked by an open item or could not be read.
    #[must_use]
    pub fn waits(&self, row: &DocumentRow) -> bool {
        if row
            .blocker()
            .is_some_and(|blocker| self.blocker_holds(blocker))
        {
            return true;
        }
        let issue = row
            .references()
            .iter()
            .find(|reference| reference.kind() == WorkKind::Issue);
        match issue.and_then(|issue| self.statuses.get(issue)) {
            None => false,
            Some(Ok(status)) => status.is_blocked(),
            Some(Err(_)) => true,
        }
    }

    /// The open rows that wait on nothing, in document order.
    pub fn actionable_rows(&self) -> impl Iterator<Item = &DocumentRow> {
        self.document
            .rows()
            .iter()
            .filter(|row| row.is_open() && !self.waits(row))
    }
}
