//! The context source that watches the working document.

use std::collections::HashSet;
use std::path::Path;

use askama::Template;
use futures::StreamExt;

use super::blocker::AnyBlockerResolver;
use super::events::{Subject, WorkEvent, WorkEvents};
use super::file::{self, DocumentFile, Snapshot};
use super::reference::WorkRef;
use super::review::{Blockers, DocumentAlertItem, DocumentReview, Statuses};
use super::schema::{AnyDocumentSchema, DocumentRow, DocumentSpec, ParsedDocument};
use super::tracker::{AnyWorkTracker, WorkItem, WorkTrackerError, kind_label};
use crate::context::{Block, ContextSource, ExternalContent};
use crate::notices::{DocumentAlert, DocumentShown};

/// How many blockers resolve at once.
const RESOLVE_CONCURRENCY: usize = 8;

#[derive(Template)]
#[template(path = "working_document.txt", escape = "none")]
struct WorkingDocumentTemplate<'a> {
    document: &'a str,
    content: Option<&'a str>,
    changes: Option<&'a str>,
    trees: Option<&'a str>,
}

/// aither's wording of the document entering the conversation.
pub fn default_shown(shown: &DocumentShown<'_>) -> String {
    crate::notices::rendered(&WorkingDocumentTemplate {
        document: shown.document,
        content: shown.content,
        changes: shown.changes,
        trees: shown.trees,
    })
}

/// One item of an issue tree, with the kind word the model reads.
struct TreeEntry<'a> {
    label: &'static str,
    item: &'a WorkItem,
}

#[derive(Template)]
#[template(path = "issue_trees.txt", escape = "none")]
struct IssueTreesTemplate<'a> {
    entries: Vec<TreeEntry<'a>>,
    unresolved: &'a [WorkTrackerError],
}

#[derive(Template)]
#[template(path = "document_alert.txt", escape = "none")]
struct DocumentAlertTemplate<'a> {
    document: &'a str,
    items: &'a [DocumentAlertItem],
    held: bool,
}

/// aither's wording of a working-document alert.
pub fn default_alert(alert: &DocumentAlert<'_>) -> String {
    crate::notices::rendered(&DocumentAlertTemplate {
        document: alert.document,
        items: alert.items,
        held: alert.held,
    })
}

/// An event taken at a request, judged when that turn ends.
struct Pending {
    event: WorkEvent,
    /// The document as the conversation held it before the event: an edit
    /// since, the one the event's own tool turn made included, records it.
    baseline: Option<String>,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("event", &self.event)
            .finish_non_exhaustive()
    }
}

/// The events a turn left unrecorded, and the rows they already name, so a
/// stale-row check does not name those again.
#[derive(Default)]
struct Judged {
    items: Vec<DocumentAlertItem>,
    /// Row lines and the references an event named on them.
    named_references: Vec<(usize, WorkRef)>,
    /// Row lines and the ids an event named on them.
    named_ids: Vec<(usize, String)>,
}

/// What the conversation holds of the working document, carried in an
/// agent checkpoint so an agent built on a restored conversation does not
/// show the document and its trees again.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentCheckpoint {
    /// The document as last shown; `None` when it was absent.
    content: Option<String>,
    /// The roots whose trees the conversation holds.
    roots: Vec<WorkRef>,
}

/// Brings the working document into the conversation and holds it to the
/// state its rows track.
///
/// The document is the canonical record of the session's work: it outlives
/// compaction, which the conversation does not. It enters the conversation
/// as durable alerts, never as a block re-rendered into every request:
/// whole at the start of the session and after a reassembly wiped the
/// earlier alerts, and otherwise only as the diff of a change, once. With a
/// [`WorkTracker`](super::WorkTracker), each root its schema reports comes
/// with its issue tree, in full, the first time the conversation sees that
/// root; a tree the backend failed to read is fetched again at the next
/// request.
///
/// After a turn that ran tools, one alert names
/// - each [`WorkEvent`] the turn left the document unchanged after, with
///   the rows it affects; an event that opens work — an issue created
///   outside the roots' trees, a pull request opened, a question asked, a
///   subagent started that has not finished — alerts when no row names it;
/// - each problem the schema finds in an edit, once per edit.
///
/// As the model ends its turn, [`review`](Self::review) reads every item
/// the rows name and every blocker they wait on; the review adds the rows
/// naming a finished item or a blocker that no longer waits, and the
/// document's problems. Whether that keeps the run going is the
/// [`StopHook`](crate::StopHook)'s decision: the document never holds a run
/// by itself. A blocker or item that cannot be read is logged and never
/// taken for unblocked.
#[derive(Debug)]
pub struct WorkingDocument {
    file: DocumentFile,
    schema: AnyDocumentSchema,
    /// Where the state changes rows track arrive.
    events: WorkEvents,
    /// Resolves blockers when the host can say what still waits.
    blocker_resolver: Option<AnyBlockerResolver>,
    /// Reads the issues and pull requests rows name.
    tracker: Option<AnyWorkTracker>,
    /// What the conversation last received of the document; `None` until
    /// the session start has shown it.
    shown: Option<Snapshot>,
    /// The roots whose trees the conversation holds in full.
    shown_roots: Vec<WorkRef>,
    /// The roots whose trees failed to read and were said so; they are
    /// fetched again at every request until they read.
    failed_roots: Vec<WorkRef>,
    /// Events the current turn is judged against.
    pending: Vec<Pending>,
    /// The document as its problems were last checked: an edit away from it
    /// is checked once. The session start sets it without checking, so a
    /// document the session inherits is judged as the turn ends.
    checked: Snapshot,
    /// Renders its alerts.
    notices: crate::AnyNotices,
}

impl WorkingDocument {
    /// Watches the document `spec` describes in `sandbox_dir`, taking the
    /// events recorded on `events`.
    #[must_use]
    pub fn new(spec: &DocumentSpec, sandbox_dir: &Path, events: WorkEvents) -> Self {
        Self {
            file: DocumentFile::new(sandbox_dir, spec.file()),
            schema: spec.schema().clone(),
            events,
            blocker_resolver: None,
            tracker: None,
            shown: None,
            shown_roots: Vec::new(),
            failed_roots: Vec::new(),
            pending: Vec::new(),
            checked: Snapshot::default(),
            notices: crate::AnyNotices::default(),
        }
    }

    /// Renders what it shows and alerts through `notices`;
    /// [`DefaultNotices`](crate::DefaultNotices) by default.
    #[must_use]
    pub fn with_notices(mut self, notices: crate::AnyNotices) -> Self {
        self.notices = notices;
        self
    }

    /// Resolves the rows' blockers through `resolver`: a blocker it reports
    /// dead no longer holds its row, which is then stale. Without it every
    /// blocker holds.
    #[must_use]
    pub fn with_blocker_resolver(mut self, resolver: AnyBlockerResolver) -> Self {
        self.blocker_resolver = Some(resolver);
        self
    }

    /// Reads the issues and pull requests rows name through `tracker`:
    /// their state makes a row stale or blocked, and each root brings its
    /// tree.
    #[must_use]
    pub fn with_work_tracker(mut self, tracker: AnyWorkTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }

    #[cfg(feature = "languages")]
    pub(crate) const fn file(&self) -> &DocumentFile {
        &self.file
    }

    /// What the conversation holds of the document; `None` before the
    /// session start showed it.
    #[must_use]
    pub fn checkpoint(&self) -> Option<DocumentCheckpoint> {
        self.shown.as_ref().map(|shown| DocumentCheckpoint {
            content: shown.0.clone(),
            roots: self.shown_roots.clone(),
        })
    }

    /// Takes `checkpoint` as what a restored conversation holds, so the next
    /// request shows only what changed since; `None` means it holds nothing
    /// and the next request shows the document whole.
    pub fn restore(&mut self, checkpoint: Option<DocumentCheckpoint>) {
        self.failed_roots.clear();
        if let Some(DocumentCheckpoint { content, roots }) = checkpoint {
            self.checked = Snapshot(content.clone());
            self.shown = Some(Snapshot(content));
            self.shown_roots = roots;
        } else {
            self.shown = None;
            self.shown_roots.clear();
        }
    }

    fn parse(&self, content: Option<&str>) -> ParsedDocument {
        content.map_or_else(ParsedDocument::new, |content| self.schema.parse(content))
    }

    /// Takes the events recorded since the last request, judged against
    /// `baseline`. An opening event the same batch already ends leaves
    /// nothing for a row to carry and is dropped.
    fn take_events(&mut self, baseline: Option<&str>) {
        let taken = self.events.take();
        for (index, event) in taken.iter().enumerate() {
            if event.opens_work() && taken.iter().skip(index + 1).any(|later| later.ends(event)) {
                continue;
            }
            self.pending.push(Pending {
                event: event.clone(),
                baseline: baseline.map(str::to_string),
            });
        }
    }

    /// Judges `pending`, the events of the turn that just ended, against
    /// `document`, read from `content`. An event the turn recorded — the
    /// document changed since its baseline — is settled; so is one that
    /// opens work a row already names, or an issue created inside a root's
    /// tree, which needs no row of its own. One that only concerns rows
    /// naming its subject is settled when none does.
    async fn judge(
        &self,
        pending: Vec<Pending>,
        content: Option<&str>,
        document: &ParsedDocument,
    ) -> Judged {
        let mut judged = Judged::default();
        let mut unrecorded: Vec<(WorkEvent, Vec<String>)> = Vec::new();
        for Pending { event, baseline } in pending {
            if baseline.as_deref() != content {
                continue;
            }
            let rows: Vec<&DocumentRow> = match event.subject() {
                Subject::Work(reference) => document.rows_naming(reference).collect(),
                Subject::Id(id) => document.rows_naming_id(id).collect(),
            };
            if event.opens_work() {
                if rows.is_empty() {
                    unrecorded.push((event, Vec::new()));
                }
                continue;
            }
            if rows.is_empty() {
                continue;
            }
            match event.subject() {
                Subject::Work(reference) => judged
                    .named_references
                    .extend(rows.iter().map(|row| (row.line(), reference.clone()))),
                Subject::Id(id) => judged
                    .named_ids
                    .extend(rows.iter().map(|row| (row.line(), id.to_string()))),
            }
            let summaries = rows.iter().map(|row| row.summary().to_string()).collect();
            unrecorded.push((event, summaries));
        }

        let created_unnamed = |event: &WorkEvent, rows: &[String]| {
            rows.is_empty() && matches!(event, WorkEvent::IssueCreated(_))
        };
        let reached = match &self.tracker {
            Some(tracker)
                if !document.roots().is_empty()
                    && unrecorded
                        .iter()
                        .any(|(event, rows)| created_unnamed(event, rows)) =>
            {
                tracker.reached(document.roots().to_vec()).await
            }
            _ => HashSet::new(),
        };
        for (event, rows) in unrecorded {
            if let WorkEvent::IssueCreated(created) = &event
                && rows.is_empty()
                && reached.iter().any(|item| item.matches(created))
            {
                continue;
            }
            judged.items.push(DocumentAlertItem::Event { event, rows });
        }
        judged
    }

    /// Resolves every blocker of `document` once, at most
    /// [`RESOLVE_CONCURRENCY`] at a time; empty without a resolver, which
    /// trusts every blocker.
    async fn resolve_blockers(&self, document: &ParsedDocument) -> Blockers {
        let Some(resolver) = &self.blocker_resolver else {
            return Blockers::new();
        };
        let blockers: HashSet<String> = document
            .rows()
            .iter()
            .filter_map(DocumentRow::blocker)
            .map(str::to_string)
            .collect();
        futures::stream::iter(blockers)
            .map(|blocker| async move {
                let open = resolver.is_open(&blocker).await;
                (blocker, open)
            })
            .buffer_unordered(RESOLVE_CONCURRENCY)
            .collect()
            .await
    }

    /// Reads the state of every item `document`'s rows name, when there is
    /// a tracker.
    async fn statuses(&self, document: &ParsedDocument) -> Statuses {
        match &self.tracker {
            Some(tracker) => {
                tracker
                    .statuses(
                        document
                            .rows()
                            .iter()
                            .flat_map(|row| row.references().iter().cloned()),
                    )
                    .await
            }
            None => Statuses::new(),
        }
    }

    /// The trees of `roots` the conversation has not received, rendered
    /// inside their external-content boundary; `None` without a tracker or
    /// with nothing to show. A tree read in full joins the shown roots; one
    /// the backend failed on is said once and fetched again next time.
    async fn trees(&mut self, roots: Vec<WorkRef>) -> Option<String> {
        let tracker = self.tracker.clone()?;
        if roots.is_empty() {
            return None;
        }
        let mut items: Vec<WorkItem> = Vec::new();
        let mut unresolved = Vec::new();
        for (root, tree) in tracker.trees(roots).await {
            if tree.is_complete() {
                self.failed_roots.retain(|failed| *failed != root);
                self.shown_roots.push(root);
            } else if self.failed_roots.contains(&root) {
                continue;
            } else {
                self.failed_roots.push(root);
            }
            for item in tree.items {
                if !items.iter().any(|known| known.reference == item.reference) {
                    items.push(item);
                }
            }
            unresolved.extend(tree.unresolved);
        }
        if items.is_empty() && unresolved.is_empty() {
            return None;
        }
        let rendered = IssueTreesTemplate {
            entries: items
                .iter()
                .map(|item| TreeEntry {
                    label: kind_label(&item.reference),
                    item,
                })
                .collect(),
            unresolved: &unresolved,
        }
        .render()
        .expect("issue trees template must render");
        Some(ExternalContent::data("issue tracker", rendered).render_with_notice())
    }

    /// Reviews the document as the model ends its turn: judges the turn's
    /// events, reads every item the rows name and resolves every blocker,
    /// concurrently, and collects what the document fell behind on.
    pub async fn review(&mut self) -> DocumentReview {
        let current = self.file.read().await;
        let document = self.parse(current.content());
        let pending = std::mem::take(&mut self.pending);
        let (judged, statuses, blockers) = futures::join!(
            self.judge(pending, current.content(), &document),
            self.statuses(&document),
            self.resolve_blockers(&document),
        );
        log_unreadable(&statuses, &blockers);
        let stale = stale_rows(&document, &statuses, &blockers, &judged);
        let mut findings = judged.items;
        findings.extend(problems(&document));
        findings.extend(stale);
        DocumentReview::new(document, statuses, blockers, findings)
    }

    /// The alert naming `review`'s findings, or nothing when there are
    /// none; `held` when the run goes on after it.
    #[must_use]
    pub fn alert(&self, review: &DocumentReview, held: bool) -> Option<Block> {
        self.alert_items(review.findings(), held)
    }

    fn alert_items(&self, items: &[DocumentAlertItem], held: bool) -> Option<Block> {
        if items.is_empty() {
            return None;
        }
        Some(Block::new(
            "document_alert",
            self.notices.document_alert(&DocumentAlert {
                document: self.file.name(),
                items,
                held,
            }),
        ))
    }
}

/// The rows of `document` that name a finished item or wait on a dead
/// blocker, skipping what `judged` already named on them.
fn stale_rows(
    document: &ParsedDocument,
    statuses: &Statuses,
    blockers: &Blockers,
    judged: &Judged,
) -> Vec<DocumentAlertItem> {
    let mut items = Vec::new();
    for row in document.rows() {
        for reference in row.references() {
            if judged
                .named_references
                .iter()
                .any(|(line, named)| *line == row.line() && named == reference)
            {
                continue;
            }
            if let Some(Ok(status)) = statuses.get(reference)
                && status.state.is_done()
            {
                items.push(DocumentAlertItem::Stale {
                    row: row.summary().to_string(),
                    reference: reference.clone(),
                    state: status.state,
                });
            }
        }
        if let Some(blocker) = row.blocker()
            && matches!(blockers.get(blocker), Some(Ok(false)))
            && !judged
                .named_ids
                .iter()
                .any(|(line, named)| *line == row.line() && named == blocker)
        {
            items.push(DocumentAlertItem::DeadBlocker {
                row: row.summary().to_string(),
                blocker: blocker.to_string(),
            });
        }
    }
    items
}

/// Logs every item and blocker the end of the turn could not read.
fn log_unreadable(statuses: &Statuses, blockers: &Blockers) {
    for (reference, status) in statuses {
        if let Err(error) = status {
            tracing::warn!(
                %reference,
                %error,
                "a working-document item could not be read; its row is taken for neither stale nor unblocked"
            );
        }
    }
    for (blocker, resolved) in blockers {
        if let Err(error) = resolved {
            tracing::warn!(
                blocker,
                %error,
                "a working-document blocker could not be resolved; it keeps holding its row"
            );
        }
    }
}

/// The problems the schema found in `document`, as alert items.
fn problems(document: &ParsedDocument) -> impl Iterator<Item = DocumentAlertItem> + '_ {
    document
        .problems()
        .iter()
        .cloned()
        .map(DocumentAlertItem::Problem)
}

impl ContextSource for WorkingDocument {
    async fn on_request(&mut self, reassembled: bool) -> Option<Block> {
        let current = self.file.read().await;
        // The events arrived since the previous request, while the document
        // was as that request showed it.
        let baseline = self.shown.as_ref().unwrap_or(&current).0.clone();
        self.take_events(baseline.as_deref());
        if self.shown.is_none() {
            self.checked.clone_from(&current);
        }

        let whole = reassembled || self.shown.is_none();
        if whole {
            self.shown_roots.clear();
            self.failed_roots.clear();
        }
        let document = self.parse(current.content());
        let roots = document.roots();
        self.shown_roots.retain(|root| roots.contains(root));
        self.failed_roots.retain(|root| roots.contains(root));
        let unseen: Vec<_> = roots
            .iter()
            .filter(|root| !self.shown_roots.contains(root))
            .cloned()
            .collect();
        let trees = self.trees(unseen).await;

        let changes = match &self.shown {
            Some(shown) if !whole => file::diff(shown, &current),
            _ => None,
        };
        let whole_content = whole.then(|| current.0.clone()).flatten();
        self.shown = Some(current);
        if whole_content.is_none() && changes.is_none() && trees.is_none() {
            return None;
        }
        Some(Block::new(
            "working_document",
            self.notices.working_document(&DocumentShown {
                document: self.file.name(),
                content: whole_content.as_deref(),
                changes: changes.as_deref(),
                trees: trees.as_deref(),
            }),
        ))
    }

    async fn on_tool_turn_end(&mut self) -> Option<Block> {
        let current = self.file.read().await;
        let document = self.parse(current.content());
        let pending = std::mem::take(&mut self.pending);
        let mut items = self
            .judge(pending, current.content(), &document)
            .await
            .items;
        if self.checked != current {
            self.checked = current;
            items.extend(problems(&document));
        }
        self.alert_items(&items, false)
    }
}

#[cfg(test)]
#[path = "document_tests.rs"]
mod tests;
