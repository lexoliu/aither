//! The context source that supervises `tasks.md`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use askama::Template;
use futures::StreamExt;

use super::events::{Subject, TaskEvent, TaskEvents};
use super::format::{Row, TasksFile, Violation};
use super::reference::{WorkKind, WorkRef};
use super::tracker::{
    AnyWorkTracker, WorkItem, WorkState, WorkStatus, WorkTrackerError, kind_label,
};
use crate::context::{Block, ContextSource, ExternalContent};
use crate::working_docs::{self, AnyTaskBlockerResolver, TaskBlockerError, WorkingDocsSnapshot};

/// How many `BLOCKING` marks resolve at once.
const RESOLVE_CONCURRENCY: usize = 8;

#[derive(Template)]
#[template(path = "tasks_document.txt", escape = "none")]
struct TasksDocumentTemplate {
    /// The whole document, at session start and after a reassembly.
    tasks: Option<String>,
    /// The diff of a change seen since the document was last shown.
    changes: Option<String>,
    /// The issue trees of the roots the conversation has not seen yet,
    /// inside their external-content boundary.
    trees: Option<String>,
}

/// One item of an issue tree, with the kind word the model reads.
struct TreeEntry<'a> {
    label: &'static str,
    item: &'a WorkItem,
}

#[derive(Template)]
#[template(path = "tasks_issue_trees.txt", escape = "none")]
struct IssueTreesTemplate<'a> {
    entries: Vec<TreeEntry<'a>>,
    unresolved: &'a [WorkTrackerError],
}

/// One thing a tasks alert says.
enum AlertItem {
    /// An event the turn left `tasks.md` unchanged after, with the rows
    /// naming its subject; none for an event that opens work no row carries.
    Event { event: TaskEvent, rows: Vec<String> },
    /// A row naming a finished issue or pull request.
    Stale {
        row: String,
        reference: WorkRef,
        state: WorkState,
    },
    /// A row whose `BLOCKING` mark no longer waits on anything.
    DeadMark { row: String, mark: String },
    /// A line breaking the format.
    Malformed { line: usize, violation: Violation },
    /// The document grew past its line limit.
    OverLimit { lines: usize, limit: usize },
}

#[derive(Template)]
#[template(path = "tasks_alert.txt", escape = "none")]
struct TasksAlertTemplate<'a> {
    items: &'a [AlertItem],
    /// Whether the alert holds the run open at its end.
    gate: bool,
}

#[derive(Template)]
#[template(path = "tasks_continue.txt", escape = "none")]
struct TasksContinueTemplate;

#[derive(Template)]
#[template(path = "system_reminder.txt", escape = "none")]
struct SystemReminderTemplate {
    content: String,
}

/// An event taken at a request, judged when that turn ends.
struct Pending {
    event: TaskEvent,
    /// `tasks.md` as the conversation held it before the event: an edit
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

/// What every `BLOCKING` mark of the document resolved to, each once.
type Marks = HashMap<String, Result<bool, TaskBlockerError>>;

/// The state of every item the document's rows name, each read once.
type Statuses = HashMap<WorkRef, Result<WorkStatus, WorkTrackerError>>;

/// The events a turn left unrecorded, and the rows they already name, so a
/// stale-row check does not name those again.
#[derive(Default)]
struct Judged {
    items: Vec<AlertItem>,
    /// Row lines and the references an event named on them.
    named_references: Vec<(usize, WorkRef)>,
    /// Row lines and the ids an event named on them.
    named_ids: Vec<(usize, String)>,
}

/// What the conversation holds of `tasks.md`, carried in an agent
/// checkpoint so an agent built on a restored conversation does not show the
/// document and its trees again.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TasksCheckpoint {
    /// The document as last shown; `None` when it was absent.
    tasks_md: Option<String>,
    /// The roots whose trees the conversation holds.
    roots: Vec<WorkRef>,
}

/// Brings `tasks.md` into the conversation and holds it to the state it
/// tracks.
///
/// The document is the canonical record of the session's work: it outlives
/// compaction, which the conversation does not. It enters the conversation
/// as durable alerts, never as a block re-rendered into every request: whole
/// at the start of the session and after a reassembly wiped the earlier
/// alerts, and otherwise only as the diff of a change, once. With a
/// [`WorkTracker`](super::WorkTracker), each root it lists comes with its
/// issue tree, in full, the first time the conversation sees that root; a
/// tree the backend failed to read is fetched again at the next request.
///
/// Every alert answers an event, once:
/// - a [`TaskEvent`] the turn that saw it left `tasks.md` unchanged after,
///   naming the rows it affects; an event that opens work — an issue
///   created outside the roots' trees, a pull request opened, a question
///   asked, a subagent started that has not finished — alerts when no row
///   names it;
/// - an edit that leaves the document malformed, naming each violation;
/// - an edit that takes it past its line limit.
///
/// Ending the turn is an event too, and the gate it meets: a malformed
/// document, or a row naming a finished issue or pull request or carrying a
/// `BLOCKING` mark that no longer waits, keeps the run going with an alert
/// naming each; otherwise a row under `Doing` or `Queued` that nothing
/// blocks does. A mark or item that cannot be read is logged and never
/// taken for unblocked. A session with nothing to track gets no alert at
/// all.
#[derive(Debug)]
pub struct TasksDocument {
    /// Sandbox root holding `tasks.md`.
    sandbox_dir: PathBuf,
    /// Lines past which the document must be rewritten.
    line_limit: usize,
    /// Where the state changes rows track arrive.
    events: TaskEvents,
    /// Resolves `BLOCKING` marks when the host can say what still waits.
    blocker_resolver: Option<AnyTaskBlockerResolver>,
    /// Reads the issues and pull requests rows name.
    tracker: Option<AnyWorkTracker>,
    /// What the conversation last received of the document; `None` until the
    /// session start has shown it.
    shown: Option<WorkingDocsSnapshot>,
    /// The roots whose trees the conversation holds in full.
    shown_roots: Vec<WorkRef>,
    /// The roots whose trees failed to read and were said so; they are
    /// fetched again at every request until they read.
    failed_roots: Vec<WorkRef>,
    /// Events the current turn is judged against.
    pending: Vec<Pending>,
    /// The document as its format was last checked (`None` = absent): an
    /// edit away from it is checked once. The session start sets it without
    /// checking, so a document the session inherits is judged at the gate.
    checked: Option<String>,
    /// Whether the document is past its line limit and was told so.
    over_limit: bool,
}

impl TasksDocument {
    /// Watches `tasks.md` in `sandbox_dir`, taking the events recorded on
    /// `events` and asking for a rewrite past `line_limit` lines.
    #[must_use]
    pub fn new(sandbox_dir: impl Into<PathBuf>, line_limit: usize, events: TaskEvents) -> Self {
        Self {
            sandbox_dir: sandbox_dir.into(),
            line_limit,
            events,
            blocker_resolver: None,
            tracker: None,
            shown: None,
            shown_roots: Vec::new(),
            failed_roots: Vec::new(),
            pending: Vec::new(),
            checked: None,
            over_limit: false,
        }
    }

    /// Resolves `BLOCKING` mark payloads through `resolver`: a mark it
    /// reports dead no longer holds its row, which is then stale. Without it
    /// every mark is trusted.
    #[must_use]
    pub fn with_blocker_resolver(mut self, resolver: AnyTaskBlockerResolver) -> Self {
        self.blocker_resolver = Some(resolver);
        self
    }

    /// Reads the issues and pull requests rows name through `tracker`: their
    /// state makes a row stale or blocked, and each root brings its tree.
    #[must_use]
    pub fn with_work_tracker(mut self, tracker: AnyWorkTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }

    /// What the conversation holds of the document; `None` before the
    /// session start showed it.
    #[must_use]
    pub fn checkpoint(&self) -> Option<TasksCheckpoint> {
        self.shown.as_ref().map(|shown| TasksCheckpoint {
            tasks_md: shown.tasks_md.clone(),
            roots: self.shown_roots.clone(),
        })
    }

    /// Takes `checkpoint` as what a restored conversation holds, so the next
    /// request shows only what changed since; `None` means it holds nothing
    /// and the next request shows the document whole.
    pub fn restore(&mut self, checkpoint: Option<TasksCheckpoint>) {
        self.failed_roots.clear();
        if let Some(TasksCheckpoint { tasks_md, roots }) = checkpoint {
            self.checked.clone_from(&tasks_md);
            self.shown = Some(WorkingDocsSnapshot { tasks_md });
            self.shown_roots = roots;
        } else {
            self.shown = None;
            self.shown_roots.clear();
        }
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
    /// `file`. An event the turn recorded — `tasks.md` changed since its
    /// baseline — is settled; so is one that opens work a row already names,
    /// or an issue created inside a root's tree, whose rows the format does
    /// not list. One that only concerns rows naming its subject is settled
    /// when none does.
    async fn judge(
        &self,
        pending: Vec<Pending>,
        docs: &WorkingDocsSnapshot,
        file: &TasksFile,
    ) -> Judged {
        let mut judged = Judged::default();
        let mut unrecorded: Vec<(TaskEvent, Vec<String>)> = Vec::new();
        for Pending { event, baseline } in pending {
            if baseline != docs.tasks_md {
                continue;
            }
            let rows: Vec<&Row> = match event.subject() {
                Subject::Work(reference) => file.rows_naming(reference).collect(),
                Subject::Id(id) => file.rows_naming_id(id).collect(),
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

        let created_unnamed = |event: &TaskEvent, rows: &[String]| {
            rows.is_empty() && matches!(event, TaskEvent::IssueCreated(_))
        };
        let reached = match &self.tracker {
            Some(tracker)
                if unrecorded
                    .iter()
                    .any(|(event, rows)| created_unnamed(event, rows)) =>
            {
                let roots = file.roots();
                if roots.is_empty() {
                    HashSet::new()
                } else {
                    tracker.reached(roots).await
                }
            }
            _ => HashSet::new(),
        };
        for (event, rows) in unrecorded {
            if let TaskEvent::IssueCreated(created) = &event
                && rows.is_empty()
                && reached.iter().any(|item| item.matches(created))
            {
                continue;
            }
            judged.items.push(AlertItem::Event { event, rows });
        }
        judged
    }

    /// Resolves every `BLOCKING` mark of `file` once, at most
    /// [`RESOLVE_CONCURRENCY`] at a time; empty without a resolver, which
    /// trusts every mark.
    async fn resolve_marks(&self, file: &TasksFile) -> Marks {
        let Some(resolver) = &self.blocker_resolver else {
            return Marks::new();
        };
        let marks: HashSet<String> = file
            .rows()
            .iter()
            .filter_map(Row::blocking_mark)
            .map(str::to_string)
            .collect();
        futures::stream::iter(marks)
            .map(|mark| async move {
                let open = resolver.is_open(&mark).await;
                (mark, open)
            })
            .buffer_unordered(RESOLVE_CONCURRENCY)
            .collect()
            .await
    }

    /// Reads the state of every item `file`'s rows name, when there is a
    /// tracker.
    async fn statuses(&self, file: &TasksFile) -> Statuses {
        match &self.tracker {
            Some(tracker) => {
                tracker
                    .statuses(file.rows().iter().flat_map(Row::references))
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
}

/// Whether `mark` still holds its row: anything but a resolver's answer
/// that it no longer waits, an unresolvable mark included.
fn mark_holds(marks: &Marks, mark: &str) -> bool {
    !matches!(marks.get(mark), Some(Ok(false)))
}

/// The rows of `file` that name a finished item or carry a dead mark,
/// skipping what `judged` already named on them.
fn stale_rows(
    file: &TasksFile,
    statuses: &Statuses,
    marks: &Marks,
    judged: &Judged,
) -> Vec<AlertItem> {
    let mut items = Vec::new();
    for row in file.rows() {
        for reference in row.references() {
            if judged
                .named_references
                .iter()
                .any(|(line, named)| *line == row.line() && *named == reference)
            {
                continue;
            }
            if let Some(Ok(status)) = statuses.get(&reference)
                && status.state.is_done()
            {
                items.push(AlertItem::Stale {
                    row: row.summary().to_string(),
                    reference,
                    state: status.state,
                });
            }
        }
        if let Some(mark) = row.blocking_mark()
            && !mark_holds(marks, mark)
            && !judged
                .named_ids
                .iter()
                .any(|(line, named)| *line == row.line() && named == mark)
        {
            items.push(AlertItem::DeadMark {
                row: row.summary().to_string(),
                mark: mark.to_string(),
            });
        }
    }
    items
}

/// Whether a row under `Doing` or `Queued` is neither marked `BLOCKING`
/// with a mark that holds nor about an issue blocked by an open one. An
/// issue that could not be read is never taken for unblocked.
fn has_actionable_rows(file: &TasksFile, statuses: &Statuses, marks: &Marks) -> bool {
    file.rows()
        .iter()
        .filter(|row| row.section().is_open_work())
        .any(|row| {
            if row
                .blocking_mark()
                .is_some_and(|mark| mark_holds(marks, mark))
            {
                return false;
            }
            let subject = row
                .references()
                .into_iter()
                .find(|reference| reference.kind() == WorkKind::Issue);
            match subject.and_then(|subject| statuses.get(&subject)) {
                None => true,
                Some(Ok(status)) => !status.is_blocked(),
                Some(Err(_)) => false,
            }
        })
}

/// Logs every item and mark the end of the turn could not read.
fn log_unreadable(statuses: &Statuses, marks: &Marks) {
    for (reference, status) in statuses {
        if let Err(error) = status {
            tracing::warn!(
                %reference,
                %error,
                "a tasks.md item could not be read; its row is taken for neither stale nor unblocked"
            );
        }
    }
    for (mark, resolved) in marks {
        if let Err(error) = resolved {
            tracing::warn!(
                mark,
                %error,
                "a tasks.md BLOCKING mark could not be resolved; it keeps holding its row"
            );
        }
    }
}

/// Renders `items` as one alert, or nothing when there are none.
fn alert(items: &[AlertItem], gate: bool) -> Option<Block> {
    if items.is_empty() {
        return None;
    }
    Some(Block::new(
        "tasks_alert",
        TasksAlertTemplate { items, gate }
            .render()
            .expect("tasks alert template must render"),
    ))
}

/// The format violations of `file`, as alert items.
fn malformed(file: &TasksFile) -> impl Iterator<Item = AlertItem> + '_ {
    file.violations()
        .iter()
        .map(|violation| AlertItem::Malformed {
            line: violation.line,
            violation: violation.violation.clone(),
        })
}

fn parse(docs: &WorkingDocsSnapshot) -> TasksFile {
    docs.tasks_md
        .as_deref()
        .map(TasksFile::parse)
        .unwrap_or_default()
}

impl ContextSource for TasksDocument {
    async fn on_request(&mut self, reassembled: bool) -> Option<Block> {
        let docs = working_docs::read_snapshot(&self.sandbox_dir).await;
        // The events arrived since the previous request, while the document
        // was as that request showed it.
        let baseline = self
            .shown
            .as_ref()
            .map_or_else(|| docs.tasks_md.clone(), |shown| shown.tasks_md.clone());
        self.take_events(baseline.as_deref());
        if self.shown.is_none() {
            self.checked.clone_from(&docs.tasks_md);
        }

        let whole = reassembled || self.shown.is_none();
        if whole {
            self.shown_roots.clear();
            self.failed_roots.clear();
        }
        let roots = parse(&docs).roots();
        self.shown_roots.retain(|root| roots.contains(root));
        self.failed_roots.retain(|root| roots.contains(root));
        let unseen: Vec<_> = roots
            .into_iter()
            .filter(|root| !self.shown_roots.contains(root))
            .collect();
        let trees = self.trees(unseen).await;

        let (tasks, changes) = match self.shown.as_ref() {
            Some(shown) if !whole => (None, working_docs::tasks_md_diff(shown, &docs)),
            _ => (docs.tasks_md.clone(), None),
        };
        self.shown = Some(docs);
        if tasks.is_none() && changes.is_none() && trees.is_none() {
            return None;
        }
        Some(Block::new(
            "tasks_document",
            TasksDocumentTemplate {
                tasks,
                changes,
                trees,
            }
            .render()
            .expect("tasks document template must render"),
        ))
    }

    async fn on_tool_turn_end(&mut self) -> Option<Block> {
        let docs = working_docs::read_snapshot(&self.sandbox_dir).await;
        let file = parse(&docs);
        let pending = std::mem::take(&mut self.pending);
        let mut items = self.judge(pending, &docs, &file).await.items;
        if self.checked != docs.tasks_md {
            self.checked = docs.tasks_md;
            items.extend(malformed(&file));
        }
        if file.lines() > self.line_limit {
            if !self.over_limit {
                self.over_limit = true;
                items.push(AlertItem::OverLimit {
                    lines: file.lines(),
                    limit: self.line_limit,
                });
            }
        } else {
            self.over_limit = false;
        }
        alert(&items, false)
    }

    async fn on_final_turn(&mut self) -> Option<Block> {
        let docs = working_docs::read_snapshot(&self.sandbox_dir).await;
        let file = parse(&docs);
        let pending = std::mem::take(&mut self.pending);
        let (judged, statuses, marks) = futures::join!(
            self.judge(pending, &docs, &file),
            self.statuses(&file),
            self.resolve_marks(&file),
        );
        log_unreadable(&statuses, &marks);
        let stale = stale_rows(&file, &statuses, &marks, &judged);
        let mut items = judged.items;
        items.extend(malformed(&file));
        items.extend(stale);
        if let Some(block) = alert(&items, true) {
            return Some(block);
        }
        if !has_actionable_rows(&file, &statuses, &marks) {
            return None;
        }
        Some(Block::new(
            "system_reminder",
            SystemReminderTemplate {
                content: TasksContinueTemplate
                    .render()
                    .expect("tasks continue template must render"),
            }
            .render()
            .expect("system reminder template must render"),
        ))
    }
}

#[cfg(test)]
#[path = "document_tests.rs"]
mod tests;
