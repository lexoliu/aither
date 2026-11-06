//! The context source that supervises `tasks.md`.

use std::collections::HashMap;
use std::path::PathBuf;

use askama::Template;

use super::events::{Subject, TaskEvent, TaskEvents};
use super::format::{TasksFile, Violation};
use super::reference::{WorkKind, WorkRef};
use super::tracker::{
    AnyWorkTracker, IssueTrees, WorkItem, WorkState, WorkTrackerError, kind_label,
};
use crate::context::{Block, ContextSource, ExternalContent};
use crate::working_docs::{self, AnyTaskBlockerResolver, WorkingDocsSnapshot};

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
    /// naming its subject.
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
    /// `tasks.md` as it was when the model was shown the event's result.
    baseline: Option<String>,
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
/// issue tree, in full, the first time the conversation sees that root.
///
/// Every alert answers an event, once:
/// - a [`TaskEvent`] the turn that saw it left `tasks.md` unchanged after,
///   naming the rows it affects;
/// - an edit that leaves the document malformed, naming each violation;
/// - an edit that takes it past its line limit.
///
/// Ending the turn is an event too, and the gate it meets: a malformed
/// document, or a row naming a finished issue or pull request or carrying a
/// `BLOCKING` mark that no longer waits, keeps the run going with an alert
/// naming each; otherwise a row under `Doing` or `Queued` that nothing
/// blocks does. A session with nothing to track gets no alert at all.
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
    /// The roots whose trees the conversation holds.
    shown_roots: Vec<WorkRef>,
    /// Events the current turn is judged against.
    pending: Vec<Pending>,
    /// The document as its format was last checked (`None` = absent): an
    /// edit away from it is checked once. The session start sets it without
    /// checking, so a document the session inherits is judged at the gate.
    checked: Option<String>,
    /// Whether the document is past its line limit and was told so.
    over_limit: bool,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("event", &self.event)
            .finish_non_exhaustive()
    }
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

    /// Judges the events of the turn that just ended against `file`: each
    /// one the turn left the document unchanged after is an alert item,
    /// unless no row names its subject and it opens no work.
    ///
    /// Returns the items and, for each, the rows and references they name,
    /// so a stale-row check does not name them again.
    fn judge(
        &mut self,
        docs: &WorkingDocsSnapshot,
        file: &TasksFile,
    ) -> (Vec<AlertItem>, Vec<(usize, WorkRef)>) {
        let mut items = Vec::new();
        let mut named = Vec::new();
        for Pending { event, baseline } in std::mem::take(&mut self.pending) {
            if baseline != docs.tasks_md {
                continue;
            }
            let rows: Vec<_> = match event.subject() {
                Subject::Work(reference) => {
                    let rows: Vec<_> = file.rows_naming(reference).collect();
                    named.extend(rows.iter().map(|row| (row.line(), reference.clone())));
                    rows.iter().map(|row| row.summary().to_string()).collect()
                }
                Subject::Id(id) => file
                    .rows_naming_id(id)
                    .map(|row| row.summary().to_string())
                    .collect(),
            };
            if rows.is_empty() && !event.opens_work() {
                continue;
            }
            items.push(AlertItem::Event { event, rows });
        }
        (items, named)
    }

    /// Whether `mark` still names something the run waits on.
    async fn mark_holds(&self, mark: &str) -> bool {
        match &self.blocker_resolver {
            Some(resolver) => resolver.is_open(mark).await,
            None => true,
        }
    }

    /// The rows of `file` that name a finished item or carry a dead mark,
    /// skipping the row and reference pairs in `named`.
    async fn stale_rows(
        &self,
        file: &TasksFile,
        lookup: &HashMap<WorkRef, Result<WorkItem, WorkTrackerError>>,
        named: &[(usize, WorkRef)],
    ) -> Vec<AlertItem> {
        let mut items = Vec::new();
        for row in file.rows() {
            for reference in row.references() {
                if named
                    .iter()
                    .any(|(line, named)| *line == row.line() && *named == reference)
                {
                    continue;
                }
                if let Some(Ok(item)) = lookup.get(&reference)
                    && item.state.is_done()
                {
                    items.push(AlertItem::Stale {
                        row: row.summary().to_string(),
                        reference,
                        state: item.state,
                    });
                }
            }
            if let Some(mark) = row.blocking_mark()
                && !self.mark_holds(mark).await
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
    /// with a mark that holds nor about an issue blocked by an open one.
    async fn has_actionable_rows(
        &self,
        file: &TasksFile,
        lookup: &HashMap<WorkRef, Result<WorkItem, WorkTrackerError>>,
    ) -> bool {
        for row in file
            .rows()
            .iter()
            .filter(|row| row.section().is_open_work())
        {
            if let Some(mark) = row.blocking_mark()
                && self.mark_holds(mark).await
            {
                continue;
            }
            let blocked = row
                .references()
                .into_iter()
                .find(|reference| reference.kind() == WorkKind::Issue)
                .and_then(|subject| lookup.get(&subject))
                .is_some_and(|item| item.as_ref().is_ok_and(WorkItem::is_blocked));
            if !blocked {
                return true;
            }
        }
        false
    }

    /// Reads every item `file`'s rows name, when there is a tracker.
    async fn lookup(
        &self,
        file: &TasksFile,
    ) -> HashMap<WorkRef, Result<WorkItem, WorkTrackerError>> {
        match &self.tracker {
            Some(tracker) => {
                tracker
                    .lookup(file.rows().iter().flat_map(super::format::Row::references))
                    .await
            }
            None => HashMap::new(),
        }
    }

    /// The trees of `roots`, rendered inside their external-content boundary;
    /// `None` without a tracker or with nothing to show.
    async fn trees(&self, roots: Vec<WorkRef>) -> Option<String> {
        let tracker = self.tracker.as_ref()?;
        if roots.is_empty() {
            return None;
        }
        let trees: IssueTrees = tracker.trees(roots).await;
        if trees.is_empty() {
            return None;
        }
        let rendered = IssueTreesTemplate {
            entries: trees
                .items
                .iter()
                .map(|item| TreeEntry {
                    label: kind_label(&item.reference),
                    item,
                })
                .collect(),
            unresolved: &trees.unresolved,
        }
        .render()
        .expect("issue trees template must render");
        Some(ExternalContent::data("issue tracker", rendered).render_with_notice())
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
        for event in self.events.take() {
            self.pending.push(Pending {
                event,
                baseline: docs.tasks_md.clone(),
            });
        }
        if self.shown.is_none() {
            self.checked.clone_from(&docs.tasks_md);
        }

        let whole = reassembled || self.shown.is_none();
        let roots = parse(&docs).roots();
        let unseen: Vec<_> = if whole {
            roots.clone()
        } else {
            roots
                .iter()
                .filter(|root| !self.shown_roots.contains(root))
                .cloned()
                .collect()
        };
        self.shown_roots = roots;
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
        let (mut items, _) = self.judge(&docs, &file);
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
        let (mut items, named) = self.judge(&docs, &file);
        items.extend(malformed(&file));
        let lookup = self.lookup(&file).await;
        items.extend(self.stale_rows(&file, &lookup, &named).await);
        if let Some(block) = alert(&items, true) {
            return Some(block);
        }
        if !self.has_actionable_rows(&file, &lookup).await {
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
