//! How the embedder reads its working document.

use std::fmt;
use std::sync::Arc;

use super::reference::{WorkRef, words};

/// Reads a working document into what the harness tracks of it.
///
/// The harness knows no document format. The embedder parses its own and
/// says which rows it holds, which of them are open work, what each names
/// and waits on, which items are the roots whose trees the conversation
/// receives, and how the document breaks its format.
pub trait DocumentSchema: Send + Sync + 'static {
    /// Reads `content`, the whole document.
    fn parse(&self, content: &str) -> ParsedDocument;
}

/// A shared [`DocumentSchema`].
#[derive(Clone)]
pub struct AnyDocumentSchema(Arc<dyn DocumentSchema>);

impl AnyDocumentSchema {
    /// Shares `schema`.
    #[must_use]
    pub fn new(schema: impl DocumentSchema) -> Self {
        Self(Arc::new(schema))
    }

    /// See [`DocumentSchema::parse`].
    #[must_use]
    pub fn parse(&self, content: &str) -> ParsedDocument {
        self.0.parse(content)
    }
}

impl fmt::Debug for AnyDocumentSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyDocumentSchema").finish_non_exhaustive()
    }
}

/// The working document an agent keeps: the file, relative to the agent's
/// sandbox, and the schema that reads it.
///
/// Every agent with a sandbox keeps its own, a subagent's in its own space.
#[derive(Debug, Clone)]
pub struct DocumentSpec {
    file: String,
    schema: AnyDocumentSchema,
}

impl DocumentSpec {
    /// The document `file`, read by `schema`. The model knows the document
    /// by `file` as written (`tasks.md`).
    #[must_use]
    pub fn new(file: impl Into<String>, schema: impl DocumentSchema) -> Self {
        Self {
            file: file.into(),
            schema: AnyDocumentSchema::new(schema),
        }
    }

    /// The file, relative to the agent's sandbox.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// The schema that reads it.
    #[must_use]
    pub const fn schema(&self) -> &AnyDocumentSchema {
        &self.schema
    }
}

/// One row of a working document: an entry the events of the session are
/// matched against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRow {
    line: usize,
    text: String,
    open: bool,
    references: Vec<WorkRef>,
    blocker: Option<String>,
}

impl DocumentRow {
    /// The row starting on `line` (from 1) with `text`, continuation lines
    /// joined by newlines. It is no open work, names no item and waits on
    /// nothing until the builder methods say so.
    #[must_use]
    pub fn new(line: usize, text: impl Into<String>) -> Self {
        Self {
            line,
            text: text.into(),
            open: false,
            references: Vec::new(),
            blocker: None,
        }
    }

    /// Marks it open work: what a stop hook may hold the run for, reported
    /// by [`DocumentReview::actionable_rows`](super::DocumentReview::actionable_rows)
    /// unless something blocks it.
    #[must_use]
    pub const fn open(mut self) -> Self {
        self.open = true;
        self
    }

    /// The issues and pull requests it names, in order: an event about one
    /// concerns this row, and the tracker's state for each makes it stale or
    /// blocked.
    #[must_use]
    pub fn naming(mut self, references: impl IntoIterator<Item = WorkRef>) -> Self {
        self.references = references.into_iter().collect();
        self
    }

    /// What it waits on: a request id or a free-text reason, resolved
    /// through the host's [`BlockerResolver`](super::BlockerResolver).
    #[must_use]
    pub fn blocked_on(mut self, blocker: impl Into<String>) -> Self {
        self.blocker = Some(blocker.into());
        self
    }

    /// Its first line's number, from 1.
    #[must_use]
    pub const fn line(&self) -> usize {
        self.line
    }

    /// Its text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The first line of its text, which names it in an alert.
    #[must_use]
    pub fn summary(&self) -> &str {
        self.text.lines().next().unwrap_or_default()
    }

    /// Whether it is open work.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.open
    }

    /// The issues and pull requests it names.
    #[must_use]
    pub fn references(&self) -> &[WorkRef] {
        &self.references
    }

    /// What it waits on, if anything.
    #[must_use]
    pub fn blocker(&self) -> Option<&str> {
        self.blocker.as_deref()
    }

    /// Whether its text names `id` — a question, job or subagent id — as a
    /// word, split by [`words`].
    #[must_use]
    pub fn names(&self, id: &str) -> bool {
        words(&self.text).any(|word| word == id)
    }

    /// Whether it names `reference`, or an item `reference` can be.
    fn names_reference(&self, reference: &WorkRef) -> bool {
        self.references.iter().any(|named| named.matches(reference))
    }
}

/// A working document as its schema read it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedDocument {
    rows: Vec<DocumentRow>,
    roots: Vec<WorkRef>,
    problems: Vec<String>,
}

impl ParsedDocument {
    /// A document with nothing in it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `row`, after the rows added before it.
    pub fn push_row(&mut self, row: DocumentRow) {
        self.rows.push(row);
    }

    /// Adds `root`, an item the session serves: its issue tree enters the
    /// conversation the first time the document lists it, and an issue
    /// created anywhere in that tree needs no row of its own. A root added
    /// twice is kept once.
    pub fn push_root(&mut self, root: WorkRef) {
        if !self.roots.contains(&root) {
            self.roots.push(root);
        }
    }

    /// Adds `problem`, a way the document breaks its format, rendered as
    /// the sentence the model reads. Every problem is alerted after an edit
    /// that leaves it, and again as the turn ends.
    pub fn push_problem(&mut self, problem: impl Into<String>) {
        self.problems.push(problem.into());
    }

    /// Every row, in the order added.
    #[must_use]
    pub fn rows(&self) -> &[DocumentRow] {
        &self.rows
    }

    /// The roots, each once, in the order added.
    #[must_use]
    pub fn roots(&self) -> &[WorkRef] {
        &self.roots
    }

    /// Every problem, in the order added.
    #[must_use]
    pub fn problems(&self) -> &[String] {
        &self.problems
    }

    /// The rows that name `reference`.
    pub(crate) fn rows_naming<'a>(
        &'a self,
        reference: &'a WorkRef,
    ) -> impl Iterator<Item = &'a DocumentRow> {
        self.rows
            .iter()
            .filter(move |row| row.names_reference(reference))
    }

    /// The rows that name `id` as a word.
    pub(crate) fn rows_naming_id<'a>(
        &'a self,
        id: &'a str,
    ) -> impl Iterator<Item = &'a DocumentRow> {
        self.rows.iter().filter(move |row| row.names(id))
    }
}
