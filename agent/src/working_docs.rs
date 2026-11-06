//! Working document utilities for long-task execution.
//!
//! `tasks.md` is a regular markdown file in the sandbox that the framework
//! treats specially by loading into context and supervising progress.

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_fs as fs;
use similar::{ChangeTag, TextDiff};

/// Decides whether a `BLOCKING <what>` mark still waits on something live.
///
/// The host resolves the mark's payload token: `true` while it names
/// something still open — an unanswered `ask_user` request id, or a
/// free-text reason the host cannot check. `false` means the mark is dead —
/// its question was answered or withdrawn — and its row is stale. An error
/// says nothing either way: the mark keeps holding its row.
pub trait TaskBlockerResolver: Send + Sync + 'static {
    /// Whether `mark` still names something the run waits on.
    fn is_open(&self, mark: &str) -> impl Future<Output = Result<bool, TaskBlockerError>> + Send;
}

/// Why a `BLOCKING` mark could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TaskBlockerError {
    /// The host's record of what the mark names could not be read.
    #[error("the mark {mark} cannot be resolved: {reason}")]
    Unavailable {
        /// The mark's payload.
        mark: String,
        /// What went wrong.
        reason: String,
    },
}

type IsOpen<'a> = Pin<Box<dyn Future<Output = Result<bool, TaskBlockerError>> + Send + 'a>>;

/// Object-safe twin of [`TaskBlockerResolver`]; boxing lives inside
/// [`AnyTaskBlockerResolver`].
trait TaskBlockerResolverImpl: Send + Sync {
    fn is_open<'a>(&'a self, mark: &'a str) -> IsOpen<'a>;
}

impl<T: TaskBlockerResolver> TaskBlockerResolverImpl for T {
    fn is_open<'a>(&'a self, mark: &'a str) -> IsOpen<'a> {
        Box::pin(TaskBlockerResolver::is_open(self, mark))
    }
}

/// A type-erased [`TaskBlockerResolver`] carried by the agent.
#[derive(Clone)]
pub struct AnyTaskBlockerResolver(Arc<dyn TaskBlockerResolverImpl>);

impl AnyTaskBlockerResolver {
    /// Wraps `resolver`.
    #[must_use]
    pub fn new(resolver: impl TaskBlockerResolver) -> Self {
        Self(Arc::new(resolver))
    }

    /// Whether `mark` still names something the run waits on.
    pub fn is_open<'a>(
        &'a self,
        mark: &'a str,
    ) -> impl Future<Output = Result<bool, TaskBlockerError>> + Send + 'a {
        self.0.is_open(mark)
    }
}

impl std::fmt::Debug for AnyTaskBlockerResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyTaskBlockerResolver")
            .finish_non_exhaustive()
    }
}

/// Snapshot of working documents currently present in the sandbox.
#[derive(Debug, Clone, Default)]
pub struct WorkingDocsSnapshot {
    /// Contents of `tasks.md`, when present.
    pub tasks_md: Option<String>,
}

/// Reads `tasks.md` from the given sandbox directory.
pub async fn read_snapshot(sandbox_dir: &Path) -> WorkingDocsSnapshot {
    let tasks_path = sandbox_dir.join("tasks.md");

    let tasks_md = read_file_if_exists(&tasks_path).await;

    WorkingDocsSnapshot { tasks_md }
}

/// Returns a compact changed-line diff for `tasks.md` when it changed.
#[must_use]
pub fn tasks_md_diff(
    previous: &WorkingDocsSnapshot,
    current: &WorkingDocsSnapshot,
) -> Option<String> {
    if previous.tasks_md == current.tasks_md {
        return None;
    }

    let diff = TextDiff::from_lines(
        previous.tasks_md.as_deref().unwrap_or_default(),
        current.tasks_md.as_deref().unwrap_or_default(),
    );

    let mut lines = Vec::new();
    for change in diff.iter_all_changes() {
        let prefix = match change.tag() {
            ChangeTag::Delete => "- ",
            ChangeTag::Insert => "+ ",
            ChangeTag::Equal => continue,
        };
        let value = change.value().trim_end_matches('\n');
        if value.is_empty() {
            continue;
        }
        let mut line = String::from(prefix);
        line.push_str(value);
        lines.push(line);
        if lines.len() == 48 {
            lines.push("...".to_string());
            break;
        }
    }

    if lines.is_empty() {
        return Some("tasks.md changed.".to_string());
    }

    Some(lines.join("\n"))
}

/// The lines a change added to `tasks.md`, or `None` when it added none.
#[cfg(feature = "languages")]
#[must_use]
pub fn tasks_md_added(
    previous: &WorkingDocsSnapshot,
    current: &WorkingDocsSnapshot,
) -> Option<String> {
    if previous.tasks_md == current.tasks_md {
        return None;
    }
    let diff = TextDiff::from_lines(
        previous.tasks_md.as_deref().unwrap_or_default(),
        current.tasks_md.as_deref().unwrap_or_default(),
    );
    let added: String = diff
        .iter_all_changes()
        .filter(|change| change.tag() == ChangeTag::Insert)
        .map(|change| change.value())
        .collect();
    (!added.trim().is_empty()).then_some(added)
}

async fn read_file_if_exists(path: &Path) -> Option<String> {
    match fs::read_to_string(path).await {
        Ok(content) => Some(content),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            tracing::warn!(path = %path.display(), error = %err, "failed to read working doc");
            None
        }
    }
}
