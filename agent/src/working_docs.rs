//! Working document utilities for long-task execution.
//!
//! TODO.md and PLAN.md are regular markdown files in the sandbox that the
//! framework treats specially by loading into context and supervising progress.

use std::path::Path;

use async_fs as fs;

/// Snapshot of working documents currently present in the sandbox.
#[derive(Debug, Clone, Default)]
pub struct WorkingDocsSnapshot {
    pub todo_md: Option<String>,
    pub plan_md: Option<String>,
}

impl WorkingDocsSnapshot {
    /// Returns true when either TODO.md or PLAN.md has unchecked markdown tasks.
    #[must_use]
    pub fn has_unchecked_items(&self) -> bool {
        self.todo_md
            .as_deref()
            .is_some_and(has_unchecked_markdown_tasks)
            || self
                .plan_md
                .as_deref()
                .is_some_and(has_unchecked_markdown_tasks)
    }
}

/// Reads TODO.md and PLAN.md from the given sandbox directory.
pub async fn read_snapshot(sandbox_dir: &Path) -> WorkingDocsSnapshot {
    let todo_path = sandbox_dir.join("TODO.md");
    let plan_path = sandbox_dir.join("PLAN.md");

    let todo_md = read_file_if_exists(&todo_path).await;
    let plan_md = read_file_if_exists(&plan_path).await;

    WorkingDocsSnapshot { todo_md, plan_md }
}

/// Detects unfinished markdown checklist items (`- [ ]`).
#[must_use]
pub fn has_unchecked_markdown_tasks(content: &str) -> bool {
    content.lines().any(is_unchecked_todo_line)
}

fn is_unchecked_todo_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    (trimmed.starts_with("- [ ]") || trimmed.starts_with("* [ ]"))
        || (trimmed.starts_with("- [	]") || trimmed.starts_with("* [	]"))
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
