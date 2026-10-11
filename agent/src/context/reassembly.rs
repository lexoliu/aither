//! Restoring working material after a compaction.
//!
//! A handoff document names the files the work depends on, but naming them is
//! not enough: the instance that reads the handoff has to spend its first
//! turns re-reading them one by one, and until it does it is reasoning about
//! code it cannot see. The restoration here front-runs that — the files the
//! handoff listed are read concurrently and placed in the window as a single
//! `<restored_context>` block, so the fresh instance starts with the same
//! material the compacted one had.
//!
//! The budget is what keeps this from undoing the compaction it follows.
//! [`ContextPolicy::restore_budget`](super::ContextPolicy::restore_budget)
//! caps the block as a fraction of the window; files are taken in the order
//! the handoff listed them, the first one that does not fit whole is
//! truncated, and the rest are reported as omitted rather than dropped
//! silently so the model knows to read them on demand.

use std::path::{Path, PathBuf};

use askama::Template;
use futures::StreamExt as _;

use super::compaction::HandoffDocument;
use super::source::Block;

/// How many files are read from disk at once.
///
/// Restoration happens at a request boundary with the user waiting, so the
/// reads are overlapped; the bound keeps a handoff that lists a hundred paths
/// from opening a hundred file descriptors at once.
const CONCURRENT_READS: usize = 8;

/// A file re-read for the restored context block.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RestoredFile {
    /// Path as the handoff named it.
    path: String,
    /// Contents, possibly truncated to fit the budget.
    content: String,
    /// Whether `content` is a prefix of the file rather than all of it.
    truncated: bool,
}

#[derive(Template)]
#[template(path = "restored_context.md", escape = "none")]
struct RestoredContextTemplate<'a> {
    files: &'a [RestoredFile],
    omitted: usize,
}

/// Re-reads the files a handoff named, capped at `budget_chars`.
///
/// Paths are resolved relative to `root` when they are not absolute. Missing
/// and unreadable paths are skipped: a handoff routinely names a file that
/// has since been moved, and that is not a reason to fail the restoration.
///
/// Returns `None` when nothing could be restored.
///
/// # Panics
///
/// Panics if the compiled-in restoration template fails to render, which
/// would mean the binary was built against a broken template.
pub async fn restore_context(
    handoff: &HandoffDocument,
    root: &Path,
    budget_chars: usize,
) -> Option<Block> {
    if budget_chars == 0 {
        return None;
    }

    let candidates = restoration_candidates(handoff);
    if candidates.is_empty() {
        return None;
    }

    let contents = read_all(root, &candidates).await;

    let mut files: Vec<RestoredFile> = Vec::new();
    let mut omitted = 0usize;
    let mut used = 0usize;
    for (path, content) in contents {
        let Some(content) = content else {
            continue;
        };
        let remaining = budget_chars.saturating_sub(used);
        if remaining <= path.len() {
            omitted += 1;
            continue;
        }
        let allowance = remaining - path.len();
        let (content, truncated) = if content.len() <= allowance {
            (content, false)
        } else {
            (truncate_on_char_boundary(&content, allowance), true)
        };
        used += path.len() + content.len();
        files.push(RestoredFile {
            path,
            content,
            truncated,
        });
    }

    if files.is_empty() {
        return None;
    }

    let rendered = RestoredContextTemplate {
        files: &files,
        omitted,
    }
    .render()
    .expect("restored context template must render");
    Some(Block::new("restored_context", rendered))
}

/// The paths worth re-reading, in handoff order and without duplicates.
///
/// `file_index` entries are rendered as `path — description`, so only the
/// path part is usable here.
fn restoration_candidates(handoff: &HandoffDocument) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let indexed = handoff
        .file_index
        .iter()
        .filter_map(|entry| entry.split(" — ").next())
        .map(str::to_string);
    handoff
        .files_paths
        .iter()
        .cloned()
        .chain(indexed)
        .filter(|path| {
            let path = path.trim();
            !path.is_empty() && !path.ends_with('/') && seen.insert(path.to_string())
        })
        .collect()
}

/// Reads every candidate concurrently, preserving the requested order.
async fn read_all(root: &Path, candidates: &[String]) -> Vec<(String, Option<String>)> {
    futures::stream::iter(candidates.to_vec())
        .map(|path| async move {
            let resolved = resolve(root, &path);
            let content = async_fs::read_to_string(&resolved).await.ok();
            if content.is_none() {
                tracing::debug!(path = %resolved.display(), "handoff file could not be restored");
            }
            (path, content)
        })
        .buffered(CONCURRENT_READS)
        .collect()
        .await
}

/// Resolves a handoff path against the sandbox root.
fn resolve(root: &Path, path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    }
}

/// Truncates to at most `limit` bytes without splitting a character.
fn truncate_on_char_boundary(content: &str, limit: usize) -> String {
    if content.len() <= limit {
        return content.to_string();
    }
    let mut end = limit;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handoff(paths: &[&str]) -> HandoffDocument {
        HandoffDocument {
            current_task: "task".to_string(),
            files_paths: paths.iter().map(|path| (*path).to_string()).collect(),
            ..HandoffDocument::default()
        }
    }

    #[tokio::test]
    async fn restores_the_files_the_handoff_named() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("a.rs"), "fn a() {}")
            .await
            .expect("write a");
        async_fs::write(root.path().join("b.rs"), "fn b() {}")
            .await
            .expect("write b");

        let block = restore_context(&handoff(&["a.rs", "b.rs"]), root.path(), 10_000)
            .await
            .expect("restoration produces a block");

        assert_eq!(block.tag(), "restored_context");
        assert!(block.content().contains("fn a() {}"));
        assert!(block.content().contains("fn b() {}"));
        assert!(!block.content().contains("left out of this restoration"));
    }

    #[tokio::test]
    async fn the_budget_truncates_and_then_omits() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("big.rs"), "x".repeat(500))
            .await
            .expect("write big");
        async_fs::write(root.path().join("next.rs"), "y".repeat(500))
            .await
            .expect("write next");

        let block = restore_context(&handoff(&["big.rs", "next.rs"]), root.path(), 100)
            .await
            .expect("restoration produces a block");

        assert!(block.content().contains("(truncated)"));
        assert!(block.content().contains("1 further file(s)"));
        assert!(
            !block.content().contains(&"y".repeat(10)),
            "the omitted file must not be inlined"
        );
    }

    #[tokio::test]
    async fn missing_files_are_skipped_not_fatal() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("present.rs"), "fn present() {}")
            .await
            .expect("write present");

        let block = restore_context(&handoff(&["gone.rs", "present.rs"]), root.path(), 10_000)
            .await
            .expect("restoration produces a block");

        assert!(block.content().contains("fn present() {}"));
        assert!(!block.content().contains("gone.rs"));
    }

    #[tokio::test]
    async fn nothing_restorable_yields_no_block() {
        let root = tempfile::tempdir().expect("tempdir");
        assert!(
            restore_context(&handoff(&["gone.rs"]), root.path(), 10_000)
                .await
                .is_none()
        );
        assert!(
            restore_context(&handoff(&[]), root.path(), 10_000)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_zero_budget_restores_nothing() {
        let root = tempfile::tempdir().expect("tempdir");
        async_fs::write(root.path().join("a.rs"), "fn a() {}")
            .await
            .expect("write a");
        assert!(
            restore_context(&handoff(&["a.rs"]), root.path(), 0)
                .await
                .is_none()
        );
    }

    #[test]
    fn candidates_merge_paths_and_the_file_index_without_duplicates() {
        let document = HandoffDocument {
            current_task: "task".to_string(),
            files_paths: vec!["src/lib.rs".to_string(), "src/lib.rs".to_string()],
            file_index: vec![
                "src/lib.rs — edited".to_string(),
                "artifacts/report.md — written".to_string(),
                "outputs/".to_string(),
            ],
            ..HandoffDocument::default()
        };
        assert_eq!(
            restoration_candidates(&document),
            vec!["src/lib.rs".to_string(), "artifacts/report.md".to_string()]
        );
    }
}
