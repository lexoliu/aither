//! Directory indexing with progress tracking.

use async_channel::{Receiver, Sender, unbounded};
use futures_core::Stream;
use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::error::Result;

/// Progress update during directory indexing.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Number of files processed so far.
    pub processed: usize,
    /// Total number of files discovered.
    pub total: usize,
    /// Current file being processed (if any).
    pub current_file: Option<PathBuf>,
    /// Current stage of indexing.
    pub stage: IndexStage,
}

impl IndexProgress {
    /// Creates a new progress update.
    #[must_use]
    pub fn new(
        processed: usize,
        total: usize,
        current_file: Option<PathBuf>,
        stage: IndexStage,
    ) -> Self {
        Self {
            processed,
            total,
            current_file,
            stage,
        }
    }
}

/// Stages of the indexing process.
#[derive(Debug, Clone)]
pub enum IndexStage {
    /// Scanning the directory for files.
    Scanning,
    /// Chunking the current file.
    Chunking,
    /// Embedding the current file's chunks.
    Embedding,
    /// Adding chunks to the index.
    Indexing,
    /// Saving the index to disk.
    Saving,
    /// Indexing completed successfully.
    Done,
    /// File was skipped due to an error.
    Skipped {
        /// Reason the file was skipped.
        reason: String,
    },
}

/// A job that indexes a directory and reports progress.
///
/// This type implements both `Future` (for awaiting completion) and `Stream`
/// (for receiving progress updates).
pub struct IndexingJob {
    /// Receiver for progress updates.
    progress_rx: Pin<Box<Receiver<IndexProgress>>>,
    /// The completion future.
    completion: Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send>>,
    /// Cached completion result.
    completion_result: Option<Result<usize>>,
}

impl std::fmt::Debug for IndexingJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexingJob").finish_non_exhaustive()
    }
}

impl IndexingJob {
    /// Creates a new indexing job.
    pub(crate) fn new<F>(future: F, progress_rx: Receiver<IndexProgress>) -> Self
    where
        F: std::future::Future<Output = Result<usize>> + Send + 'static,
    {
        Self {
            progress_rx: Box::pin(progress_rx),
            completion: Box::pin(future),
            completion_result: None,
        }
    }

    /// Creates a progress channel pair.
    pub(crate) fn channel() -> (Sender<IndexProgress>, Receiver<IndexProgress>) {
        unbounded()
    }
}

impl std::future::Future for IndexingJob {
    type Output = Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        if let Some(result) = this.completion_result.take() {
            return Poll::Ready(result);
        }

        this.completion.as_mut().poll(cx)
    }
}

impl Stream for IndexingJob {
    type Item = IndexProgress;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = unsafe { self.get_unchecked_mut() };

        // Drive the completion future to make progress
        if this.completion_result.is_none() {
            if let Poll::Ready(result) = this.completion.as_mut().poll(cx) {
                this.completion_result = Some(result);
            }
        }

        // Poll for progress updates
        this.progress_rx.as_mut().poll_next(cx)
    }
}

/// Collects all files from a directory tree.
pub(crate) fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(path) = stack.pop() {
        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(_) => continue,
        };

        if metadata.is_dir() {
            let entries = fs::read_dir(&path)?;
            for entry in entries {
                let entry = entry?;
                stack.push(entry.path());
            }
        } else if metadata.is_file() {
            files.push(path);
        }
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn collect_files_from_directory() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(dir.path().join("file2.txt"), "content2").unwrap();

        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("file3.txt"), "content3").unwrap();

        let files = collect_files(dir.path()).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn collect_files_empty_directory() {
        let dir = tempdir().unwrap();
        let files = collect_files(dir.path()).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn progress_creation() {
        let progress = IndexProgress::new(
            5,
            10,
            Some(PathBuf::from("/test/file.txt")),
            IndexStage::Embedding,
        );
        assert_eq!(progress.processed, 5);
        assert_eq!(progress.total, 10);
        assert!(progress.current_file.is_some());
    }
}
