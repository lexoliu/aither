//! Directory indexing with progress tracking.

use std::fs;
use std::path::{Path, PathBuf};

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
