//! Error type returned by [`crate::apply_patch_text`].
//!
//! Every variant carries enough context for a caller — human or model — to fix
//! the patch or the target file without re-running anything.

use std::io;

/// Failure while reading, validating, or writing a patch.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PatchError {
    /// The patch text read from standard input was empty or all whitespace.
    #[error("patch input is empty")]
    EmptyInput,

    /// The patch text itself was not valid UTF-8.
    #[error("patch input is not valid UTF-8")]
    InputNotUtf8,

    /// The patch text is not in the compact patch format.
    #[error("Invalid patch format: {0}")]
    InvalidFormat(String),

    /// A hunk's context and deleted lines match nowhere it may apply.
    #[error("Patch conflict: in {path}: hunk {hunk}: {reason}")]
    Conflict {
        /// The patch-relative path of the file being updated.
        path: String,
        /// The 1-based number of the failing hunk within that file's update.
        hunk: usize,
        /// Which line failed to match and where.
        reason: String,
    },

    /// A hunk's context and deleted lines match in more than one place.
    #[error(
        "Ambiguous patch: in {path}: hunk {hunk} matches at file lines {}; add surrounding context or an `@@ anchor` line to select one",
        join_lines(starts)
    )]
    Ambiguous {
        /// The patch-relative path of the file being updated.
        path: String,
        /// The 1-based number of the ambiguous hunk within that file's update.
        hunk: usize,
        /// The 1-based file lines where the hunk's old side starts.
        starts: Vec<usize>,
    },

    /// A `Delete File` directive's content lines differ from the file.
    #[error("Patch conflict: in {0}: content to delete does not match the file's content")]
    DeleteMismatch(String),

    /// An Update or Delete directive named a path that does not exist.
    #[error("{0}: file not found")]
    FileNotFound(String),

    /// An Add or Move directive named a path that already exists.
    #[error("{0}: path already exists")]
    AlreadyExists(String),

    /// A file directive named a path that is not a regular file.
    #[error("{0}: not a regular file")]
    NotAFile(String),

    /// A target file on disk is not valid UTF-8 and cannot be patched
    /// textually.
    #[error("{0}: file is not valid UTF-8")]
    NotUtf8(String),

    /// A filesystem operation failed; `path` names the file involved.
    #[error("{path}: {source}")]
    Io {
        /// The patch-relative path that failed.
        path: String,
        /// The underlying OS error.
        source: io::Error,
    },
}

fn join_lines(lines: &[usize]) -> String {
    lines
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
