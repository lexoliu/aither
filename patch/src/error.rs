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

    /// The patch could not be parsed, or one of its hunks did not apply.
    #[error("{0}")]
    Patch(#[from] zenpatch::ZenpatchError),

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
