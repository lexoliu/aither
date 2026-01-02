//! Error types for the RAG crate.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur in RAG operations.
#[derive(Debug, Error)]
pub enum RagError {
    /// Embedding operation failed.
    #[error("embedding failed: {0}")]
    Embedding(#[source] anyhow::Error),

    /// Vector index operation failed.
    #[error("index error: {0}")]
    Index(String),

    /// Persistence operation failed.
    #[error("persistence error at {path}: {source}")]
    Persistence {
        /// Path where the error occurred.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// IO operation failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Dimension mismatch between embedding and index.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension provided.
        actual: usize,
    },

    /// Document not found.
    #[error("document not found: {0}")]
    NotFound(String),

    /// Chunking operation failed.
    #[error("chunking error: {0}")]
    Chunking(String),

    /// Database operation failed.
    #[error("database error: {0}")]
    Database(String),
}

/// Result type alias for RAG operations.
pub type Result<T> = std::result::Result<T, RagError>;
