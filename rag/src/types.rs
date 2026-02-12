//! Core types for the RAG crate.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Key/value metadata attached to documents and chunks.
pub type Metadata = BTreeMap<String, String>;

/// A document to be indexed in the RAG store.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Stable identifier for the document.
    pub id: String,
    /// Raw text content.
    pub text: String,
    /// Arbitrary metadata for filtering/citations.
    pub metadata: Metadata,
}

impl Document {
    /// Creates a new document with empty metadata.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// Creates a new document with metadata.
    #[must_use]
    pub fn with_metadata(
        id: impl Into<String>,
        text: impl Into<String>,
        metadata: Metadata,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata,
        }
    }
}

/// A chunk of text derived from a document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk (format: `{doc_id}#chunk_{n}`).
    pub id: String,
    /// Text content of the chunk.
    pub text: String,
    /// Parent document ID.
    pub source_id: String,
    /// Index of this chunk within the document.
    pub index: usize,
    /// Inherited and chunk-specific metadata.
    pub metadata: Metadata,
    /// Content hash for deduplication.
    pub content_hash: u64,
}

impl Chunk {
    /// Creates a new chunk.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        text: impl Into<String>,
        source_id: impl Into<String>,
        index: usize,
        content_hash: u64,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            source_id: source_id.into(),
            index,
            metadata: Metadata::new(),
            content_hash,
        }
    }

    /// Creates a new chunk with metadata.
    #[must_use]
    pub fn with_metadata(
        id: impl Into<String>,
        text: impl Into<String>,
        source_id: impl Into<String>,
        index: usize,
        content_hash: u64,
        metadata: Metadata,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            source_id: source_id.into(),
            index,
            metadata,
            content_hash,
        }
    }
}

/// A search result containing a chunk and its similarity score.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching chunk.
    pub chunk: Chunk,
    /// Similarity score (higher is better, typically 0.0 to 1.0 for cosine similarity).
    pub score: f32,
}

/// Internal entry stored in the index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexEntry {
    /// The chunk.
    pub chunk: Chunk,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

impl IndexEntry {
    /// Creates a new index entry.
    #[must_use]
    pub const fn new(chunk: Chunk, embedding: Vec<f32>) -> Self {
        Self { chunk, embedding }
    }
}
