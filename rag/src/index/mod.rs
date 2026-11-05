//! Vector index implementations for RAG.
//!
//! This module provides the [`VectorIndex`] trait and the [`HnswIndex`]
//! implementation for efficient approximate nearest neighbor search.

mod hnsw;

pub use hnsw::HnswIndex;

use crate::error::Result;
use crate::types::{Chunk, IndexEntry, SearchResult};

/// Trait for vector index implementations.
///
/// A vector index stores chunks with their embedding vectors and supports
/// efficient similarity search.
pub trait VectorIndex: Send + Sync {
    /// Inserts or updates a chunk with its embedding vector.
    ///
    /// If a chunk with the same ID already exists, it will be replaced.
    /// # Errors
    ///
    /// Returns an error if the embedding's dimension does not match the index.
    fn insert(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()>;

    /// Removes a chunk by its ID.
    ///
    /// Returns `true` if a chunk was removed, `false` if not found.
    fn remove(&self, chunk_id: &str) -> bool;

    /// Removes every chunk of the document `source_id`, whatever their IDs.
    ///
    /// Returns the number of chunks removed.
    fn remove_source(&self, source_id: &str) -> usize;

    /// Searches for the most similar chunks to the query vector.
    ///
    /// # Arguments
    /// * `query` - The query embedding vector
    /// * `top_k` - Maximum number of results to return
    /// * `threshold` - Minimum similarity score (0.0 to 1.0 for cosine)
    /// # Errors
    ///
    /// Returns an error if the query's dimension does not match the index.
    fn search(&self, query: &[f32], top_k: usize, threshold: f32) -> Result<Vec<SearchResult>>;

    /// Returns the embedding dimension.
    fn dimension(&self) -> usize;

    /// Returns the number of indexed chunks.
    fn len(&self) -> usize;

    /// Returns `true` if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all entries from the index.
    fn clear(&self);

    /// Returns an iterator over all index entries.
    fn entries(&self) -> Vec<IndexEntry>;

    /// Loads entries into the index, replacing existing content.
    /// # Errors
    ///
    /// Returns an error if any entry's dimension does not match the index.
    fn load(&self, entries: Vec<IndexEntry>) -> Result<()>;

    /// Checks if a content hash already exists in the index.
    fn contains_hash(&self, hash: u64) -> bool;
}
