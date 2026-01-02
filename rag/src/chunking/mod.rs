//! Text chunking strategies for RAG.
//!
//! This module provides the [`Chunker`] trait and implementations for
//! splitting documents into smaller, indexable chunks.

mod fixed;
mod sentence;

pub use fixed::FixedSizeChunker;
pub use sentence::SentenceChunker;

use crate::error::Result;
use crate::types::{Chunk, Document};

/// Trait for text chunking strategies.
///
/// Chunkers split documents into smaller pieces that can be individually
/// embedded and searched. Different strategies suit different use cases:
///
/// - [`FixedSizeChunker`]: Simple character-based chunking with overlap
/// - [`SentenceChunker`]: Respects sentence boundaries for more coherent chunks
pub trait Chunker: Send + Sync {
    /// Splits a document into chunks.
    ///
    /// # Arguments
    /// * `doc` - The document to chunk
    ///
    /// # Returns
    /// A vector of chunks, each with a unique ID derived from the document ID.
    fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>>;

    /// Returns the name of this chunking strategy.
    fn name(&self) -> &'static str;
}
