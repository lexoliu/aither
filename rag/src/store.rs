//! Core RAG store implementation.

use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;

use crate::chunking::{Chunker, FixedSizeChunker};
use crate::config::RagConfig;
use crate::error::{RagError, Result};
use crate::index::{HnswIndex, VectorIndex};
use crate::types::{Chunk, Document, SearchResult};

/// The core RAG store that manages documents, chunking, and indexing.
///
/// `RagStore` combines an embedding model with a vector index to provide
/// efficient semantic search over documents.
pub struct RagStore<M: EmbeddingModel> {
    embedder: Arc<M>,
    index: Arc<HnswIndex>,
    chunker: Arc<dyn Chunker>,
    config: RagConfig,
}

impl<M: EmbeddingModel> Clone for RagStore<M> {
    fn clone(&self) -> Self {
        Self {
            embedder: Arc::clone(&self.embedder),
            index: Arc::clone(&self.index),
            chunker: Arc::clone(&self.chunker),
            config: self.config.clone(),
        }
    }
}

impl<M: EmbeddingModel> std::fmt::Debug for RagStore<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagStore")
            .field("index", &self.index)
            .field("chunker", &self.chunker.name())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<M> RagStore<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a new RAG store with the given embedding model.
    ///
    /// Uses default configuration and fixed-size chunking.
    #[must_use]
    pub fn new(embedder: M) -> Self {
        let dimension = embedder.dim();
        Self {
            embedder: Arc::new(embedder),
            index: Arc::new(HnswIndex::new(dimension)),
            chunker: Arc::new(FixedSizeChunker::default()),
            config: RagConfig::default(),
        }
    }

    /// Creates a new RAG store with custom configuration.
    #[must_use]
    pub fn with_config(embedder: M, config: RagConfig) -> Self {
        let dimension = embedder.dim();
        Self {
            embedder: Arc::new(embedder),
            index: Arc::new(HnswIndex::new(dimension)),
            chunker: Arc::new(FixedSizeChunker::default()),
            config,
        }
    }

    /// Sets a custom chunker for this store.
    #[must_use]
    pub fn with_chunker(mut self, chunker: impl Chunker + 'static) -> Self {
        self.chunker = Arc::new(chunker);
        self
    }

    /// Inserts a document into the store.
    ///
    /// The document is chunked, deduplicated (if enabled), embedded, and indexed.
    ///
    /// # Returns
    /// The number of chunks actually inserted (may be less than total chunks
    /// if deduplication is enabled and duplicates are found).
    pub async fn insert(&self, document: Document) -> Result<usize> {
        let chunks = self.chunker.chunk(&document)?;
        let mut inserted = 0;

        for chunk in chunks {
            // Check for duplicate content
            if self.config.deduplication && self.index.contains_hash(chunk.content_hash) {
                continue;
            }

            let embedding = self
                .embedder
                .embed(&chunk.text)
                .await
                .map_err(RagError::Embedding)?;

            self.index.insert(chunk, embedding)?;
            inserted += 1;
        }

        Ok(inserted)
    }

    /// Inserts multiple documents into the store.
    ///
    /// # Returns
    /// The total number of chunks inserted across all documents.
    pub async fn insert_batch(&self, documents: Vec<Document>) -> Result<usize> {
        let mut total_inserted = 0;
        for doc in documents {
            total_inserted += self.insert(doc).await?;
        }
        Ok(total_inserted)
    }

    /// Inserts a pre-chunked chunk with a precomputed embedding.
    ///
    /// Useful for loading from persistence or custom chunking workflows.
    pub fn insert_with_embedding(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()> {
        self.index.insert(chunk, embedding)
    }

    /// Deletes a document and all its chunks from the store.
    ///
    /// # Returns
    /// `true` if any chunks were removed.
    pub fn delete(&self, doc_id: &str) -> bool {
        // Delete the main document chunk
        let mut removed = self.index.remove(doc_id);

        // Delete all numbered chunks
        let mut chunk_idx = 0;
        loop {
            let chunk_id = format!("{}#chunk_{}", doc_id, chunk_idx);
            if self.index.remove(&chunk_id) {
                removed = true;
                chunk_idx += 1;
            } else {
                break;
            }
        }

        removed
    }

    /// Searches for chunks similar to the query.
    ///
    /// # Arguments
    /// * `query` - The search query text
    ///
    /// # Returns
    /// Search results sorted by similarity score (highest first).
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.search_with_k(query, self.config.default_top_k).await
    }

    /// Searches for chunks similar to the query with a custom result count.
    ///
    /// # Arguments
    /// * `query` - The search query text
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    /// Search results sorted by similarity score (highest first).
    pub async fn search_with_k(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let embedding = self
            .embedder
            .embed(query)
            .await
            .map_err(RagError::Embedding)?;

        self.index
            .search(&embedding, top_k, self.config.similarity_threshold)
    }

    /// Returns the number of indexed chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clears all data from the store.
    pub fn clear(&self) {
        self.index.clear();
    }

    /// Returns a reference to the underlying index.
    pub fn index(&self) -> &HnswIndex {
        &self.index
    }

    /// Returns a reference to the embedder.
    pub fn embedder(&self) -> &M {
        &self.embedder
    }

    /// Returns the configuration.
    pub fn config(&self) -> &RagConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::EmbeddingModel;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct MockEmbedder {
        dimension: usize,
        calls: Arc<AtomicUsize>,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl EmbeddingModel for MockEmbedder {
        fn dim(&self) -> usize {
            self.dimension
        }

        async fn embed(&self, text: &str) -> aither_core::Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut vec = vec![0.0; self.dimension];
            for (idx, value) in vec.iter_mut().enumerate() {
                *value = ((text.len() + idx) % 10) as f32 / 10.0;
            }
            Ok(vec)
        }
    }

    #[tokio::test]
    async fn insert_and_search() {
        let embedder = MockEmbedder::new(4);
        let store = RagStore::new(embedder);

        let doc = Document::new("doc1", "Hello world");
        store.insert(doc).await.unwrap();

        assert_eq!(store.len(), 1);

        let results = store.search("hello").await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn delete_document() {
        let embedder = MockEmbedder::new(4);
        let store = RagStore::new(embedder);

        let doc = Document::new("doc1", "Hello world");
        store.insert(doc).await.unwrap();

        assert!(store.delete("doc1"));
        assert_eq!(store.len(), 0);
        assert!(!store.delete("doc1"));
    }

    #[tokio::test]
    async fn deduplication() {
        let embedder = MockEmbedder::new(4);
        let config = RagConfig::builder().deduplication(true).build();
        let store = RagStore::with_config(embedder, config);

        // Insert same content twice with different IDs
        let doc1 = Document::new("doc1", "Same content");
        let doc2 = Document::new("doc2", "Same content");

        let inserted1 = store.insert(doc1).await.unwrap();
        let inserted2 = store.insert(doc2).await.unwrap();

        assert_eq!(inserted1, 1);
        assert_eq!(inserted2, 0); // Deduplicated
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn no_deduplication() {
        let embedder = MockEmbedder::new(4);
        let config = RagConfig::builder().deduplication(false).build();
        let store = RagStore::with_config(embedder, config);

        let doc1 = Document::new("doc1", "Same content");
        let doc2 = Document::new("doc2", "Same content");

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        assert_eq!(store.len(), 2);
    }
}
