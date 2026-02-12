//! High-level RAG orchestrator.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;

use crate::chunking::{Chunker, FixedSizeChunker, SentenceChunker};
use crate::config::{RagConfig, RagConfigBuilder};
use crate::error::Result;
use crate::index::VectorIndex;
use crate::indexing::{IndexProgress, IndexStage};
use crate::persistence::{Persistence, RedbPersistence};
use crate::store::RagStore;
use crate::types::{Document, Metadata, SearchResult};

/// High-level RAG orchestrator that provides a simple API for common RAG workflows.
///
/// `Rag` wraps a [`RagStore`] and adds directory indexing, persistence, and
/// convenient configuration through a builder pattern.
///
/// # Example
///
/// ```rust,no_run
/// use aither_rag::Rag;
/// use aither_core::EmbeddingModel;
///
/// async fn example<E: EmbeddingModel + Send + Sync + 'static>(embedder: E) {
///     // Simple usage with defaults
///     let mut rag = Rag::new(embedder);
///
///     // Index a directory
///     let count = rag.index_directory("./docs").await.unwrap();
///
///     // Search
///     let results = rag.search("query").await.unwrap();
/// }
/// ```
pub struct Rag<M: EmbeddingModel> {
    store: RagStore<M>,
    persistence: Arc<dyn Persistence>,
    config: RagConfig,
}

impl<M: EmbeddingModel> std::fmt::Debug for Rag<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rag")
            .field("store", &self.store)
            .field("persistence_path", &self.persistence.path())
            .field("config", &self.config)
            .finish()
    }
}

impl<M> Rag<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a new RAG instance with default configuration.
    ///
    /// This uses:
    /// - Fixed-size chunking (512 chars, 64 overlap)
    /// - Redb persistence at `./rag_index.redb`
    /// - Deduplication enabled
    /// - Auto-save enabled
    pub fn new(embedder: M) -> Self {
        let config = RagConfig::default();
        let persistence =
            RedbPersistence::new(&config.index_path).expect("Failed to create default persistence");

        Self {
            store: RagStore::new(embedder),
            persistence: Arc::new(persistence),
            config,
        }
    }

    /// Creates a builder for custom configuration.
    pub fn builder(embedder: M) -> RagBuilder<M> {
        RagBuilder::new(embedder)
    }

    /// Loads the index from persistence.
    ///
    /// # Returns
    /// The number of entries loaded.
    pub fn load(&self) -> Result<usize> {
        let entries = self.persistence.load()?;
        let count = entries.len();
        self.store.index().load(entries)?;
        Ok(count)
    }

    /// Saves the index to persistence.
    pub fn save(&self) -> Result<()> {
        let entries = self.store.index().entries();
        self.persistence.save(&entries)
    }

    /// Indexes all files in a directory.
    ///
    /// This method:
    /// 1. Recursively collects all files
    /// 2. Reads each file as text
    /// 3. Chunks, embeds, and indexes the content
    /// 4. Saves the index (if auto-save is enabled)
    ///
    /// Returns the number of files indexed and a receiver for progress updates.
    pub async fn index_directory<P: AsRef<Path>>(&self, dir: P) -> Result<usize> {
        self.index_directory_with_progress(dir, |_| {}).await
    }

    /// Indexes all files in a directory with progress callback.
    ///
    /// The callback is called for each progress update during indexing.
    pub async fn index_directory_with_progress<P, F>(
        &self,
        dir: P,
        mut on_progress: F,
    ) -> Result<usize>
    where
        P: AsRef<Path>,
        F: FnMut(IndexProgress),
    {
        use crate::indexing::collect_files;

        let dir_path = dir.as_ref().to_path_buf();

        // Collect files
        on_progress(IndexProgress::new(0, 0, None, IndexStage::Scanning));

        let files = collect_files(&dir_path)?;
        let total = files.len();

        let mut indexed = 0usize;

        for (idx, path) in files.into_iter().enumerate() {
            // Generate document ID from relative path
            let relative_id = path
                .strip_prefix(&dir_path)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            // Read file content
            let content = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    on_progress(IndexProgress::new(
                        idx,
                        total,
                        Some(path.clone()),
                        IndexStage::Skipped {
                            reason: err.to_string(),
                        },
                    ));
                    continue;
                }
            };

            // Create document with metadata
            let mut metadata = Metadata::new();
            metadata.insert("path".into(), path.display().to_string());

            let doc = Document::with_metadata(relative_id, content, metadata);

            // Embed and index
            on_progress(IndexProgress::new(
                idx,
                total,
                Some(path.clone()),
                IndexStage::Embedding,
            ));

            self.store.insert(doc).await?;
            indexed += 1;

            on_progress(IndexProgress::new(
                idx + 1,
                total,
                Some(path),
                IndexStage::Indexing,
            ));
        }

        // Save if auto-save is enabled
        if self.config.auto_save {
            on_progress(IndexProgress::new(indexed, total, None, IndexStage::Saving));

            let entries = self.store.index().entries();
            self.persistence.save(&entries)?;
        }

        on_progress(IndexProgress::new(indexed, total, None, IndexStage::Done));

        Ok(indexed)
    }

    /// Inserts a single document.
    ///
    /// # Returns
    /// The number of chunks inserted.
    pub async fn insert(&self, document: Document) -> Result<usize> {
        self.store.insert(document).await
    }

    /// Deletes a document and all its chunks.
    ///
    /// # Returns
    /// `true` if anything was deleted.
    pub fn delete(&self, doc_id: &str) -> bool {
        self.store.delete(doc_id)
    }

    /// Searches for similar content.
    ///
    /// Uses the configured default `top_k`.
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.store.search(query).await
    }

    /// Searches with a custom result count.
    pub async fn search_with_k(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.store.search_with_k(query, top_k).await
    }

    /// Returns a reference to the underlying store.
    pub const fn store(&self) -> &RagStore<M> {
        &self.store
    }

    /// Returns the number of indexed chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clears all indexed data.
    pub fn clear(&self) {
        self.store.clear();
    }
}

/// Builder for configuring a [`Rag`] instance.
pub struct RagBuilder<M: EmbeddingModel> {
    embedder: M,
    config_builder: RagConfigBuilder,
    chunker: Option<Arc<dyn Chunker>>,
}

impl<M: EmbeddingModel> std::fmt::Debug for RagBuilder<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagBuilder")
            .field("config_builder", &self.config_builder)
            .field(
                "chunker",
                &self.chunker.as_ref().map_or("default", |c| c.name()),
            )
            .finish_non_exhaustive()
    }
}

impl<M> RagBuilder<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a new builder with the given embedder.
    fn new(embedder: M) -> Self {
        Self {
            embedder,
            config_builder: RagConfigBuilder::new(),
            chunker: None,
        }
    }

    /// Sets the index persistence path.
    #[must_use]
    pub fn index_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config_builder = self.config_builder.index_path(path);
        self
    }

    /// Sets the minimum similarity threshold.
    #[must_use]
    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.config_builder = self.config_builder.similarity_threshold(threshold);
        self
    }

    /// Sets the default number of search results.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.config_builder = self.config_builder.default_top_k(k);
        self
    }

    /// Enables or disables deduplication.
    #[must_use]
    pub fn deduplication(mut self, enabled: bool) -> Self {
        self.config_builder = self.config_builder.deduplication(enabled);
        self
    }

    /// Enables or disables auto-save.
    #[must_use]
    pub fn auto_save(mut self, enabled: bool) -> Self {
        self.config_builder = self.config_builder.auto_save(enabled);
        self
    }

    /// Uses a custom chunker.
    #[must_use]
    pub fn chunker(mut self, chunker: impl Chunker + 'static) -> Self {
        self.chunker = Some(Arc::new(chunker));
        self
    }

    /// Uses fixed-size chunking with custom parameters.
    #[must_use]
    pub fn fixed_chunking(mut self, chunk_size: usize, overlap: usize) -> Self {
        self.chunker = Some(Arc::new(FixedSizeChunker::new(chunk_size, overlap)));
        self
    }

    /// Uses sentence-based chunking.
    #[must_use]
    pub fn sentence_chunking(mut self, max_chunk_size: usize) -> Self {
        self.chunker = Some(Arc::new(SentenceChunker::new(max_chunk_size)));
        self
    }

    /// Builds the [`Rag`] instance.
    ///
    /// # Errors
    /// Returns an error if persistence cannot be initialized.
    pub fn build(self) -> Result<Rag<M>> {
        let config = self.config_builder.build();
        let persistence = RedbPersistence::new(&config.index_path)?;

        let mut store = RagStore::with_config(self.embedder, config.clone());
        if let Some(chunker) = self.chunker {
            // Need to rebuild store with custom chunker
            store = store.with_chunker(ChunkerWrapper(chunker));
        }

        Ok(Rag {
            store,
            persistence: Arc::new(persistence),
            config,
        })
    }
}

/// Wrapper to convert Arc<dyn Chunker> into a Chunker implementation.
struct ChunkerWrapper(Arc<dyn Chunker>);

impl Chunker for ChunkerWrapper {
    fn chunk(&self, doc: &Document) -> Result<Vec<crate::types::Chunk>> {
        self.0.chunk(doc)
    }

    fn name(&self) -> &'static str {
        // We can't get the inner name without an additional method
        "custom"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::EmbeddingModel;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

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
    async fn index_directory_and_search() {
        let embedder = MockEmbedder::new(4);
        let index_dir = tempdir().unwrap();
        let data_dir = tempdir().unwrap();

        // Create test files
        fs::write(data_dir.path().join("file1.txt"), "Hello world").unwrap();
        fs::write(data_dir.path().join("file2.txt"), "Goodbye world").unwrap();

        let rag = Rag::builder(embedder)
            .index_path(index_dir.path().join("index.redb"))
            .auto_save(false)
            .build()
            .unwrap();

        // Index directory
        let mut saw_done = false;
        let count = rag
            .index_directory_with_progress(data_dir.path(), |progress| {
                if matches!(progress.stage, IndexStage::Done) {
                    saw_done = true;
                }
            })
            .await
            .unwrap();

        assert!(saw_done);
        assert_eq!(count, 2);
        assert_eq!(rag.len(), 2);

        // Search
        let results = rag.search("hello").await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn save_and_load() {
        let index_dir = tempdir().unwrap();

        // Create and populate
        {
            let embedder = MockEmbedder::new(4);
            let rag = Rag::builder(embedder)
                .index_path(index_dir.path().join("index.redb"))
                .build()
                .unwrap();

            rag.insert(Document::new("doc1", "Hello world"))
                .await
                .unwrap();
            rag.save().unwrap();
        }

        // Reload
        {
            let embedder = MockEmbedder::new(4);
            let rag = Rag::builder(embedder)
                .index_path(index_dir.path().join("index.redb"))
                .build()
                .unwrap();

            let count = rag.load().unwrap();
            assert_eq!(count, 1);
            assert_eq!(rag.len(), 1);
        }
    }

    #[tokio::test]
    async fn builder_configuration() {
        let index_dir = tempdir().unwrap();
        let embedder = MockEmbedder::new(4);

        let rag = Rag::builder(embedder)
            .index_path(index_dir.path().join("custom.redb"))
            .similarity_threshold(0.5)
            .top_k(10)
            .deduplication(false)
            .sentence_chunking(256)
            .build()
            .unwrap();

        assert!(!rag.config.deduplication);
        assert_eq!(rag.config.default_top_k, 10);
        assert_eq!(rag.config.similarity_threshold, 0.5);
    }
}
