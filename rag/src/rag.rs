//! High-level RAG orchestrator.

use std::fs;
use std::path::Path;

use aither_core::embedding::EmbeddingModel;

use crate::chunking::{Chunker, CodeChunker, FixedSizeChunker, ParagraphChunker, SentenceChunker};
use crate::cleaning::{BasicCleaner, Cleaner};
use crate::config::{RagConfig, RagConfigBuilder};
use crate::error::Result;
use crate::index::VectorIndex;
use crate::indexing::{IndexProgress, IndexStage};
use crate::persistence::{Persistence, RedbPersistence};
use crate::store::RagStore;
use crate::types::{Document, Metadata, SearchResult};

/// High-level RAG orchestrator that provides a simple API for common RAG workflows.
pub struct Rag<
    M: EmbeddingModel,
    C: Chunker = FixedSizeChunker,
    L: Cleaner = BasicCleaner,
    P: Persistence = RedbPersistence,
> {
    store: RagStore<M, C, L>,
    persistence: P,
    config: RagConfig,
}

impl<M: EmbeddingModel, C: Chunker, L: Cleaner, P: Persistence> std::fmt::Debug
    for Rag<M, C, L, P>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rag")
            .field("store", &self.store)
            .field("persistence_path", &self.persistence.path())
            .field("config", &self.config)
            .finish()
    }
}

impl<M> Rag<M, FixedSizeChunker, BasicCleaner, RedbPersistence>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a new RAG instance with default configuration.
    pub fn new(embedder: M) -> Self {
        let config = RagConfig::default();
        let persistence =
            RedbPersistence::new(&config.index_path).expect("Failed to create default persistence");
        let store = RagStore::with_config(embedder, config.clone());

        Self {
            store,
            persistence,
            config,
        }
    }

    /// Creates a builder for custom configuration.
    pub fn builder(embedder: M) -> RagBuilder<M> {
        RagBuilder::new(embedder)
    }
}

impl<M, C, L, P> Rag<M, C, L, P>
where
    M: EmbeddingModel + Send + Sync + 'static,
    C: Chunker,
    L: Cleaner,
    P: Persistence,
{
    /// Loads the index from persistence.
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
    pub async fn index_directory<Pth: AsRef<Path>>(&self, dir: Pth) -> Result<usize> {
        self.index_directory_with_progress(dir, |_| {}).await
    }

    /// Indexes all files in a directory with progress callback.
    pub async fn index_directory_with_progress<Pth, F>(
        &self,
        dir: Pth,
        mut on_progress: F,
    ) -> Result<usize>
    where
        Pth: AsRef<Path>,
        F: FnMut(IndexProgress),
    {
        use crate::indexing::collect_files;

        let dir_path = dir.as_ref().to_path_buf();

        on_progress(IndexProgress::new(0, 0, None, IndexStage::Scanning));

        let files = collect_files(&dir_path)?;
        let total = files.len();
        let mut indexed = 0usize;

        for (idx, path) in files.into_iter().enumerate() {
            let relative_id = path
                .strip_prefix(&dir_path)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

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

            let mut metadata = Metadata::new();
            metadata.insert("path".into(), path.display().to_string());

            let doc = Document::with_metadata(relative_id, content, metadata);

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

        if self.config.auto_save {
            on_progress(IndexProgress::new(indexed, total, None, IndexStage::Saving));
            self.save()?;
        }

        on_progress(IndexProgress::new(indexed, total, None, IndexStage::Done));

        Ok(indexed)
    }

    /// Inserts a single document.
    pub async fn insert(&self, document: Document) -> Result<usize> {
        self.store.insert(document).await
    }

    /// Deletes a document and all its chunks.
    pub fn delete(&self, doc_id: &str) -> bool {
        self.store.delete(doc_id)
    }

    /// Searches for similar content.
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.store.search(query).await
    }

    /// Searches with a custom result count.
    pub async fn search_with_k(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.store.search_with_k(query, top_k).await
    }

    /// Returns a reference to the underlying store.
    pub const fn store(&self) -> &RagStore<M, C, L> {
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

    /// Returns configuration.
    pub const fn config(&self) -> &RagConfig {
        &self.config
    }
}

/// Builder for configuring a [`Rag`] instance.
pub struct RagBuilder<M: EmbeddingModel, C: Chunker = FixedSizeChunker, L: Cleaner = BasicCleaner> {
    embedder: M,
    config_builder: RagConfigBuilder,
    chunker: C,
    cleaner: L,
}

impl<M: EmbeddingModel, C: Chunker, L: Cleaner> std::fmt::Debug for RagBuilder<M, C, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagBuilder")
            .field("config_builder", &self.config_builder)
            .field("chunker", &self.chunker.name())
            .field("cleaner", &self.cleaner.name())
            .finish_non_exhaustive()
    }
}

impl<M> RagBuilder<M, FixedSizeChunker, BasicCleaner>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    fn new(embedder: M) -> Self {
        Self {
            embedder,
            config_builder: RagConfigBuilder::new(),
            chunker: FixedSizeChunker::default(),
            cleaner: BasicCleaner,
        }
    }
}

impl<M, C, L> RagBuilder<M, C, L>
where
    M: EmbeddingModel + Send + Sync + 'static,
    C: Chunker,
    L: Cleaner,
{
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
    pub fn chunker<C2: Chunker>(self, chunker: C2) -> RagBuilder<M, C2, L> {
        RagBuilder {
            embedder: self.embedder,
            config_builder: self.config_builder,
            chunker,
            cleaner: self.cleaner,
        }
    }

    /// Uses a custom cleaner.
    #[must_use]
    pub fn cleaner<L2: Cleaner>(self, cleaner: L2) -> RagBuilder<M, C, L2> {
        RagBuilder {
            embedder: self.embedder,
            config_builder: self.config_builder,
            chunker: self.chunker,
            cleaner,
        }
    }

    /// Uses fixed-size chunking with custom parameters.
    #[must_use]
    pub fn fixed_chunking(
        self,
        chunk_size: usize,
        overlap: usize,
    ) -> RagBuilder<M, FixedSizeChunker, L> {
        self.chunker(FixedSizeChunker::new(chunk_size, overlap))
    }

    /// Uses sentence-based chunking.
    #[must_use]
    pub fn sentence_chunking(self, max_chunk_size: usize) -> RagBuilder<M, SentenceChunker, L> {
        self.chunker(SentenceChunker::new(max_chunk_size))
    }

    /// Uses paragraph-based chunking.
    #[must_use]
    pub fn paragraph_chunking(self, max_chunk_size: usize) -> RagBuilder<M, ParagraphChunker, L> {
        self.chunker(ParagraphChunker::new(max_chunk_size))
    }

    /// Uses tree-sitter semantic code chunking.
    #[must_use]
    pub fn code_chunking(self, max_chunk_size: usize) -> RagBuilder<M, CodeChunker, L> {
        self.chunker(CodeChunker::new(max_chunk_size))
    }

    /// Builds the [`Rag`] instance using Redb persistence.
    pub fn build(self) -> Result<Rag<M, C, L, RedbPersistence>> {
        let config = self.config_builder.build();
        let persistence = RedbPersistence::new(&config.index_path)?;
        let store =
            RagStore::with_components(self.embedder, config.clone(), self.chunker, self.cleaner);

        Ok(Rag {
            store,
            persistence,
            config,
        })
    }

    /// Builds the [`Rag`] instance using a provided persistence backend.
    pub fn build_with_persistence<P: Persistence>(self, persistence: P) -> Result<Rag<M, C, L, P>> {
        let config = self.config_builder.build();
        let store =
            RagStore::with_components(self.embedder, config.clone(), self.chunker, self.cleaner);

        Ok(Rag {
            store,
            persistence,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::EmbeddingModel;
    use std::sync::Arc;
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

        fs::write(data_dir.path().join("file1.txt"), "Hello world").unwrap();
        fs::write(data_dir.path().join("file2.txt"), "Goodbye world").unwrap();

        let rag = Rag::builder(embedder)
            .index_path(index_dir.path().join("index.redb"))
            .auto_save(false)
            .build()
            .unwrap();

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

        let results = rag.search("hello").await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn save_and_load() {
        let index_dir = tempdir().unwrap();

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

        assert!(!rag.config().deduplication);
        assert_eq!(rag.config().default_top_k, 10);
        assert_eq!(rag.config().similarity_threshold, 0.5);
    }
}
