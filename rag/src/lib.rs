//! Retrieval-Augmented Generation helper crate.
//!
//! The [`RagStore`] type glues any [`EmbeddingModel`](aither_core::EmbeddingModel) to a built-in
//! parallel vector index, exposing a tiny API surface:
//! - [`RagStore::insert`] – chunk/index new documents.
//! - [`RagStore::delete`] – drop vectors by identifier.
//! - [`RagStore::query`] – embed a prompt and fetch the best matching context.
//!
//! The default index keeps everything in-memory and performs cosine similarity scoring using a
//! parallel iterator, which works well up to hundreds of thousands of vectors without pulling in a
//! full external database. Replace it with your own backend later by reusing the public document and
//! scoring types.

use aither_core::{Result, embedding::EmbeddingModel, llm::tool::Tool};
use async_channel::{Receiver as AsyncReceiver, unbounded};
use futures_core::Stream;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json;
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, RwLock},
    task::{Context, Poll},
};

/// Key/value metadata attached to stored documents.
pub type Metadata = BTreeMap<String, String>;

/// Represents a chunk of text tracked inside the RAG store.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Document {
    /// Stable identifier used for deduplication and deletions.
    pub id: String,
    /// Raw text that produced the embedding vector.
    pub text: String,
    /// Arbitrary metadata that higher-level applications can use for filtering/citations.
    pub metadata: Metadata,
}

impl Document {
    /// Creates a document with empty metadata.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// Creates a document with metadata.
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

/// Result row returned by [`RagStore::query`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RetrievedDocument {
    /// Stored document.
    pub document: Document,
    /// Cosine similarity score (1.0 = identical).
    pub score: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PersistedEntry {
    document: Document,
    embedding: Vec<f32>,
}

/// High-level entry point for retrieval augmented workflows.
///
/// It owns both an embedding model and the internal vector index. All operations are async-friendly
/// and return the workspace-wide [`Result`].
pub struct RagStore<M> {
    embedder: Arc<M>,
    index: Arc<ParallelIndex>,
}

impl<M> Clone for RagStore<M> {
    fn clone(&self) -> Self {
        Self {
            embedder: Arc::clone(&self.embedder),
            index: Arc::clone(&self.index),
        }
    }
}

impl<M> RagStore<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a store with the default parallel index sized to the embedder's dimension.
    pub fn new(embedder: M) -> Self {
        let dimension = embedder.dim();
        Self {
            embedder: Arc::new(embedder),
            index: Arc::new(ParallelIndex::new(dimension)),
        }
    }

    /// Inserts or replaces a document.
    pub async fn insert(&self, document: Document) -> Result<()> {
        let embedding = self.embedder.embed(&document.text).await?;
        self.index.upsert(document, embedding);
        Ok(())
    }

    /// Deletes a document by id. Returns true if something was removed.
    pub fn delete(&self, doc_id: &str) -> bool {
        self.index.remove(doc_id)
    }

    /// Embeds a free-form query and returns the best matching documents.
    pub async fn query(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedDocument>> {
        let vector = self.embedder.embed(query).await?;
        Ok(self.index.search(&vector, top_k))
    }

    /// Returns a clone of the inner embedder handle for advanced flows.
    pub fn embedder(&self) -> Arc<M> {
        Arc::clone(&self.embedder)
    }

    /// Inserts a document using a precomputed embedding vector.
    pub fn insert_with_embedding(&self, document: Document, embedding: Vec<f32>) -> Result<()> {
        self.index.upsert(document, embedding);
        Ok(())
    }

    /// Returns a snapshot of all stored entries for persistence.
    pub(crate) fn snapshot(&self) -> Vec<PersistedEntry> {
        self.index.snapshot()
    }
}

/// Parallel cosine-similarity index optimized for medium-sized corpora.
#[derive(Debug)]
pub struct ParallelIndex {
    dimension: usize,
    entries: RwLock<Vec<IndexedDocument>>,
}

impl ParallelIndex {
    /// Creates an empty index. The dimension must match the embedder's output size.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: RwLock::new(Vec::new()),
        }
    }

    fn upsert(&self, doc: Document, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dimension,
            "embedding dimension mismatch"
        );
        let mut entries = self
            .entries
            .write()
            .expect("parallel index poisoned while inserting");
        if let Some(existing) = entries.iter_mut().find(|entry| entry.document.id == doc.id) {
            existing.document = doc;
            existing.embedding = embedding;
            return;
        }
        entries.push(IndexedDocument {
            document: doc,
            embedding,
        });
    }

    fn remove(&self, doc_id: &str) -> bool {
        let mut entries = self
            .entries
            .write()
            .expect("parallel index poisoned while deleting");
        let len_before = entries.len();
        entries.retain(|entry| entry.document.id != doc_id);
        len_before != entries.len()
    }

    fn search(&self, vector: &[f32], top_k: usize) -> Vec<RetrievedDocument> {
        assert_eq!(vector.len(), self.dimension, "query dimension mismatch");
        let entries = self
            .entries
            .read()
            .expect("parallel index poisoned while querying");
        if entries.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<RetrievedDocument> = entries
            .par_iter()
            .map(|entry| {
                let score = cosine_similarity(&entry.embedding, vector);
                RetrievedDocument {
                    document: entry.document.clone(),
                    score,
                }
            })
            .collect();

        scored
            .par_sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scored.truncate(scored.len().min(top_k));
        scored
    }

    fn snapshot(&self) -> Vec<PersistedEntry> {
        self.entries
            .read()
            .expect("parallel index poisoned during snapshot")
            .iter()
            .map(|entry| PersistedEntry {
                document: entry.document.clone(),
                embedding: entry.embedding.clone(),
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct IndexedDocument {
    document: Document,
    embedding: Vec<f32>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (lhs, rhs) in a.iter().zip(b) {
        dot += lhs * rhs;
        norm_a += lhs * lhs;
        norm_b += rhs * rhs;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

const SNAPSHOT_FILE: &str = "index.json";

/// High-level orchestrator that knows how to index file trees and expose a tool interface.
#[derive(Clone)]
pub struct Rag<M> {
    store: RagStore<M>,
    index_dir: Arc<RwLock<Option<PathBuf>>>,
}

impl<M> Rag<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a high-level RAG controller around a [`RagStore`].
    pub fn new(embedder: M) -> Self {
        Self {
            store: RagStore::new(embedder),
            index_dir: Arc::new(RwLock::new(None)),
        }
    }

    /// Sets the directory used to persist index snapshots.
    pub fn set_index_dir<P: Into<PathBuf>>(&self, dir: P) -> Result<()> {
        let path = dir.into();
        fs::create_dir_all(&path)?;
        *self.index_dir.write().expect("index dir poisoned") = Some(path);
        Ok(())
    }

    /// Loads a previously persisted index snapshot.
    pub fn load_index(&self) -> Result<usize> {
        let dir = self
            .index_dir
            .read()
            .expect("index dir poisoned")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("index directory not configured"))?;
        let snapshot_path = dir.join(SNAPSHOT_FILE);
        if !snapshot_path.exists() {
            return Ok(0);
        }
        let payload = fs::read_to_string(&snapshot_path)?;
        let entries: Vec<PersistedEntry> = serde_json::from_str(&payload)?;
        let count = entries.len();
        for entry in entries {
            self.store
                .insert_with_embedding(entry.document, entry.embedding)?;
        }
        Ok(count)
    }

    /// Kicks off indexing for an entire directory tree.
    ///
    /// Returns a [`IndexingJob`] that can be awaited for completion while also streaming progress.
    pub fn index_directory<P: AsRef<Path>>(&self, dir: P) -> Result<IndexingJob> {
        let dir_path = dir.as_ref().to_path_buf();
        let save_dir = self
            .index_dir
            .read()
            .expect("index dir poisoned")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("call set_index_dir before indexing"))?;
        let store = self.store.clone();
        let (progress_tx, progress_rx) = unbounded();
        let job = async move {
            let files = collect_files(&dir_path)?;
            let total = files.len();
            let _ = progress_tx
                .send(IndexProgress::new(0, total, None, IndexStage::Enumerating))
                .await;
            let mut indexed = 0usize;
            for (idx, path) in files.into_iter().enumerate() {
                let relative_id = path
                    .strip_prefix(&dir_path)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                let mut metadata = Metadata::new();
                metadata.insert("path".into(), path.display().to_string());
                let content = match fs::read_to_string(&path) {
                    Ok(text) => text,
                    Err(err) => {
                        let _ = progress_tx
                            .send(IndexProgress::new(
                                idx,
                                total,
                                Some(path.clone()),
                                IndexStage::Skipped {
                                    reason: err.to_string(),
                                },
                            ))
                            .await;
                        continue;
                    }
                };
                let doc = Document::with_metadata(relative_id, content, metadata);
                let _ = progress_tx
                    .send(IndexProgress::new(
                        idx,
                        total,
                        Some(path.clone()),
                        IndexStage::Embedding,
                    ))
                    .await;
                store.insert(doc.clone()).await?;
                indexed += 1;
                let _ = progress_tx
                    .send(IndexProgress::new(
                        idx + 1,
                        total,
                        Some(path),
                        IndexStage::Embedded,
                    ))
                    .await;
            }
            let _ = progress_tx
                .send(IndexProgress::new(
                    indexed,
                    total,
                    None,
                    IndexStage::Persisting,
                ))
                .await;
            persist_snapshot(&store, &save_dir)?;
            let _ = progress_tx
                .send(IndexProgress::new(
                    indexed,
                    total,
                    None,
                    IndexStage::Finished,
                ))
                .await;
            Ok(indexed)
        };
        Ok(IndexingJob::new(job, progress_rx))
    }

    /// Exposes the underlying store for manual control.
    pub fn store(&self) -> &RagStore<M> {
        &self.store
    }
}

impl<M> Tool for Rag<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        "indexed_knowledge_base".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Retrieve contextual documents from the local knowledge base".into()
    }

    type Arguments = RagToolArgs;

    fn call(
        &mut self,
        arguments: Self::Arguments,
    ) -> impl core::future::Future<Output = Result> + Send {
        let top_k = arguments.top_k;
        let query = arguments.query;
        let store = self.store.clone();
        async move {
            let hits = store.query(&query, top_k).await?;
            let response: Vec<RagToolResponse> = hits
                .into_iter()
                .map(|hit| RagToolResponse {
                    id: hit.document.id,
                    text: hit.document.text,
                    metadata: hit.document.metadata,
                    score: hit.score,
                })
                .collect();
            Ok(serde_json::to_string(&response)?)
        }
    }
}

/// Arguments available when the RAG engine is used as a tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RagToolArgs {
    /// Natural language query.
    pub query: String,
    /// Number of documents to retrieve (defaults to 3).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Debug, serde::Serialize)]
struct RagToolResponse {
    id: String,
    text: String,
    metadata: Metadata,
    score: f32,
}

const fn default_top_k() -> usize {
    3
}

/// Combined future + stream that drives directory indexing.
pub struct IndexingJob {
    completion: Pin<Box<dyn core::future::Future<Output = Result<usize>> + Send>>,
    completion_result: Option<Result<usize>>,
    progress: Pin<Box<AsyncReceiver<IndexProgress>>>,
}

impl IndexingJob {
    fn new<F>(future: F, progress: AsyncReceiver<IndexProgress>) -> Self
    where
        F: core::future::Future<Output = Result<usize>> + Send + 'static,
    {
        Self {
            completion: Box::pin(future),
            completion_result: None,
            progress: Box::pin(progress),
        }
    }
}

impl core::future::Future for IndexingJob {
    type Output = Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        if let Some(result) = this.completion_result.take() {
            return Poll::Ready(result);
        }
        this.completion.as_mut().poll(cx)
    }
}

impl Stream for IndexingJob {
    type Item = IndexProgress;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.completion_result.is_none() {
            if let Poll::Ready(result) = this.completion.as_mut().poll(cx) {
                this.completion_result = Some(result);
            }
        }
        this.progress.as_mut().poll_next(cx)
    }
}

/// Granular states emitted while indexing a directory.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Files processed successfully.
    pub processed: usize,
    /// Total files discovered.
    pub total: usize,
    /// Current file path (if applicable).
    pub path: Option<PathBuf>,
    /// Current indexing stage.
    pub stage: IndexStage,
}

impl IndexProgress {
    fn new(processed: usize, total: usize, path: Option<PathBuf>, stage: IndexStage) -> Self {
        Self {
            processed,
            total,
            path,
            stage,
        }
    }
}

/// Discrete phases emitted through [`IndexProgress`].
#[derive(Debug, Clone)]
pub enum IndexStage {
    /// Walking the directory to discover files.
    Enumerating,
    /// Embedding the current file.
    Embedding,
    /// Embedding finished for the current file.
    Embedded,
    /// Persisting the snapshot to disk.
    Persisting,
    /// Indexing completed successfully.
    Finished,
    /// File was skipped due to an error.
    Skipped { reason: String },
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
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

fn persist_snapshot<M>(store: &RagStore<M>, dir: &Path) -> Result<()>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    let data = store.snapshot();
    let payload = serde_json::to_vec_pretty(&data)?;
    fs::write(dir.join(SNAPSHOT_FILE), payload)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::EmbeddingModel;
    use futures_lite::StreamExt;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tempfile::tempdir;

    #[derive(Clone)]
    struct MockEmbedding {
        dimension: usize,
        embeds: Arc<AtomicUsize>,
    }

    impl EmbeddingModel for MockEmbedding {
        fn dim(&self) -> usize {
            self.dimension
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            self.embeds.fetch_add(1, AtomicOrdering::SeqCst);
            let mut vec = vec![0.0; self.dimension];
            for (idx, value) in vec.iter_mut().enumerate() {
                *value = ((text.len() + idx) % 10) as f32;
            }
            Ok(vec)
        }
    }

    #[tokio::test]
    async fn insert_and_query_documents() {
        let embeds = Arc::new(AtomicUsize::new(0));
        let embedder = MockEmbedding {
            dimension: 4,
            embeds: embeds.clone(),
        };
        let rag = RagStore::new(embedder);

        rag.insert(Document::new("doc-1", "rust programming language"))
            .await
            .unwrap();
        rag.insert(Document::new("doc-2", "advanced rag techniques"))
            .await
            .unwrap();

        let hits = rag.query("rag search", 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(matches!(hits[0].document.id.as_str(), "doc-1" | "doc-2"));
        assert!(hits[0].score.is_finite());
        assert!(embeds.load(AtomicOrdering::SeqCst) >= 3);
    }

    #[tokio::test]
    async fn delete_documents() {
        let embedder = MockEmbedding {
            dimension: 4,
            embeds: Arc::new(AtomicUsize::new(0)),
        };
        let rag = RagStore::new(embedder);
        rag.insert(Document::new("doc-a", "foo bar")).await.unwrap();
        assert!(rag.delete("doc-a"));
        assert!(!rag.delete("doc-a"));
        assert!(rag.query("foo", 5).await.unwrap().is_empty());
    }

    #[test]
    fn cosine_similarity_handles_zero_norms() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn rag_indexing_job_streams_progress() {
        let embedder = MockEmbedding {
            dimension: 4,
            embeds: Arc::new(AtomicUsize::new(0)),
        };
        let rag = Rag::new(embedder);
        let index_dir = tempdir().unwrap();
        rag.set_index_dir(index_dir.path()).unwrap();

        let data_dir = tempdir().unwrap();
        fs::write(
            data_dir.path().join("rust.txt"),
            "Rust lets you build fast and safe systems.",
        )
        .unwrap();
        fs::write(
            data_dir.path().join("rag.txt"),
            "RAG stitches embeddings together.",
        )
        .unwrap();

        let mut job = rag.index_directory(data_dir.path()).unwrap();
        let mut saw_finished = false;
        while let Some(progress) = job.next().await {
            if matches!(progress.stage, IndexStage::Finished) {
                saw_finished = true;
            }
        }
        let count = job.await.unwrap();
        assert_eq!(count, 2);
        assert!(saw_finished);
        assert!(index_dir.path().join(SNAPSHOT_FILE).exists());

        let mut rag_tool = rag.clone();
        let tool_output = rag_tool
            .call(RagToolArgs {
                query: "safe systems".into(),
                top_k: 1,
            })
            .await
            .unwrap();
        assert!(tool_output.contains("Rust"));
    }
}
