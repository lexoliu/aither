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

use aither_core::{Result, embedding::EmbeddingModel};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    sync::{Arc, RwLock},
};

/// Key/value metadata attached to stored documents.
pub type Metadata = BTreeMap<String, String>;

/// Represents a chunk of text tracked inside the RAG store.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RetrievedDocument {
    /// Stored document.
    pub document: Document,
    /// Cosine similarity score (1.0 = identical).
    pub score: f32,
}

/// High-level entry point for retrieval augmented workflows.
///
/// It owns both an embedding model and the internal vector index. All operations are async-friendly
/// and return the workspace-wide [`Result`].
#[derive(Clone)]
pub struct RagStore<M> {
    embedder: Arc<M>,
    index: Arc<ParallelIndex>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::EmbeddingModel;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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
}
