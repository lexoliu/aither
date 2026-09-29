//! Tool trait implementation for RAG.

use aither_core::embedding::EmbeddingModel;
use aither_core::llm::ToolContext;
use aither_core::llm::tool::{Tool, ToolResult};
use schemars::JsonSchema;
use serde::Deserialize;
use std::borrow::Cow;

use crate::rag::Rag;
use crate::types::Metadata;

/// Arguments for the RAG search tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RagToolArgs {
    /// The search query.
    pub query: String,
    /// Number of results to return (defaults to 5).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

const fn default_top_k() -> usize {
    5
}

/// Response from the RAG search tool.
#[derive(Debug, serde::Serialize)]
pub struct RagToolResponse {
    /// Chunk ID.
    pub id: String,
    /// Chunk text content.
    pub text: String,
    /// Chunk metadata.
    pub metadata: Metadata,
    /// Similarity score.
    pub score: f32,
}

impl<M> Tool for Rag<M>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        "rag_search".into()
    }

    type Arguments = RagToolArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let results = self
            .store()
            .search_with_k(&arguments.query, arguments.top_k)
            .await?;

        let response: Vec<RagToolResponse> = results
            .into_iter()
            .map(|r| RagToolResponse {
                id: r.chunk.id,
                text: r.chunk.text,
                metadata: r.chunk.metadata,
                score: r.score,
            })
            .collect();

        ToolResult::json(&response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Document;
    use aither_core::EmbeddingModel;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

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

        fn embed(
            &self,
            text: &str,
        ) -> impl std::future::Future<Output = aither_core::Result<Vec<f32>>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut vec = vec![0.0; self.dimension];
            for (idx, value) in vec.iter_mut().enumerate() {
                let digit = u8::try_from((text.len() + idx) % 10).expect("modulo result fits u8");
                *value = f32::from(digit) / 10.0;
            }
            std::future::ready(Ok(vec))
        }
    }

    #[tokio::test]
    async fn tool_interface() {
        let embedder = MockEmbedder::new(4);
        let index_dir = tempdir().unwrap();

        let rag = Rag::builder(embedder)
            .index_path(index_dir.path().join("index.redb"))
            .build()
            .unwrap();

        // Insert some data
        rag.insert(Document::new("doc1", "Rust programming language"))
            .await
            .unwrap();

        // Use tool interface
        let args = RagToolArgs {
            query: "rust".to_string(),
            top_k: 5,
        };

        let result = rag.call(args, ToolContext::new()).await.unwrap();
        assert!(result.render_for_model().unwrap().contains("Rust"));
    }
}
