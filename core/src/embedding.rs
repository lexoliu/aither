//! # Embedding Module
//!
//! This module provides types and traits for working with text embeddings.
//!
//! ## What are Embeddings?
//!
//! Embeddings are dense vector representations of text that capture semantic meaning.
//! They transform human-readable text into numerical vectors that machine learning
//! models can process effectively. Similar texts produce similar embedding vectors,
//! making them useful for:
//!
//! - **Semantic search**: Finding relevant documents based on meaning rather than exact keywords
//! - **Text similarity**: Measuring how similar two pieces of text are
//! - **Classification**: Categorizing text based on content
//! - **Clustering**: Grouping similar texts together
//! - **Recommendation systems**: Finding related content
//!
//! ## Embedding Models
//!
//! An embedding model is a neural network that has been trained to convert text into
//! meaningful vector representations. Different models have different characteristics:
//!
//! - **Dimension**: The length of the embedding vector (e.g., 768, 1536)
//! - **Domain**: Some models are optimized for specific types of content
//! - **Performance**: Trade-offs between speed, accuracy, and resource usage
//!
//! Popular embedding models include:
//! - OpenAI's `text-embedding-ada-002` (1536 dimensions)
//! - Sentence Transformers like `all-MiniLM-L6-v2` (384 dimensions)
//! - Cohere's embedding models
//!
//! ## Usage
//!
//! This module provides the [`EmbeddingModel`] trait that abstracts over different
//! embedding implementations, allowing you to switch between providers while
//! maintaining the same interface.
//!
//! ```rust
//! use aither::EmbeddingModel;
//!
//! async fn example<T: EmbeddingModel>(model: &T) -> aither::Result<()> {
//!     // Get the embedding dimension
//!     let dim = model.dim();
//!     println!("Model produces {}-dimensional embeddings", dim);
//!
//!     // Convert text to embedding
//!     let embedding = model.embed("Hello, world!").await?;
//!     assert_eq!(embedding.len(), dim);
//!
//!     Ok(())
//! }
//! ```

use alloc::vec::Vec;
use core::future::Future;

/// A type alias for an embedding vector of 32-bit floats.
///
/// Embeddings are dense vector representations where each dimension captures
/// different semantic features of the input text. The vector length is determined
/// by the embedding model's architecture.
pub type Embedding = Vec<f32>;

/// Converts text to vector representations.
///
/// This trait provides a unified interface for different embedding model implementations,
/// allowing you to switch between providers (`OpenAI`, `Cohere`, `Hugging Face`, etc.) while
/// maintaining the same API.
///
/// See the [module documentation](crate::embedding) for more details on embeddings and their use cases.
///
/// # Implementation Requirements
///
/// - The [`embed`](EmbeddingModel::embed) method must return vectors with length equal to [`dim`](EmbeddingModel::dim)
/// - Embeddings should be normalized if the underlying model requires it
/// - The implementation should handle errors gracefully (network issues, API limits, etc.)
///
/// # Example
///
/// ```rust
/// use aither::EmbeddingModel;
///
/// struct MyEmbedding {
///     api_key: String,
/// }
///
/// impl EmbeddingModel for MyEmbedding {
///     fn dim(&self) -> usize {
///         1536 // OpenAI text-embedding-ada-002 dimension
///     }
///     
///     async fn embed(&self, text: &str) -> aither::Result<Vec<f32>> {
///         // In a real implementation, this would call the embedding API
///         Ok(vec![0.0; self.dim()])
///     }
/// }
///
/// # tokio_test::block_on(async {
/// let model = MyEmbedding { api_key: "sk-...".to_string() };
/// let embedding = model.embed("The quick brown fox").await.unwrap();
/// assert_eq!(embedding.len(), 1536);
/// # });
/// ```
///
/// # Performance Considerations
///
/// - Batch multiple texts when possible to reduce API calls
/// - Consider caching embeddings for frequently used texts
/// - Be aware of rate limits when using cloud-based embedding services
pub trait EmbeddingModel: Send + Sized + Send + Sync {
    /// Returns the embedding vector dimension.
    ///
    /// This value determines the length of vectors returned by [`embed`](EmbeddingModel::embed).
    /// Common dimensions include:
    /// - 384 (`Sentence Transformers MiniLM`)
    /// - 768 (`BERT-base`)
    /// - 1536 (`OpenAI text-embedding-ada-002`)
    /// - 3072 (`OpenAI text-embedding-3-large`)
    fn dim(&self) -> usize;

    /// Converts text to an embedding vector.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to embed. Can be a word, sentence, paragraph, or document.
    ///
    /// # Returns
    ///
    /// A [`Vec<f32>`] with length equal to [`Self::dim`](EmbeddingModel::dim).
    /// The vector represents the semantic meaning of the input text in high-dimensional space.
    fn embed(&self, text: &str) -> impl Future<Output = crate::Result<Vec<f32>>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    struct MockEmbeddingModel {
        dimension: usize,
    }

    impl EmbeddingModel for MockEmbeddingModel {
        fn dim(&self) -> usize {
            self.dimension
        }

        #[allow(clippy::cast_precision_loss)]
        async fn embed(&self, text: &str) -> crate::Result<Vec<f32>> {
            // Create a simple mock embedding based on text length
            let mut embedding = vec![0.0; self.dimension];
            let text_len = text.len();

            for (i, value) in embedding.iter_mut().enumerate() {
                *value = (text_len + i) as f32 * 0.01;
            }

            Ok(embedding)
        }
    }

    #[tokio::test]
    async fn embedding_model_dimension() {
        let model = MockEmbeddingModel { dimension: 768 };
        assert_eq!(model.dim(), 768);
    }

    #[tokio::test]
    async fn embedding_generation() {
        let model = MockEmbeddingModel { dimension: 4 };
        let embedding = model.embed("test").await.unwrap();

        assert_eq!(embedding.len(), 4);
        assert!((embedding[0] - 0.04).abs() < f32::EPSILON); // text length 4 + index 0 = 4 * 0.01
        assert!((embedding[1] - 0.05).abs() < f32::EPSILON); // text length 4 + index 1 = 5 * 0.01
        assert!((embedding[2] - 0.06).abs() < f32::EPSILON); // text length 4 + index 2 = 6 * 0.01
        assert!((embedding[3] - 0.07).abs() < f32::EPSILON); // text length 4 + index 3 = 7 * 0.01
    }

    #[tokio::test]
    #[allow(clippy::float_cmp)]
    async fn embedding_different_texts() {
        let model = MockEmbeddingModel { dimension: 2 };

        let embedding1 = model.embed("a").await.unwrap();
        let embedding2 = model.embed("ab").await.unwrap();

        assert_eq!(embedding1.len(), 2);
        assert_eq!(embedding2.len(), 2);

        // Different text lengths should produce different embeddings
        assert_ne!(embedding1[0], embedding2[0]);
        assert_ne!(embedding1[1], embedding2[1]);
    }

    #[tokio::test]
    #[allow(clippy::float_cmp)]
    async fn embedding_empty_text() {
        let model = MockEmbeddingModel { dimension: 3 };
        let embedding = model.embed("").await.unwrap();

        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0], 0.00); // length 0 + index 0 = 0 * 0.01
        assert_eq!(embedding[1], 0.01); // length 0 + index 1 = 1 * 0.01
        assert_eq!(embedding[2], 0.02); // length 0 + index 2 = 2 * 0.01
    }

    #[tokio::test]
    async fn embedding_large_dimension() {
        let model = MockEmbeddingModel { dimension: 1536 }; // Common OpenAI dimension
        let embedding = model.embed("test text").await.unwrap();

        assert_eq!(embedding.len(), 1536);
        assert!((embedding[0] - 0.09).abs() < f32::EPSILON); // text length 9 + index 0 = 9 * 0.01
        assert!((embedding[1535] - 15.44).abs() < 0.01); // text length 9 + index 1535 = 1544 * 0.01
    }
}
