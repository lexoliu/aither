//! Production-ready Retrieval-Augmented Generation (RAG) crate.
//!
//! This crate provides a complete RAG solution with:
//! - **HNSW indexing** for fast approximate nearest neighbor search
//! - **Text chunking** strategies (fixed-size and sentence-based)
//! - **Persistence** backends (rkyv binary and redb embedded database)
//! - **Deduplication** using content hashing
//! - **Tool integration** for LLM function calling
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use aither_rag::{Rag, Document};
//! use aither_core::EmbeddingModel;
//!
//! async fn example<E: EmbeddingModel + Send + Sync + 'static>(embedder: E) {
//!     // Create RAG instance with defaults
//!     let rag = Rag::new(embedder);
//!
//!     // Index a directory
//!     let job = rag.index_directory("./docs").unwrap();
//!     let _count = job.await.unwrap();
//!
//!     // Search
//!     let results = rag.search("query").await.unwrap();
//!     for result in results {
//!         println!("{}: {:.2}", result.chunk.id, result.score);
//!     }
//! }
//! ```
//!
//! # Custom Configuration
//!
//! ```rust,no_run
//! use aither_rag::{Rag, Document};
//! use aither_rag::chunking::SentenceChunker;
//! use aither_core::EmbeddingModel;
//!
//! async fn example<E: EmbeddingModel + Send + Sync + 'static>(embedder: E) {
//!     let rag = Rag::builder(embedder)
//!         .index_path("./custom-index.redb")
//!         .sentence_chunking(1024)
//!         .similarity_threshold(0.7)
//!         .top_k(10)
//!         .deduplication(true)
//!         .build()
//!         .unwrap();
//! }
//! ```
//!
//! # Architecture
//!
//! The crate is organized into several modules:
//!
//! - [`chunking`] - Text chunking strategies
//! - [`index`] - Vector index implementations
//! - [`persistence`] - Storage backends
//! - [`config`] - Configuration types
//!
//! The main entry points are:
//!
//! - [`Rag`] - High-level orchestrator with directory indexing and persistence
//! - [`RagStore`] - Lower-level store for manual control

pub mod chunking;
pub mod config;
mod dedup;
pub mod error;
pub mod index;
pub mod indexing;
pub mod persistence;
mod rag;
mod store;
mod tool;
pub mod types;

// Re-exports for convenience
pub use chunking::{Chunker, FixedSizeChunker, SentenceChunker};
pub use config::{RagConfig, RagConfigBuilder};
pub use error::{RagError, Result};
pub use index::{HnswIndex, VectorIndex};
pub use indexing::{IndexProgress, IndexStage, IndexingJob};
pub use persistence::{Persistence, RedbPersistence, RkyvPersistence};
pub use rag::{Rag, RagBuilder};
pub use store::RagStore;
pub use tool::{RagToolArgs, RagToolResponse};
pub use types::{Chunk, Document, IndexEntry, Metadata, SearchResult};

