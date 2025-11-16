//! Basic RAG flow using the in-memory store and a toy embedder.

use aither_core::{EmbeddingModel, Result};
use aither_rag::{Document, Metadata, RagStore};

#[derive(Clone)]
struct DemoEmbedder;

impl EmbeddingModel for DemoEmbedder {
    fn dim(&self) -> usize {
        4
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.dim()];
        for (idx, byte) in text.bytes().enumerate() {
            let bucket = idx % self.dim();
            vector[bucket] += byte as f32;
        }
        Ok(vector)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let rag = RagStore::new(DemoEmbedder);

    let docs = [
        Document::with_metadata(
            "rag-basics",
            "Retrieval-Augmented Generation uses embeddings to fetch context.",
            Metadata::from([("source".into(), "notes".into())]),
        ),
        Document::with_metadata(
            "chunking",
            "Chunking splits large files into overlapping passages for indexing.",
            Metadata::from([("source".into(), "docs".into())]),
        ),
        Document::with_metadata(
            "rust",
            "Rust focuses on performance and safety using ownership and borrowing.",
            Metadata::from([("source".into(), "book".into())]),
        ),
    ];

    for doc in docs {
        rag.insert(doc).await?;
    }

    let results = rag.query("How do I prep documents for RAG?", 2).await?;
    println!("Top matches:");
    for (rank, hit) in results.iter().enumerate() {
        println!(
            "{rank}: {} (score = {:.3}) - {}",
            hit.document.id, hit.score, hit.document.text
        );
    }

    Ok(())
}
