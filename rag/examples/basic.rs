//! Basic RAG flow using the HNSW index and a toy embedder.

use aither_core::{EmbeddingModel, Result, llm::Tool};
use aither_rag::{IndexStage, Rag, RagToolArgs};
use std::{env, fs, path::PathBuf};

#[derive(Clone)]
struct DemoEmbedder;

impl EmbeddingModel for DemoEmbedder {
    fn dim(&self) -> usize {
        4
    }

    async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.dim()];
        for (idx, byte) in text.bytes().enumerate() {
            let bucket = idx % self.dim();
            vector[bucket] += byte as f32;
        }
        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        Ok(vector)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let working_dir = prepare_source_tree()?;
    let index_path = working_dir.join("index.redb");

    // Create RAG instance with custom configuration
    let mut rag = Rag::builder(DemoEmbedder)
        .index_path(&index_path)
        .top_k(3)
        .auto_save(true)
        .build()
        .expect("Failed to create RAG instance");

    println!("Indexing {}", working_dir.display());
    let indexed = rag
        .index_directory_with_progress(working_dir.join("notes"), |progress| match progress.stage {
            IndexStage::Scanning => println!("Scanning files..."),
            IndexStage::Embedding | IndexStage::Indexing => {
                if let Some(path) = progress.current_file.as_ref() {
                    println!(
                        "[{}/{}] Indexing {}",
                        progress.processed,
                        progress.total,
                        path.display()
                    );
                }
            }
            IndexStage::Saving => println!("Saving index..."),
            IndexStage::Done => println!("Done indexing!"),
            IndexStage::Skipped { ref reason } => {
                if let Some(path) = progress.current_file.as_ref() {
                    eprintln!("Skipped {}: {}", path.display(), reason);
                }
            }
            _ => {}
        })
        .await?;

    println!(
        "Indexed {indexed} files. Index stored at {}",
        index_path.display()
    );

    // Use the RAG as a tool
    let response = rag
        .call(RagToolArgs {
            query: "How do I prep documents for RAG?".into(),
            top_k: 2,
        })
        .await?;
    println!("\nTool response:\n{response}");

    // Direct search
    println!("\nDirect search results:");
    let results = rag.search("rust safety").await?;
    for result in results {
        println!(
            "  [{:.3}] {}: {}",
            result.score,
            result.chunk.id,
            &result.chunk.text[..50.min(result.chunk.text.len())]
        );
    }

    Ok(())
}

fn prepare_source_tree() -> Result<PathBuf> {
    let base = env::temp_dir().join("aither-rag-demo");
    if base.exists() {
        fs::remove_dir_all(&base)?;
    }
    let notes_dir = base.join("notes");
    fs::create_dir_all(&notes_dir)?;

    fs::write(
        notes_dir.join("rag.txt"),
        "Retrieval-Augmented Generation uses embeddings plus vector search to provide context to LLMs.",
    )?;
    fs::write(
        notes_dir.join("chunking.md"),
        "Chunking splits large files into overlapping windows for indexing. This helps with retrieval accuracy.",
    )?;
    fs::write(
        notes_dir.join("rust.md"),
        "Rust focuses on performance and safety using ownership and borrowing. It prevents memory bugs at compile time.",
    )?;

    Ok(base)
}
