//! Basic RAG flow using the in-memory store and a toy embedder.

use aither_core::{EmbeddingModel, Result, llm::Tool};
use aither_rag::{IndexStage, Rag, RagToolArgs};
use futures_lite::StreamExt;
use std::{env, fs, path::PathBuf};

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
    let rag = Rag::new(DemoEmbedder);
    let working_dir = prepare_source_tree()?;
    let index_dir = working_dir.join("index");
    rag.set_index_dir(&index_dir)?;

    println!("Indexing {}", working_dir.display());
    let mut job = rag.index_directory(working_dir.join("notes"))?;
    while let Some(progress) = job.next().await {
        match progress.stage {
            IndexStage::Enumerating => println!("Scanning files..."),
            IndexStage::Embedding | IndexStage::Embedded => {
                if let Some(path) = progress.path.as_ref() {
                    println!(
                        "[{}/{}] Indexed {}",
                        progress.processed,
                        progress.total,
                        path.display()
                    );
                }
            }
            IndexStage::Persisting => println!("Persisting snapshot..."),
            IndexStage::Finished => println!("Done indexing!"),
            IndexStage::Skipped { ref reason } => {
                if let Some(path) = progress.path.as_ref() {
                    eprintln!("Skipped {}: {}", path.display(), reason);
                }
            }
        }
    }
    let indexed = job.await?;
    println!(
        "Indexed {indexed} files. Snapshot stored at {}",
        index_dir.display()
    );

    let mut tool_rag = rag.clone();
    let response = tool_rag
        .call(RagToolArgs {
            query: "How do I prep documents for RAG?".into(),
            top_k: 2,
        })
        .await?;
    println!("Tool response: {response}");

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
        "Retrieval-Augmented Generation uses embeddings plus vector search.",
    )?;
    fs::write(
        notes_dir.join("chunking.md"),
        "Chunking splits large files into overlapping windows for indexing.",
    )?;
    fs::write(
        notes_dir.join("rust.md"),
        "Rust focuses on performance and safety using ownership and borrowing.",
    )?;
    Ok(base)
}
