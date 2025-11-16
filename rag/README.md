<div align="center">

# aither-rag

High-performance Retrieval-Augmented Generation plumbing for any embedding model.

</div>

`aither-rag` pairs the provider-agnostic [`EmbeddingModel`](https://docs.rs/aither-core/latest/aither_core/embedding/trait.EmbeddingModel.html) trait with a parallel in-memory vector index. It gives you a battery-included RAG store that can ingest documents, remove them, and retrieve relevant context for prompts—all without standing up an external database.

## Features

- 🔌 **Plug in any embedder** – Works with `aither-core`/`aither` embedders (OpenAI, Gemini, local adapters, etc.).
- ⚡ **Built-in vector DB** – SIMD-friendly cosine similarity with Rayon parallelism scales to hundreds of thousands of vectors.
- 📁 **Directory ingestion** – `Rag::index_directory` walks any folder, indexes files, and streams progress updates.
- 💾 **Snapshot persistence** – Store embeddings to disk with `set_index_dir`/`load_index`.
- 🧰 **LLM Tool ready** – `Rag` implements the `Tool` trait so models can call it directly.
- 🧱 **Low-level access** – Use `RagStore` for manual control when you only need insert/delete/query primitives.

## Quick Start

```rust
use aither_rag::{IndexStage, Rag, RagToolArgs};
use aither_core::EmbeddingModel;
use futures_lite::StreamExt;

#[derive(Clone)]
struct DemoEmbedder;

impl EmbeddingModel for DemoEmbedder {
    fn dim(&self) -> usize { 4 }
    async fn embed(&self, text: &str) -> aither_core::Result<Vec<f32>> {
        let mut vec = vec![0.0; self.dim()];
        for (idx, ch) in text.bytes().enumerate() {
            vec[idx % 4] += ch as f32;
        }
        Ok(vec)
    }
}

#[tokio::main]
async fn main() -> aither_core::Result<()> {
    let rag = Rag::new(DemoEmbedder);
    rag.set_index_dir("./.rag-index")?;

    let mut job = rag.index_directory("./notes")?;
    while let Some(progress) = job.next().await {
        if let Some(path) = progress.path.as_ref() {
            println!("[{}/{}] {:?} {}", progress.processed, progress.total, progress.stage, path.display());
        }
    }
    job.await?;

    let response = rag
        .clone()
        .call(RagToolArgs {
            query: "What is Rust?".into(),
            top_k: 1,
        })
        .await?;
    println!("LLM-friendly response: {response}");
    Ok(())
}
```

## Example

A complete runnable demo lives in [`examples/basic.rs`](examples/basic.rs):

```bash
cargo run -p aither-rag --example basic
```

It wires a tiny deterministic embedder into the high-level `Rag` helper, indexes a temporary directory while streaming progress, persists the snapshot, and finishes by exercising the tool interface.

## API Overview

### High-level `Rag`

| Method | Description |
| ------ | ----------- |
| `Rag::set_index_dir(path)` | Configure where snapshots are written (`index.json`). |
| `Rag::load_index()` | Hydrate the in-memory store from the snapshot directory. |
| `Rag::index_directory(path)` | Walk a directory tree; returns an `IndexingJob` implementing both `Future` and `Stream`. |
| `Rag::call(args)` | Invoke the tool interface—perfect for LLM function calling. |

### Low-level `RagStore`

| Method | Description |
| ------ | ----------- |
| `RagStore::insert(document)` | Embeds the document text and upserts it into the vector store. |
| `RagStore::delete(id)` | Removes a document by identifier; returns `true` when something was deleted. |
| `RagStore::query(query, top_k)` | Embeds the query text and returns the best matching documents with cosine similarity scores. |
| `RagStore::embedder()` | Clones the inner embedder handle so you can perform custom chunking or batched embeddings. |

Need persistent storage? Swap the provided `ParallelIndex` with your own backend while reusing the public `Document` and `RetrievedDocument` types.

## Status & Contributing

The crate is part of the `aither` workspace and follows the same lint/test gates:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features --workspace -- -D warnings
cargo test -p aither-rag --all-features
```

Issues and PRs welcome! Use the workspace’s AGENTS.md for style and testing guidelines.
