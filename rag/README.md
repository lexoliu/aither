<div align="center">

# aither-rag

High-performance Retrieval-Augmented Generation plumbing for any embedding model.

</div>

`aither-rag` pairs the provider-agnostic [`EmbeddingModel`](https://docs.rs/aither-core/latest/aither_core/embedding/trait.EmbeddingModel.html) trait with a parallel in-memory vector index. It gives you a battery-included RAG store that can ingest documents, remove them, and retrieve relevant context for prompts—all without standing up an external database.

## Features

- 🔌 **Plug in any embedder** – Works with `aither-core`/`aither` embedders (OpenAI, Gemini, local adapters, etc.).
- ⚡ **Built-in vector DB** – SIMD-friendly cosine similarity with Rayon parallelism scales to hundreds of thousands of vectors.
- 🧱 **Tiny API surface** – Only three async methods: `.insert()`, `.delete()`, `.query()`.
- 🧾 **Metadata-first** – Store arbitrary key/value metadata with each chunk for filtering and citations.
- 🧪 **Test friendly** – Ships with an in-memory mock embedder example; no network calls required.

## Quick Start

```rust
use aither_rag::{Document, RagStore};
use aither_core::EmbeddingModel;

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
    let rag = RagStore::new(DemoEmbedder);
    rag.insert(Document::new("rust-book", "Rust is a systems programming language.")).await?;
    let hits = rag.query("What is Rust?", 1).await?;
    println!("Top result: {} ({:.3})", hits[0].document.id, hits[0].score);
    Ok(())
}
```

## Example

A complete runnable demo lives in [`examples/basic.rs`](examples/basic.rs):

```bash
cargo run -p aither-rag --example basic
```

It wires a tiny deterministic embedder into `RagStore`, ingests a few knowledge snippets, and prints the retrieval hits for a sample question.

## API Overview

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
