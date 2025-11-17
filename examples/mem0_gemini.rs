//! Mem0 extraction/update demo backed by Gemini.
//!
//! Inspired by and thanks to the “Memo: Building Production-Ready AI Agents with
//! Scalable Long-Term Memory” paper (<https://arxiv.org/abs/2504.19413>).
//! This example runs the dense-memory variant, so no graph database is required.
//!
//! Requirements:
//! - `GEMINI_API_KEY` must be present in your environment.
//! - Running the example triggers real LLM + embedding calls which incur usage.
//! - Invoke with `cargo run --example mem0_gemini`.

use aither_core::llm::Message;
use aither_gemini::GeminiBackend;
use aither_mem0::{Mem0, Mem0Config, OperationKind};
use anyhow::{Context, Result};

const TEXT_MODEL: &str = "gemini-2.0-flash";

#[tokio::main]
async fn main() -> Result<()> {
    let api_key =
        std::env::var("GEMINI_API_KEY").context("set GEMINI_API_KEY in your environment")?;

    let gemini = GeminiBackend::new(api_key).with_text_model(TEXT_MODEL);

    let config = Mem0Config::default()
        .with_recency_window(8)
        .with_similar_memories(6)
        .with_summary_refresh_interval(2);
    let mut mem0 = Mem0::with_config(gemini.clone(), gemini.clone(), config);

    let exchanges = vec![
        (
            Message::user(
                "Hey, I'm Priya from Vancouver. I'm vegetarian, avoid dairy, and I \
                 book-tracked a Garibaldi Lake hike for 14 July.",
            ),
            Message::assistant(
                "Got it—Vancouver-based Priya, vegetarian + dairy-free, hiking Garibaldi on 14 July.",
            ),
        ),
        (
            Message::user(
                "Could you remember that I started a remote data viz job at Helios Analytics last March? \
                 Also, peanuts make me break out.",
            ),
            Message::assistant(
                "I’ll remember the Helios remote role (since March) and flag your peanut allergy.",
            ),
        ),
        (
            Message::user(
                "Next month I’ll be in Kyoto for a workshop—prefer boutique hotels near trains.",
            ),
            Message::assistant(
                "I’ll search boutique hotels close to Kyoto Station before your workshop.",
            ),
        ),
    ];

    println!(
        "Feeding {} exchanges into the Mem0 pipeline…",
        exchanges.len()
    );
    for (idx, (previous, current)) in exchanges.into_iter().enumerate() {
        let changes = mem0.ingest_exchange(previous, current).await?;
        println!("Exchange {} produced {} change(s):", idx + 1, changes.len());
        for change in changes {
            match change.operation {
                OperationKind::Add | OperationKind::Update => {
                    if let Some(entry) = change.entry {
                        println!("  {:?}: {}", change.operation, entry.payload.content.trim());
                    }
                }
                OperationKind::Delete => {
                    if let Some(entry) = change.entry {
                        println!("  Delete removed {}", entry.payload.content.trim());
                    }
                }
                OperationKind::Noop => println!("  Noop (candidate skipped)"),
            }
        }
    }

    if let Some(summary) = mem0.summary() {
        println!("\nConversation summary snapshot:\n{summary}\n");
    }

    let recall_query = "What should I remember about Priya's dietary rules and work?";
    let hits = mem0.recall(recall_query, 3).await?;
    println!("Recall results for “{recall_query}”:");
    for entry in hits {
        println!(
            "- {} (speaker: {})",
            entry.payload.content.trim(),
            entry
                .payload
                .speaker
                .as_deref()
                .unwrap_or("unknown speaker")
        );
    }

    Ok(())
}
