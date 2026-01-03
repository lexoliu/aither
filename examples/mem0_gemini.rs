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
use aither_gemini::Gemini;
use aither_mem0::{Config, Mem0, store::InMemoryStore};
use anyhow::{Context, Result};

const TEXT_MODEL: &str = "gemini-flash-latest";

#[tokio::main]
async fn main() -> Result<()> {
    let api_key =
        std::env::var("GEMINI_API_KEY").context("set GEMINI_API_KEY in your environment")?;

    let gemini = Gemini::new(api_key).with_text_model(TEXT_MODEL);

    let config = Config {
        retrieve_count: 3,
        user_id: Some("priya_123".to_string()),
        agent_id: None,
    };

    let store = InMemoryStore::new();
    let mem0 = Mem0::new(gemini.clone(), gemini.clone(), store, config);

    let exchanges = vec![
        vec![
            Message::user(
                "Hey, I'm Priya from Vancouver. I'm vegetarian, avoid dairy, and I \
                 book-tracked a Garibaldi Lake hike for 14 July.",
            ),
            Message::assistant(
                "Got it—Vancouver-based Priya, vegetarian + dairy-free, hiking Garibaldi on 14 July.",
            ),
        ],
        vec![
            Message::user(
                "Could you remember that I started a remote data viz job at Helios Analytics last March? \
                 Also, peanuts make me break out.",
            ),
            Message::assistant(
                "I’ll remember the Helios remote role (since March) and flag your peanut allergy.",
            ),
        ],
        vec![
            Message::user(
                "Next month I’ll be in Kyoto for a workshop—prefer boutique hotels near trains.",
            ),
            Message::assistant(
                "I’ll search boutique hotels close to Kyoto Station before your workshop.",
            ),
        ],
    ];

    println!(
        "Feeding {} exchanges into the Mem0 pipeline…",
        exchanges.len()
    );

    for (idx, msgs) in exchanges.into_iter().enumerate() {
        println!("Processing exchange {}...", idx + 1);
        mem0.add(&msgs).await?;
    }

    let recall_query = "What should I remember about Priya's dietary rules and work?";
    println!("\nRecalling memories for query: “{recall_query}”");
    let hits = mem0.search(recall_query, 5).await?;
    for entry in hits {
        println!(
            "- {} (Score: {:.2})",
            entry.memory.content.trim(),
            entry.score
        );
    }

    Ok(())
}
