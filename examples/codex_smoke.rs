//! Smoke test for the `ChatGPT` Codex subscription backend.
//!
//! Reads credentials from `~/.codex/auth.json` (written by `codex login`),
//! lists the backend's model catalog, and runs one Responses request.
//! Run: `cargo run --example codex_smoke -- [model-slug]`

use aither_codex::auth::CodexCredentials;
use aither_codex::{CodexProvider, LanguageModel, LanguageModelProvider};
use aither_core::llm::{Event, LLMRequest, Message};
use anyhow::{Context, Result};
use futures_lite::StreamExt;
use serde::Deserialize;

/// The subset of codex-cli's `auth.json` needed here.
#[derive(Deserialize)]
struct CodexCliAuth {
    tokens: CodexCliTokens,
}

#[derive(Deserialize)]
struct CodexCliTokens {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    account_id: Option<String>,
}

fn load_credentials() -> Result<CodexCredentials> {
    let path = std::env::home_dir()
        .context("no home dir")?
        .join(".codex/auth.json");
    let raw = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    let auth: CodexCliAuth = serde_json::from_str(&raw)?;
    let mut credentials =
        CodexCredentials::new(auth.tokens.access_token, auth.tokens.refresh_token);
    if let Some(id_token) = auth.tokens.id_token {
        credentials = credentials.with_id_token(id_token);
    }
    if auth.tokens.account_id.is_some() {
        credentials.account_id = auth.tokens.account_id;
    }
    Ok(credentials)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let credentials = load_credentials()?;
    println!("account_id: {:?}", credentials.account_id);
    println!("residency: {:?}", credentials.residency);

    let provider = CodexProvider::new(credentials.clone());
    let models = provider.list_models().await?;
    println!("\n=== {} models ===", models.len());
    for model in &models {
        println!("  {} (ctx {})", model.slug, model.context_length);
    }

    let slug = std::env::args()
        .nth(1)
        .or_else(|| models.first().map(|m| m.slug.clone()))
        .context("no model available")?;
    println!("\n=== respond() on {slug} ===");

    let model = provider.get_model(&slug).await?;
    let request = LLMRequest::new(vec![Message::user("Reply with exactly one word: pong")]);
    let mut stream = std::pin::pin!(model.respond(request));
    while let Some(event) = stream.next().await {
        match event? {
            Event::Text(text) => print!("{text}"),
            other => println!("\n[event: {other:?}]"),
        }
    }
    println!();
    Ok(())
}
