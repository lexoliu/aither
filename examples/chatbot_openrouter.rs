//! Streaming CLI chatbot using `OpenRouter` (defaults to `DeepSeek`).
//! Requires `OPENROUTER_API_KEY` in your environment.
//! Run: `cargo run --example chatbot_openrouter`

use std::io::{self, Write};

use aither::llm::LLMRequest;
use aither_core::{LanguageModel, llm::Message};
use aither_openai::{GPT5_MINI, OpenAI};
use anyhow::{Context, Result};
use futures_lite::{StreamExt, pin};

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .context("set OPENROUTER_API_KEY in your environment")?;

    let model = OpenAI::openrouter(api_key).with_model(GPT5_MINI);
    let mut messages = vec![Message::system(
        "You are a friendly Rust assistant. Answer concisely.",
    )];
    let stdin = io::stdin();

    loop {
        print!("You> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        let read = stdin.read_line(&mut input)?;
        if read == 0 {
            break;
        }
        let user = input.trim();
        if user.is_empty() {
            continue;
        }
        if user.eq_ignore_ascii_case(":exit") || user.eq_ignore_ascii_case(":quit") {
            break;
        }

        messages.push(Message::user(user));

        println!("Model>");
        let response = {
            let stream = model.respond(LLMRequest::new(messages.as_slice()));
            pin!(stream);
            let mut response = String::new();
            while let Some(chunk) = stream.next().await {
                let text = chunk?;
                print!("{text}");
                io::stdout().flush().ok();
                response.push_str(&text);
            }
            println!();
            response
        };
        messages.push(Message::assistant(response));
    }

    println!("Goodbye!");
    Ok(())
}
