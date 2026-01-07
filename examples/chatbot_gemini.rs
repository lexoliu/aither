//! Streaming CLI chatbot that talks to `gemini-flash-latest`.
//! Requires `GEMINI_API_KEY` in your environment.
//! Run: `cargo run --example chatbot_gemini`

use std::io::{self, Write};

use aither::llm::{Event, LLMRequest};
use aither_core::{LanguageModel, llm::Message};
use aither_gemini::Gemini;
use anyhow::{Context, Result};
use futures_lite::{StreamExt, pin};

const MODEL: &str = "gemini-flash-latest";

#[tokio::main]
async fn main() -> Result<()> {
    let api_key =
        std::env::var("GEMINI_API_KEY").context("set GEMINI_API_KEY in your environment")?;

    let model = Gemini::new(api_key).with_text_model(MODEL);
    let mut messages = vec![Message::system(
        "You are a friendly assistant. Keep replies short unless asked otherwise.",
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
        if matches!(user, ":exit" | ":quit") {
            break;
        }

        messages.push(Message::user(user));

        println!("Gemini>");
        let response = {
            let stream = model.respond(LLMRequest::new(messages.as_slice()));
            pin!(stream);
            let mut response = String::new();
            while let Some(chunk) = stream.next().await {
                let event = chunk?;
                if let Event::Text(text) = event {
                    print!("{text}");
                    io::stdout().flush().ok();
                    response.push_str(&text);
                }
            }
            println!();
            response
        };
        messages.push(Message::assistant(response));
    }

    println!("Goodbye!");
    Ok(())
}
