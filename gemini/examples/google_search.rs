//! Example of using the native Google Search tool with aither-gemini.
//!
//! To run this example:
//! 1. Set your `GEMINI_API_KEY` environment variable.
//! 2. Run `cargo run --example google_search`
//!
//! This example demonstrates how to answer a question that requires up-to-date
//! information by enabling the native Google Search tool.

use aither_core::llm::{Event, LanguageModel, Message, model::Parameters};
use aither_gemini::Gemini;
use futures_lite::StreamExt;
use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable not set");

    let backend = Gemini::new(api_key);

    println!("Welcome to the interactive Google Search example!");
    println!("Type your questions, or 'quit'/'exit' to stop.");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Exiting...");
            break;
        }

        let messages = vec![Message::user(input)];

        let request = aither_core::llm::LLMRequest::new(messages)
            .with_parameters(Parameters::default().websearch(true));

        println!("Gemini (searching): ");
        let mut stream = backend.respond(request);
        let mut full_response = String::new();
        while let Some(event) = stream.next().await {
            if let Event::Text(text) = event? {
                print!("{text}");
                io::stdout().flush()?;
                full_response.push_str(&text);
            }
        }
        println!();
    }

    Ok(())
}
