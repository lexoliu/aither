//! # Tool Macro Examples
//!
//! This example demonstrates various ways to use the `#[tool]` macro from `aither-derive`
//! to convert async functions into AI tools.
//!
//! Run this example with: `cargo run --example tool_macro`

#![allow(clippy::missing_errors_doc)]
#![allow(missing_docs)]
#![allow(clippy::unused_async)]
use aither::Result;
use aither_derive::tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Basic tool example - no parameters needed
#[tool(description = "Get current time in UTC")]
pub async fn time() -> Result<&'static str> {
    Ok("2023-10-01T12:00:00Z")
}

/// Return type for search results
#[derive(Debug, Serialize)]
pub struct SearchResult {
    title: String,
    url: String,
}

// Tool with multiple simple parameters
#[tool(description = "Search on the web")]
pub async fn search(keywords: Vec<String>, max_results: u32) -> Result<Vec<SearchResult>> {
    // Simulate a search result
    let results = keywords
        .into_iter()
        .take(max_results as usize)
        .map(|keyword| SearchResult {
            title: format!("Result for {keyword}",),
            url: format!("https://example.com/search?q={keyword}",),
        })
        .collect();
    Ok(results)
}

/// Arguments for image generation with comprehensive documentation
#[derive(Debug, JsonSchema, Deserialize)]
pub struct GenerateImageArgs {
    /// The prompt for the image generation.
    pub prompt: String,
    /// Optional images to guide the generation process.
    pub images: Vec<String>,
}

// Tool with complex documented arguments using a single struct parameter
#[tool(description = "Generate an image from a text prompt")]
pub async fn generate_image(args: GenerateImageArgs) -> aither::Result<String> {
    let file_name = format!("image_{}.png", args.prompt.replace(' ', "_"));
    // Simulate image generation
    Ok(format!(
        "Generated image '{file_name}' with prompt '{}'",
        args.prompt
    ))
}

fn main() {}
