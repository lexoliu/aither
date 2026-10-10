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
use aither::llm::ToolContext;
use aither::llm::tool::Progress;
use aither_derive::tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Basic tool example - no parameters needed. The function's rustdoc is the
// tool description.
/// Get the current UTC time.
#[tool]
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
/// Search the web for the given keywords.
#[tool]
pub async fn search(keywords: Vec<String>, max_results: u32) -> Result<Vec<SearchResult>> {
    // Simulate a search result
    let results = keywords
        .into_iter()
        .take(max_results as usize)
        .map(|keyword| SearchResult {
            title: format!("Result for {keyword}"),
            url: format!("https://example.com/search?q={keyword}"),
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
#[tool]
pub async fn generate_image(args: GenerateImageArgs) -> aither::Result<String> {
    let file_name = format!("image_{}.png", args.prompt.replace(' ', "_"));
    // Simulate image generation
    Ok(format!(
        "Generated image '{file_name}' with prompt '{}'",
        args.prompt
    ))
}

// Long-running tool that reports progress through its call context. The
// `ToolContext` parameter is not part of the argument schema; reporting is a
// no-op when the caller does not listen for progress.
/// Fetch every URL, reporting each one as it completes.
#[tool]
pub async fn crawl(urls: Vec<String>, mut cx: ToolContext) -> Result<u32> {
    let total = u32::try_from(urls.len())?;
    for (done, url) in (1..=total).zip(&urls) {
        // Simulate fetching `url`
        let progress = Progress::new(f64::from(done))
            .with_total(f64::from(total))
            .with_message(format!("fetched {url}"));
        cx.report_progress(progress).await?;
    }
    Ok(total)
}

fn main() {
    // Every generated tool carries a description and registers cleanly.
    let mut tools = aither::llm::tool::Tools::new();
    tools.register(Time).expect("time registers");
    tools.register(Search).expect("search registers");
    tools
        .register(GenerateImage)
        .expect("generate_image registers");
    tools.register(Crawl).expect("crawl registers");
}
