//! # Web Content Fetching Library
//!
//! An async runtime-independent library for fetching web content and converting it to clean markdown.
//! Uses:
//! - `zenwave` for async HTTP fetching (runtime-agnostic)
//! - `readability` for main content extraction (Mozilla Readability port)
//! - `htmd` for HTML-to-Markdown conversion
//!
//! # Example
//!
//! ```no_run
//! use aither_webfetch::{fetch, FetchResult};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let result = fetch("https://example.com").await?;
//! println!("Title: {:?}", result.title);
//! println!("Content: {}", result.content);
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// Result of fetching web content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResult {
    /// The original URL that was fetched.
    pub url: String,
    /// The page title, if found.
    pub title: Option<String>,
    /// The content in markdown format.
    pub content: String,
}

/// Fetches a web page and converts it to clean markdown.
///
/// Uses `zenwave` for async HTTP, `readability` for content extraction,
/// and `htmd` for HTML-to-Markdown conversion.
pub async fn fetch(url: &str) -> Result<FetchResult> {
    let html = fetch_html(url).await?;

    // Parse URL for readability
    let parsed_url = url::Url::parse(url).map_err(|e| anyhow!("Invalid URL: {e}"))?;

    // Use readability to extract main content
    let mut cursor = Cursor::new(html.as_bytes());
    let extracted = readability::extractor::extract(&mut cursor, &parsed_url)
        .map_err(|e| anyhow!("Content extraction failed: {e}"))?;

    // Convert extracted HTML to markdown
    let content = htmd::convert(&extracted.content).map_err(|e| anyhow!("{e}"))?;

    Ok(FetchResult {
        url: url.to_string(),
        title: Some(extracted.title),
        content,
    })
}

/// Fetches raw HTML from a URL using zenwave (async HTTP client).
async fn fetch_html(url: &str) -> Result<String> {
    let response = zenwave::get(url)
        .await
        .map_err(|e| anyhow!("HTTP request failed: {e}"))?;

    let html = response
        .into_body()
        .into_string()
        .await
        .map_err(|e| anyhow!("Failed to read response body: {e}"))?;

    Ok(html.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fetch_example_com() {
        let result = fetch("https://example.com").await.unwrap();
        assert!(result.title.is_some());
        assert!(!result.content.is_empty());
    }
}
