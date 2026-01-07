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
//! use aither_webfetch::{fetch, FetchResult, WebFetchTool};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Direct fetch
//! let result = fetch("https://example.com").await?;
//! println!("Title: {:?}", result.title);
//! println!("Content: {}", result.content);
//!
//! // Or use as an LLM tool
//! let tool = WebFetchTool::new();
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::io::Cursor;

use aither_core::llm::Tool;
use anyhow::{Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

/// Arguments for the web fetch tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebFetchArgs {
    /// The URL to fetch content from. Must be a valid HTTP or HTTPS URL.
    pub url: String,
}

/// Web content fetching tool for LLM agents.
///
/// Fetches web pages, extracts main content using readability, and converts
/// to clean markdown format suitable for LLM consumption.
///
/// # Example
///
/// ```no_run
/// use aither_webfetch::WebFetchTool;
/// use aither_core::llm::Tool;
///
/// # async fn example() -> anyhow::Result<()> {
/// let tool = WebFetchTool::new();
/// // Use with agent...
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct WebFetchTool {
    name: String,
    description: String,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    /// Create a new web fetch tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "web_fetch".into(),
            description: include_str!("prompt.md").into(),
        }
    }

    /// Create a web fetch tool with custom name.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Create a web fetch tool with custom description.
    #[must_use]
    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl Tool for WebFetchTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = WebFetchArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result {
        let result = fetch(&arguments.url).await?;

        let mut output = String::new();
        if let Some(title) = &result.title {
            output.push_str(&format!("# {title}\n\n"));
        }
        output.push_str(&format!("Source: {}\n\n", result.url));
        output.push_str(&result.content);

        Ok(output)
    }
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
