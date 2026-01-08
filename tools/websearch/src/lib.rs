//! Web search tool for aither agents.
//!
//! This crate provides a unified interface for web search providers that can be
//! used with LLM agents to perform real-time internet searches.
//!
//! # Default Provider
//!
//! By default, uses [SearXNG](https://docs.searxng.org/) - a free, open-source
//! metasearch engine that requires no API key.
//!
//! ```no_run
//! use aither_websearch::WebSearchTool;
//!
//! // Uses SearXNG - no API key needed
//! let tool = WebSearchTool::default();
//! ```
//!
//! # All Providers
//!
//! | Provider | API Key | Description |
//! |----------|---------|-------------|
//! | [`SearXNG`] | Not required | **Default** - Free metasearch engine |
//! | [`DuckDuckGo`] | Not required | Instant answers only (not full web search) |
//! | [`BraveSearch`] | Required | Privacy-first search with independent index |
//! | [`Tavily`] | Required | AI-optimized search for RAG workflows |
//! | [`GoogleSearch`] | Required (+CX ID) | Google Custom Search API |
//! | [`Serper`] | Required | Fast Google SERP API |
//!
//! # Custom Provider
//!
//! ```no_run
//! use aither_websearch::{Tavily, WebSearchTool};
//!
//! let tool = WebSearchTool::new(Tavily::new("YOUR_API_KEY"));
//! ```

mod providers;

pub use providers::*;

use std::borrow::Cow;

use aither_core::llm::{Tool, ToolOutput};
use aither_core::llm::tool::json;
use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebSearchArgs {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub trait SearchProvider: Send + Sync {
    fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
}

#[derive(Debug, Clone)]
pub struct WebSearchTool<P> {
    provider: P,
    name: String,
    description: String,
}

impl Default for WebSearchTool<SearXNG> {
    fn default() -> Self {
        Self::new(SearXNG::default())
    }
}

impl<P> WebSearchTool<P> {
    /// Create a web search tool with a custom provider.
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            name: "websearch".into(),
            description: include_str!("prompt.md").into(),
        }
    }

    /// Create a web search tool with custom name and description.
    pub fn with_metadata(
        provider: P,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Maximum retry attempts when search returns empty results.
const MAX_RETRIES: u32 = 3;

/// Delay between retry attempts in milliseconds.
const RETRY_DELAY_MS: u64 = 500;

/// Check if an error is non-retryable (e.g., CAPTCHA).
fn is_non_retryable(e: &anyhow::Error) -> bool {
    e.to_string().contains("CAPTCHA")
}

impl<P> Tool for WebSearchTool<P>
where
    P: SearchProvider + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = WebSearchArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let limit = arguments.limit.clamp(1, 10);

        // Retry on empty results (search engines may temporarily fail)
        for attempt in 0..MAX_RETRIES {
            match self.provider.search(&arguments.query, limit).await {
                Ok(results) if !results.is_empty() => {
                    return Ok(ToolOutput::text(json(&results)));
                }
                Ok(_empty) if attempt < MAX_RETRIES - 1 => {
                    // Empty results, retry after delay
                    async_io::Timer::after(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
                }
                Ok(empty) => {
                    // Final attempt still empty, return empty results
                    return Ok(ToolOutput::text(json(&empty)));
                }
                Err(e) if is_non_retryable(&e) => {
                    // Non-retryable error (e.g., CAPTCHA), fail immediately
                    return Err(e.into());
                }
                Err(e) if attempt < MAX_RETRIES - 1 => {
                    // Retryable error, retry after delay
                    async_io::Timer::after(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
                    tracing::warn!(attempt, error = %e, "websearch failed, retrying");
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }

        unreachable!()
    }
}
