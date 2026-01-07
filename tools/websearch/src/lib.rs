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

use aither_core::llm::Tool;
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
            name: "web_search".into(),
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

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result {
        let limit = arguments.limit.clamp(1, 10);
        let results = self.provider.search(&arguments.query, limit).await?;
        Ok(json(&results))
    }
}
