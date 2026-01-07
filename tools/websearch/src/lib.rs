//! Web search tool for aither agents.
//!
//! This crate provides a unified interface for web search providers that can be
//! used with LLM agents to perform real-time internet searches.
//!
//! # Providers
//!
//! The following search providers are available:
//!
//! | Provider | API Key | Description |
//! |----------|---------|-------------|
//! | [`BraveSearch`] | Required | Privacy-first search with independent index |
//! | [`DuckDuckGo`] | Not required | Free instant answers API |
//! | [`Tavily`] | Required | AI-optimized search for RAG workflows |
//! | [`SearXNG`] | Not required | Self-hosted metasearch engine |
//! | [`GoogleSearch`] | Required (+CX ID) | Google Custom Search API |
//! | [`Serper`] | Required | Fast Google SERP API |
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{BraveSearch, SearchProvider, WebSearchTool};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a search provider
//! let provider = BraveSearch::new("YOUR_API_KEY");
//!
//! // Use directly
//! let results = provider.search("rust async programming", 5).await?;
//!
//! // Or wrap in a tool for LLM agents
//! let tool = WebSearchTool::new(provider);
//! # Ok(())
//! # }
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

impl<P> WebSearchTool<P> {
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            name: "web_search".into(),
            description: "Searches the web and returns relevant documents.".into(),
        }
    }

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
