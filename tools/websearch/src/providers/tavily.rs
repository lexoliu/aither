//! Tavily AI-optimized search API provider.
//!
//! [Tavily](https://tavily.com/) is a search engine specifically designed for AI agents
//! and LLMs. It aggregates multiple sources and provides cleaned, structured content
//! optimized for RAG workflows.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{Tavily, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = Tavily::new("YOUR_API_KEY");
//! let results = provider.search("latest AI developments", 5).await?;
//! for result in results {
//!     println!("{}: {}", result.title, result.url);
//! }
//! # Ok(())
//! # }
//! ```

use crate::{SearchProvider, SearchResult};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use zenwave::{Client, client, header};

/// Tavily API endpoint.
const TAVILY_API_URL: &str = "https://api.tavily.com/search";

/// Tavily AI-optimized search provider.
///
/// Uses the [Tavily API](https://tavily.com/) which is designed specifically for
/// AI agents and RAG applications. Provides cleaned, structured search results
/// with optional content extraction.
#[derive(Debug, Clone)]
pub struct Tavily {
    api_key: String,
    search_depth: SearchDepth,
    include_answer: bool,
}

/// Search depth for Tavily queries.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchDepth {
    /// Basic search (faster, fewer results).
    #[default]
    Basic,
    /// Advanced search (slower, more comprehensive).
    Advanced,
}

impl Tavily {
    /// Create a new Tavily provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            search_depth: SearchDepth::Basic,
            include_answer: false,
        }
    }

    /// Set the search depth (basic or advanced).
    #[must_use]
    pub const fn with_search_depth(mut self, depth: SearchDepth) -> Self {
        self.search_depth = depth;
        self
    }

    /// Enable AI-generated answer in response.
    #[must_use]
    pub const fn with_answer(mut self, include: bool) -> Self {
        self.include_answer = include;
        self
    }
}

impl SearchProvider for Tavily {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let request = TavilyRequest {
            api_key: &self.api_key,
            query,
            search_depth: self.search_depth,
            include_answer: self.include_answer,
            max_results: limit,
        };

        let mut backend = client();
        let response: TavilyResponse = backend
            .post(TAVILY_API_URL)
            .map_err(|e| anyhow!("{e}"))?
            .header(header::CONTENT_TYPE.as_str(), "application/json")
            .map_err(|e| anyhow!("{e}"))?
            .header(header::ACCEPT.as_str(), "application/json")
            .map_err(|e| anyhow!("{e}"))?
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1")
            .map_err(|e| anyhow!("{e}"))?
            .json_body(&request)
            .map_err(|e| anyhow!("{e}"))?
            .json()
            .await
            .map_err(|e| anyhow!("{e}"))?;

        Ok(response
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
            })
            .collect())
    }
}

#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    search_depth: SearchDepth,
    include_answer: bool,
    max_results: usize,
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}
