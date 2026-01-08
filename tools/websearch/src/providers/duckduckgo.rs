//! DuckDuckGo Instant Answer API provider.
//!
//! [DuckDuckGo](https://duckduckgo.com/) provides a free instant answer API
//! that doesn't require an API key. Note that this API returns instant answers
//! and related topics rather than traditional web search results.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{DuckDuckGo, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = DuckDuckGo::new();
//! let results = provider.search("rust programming language", 5).await?;
//! for result in results {
//!     println!("{}: {}", result.title, result.url);
//! }
//! # Ok(())
//! # }
//! ```

use crate::{SearchProvider, SearchResult};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use zenwave::{Client, client, header};

/// DuckDuckGo Instant Answer API endpoint.
const DDG_API_URL: &str = "https://api.duckduckgo.com/";

/// DuckDuckGo Instant Answer API provider.
///
/// Uses the [DuckDuckGo Instant Answer API](https://api.duckduckgo.com/api) which
/// is free and doesn't require authentication. Returns instant answers, abstract,
/// and related topics.
#[derive(Debug, Clone, Default)]
pub struct DuckDuckGo {
    _private: (),
}

impl DuckDuckGo {
    /// Create a new DuckDuckGo provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl SearchProvider for DuckDuckGo {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let url = format!(
            "{}?q={}&format=json&no_html=1&skip_disambig=1",
            DDG_API_URL,
            urlencoded(query)
        );

        let mut backend = client();
        let response: DdgResponse = backend
            .get(&url)
            .map_err(|e| anyhow!("{e}"))?
            .header(header::ACCEPT.as_str(), "application/json")
            .map_err(|e| anyhow!("{e}"))?
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1")
            .map_err(|e| anyhow!("{e}"))?
            .json()
            .await
            .map_err(|e| anyhow!("{e}"))?;

        let mut results = Vec::new();

        // Add abstract if available
        if !response.abstract_text.is_empty() {
            results.push(SearchResult {
                title: response.heading.clone(),
                url: response.abstract_url.clone(),
                snippet: response.abstract_text.clone(),
            });
        }

        // Add related topics
        for topic in response
            .related_topics
            .into_iter()
            .take(limit.saturating_sub(results.len()))
        {
            if let Some(text) = topic.text {
                results.push(SearchResult {
                    title: topic.first_url.as_deref().unwrap_or("Related").to_string(),
                    url: topic.first_url.unwrap_or_default(),
                    snippet: text,
                });
            }
        }

        Ok(results.into_iter().take(limit).collect())
    }
}

#[derive(Debug, Deserialize)]
struct DdgResponse {
    #[serde(rename = "Heading", default)]
    heading: String,
    #[serde(rename = "AbstractText", default)]
    abstract_text: String,
    #[serde(rename = "AbstractURL", default)]
    abstract_url: String,
    #[serde(rename = "RelatedTopics", default)]
    related_topics: Vec<DdgTopic>,
}

#[derive(Debug, Deserialize)]
struct DdgTopic {
    #[serde(rename = "Text")]
    text: Option<String>,
    #[serde(rename = "FirstURL")]
    first_url: Option<String>,
}

fn urlencoded(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}
