//! SearXNG metasearch engine provider.
//!
//! [SearXNG](https://docs.searxng.org/) is a free, open-source metasearch engine
//! that aggregates results from multiple search engines. It can be self-hosted
//! for privacy-focused applications.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{SearXNG, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Connect to a self-hosted SearXNG instance
//! let provider = SearXNG::new("http://localhost:8080");
//! let results = provider.search("privacy search engines", 5).await?;
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

/// SearXNG metasearch engine provider.
///
/// Connects to a self-hosted [SearXNG](https://github.com/searxng/searxng) instance.
/// Requires the instance to have JSON format enabled in settings.
#[derive(Debug, Clone)]
pub struct SearXNG {
    base_url: String,
    engines: Option<String>,
}

impl SearXNG {
    /// Create a new SearXNG provider with the given instance URL.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of your SearXNG instance (e.g., `http://localhost:8080`)
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            engines: None,
        }
    }

    /// Specify which engines to use (comma-separated).
    ///
    /// # Example
    ///
    /// ```
    /// use aither_websearch::SearXNG;
    ///
    /// let provider = SearXNG::new("http://localhost:8080")
    ///     .with_engines("google,duckduckgo,bing");
    /// ```
    #[must_use]
    pub fn with_engines(mut self, engines: impl Into<String>) -> Self {
        self.engines = Some(engines.into());
        self
    }
}

impl SearchProvider for SearXNG {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut url = format!(
            "{}/search?q={}&format=json",
            self.base_url,
            urlencoded(query)
        );

        if let Some(ref engines) = self.engines {
            url.push_str(&format!("&engines={engines}"));
        }

        let mut backend = client();
        let builder = backend
            .get(&url)
            .header(header::ACCEPT.as_str(), "application/json")
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1");

        let response: SearxngResponse = builder.json().await.map_err(|e| anyhow!("{e}"))?;

        Ok(response
            .results
            .into_iter()
            .take(limit)
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content.unwrap_or_default(),
            })
            .collect())
    }
}

#[derive(Debug, Deserialize)]
struct SearxngResponse {
    results: Vec<SearxngResult>,
}

#[derive(Debug, Deserialize)]
struct SearxngResult {
    title: String,
    url: String,
    content: Option<String>,
}

fn urlencoded(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}
