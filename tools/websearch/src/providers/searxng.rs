//! SearXNG metasearch engine provider.
//!
//! [SearXNG](https://docs.searxng.org/) is a free, open-source metasearch engine
//! that aggregates results from multiple search engines. No API key required.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{SearXNG, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Use the default public endpoint
//! let provider = SearXNG::default();
//! let results = provider.search("rust programming", 5).await?;
//!
//! // Or use a custom instance
//! let custom = SearXNG::new("http://localhost:8080");
//! # Ok(())
//! # }
//! ```

use crate::{SearchProvider, SearchResult};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use zenwave::{Client, client, header};

/// Default public SearXNG endpoint.
pub const DEFAULT_SEARXNG_URL: &str = "https://serxng-deployment-production.up.railway.app";

/// SearXNG metasearch engine provider.
///
/// A free, open-source metasearch engine that requires no API key.
/// Uses a public instance by default, or can be configured with a custom URL.
#[derive(Debug, Clone)]
pub struct SearXNG {
    base_url: String,
    engines: Option<String>,
}

impl Default for SearXNG {
    fn default() -> Self {
        Self::new(DEFAULT_SEARXNG_URL)
    }
}

impl SearXNG {
    /// Create a new SearXNG provider with a custom instance URL.
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
    /// let provider = SearXNG::default()
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
        let response: SearxngResponse = backend
            .get(&url)
            .map_err(|e| anyhow!("{e}"))?
            .header(header::ACCEPT.as_str(), "application/json")
            .map_err(|e| anyhow!("{e}"))?
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1")
            .map_err(|e| anyhow!("{e}"))?
            .json()
            .await
            .map_err(|e| anyhow!("{e}"))?;

        // Check for CAPTCHA errors when no results
        if response.results.is_empty() {
            let captcha_engines: Vec<_> = response
                .unresponsive_engines
                .iter()
                .filter(|(_, reason)| reason.contains("CAPTCHA"))
                .map(|(engine, _)| engine.as_str())
                .collect();

            if !captcha_engines.is_empty() {
                return Err(anyhow!(
                    "CAPTCHA detected for engines: {}. Try a different SearXNG instance.",
                    captcha_engines.join(", ")
                ));
            }
        }

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
    #[serde(default)]
    unresponsive_engines: Vec<(String, String)>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn search_returns_results() {
        let provider = SearXNG::default();
        let results = provider.search("rust programming language", 5).await;
        match results {
            Ok(r) if !r.is_empty() => {} // Success
            Ok(_) => eprintln!("Warning: SearXNG returned no results - endpoint may be degraded"),
            Err(e) if e.to_string().contains("CAPTCHA") => {
                eprintln!("Warning: SearXNG CAPTCHA detected - skipping test");
            }
            Err(e) => panic!("SearXNG search failed: {e}"),
        }
    }

    #[tokio::test]
    async fn search_unicode_query() {
        let provider = SearXNG::default();
        // Test with unicode characters (Japanese: "weather forecast")
        let results = provider.search("天気予報", 5).await;
        match results {
            Ok(_) => {} // Success (empty is ok for unicode)
            Err(e) if e.to_string().contains("CAPTCHA") => {
                eprintln!("Warning: SearXNG CAPTCHA detected - skipping test");
            }
            Err(e) => panic!("SearXNG unicode search failed: {e}"),
        }
    }
}
