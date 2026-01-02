//! Google Custom Search API provider.
//!
//! [Google Custom Search](https://developers.google.com/custom-search/) provides
//! programmatic access to Google search results through a Programmable Search Engine.
//!
//! # Setup
//!
//! 1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
//! 2. Enable the Custom Search API
//! 3. Create credentials (API key)
//! 4. Create a [Programmable Search Engine](https://programmablesearchengine.google.com/)
//! 5. Get your Search Engine ID (cx)
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{GoogleSearch, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = GoogleSearch::new("YOUR_API_KEY", "YOUR_SEARCH_ENGINE_ID");
//! let results = provider.search("machine learning tutorials", 5).await?;
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

/// Google Custom Search API endpoint.
const GOOGLE_API_URL: &str = "https://www.googleapis.com/customsearch/v1";

/// Google Custom Search API provider.
///
/// Uses the [Google Custom Search JSON API](https://developers.google.com/custom-search/v1/overview)
/// to perform web searches. Requires an API key and a Custom Search Engine ID (cx).
#[derive(Debug, Clone)]
pub struct GoogleSearch {
    api_key: String,
    cx: String,
}

impl GoogleSearch {
    /// Create a new Google Custom Search provider.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Google API key
    /// * `cx` - Your Custom Search Engine ID
    #[must_use]
    pub fn new(api_key: impl Into<String>, cx: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            cx: cx.into(),
        }
    }
}

impl SearchProvider for GoogleSearch {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // Google CSE has a max of 10 results per request
        let num = limit.min(10);

        let url = format!(
            "{}?key={}&cx={}&q={}&num={}",
            GOOGLE_API_URL,
            urlencoded(&self.api_key),
            urlencoded(&self.cx),
            urlencoded(query),
            num
        );

        let mut backend = client();
        let builder = backend
            .get(&url)
            .header(header::ACCEPT.as_str(), "application/json")
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1");

        let response: GoogleResponse = builder.json().await.map_err(|e| anyhow!("{e}"))?;

        Ok(response
            .items
            .unwrap_or_default()
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.link,
                snippet: r.snippet.unwrap_or_default(),
            })
            .collect())
    }
}

#[derive(Debug, Deserialize)]
struct GoogleResponse {
    items: Option<Vec<GoogleResult>>,
}

#[derive(Debug, Deserialize)]
struct GoogleResult {
    title: String,
    link: String,
    snippet: Option<String>,
}

fn urlencoded(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}
