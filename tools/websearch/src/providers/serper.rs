//! Serper.dev Google SERP API provider.
//!
//! [Serper](https://serper.dev/) provides a fast and affordable Google Search API
//! that returns structured SERP (Search Engine Results Page) data.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{Serper, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = Serper::new("YOUR_API_KEY");
//! let results = provider.search("best programming practices", 5).await?;
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

/// Serper API endpoint.
const SERPER_API_URL: &str = "https://google.serper.dev/search";

/// Search type for Serper queries.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SerperSearchType {
    /// Standard web search.
    #[default]
    Search,
    /// News search.
    News,
    /// Image search.
    Images,
    /// Places/local search.
    Places,
}

/// Serper.dev Google SERP API provider.
///
/// Uses the [Serper API](https://serper.dev/) to get Google search results
/// in a structured JSON format.
#[derive(Debug, Clone)]
pub struct Serper {
    api_key: String,
    search_type: SerperSearchType,
    country: Option<String>,
    locale: Option<String>,
}

impl Serper {
    /// Create a new Serper provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            search_type: SerperSearchType::default(),
            country: None,
            locale: None,
        }
    }

    /// Set the search type (search, news, images, places).
    #[must_use]
    pub const fn with_search_type(mut self, search_type: SerperSearchType) -> Self {
        self.search_type = search_type;
        self
    }

    /// Set the country code for localized results (e.g., "us", "uk", "de").
    #[must_use]
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the locale for results (e.g., "en", "de", "fr").
    #[must_use]
    pub fn with_locale(mut self, locale: impl Into<String>) -> Self {
        self.locale = Some(locale.into());
        self
    }
}

impl SearchProvider for Serper {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let request = SerperRequest {
            q: query,
            num: Some(limit.min(100)), // Serper supports up to 100 results
            gl: self.country.as_deref(),
            hl: self.locale.as_deref(),
        };

        let mut backend = client();
        let builder = backend
            .post(SERPER_API_URL)
            .header("X-API-KEY", &self.api_key)
            .header(header::CONTENT_TYPE.as_str(), "application/json")
            .header(header::ACCEPT.as_str(), "application/json")
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1");

        let response: SerperResponse = builder
            .json_body(&request)
            .json()
            .await
            .map_err(|e| anyhow!("{e}"))?;

        Ok(response
            .organic
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

#[derive(Debug, Serialize)]
struct SerperRequest<'a> {
    q: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    num: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gl: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hl: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct SerperResponse {
    organic: Option<Vec<SerperResult>>,
}

#[derive(Debug, Deserialize)]
struct SerperResult {
    title: String,
    link: String,
    snippet: Option<String>,
}
