//! Exa Search API provider.
//!
//! [Exa](https://exa.ai) provides semantic web search and content extraction
//! optimized for AI applications.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{Exa, SearchProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = Exa::new("YOUR_API_KEY");
//! let results = provider.search("latest rust async patterns", 5).await?;
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

/// Exa Search API endpoint.
const EXA_SEARCH_API_URL: &str = "https://api.exa.ai/search";

/// Exa Search provider.
///
/// Uses the Exa Search API to retrieve web results with page text content.
/// Requires an API key.
#[derive(Debug, Clone)]
pub struct Exa {
    api_key: String,
}

impl Exa {
    /// Create a new Exa provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
    }
}

impl SearchProvider for Exa {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let request = build_request(query, limit.min(100));

        let mut backend = client();
        let response: ExaResponse = backend
            .post(EXA_SEARCH_API_URL)
            .map_err(|e| anyhow!("{e}"))?
            .header("x-api-key", &self.api_key)
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

        Ok(response.results.into_iter().map(map_result).collect())
    }
}

fn map_result(result: ExaResult) -> SearchResult {
    SearchResult {
        title: result.title.unwrap_or_else(|| result.url.clone()),
        url: result.url,
        snippet: result.text.or(result.summary).unwrap_or_default(),
    }
}

fn build_request(query: &str, limit: usize) -> ExaRequest<'_> {
    ExaRequest {
        query,
        num_results: limit,
        contents: ExaContentsRequest { text: true },
    }
}

#[derive(Debug, Serialize)]
struct ExaRequest<'a> {
    query: &'a str,
    #[serde(rename = "numResults")]
    num_results: usize,
    contents: ExaContentsRequest,
}

#[derive(Debug, Serialize)]
struct ExaContentsRequest {
    text: bool,
}

#[derive(Debug, Deserialize)]
struct ExaResponse {
    #[serde(default)]
    results: Vec<ExaResult>,
}

#[derive(Debug, Deserialize)]
struct ExaResult {
    title: Option<String>,
    url: String,
    text: Option<String>,
    summary: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_expected_fields() {
        let request = build_request("rust async", 7);
        let json = serde_json::to_value(request).expect("serialize request");

        assert_eq!(json["query"], "rust async");
        assert_eq!(json["numResults"], 7);
        assert_eq!(json["contents"]["text"], true);
    }

    #[test]
    fn response_maps_results() {
        let response_json = serde_json::json!({
            "results": [
                {
                    "title": "Rust Async Patterns",
                    "url": "https://example.com/rust-async",
                    "text": "Structured concurrency patterns in Rust."
                }
            ]
        });
        let response: ExaResponse = serde_json::from_value(response_json).expect("parse response");
        let mapped: Vec<SearchResult> = response.results.into_iter().map(map_result).collect();

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].title, "Rust Async Patterns");
        assert_eq!(mapped[0].url, "https://example.com/rust-async");
        assert_eq!(
            mapped[0].snippet,
            "Structured concurrency patterns in Rust."
        );
    }

    #[test]
    fn response_handles_empty_results() {
        let response_json = serde_json::json!({ "results": [] });
        let response: ExaResponse = serde_json::from_value(response_json).expect("parse response");
        let mapped: Vec<SearchResult> = response.results.into_iter().map(map_result).collect();

        assert!(mapped.is_empty());
    }

    #[test]
    fn mapping_handles_missing_optional_fields() {
        let response_json = serde_json::json!({
            "results": [
                {
                    "url": "https://example.com/no-title",
                    "summary": "Fallback summary text."
                }
            ]
        });
        let response: ExaResponse = serde_json::from_value(response_json).expect("parse response");
        let mapped: Vec<SearchResult> = response.results.into_iter().map(map_result).collect();

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].title, "https://example.com/no-title");
        assert_eq!(mapped[0].url, "https://example.com/no-title");
        assert_eq!(mapped[0].snippet, "Fallback summary text.");
    }
}
