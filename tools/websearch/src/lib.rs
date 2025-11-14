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

    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        let limit = arguments.limit.clamp(1, 10);
        let results = self.provider.search(&arguments.query, limit).await?;
        Ok(json(&results))
    }
}

#[derive(Debug, Clone)]
pub struct InMemorySearchProvider {
    corpus: Vec<SearchResult>,
}

impl InMemorySearchProvider {
    pub fn new(corpus: Vec<SearchResult>) -> Self {
        Self { corpus }
    }
}

impl Default for InMemorySearchProvider {
    fn default() -> Self {
        Self {
            corpus: vec![SearchResult {
                title: "Knowledge base is empty".into(),
                url: "https://example.com".into(),
                snippet: "Provide your own provider to enable real search.".into(),
            }],
        }
    }
}

impl SearchProvider for InMemorySearchProvider {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let needle = query.to_lowercase();
        let mut matches: Vec<_> = self
            .corpus
            .iter()
            .filter(|entry| {
                entry.title.to_lowercase().contains(&needle)
                    || entry.snippet.to_lowercase().contains(&needle)
            })
            .cloned()
            .collect();
        matches.truncate(limit);
        Ok(matches)
    }
}
