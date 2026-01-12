//! Claude provider for listing and instantiating models.

use crate::{Claude, error::ClaudeError, constant::{CLAUDE_BASE_URL, ANTHROPIC_VERSION}};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use serde::Deserialize;
use std::{future::Future, sync::Arc};
use zenwave::{Client, client};

/// Provider capable of listing and instantiating `Claude` models.
#[derive(Clone, Debug)]
pub struct ClaudeProvider {
    inner: Arc<ProviderConfig>,
}

impl ClaudeProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                api_key: api_key.into(),
                base_url: CLAUDE_BASE_URL.to_string(),
            }),
        }
    }

    /// Override the REST base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = url.into();
        self
    }
}

impl LanguageModelProvider for ClaudeProvider {
    type Model = Claude;
    type Error = ClaudeError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let endpoint = format!("{}/models", cfg.base_url.trim_end_matches('/'));
            let mut backend = client();
            let response: ModelListResponse = backend
                .get(endpoint)
                .map_err(ClaudeError::Http)?
                .header("x-api-key", cfg.api_key.clone())
                .map_err(ClaudeError::Http)?
                .header("anthropic-version", ANTHROPIC_VERSION)
                .map_err(ClaudeError::Http)?
                .json()
                .await
                .map_err(ClaudeError::Http)?;

            Ok(response
                .data
                .into_iter()
                .map(|model| {
                    ModelProfile::new(
                        model.id.clone(),
                        "anthropic",
                        model.id,
                        model.display_name,
                        200_000, // Claude models generally have 200k context
                    )
                })
                .collect())
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let name = name.to_string();
        async move {
            Ok(Claude::new(cfg.api_key.clone())
                .with_base_url(cfg.base_url.clone())
                .with_model(name))
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("anthropic", "Anthropic Claude API provider")
    }
}

#[derive(Debug, Clone)]
struct ProviderConfig {
    api_key: String,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelDescriptor>,
}

#[derive(Debug, Deserialize)]
struct ModelDescriptor {
    id: String,
    display_name: String,
}
