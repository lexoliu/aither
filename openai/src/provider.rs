use crate::{
    DEEPSEEK_BASE_URL, DEFAULT_BASE_URL, OPENROUTER_BASE_URL, client::OpenAI, error::OpenAIError,
};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use serde::Deserialize;
use std::{future::Future, sync::Arc};
use zenwave::{Client, client, header};

/// Provider capable of listing and instantiating `OpenAI` models.
#[derive(Clone, Debug)]
pub struct OpenAIProvider {
    inner: Arc<ProviderConfig>,
}

impl OpenAIProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                organization: None,
            }),
        }
    }

    /// Convenience constructor targeting [`Deepseek`](https://api-docs.deepseek.com)'s API.
    #[must_use]
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new(api_key).base_url(DEEPSEEK_BASE_URL)
    }

    /// Convenience constructor targeting [`OpenRouter`](https://openrouter.ai)'s API.
    #[must_use]
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(api_key).base_url(OPENROUTER_BASE_URL)
    }

    /// Override the REST base URL (useful for Azure or sandboxes).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = url.into();
        self
    }

    /// Attach an organization header for model management calls.
    #[must_use]
    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).organization = Some(organization.into());
        self
    }

    fn client_for(&self, model: impl Into<String>) -> OpenAI {
        let mut builder = OpenAI::builder(self.inner.api_key.clone())
            .base_url(self.inner.base_url.clone())
            .model(model);
        if let Some(org) = &self.inner.organization {
            builder = builder.organization(org.clone());
        }
        builder.build()
    }
}

impl LanguageModelProvider for OpenAIProvider {
    type Model = OpenAI;
    type Error = OpenAIError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let endpoint = format!("{}/models", cfg.base_url.trim_end_matches('/'));
            let mut backend = client();
            let mut builder = backend
                .get(endpoint)
                .map_err(OpenAIError::Http)?
                .header(
                    header::AUTHORIZATION.as_str(),
                    format!("Bearer {}", cfg.api_key),
                )
                .map_err(OpenAIError::Http)?;
            if let Some(org) = &cfg.organization {
                builder = builder
                    .header("OpenAI-Organization", org.clone())
                    .map_err(OpenAIError::Http)?;
            }
            let response: ModelListResponse = builder.json().await.map_err(OpenAIError::Http)?;
            Ok(response
                .data
                .into_iter()
                .map(|model| {
                    ModelProfile::new(
                        model.id.clone(),
                        model.owned_by.clone().unwrap_or_else(|| "openai".into()),
                        model.id,
                        "OpenAI model",
                        128_000,
                    )
                })
                .collect())
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send {
        let client = self.client_for(name.to_string());
        async move { Ok(client) }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("openai", "Official OpenAI API provider")
    }
}

#[derive(Debug, Clone)]
struct ProviderConfig {
    api_key: String,
    base_url: String,
    organization: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelDescriptor>,
}

#[derive(Debug, Deserialize)]
struct ModelDescriptor {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
}
