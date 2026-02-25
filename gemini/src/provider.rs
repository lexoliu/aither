use crate::{GEMINI_API_BASE_URL, Gemini, config::AuthMode, error::GeminiError};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use aither_models::lookup as lookup_model_info;
use serde::Deserialize;
use std::{future::Future, sync::Arc};
use zenwave::{Client, client};

/// Provider capable of listing and instantiating `Gemini` models.
#[derive(Clone, Debug)]
pub struct GeminiProvider {
    inner: Arc<ProviderConfig>,
}

impl GeminiProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                api_key: api_key.into(),
                base_url: GEMINI_API_BASE_URL.to_string(),
                auth: AuthMode::Query,
            }),
        }
    }

    /// Override the REST base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = url.into();
        self
    }

    /// Select header-based authentication.
    #[must_use]
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        Arc::make_mut(&mut self.inner).auth = mode;
        self
    }
}

impl LanguageModelProvider for GeminiProvider {
    type Model = Gemini;
    type Error = GeminiError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let mut endpoint = format!("{}/models", cfg.base_url.trim_end_matches('/'));
            if cfg.auth == AuthMode::Query {
                let separator = if endpoint.contains('?') { '&' } else { '?' };
                endpoint.push(separator);
                endpoint.push_str("key=");
                endpoint.push_str(&cfg.api_key);
            }

            let mut backend = client();
            let mut builder = backend.get(endpoint).map_err(GeminiError::Http)?;

            if cfg.auth == AuthMode::Header {
                builder = builder
                    .header("x-goog-api-key", cfg.api_key.clone())
                    .map_err(GeminiError::Http)?;
            }

            let response: ModelListResponse = builder.json().await.map_err(GeminiError::Http)?;

            Ok(response
                .models
                .into_iter()
                .filter_map(|model| {
                    let model_id = model.name.trim_start_matches("models/").to_string();
                    let context_length = model
                        .input_token_limit
                        .and_then(|value| u32::try_from(value).ok())
                        .or_else(|| lookup_model_info(&model_id).map(|info| info.context_window))
                        .or_else(|| lookup_model_info(&model.name).map(|info| info.context_window));
                    let Some(context_length) = context_length else {
                        tracing::warn!(
                            model = %model.name,
                            "skip model from Gemini list: missing input_token_limit and no aither-models fallback"
                        );
                        return None;
                    };
                    Some(ModelProfile::new(
                        model_id,
                        "google",
                        model.name,
                        model.description.unwrap_or_default(),
                        context_length,
                    ))
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
            let backend = Gemini::new(cfg.api_key.clone())
                .with_base_url(cfg.base_url.clone())
                .with_auth_mode(cfg.auth)
                .with_text_model(name);
            Ok(backend)
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("gemini", "Google Gemini API provider")
    }
}

#[derive(Debug, Clone)]
struct ProviderConfig {
    api_key: String,
    base_url: String,
    auth: AuthMode,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    models: Vec<ModelDescriptor>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelDescriptor {
    name: String,
    display_name: Option<String>,
    description: Option<String>,
    input_token_limit: Option<i32>,
    output_token_limit: Option<i32>,
    supported_generation_methods: Option<Vec<String>>,
}
