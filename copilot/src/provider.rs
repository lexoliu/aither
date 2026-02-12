//! Copilot provider for listing and instantiating models.

use crate::{
    Copilot,
    constant::{COPILOT_BASE_URL, COPILOT_INTEGRATION_ID, EDITOR_VERSION},
    error::CopilotError,
};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use serde::Deserialize;
use std::{future::Future, sync::Arc};
use zenwave::{Client, client, header};

/// Provider capable of listing and instantiating `Copilot` models.
#[derive(Clone, Debug)]
pub struct CopilotProvider {
    inner: Arc<ProviderConfig>,
}

impl CopilotProvider {
    /// Create a new provider with the given session token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                token: token.into(),
                base_url: COPILOT_BASE_URL.to_string(),
                editor_version: EDITOR_VERSION.to_string(),
                integration_id: COPILOT_INTEGRATION_ID.to_string(),
            }),
        }
    }

    /// Set the base URL for API calls.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = url.into();
        self
    }
}

impl LanguageModelProvider for CopilotProvider {
    type Model = Copilot;
    type Error = CopilotError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let endpoint = format!("{}/models", cfg.base_url.trim_end_matches('/'));
            let mut backend = client();
            let response: ModelListResponse = backend
                .get(endpoint)
                .map_err(CopilotError::Http)?
                .header(
                    header::AUTHORIZATION.as_str(),
                    format!("Bearer {}", cfg.token),
                )
                .map_err(CopilotError::Http)?
                .header(header::USER_AGENT.as_str(), "aither-copilot/0.1")
                .map_err(CopilotError::Http)?
                .header(header::ACCEPT.as_str(), "application/json")
                .map_err(CopilotError::Http)?
                .header("Editor-Version", cfg.editor_version.clone())
                .map_err(CopilotError::Http)?
                .header("Copilot-Integration-Id", cfg.integration_id.clone())
                .map_err(CopilotError::Http)?
                .json()
                .await
                .map_err(CopilotError::Http)?;

            Ok(response
                .data
                .into_iter()
                .map(|model| {
                    ModelProfile::new(
                        model.id.clone(),
                        "copilot",
                        model.id,
                        model.owned_by.unwrap_or_else(|| "github".to_string()),
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
        let cfg = self.inner.clone();
        let name = name.to_string();
        async move {
            Ok(Copilot::new(cfg.token.clone())
                .with_base_url(cfg.base_url.clone())
                .with_model(name))
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("copilot", "GitHub Copilot API provider")
    }
}

#[derive(Debug, Clone)]
struct ProviderConfig {
    token: String,
    base_url: String,
    editor_version: String,
    integration_id: String,
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
