//! Copilot provider for listing and instantiating models.

use crate::{
    Copilot,
    auth::get_session_token,
    constant::{COPILOT_BASE_URL, COPILOT_INTEGRATION_ID, EDITOR_VERSION},
    error::CopilotError,
};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use aither_models::lookup as lookup_model_info;
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
                oauth_token: None,
            }),
        }
    }

    /// Set the base URL for API calls.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = url.into();
        self
    }

    /// Provide OAuth token so short-lived session tokens can be refreshed.
    #[must_use]
    pub fn oauth_token(mut self, token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).oauth_token = Some(token.into());
        self
    }
}

impl LanguageModelProvider for CopilotProvider {
    type Model = Copilot;
    type Error = CopilotError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let mut cfg = cfg.as_ref().clone();
            let response = match fetch_models(&cfg).await {
                Ok(response) => response,
                Err(err) if is_unauthorized(&err) => match refresh_provider_config(&cfg).await? {
                    Some(refreshed) => {
                        cfg = refreshed;
                        fetch_models(&cfg).await?
                    }
                    None => return Err(err),
                },
                Err(err) => return Err(err),
            };

            Ok(response
                .data
                .into_iter()
                .filter_map(|model| {
                    let context_length = model
                        .context_length
                        .or(model.max_tokens)
                        .or(model.input_token_limit)
                        .or_else(|| lookup_model_info(&model.id).map(|info| info.context_window));
                    let Some(context_length) = context_length else {
                        tracing::warn!(
                            model = %model.id,
                            "skip model from Copilot list: missing context_length and no aither-models fallback"
                        );
                        return None;
                    };
                    Some(ModelProfile::new(
                        model.id.clone(),
                        "copilot",
                        model.id,
                        model.owned_by.unwrap_or_else(|| "github".to_string()),
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
            let mut builder = Copilot::builder(cfg.token.clone())
                .base_url(cfg.base_url.clone())
                .model(name);
            if let Some(oauth_token) = &cfg.oauth_token {
                builder = builder.oauth_token(oauth_token.clone());
            }
            Ok(builder.build())
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
    oauth_token: Option<String>,
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
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default, alias = "inputTokenLimit", alias = "input_token_limit")]
    input_token_limit: Option<u32>,
}

async fn fetch_models(cfg: &ProviderConfig) -> Result<ModelListResponse, CopilotError> {
    let endpoint = format!("{}/models", cfg.base_url.trim_end_matches('/'));
    let mut backend = client();
    backend
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
        .map_err(CopilotError::Http)
}

const fn is_unauthorized(err: &CopilotError) -> bool {
    matches!(
        err,
        CopilotError::Http(zenwave::Error::Http { status, .. }) if status.as_u16() == 401
    )
}

async fn refresh_provider_config(
    cfg: &ProviderConfig,
) -> Result<Option<ProviderConfig>, CopilotError> {
    let Some(oauth_token) = cfg.oauth_token.as_deref() else {
        return Ok(None);
    };
    let session = get_session_token(oauth_token).await?;
    let mut refreshed = cfg.clone();
    refreshed.token = session.token;
    refreshed.base_url = session.api_endpoint;
    Ok(Some(refreshed))
}
