//! [`LanguageModelProvider`] for the Codex subscription backend.
//!
//! The backend exposes its own model catalog at
//! `GET {base_url}/models?client_version=…`, which is authoritative for what a
//! subscription can run — [`CodexProvider::list_models`] surfaces it directly.

use serde::Deserialize;
use zenwave::{Client as _, client};

use aither_core::llm::model::Profile as ModelProfile;
use aither_core::llm::provider::{LanguageModelProvider, Profile};

use crate::auth::CodexCredentials;
use crate::client::{Codex, OnRefresh};
use crate::constant::{ACCOUNT_ID_HEADER, CLIENT_VERSION, CODEX_BASE_URL};
use crate::error::CodexError;

/// Provider over the `ChatGPT` Codex backend.
///
/// Holds the credential set used for both catalog listing and model clients.
/// Token lifecycle (expiry refresh, rotation persistence) is handled inside
/// each [`Codex`] model client created by [`get_model`].
#[derive(Clone)]
pub struct CodexProvider {
    credentials: CodexCredentials,
    base_url: String,
    on_refresh: Option<OnRefresh>,
}

impl std::fmt::Debug for CodexProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodexProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl CodexProvider {
    /// Create a provider from OAuth credentials.
    #[must_use]
    pub fn new(credentials: CodexCredentials) -> Self {
        Self {
            credentials,
            base_url: CODEX_BASE_URL.into(),
            on_refresh: None,
        }
    }

    /// Override the backend base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Propagate a token-rotation persistence callback to model clients.
    #[must_use]
    pub fn on_refresh(
        mut self,
        callback: impl Fn(&CodexCredentials) + Send + Sync + 'static,
    ) -> Self {
        self.on_refresh = Some(std::sync::Arc::new(callback));
        self
    }

    /// The credentials backing this provider.
    #[must_use]
    pub const fn credentials(&self) -> &CodexCredentials {
        &self.credentials
    }
}

/// Response of `GET /codex/models`.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    models: Vec<ModelDescriptor>,
}

#[derive(Debug, Deserialize)]
struct ModelDescriptor {
    slug: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    context_window: Option<u32>,
    /// Catalog visibility; `"hide"` entries are filtered out.
    #[serde(default)]
    visibility: Option<String>,
    /// Sort priority (higher first); missing sorts last.
    #[serde(default)]
    priority: Option<i64>,
}

impl ModelDescriptor {
    fn visible(&self) -> bool {
        self.visibility.as_deref() != Some("hide")
    }

    fn into_profile(self) -> ModelProfile {
        ModelProfile::new(
            self.display_name.unwrap_or_else(|| self.slug.clone()),
            "openai",
            self.slug,
            self.description.unwrap_or_default(),
            self.context_window.unwrap_or_default(),
        )
    }
}

impl LanguageModelProvider for CodexProvider {
    type Model = Codex;
    type Error = CodexError;

    async fn list_models(&self) -> Result<Vec<ModelProfile>, Self::Error> {
        let url = format!("{}/models?client_version={CLIENT_VERSION}", self.base_url);
        let mut backend = client();
        let mut request = backend.get(&url)?.header(
            "authorization",
            format!("Bearer {}", self.credentials.access_token),
        )?;
        if let Some(account_id) = &self.credentials.account_id {
            request = request.header(ACCOUNT_ID_HEADER, account_id)?;
        }
        let mut response: ModelsResponse = request.json().await?;
        response
            .models
            .sort_by_key(|m| std::cmp::Reverse(m.priority.unwrap_or(0)));
        Ok(response
            .models
            .into_iter()
            .filter(ModelDescriptor::visible)
            .map(ModelDescriptor::into_profile)
            .collect())
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl core::future::Future<Output = Result<Self::Model, Self::Error>> + Send {
        let mut model = Codex::new(self.credentials.clone(), name);
        if let Some(callback) = &self.on_refresh {
            model = model.on_refresh({
                let callback = callback.clone();
                move |creds| callback(creds)
            });
        }
        std::future::ready(Ok(model))
    }

    fn profile() -> Profile {
        Profile::new("codex", "OpenAI models through a ChatGPT subscription")
    }
}
