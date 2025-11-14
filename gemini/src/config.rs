use std::sync::Arc;

use aither_core::llm::model::Ability;

/// Gemini REST base URL used by the Developer API.
pub const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub const USER_AGENT: &str = "aither-gemini/0.1";
pub const DEFAULT_MODEL: &str = "gemini-2.0-flash";
pub const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-001";
pub const DEFAULT_IMAGE_MODEL: &str = "gemini-2.5-flash-image";
pub const DEFAULT_TTS_MODEL: &str = "gemini-2.5-flash-preview-tts";
pub const DEFAULT_TTS_VOICE: &str = "Kore";

/// Authentication strategy supported by the Gemini backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// Attach `?key=API_KEY` to every request (default).
    Query,
    /// Send the API key via `x-goog-api-key` header.
    Header,
}

/// Native Gemini backend wired up to the `aither` traits.
#[derive(Clone, Debug)]
pub struct GeminiBackend {
    inner: Arc<GeminiConfig>,
}

impl GeminiBackend {
    /// Create a backend using the default chat/embedding models.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(GeminiConfig {
                api_key: api_key.into(),
                base_url: GEMINI_API_BASE_URL.to_string(),
                auth: AuthMode::Query,
                text_model: sanitize_model(DEFAULT_MODEL),
                embedding_model: sanitize_model(DEFAULT_EMBEDDING_MODEL),
                embedding_dimensions: 3072,
                image_model: Some(sanitize_model(DEFAULT_IMAGE_MODEL)),
                tts_model: Some(sanitize_model(DEFAULT_TTS_MODEL)),
                tts_voice: DEFAULT_TTS_VOICE.to_string(),
                native_abilities: vec![Ability::Pdf],
            }),
        }
    }

    /// Override the REST base URL (useful for sandboxes or proxies).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Select header-based authentication.
    #[must_use]
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        Arc::make_mut(&mut self.inner).auth = mode;
        self
    }

    /// Override the default chat model.
    #[must_use]
    pub fn with_text_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).text_model = sanitize_model(model);
        self
    }

    /// Override the embedding model and (optionally) its dimensionality.
    #[must_use]
    pub fn with_embedding_model(mut self, model: impl Into<String>, dim: usize) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.embedding_model = sanitize_model(model);
        cfg.embedding_dimensions = dim;
        self
    }

    /// Override the optional image model.
    #[must_use]
    pub fn with_image_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).image_model = Some(sanitize_model(model));
        self
    }

    /// Disable image generation support.
    #[must_use]
    pub fn without_image_model(mut self) -> Self {
        Arc::make_mut(&mut self.inner).image_model = None;
        self
    }

    /// Override the optional text-to-speech model + voice.
    #[must_use]
    pub fn with_tts(mut self, model: impl Into<String>, voice: impl Into<String>) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.tts_model = Some(sanitize_model(model));
        cfg.tts_voice = voice.into();
        self
    }

    pub(crate) fn config(&self) -> Arc<GeminiConfig> {
        self.inner.clone()
    }

    /// Declare native capabilities exposed by the selected Gemini model.
    #[must_use]
    pub fn with_native_capabilities(
        mut self,
        abilities: impl IntoIterator<Item = Ability>,
    ) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        for ability in abilities {
            if !cfg.native_abilities.contains(&ability) {
                cfg.native_abilities.push(ability);
            }
        }
        self
    }

    /// Declare a single native capability.
    #[must_use]
    pub fn with_native_capability(self, ability: Ability) -> Self {
        self.with_native_capabilities([ability])
    }
}

#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) auth: AuthMode,
    pub(crate) text_model: String,
    pub(crate) embedding_model: String,
    pub(crate) embedding_dimensions: usize,
    pub(crate) image_model: Option<String>,
    pub(crate) tts_model: Option<String>,
    pub(crate) tts_voice: String,
    pub(crate) native_abilities: Vec<Ability>,
}

impl GeminiConfig {
    pub(crate) fn endpoint(&self, suffix: &str) -> String {
        let mut url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        );
        if self.auth == AuthMode::Query {
            let separator = if url.contains('?') { '&' } else { '?' };
            url.push(separator);
            url.push_str("key=");
            url.push_str(&self.api_key);
        }
        url
    }

    pub(crate) fn model_endpoint(&self, model: &str, action: &str) -> String {
        let model = sanitize_model(model);
        self.endpoint(&format!("{model}:{action}"))
    }
}

pub fn sanitize_model(model: impl Into<String>) -> String {
    let model = model.into();
    if model.starts_with("models/") {
        model
    } else {
        format!("models/{model}")
    }
}
