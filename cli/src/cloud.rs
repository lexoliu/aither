//! Unified cloud provider supporting OpenAI, Anthropic Claude, and Google Gemini.
//!
//! Auto-detects the provider based on environment variables or model name prefix.

use aither_claude::Claude;
use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, model::Profile},
};
use aither_gemini::Gemini;
use aither_openai::OpenAI;
use anyhow::{Result, bail};
use futures_core::Stream;
use futures_lite::StreamExt;

/// Supported cloud providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Provider {
    /// OpenAI GPT models (gpt-4o, o1, etc.)
    OpenAI,
    /// Anthropic Claude models (claude-3, claude-4, etc.)
    Claude,
    /// Google Gemini models (gemini-2.5-pro, gemini-2.5-flash, etc.)
    #[default]
    Gemini,
}

impl Provider {
    /// Detect provider from model name prefix.
    pub fn from_model(model: &str) -> Option<Self> {
        let lower = model.to_lowercase();
        if lower.starts_with("gpt") || lower.starts_with("o1") || lower.starts_with("o3") {
            Some(Self::OpenAI)
        } else if lower.starts_with("claude") {
            Some(Self::Claude)
        } else if lower.starts_with("gemini") {
            Some(Self::Gemini)
        } else {
            None
        }
    }

    /// Parse provider from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" | "gpt" => Some(Self::OpenAI),
            "claude" | "anthropic" => Some(Self::Claude),
            "gemini" | "google" => Some(Self::Gemini),
            _ => None,
        }
    }

    /// Get environment variable name for API key.
    pub const fn env_var(self) -> &'static str {
        match self {
            Self::OpenAI => "OPENAI_API_KEY",
            Self::Claude => "ANTHROPIC_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
        }
    }

    /// Get default model for this provider.
    pub const fn default_model(self) -> &'static str {
        match self {
            Self::OpenAI => "gpt-4o-mini",
            Self::Claude => "claude-sonnet-4-20250514",
            Self::Gemini => "gemini-2.5-flash",
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI => write!(f, "OpenAI"),
            Self::Claude => write!(f, "Claude"),
            Self::Gemini => write!(f, "Gemini"),
        }
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| format!("Unknown provider: {s}"))
    }
}

/// Unified cloud provider wrapping OpenAI, Claude, and Gemini.
#[derive(Clone)]
pub enum CloudProvider {
    /// OpenAI GPT models.
    OpenAI(OpenAI),
    /// Anthropic Claude models.
    Claude(Claude),
    /// Google Gemini models.
    Gemini(Gemini),
}

impl CloudProvider {
    /// Create a cloud provider from explicit provider type and model.
    pub fn new(provider: Provider, model: &str) -> Result<Self> {
        Self::with_base_url(provider, model, None)
    }

    /// Create a cloud provider with a custom base URL.
    pub fn with_base_url(
        provider: Provider,
        model: &str,
        base_url: Option<&str>,
    ) -> Result<Self> {
        let api_key = std::env::var(provider.env_var())
            .map_err(|_| anyhow::anyhow!("Set {} in your environment", provider.env_var()))?;

        Ok(match provider {
            Provider::OpenAI => {
                // Use Responses API by default for better caching and reasoning
                // preservation. Fall back to Chat Completions for proxies.
                let mut client = OpenAI::new(api_key).with_model(model);
                if let Some(url) = base_url {
                    client = client.with_base_url(url);
                    // Use Chat Completions for non-OpenAI endpoints (proxies)
                    if !url.contains("api.openai.com") {
                        client = client.with_chat_completions_api();
                    }
                }
                Self::OpenAI(client)
            }
            Provider::Claude => {
                let mut client = Claude::new(api_key).with_model(model);
                if let Some(url) = base_url {
                    client = client.with_base_url(url);
                }
                Self::Claude(client)
            }
            Provider::Gemini => {
                let mut client = Gemini::new(api_key).with_text_model(model);
                if let Some(url) = base_url {
                    client = client.with_base_url(url);
                }
                Self::Gemini(client)
            }
        })
    }

    /// Auto-detect provider from model name or available API keys.
    ///
    /// Priority:
    /// 1. If model prefix matches a provider, use that provider
    /// 2. Otherwise, check available API keys in order: Gemini, OpenAI, Claude
    pub fn auto(model: Option<&str>, base_url: Option<&str>) -> Result<(Self, String)> {
        // Try to detect from model name
        if let Some(model) = model {
            if let Some(provider) = Provider::from_model(model) {
                return Ok((
                    Self::with_base_url(provider, model, base_url)?,
                    model.to_string(),
                ));
            }
        }

        // Auto-detect from available API keys
        let providers = [Provider::Gemini, Provider::OpenAI, Provider::Claude];

        for provider in providers {
            if std::env::var(provider.env_var()).is_ok() {
                let model = model.unwrap_or(provider.default_model());
                return Ok((
                    Self::with_base_url(provider, model, base_url)?,
                    model.to_string(),
                ));
            }
        }

        bail!(
            "No API key found. Set one of: GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        );
    }

    /// Get the provider type.
    pub const fn provider(&self) -> Provider {
        match self {
            Self::OpenAI(_) => Provider::OpenAI,
            Self::Claude(_) => Provider::Claude,
            Self::Gemini(_) => Provider::Gemini,
        }
    }
}

impl std::fmt::Debug for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI(_) => f.debug_tuple("CloudProvider::OpenAI").finish(),
            Self::Claude(_) => f.debug_tuple("CloudProvider::Claude").finish(),
            Self::Gemini(_) => f.debug_tuple("CloudProvider::Gemini").finish(),
        }
    }
}

/// Error type for cloud provider operations.
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    /// OpenAI API error.
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] aither_openai::OpenAIError),
    /// Claude API error.
    #[error("Claude error: {0}")]
    Claude(#[from] aither_claude::ClaudeError),
    /// Gemini API error.
    #[error("Gemini error: {0}")]
    Gemini(#[from] aither_gemini::GeminiError),
}

impl LanguageModel for CloudProvider {
    type Error = CloudError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        // Clone self to move into the async block
        let provider = match self {
            Self::OpenAI(inner) => ProviderInner::OpenAI(inner.clone()),
            Self::Claude(inner) => ProviderInner::Claude(inner.clone()),
            Self::Gemini(inner) => ProviderInner::Gemini(inner.clone()),
        };

        async_stream::stream! {
            match provider {
                ProviderInner::OpenAI(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
                ProviderInner::Claude(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
                ProviderInner::Gemini(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
            }
        }
    }

    fn profile(&self) -> impl std::future::Future<Output = Profile> + Send {
        async move {
            match self {
                Self::OpenAI(inner) => inner.profile().await,
                Self::Claude(inner) => inner.profile().await,
                Self::Gemini(inner) => inner.profile().await,
            }
        }
    }
}

/// Internal enum to allow cloning the provider for async block.
enum ProviderInner {
    OpenAI(OpenAI),
    Claude(Claude),
    Gemini(Gemini),
}
