//! Provider detection and construction for the CLI.

use aither_cloud::{Claude, CloudProvider, Gemini, OpenAI};
use anyhow::{Result, bail};

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
    #[must_use]
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
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" | "gpt" => Some(Self::OpenAI),
            "claude" | "anthropic" => Some(Self::Claude),
            "gemini" | "google" => Some(Self::Gemini),
            _ => None,
        }
    }

    /// Get environment variable name for API key.
    #[must_use]
    pub const fn env_var(self) -> &'static str {
        match self {
            Self::OpenAI => "OPENAI_API_KEY",
            Self::Claude => "ANTHROPIC_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
        }
    }

    /// Get default model for this provider.
    #[must_use]
    pub const fn default_model(self) -> &'static str {
        match self {
            Self::OpenAI => "gpt-4o-mini",
            Self::Claude => "claude-sonnet-4-20250514",
            Self::Gemini => "gemini-2.5-flash",
        }
    }

    /// Create a cloud provider from this provider type.
    pub fn create(self, model: &str, base_url: Option<&str>) -> Result<CloudProvider> {
        let api_key = std::env::var(self.env_var())
            .map_err(|_| anyhow::anyhow!("Set {} in your environment", self.env_var()))?;

        Ok(match self {
            Self::OpenAI => {
                let mut client = OpenAI::new(api_key).with_model(model);
                if let Some(url) = base_url {
                    client = client.with_base_url(url);
                    // Use Chat Completions for non-OpenAI endpoints (proxies)
                    if !url.contains("api.openai.com") {
                        client = client.with_chat_completions_api();
                    }
                }
                CloudProvider::from(client)
            }
            Self::Claude => {
                let mut client = Claude::new(api_key).with_model(model);
                if let Some(url) = base_url {
                    client = client.with_base_url(url);
                }
                CloudProvider::from(client)
            }
            Self::Gemini => {
                let mut client = Gemini::new(api_key).with_text_model(model);
                if let Some(url) = base_url {
                    client = client.with_base_url(url);
                }
                CloudProvider::from(client)
            }
        })
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

/// Auto-detect provider from model name or available API keys.
///
/// Priority:
/// 1. If model prefix matches a provider, use that provider
/// 2. Otherwise, check available API keys in order: Gemini, OpenAI, Claude
pub fn auto_detect(model: Option<&str>, base_url: Option<&str>) -> Result<(CloudProvider, String)> {
    // Try to detect from model name
    if let Some(model) = model {
        if let Some(provider) = Provider::from_model(model) {
            return Ok((provider.create(model, base_url)?, model.to_string()));
        }
    }

    // Auto-detect from available API keys
    let providers = [Provider::Gemini, Provider::OpenAI, Provider::Claude];

    for provider in providers {
        if std::env::var(provider.env_var()).is_ok() {
            let model = model.unwrap_or(provider.default_model());
            return Ok((provider.create(model, base_url)?, model.to_string()));
        }
    }

    bail!("No API key found. Set one of: GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY");
}
