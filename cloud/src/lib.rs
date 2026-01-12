//! Unified cloud provider wrapping OpenAI, Claude, Gemini, and GitHub Copilot.
//!
//! This crate provides a single `CloudProvider` enum that implements `LanguageModel`,
//! allowing seamless switching between providers.
//!
//! It also provides `CloudModelProvider` enum that implements `LanguageModelProvider`,
//! allowing unified model listing and instantiation across providers.

pub use aither_claude::{self as claude, Claude, ClaudeProvider};
pub use aither_copilot::{self as copilot, Copilot, CopilotProvider};
pub use aither_gemini::{self as gemini, Gemini, GeminiProvider};
pub use aither_openai::{self as openai, OpenAI, OpenAIProvider};

use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, LanguageModelProvider, model::Profile, provider::Profile as ProviderProfile},
};
use futures_core::Stream;
use futures_lite::StreamExt;

/// Unified cloud provider wrapping OpenAI, Claude, Gemini, and GitHub Copilot.
#[derive(Clone)]
pub enum CloudProvider {
    /// OpenAI GPT models.
    OpenAI(OpenAI),
    /// Anthropic Claude models.
    Claude(Claude),
    /// Google Gemini models.
    Gemini(Gemini),
    /// GitHub Copilot models.
    Copilot(Copilot),
}

impl From<OpenAI> for CloudProvider {
    fn from(client: OpenAI) -> Self {
        Self::OpenAI(client)
    }
}

impl From<Claude> for CloudProvider {
    fn from(client: Claude) -> Self {
        Self::Claude(client)
    }
}

impl From<Gemini> for CloudProvider {
    fn from(client: Gemini) -> Self {
        Self::Gemini(client)
    }
}

impl From<Copilot> for CloudProvider {
    fn from(client: Copilot) -> Self {
        Self::Copilot(client)
    }
}

impl std::fmt::Debug for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI(_) => f.debug_tuple("CloudProvider::OpenAI").finish(),
            Self::Claude(_) => f.debug_tuple("CloudProvider::Claude").finish(),
            Self::Gemini(_) => f.debug_tuple("CloudProvider::Gemini").finish(),
            Self::Copilot(_) => f.debug_tuple("CloudProvider::Copilot").finish(),
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
    /// GitHub Copilot API error.
    #[error("Copilot error: {0}")]
    Copilot(#[from] aither_copilot::CopilotError),
}

impl LanguageModel for CloudProvider {
    type Error = CloudError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let provider = match self {
            Self::OpenAI(inner) => ProviderInner::OpenAI(inner.clone()),
            Self::Claude(inner) => ProviderInner::Claude(inner.clone()),
            Self::Gemini(inner) => ProviderInner::Gemini(inner.clone()),
            Self::Copilot(inner) => ProviderInner::Copilot(inner.clone()),
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
                ProviderInner::Copilot(inner) => {
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
                Self::Copilot(inner) => inner.profile().await,
            }
        }
    }
}

/// Internal enum to allow cloning the provider for async block.
enum ProviderInner {
    OpenAI(OpenAI),
    Claude(Claude),
    Gemini(Gemini),
    Copilot(Copilot),
}

/// Unified model provider wrapping OpenAI, Claude, Gemini, and Copilot providers.
///
/// Implements `LanguageModelProvider` to allow unified model listing and instantiation.
#[derive(Clone, Debug)]
pub enum CloudModelProvider {
    /// OpenAI provider.
    OpenAI(OpenAIProvider),
    /// Anthropic Claude provider.
    Claude(ClaudeProvider),
    /// Google Gemini provider.
    Gemini(GeminiProvider),
    /// GitHub Copilot provider.
    Copilot(CopilotProvider),
}

impl From<OpenAIProvider> for CloudModelProvider {
    fn from(provider: OpenAIProvider) -> Self {
        Self::OpenAI(provider)
    }
}

impl From<ClaudeProvider> for CloudModelProvider {
    fn from(provider: ClaudeProvider) -> Self {
        Self::Claude(provider)
    }
}

impl From<GeminiProvider> for CloudModelProvider {
    fn from(provider: GeminiProvider) -> Self {
        Self::Gemini(provider)
    }
}

impl From<CopilotProvider> for CloudModelProvider {
    fn from(provider: CopilotProvider) -> Self {
        Self::Copilot(provider)
    }
}

impl LanguageModelProvider for CloudModelProvider {
    type Model = CloudProvider;
    type Error = CloudError;

    fn list_models(&self) -> impl std::future::Future<Output = Result<Vec<Profile>, Self::Error>> + Send {
        let provider = self.clone();
        async move {
            match provider {
                CloudModelProvider::OpenAI(p) => p.list_models().await.map_err(CloudError::from),
                CloudModelProvider::Claude(p) => p.list_models().await.map_err(CloudError::from),
                CloudModelProvider::Gemini(p) => p.list_models().await.map_err(CloudError::from),
                CloudModelProvider::Copilot(p) => p.list_models().await.map_err(CloudError::from),
            }
        }
    }

    fn get_model(&self, name: &str) -> impl std::future::Future<Output = Result<Self::Model, Self::Error>> + Send {
        let provider = self.clone();
        let name = name.to_string();
        async move {
            match provider {
                CloudModelProvider::OpenAI(p) => p.get_model(&name).await.map(CloudProvider::from).map_err(CloudError::from),
                CloudModelProvider::Claude(p) => p.get_model(&name).await.map(CloudProvider::from).map_err(CloudError::from),
                CloudModelProvider::Gemini(p) => p.get_model(&name).await.map(CloudProvider::from).map_err(CloudError::from),
                CloudModelProvider::Copilot(p) => p.get_model(&name).await.map(CloudProvider::from).map_err(CloudError::from),
            }
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("cloud", "Unified cloud provider")
    }
}
