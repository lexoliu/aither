use crate::Mistral;
use crate::error::MistralError;
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use std::{future::Future, sync::Arc};

/// Provider capable of listing and instantiating local mistral.rs models.
#[derive(Clone, Debug)]
pub struct MistralProvider {
    inner: Arc<ProviderConfig>,
}

impl MistralProvider {
    /// Create a new provider with no preconfigured model IDs.
    #[must_use] 
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ProviderConfig::default()),
        }
    }

    /// Set default local model id for LLM inference.
    #[must_use]
    pub fn with_llm_model(mut self, model_id: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).llm_model = Some(model_id.into());
        self
    }

    /// Set default local model id for embeddings.
    #[must_use]
    pub fn with_embedding_model(mut self, model_id: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).embedding_model = Some(model_id.into());
        self
    }

    /// Set default local model id for image generation.
    #[must_use]
    pub fn with_image_model(mut self, model_id: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).image_model = Some(model_id.into());
        self
    }
}

impl Default for MistralProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageModelProvider for MistralProvider {
    type Model = Mistral;
    type Error = MistralError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        async move {
            Ok(aither_models::models_for_provider("mistral")
                .map(|model| {
                    ModelProfile::new(
                        model.id,
                        "mistral",
                        model.id,
                        model.name,
                        model.context_window,
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
        let requested = name.to_owned();
        async move {
            let llm_model = cfg.llm_model.clone().unwrap_or(requested);
            let mut model = Mistral::new().with_llm_model(llm_model);
            if let Some(embedding_model) = &cfg.embedding_model {
                model = model.with_embedding_model(embedding_model.clone());
            }
            if let Some(image_model) = &cfg.image_model {
                model = model.with_image_model(image_model.clone());
            }
            Ok(model)
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("mistral", "Local mistral.rs inference provider")
    }
}

#[derive(Clone, Debug, Default)]
struct ProviderConfig {
    llm_model: Option<String>,
    embedding_model: Option<String>,
    image_model: Option<String>,
}
