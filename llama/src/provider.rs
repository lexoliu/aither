use crate::{Llama, LlamaError};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use std::{future::Future, path::PathBuf, sync::Arc};

/// Provider for local llama.cpp GGUF models from disk.
#[derive(Clone, Debug)]
pub struct LlamaProvider {
    inner: Arc<ProviderConfig>,
}

impl LlamaProvider {
    /// Create a provider rooted at a directory containing GGUF models.
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                models_dir: models_dir.into(),
            }),
        }
    }
}

impl LanguageModelProvider for LlamaProvider {
    type Model = Llama;
    type Error = LlamaError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let mut profiles = Vec::new();
            let entries = std::fs::read_dir(&cfg.models_dir)
                .map_err(|err| LlamaError::Model(err.to_string()))?;

            for entry in entries {
                let entry = entry.map_err(|err| LlamaError::Model(err.to_string()))?;
                let path = entry.path();
                let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
                    continue;
                };
                if ext != "gguf" {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                profiles.push(ModelProfile::new(
                    name.to_string(),
                    "llama.cpp",
                    name.to_string(),
                    "Local GGUF model",
                    8192,
                ));
            }

            Ok(profiles)
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let name = name.to_string();
        async move {
            let path = cfg.models_dir.join(name);
            Llama::from_file(path)
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("llama", "Local llama.cpp GGUF provider")
    }
}

#[derive(Debug, Clone)]
struct ProviderConfig {
    models_dir: PathBuf,
}
