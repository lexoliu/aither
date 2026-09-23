//! Feature-gated async fetch for `LiteLLM` model data.
//!
//! Enable with `features = ["remote"]`.

use std::path::Path;

use crate::registry::ModelRegistry;
use async_fs;

const LITELLM_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

const CACHE_MAX_AGE_SECS: u64 = 24 * 60 * 60;

/// Fetch the latest `LiteLLM` model data and build a registry.
///
/// If `cache_path` is provided and the file is less than 24 hours old,
/// the cached version is used instead of fetching.
///
/// # Errors
///
/// Returns an error if the network request fails or the JSON is invalid.
pub async fn fetch_litellm(cache_path: Option<&Path>) -> Result<ModelRegistry, FetchError> {
    let bytes = fetch_with_cache(LITELLM_URL, cache_path).await?;
    Ok(ModelRegistry::from_litellm_json(&bytes))
}

/// Error type for remote fetch operations.
#[derive(Debug)]
pub enum FetchError {
    /// Network request failed.
    Network(Box<zenwave::Error>),
    /// Body read error.
    Body(zenwave::BodyError),
    /// I/O error reading/writing cache.
    Io(std::io::Error),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "network error: {e}"),
            Self::Body(e) => write!(f, "body read error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for FetchError {}

impl From<zenwave::Error> for FetchError {
    fn from(e: zenwave::Error) -> Self {
        Self::Network(Box::new(e))
    }
}

impl From<zenwave::BodyError> for FetchError {
    fn from(e: zenwave::BodyError) -> Self {
        Self::Body(e)
    }
}

impl From<std::io::Error> for FetchError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

async fn fetch_with_cache(url: &str, cache_path: Option<&Path>) -> Result<Vec<u8>, FetchError> {
    // Check cache freshness
    if let Some(path) = cache_path
        && let Ok(meta) = async_fs::metadata(path).await
        && let Ok(modified) = meta.modified()
    {
        let age = modified.elapsed().unwrap_or_default();
        if age.as_secs() < CACHE_MAX_AGE_SECS {
            return Ok(async_fs::read(path).await?);
        }
    }

    // Fetch from network
    let response = zenwave::get(url).await?;
    let bytes = response.into_body().into_bytes().await?.to_vec();

    // Write cache
    if let Some(path) = cache_path {
        if let Some(parent) = path.parent() {
            async_fs::create_dir_all(parent).await?;
        }
        async_fs::write(path, &bytes).await?;
    }

    Ok(bytes)
}
