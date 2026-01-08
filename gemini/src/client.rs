use serde::{Deserialize, Serialize};
use zenwave::{Client, client, header};

use crate::{
    config::{AuthMode, GeminiConfig, USER_AGENT},
    error::GeminiError,
    types::{
        EmbedContentRequest, EmbedContentResponse, GenerateContentRequest, GenerateContentResponse,
    },
};

/// Response from GET /models/{model} endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Model name (e.g., "models/gemini-2.0-flash")
    #[allow(dead_code)]
    pub name: String,
    /// Maximum input tokens the model can accept.
    pub input_token_limit: u32,
    /// Maximum output tokens the model can generate.
    #[allow(dead_code)]
    pub output_token_limit: u32,
}

pub async fn call_generate(
    cfg: &GeminiConfig,
    model: &str,
    request: GenerateContentRequest,
) -> Result<GenerateContentResponse, GeminiError> {
    post_json(cfg, cfg.model_endpoint(model, "generateContent"), &request).await
}

/// Fetch model info including context window size.
pub async fn get_model_info(cfg: &GeminiConfig, model: &str) -> Result<ModelInfo, GeminiError> {
    let model = crate::config::sanitize_model(model);
    get_json(cfg, cfg.endpoint(&model)).await
}

pub async fn embed_content(
    cfg: &GeminiConfig,
    request: EmbedContentRequest,
) -> Result<EmbedContentResponse, GeminiError> {
    post_json(
        cfg,
        cfg.model_endpoint(&cfg.embedding_model, "embedContent"),
        &request,
    )
    .await
}

#[allow(clippy::future_not_send)]
async fn get_json<T: for<'de> serde::Deserialize<'de>>(
    cfg: &GeminiConfig,
    endpoint: String,
) -> Result<T, GeminiError> {
    let mut backend = client();
    let mut builder = backend
        .get(endpoint)
        .map_err(|e| GeminiError::Http(e))?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(|e| GeminiError::Http(e))?;
    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(|e| GeminiError::Http(e))?;
    }
    builder.json().await.map_err(|e| GeminiError::Http(e))
}

#[allow(clippy::future_not_send)]
async fn post_json<T: for<'de> serde::Deserialize<'de> + serde::Serialize, S: Serialize>(
    cfg: &GeminiConfig,
    endpoint: String,
    body: &S,
) -> Result<T, GeminiError> {
    let debug = std::env::var("AITHER_GEMINI_DEBUG").as_deref() == Ok("1");
    if debug {
        if let Ok(json) = serde_json::to_string_pretty(body) {
            eprintln!("Gemini request to {endpoint}:\n{json}");
        }
        if cfg.auth == AuthMode::Query {
            eprintln!("Gemini using query auth (key redacted)");
        }
    }

    let mut attempt = 0;
    loop {
        attempt += 1;
        let mut backend = client();
        let mut builder = backend
            .post(endpoint.clone())
            .map_err(|e| GeminiError::Http(e))?
            .header(header::USER_AGENT.as_str(), USER_AGENT)
            .map_err(|e| GeminiError::Http(e))?;
        if cfg.auth == AuthMode::Header {
            builder = builder
                .header("x-goog-api-key", cfg.api_key.clone())
                .map_err(|e| GeminiError::Http(e))?;
        }
        let builder = builder.json_body(body).map_err(|e| GeminiError::Http(e))?;

        match builder.json().await {
            Ok(res) => {
                if debug {
                    if let Ok(json) = serde_json::to_string_pretty(&res) {
                        eprintln!("Gemini response from {endpoint}:\n{json}");
                    }
                }
                return Ok(res);
            }
            Err(error) => {
                let msg = error.to_string();
                let is_connect = msg.to_ascii_lowercase().contains("connect");
                let should_retry = is_connect && attempt < 3;
                if !should_retry {
                    return Err(GeminiError::Http(error));
                }
                if debug {
                    eprintln!("Gemini connect error, retrying attempt {attempt}: {msg}");
                }
                std::thread::sleep(std::time::Duration::from_millis(200 * attempt as u64));
            }
        }
    }
}
