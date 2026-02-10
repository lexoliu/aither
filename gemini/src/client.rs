use std::time::Duration;

use async_io::Timer;
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

/// Maximum retry attempts for transient errors.
const MAX_RETRIES: u32 = 3;

/// Stream generation responses using SSE with retry logic.
pub async fn stream_generate(
    cfg: &GeminiConfig,
    model: &str,
    request: GenerateContentRequest,
) -> Result<
    impl futures_core::Stream<Item = Result<GenerateContentResponse, GeminiError>>,
    GeminiError,
> {
    let endpoint = cfg.model_endpoint(model, "streamGenerateContent") + "&alt=sse";
    let debug = std::env::var("AITHER_GEMINI_DEBUG").as_deref() == Ok("1");

    let mut attempt = 0u32;
    let sse_stream = loop {
        attempt += 1;

        let mut backend = client();
        let mut builder = backend
            .post(endpoint.clone())
            .map_err(GeminiError::from_http)?
            .header(header::USER_AGENT.as_str(), USER_AGENT)
            .map_err(GeminiError::from_http)?;

        if cfg.auth == AuthMode::Header {
            builder = builder
                .header("x-goog-api-key", cfg.api_key.clone())
                .map_err(GeminiError::from_http)?;
        }

        let builder = builder
            .json_body(&request)
            .map_err(GeminiError::from_http)?;

        match builder.sse().await {
            Ok(stream) => break stream,
            Err(e) => {
                let err = GeminiError::from_http(e);
                if err.is_retryable() && attempt < MAX_RETRIES {
                    let delay_secs = err.retry_delay_secs().unwrap_or(2u64.pow(attempt));
                    if debug {
                        tracing::info!(
                            "Gemini streaming error (attempt {}/{}), retrying in {}s: {}",
                            attempt,
                            MAX_RETRIES,
                            delay_secs,
                            err
                        );
                    }
                    Timer::after(Duration::from_secs(delay_secs)).await;
                    continue;
                }
                return Err(err);
            }
        }
    };

    Ok(futures_lite::stream::unfold(
        sse_stream,
        |mut stream| async move {
            use futures_lite::StreamExt;
            loop {
                match stream.next().await {
                    None => return None,
                    Some(result) => {
                        let event = match result {
                            Ok(e) => e,
                            Err(e) => {
                                return Some((Err(GeminiError::Parse(e.to_string())), stream));
                            }
                        };
                        let data = event.text_data();
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<GenerateContentResponse>(data) {
                            Ok(response) => return Some((Ok(response), stream)),
                            Err(e) => {
                                tracing::debug!("SSE parse error: {} for data: {}", e, data);
                                continue;
                            }
                        }
                    }
                }
            }
        },
    ))
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
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;
    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }
    builder.json().await.map_err(GeminiError::from_http)
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

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let mut backend = client();
        let mut builder = backend
            .post(endpoint.clone())
            .map_err(GeminiError::from_http)?
            .header(header::USER_AGENT.as_str(), USER_AGENT)
            .map_err(GeminiError::from_http)?;
        if cfg.auth == AuthMode::Header {
            builder = builder
                .header("x-goog-api-key", cfg.api_key.clone())
                .map_err(GeminiError::from_http)?;
        }
        let builder = builder.json_body(body).map_err(GeminiError::from_http)?;

        match builder.json().await {
            Ok(res) => {
                if debug {
                    if let Ok(json) = serde_json::to_string_pretty(&res) {
                        eprintln!("Gemini response from {endpoint}:\n{json}");
                    }
                }
                return Ok(res);
            }
            Err(e) => {
                let err = GeminiError::from_http(e);
                if err.is_retryable() && attempt < MAX_RETRIES {
                    let delay_secs = err.retry_delay_secs().unwrap_or(2u64.pow(attempt));
                    if debug {
                        tracing::info!(
                            "Gemini POST error (attempt {}/{}), retrying in {}s: {}",
                            attempt,
                            MAX_RETRIES,
                            delay_secs,
                            err
                        );
                    }
                    Timer::after(Duration::from_secs(delay_secs)).await;
                    continue;
                }
                return Err(err);
            }
        }
    }
}
