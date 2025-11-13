use std::sync::Arc;

use serde::Serialize;
use zenwave::{Client, client, header};

use crate::{
    config::{AuthMode, GeminiConfig, USER_AGENT},
    error::GeminiError,
    types::{
        EmbedContentRequest, EmbedContentResponse, GenerateContentRequest, GenerateContentResponse,
    },
};

pub(crate) async fn call_generate(
    cfg: Arc<GeminiConfig>,
    model: &str,
    request: GenerateContentRequest,
) -> Result<GenerateContentResponse, GeminiError> {
    post_json(
        cfg.clone(),
        cfg.model_endpoint(model, "generateContent"),
        &request,
    )
    .await
}

pub(crate) async fn embed_content(
    cfg: Arc<GeminiConfig>,
    request: EmbedContentRequest,
) -> Result<EmbedContentResponse, GeminiError> {
    post_json(
        cfg.clone(),
        cfg.model_endpoint(&cfg.embedding_model, "embedContent"),
        &request,
    )
    .await
}

async fn post_json<T: for<'de> serde::Deserialize<'de>, S: Serialize>(
    cfg: Arc<GeminiConfig>,
    endpoint: String,
    body: &S,
) -> Result<T, GeminiError> {
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::USER_AGENT.as_str(), USER_AGENT);
    if let AuthMode::Header = cfg.auth {
        builder = builder.header("x-goog-api-key", cfg.api_key.clone());
    }
    builder = builder.json_body(body).map_err(GeminiError::from)?;
    builder.json().await.map_err(GeminiError::from)
}
