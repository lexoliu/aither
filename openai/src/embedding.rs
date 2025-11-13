use crate::{
    client::{Config, OpenAI},
    error::OpenAIError,
};
use aither_core::{EmbeddingModel, Result as CoreResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zenwave::{Client, client, header};

impl EmbeddingModel for OpenAI {
    fn dim(&self) -> usize {
        self.config().embedding_dimensions
    }

    fn embed(&self, text: &str) -> impl core::future::Future<Output = CoreResult<Vec<f32>>> + Send {
        let cfg = self.config();
        let input = text.to_owned();
        async move {
            let vector = embed_once(cfg, input).await?;
            Ok(vector)
        }
    }
}

async fn embed_once(cfg: Arc<Config>, input: String) -> Result<Vec<f32>, OpenAIError> {
    let endpoint = cfg.request_url("/embeddings");
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
    builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
    if let Some(org) = &cfg.organization {
        builder = builder.header("OpenAI-Organization", org.clone());
    }
    let request = EmbeddingRequest {
        model: &cfg.embedding_model,
        input: &input,
    };
    builder = builder.json_body(&request).map_err(OpenAIError::from)?;
    let response: EmbeddingResponse = builder.json().await.map_err(OpenAIError::from)?;
    response
        .data
        .into_iter()
        .next()
        .map(|item| item.embedding)
        .ok_or_else(|| OpenAIError::Api("embedding response missing vector data".into()))
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}
