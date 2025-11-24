use crate::{
    client::{Config, OpenAI},
    error::OpenAIError,
};
use aither_core::moderation::{Moderation, ModerationCategory, ModerationResult};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, future::Future, sync::Arc};
use zenwave::{Client, client, error::BoxHttpError, header};

impl Moderation for OpenAI {
    type Error = OpenAIError;

    fn moderate(
        &self,
        content: &str,
    ) -> impl Future<Output = Result<ModerationResult, Self::Error>> + Send {
        let cfg = self.config();
        let text = content.to_owned();
        async move { moderate_once(cfg, text).await }
    }
}

async fn moderate_once(cfg: Arc<Config>, content: String) -> Result<ModerationResult, OpenAIError> {
    let endpoint = cfg.request_url("/moderations");
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
    builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
    if let Some(org) = &cfg.organization {
        builder = builder.header("OpenAI-Organization", org.clone());
    }

    let request = ModerationRequest {
        model: &cfg.moderation_model,
        input: &content,
    };
    builder = builder.json_body(&request);
    let response: ModerationResponse = builder
        .json()
        .await
        .map_err(|error| OpenAIError::Http(BoxHttpError::from(Box::new(error))))?;

    response.into_result()
}

#[derive(Debug, Serialize)]
struct ModerationRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Debug, Deserialize)]
struct ModerationResponse {
    results: Vec<ModerationPayload>,
}

impl ModerationResponse {
    fn into_result(self) -> Result<ModerationResult, OpenAIError> {
        let payload = self
            .results
            .into_iter()
            .next()
            .ok_or_else(|| OpenAIError::Api("moderation response missing results".into()))?;
        let categories = collect_categories(&payload);
        Ok(ModerationResult::new(payload.flagged, categories))
    }
}

#[derive(Debug, Deserialize)]
struct ModerationPayload {
    flagged: bool,
    #[serde(default)]
    categories: HashMap<String, bool>,
    #[serde(default)]
    category_scores: HashMap<String, f32>,
}

fn collect_categories(payload: &ModerationPayload) -> Vec<ModerationCategory> {
    let mut categories = Vec::new();
    if let Some(score) = select_score(payload, "hate") {
        categories.push(ModerationCategory::Hate { score });
    }
    if let Some(score) = select_score(payload, "harassment") {
        categories.push(ModerationCategory::Harassment { score });
    }
    if let Some(score) = select_score(payload, "self-harm") {
        categories.push(ModerationCategory::SelfHarm { score });
    }
    if let Some(score) = select_score(payload, "sexual") {
        categories.push(ModerationCategory::Sexual { score });
    }
    if let Some(score) = select_score(payload, "violence") {
        categories.push(ModerationCategory::Violence { score });
    }
    categories
}

fn select_score(payload: &ModerationPayload, prefix: &str) -> Option<f32> {
    let flagged = payload
        .categories
        .iter()
        .any(|(name, value)| *value && name.starts_with(prefix));
    let mut best = None;
    for (name, score) in &payload.category_scores {
        if name.starts_with(prefix) {
            best = Some(best.map_or(*score, |current: f32| current.max(*score)));
        }
    }
    match (flagged, best) {
        (true, Some(score)) => Some(score),
        (true, None) => Some(1.0),
        (false, Some(score)) if score > 0.0 => Some(score),
        _ => None,
    }
}
