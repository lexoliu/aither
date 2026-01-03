use aither_core::moderation::{Moderation, ModerationCategory, ModerationResult};

use crate::{
    client::call_generate,
    config::Gemini,
    error::GeminiError,
    types::{
        Candidate, GeminiContent, GenerateContentRequest, GenerationConfig, SafetyRating,
        SafetySetting,
    },
};

impl Moderation for Gemini {
    type Error = GeminiError;

    fn moderate(
        &self,
        content: &str,
    ) -> impl core::future::Future<Output = Result<ModerationResult, Self::Error>> + Send {
        let cfg = self.config();
        let text = content.to_owned();
        async move {
            let request = GenerateContentRequest {
                system_instruction: None,
                contents: vec![GeminiContent::text("user", text)],
                generation_config: Some(GenerationConfig {
                    response_mime_type: Some("text/plain".into()),
                    ..GenerationConfig::default()
                }),
                tools: Vec::new(),
                tool_config: None,
                safety_settings: default_safety_settings(),
            };
            let model_id = cfg.text_model.clone();
            let response = call_generate(cfg, &model_id, request).await?;
            let mut flagged = false;
            let mut categories = Vec::new();
            if let Some(feedback) = response.prompt_feedback {
                for rating in feedback.safety_ratings {
                    if let Some(cat) = to_moderation_category(&rating) {
                        flagged |= rating.blocked.unwrap_or(false);
                        categories.push(cat);
                    }
                }
                flagged |= feedback.block_reason.is_some();
            }
            for candidate in response.candidates {
                categories.extend(ratings_to_categories(&candidate));
                flagged |= candidate
                    .safety_ratings
                    .iter()
                    .any(|rating| rating.blocked.unwrap_or(false));
            }
            Ok(ModerationResult::new(flagged, categories))
        }
    }
}

fn default_safety_settings() -> Vec<SafetySetting> {
    vec![
        SafetySetting::new("HARM_CATEGORY_HATE_SPEECH"),
        SafetySetting::new("HARM_CATEGORY_SEXUALLY_EXPLICIT"),
        SafetySetting::new("HARM_CATEGORY_DANGEROUS_CONTENT"),
        SafetySetting::new("HARM_CATEGORY_HARASSMENT"),
        SafetySetting::new("HARM_CATEGORY_CIVIC_INTEGRITY"),
    ]
}

fn ratings_to_categories(candidate: &Candidate) -> Vec<ModerationCategory> {
    candidate
        .safety_ratings
        .iter()
        .filter_map(to_moderation_category)
        .collect()
}

fn to_moderation_category(rating: &SafetyRating) -> Option<ModerationCategory> {
    let score = probability_to_score(rating.probability.as_deref());
    match rating.category.as_str() {
        "HARM_CATEGORY_HATE_SPEECH" => Some(ModerationCategory::Hate { score }),
        "HARM_CATEGORY_HARASSMENT" => Some(ModerationCategory::Harassment { score }),
        "HARM_CATEGORY_SEXUALLY_EXPLICIT" => Some(ModerationCategory::Sexual { score }),
        "HARM_CATEGORY_DANGEROUS_CONTENT" => Some(ModerationCategory::Violence { score }),
        _ => None,
    }
}

fn probability_to_score(label: Option<&str>) -> f32 {
    match label {
        Some("NEGLIGIBLE") => 0.0,
        Some("VERY_LOW") => 0.1,
        Some("LOW") => 0.3,
        Some("MEDIUM") => 0.6,
        Some("HIGH") => 0.9,
        _ => 0.5,
    }
}
