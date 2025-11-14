use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::sanitize_model;

#[derive(Debug, Clone, Serialize)]
pub struct GenerateContentRequest {
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<GeminiContent>,
    pub(crate) contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) generation_config: Option<GenerationConfig>,
    #[serde(rename = "tools", default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<GeminiTool>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<ToolConfig>,
    #[serde(
        rename = "safetySettings",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub(crate) safety_settings: Vec<SafetySetting>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    pub(crate) parts: Vec<Part>,
}

impl GeminiContent {
    pub(crate) fn text(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Some(role.into()),
            parts: vec![Part::text(text)],
        }
    }

    pub(crate) fn with_parts(role: impl Into<String>, parts: Vec<Part>) -> Self {
        Self {
            role: Some(role.into()),
            parts,
        }
    }

    pub(crate) fn function_response(name: impl Into<String>, response: Value) -> Self {
        Self {
            role: Some("user".into()),
            parts: vec![Part::function_response(name.into(), response)],
        }
    }

    pub(crate) fn text_chunks(&self) -> Vec<String> {
        self.parts
            .iter()
            .filter_map(|part| part.text.clone())
            .collect()
    }

    pub(crate) fn first_function_call(&self) -> Option<FunctionCall> {
        self.parts
            .iter()
            .find_map(|part| part.function_call.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    pub(crate) inline_data: Option<InlineData>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    function_call: Option<FunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    function_response: Option<FunctionResponse>,
}

impl Part {
    pub(crate) fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            inline_data: None,
            function_call: None,
            function_response: None,
        }
    }

    pub(crate) fn inline_image(data: Vec<u8>) -> Self {
        Self {
            text: None,
            inline_data: Some(InlineData::new("image/png", data)),
            function_call: None,
            function_response: None,
        }
    }

    pub(crate) fn inline_mask(data: Vec<u8>) -> Self {
        Self {
            text: None,
            inline_data: Some(InlineData::new("application/octet-stream", data)),
            function_call: None,
            function_response: None,
        }
    }

    pub(crate) fn inline_audio(data: Vec<u8>) -> Self {
        Self {
            text: None,
            inline_data: Some(InlineData::new("audio/wav", data)),
            function_call: None,
            function_response: None,
        }
    }

    pub(crate) const fn function_response(name: String, response: Value) -> Self {
        Self {
            text: None,
            inline_data: None,
            function_call: None,
            function_response: Some(FunctionResponse { name, response }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

impl InlineData {
    fn new(mime_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            mime_type: mime_type.into(),
            data: BASE64.encode(data),
        }
    }

    pub(crate) fn decode(&self) -> Result<Vec<u8>, base64::DecodeError> {
        BASE64.decode(&self.data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub(crate) name: String,
    pub(crate) response: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub(crate) function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDeclaration {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parameters: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ToolConfig {
    #[serde(
        rename = "functionCallingConfig",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) function_calling_config: Option<FunctionCallingConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallingConfig {
    pub(crate) mode: FunctionCallingMode,
    #[serde(
        rename = "allowedFunctionNames",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    Auto,
    Any,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<u32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<i32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub(crate) stop_sequences: Option<Vec<String>>,
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    pub(crate) response_mime_type: Option<String>,
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    pub(crate) response_schema: Option<Value>,
    #[serde(rename = "responseModalities", skip_serializing_if = "Option::is_none")]
    pub(crate) response_modalities: Option<Vec<String>>,
    #[serde(rename = "speechConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) speech_config: Option<SpeechConfig>,
}

impl GenerationConfig {
    pub(crate) const fn is_meaningful(&self) -> bool {
        self.temperature.is_some()
            || self.top_p.is_some()
            || self.top_k.is_some()
            || self.max_output_tokens.is_some()
            || self.stop_sequences.is_some()
            || self.response_mime_type.is_some()
            || self.response_schema.is_some()
            || self.response_modalities.is_some()
            || self.speech_config.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    #[serde(rename = "voiceConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) voice_config: Option<VoiceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    #[serde(rename = "prebuiltVoiceConfig")]
    pub(crate) prebuilt: PrebuiltVoiceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrebuiltVoiceConfig {
    #[serde(rename = "voiceName")]
    pub(crate) voice_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub(crate) category: &'static str,
    pub(crate) threshold: &'static str,
}

impl SafetySetting {
    pub(crate) const fn new(category: &'static str) -> Self {
        Self {
            category,
            threshold: "BLOCK_MEDIUM_AND_ABOVE",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub(crate) candidates: Vec<Candidate>,
    #[serde(rename = "promptFeedback")]
    #[serde(default)]
    pub(crate) prompt_feedback: Option<PromptFeedback>,
}

impl GenerateContentResponse {
    pub(crate) fn primary_candidate(&self) -> Option<&Candidate> {
        self.candidates.first()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidate {
    pub(crate) content: Option<GeminiContent>,
    #[serde(rename = "safetyRatings", default)]
    pub(crate) safety_ratings: Vec<SafetyRating>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptFeedback {
    #[serde(rename = "safetyRatings", default)]
    pub(crate) safety_ratings: Vec<SafetyRating>,
    #[serde(rename = "blockReason")]
    #[serde(default)]
    pub(crate) block_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafetyRating {
    pub(crate) category: String,
    #[serde(default)]
    pub(crate) probability: Option<String>,
    #[serde(default)]
    pub(crate) blocked: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbedContentRequest {
    pub(crate) model: String,
    pub(crate) content: GeminiContent,
}

impl EmbedContentRequest {
    pub(crate) fn new(model: &str, content: GeminiContent) -> Self {
        Self {
            model: sanitize_model(model),
            content,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbedContentResponse {
    pub(crate) embedding: EmbeddingValue,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingValue {
    pub(crate) values: Vec<f32>,
}
