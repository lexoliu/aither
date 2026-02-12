use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::sanitize_model;

#[derive(Debug, Clone, Serialize)]
pub struct GenerateContentRequest {
    #[serde(
        rename = "systemInstruction",
        alias = "system_instruction",
        skip_serializing_if = "Option::is_none"
    )]
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
    /// Parts of the content. Defaults to empty if not present in response.
    #[serde(default)]
    pub(crate) parts: Vec<Part>,
}

impl GeminiContent {
    pub(crate) fn text(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Some(role.into()),
            parts: vec![Part::text(text)],
        }
    }

    pub(crate) const fn system(parts: Vec<Part>) -> Self {
        Self { role: None, parts }
    }

    pub(crate) fn with_parts(role: impl Into<String>, parts: Vec<Part>) -> Self {
        Self {
            role: Some(role.into()),
            parts,
        }
    }

    pub(crate) fn function_response_with_signature(
        name: impl Into<String>,
        response: Value,
        thought_signature: Option<String>,
    ) -> Self {
        Self {
            role: Some("user".into()),
            parts: vec![Part::function_response_with_signature(
                name.into(),
                response,
                thought_signature,
            )],
        }
    }

    pub(crate) fn text_chunks(&self) -> Vec<String> {
        self.parts.iter().filter_map(Part::text_chunk).collect()
    }

    pub(crate) fn function_call_parts(&self) -> Vec<(FunctionCall, Option<String>)> {
        self.parts
            .iter()
            .filter_map(|part| {
                part.function_call
                    .clone()
                    .map(|call| (call, part.thought_signature.clone()))
            })
            .collect()
    }

    pub(crate) fn reasoning_chunks(&self) -> Vec<String> {
        let mut chunks = Vec::new();
        for part in &self.parts {
            part.collect_thoughts(&mut chunks);
            part.collect_reasoning(&mut chunks);
        }
        chunks
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub(crate) thought_signature: Option<String>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    pub(crate) inline_data: Option<InlineData>,
    #[serde(rename = "fileData", skip_serializing_if = "Option::is_none")]
    pub(crate) file_data: Option<FileData>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    function_call: Option<FunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    function_response: Option<FunctionResponse>,
    #[serde(rename = "executableCode", skip_serializing_if = "Option::is_none")]
    pub(crate) executable_code: Option<ExecutableCode>,
    #[serde(
        rename = "codeExecutionResult",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) code_execution_result: Option<CodeExecutionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
}

impl Part {
    pub(crate) fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    pub(crate) fn inline_image(data: Vec<u8>) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: Some(InlineData::new("image/png", data)),
            file_data: None,
            function_call: None,
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    pub(crate) fn inline_mask(data: Vec<u8>) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: Some(InlineData::new("application/octet-stream", data)),
            file_data: None,
            function_call: None,
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    pub(crate) fn inline_audio(data: Vec<u8>) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: Some(InlineData::new("audio/wav", data)),
            file_data: None,
            function_call: None,
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    /// Create a part with inline media data of any supported MIME type.
    pub(crate) fn inline_media(mime_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: Some(InlineData::new(mime_type, data)),
            file_data: None,
            function_call: None,
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    /// Create a part referencing an uploaded file via the Files API.
    pub(crate) fn from_file(mime_type: impl Into<String>, file_uri: impl Into<String>) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: Some(FileData::new(mime_type, file_uri)),
            function_call: None,
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    pub(crate) const fn function_response_with_signature(
        name: String,
        response: Value,
        thought_signature: Option<String>,
    ) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature,
            inline_data: None,
            file_data: None,
            function_call: None,
            function_response: Some(FunctionResponse { name, response }),
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    pub(crate) const fn function_call(name: String, args: Value) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
            function_call: Some(FunctionCall { name, args }),
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    pub(crate) const fn function_call_with_signature(
        name: String,
        args: Value,
        thought_signature: Option<String>,
    ) -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature,
            inline_data: None,
            file_data: None,
            function_call: Some(FunctionCall { name, args }),
            function_response: None,
            executable_code: None,
            code_execution_result: None,
            metadata: None,
        }
    }

    fn text_chunk(&self) -> Option<String> {
        if self.is_thought() {
            None
        } else {
            self.text.clone()
        }
    }

    fn collect_thoughts(&self, output: &mut Vec<String>) {
        if self.is_thought() {
            if let Some(text) = &self.text {
                if !text.is_empty() {
                    output.push(text.clone());
                }
            }
        }
    }

    fn collect_reasoning(&self, output: &mut Vec<String>) {
        if let Some(meta) = &self.metadata {
            collect_reasoning_values(meta, output, false);
        }
    }

    fn is_thought(&self) -> bool {
        self.thought.unwrap_or(false)
    }
}

fn collect_reasoning_values(value: &Value, output: &mut Vec<String>, matched: bool) {
    match value {
        Value::String(text) => {
            if matched && !text.is_empty() {
                output.push(text.clone());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_reasoning_values(item, output, matched);
            }
        }
        Value::Object(map) => {
            for (key, val) in map {
                let key_lower = key.to_ascii_lowercase();
                let next_matched = matched
                    || key_lower.contains("reason")
                    || key_lower.contains("think")
                    || key_lower.contains("summary");
                collect_reasoning_values(val, output, next_matched);
            }
        }
        _ => {}
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

/// Reference to a file uploaded via the Files API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    /// MIME type of the file.
    mime_type: String,
    /// URI of the uploaded file (e.g., "<https://generativelanguage.googleapis.com/v1beta/files/xyz>").
    file_uri: String,
}

impl FileData {
    /// Create a new file data reference.
    pub fn new(mime_type: impl Into<String>, file_uri: impl Into<String>) -> Self {
        Self {
            mime_type: mime_type.into(),
            file_uri: file_uri.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableCode {
    pub(crate) language: String,
    pub(crate) code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    pub(crate) outcome: String,
    pub(crate) output: String,
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
#[serde(untagged)] // This allows deserialization to try matching one variant after another
pub enum GeminiTool {
    FunctionTool {
        #[serde(rename = "functionDeclarations")]
        function_declarations: Vec<FunctionDeclaration>,
    },
    GoogleSearchTool {
        #[serde(rename = "googleSearch")]
        google_search: GoogleSearch,
    },
    CodeExecutionTool {
        #[serde(rename = "codeExecution")]
        code_execution: CodeExecution,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearch {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecution {}

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
    #[serde(rename = "responseJsonSchema", skip_serializing_if = "Option::is_none")]
    pub(crate) response_json_schema: Option<Value>,
    #[serde(rename = "responseModalities", skip_serializing_if = "Option::is_none")]
    pub(crate) response_modalities: Option<Vec<String>>,
    #[serde(rename = "speechConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) speech_config: Option<SpeechConfig>,
    #[serde(rename = "imageConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) image_config: Option<ImageConfig>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_config: Option<ThinkingConfig>,
    #[serde(rename = "candidateCount", skip_serializing_if = "Option::is_none")]
    pub(crate) candidate_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<i32>,
    #[serde(rename = "presencePenalty", skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f32>,
    #[serde(rename = "frequencyPenalty", skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f32>,
    #[serde(rename = "responseLogprobs", skip_serializing_if = "Option::is_none")]
    pub(crate) response_logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<i32>,
    #[serde(
        rename = "enableEnhancedCivicAnswers",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) enable_enhanced_civic_answers: Option<bool>,
}

impl GenerationConfig {
    pub(crate) const fn is_meaningful(&self) -> bool {
        self.temperature.is_some()
            || self.top_p.is_some()
            || self.top_k.is_some()
            || self.max_output_tokens.is_some()
            || self.stop_sequences.is_some()
            || self.response_mime_type.is_some()
            || self.response_json_schema.is_some()
            || self.response_modalities.is_some()
            || self.speech_config.is_some()
            || self.image_config.is_some()
            || self.thinking_config.is_some()
            || self.candidate_count.is_some()
            || self.seed.is_some()
            || self.presence_penalty.is_some()
            || self.frequency_penalty.is_some()
            || self.response_logprobs.is_some()
            || self.logprobs.is_some()
            || self.enable_enhanced_civic_answers.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    #[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
    pub(crate) aspect_ratio: Option<String>,
    #[serde(rename = "imageSize", skip_serializing_if = "Option::is_none")]
    pub(crate) image_size: Option<String>,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Candidate {
    pub(crate) content: Option<GeminiContent>,
    #[serde(rename = "finishReason", default)]
    pub(crate) finish_reason: Option<String>,
    #[serde(rename = "safetyRatings", default)]
    pub(crate) safety_ratings: Vec<SafetyRating>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptFeedback {
    #[serde(rename = "safetyRatings", default)]
    pub(crate) safety_ratings: Vec<SafetyRating>,
    #[serde(rename = "blockReason")]
    #[serde(default)]
    pub(crate) block_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbedContentResponse {
    pub(crate) embedding: EmbeddingValue,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingValue {
    pub(crate) values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "includeThoughts", skip_serializing_if = "Option::is_none")]
    pub(crate) include_thoughts: Option<bool>,
    #[serde(rename = "tokenBudget", skip_serializing_if = "Option::is_none")]
    pub(crate) token_budget: Option<i32>,
    #[serde(rename = "thinkingLevel", skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_level: Option<String>, // enum in doc, string here for simplicity
}
