use aither_core::llm::{
    Message, Role,
    model::{Parameters, ReasoningEffort, ToolChoice},
    tool::ToolDefinition,
};
use url::Url;

use schemars::Schema;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::attachments::parse_openai_file_url;
use crate::error::OpenAIError;
#[derive(Clone)]
pub struct ParameterSnapshot {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) stop: Option<Vec<String>>,
    pub(crate) logit_bias: Option<HashMap<String, f32>>,
    pub(crate) seed: Option<u32>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) logprobs: Option<bool>,
    pub(crate) top_logprobs: Option<u8>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) include_reasoning: bool,
    pub(crate) structured_outputs: bool,
    pub(crate) response_format: Option<Schema>,
    pub(crate) websearch: bool,
    pub(crate) code_execution: bool,
    pub(crate) legacy_max_tokens: bool,
}

impl From<&Parameters> for ParameterSnapshot {
    fn from(value: &Parameters) -> Self {
        Self {
            temperature: value.temperature,
            top_p: value.top_p,
            max_tokens: value.max_tokens,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop: value.stop.clone(),
            logit_bias: value
                .logit_bias
                .as_ref()
                .map(|pairs| pairs.iter().cloned().collect()),
            seed: value.seed,
            tool_choice: value.tool_choice.clone(),
            logprobs: value.logprobs,
            top_logprobs: value.top_logprobs,
            reasoning_effort: value.reasoning_effort,
            include_reasoning: value.include_reasoning,
            structured_outputs: value.structured_outputs,
            response_format: value.response_format.clone(),
            websearch: value.websearch,
            code_execution: value.code_execution,
            legacy_max_tokens: false,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessagePayload>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max_completion_tokens")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoicePayload>,
    /// Enable parallel tool calls (default: true when tools provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
}

impl ChatCompletionRequest {
    pub(crate) fn new(
        model: String,
        messages: Vec<ChatMessagePayload>,
        params: &ParameterSnapshot,
        tools: Option<Vec<ToolPayload>>,
        stream: bool,
    ) -> Self {
        let has_tools = tools.as_ref().map_or(false, |t| !t.is_empty());
        Self {
            model,
            messages,
            stream,
            temperature: params.temperature,
            top_p: params.top_p,
            max_completion_tokens: params.max_tokens,
            max_tokens: if params.legacy_max_tokens {
                params.max_tokens
            } else {
                None
            },
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            stop: params.stop.clone(),
            logit_bias: params.logit_bias.clone(),
            seed: params.seed,
            logprobs: params.logprobs,
            top_logprobs: params.top_logprobs,
            tools,
            tool_choice: tool_choice(params, has_tools),
            parallel_tool_calls: if has_tools { Some(true) } else { None },
            response_format: response_format(params),
            reasoning: reasoning(params),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessagePayload {
    role: &'static str,
    content: ContentPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Message content - either simple string or array of content parts (for vision).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ContentPayload {
    /// Simple text content.
    Text(String),
    /// Array of content parts (for multimodal messages).
    Parts(Vec<ContentPart>),
}

/// Content part for multimodal messages.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content part.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content part.
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPayload },
}

/// Image URL payload for vision.
#[derive(Debug, Clone, Serialize)]
pub struct ImageUrlPayload {
    /// URL to the image (can be data URL with base64).
    url: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolPayload {
    r#type: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoicePayload {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct ToolChoiceFunction {
    name: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormatPayload {
    JsonSchema { json_schema: JsonSchemaPayload },
    JsonObject,
}

#[derive(Debug, Serialize)]
struct JsonSchemaPayload {
    name: String,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ReasoningPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'static str>,
}

pub fn to_chat_messages(messages: &[Message]) -> Vec<ChatMessagePayload> {
    messages
        .iter()
        .map(|message| {
            let role = match message.role() {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };

            // Handle tool_call_id for Tool messages
            let tool_call_id = message.tool_call_id().map(String::from);

            // Handle tool_calls for Assistant messages
            let tool_calls = if !message.tool_calls().is_empty() {
                Some(
                    message
                        .tool_calls()
                        .iter()
                        .map(|tc| ChatToolCallPayload {
                            id: tc.id.clone(),
                            kind: "function",
                            function: ChatToolFunctionPayload {
                                name: tc.name.clone(),
                                arguments: tc.arguments.to_string(),
                            },
                        })
                        .collect(),
                )
            } else {
                None
            };

            // Build content - use multimodal format if there are attachments
            let content = build_content(message);

            ChatMessagePayload {
                role,
                content,
                tool_calls,
                tool_call_id,
            }
        })
        .collect()
}

/// Build content payload for a message.
///
/// Returns simple text for messages without attachments,
/// or multimodal content parts for messages with attachments.
fn build_content(message: &Message) -> ContentPayload {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return ContentPayload::Text(message.content().to_owned());
    }

    let mut parts = Vec::new();

    // Add image parts first
    for attachment in attachments {
        if let Some(data_url) = url_to_data_url(attachment) {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrlPayload { url: data_url },
            });
        }
    }

    // Add text content
    if !message.content().is_empty() {
        parts.push(ContentPart::Text {
            text: message.content().to_owned(),
        });
    }

    ContentPayload::Parts(parts)
}

/// Flatten message content to a simple string.
///
/// For non-vision contexts (like Responses API), just returns the text content.
fn flatten_content(message: &Message) -> String {
    message.content().to_owned()
}

/// Convert a URL to a data URL suitable for OpenAI vision.
///
/// Handles:
/// - `data:...` URLs - passed through as-is
/// - `file:///path` URLs - reads file and converts to base64 data URL
/// - HTTP/HTTPS URLs - passed through as-is (OpenAI can fetch them)
fn url_to_data_url(url: &url::Url) -> Option<String> {
    match url.scheme() {
        "data" => Some(url.as_str().to_string()),
        "http" | "https" => Some(url.as_str().to_string()),
        "file" => read_file_to_data_url(url),
        _ => {
            tracing::warn!("Unsupported attachment URL scheme: {}", url.scheme());
            None
        }
    }
}

/// Read a file:// URL and convert to a data URL.
fn read_file_to_data_url(url: &url::Url) -> Option<String> {
    use base64::Engine;

    let path = url.to_file_path().ok()?;
    let data = std::fs::read(&path).ok()?;
    let mime_type = mime_from_path(&path)?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

    Some(format!("data:{};base64,{}", mime_type, base64_data))
}

/// Get MIME type from file path extension.
fn mime_from_path(path: &std::path::Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase()
        .as_str()
    {
        // Images
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        // Video (for providers that support it)
        "mp4" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        // Audio (for providers that support it)
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        // Documents
        "pdf" => Some("application/pdf"),
        _ => None,
    }
}

pub fn convert_tools(definitions: Vec<ToolDefinition>) -> Vec<ToolPayload> {
    definitions
        .into_iter()
        .map(|tool| ToolPayload {
            r#type: "function",
            function: ToolFunction {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.arguments_openai_schema(),
            },
        })
        .collect()
}

fn schema_to_value(schema: &Schema) -> Value {
    serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(Map::new()))
}

fn tool_choice(params: &ParameterSnapshot, has_tools: bool) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => Some(ToolChoicePayload::Mode("auto")),
        ToolChoice::None => Some(ToolChoicePayload::Mode("none")),
        ToolChoice::Required => Some(ToolChoicePayload::Mode("required")),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Function {
            kind: "function",
            function: ToolChoiceFunction { name: name.clone() },
        }),
    }
}

fn response_format(params: &ParameterSnapshot) -> Option<ResponseFormatPayload> {
    params
        .response_format
        .as_ref()
        .map(|schema| ResponseFormatPayload::JsonSchema {
            json_schema: JsonSchemaPayload {
                name: "aither.response".into(),
                schema: schema_to_value(schema),
                strict: Some(params.structured_outputs),
            },
        })
        .or_else(|| {
            if params.structured_outputs {
                Some(ResponseFormatPayload::JsonObject)
            } else {
                None
            }
        })
}

fn reasoning(params: &ParameterSnapshot) -> Option<ReasoningPayload> {
    params.reasoning_effort.map(|effort| ReasoningPayload {
        effort: Some(effort.as_str()),
    })
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatToolCallPayload {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: ChatToolFunctionPayload,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatToolFunctionPayload {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[allow(dead_code)]
impl ChatMessagePayload {
    pub(crate) fn tool_output(call_id: String, output: String) -> Self {
        Self {
            role: "tool",
            content: ContentPayload::Text(output),
            tool_calls: None,
            tool_call_id: Some(call_id),
        }
    }

    pub(crate) fn assistant_tool_calls(
        content: String,
        tool_calls: Vec<ChatToolCallPayload>,
    ) -> Self {
        Self {
            role: "assistant",
            content: ContentPayload::Text(content),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponsesInputItem {
    Message {
        role: String,
        content: ResponsesMessageContent,
    },
    FunctionCall {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponsesMessageContent {
    Text(String),
    Parts(Vec<ResponsesInputContent>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputContent {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(flatten)]
        source: InputImageSource,
    },
    InputFile {
        file_id: String,
    },
}

#[derive(Debug, Serialize, Clone)]
struct InputImageSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
}

impl InputImageSource {
    fn from_url(url: String) -> Self {
        Self {
            image_url: Some(url),
            file_id: None,
        }
    }

    fn from_file_id(file_id: String) -> Self {
        Self {
            image_url: None,
            file_id: Some(file_id),
        }
    }
}

#[allow(dead_code)]
impl ResponsesInputItem {
    pub(crate) fn message(role: impl Into<String>, content: ResponsesMessageContent) -> Self {
        Self::Message {
            role: role.into(),
            content,
        }
    }

    pub(crate) fn function_call_output(call_id: impl Into<String>, output: String) -> Self {
        Self::FunctionCallOutput {
            kind: "function_call_output",
            call_id: call_id.into(),
            output,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max_output_tokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<&'static str>>,
}

impl ResponsesRequest {
    pub(crate) fn new(
        model: String,
        input: Vec<ResponsesInputItem>,
        params: &ParameterSnapshot,
        tools: Option<Vec<ResponsesTool>>,
        tool_choice: Option<ResponsesToolChoice>,
        stream: bool,
    ) -> Self {
        Self {
            model,
            input,
            stream,
            temperature: params.temperature,
            top_p: params.top_p,
            max_output_tokens: params.max_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            logit_bias: params.logit_bias.clone(),
            seed: params.seed,
            top_logprobs: params.top_logprobs,
            tools,
            tool_choice,
            text: responses_text(params),
            reasoning: reasoning(params),
            include: responses_include(params),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponseTextConfig {
    format: ResponseTextFormat,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    JsonSchema {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    JsonObject,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function {
        name: String,
        description: String,
        parameters: Value,
    },
    WebSearch,
    CodeInterpreter,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub(crate) enum ResponsesToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: ToolChoiceFunction,
    },
}

pub fn to_responses_input(messages: &[Message]) -> Result<Vec<ResponsesInputItem>, OpenAIError> {
    let mut items = Vec::new();

    for message in messages {
        match message.role() {
            Role::User => {
                let attachments = message.attachments();
                if attachments.is_empty() {
                    items.push(ResponsesInputItem::message(
                        "user",
                        ResponsesMessageContent::Text(flatten_content(message)),
                    ));
                } else {
                    let mut parts = Vec::new();
                    for attachment in attachments {
                        parts.push(attachment_to_responses_part(attachment)?);
                    }
                    if !message.content().is_empty() {
                        parts.push(ResponsesInputContent::InputText {
                            text: message.content().to_owned(),
                        });
                    }
                    items.push(ResponsesInputItem::message(
                        "user",
                        ResponsesMessageContent::Parts(parts),
                    ));
                }
            }
            Role::System => {
                items.push(ResponsesInputItem::message(
                    "developer",
                    ResponsesMessageContent::Text(flatten_content(message)),
                ));
            }
            Role::Assistant => {
                let tool_calls = message.tool_calls();
                if tool_calls.is_empty() {
                    // Regular text response
                    items.push(ResponsesInputItem::message(
                        "assistant",
                        ResponsesMessageContent::Text(flatten_content(message)),
                    ));
                } else {
                    // Assistant message with function calls
                    // First add text content if present
                    if !message.content().is_empty() {
                        items.push(ResponsesInputItem::message(
                            "assistant",
                            ResponsesMessageContent::Text(message.content().to_string()),
                        ));
                    }
                    // Add function call items
                    for tc in tool_calls {
                        items.push(ResponsesInputItem::FunctionCall {
                            kind: "function_call",
                            call_id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.to_string(),
                        });
                    }
                }
            }
            Role::Tool => {
                // Tool results must be sent as FunctionCallOutput
                if let Some(call_id) = message.tool_call_id() {
                    items.push(ResponsesInputItem::function_call_output(
                        call_id,
                        message.content().to_string(),
                    ));
                } else {
                    // Fallback if no call_id (shouldn't happen)
                    items.push(ResponsesInputItem::message(
                        "user",
                        ResponsesMessageContent::Text(flatten_content(message)),
                    ));
                }
            }
        }
    }

    Ok(items)
}

fn attachment_to_responses_part(url: &Url) -> Result<ResponsesInputContent, OpenAIError> {
    if let Some((kind, id)) = parse_openai_file_url(url) {
        if kind.is_image() {
            return Ok(ResponsesInputContent::InputImage {
                source: InputImageSource::from_file_id(id),
            });
        }
        return Ok(ResponsesInputContent::InputFile { file_id: id });
    }

    match url.scheme() {
        "http" | "https" | "data" => Ok(ResponsesInputContent::InputImage {
            source: InputImageSource::from_url(url.as_str().to_string()),
        }),
        "file" => Err(OpenAIError::Api(
            "file:// attachments must be uploaded via Files API".to_string(),
        )),
        other => Err(OpenAIError::Api(format!(
            "Unsupported attachment URL scheme: {other}"
        ))),
    }
}

pub fn convert_responses_tools(definitions: Vec<ToolDefinition>) -> Vec<ResponsesTool> {
    definitions
        .into_iter()
        .map(|tool| ResponsesTool::Function {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.arguments_openai_schema(),
        })
        .collect()
}

pub(crate) fn responses_tool_choice(
    params: &ParameterSnapshot,
    has_tools: bool,
) -> Option<ResponsesToolChoice> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(ResponsesToolChoice::Mode("none")),
        ToolChoice::Required => Some(ResponsesToolChoice::Mode("required")),
        ToolChoice::Exact(_) => None,
    }
}

fn responses_text(params: &ParameterSnapshot) -> Option<ResponseTextConfig> {
    params
        .response_format
        .as_ref()
        .map(|schema| ResponseTextConfig {
            format: ResponseTextFormat::JsonSchema {
                name: Some("aither.response".into()),
                schema: schema_to_value(schema),
                strict: Some(params.structured_outputs),
            },
        })
        .or_else(|| {
            if params.structured_outputs {
                Some(ResponseTextConfig {
                    format: ResponseTextFormat::JsonObject,
                })
            } else {
                None
            }
        })
}

fn responses_include(params: &ParameterSnapshot) -> Option<Vec<&'static str>> {
    let mut include = Vec::new();
    if params.logprobs.unwrap_or(false) {
        include.push("message.output_text.logprobs");
    }
    if params.include_reasoning {
        include.push("reasoning.encrypted_content");
    }
    if include.is_empty() {
        None
    } else {
        Some(include)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::model::Parameters;

    #[test]
    fn chat_json_object_when_structured_outputs_without_schema() {
        let params = Parameters {
            structured_outputs: true,
            ..Parameters::default()
        };
        let snapshot = ParameterSnapshot::from(&params);
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, false);
        let value = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(value["response_format"]["type"], "json_object");
    }

    #[test]
    fn responses_json_object_when_structured_outputs_without_schema() {
        let params = Parameters {
            structured_outputs: true,
            ..Parameters::default()
        };
        let snapshot = ParameterSnapshot::from(&params);
        let req = ResponsesRequest::new(
            "gpt-5".into(),
            vec![ResponsesInputItem::message(
                "user",
                ResponsesMessageContent::Text("hi".to_string()),
            )],
            &snapshot,
            None,
            responses_tool_choice(&snapshot, false),
            false,
        );
        let value = serde_json::to_value(&req).expect("serialize responses request");
        assert_eq!(value["text"]["format"]["type"], "json_object");
    }
}
