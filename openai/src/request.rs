use aither_core::llm::{
    Attachment, Message, Role,
    model::{
        OpenAIAutoContainer, OpenAICodeInterpreterContainer, OpenAICodeInterpreterTool,
        OpenAIComputerUseTool, OpenAIFileSearchTool, OpenAIImageGenerationTool, OpenAIMcpTool,
        OpenAINativeTools, OpenAIPromptCacheRetention, OpenAIWebSearchTool, Parameters,
        ReasoningEffort, ToolChoice,
    },
    tool::ToolDefinition,
};
use schemars::Schema;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::PROVIDER_NAME;
use crate::attachments::parse_openai_file_url;
use crate::error::OpenAIError;
#[allow(clippy::struct_excessive_bools)]
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
    pub(crate) parallel_tool_calls: Option<bool>,
    pub(crate) logprobs: Option<bool>,
    pub(crate) top_logprobs: Option<u8>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) include_reasoning: bool,
    pub(crate) structured_outputs: bool,
    pub(crate) response_format: Option<Schema>,
    pub(crate) websearch: bool,
    pub(crate) code_execution: bool,
    pub(crate) openai_tools: OpenAINativeTools,
    pub(crate) legacy_max_tokens: bool,
    pub(crate) prompt_cache_key: Option<String>,
    pub(crate) prompt_cache_retention: Option<OpenAIPromptCacheRetention>,
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
            parallel_tool_calls: value.parallel_tool_calls,
            logprobs: value.logprobs,
            top_logprobs: value.top_logprobs,
            reasoning_effort: value.reasoning_effort,
            include_reasoning: value.include_reasoning,
            structured_outputs: value.structured_outputs,
            response_format: value.response_format.clone(),
            websearch: value.websearch,
            code_execution: value.code_execution,
            openai_tools: value.native_tools.openai.clone(),
            legacy_max_tokens: false,
            prompt_cache_key: value
                .cache
                .openai
                .as_ref()
                .and_then(|cache| cache.key.clone()),
            prompt_cache_retention: value
                .cache
                .openai
                .as_ref()
                .and_then(|cache| cache.retention),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessagePayload>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
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
    /// Whether the model may request several tools per turn. Omitted unless the
    /// caller sets it, so the provider's own default applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

impl ChatCompletionRequest {
    pub(crate) fn new(
        model: String,
        messages: Vec<ChatMessagePayload>,
        params: &ParameterSnapshot,
        tools: Option<Vec<ToolPayload>>,
        stream: bool,
    ) -> Self {
        let has_tools = tools.as_ref().is_some_and(|t| !t.is_empty());
        Self {
            model,
            messages,
            stream,
            stream_options: stream.then_some(StreamOptions {
                include_usage: true,
            }),
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
            parallel_tool_calls: has_tools.then_some(params.parallel_tool_calls).flatten(),
            response_format: response_format(params),
            reasoning: reasoning(params),
            prompt_cache_key: params.prompt_cache_key.clone(),
            prompt_cache_retention: prompt_cache_retention(params),
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
    /// Base64 audio input content part.
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudioPayload },
}

/// Image URL payload for vision.
#[derive(Debug, Clone, Serialize)]
pub struct ImageUrlPayload {
    /// URL to the image (can be data URL with base64).
    url: String,
}

/// Base64 audio payload for Chat Completions.
#[derive(Debug, Clone, Serialize)]
pub struct InputAudioPayload {
    data: String,
    format: &'static str,
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
pub struct ToolChoiceFunction {
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
    /// Responses API only: requests reasoning summaries.
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<&'static str>,
}

pub async fn to_chat_messages(
    messages: &[Message],
) -> Result<Vec<ChatMessagePayload>, OpenAIError> {
    let mut payloads = Vec::with_capacity(messages.len());
    for message in messages {
        let role = match message.role() {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        };

        let tool_call_id = message.tool_call_id().map(String::from);
        let tool_calls = if message.tool_calls().is_empty() {
            None
        } else {
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
        };

        let content = build_content(message).await?;
        payloads.push(ChatMessagePayload {
            role,
            content,
            tool_calls,
            tool_call_id,
        });
    }
    Ok(payloads)
}

/// Build content payload for a message.
///
/// Returns simple text for messages without attachments,
/// or multimodal content parts for messages with attachments.
async fn build_content(message: &Message) -> Result<ContentPayload, OpenAIError> {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return Ok(ContentPayload::Text(message.content().to_owned()));
    }

    let mut parts = Vec::with_capacity(attachments.len() + 1);
    for attachment in attachments {
        let media_type = attachment.media_type().as_ref();
        if media_type.starts_with("image/") {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrlPayload {
                    url: attachment_image_url(attachment).await?,
                },
            });
        } else if media_type.starts_with("audio/") {
            parts.push(ContentPart::InputAudio {
                input_audio: attachment_audio_payload(attachment).await?,
            });
        } else {
            return Err(OpenAIError::Api(format!(
                "Chat Completions does not support attachment MIME type '{media_type}'"
            )));
        }
    }

    if !message.content().is_empty() {
        parts.push(ContentPart::Text {
            text: message.content().to_owned(),
        });
    }

    Ok(ContentPayload::Parts(parts))
}

/// Flatten message content to a simple string.
fn flatten_content(message: &Message) -> String {
    message.content().to_owned()
}

async fn attachment_image_url(attachment: &Attachment) -> Result<String, OpenAIError> {
    let url = attachment.url();
    match url.scheme() {
        "data" | "http" | "https" => Ok(url.as_str().to_string()),
        "file" => read_file_to_data_url(attachment).await,
        scheme => Err(OpenAIError::Api(format!(
            "OpenAI does not support image attachment URL scheme '{scheme}'"
        ))),
    }
}

async fn attachment_audio_payload(
    attachment: &Attachment,
) -> Result<InputAudioPayload, OpenAIError> {
    let media_type = attachment.media_type().as_ref();
    let format = match media_type {
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/x-wav" => "wav",
        _ => {
            return Err(OpenAIError::Api(format!(
                "OpenAI audio input supports only MP3 and WAV, not '{media_type}'"
            )));
        }
    };

    let data = match attachment.url().scheme() {
        "data" => {
            let after_data = attachment
                .url()
                .as_str()
                .strip_prefix("data:")
                .ok_or_else(|| OpenAIError::Api("Malformed audio data URL".to_string()))?;
            let (header, data) = after_data
                .split_once(',')
                .ok_or_else(|| OpenAIError::Api("Audio data URL is missing payload".to_string()))?;
            let encoded_media_type = header.strip_suffix(";base64").ok_or_else(|| {
                OpenAIError::Api("Audio data URL must use base64 encoding".to_string())
            })?;
            if encoded_media_type != media_type {
                return Err(OpenAIError::Api(format!(
                    "Attachment MIME type '{media_type}' does not match data URL MIME type '{encoded_media_type}'"
                )));
            }
            data.to_string()
        }
        "file" => read_file_base64(attachment.url()).await?,
        scheme => {
            return Err(OpenAIError::Api(format!(
                "OpenAI audio input requires a data: or file: URL, not '{scheme}'"
            )));
        }
    };

    Ok(InputAudioPayload { data, format })
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_file_to_data_url(attachment: &Attachment) -> Result<String, OpenAIError> {
    let data = read_file_base64(attachment.url()).await?;
    Ok(format!("data:{};base64,{data}", attachment.media_type()))
}

#[cfg(target_arch = "wasm32")]
async fn read_file_to_data_url(_attachment: &Attachment) -> Result<String, OpenAIError> {
    Err(OpenAIError::Api(
        "file:// attachments are not supported on wasm32".to_string(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_file_base64(url: &url::Url) -> Result<String, OpenAIError> {
    use base64::Engine;

    let path = url.to_file_path().map_err(|()| {
        OpenAIError::Api("Attachment file URL could not be converted to a path".to_string())
    })?;
    let data = async_fs::read(&path).await.map_err(|error| {
        OpenAIError::Api(format!(
            "Failed to read attachment '{}': {error}",
            path.display()
        ))
    })?;
    Ok(base64::engine::general_purpose::STANDARD.encode(data))
}

#[cfg(target_arch = "wasm32")]
async fn read_file_base64(_url: &url::Url) -> Result<String, OpenAIError> {
    Err(OpenAIError::Api(
        "file:// attachments are not supported on wasm32".to_string(),
    ))
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
        .or({
            if params.structured_outputs {
                Some(ResponseFormatPayload::JsonObject)
            } else {
                None
            }
        })
}

/// Maps the portable effort ladder onto `OpenAI`'s `reasoning_effort` vocabulary.
///
/// `OpenAI` accepts the whole ladder, though which levels a given model honours
/// varies by model — the API rejects an unsupported one, which is the failure
/// the caller should see rather than a silently substituted level.
const fn openai_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

fn reasoning(params: &ParameterSnapshot) -> Option<ReasoningPayload> {
    params.reasoning_effort.map(|effort| ReasoningPayload {
        effort: Some(openai_effort(effort)),
        summary: None,
    })
}

/// Builds the Responses API `reasoning` config.
///
/// The Responses API returns reasoning summaries only when `summary` is
/// requested. Without it the `response.reasoning_summary_text.*` events never
/// arrive, so asking for reasoning without asking for the summary is silently
/// a no-op.
fn responses_reasoning(params: &ParameterSnapshot) -> Option<ReasoningPayload> {
    let summary = params.include_reasoning.then_some("auto");
    let effort = params.reasoning_effort.map(openai_effort);
    if effort.is_none() && summary.is_none() {
        return None;
    }
    Some(ReasoningPayload { effort, summary })
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
    pub(crate) const fn tool_output(call_id: String, output: String) -> Self {
        Self {
            role: "tool",
            content: ContentPayload::Text(output),
            tool_calls: None,
            tool_call_id: Some(call_id),
        }
    }

    pub(crate) const fn assistant_tool_calls(
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
    /// A reasoning item replayed verbatim from a previous response.
    ///
    /// `OpenAI`'s stateless flow requires the reasoning item to be appended back
    /// into `input` alongside the function call it produced; without it the
    /// model loses the reasoning behind the call it is being given a result for.
    Reasoning {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
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
#[allow(clippy::enum_variant_names)]
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
        #[serde(flatten)]
        source: InputFileSource,
    },
}

#[derive(Debug, Serialize, Clone)]
pub struct InputImageSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
}

impl InputImageSource {
    const fn from_url(url: String) -> Self {
        Self {
            image_url: Some(url),
            file_id: None,
        }
    }

    const fn from_file_id(file_id: String) -> Self {
        Self {
            image_url: None,
            file_id: Some(file_id),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct InputFileSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_url: Option<String>,
}

impl InputFileSource {
    const fn from_file_id(file_id: String) -> Self {
        Self {
            file_id: Some(file_id),
            file_url: None,
        }
    }

    const fn from_url(file_url: String) -> Self {
        Self {
            file_id: None,
            file_url: Some(file_url),
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

fn require_non_empty_call_id<'a>(
    call_id: Option<&'a str>,
    message_kind: &str,
) -> Result<&'a str, OpenAIError> {
    let Some(call_id) = call_id else {
        return Err(OpenAIError::Api(format!(
            "{message_kind} is missing tool_call_id"
        )));
    };
    if call_id.trim().is_empty() {
        return Err(OpenAIError::Api(format!(
            "{message_kind} has an empty tool_call_id"
        )));
    }
    Ok(call_id)
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
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
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
        let has_tools = tools.as_ref().is_some_and(|items| !items.is_empty());
        Self {
            model,
            input,
            stream,
            temperature: params.temperature,
            top_p: params.top_p,
            max_output_tokens: params.max_tokens,
            top_logprobs: params.top_logprobs,
            tools,
            tool_choice,
            parallel_tool_calls: has_tools.then_some(params.parallel_tool_calls).flatten(),
            text: responses_text(params),
            reasoning: responses_reasoning(params),
            include: responses_include(params),
            prompt_cache_key: params.prompt_cache_key.clone(),
            prompt_cache_retention: prompt_cache_retention(params),
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
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        external_web_access: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<Value>,
    },
    FileSearch {
        vector_store_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<Value>,
    },
    CodeInterpreter {
        container: CodeInterpreterContainerPayload,
    },
    ImageGeneration {
        #[serde(skip_serializing_if = "Option::is_none")]
        partial_images: Option<u8>,
    },
    Mcp {
        server_label: String,
        server_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        require_approval: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        allowed_tools: Vec<String>,
    },
    ComputerUsePreview {
        display_width: u32,
        display_height: u32,
        environment: String,
    },
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum CodeInterpreterContainerPayload {
    Auto {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        memory_limit: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        file_ids: Vec<String>,
    },
    Existing(String),
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
    },
}

/// Decodes reasoning this crate previously emitted back into input items.
///
/// The payload is this crate's own encoding of the response's `reasoning`
/// output item. State from another provider, and state that no longer parses,
/// is dropped rather than replayed into an API that cannot verify it.
fn replayed_reasoning_items(message: &Message) -> Vec<ResponsesInputItem> {
    #[derive(serde::Deserialize)]
    struct Replayed {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        encrypted_content: Option<String>,
    }

    message
        .reasoning()
        .iter()
        .filter_map(|state| {
            let payload = state.payload_for(PROVIDER_NAME).or_else(|| {
                tracing::debug!(
                    provider = state.provider(),
                    "dropping reasoning state from another provider"
                );
                None
            })?;
            match serde_json::from_str::<Replayed>(payload) {
                Ok(replayed) => Some(ResponsesInputItem::Reasoning {
                    kind: "reasoning",
                    id: replayed.id,
                    encrypted_content: replayed.encrypted_content,
                }),
                Err(error) => {
                    tracing::debug!(%error, "dropping unparsable OpenAI reasoning state");
                    None
                }
            }
        })
        .collect()
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
                    // Assistant message with function calls.
                    // Reasoning leads, because it precedes the calls it
                    // produced in the response it came from.
                    items.extend(replayed_reasoning_items(message));
                    // First add text content if present
                    if !message.content().is_empty() {
                        items.push(ResponsesInputItem::message(
                            "assistant",
                            ResponsesMessageContent::Text(message.content().to_string()),
                        ));
                    }
                    // Add function call items
                    for tc in tool_calls {
                        let call_id =
                            require_non_empty_call_id(Some(tc.id.as_str()), "assistant tool call")?;
                        items.push(ResponsesInputItem::FunctionCall {
                            kind: "function_call",
                            call_id: call_id.to_string(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.to_string(),
                        });
                    }
                }
            }
            Role::Tool => {
                // Tool results must be sent as FunctionCallOutput
                let call_id = require_non_empty_call_id(message.tool_call_id(), "tool result")?;
                items.push(ResponsesInputItem::function_call_output(
                    call_id,
                    message.content().to_string(),
                ));
            }
        }
    }

    validate_responses_input(&items)?;
    Ok(items)
}

fn validate_responses_input(items: &[ResponsesInputItem]) -> Result<(), OpenAIError> {
    for (index, item) in items.iter().enumerate() {
        match item {
            ResponsesInputItem::FunctionCall { call_id, name, .. } => {
                if call_id.trim().is_empty() {
                    return Err(OpenAIError::Api(format!(
                        "responses input[{index}] function_call '{name}' has an empty call_id"
                    )));
                }
            }
            ResponsesInputItem::FunctionCallOutput { call_id, .. } => {
                if call_id.trim().is_empty() {
                    return Err(OpenAIError::Api(format!(
                        "responses input[{index}] function_call_output has an empty call_id"
                    )));
                }
            }
            // Reasoning items carry no call_id to validate; they are replayed
            // verbatim or not at all.
            ResponsesInputItem::Message { .. } | ResponsesInputItem::Reasoning { .. } => {}
        }
    }
    Ok(())
}

fn attachment_to_responses_part(
    attachment: &Attachment,
) -> Result<ResponsesInputContent, OpenAIError> {
    let url = attachment.url();
    let media_type = attachment.media_type().as_ref();
    if media_type.starts_with("audio/") || media_type.starts_with("video/") {
        return Err(OpenAIError::Api(format!(
            "Responses API does not support attachment MIME type '{media_type}'"
        )));
    }

    if let Some((kind, id)) = parse_openai_file_url(url) {
        if media_type.starts_with("image/") {
            if !kind.is_image() {
                return Err(OpenAIError::Api(
                    "Uploaded image attachment was encoded as a generic file".to_string(),
                ));
            }
            return Ok(ResponsesInputContent::InputImage {
                source: InputImageSource::from_file_id(id),
            });
        }
        return Ok(ResponsesInputContent::InputFile {
            source: InputFileSource::from_file_id(id),
        });
    }

    match url.scheme() {
        "http" | "https" if media_type.starts_with("image/") => {
            Ok(ResponsesInputContent::InputImage {
                source: InputImageSource::from_url(url.as_str().to_string()),
            })
        }
        "data" if media_type.starts_with("image/") => Ok(ResponsesInputContent::InputImage {
            source: InputImageSource::from_url(url.as_str().to_string()),
        }),
        "http" | "https" => Ok(ResponsesInputContent::InputFile {
            source: InputFileSource::from_url(url.as_str().to_string()),
        }),
        "file" => Err(OpenAIError::Api(
            "file:// attachments must be uploaded via Files API".to_string(),
        )),
        "data" => Err(OpenAIError::Api(
            "Non-image data: attachments must be persisted and uploaded via Files API".to_string(),
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

pub fn responses_tool_choice(
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
        ToolChoice::Exact(name) => Some(ResponsesToolChoice::Function {
            kind: "function",
            name: name.clone(),
        }),
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
        .or({
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
    if params
        .openai_tools
        .file_search
        .iter()
        .any(|tool| tool.include_results)
    {
        include.push("file_search_call.results");
    }
    if include.is_empty() {
        None
    } else {
        Some(include)
    }
}

impl From<OpenAIWebSearchTool> for ResponsesTool {
    fn from(tool: OpenAIWebSearchTool) -> Self {
        Self::WebSearch {
            external_web_access: tool.external_web_access,
            filters: tool.filters,
            user_location: tool.user_location,
        }
    }
}

impl From<OpenAIFileSearchTool> for ResponsesTool {
    fn from(tool: OpenAIFileSearchTool) -> Self {
        Self::FileSearch {
            vector_store_ids: tool.vector_store_ids,
            max_num_results: tool.max_num_results,
            filters: tool.filters,
        }
    }
}

impl From<OpenAICodeInterpreterTool> for ResponsesTool {
    fn from(tool: OpenAICodeInterpreterTool) -> Self {
        Self::CodeInterpreter {
            container: CodeInterpreterContainerPayload::from(tool.container),
        }
    }
}

impl From<OpenAICodeInterpreterContainer> for CodeInterpreterContainerPayload {
    fn from(container: OpenAICodeInterpreterContainer) -> Self {
        match container {
            OpenAICodeInterpreterContainer::Auto(container) => Self::from(container),
            OpenAICodeInterpreterContainer::Existing(id) => Self::Existing(id),
        }
    }
}

impl From<OpenAIAutoContainer> for CodeInterpreterContainerPayload {
    fn from(container: OpenAIAutoContainer) -> Self {
        Self::Auto {
            kind: "auto",
            memory_limit: container.memory_limit,
            file_ids: container.file_ids,
        }
    }
}

impl From<OpenAIImageGenerationTool> for ResponsesTool {
    fn from(tool: OpenAIImageGenerationTool) -> Self {
        Self::ImageGeneration {
            partial_images: tool.partial_images,
        }
    }
}

impl From<OpenAIMcpTool> for ResponsesTool {
    fn from(tool: OpenAIMcpTool) -> Self {
        Self::Mcp {
            server_label: tool.server_label,
            server_url: tool.server_url,
            require_approval: tool.require_approval,
            allowed_tools: tool.allowed_tools,
        }
    }
}

impl From<OpenAIComputerUseTool> for ResponsesTool {
    fn from(tool: OpenAIComputerUseTool) -> Self {
        Self::ComputerUsePreview {
            display_width: tool.display_width,
            display_height: tool.display_height,
            environment: tool.environment,
        }
    }
}

fn prompt_cache_retention(params: &ParameterSnapshot) -> Option<&'static str> {
    params
        .prompt_cache_retention
        .map(OpenAIPromptCacheRetention::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::model::{
        OpenAICodeInterpreterTool, OpenAIFileSearchTool, OpenAIImageGenerationTool, OpenAIMcpTool,
        OpenAINativeTools, OpenAIPromptCacheRetention, OpenAIWebSearchTool, Parameters, ToolChoice,
    };
    use aither_core::llm::{Attachment, Message, ReasoningState, ToolCall};

    #[tokio::test]
    async fn chat_serializes_typed_image_and_audio_parts() {
        let image = Attachment::new(
            "data:image/png;base64,AA==".parse().expect("image URL"),
            "image/png".parse().expect("image MIME"),
        );
        let audio = Attachment::new(
            "data:audio/wav;base64,AA==".parse().expect("audio URL"),
            "audio/wav".parse().expect("audio MIME"),
        );
        let messages = vec![Message::user("describe both").with_attachments([image, audio])];
        let payload = to_chat_messages(&messages)
            .await
            .expect("encode chat attachments");
        let value = serde_json::to_value(&payload[0]).expect("serialize chat message");
        assert_eq!(value["content"][0]["type"], "image_url");
        assert_eq!(value["content"][1]["type"], "input_audio");
        assert_eq!(value["content"][1]["input_audio"]["format"], "wav");
        assert_eq!(value["content"][2]["type"], "text");
    }

    #[test]
    fn responses_serializes_remote_pdf_as_input_file() {
        let attachment = Attachment::new(
            "https://platform.openai.com/docs/guides/pdf-files/sample.pdf"
                .parse()
                .expect("PDF URL"),
            "application/pdf".parse().expect("PDF MIME"),
        );
        let messages = vec![Message::user("summarize").with_attachment(attachment)];
        let input = to_responses_input(&messages).expect("encode Responses attachment");
        let value = serde_json::to_value(&input[0]).expect("serialize Responses input");
        assert_eq!(value["content"][0]["type"], "input_file");
        assert_eq!(
            value["content"][0]["file_url"],
            "https://platform.openai.com/docs/guides/pdf-files/sample.pdf"
        );
    }

    #[test]
    fn responses_rejects_audio_attachment() {
        let attachment = Attachment::new(
            "https://platform.openai.com/docs/guides/audio/sample.wav"
                .parse()
                .expect("audio URL"),
            "audio/wav".parse().expect("audio MIME"),
        );
        let messages = vec![Message::user("transcribe").with_attachment(attachment)];
        let error = to_responses_input(&messages).expect_err("Responses audio must fail");
        assert!(
            error
                .to_string()
                .contains("does not support attachment MIME type")
        );
    }

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

    #[test]
    fn chat_stream_request_includes_usage_option() {
        let snapshot = ParameterSnapshot::from(&Parameters::default());
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, true);
        let value = serde_json::to_value(&req).expect("serialize stream chat request");
        assert_eq!(value["stream_options"]["include_usage"], true);
    }

    /// Builds a chat request carrying one function tool, so the
    /// `parallel_tool_calls` field is in play.
    fn chat_request_with_tool(params: &Parameters) -> serde_json::Value {
        let snapshot = ParameterSnapshot::from(params);
        let req = ChatCompletionRequest::new(
            "gpt-5".into(),
            Vec::new(),
            &snapshot,
            Some(vec![ToolPayload {
                r#type: "function",
                function: ToolFunction {
                    name: "lookup".to_string(),
                    description: "lookup data".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
            }]),
            false,
        );
        serde_json::to_value(&req).expect("serialize chat request")
    }

    #[test]
    fn chat_request_omits_parallel_tool_calls_by_default() {
        // Unset must stay off the wire so OpenAI applies its own default,
        // rather than aither silently serializing every agent's tool calls.
        let value = chat_request_with_tool(&Parameters::default());
        assert_eq!(value.get("parallel_tool_calls"), None);
    }

    #[test]
    fn chat_request_forwards_explicit_parallel_tool_calls() {
        let enabled = chat_request_with_tool(&Parameters::default().parallel_tool_calls(true));
        assert_eq!(enabled["parallel_tool_calls"], true);

        let disabled = chat_request_with_tool(&Parameters::default().parallel_tool_calls(false));
        assert_eq!(disabled["parallel_tool_calls"], false);
    }

    #[test]
    fn chat_request_omits_parallel_tool_calls_without_tools() {
        let snapshot = ParameterSnapshot::from(&Parameters::default().parallel_tool_calls(true));
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, false);
        let value = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(value.get("parallel_tool_calls"), None);
    }

    /// Builds a Responses request carrying one function tool, so the
    /// `parallel_tool_calls` field is in play.
    fn responses_request_with_tool(params: &Parameters) -> serde_json::Value {
        let snapshot = ParameterSnapshot::from(params);
        let req = ResponsesRequest::new(
            "gpt-5".into(),
            vec![ResponsesInputItem::message(
                "user",
                ResponsesMessageContent::Text("hi".to_string()),
            )],
            &snapshot,
            Some(vec![ResponsesTool::Function {
                name: "lookup".to_string(),
                description: "lookup data".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }]),
            responses_tool_choice(&snapshot, true),
            false,
        );
        serde_json::to_value(&req).expect("serialize responses request")
    }

    /// The Responses API is not Chat Completions. These five are Chat-only —
    /// verified against `ResponseCreateParamsBase` in the `openai-python` SDK,
    /// which is generated from `OpenAI`'s own spec and lists none of them.
    /// Sending them is at best ignored and at worst a 400.
    #[test]
    fn responses_request_sends_no_chat_only_sampling_params() {
        let params = Parameters::default()
            .presence_penalty(0.5)
            .frequency_penalty(0.5)
            .seed(42)
            .logit_bias(vec![("tok".to_string(), 1.0)])
            .stop(vec!["END".to_string()]);
        let value = responses_request_with_tool(&params);
        for field in [
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
            "stop",
        ] {
            assert_eq!(
                value.get(field),
                None,
                "{field} is not a Responses API parameter"
            );
        }
    }

    /// The control for the test above: the same knobs must still reach Chat
    /// Completions, where they are real.
    #[test]
    fn chat_request_still_sends_those_sampling_params() {
        let params = Parameters::default()
            .presence_penalty(0.5)
            .frequency_penalty(0.25)
            .seed(42)
            .stop(vec!["END".to_string()]);
        let value = chat_request_with_tool(&params);
        assert_eq!(value["presence_penalty"], 0.5);
        assert_eq!(value["frequency_penalty"], 0.25);
        assert_eq!(value["seed"], 42);
        assert_eq!(value["stop"][0], "END");
    }

    #[test]
    fn responses_request_omits_parallel_tool_calls_by_default() {
        let value = responses_request_with_tool(&Parameters::default());
        assert_eq!(value.get("parallel_tool_calls"), None);
    }

    #[test]
    fn responses_request_forwards_explicit_parallel_tool_calls() {
        let enabled = responses_request_with_tool(&Parameters::default().parallel_tool_calls(true));
        assert_eq!(enabled["parallel_tool_calls"], true);

        let disabled =
            responses_request_with_tool(&Parameters::default().parallel_tool_calls(false));
        assert_eq!(disabled["parallel_tool_calls"], false);
    }

    #[test]
    fn responses_reasoning_requests_a_summary_when_reasoning_is_included() {
        // Without `summary`, the Responses API never emits
        // `response.reasoning_summary_text.*`, so include_reasoning would be a
        // silent no-op no matter what the response parser handles.
        let snapshot = ParameterSnapshot::from(&Parameters::default().include_reasoning(true));
        let payload = responses_reasoning(&snapshot).expect("reasoning payload");
        let value = serde_json::to_value(payload).expect("serialize reasoning");
        assert_eq!(value["summary"], "auto");
    }

    #[test]
    fn responses_reasoning_carries_effort_without_a_summary() {
        let snapshot =
            ParameterSnapshot::from(&Parameters::default().reasoning_effort(ReasoningEffort::High));
        let payload = responses_reasoning(&snapshot).expect("reasoning payload");
        let value = serde_json::to_value(payload).expect("serialize reasoning");
        assert_eq!(value["effort"], "high");
        assert_eq!(value.get("summary"), None);
    }

    /// The round trip OAI-2 was missing: encrypted reasoning must return to
    /// `input`, ahead of the function call it produced.
    #[test]
    fn reasoning_state_round_trips_ahead_of_its_function_call() {
        let state = ReasoningState::new(
            PROVIDER_NAME,
            serde_json::json!({
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "cipher",
            })
            .to_string(),
        );
        let message = Message::assistant_with_reasoning(
            "",
            vec![ToolCall::new("call_1", "lookup", serde_json::json!({}))],
            vec![state],
        );
        let items = to_responses_input(&[message]).expect("build responses input");
        let value = serde_json::to_value(&items).expect("serialize input");

        assert_eq!(value[0]["type"], "reasoning");
        assert_eq!(value[0]["id"], "rs_1");
        assert_eq!(value[0]["encrypted_content"], "cipher");
        assert_eq!(value[1]["type"], "function_call");
    }

    #[test]
    fn foreign_reasoning_state_is_dropped() {
        let message = Message::assistant_with_reasoning(
            "",
            Vec::new(),
            vec![ReasoningState::new(
                "anthropic",
                serde_json::json!({"type": "thinking", "signature": "sig"}).to_string(),
            )],
        );
        assert!(replayed_reasoning_items(&message).is_empty());
    }

    #[test]
    fn responses_reasoning_is_absent_when_nothing_is_requested() {
        let snapshot = ParameterSnapshot::from(&Parameters::default());
        assert!(responses_reasoning(&snapshot).is_none());
    }

    /// Chat Completions has no `summary` field, so the shared payload must not
    /// grow one just because the Responses builder needs it.
    #[test]
    fn chat_reasoning_never_carries_a_summary() {
        let params = Parameters::default()
            .include_reasoning(true)
            .reasoning_effort(ReasoningEffort::Low);
        let snapshot = ParameterSnapshot::from(&params);
        let payload = reasoning(&snapshot).expect("reasoning payload");
        let value = serde_json::to_value(payload).expect("serialize reasoning");
        assert_eq!(value["effort"], "low");
        assert_eq!(value.get("summary"), None);
    }

    #[test]
    fn responses_tool_choice_exact_serializes_function_choice() {
        let params = Parameters::default().tool_choice(ToolChoice::Exact("lookup".to_string()));
        let snapshot = ParameterSnapshot::from(&params);
        let choice = responses_tool_choice(&snapshot, true).expect("tool choice should exist");
        let json = serde_json::to_value(choice).expect("serialize tool choice");
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "lookup");
    }

    #[test]
    fn responses_request_serializes_prompt_cache_fields() {
        let params = Parameters::default()
            .prompt_cache_key("session:alpha")
            .prompt_cache_retention(OpenAIPromptCacheRetention::Hours24);
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
        assert_eq!(value["prompt_cache_key"], "session:alpha");
        assert_eq!(value["prompt_cache_retention"], "24h");
    }

    #[test]
    fn chat_request_serializes_prompt_cache_fields() {
        let params = Parameters::default()
            .prompt_cache_key("session:beta")
            .prompt_cache_retention(OpenAIPromptCacheRetention::InMemory);
        let snapshot = ParameterSnapshot::from(&params);
        let req = ChatCompletionRequest::new("gpt-5".into(), Vec::new(), &snapshot, None, false);
        let value = serde_json::to_value(&req).expect("serialize chat request");
        assert_eq!(value["prompt_cache_key"], "session:beta");
        assert_eq!(value["prompt_cache_retention"], "in-memory");
    }

    #[test]
    fn responses_request_serializes_openai_native_tools() {
        let native_tools = OpenAINativeTools::default()
            .with_web_search(OpenAIWebSearchTool::default().external_web_access(true))
            .with_file_search(
                OpenAIFileSearchTool::new(vec!["vs_123".to_string()])
                    .max_num_results(5)
                    .include_results(true),
            )
            .with_code_interpreter(OpenAICodeInterpreterTool::auto())
            .with_image_generation(OpenAIImageGenerationTool::default().partial_images(2))
            .with_mcp(OpenAIMcpTool::new("docs", "https://example.com/mcp"));
        let params = Parameters::default().openai_tools(native_tools);
        let snapshot = ParameterSnapshot::from(&params);
        let tools = vec![
            ResponsesTool::from(
                snapshot
                    .openai_tools
                    .web_search
                    .clone()
                    .expect("web search"),
            ),
            ResponsesTool::from(snapshot.openai_tools.file_search[0].clone()),
            ResponsesTool::from(
                snapshot
                    .openai_tools
                    .code_interpreter
                    .clone()
                    .expect("code interpreter"),
            ),
            ResponsesTool::from(
                snapshot
                    .openai_tools
                    .image_generation
                    .clone()
                    .expect("image generation"),
            ),
            ResponsesTool::from(snapshot.openai_tools.mcp[0].clone()),
        ];
        let req = ResponsesRequest::new(
            "gpt-5".into(),
            vec![ResponsesInputItem::message(
                "user",
                ResponsesMessageContent::Text("hi".to_string()),
            )],
            &snapshot,
            Some(tools),
            responses_tool_choice(&snapshot, true),
            false,
        );
        let value = serde_json::to_value(&req).expect("serialize responses request");
        assert_eq!(value["tools"][0]["type"], "web_search");
        assert_eq!(value["tools"][0]["external_web_access"], true);
        assert_eq!(value["tools"][1]["type"], "file_search");
        assert_eq!(value["tools"][1]["vector_store_ids"][0], "vs_123");
        assert_eq!(value["tools"][2]["type"], "code_interpreter");
        assert_eq!(value["tools"][2]["container"]["type"], "auto");
        assert_eq!(value["tools"][3]["type"], "image_generation");
        assert_eq!(value["tools"][3]["partial_images"], 2);
        assert_eq!(value["tools"][4]["type"], "mcp");
        assert_eq!(value["include"][0], "file_search_call.results");
    }

    #[test]
    fn responses_input_rejects_empty_assistant_tool_call_id() {
        let messages = vec![Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new("", "lookup", serde_json::json!({"id": 1}))],
        )];
        let error = to_responses_input(&messages).expect_err("empty tool call id must fail");
        assert_eq!(
            error.to_string(),
            "assistant tool call has an empty tool_call_id"
        );
    }

    #[test]
    fn responses_input_rejects_empty_tool_result_call_id() {
        let messages = vec![Message::tool("", "done")];
        let error = to_responses_input(&messages).expect_err("empty tool result id must fail");
        assert_eq!(error.to_string(), "tool result has an empty tool_call_id");
    }

    #[test]
    fn responses_input_validation_rejects_empty_function_call_item() {
        let error = validate_responses_input(&[ResponsesInputItem::FunctionCall {
            kind: "function_call",
            call_id: String::new(),
            name: "lookup".to_string(),
            arguments: "{}".to_string(),
        }])
        .expect_err("empty function call id must fail");
        assert_eq!(
            error.to_string(),
            "responses input[0] function_call 'lookup' has an empty call_id"
        );
    }
}
