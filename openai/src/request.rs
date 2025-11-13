use aither_core::llm::{Annotation, Message, Role, model::Parameters, tool::ToolDefinition};
use schemars::Schema;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Clone)]
pub struct ParameterSnapshot {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) stop: Option<Vec<String>>,
    pub(crate) logit_bias: Option<HashMap<String, f32>>,
    pub(crate) response_format: Option<Schema>,
    pub(crate) structured_outputs: bool,
    pub(crate) seed: Option<u32>,
    pub(crate) tool_choice: Option<Vec<String>>,
    pub(crate) logprobs: Option<bool>,
    pub(crate) top_logprobs: Option<u8>,
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
            response_format: value.response_format.clone(),
            structured_outputs: value.structured_outputs,
            seed: value.seed,
            tool_choice: value.tool_choice.clone(),
            logprobs: value.logprobs,
            top_logprobs: value.top_logprobs,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatPayload>,
}

impl ChatCompletionRequest {
    pub(crate) fn new(
        model: String,
        messages: Vec<ChatMessagePayload>,
        params: &ParameterSnapshot,
        tools: Option<Vec<ToolPayload>>,
        stream: bool,
    ) -> Self {
        Self {
            model,
            messages,
            stream,
            temperature: params.temperature,
            top_p: params.top_p,
            max_tokens: params.max_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            stop: params.stop.clone(),
            logit_bias: params.logit_bias.clone(),
            seed: params.seed,
            logprobs: params.logprobs,
            top_logprobs: params.top_logprobs,
            tools,
            tool_choice: tool_choice(params),
            response_format: response_format(params),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatMessagePayload {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
pub struct ToolPayload {
    r#type: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoicePayload {
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize)]
struct ToolChoiceFunction {
    name: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormatPayload {
    JsonSchema { json_schema: JsonSchemaPayload },
}

#[derive(Debug, Serialize)]
struct JsonSchemaPayload {
    name: String,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

pub fn to_chat_messages(messages: &[Message]) -> Vec<ChatMessagePayload> {
    messages
        .iter()
        .map(|message| ChatMessagePayload {
            role: match message.role() {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            },
            content: flatten_content(message),
        })
        .collect()
}

fn flatten_content(message: &Message) -> String {
    let mut content = message.content().to_owned();

    if !message.attachments().is_empty() {
        content.push_str("\n\nAttachments:\n");
        for attachment in message.attachments() {
            content.push_str("- ");
            content.push_str(attachment.as_str());
            content.push('\n');
        }
    }

    if !message.annotations().is_empty() {
        content.push_str("\n\nReferences:\n");
        for annotation in message.annotations() {
            match annotation {
                Annotation::Url(url) => {
                    content.push_str("- ");
                    content.push_str(&url.title);
                    content.push_str(": ");
                    content.push_str(url.url.as_str());
                    if !url.content.is_empty() {
                        content.push_str(" (");
                        content.push_str(&url.content);
                        content.push(')');
                    }
                    content.push('\n');
                }
            }
        }
    }

    content
}

pub fn convert_tools(definitions: Vec<ToolDefinition>) -> Option<Vec<ToolPayload>> {
    if definitions.is_empty() {
        return None;
    }

    Some(
        definitions
            .into_iter()
            .map(|tool| ToolPayload {
                r#type: "function",
                function: ToolFunction {
                    name: tool.name.to_string(),
                    description: tool.description.to_string(),
                    parameters: schema_to_value(&tool.arguments),
                },
            })
            .collect(),
    )
}

fn schema_to_value(schema: &Schema) -> Value {
    serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(Map::new()))
}

fn tool_choice(params: &ParameterSnapshot) -> Option<ToolChoicePayload> {
    params.tool_choice.as_ref().and_then(|choices| {
        choices.first().map(|name| ToolChoicePayload::Function {
            kind: "function",
            function: ToolChoiceFunction { name: name.clone() },
        })
    })
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
                Some(ResponseFormatPayload::JsonSchema {
                    json_schema: JsonSchemaPayload {
                        name: "aither.response".into(),
                        schema: Value::Object(Map::new()),
                        strict: Some(true),
                    },
                })
            } else {
                None
            }
        })
}
