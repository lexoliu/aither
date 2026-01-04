use aither_core::llm::{
    Annotation, Message, Role,
    model::{Parameters, ReasoningEffort, ToolChoice},
    tool::ToolDefinition,
};
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
            response_format: response_format(params),
            reasoning: reasoning(params),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessagePayload {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
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
        .map(|message| ChatMessagePayload {
            role: match message.role() {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            },
            content: flatten_content(message),
            tool_calls: None,
            tool_call_id: None,
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
        ToolChoice::Auto => None,
        ToolChoice::None => Some(ToolChoicePayload::Mode("none")),
        ToolChoice::Required => Some(ToolChoicePayload::Mode("required")),
        ToolChoice::Exact(_) => None,
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

impl ChatMessagePayload {
    pub(crate) fn tool_output(call_id: String, output: String) -> Self {
        Self {
            role: "tool",
            content: output,
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
            content,
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
        content: String,
    },
    FunctionCallOutput {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        output: String,
    },
}

impl ResponsesInputItem {
    pub(crate) fn message(role: impl Into<String>, content: String) -> Self {
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
    ) -> Self {
        Self {
            model,
            input,
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

pub fn to_responses_input(messages: &[Message]) -> Vec<ResponsesInputItem> {
    messages
        .iter()
        .map(|message| {
            let role = match message.role() {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "developer",
                Role::Tool => "tool",
            };
            ResponsesInputItem::message(role, flatten_content(message))
        })
        .collect()
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
            vec![ResponsesInputItem::message("user", "hi".to_string())],
            &snapshot,
            None,
            responses_tool_choice(&snapshot, false),
        );
        let value = serde_json::to_value(&req).expect("serialize responses request");
        assert_eq!(value["text"]["format"]["type"], "json_object");
    }
}
