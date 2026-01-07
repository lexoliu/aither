use serde::Deserialize;
use serde_json::Value;
use zenwave::sse::Event;

// ============================================================================
// Responses API Streaming Events
// ============================================================================

/// Streaming event from the Responses API.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesStreamEvent {
    /// Response creation started
    #[serde(rename = "response.created")]
    ResponseCreated {
        #[serde(default)]
        response: Option<ResponsesStreamResponse>,
    },
    /// Response is in progress
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,
    /// Response completed
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        response: ResponsesStreamResponse,
    },
    /// Response failed
    #[serde(rename = "response.failed")]
    ResponseFailed {
        #[serde(default)]
        error: Option<ResponsesError>,
    },
    /// Output item added
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        item: ResponsesOutputItem,
        #[serde(default)]
        output_index: usize,
    },
    /// Output item done
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        item: ResponsesOutputItem,
        #[serde(default)]
        output_index: usize,
    },
    /// Content part added
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded,
    /// Content part done
    #[serde(rename = "response.content_part.done")]
    ContentPartDone,
    /// Text delta
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        delta: String,
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        content_index: usize,
    },
    /// Text done
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        #[serde(default)]
        text: String,
    },
    /// Reasoning text delta
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        delta: String,
        #[serde(default)]
        item_id: Option<String>,
    },
    /// Reasoning text done
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone,
    /// Reasoning summary text delta
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        delta: String,
    },
    /// Reasoning summary done
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone,
    /// Function call arguments delta
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        delta: String,
        item_id: String,
        #[serde(default)]
        output_index: usize,
    },
    /// Function call arguments done
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        arguments: String,
        item_id: String,
        #[serde(default)]
        output_index: usize,
    },
    /// Error event
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        message: Option<String>,
        #[serde(default)]
        code: Option<String>,
    },
    /// Catch-all for unknown events
    #[serde(other)]
    Unknown,
}

/// Response object in streaming events
#[derive(Debug, Deserialize, Default)]
pub struct ResponsesStreamResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub output: Vec<ResponsesOutputItem>,
}

/// Output item in streaming response
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    /// Text message
    Message {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        role: Option<String>,
        #[serde(default)]
        content: Vec<ResponsesContentPart>,
    },
    /// Function call
    FunctionCall {
        id: String,
        #[serde(default)]
        call_id: Option<String>,
        name: String,
        arguments: String,
    },
    /// Reasoning output
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummary>,
    },
    /// Catch-all
    #[serde(other)]
    Other,
}

/// Content part in message
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        #[serde(default)]
        text: String,
    },
    #[serde(other)]
    Other,
}

/// Reasoning summary
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningSummary {
    SummaryText {
        #[serde(default)]
        text: String,
    },
    #[serde(other)]
    Other,
}

/// Error in response
#[derive(Debug, Deserialize, Default)]
pub struct ResponsesError {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

// ============================================================================
// Chat Completions Streaming (legacy)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub delta: DeltaMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DeltaMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_content",
        alias = "reasoningContent",
        alias = "thinking",
        alias = "thinking_content",
        alias = "thinkingContent"
    )]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DeltaToolCall {
    pub index: Option<usize>,
    pub id: Option<String>,
    pub function: Option<DeltaToolFunction>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DeltaToolFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

pub fn should_skip_event(event: &Event) -> bool {
    let text = event.text_data();
    text.is_empty() || text.eq_ignore_ascii_case(": ping")
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

impl ChatCompletionResponse {
    pub(crate) fn into_primary(self) -> Option<ChatMessage> {
        self.choices.into_iter().next().map(|choice| choice.message)
    }
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatMessage {
    #[serde(default)]
    content: Option<MessageContent>,
    #[serde(default)]
    tool_calls: Vec<ChatToolCall>,
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_content",
        alias = "reasoningContent",
        alias = "thinking",
        alias = "thinking_content",
        alias = "thinkingContent"
    )]
    reasoning: Option<MessageContent>,
}

impl ChatMessage {
    pub(crate) fn into_parts(self) -> (Vec<String>, Vec<String>, Vec<ChatToolCall>) {
        let texts = self
            .content
            .map(MessageContent::into_segments)
            .unwrap_or_default();
        let reasoning = self
            .reasoning
            .map(MessageContent::into_segments)
            .unwrap_or_default();
        (texts, reasoning, self.tool_calls)
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Blocks(Vec<ContentPart>),
    Text(String),
}

impl MessageContent {
    fn into_segments(self) -> Vec<String> {
        match self {
            Self::Blocks(parts) => parts.into_iter().map(ContentPart::into_text).collect(),
            Self::Text(text) => {
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![text]
                }
            }
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum ContentPart {
    Text {
        #[serde(rename = "type", default)]
        #[allow(dead_code)]
        _kind: Option<String>,
        text: String,
    },
    Inline(String),
}

impl ContentPart {
    fn into_text(self) -> String {
        match self {
            Self::Text { text, .. } | Self::Inline(text) => text,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatToolCall {
    pub(crate) id: String,
    #[serde(rename = "type", default)]
    _kind: Option<String>,
    pub(crate) function: ChatToolFunction,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatToolFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesOutput {
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    output: Vec<ResponseOutputItem>,
}

impl ResponsesOutput {
    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<String>,
        Vec<String>,
        Vec<ResponseToolCall>,
        Option<String>,
    ) {
        let mut text = Vec::new();
        let mut reasoning = Vec::new();
        let mut tool_calls = Vec::new();

        for item in self.output {
            match item {
                ResponseOutputItem::Message { content } => {
                    for part in content {
                        if let ResponseContentPart::OutputText { text: chunk } = part {
                            if !chunk.is_empty() {
                                text.push(chunk);
                            }
                        }
                    }
                }
                ResponseOutputItem::Reasoning { summary, content } => {
                    reasoning.extend(extract_reasoning(summary));
                    reasoning.extend(extract_reasoning(content));
                }
                ResponseOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } => tool_calls.push(ResponseToolCall {
                    call_id,
                    name,
                    arguments,
                }),
                ResponseOutputItem::Unknown => {}
            }
        }

        (text, reasoning, tool_calls, self.id)
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        content: Vec<ResponseContentPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default)]
        summary: Option<Value>,
        #[serde(default)]
        content: Option<Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug)]
pub struct ResponseToolCall {
    pub(crate) call_id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

fn extract_reasoning(value: Option<Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    match value {
        Value::String(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![text]
            }
        }
        Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::String(text) if !text.is_empty() => Some(text),
                Value::Object(map) => map
                    .get("text")
                    .and_then(|inner| inner.as_str())
                    .filter(|text| !text.is_empty())
                    .map(|text| text.to_string()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
