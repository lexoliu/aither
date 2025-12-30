use aither_core::llm::ResponseChunk;
use serde::Deserialize;
use serde_json::Value;
use zenwave::sse::Event;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    pub(crate) fn into_chunk(self) -> ResponseChunk {
        self.into_chunk_filtered(true)
    }

    pub(crate) fn into_chunk_filtered(self, include_reasoning: bool) -> ResponseChunk {
        let mut chunk = ResponseChunk::default();
        for choice in self.choices {
            if let Some(content) = choice.delta.content {
                for text in content.into_segments() {
                    chunk.push_text(text);
                }
            }
            if include_reasoning {
                if let Some(reasoning) = choice.delta.reasoning {
                    for step in reasoning.into_segments() {
                        chunk.push_reasoning(step);
                    }
                }
            }
        }
        chunk
    }
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: DeltaMessage,
}

#[derive(Debug, Deserialize, Default)]
struct DeltaMessage {
    #[serde(default)]
    content: Option<MessageContent>,
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

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Blocks(Vec<DeltaContent>),
    Text(String),
}

impl MessageContent {
    fn into_segments(self) -> Vec<String> {
        match self {
            Self::Blocks(parts) => parts.into_iter().map(DeltaContent::into_text).collect(),
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
enum DeltaContent {
    Text {
        #[serde(rename = "type", default)]
        #[allow(dead_code)]
        _kind: Option<String>,
        text: String,
    },
    Inline(String),
}

impl DeltaContent {
    fn into_text(self) -> String {
        match self {
            Self::Text { text, .. } | Self::Inline(text) => text,
        }
    }
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
pub struct ChatToolCall {
    pub(crate) id: String,
    #[serde(rename = "type", default)]
    pub(crate) kind: Option<String>,
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
