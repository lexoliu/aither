use aither_core::llm::ResponseChunk;
use serde::Deserialize;
use zenwave::sse::Event;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    pub(crate) fn into_chunk(self) -> ResponseChunk {
        let mut chunk = ResponseChunk::default();
        for choice in self.choices {
            if let Some(content) = choice.delta.content {
                for text in content.into_segments() {
                    chunk.push_text(text);
                }
            }
            if let Some(reasoning) = choice.delta.reasoning {
                for step in reasoning.into_segments() {
                    chunk.push_reasoning(step);
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

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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
