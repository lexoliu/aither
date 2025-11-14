use serde::Deserialize;
use zenwave::sse::Event;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    pub(crate) fn into_text(self) -> Option<String> {
        let mut buffer = String::new();
        for choice in self.choices {
            if let Some(content) = choice.delta.content {
                buffer.push_str(&content.into_text());
            }
        }
        if buffer.is_empty() {
            None
        } else {
            Some(buffer)
        }
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
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Blocks(Vec<DeltaContent>),
    Text(String),
}

impl MessageContent {
    fn into_text(self) -> String {
        match self {
            Self::Blocks(parts) => parts.into_iter().map(DeltaContent::into_text).collect(),
            Self::Text(text) => text,
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
