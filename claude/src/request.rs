//! Request building and message conversion for the Claude API.

use aither_core::llm::{Annotation, Message, Role, model::Parameters, tool::ToolDefinition};
use serde::Serialize;
use serde_json::Value;

/// Claude Messages API request body.
#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    /// Model identifier.
    pub model: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Conversation messages.
    pub messages: Vec<MessagePayload>,
    /// System prompt (extracted from messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Enable streaming.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Available tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolPayload>>,
}

/// Individual message in Claude format.
#[derive(Debug, Clone, Serialize)]
pub struct MessagePayload {
    /// Role: "user" or "assistant".
    pub role: &'static str,
    /// Message content.
    pub content: ContentPayload,
}

/// Message content - either a simple string or array of content blocks.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ContentPayload {
    /// Simple text content.
    Text(String),
    /// Array of content blocks (for multimodal or tool results).
    Blocks(Vec<ContentBlock>),
}

/// Content block types for multimodal messages.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content block.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// Image content block.
    #[serde(rename = "image")]
    Image {
        /// Image source (base64 or URL).
        source: ImageSource,
    },
    /// Tool use block (in assistant responses).
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments.
        input: Value,
    },
    /// Tool result block (in user messages).
    #[serde(rename = "tool_result")]
    ToolResult {
        /// ID of the `tool_use` this is responding to.
        tool_use_id: String,
        /// Tool output content.
        content: String,
    },
}

/// Image source for vision requests.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    /// Base64-encoded image data.
    #[serde(rename = "base64")]
    Base64 {
        /// MIME type (image/jpeg, image/png, image/gif, image/webp).
        media_type: String,
        /// Base64-encoded image data.
        data: String,
    },
    /// URL-referenced image.
    #[serde(rename = "url")]
    Url {
        /// Full URL to the image.
        url: String,
    },
}

/// Tool definition in Claude format.
#[derive(Debug, Clone, Serialize)]
pub struct ToolPayload {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON schema for tool input.
    pub input_schema: Value,
}

/// Snapshot of parameters for request building.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct ParameterSnapshot {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to include reasoning/thinking.
    pub include_reasoning: bool,
}

impl From<&Parameters> for ParameterSnapshot {
    fn from(params: &Parameters) -> Self {
        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            max_tokens: params.max_tokens,
            stop_sequences: params.stop.clone(),
            include_reasoning: params.include_reasoning,
        }
    }
}

/// Convert aither messages to Claude format, extracting system messages.
///
/// Returns (system_prompt, messages) where system messages are concatenated
/// into a single system prompt.
pub fn to_claude_messages(messages: &[Message]) -> (Option<String>, Vec<MessagePayload>) {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut claude_messages: Vec<MessagePayload> = Vec::new();

    for message in messages {
        match message.role() {
            Role::System => {
                system_parts.push(message.content());
            }
            Role::User | Role::Tool => {
                let content = build_user_content(message);
                claude_messages.push(MessagePayload {
                    role: "user",
                    content,
                });
            }
            Role::Assistant => {
                claude_messages.push(MessagePayload {
                    role: "assistant",
                    content: ContentPayload::Text(flatten_content(message)),
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    (system, claude_messages)
}

/// Build content for a user message, handling vision attachments.
fn build_user_content(message: &Message) -> ContentPayload {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return ContentPayload::Text(flatten_content(message));
    }

    let mut blocks: Vec<ContentBlock> = Vec::new();

    // Process image attachments
    for attachment in attachments {
        let url_str = attachment.as_str();
        if let Some(source) = parse_image_source(url_str) {
            blocks.push(ContentBlock::Image { source });
        }
    }

    // Add text content
    let text = flatten_content(message);
    if !text.is_empty() {
        blocks.push(ContentBlock::Text { text });
    }

    // Optimize: if only one text block, use simple string
    if blocks.len() == 1 {
        if let Some(ContentBlock::Text { text }) = blocks.pop() {
            return ContentPayload::Text(text);
        }
    }

    ContentPayload::Blocks(blocks)
}

/// Parse a URL string into an image source.
fn parse_image_source(url: &str) -> Option<ImageSource> {
    if url.starts_with("data:image/") {
        // Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
        let after_data = url.strip_prefix("data:")?;
        let (header, data) = after_data.split_once(',')?;
        let media_type = header.strip_suffix(";base64")?;
        Some(ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        })
    } else if is_image_url(url) {
        Some(ImageSource::Url {
            url: url.to_string(),
        })
    } else {
        None
    }
}

/// Check if a URL appears to be an image.
fn is_image_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".png")
        || lower.ends_with(".gif")
        || lower.ends_with(".webp")
        || lower.contains("/image")
}

/// Flatten message content including annotations.
fn flatten_content(message: &Message) -> String {
    let mut content = message.content().to_owned();

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

/// Convert aither tool definitions to Claude format.
pub fn convert_tools(definitions: Vec<ToolDefinition>) -> Vec<ToolPayload> {
    definitions
        .into_iter()
        .map(|tool| ToolPayload {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.arguments_openai_schema(),
        })
        .collect()
}
