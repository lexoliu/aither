//! SSE response parsing for the Claude API.

use aither_core::llm::Event as LLMEvent;
use serde::Deserialize;
use serde_json::Value;
use zenwave::sse::Event;

use crate::error::ClaudeError;

/// Initial `message_start` event data.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MessageStartEvent {
    /// The message object.
    pub message: MessageObject,
}

/// Message object from `message_start`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MessageObject {
    /// Unique message ID.
    pub id: String,
    /// Model that generated the response.
    pub model: String,
    /// Reason the response stopped (null while streaming).
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Token usage information.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Token usage information.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Usage {
    /// Number of input tokens.
    pub input_tokens: u32,
    /// Number of output tokens.
    pub output_tokens: u32,
}

/// Content block start event data.
#[derive(Debug, Deserialize)]
pub struct ContentBlockStartEvent {
    /// Index of this content block.
    pub index: usize,
    /// The content block being started.
    pub content_block: ContentBlockType,
}

/// Types of content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum ContentBlockType {
    /// Text content block.
    #[serde(rename = "text")]
    Text {
        /// Initial text (usually empty).
        text: String,
    },
    /// Thinking/reasoning content block.
    #[serde(rename = "thinking")]
    Thinking {
        /// Initial thinking text.
        thinking: String,
    },
    /// Tool use content block.
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input (builds up via deltas).
        #[serde(default)]
        input: Value,
    },
}

/// Content block delta event data.
#[derive(Debug, Deserialize)]
pub struct ContentBlockDeltaEvent {
    /// Index of the content block being updated.
    pub index: usize,
    /// The delta update.
    pub delta: DeltaType,
}

/// Types of delta updates.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
pub enum DeltaType {
    /// Text delta.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// Text fragment to append.
        text: String,
    },
    /// Thinking delta.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        /// Thinking fragment to append.
        thinking: String,
    },
    /// Input JSON delta for tool use.
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        /// Partial JSON to append.
        partial_json: String,
    },
}

/// Message delta event data (final updates).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MessageDeltaEvent {
    /// Delta updates to the message.
    pub delta: MessageDelta,
    /// Updated usage information.
    #[serde(default)]
    pub usage: Option<DeltaUsage>,
}

/// Message-level delta updates.
#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    /// Reason the response stopped.
    pub stop_reason: Option<String>,
}

/// Usage information in message delta.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct DeltaUsage {
    /// Total output tokens.
    pub output_tokens: u32,
}

/// Parsed tool call from a completed `tool_use` block.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Unique ID for this tool use.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed tool input.
    pub input: Value,
}

/// State for tracking content blocks during streaming.
#[derive(Debug, Default)]
pub struct StreamState {
    /// Current content blocks being built.
    pub blocks: Vec<BlockState>,
    /// Completed tool calls.
    pub tool_calls: Vec<ToolCall>,
    /// Final stop reason.
    pub stop_reason: Option<String>,
}

/// State of an individual content block.
#[derive(Debug, Clone)]
pub enum BlockState {
    /// Text block with accumulated text.
    Text(String),
    /// Thinking block with accumulated reasoning.
    Thinking(String),
    /// Tool use block with accumulated JSON.
    ToolUse {
        /// Tool use ID.
        id: String,
        /// Tool name.
        name: String,
        /// Accumulated input JSON string.
        input_json: String,
    },
}

impl StreamState {
    /// Create a new empty stream state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the response requested tool use.
    pub const fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Parse a single SSE event into LLM events.
///
/// Updates the stream state and returns events to emit.
pub fn parse_event(event: &Event, state: &mut StreamState) -> Result<Vec<LLMEvent>, ClaudeError> {
    let event_name = event.event();
    let event_type = event_name.unwrap_or("");
    let data = event.text_data();

    let mut events = Vec::new();

    match event_type {
        "message_start" => {
            // Store message metadata if needed
            let _ev: MessageStartEvent = serde_json::from_str(data)?;
        }
        "content_block_start" => {
            let ev: ContentBlockStartEvent = serde_json::from_str(data)?;
            ensure_block_capacity(state, ev.index);

            match ev.content_block {
                ContentBlockType::Text { text } => {
                    state.blocks[ev.index] = BlockState::Text(text.clone());
                    if !text.is_empty() {
                        events.push(LLMEvent::Text(text));
                    }
                }
                ContentBlockType::Thinking { thinking } => {
                    state.blocks[ev.index] = BlockState::Thinking(thinking.clone());
                    if !thinking.is_empty() {
                        events.push(LLMEvent::Reasoning(thinking));
                    }
                }
                ContentBlockType::ToolUse { id, name, .. } => {
                    state.blocks[ev.index] = BlockState::ToolUse {
                        id,
                        name,
                        input_json: String::new(),
                    };
                }
            }
        }
        "content_block_delta" => {
            let ev: ContentBlockDeltaEvent = serde_json::from_str(data)?;

            if let Some(block) = state.blocks.get_mut(ev.index) {
                match (&mut *block, ev.delta) {
                    (BlockState::Text(text), DeltaType::TextDelta { text: delta }) => {
                        text.push_str(&delta);
                        events.push(LLMEvent::Text(delta));
                    }
                    (
                        BlockState::Thinking(thinking),
                        DeltaType::ThinkingDelta { thinking: delta },
                    ) => {
                        thinking.push_str(&delta);
                        events.push(LLMEvent::Reasoning(delta));
                    }
                    (
                        BlockState::ToolUse { input_json, .. },
                        DeltaType::InputJsonDelta { partial_json },
                    ) => {
                        input_json.push_str(&partial_json);
                    }
                    _ => {
                        // Mismatched delta type - ignore
                    }
                }
            }
        }
        "content_block_stop" => {
            // Block finished - finalize any tool calls
            #[derive(Deserialize)]
            struct StopEvent {
                index: usize,
            }
            let ev: StopEvent = serde_json::from_str(data)?;

            if let Some(BlockState::ToolUse {
                id,
                name,
                input_json,
            }) = state.blocks.get(ev.index)
            {
                // Parse the accumulated JSON
                let input = serde_json::from_str(input_json)
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                state.tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });
            }
        }
        "message_delta" => {
            let ev: MessageDeltaEvent = serde_json::from_str(data)?;
            state.stop_reason = ev.delta.stop_reason;
        }
        "message_stop" | "ping" | "" => {
            // Stream complete or keepalive - no action needed
        }
        "error" => {
            // Parse error response
            #[derive(Deserialize)]
            struct ErrorEvent {
                error: ErrorDetail,
            }
            #[derive(Deserialize)]
            struct ErrorDetail {
                message: String,
            }
            if let Ok(ev) = serde_json::from_str::<ErrorEvent>(data) {
                return Err(ClaudeError::Api(ev.error.message));
            }
            return Err(ClaudeError::Api(data.to_string()));
        }
        _ => {
            // Unknown event type - log but don't fail
            tracing::debug!("Unknown Claude SSE event type: {event_type}");
        }
    }

    Ok(events)
}

/// Ensure the blocks vector has capacity for the given index.
fn ensure_block_capacity(state: &mut StreamState, index: usize) {
    while state.blocks.len() <= index {
        state.blocks.push(BlockState::Text(String::new()));
    }
}

/// Check if an SSE event should be skipped.
pub fn should_skip_event(event: &Event) -> bool {
    let data = event.text_data();
    data.is_empty() || event.event() == Some("ping")
}
