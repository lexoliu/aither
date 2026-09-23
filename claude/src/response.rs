//! SSE response parsing for the Claude API.

use aither_core::llm::{Event as LLMEvent, ReasoningState, Usage as TokenUsage};

use crate::PROVIDER_NAME;
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
    /// Number of input tokens written into cache.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of input tokens read from cache.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
    /// Detailed cache creation token counts by TTL.
    #[serde(default)]
    pub cache_creation: Option<CacheCreation>,
}

impl Usage {
    fn cache_write_tokens(&self) -> Option<u32> {
        self.cache_creation_input_tokens
            .or_else(|| self.cache_creation.as_ref().and_then(CacheCreation::total))
    }
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
        /// Initial thinking text. Empty when `display` is `omitted`.
        thinking: String,
        /// Encrypted copy of the full reasoning, replayed back unmodified.
        #[serde(default)]
        signature: String,
    },
    /// Safety-redacted thinking. Carries no readable text, only ciphertext.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        /// Opaque encrypted thinking, replayed back unmodified.
        data: String,
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
    /// Signature delta, arriving once just before the thinking block closes.
    #[serde(rename = "signature_delta")]
    SignatureDelta {
        /// The block's signature.
        signature: String,
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
    #[serde(default)]
    pub output_tokens: Option<u32>,
    /// Number of input tokens written into cache.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of input tokens read from cache.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
    /// Detailed cache creation token counts by TTL.
    #[serde(default)]
    pub cache_creation: Option<CacheCreation>,
}

impl DeltaUsage {
    fn cache_write_tokens(&self) -> Option<u32> {
        self.cache_creation_input_tokens
            .or_else(|| self.cache_creation.as_ref().and_then(CacheCreation::total))
    }
}

/// Detailed cache creation token counts by TTL.
#[derive(Debug, Deserialize)]
pub struct CacheCreation {
    /// Tokens created with 5-minute TTL.
    #[serde(default)]
    pub ephemeral_5m_input_tokens: Option<u32>,
    /// Tokens created with 1-hour TTL.
    #[serde(default)]
    pub ephemeral_1h_input_tokens: Option<u32>,
}

impl CacheCreation {
    fn total(&self) -> Option<u32> {
        // `None` and `Some(0)` mean different things here: the former is
        // "the API said nothing about cache creation", the latter "nothing was
        // cached", so only sum when at least one field was reported.
        match (
            self.ephemeral_5m_input_tokens,
            self.ephemeral_1h_input_tokens,
        ) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
        }
    }
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
    /// Prompt/input token usage.
    pub prompt_tokens: Option<u32>,
    /// Uncached input tokens (`usage.input_tokens` from Claude).
    pub uncached_input_tokens: Option<u32>,
    /// Completion/output token usage.
    pub completion_tokens: Option<u32>,
    /// Prompt cache read token usage.
    pub cache_read_tokens: Option<u32>,
    /// Prompt cache write token usage.
    pub cache_write_tokens: Option<u32>,
    /// Whether usage has already been emitted.
    pub usage_emitted: bool,
}

/// State of an individual content block.
#[derive(Debug, Clone)]
pub enum BlockState {
    /// Text block with accumulated text.
    Text(String),
    /// Thinking block with accumulated reasoning and its signature.
    Thinking {
        /// Accumulated thinking text; empty when `display` is `omitted`.
        text: String,
        /// Signature, which arrives as a single delta before the block closes.
        signature: String,
    },
    /// Safety-redacted thinking block.
    RedactedThinking(String),
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

    fn maybe_usage_event(&mut self) -> Option<LLMEvent> {
        if self.usage_emitted {
            return None;
        }
        if self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.cache_read_tokens.is_none()
            && self.cache_write_tokens.is_none()
            && self.stop_reason.is_none()
        {
            return None;
        }
        self.usage_emitted = true;
        let total_tokens = match (self.prompt_tokens, self.completion_tokens) {
            (Some(prompt), Some(completion)) => Some(prompt + completion),
            _ => None,
        };
        Some(LLMEvent::Usage(TokenUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens,
            reasoning_tokens: None,
            cache_read_tokens: self.cache_read_tokens,
            cache_write_tokens: self.cache_write_tokens,
            cost_usd: None,
            stop_reason: self.stop_reason.clone(),
        }))
    }

    fn refresh_prompt_tokens(&mut self) {
        // As above: report a total only when the API reported at least one of
        // the components, so "not stated" stays distinct from "zero".
        let parts = [
            self.uncached_input_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
        ];
        self.prompt_tokens = parts.iter().any(Option::is_some).then(|| {
            parts
                .iter()
                .flatten()
                .fold(0u32, |total, part| total.saturating_add(*part))
        });
    }
}

/// Parse a single SSE event into LLM events.
///
/// Updates the stream state and returns events to emit.
///
/// # Errors
///
/// Returns [`ClaudeError`] if the event body does not parse, or if the API sent
/// an `error` event.
pub fn parse_event(event: &Event, state: &mut StreamState) -> Result<Vec<LLMEvent>, ClaudeError> {
    let data = event.text_data();

    match event.event().unwrap_or("") {
        "message_start" => {
            state.apply_message_start(serde_json::from_str(data)?);
            Ok(Vec::new())
        }
        "content_block_start" => Ok(state.begin_block(serde_json::from_str(data)?)),
        "content_block_delta" => Ok(state.apply_block_delta(serde_json::from_str(data)?)),
        "content_block_stop" => {
            #[derive(Deserialize)]
            struct StopEvent {
                index: usize,
            }
            let ev: StopEvent = serde_json::from_str(data)?;
            Ok(state.finish_block(ev.index))
        }
        "message_delta" => {
            state.apply_message_delta(serde_json::from_str(data)?);
            Ok(Vec::new())
        }
        "message_stop" => Ok(state.maybe_usage_event().into_iter().collect()),
        // Keepalives carry nothing to emit.
        "ping" | "" => Ok(Vec::new()),
        "error" => Err(parse_error_event(data)),
        unknown => {
            // Anthropic adds event types over time; an unfamiliar one is not a
            // reason to fail the stream.
            tracing::debug!("Unknown Claude SSE event type: {unknown}");
            Ok(Vec::new())
        }
    }
}

/// Reads the message out of an `error` event, falling back to its raw body.
fn parse_error_event(data: &str) -> ClaudeError {
    #[derive(Deserialize)]
    struct ErrorEvent {
        error: ErrorDetail,
    }
    #[derive(Deserialize)]
    struct ErrorDetail {
        message: String,
    }

    serde_json::from_str::<ErrorEvent>(data).map_or_else(
        |_| ClaudeError::Api(data.to_string()),
        |ev| ClaudeError::Api(ev.error.message),
    )
}

impl StreamState {
    /// Records the usage figures the API reports when a message opens.
    fn apply_message_start(&mut self, ev: MessageStartEvent) {
        let Some(usage) = ev.message.usage else {
            return;
        };
        self.uncached_input_tokens = Some(usage.input_tokens);
        self.completion_tokens = Some(usage.output_tokens);
        self.cache_write_tokens = usage.cache_write_tokens();
        self.cache_read_tokens = usage.cache_read_input_tokens;
        self.refresh_prompt_tokens();
    }

    /// Opens a content block, emitting whatever text arrived with it.
    fn begin_block(&mut self, ev: ContentBlockStartEvent) -> Vec<LLMEvent> {
        ensure_block_capacity(self, ev.index);

        match ev.content_block {
            ContentBlockType::Text { text } => {
                self.blocks[ev.index] = BlockState::Text(text.clone());
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![LLMEvent::Text(text)]
                }
            }
            ContentBlockType::Thinking {
                thinking,
                signature,
            } => {
                self.blocks[ev.index] = BlockState::Thinking {
                    text: thinking.clone(),
                    signature,
                };
                if thinking.is_empty() {
                    Vec::new()
                } else {
                    vec![LLMEvent::Reasoning(thinking)]
                }
            }
            // Redacted thinking has no readable text, so it produces no
            // Reasoning event — only state, emitted when the block closes.
            ContentBlockType::RedactedThinking { data } => {
                self.blocks[ev.index] = BlockState::RedactedThinking(data);
                Vec::new()
            }
            ContentBlockType::ToolUse { id, name, .. } => {
                // Emit an initial delta so a UI can show the tool name before
                // any arguments have streamed in.
                let started = LLMEvent::ToolCallDelta {
                    id: id.clone(),
                    name: name.clone(),
                    arguments_fragment: String::new(),
                };
                self.blocks[ev.index] = BlockState::ToolUse {
                    id,
                    name,
                    input_json: String::new(),
                };
                vec![started]
            }
        }
    }

    /// Appends a delta to its block, emitting it if it is text the caller sees.
    ///
    /// A delta whose type does not match the block it names is dropped: the two
    /// disagreeing leaves nothing meaningful to append.
    fn apply_block_delta(&mut self, ev: ContentBlockDeltaEvent) -> Vec<LLMEvent> {
        let Some(block) = self.blocks.get_mut(ev.index) else {
            return Vec::new();
        };

        match (block, ev.delta) {
            (BlockState::Text(text), DeltaType::TextDelta { text: delta }) => {
                text.push_str(&delta);
                vec![LLMEvent::Text(delta)]
            }
            (BlockState::Thinking { text, .. }, DeltaType::ThinkingDelta { thinking: delta }) => {
                text.push_str(&delta);
                vec![LLMEvent::Reasoning(delta)]
            }
            // Arrives once, just before content_block_stop. With
            // `display: "omitted"` it is the only delta the block produces.
            (
                BlockState::Thinking { signature, .. },
                DeltaType::SignatureDelta {
                    signature: delta, ..
                },
            ) => {
                signature.push_str(&delta);
                Vec::new()
            }
            (
                BlockState::ToolUse {
                    id,
                    name,
                    input_json,
                },
                DeltaType::InputJsonDelta { partial_json },
            ) => {
                input_json.push_str(&partial_json);
                vec![LLMEvent::ToolCallDelta {
                    id: id.clone(),
                    name: name.clone(),
                    arguments_fragment: input_json.clone(),
                }]
            }
            _ => Vec::new(),
        }
    }

    /// Closes a content block, turning a finished tool-use block into a call.
    ///
    /// Arguments that did not parse become an empty object rather than failing
    /// the stream: the tool reports the mismatch far more usefully than a
    /// truncated JSON error would.
    /// A closed thinking block emits its state here rather than at block start,
    /// because the signature arrives last: only now is the block complete
    /// enough to be replayed. The payload is the whole block re-serialized, so
    /// that `thinking` and `redacted_thinking` — which have different shapes —
    /// both round-trip without core knowing either.
    fn finish_block(&mut self, index: usize) -> Vec<LLMEvent> {
        match self.blocks.get(index) {
            Some(BlockState::ToolUse {
                id,
                name,
                input_json,
            }) => {
                let input = serde_json::from_str(input_json)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                self.tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });
                Vec::new()
            }
            Some(BlockState::Thinking { text, signature }) => {
                // An unsigned block cannot be verified on replay, so there is
                // nothing worth preserving.
                if signature.is_empty() {
                    return Vec::new();
                }
                let block = serde_json::json!({
                    "type": "thinking",
                    "thinking": text,
                    "signature": signature,
                });
                vec![LLMEvent::ReasoningState(ReasoningState::new(
                    PROVIDER_NAME,
                    block.to_string(),
                ))]
            }
            Some(BlockState::RedactedThinking(data)) => {
                let block = serde_json::json!({
                    "type": "redacted_thinking",
                    "data": data,
                });
                vec![LLMEvent::ReasoningState(ReasoningState::new(
                    PROVIDER_NAME,
                    block.to_string(),
                ))]
            }
            Some(BlockState::Text(_)) | None => Vec::new(),
        }
    }

    /// Folds in the running usage figures the API reports as it generates.
    fn apply_message_delta(&mut self, ev: MessageDeltaEvent) {
        self.stop_reason = ev.delta.stop_reason;
        let Some(usage) = ev.usage else {
            return;
        };
        if let Some(output_tokens) = usage.output_tokens {
            self.completion_tokens = Some(output_tokens);
        }
        if let Some(cache_write_tokens) = usage.cache_write_tokens() {
            self.cache_write_tokens = Some(cache_write_tokens);
        }
        if let Some(cache_read_tokens) = usage.cache_read_input_tokens {
            self.cache_read_tokens = Some(cache_read_tokens);
        }
        self.refresh_prompt_tokens();
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_event_includes_claude_cache_token_stats() {
        let mut state = StreamState::new();
        state.prompt_tokens = Some(120);
        state.completion_tokens = Some(30);
        state.cache_read_tokens = Some(90);
        state.cache_write_tokens = Some(45);
        state.stop_reason = Some("end_turn".to_string());

        let event = state
            .maybe_usage_event()
            .expect("usage event should be emitted");
        match event {
            LLMEvent::Usage(usage) => {
                assert_eq!(usage.prompt_tokens, Some(120));
                assert_eq!(usage.completion_tokens, Some(30));
                assert_eq!(usage.total_tokens, Some(150));
                assert_eq!(usage.cache_read_tokens, Some(90));
                assert_eq!(usage.cache_write_tokens, Some(45));
                assert_eq!(usage.stop_reason.as_deref(), Some("end_turn"));
            }
            _ => panic!("expected usage event"),
        }
    }

    #[test]
    fn usage_event_is_not_reemitted() {
        let mut state = StreamState::new();
        state.prompt_tokens = Some(1);
        assert!(state.maybe_usage_event().is_some());
        assert!(state.maybe_usage_event().is_none());
    }

    #[test]
    fn cache_creation_total_sums_5m_and_1h_tokens() {
        let cache_creation = CacheCreation {
            ephemeral_5m_input_tokens: Some(20),
            ephemeral_1h_input_tokens: Some(30),
        };
        assert_eq!(cache_creation.total(), Some(50));
    }

    #[test]
    fn refresh_prompt_tokens_adds_uncached_and_cache_tokens() {
        let mut state = StreamState::new();
        state.uncached_input_tokens = Some(40);
        state.cache_read_tokens = Some(30);
        state.cache_write_tokens = Some(10);
        state.refresh_prompt_tokens();
        assert_eq!(state.prompt_tokens, Some(80));
    }
}
