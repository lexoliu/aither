// These types mirror the shape of the provider's wire format. Some fields are
// declared so serde can parse a response faithfully even where this crate does
// not consume them; dropping them would change how the payload deserializes.
#![allow(dead_code)]

use serde::Deserialize;
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
    ResponseCompleted { response: ResponsesStreamResponse },
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
        item_id: Option<String>,
        #[serde(default)]
        output_index: usize,
    },
    /// Output item done
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        item: ResponsesOutputItem,
        #[serde(default)]
        item_id: Option<String>,
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
    ReasoningSummaryTextDelta { delta: String },
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
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
    #[serde(default)]
    pub input_token_details: Option<ResponsesInputTokenDetails>,
    #[serde(default)]
    pub output_token_details: Option<ResponsesOutputTokenDetails>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ResponsesInputTokenDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ResponsesOutputTokenDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
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
        /// Encrypted chain of thought, present when the request asked for
        /// `reasoning.encrypted_content`. Replayed into the next request's
        /// `input` so reasoning survives a tool-call round trip.
        #[serde(default)]
        encrypted_content: Option<String>,
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
    #[serde(default)]
    pub usage: Option<ChatCompletionUsage>,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ChatCompletionUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens_details: Option<ChatPromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<ChatCompletionTokensDetails>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ChatPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ChatCompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
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

// ============================================================================
// Model Info (for fetching context window)
// ============================================================================

/// Response from GET /v1/models endpoint (`OpenRouter` format).
#[derive(Debug, Deserialize)]
pub struct ModelsListResponse {
    pub data: Vec<ModelInfo>,
}

/// Individual model info from /v1/models.
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Context window size (`OpenRouter` returns this)
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Some providers use `max_tokens` instead
    #[serde(default)]
    pub max_tokens: Option<u32>,
}
