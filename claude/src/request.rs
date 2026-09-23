//! Request building and message conversion for the Claude API.

use aither_core::llm::{
    Attachment, Message, Role,
    model::{
        ClaudeCacheBreakpointTarget, ClaudeExplicitCacheBreakpoints, ClaudeNativeTools,
        ClaudePromptCache, ClaudePromptCacheStrategy, ClaudePromptCacheTtl, Parameters,
        ReasoningEffort, ToolChoice,
    },
    tool::ToolDefinition,
};
// Only the native file:// reader encodes anything here.
use crate::PROVIDER_NAME;
use crate::error::ClaudeError;
#[cfg(not(target_arch = "wasm32"))]
use base64::Engine;
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
    pub system: Option<SystemPayload>,
    /// Enable streaming.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Sampling temperature.
    ///
    /// # Deprecated by Anthropic
    ///
    /// Anthropic marks `temperature` deprecated. Models newer than the Claude 4
    /// generation accept only `1.0`, for backwards compatibility; other values
    /// are rejected with a 400. It is still sent when set, because callers
    /// pointing at legacy models legitimately need it — steer newer models with
    /// [`crate::request::MessagesRequest::output_config`] effort instead.
    ///
    /// See <https://platform.claude.com/docs/en/api/messages>.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    ///
    /// # Deprecated by Anthropic
    ///
    /// Deprecated alongside [`Self::temperature`]. Newer models accept only
    /// values `>= 0.99`. Kept for legacy models; see [`Self::temperature`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    ///
    /// # Deprecated by Anthropic
    ///
    /// The most restricted of the three: newer models reject `top_k` outright
    /// rather than accepting a compatibility value. Kept for legacy models; see
    /// [`Self::temperature`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Available tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolPayload>>,
    /// Tool choice policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoicePayload>,
    /// Thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingPayload>,
    /// Output shaping, including reasoning effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfigPayload>,
    /// Prompt cache control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlPayload>,
}

/// Claude thinking configuration.
///
/// Only adaptive thinking is modelled. The manual
/// `{"type": "enabled", "budget_tokens": N}` mode is deprecated on the 4.6
/// generation and returns a 400 on every model after it, so aither does not
/// offer a way to construct it.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingPayload {
    /// Let Claude decide when and how deeply to think, steered by effort.
    Adaptive {
        /// Whether thinking text is returned. Omitted uses the model default,
        /// which is `omitted` on the newest models.
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    /// Turn thinking off.
    ///
    /// Rejected at `xhigh` and `max` effort, and on models whose thinking is
    /// always on.
    Disabled,
}

/// Whether Claude returns its thinking text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    /// Return a summary of the thinking.
    Summarized,
    /// Return thinking blocks with empty text, carrying only the signature.
    Omitted,
}

/// Claude output configuration.
#[derive(Debug, Clone, Serialize)]
pub struct OutputConfigPayload {
    /// How much effort Claude puts into the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'static str>,
}

/// Maps the portable effort ladder onto Claude's `output_config.effort`.
///
/// [`ReasoningEffort::None`] is absent here on purpose: Claude expresses "do not
/// think" as `thinking: {"type": "disabled"}`, not as an effort level, so
/// [`build_thinking`] handles it and this returns `None` for it.
///
/// # Errors
///
/// Returns [`ClaudeError`] for [`ReasoningEffort::Minimal`], which Claude's
/// effort scale does not have. Rounding it up to `low` would silently buy the
/// caller more reasoning than they asked for.
pub fn claude_effort(effort: ReasoningEffort) -> Result<Option<&'static str>, ClaudeError> {
    match effort {
        ReasoningEffort::None => Ok(None),
        ReasoningEffort::Low => Ok(Some("low")),
        ReasoningEffort::Medium => Ok(Some("medium")),
        ReasoningEffort::High => Ok(Some("high")),
        ReasoningEffort::XHigh => Ok(Some("xhigh")),
        ReasoningEffort::Max => Ok(Some("max")),
        ReasoningEffort::Minimal => Err(ClaudeError::Api(
            "Claude effort has no 'minimal' level; the lowest is Low".to_string(),
        )),
    }
}

/// Builds the `thinking` field.
///
/// Returns `None` when the caller expressed no preference, leaving the model's
/// own default in force — which is thinking-on for the Claude 5 generation and
/// thinking-off for 4.6 through 4.8.
#[must_use]
pub const fn build_thinking(params: &ParameterSnapshot) -> Option<ThinkingPayload> {
    match (params.reasoning_effort, params.include_reasoning) {
        (Some(ReasoningEffort::None), _) => Some(ThinkingPayload::Disabled),
        // The caller wants to read the thinking.
        (_, true) => Some(ThinkingPayload::Adaptive {
            display: Some(ThinkingDisplay::Summarized),
        }),
        // An effort was set but the text is unwanted: ask for thinking without
        // it, which also lets the server skip streaming thinking tokens.
        (Some(_), false) => Some(ThinkingPayload::Adaptive {
            display: Some(ThinkingDisplay::Omitted),
        }),
        // No preference at all: leave the model's own default in force.
        (None, false) => None,
    }
}

/// Builds the `output_config` field.
///
/// # Errors
///
/// Returns [`ClaudeError`] when the requested effort has no Claude equivalent.
pub fn build_output_config(
    params: &ParameterSnapshot,
) -> Result<Option<OutputConfigPayload>, ClaudeError> {
    let Some(effort) = params.reasoning_effort else {
        return Ok(None);
    };
    let effort = claude_effort(effort)?;
    Ok(effort.map(|effort| OutputConfigPayload {
        effort: Some(effort),
    }))
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

/// System prompt payload in Claude format.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SystemPayload {
    /// Simple system text.
    Text(String),
    /// System content blocks.
    Blocks(Vec<SystemTextBlock>),
}

/// System text block.
#[derive(Debug, Clone, Serialize)]
pub struct SystemTextBlock {
    /// Claude text block kind.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Block text.
    pub text: String,
    /// Optional explicit cache control for this block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlPayload>,
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
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// Image content block.
    #[serde(rename = "image")]
    Image {
        /// Image source (base64 or URL).
        source: MediaSource,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// PDF document content block.
    #[serde(rename = "document")]
    Document {
        /// PDF source (base64 or URL).
        source: MediaSource,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// Thinking block replayed from a previous turn.
    ///
    /// Anthropic requires these back "complete and unmodified", so the text and
    /// signature are the ones Claude produced, never regenerated.
    #[serde(rename = "thinking")]
    Thinking {
        /// The thinking text, empty when the turn used `display: "omitted"`.
        thinking: String,
        /// The signature Claude issued for this block.
        signature: String,
    },
    /// Safety-redacted thinking block replayed from a previous turn.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        /// Opaque encrypted thinking.
        data: String,
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
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
    /// Tool result block (in user messages).
    #[serde(rename = "tool_result")]
    ToolResult {
        /// ID of the `tool_use` this is responding to.
        tool_use_id: String,
        /// Tool output content.
        content: String,
        /// Optional explicit cache control for this block.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlPayload>,
    },
}

/// Image source for vision requests.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum MediaSource {
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
#[serde(untagged)]
pub enum ToolPayload {
    /// User-defined tool.
    Custom(CustomToolPayload),
    /// Anthropic-defined tool.
    Native(NativeToolPayload),
}

/// Custom tool definition in Claude format.
#[derive(Debug, Clone, Serialize)]
pub struct CustomToolPayload {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON schema for tool input.
    pub input_schema: Value,
    /// Optional explicit cache control for this tool block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControlPayload>,
}

/// Anthropic-defined tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct NativeToolPayload {
    /// Tool type.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Tool name.
    pub name: &'static str,
    /// Optional maximum character count for text editor views.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_characters: Option<u32>,
}

/// Claude prompt cache control payload.
#[derive(Debug, Clone, Serialize)]
pub struct CacheControlPayload {
    /// Cache type.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Optional TTL override (`1h`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<&'static str>,
}

impl From<ClaudePromptCache> for CacheControlPayload {
    fn from(cache: ClaudePromptCache) -> Self {
        let ttl = match cache.ttl {
            ClaudePromptCacheTtl::FiveMinutes => None,
            ClaudePromptCacheTtl::OneHour => Some(ClaudePromptCacheTtl::OneHour.as_str()),
        };
        Self {
            kind: "ephemeral",
            ttl,
        }
    }
}

/// Tool choice payload for Claude Messages API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoicePayload {
    /// Let Claude decide when to call a tool.
    Auto {
        /// Forbids Claude from requesting more than one tool per turn.
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Require Claude to call at least one tool.
    Any {
        /// Forbids Claude from requesting more than one tool per turn.
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Restrict tool calling to a single tool by name.
    Tool {
        /// Name of the tool Claude is allowed to call.
        name: String,
        /// Forbids Claude from requesting more than one tool per turn.
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Forbid tool calls outright.
    None,
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
    /// Whether Claude may request several tools per turn. `None` keeps Anthropic's default.
    pub parallel_tool_calls: Option<bool>,
    /// Requested reasoning effort.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Tool choice policy.
    pub tool_choice: ToolChoice,
    /// Claude-specific cache controls.
    pub cache: Option<ClaudePromptCache>,
    /// Claude-native tools.
    pub native_tools: ClaudeNativeTools,
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
            parallel_tool_calls: params.parallel_tool_calls,
            reasoning_effort: params.reasoning_effort,
            tool_choice: params.tool_choice.clone(),
            cache: params.cache.claude,
            native_tools: params.native_tools.claude.clone(),
        }
    }
}

/// Convert aither messages to Claude format, extracting system messages.
///
/// Returns (`system_prompt`, messages). A single system message is emitted as
/// plain text, while multiple system messages are preserved as system blocks.
pub async fn to_claude_messages(
    messages: &[Message],
) -> Result<(Option<SystemPayload>, Vec<MessagePayload>), String> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut claude_messages: Vec<MessagePayload> = Vec::new();

    for message in messages {
        match message.role() {
            Role::System => {
                system_parts.push(message.content().to_string());
            }
            Role::User | Role::Tool => {
                let content = if matches!(message.role(), Role::Tool) {
                    build_tool_result_content(message)
                } else {
                    build_user_content(message).await?
                };
                claude_messages.push(MessagePayload {
                    role: "user",
                    content,
                });
            }
            Role::Assistant => {
                claude_messages.push(MessagePayload {
                    role: "assistant",
                    content: build_assistant_content(message),
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else if system_parts.len() == 1 {
        Some(SystemPayload::Text(system_parts.remove(0)))
    } else {
        Some(SystemPayload::Blocks(
            system_parts
                .into_iter()
                .map(|text| SystemTextBlock {
                    kind: "text",
                    text,
                    cache_control: None,
                })
                .collect(),
        ))
    };

    Ok((system, claude_messages))
}

/// Build content for a user message, handling image and PDF attachments.
async fn build_user_content(message: &Message) -> Result<ContentPayload, String> {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return Ok(ContentPayload::Text(flatten_content(message)));
    }

    let mut blocks: Vec<ContentBlock> = Vec::with_capacity(attachments.len() + 1);
    for attachment in attachments {
        let source = parse_media_source(attachment).await?;
        let media_type = attachment.media_type().as_ref();
        if media_type.starts_with("image/") {
            blocks.push(ContentBlock::Image {
                source,
                cache_control: None,
            });
        } else if media_type == "application/pdf" {
            blocks.push(ContentBlock::Document {
                source,
                cache_control: None,
            });
        } else {
            return Err(format!(
                "Claude Messages API does not support attachment MIME type '{media_type}'"
            ));
        }
    }

    let text = flatten_content(message);
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text,
            cache_control: None,
        });
    }

    if blocks.len() == 1
        && let Some(ContentBlock::Text {
            text,
            cache_control: None,
        }) = blocks.pop()
    {
        return Ok(ContentPayload::Text(text));
    }

    Ok(ContentPayload::Blocks(blocks))
}

/// Decodes reasoning this crate previously emitted back into content blocks.
///
/// The payload is this crate's own encoding, so decoding it here keeps core
/// ignorant of Claude's block taxonomy while still round-tripping both
/// `thinking` and `redacted_thinking`. State from another provider, and state
/// that no longer parses, is dropped: Anthropic documents stripping foreign
/// thinking rather than rejecting the turn.
fn replayed_thinking_blocks(message: &Message) -> Vec<ContentBlock> {
    #[derive(serde::Deserialize)]
    #[serde(tag = "type")]
    enum Replayed {
        #[serde(rename = "thinking")]
        Thinking { thinking: String, signature: String },
        #[serde(rename = "redacted_thinking")]
        RedactedThinking { data: String },
    }

    message
        .reasoning()
        .iter()
        .filter_map(|state| {
            let payload = state.payload_for(PROVIDER_NAME).or_else(|| {
                tracing::debug!(
                    provider = state.provider(),
                    "dropping reasoning state from another provider"
                );
                None
            })?;
            match serde_json::from_str::<Replayed>(payload) {
                Ok(Replayed::Thinking {
                    thinking,
                    signature,
                }) => Some(ContentBlock::Thinking {
                    thinking,
                    signature,
                }),
                Ok(Replayed::RedactedThinking { data }) => {
                    Some(ContentBlock::RedactedThinking { data })
                }
                Err(error) => {
                    tracing::debug!(%error, "dropping unparsable Claude reasoning state");
                    None
                }
            }
        })
        .collect()
}

fn build_assistant_content(message: &Message) -> ContentPayload {
    let tool_calls = message.tool_calls();
    // Thinking must lead the turn, so its presence alone forces block form even
    // when there are no tool calls to report.
    let thinking = replayed_thinking_blocks(message);
    if tool_calls.is_empty() && thinking.is_empty() {
        return ContentPayload::Text(flatten_content(message));
    }

    let mut blocks = thinking;
    let text = flatten_content(message);
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text,
            cache_control: None,
        });
    }

    for call in tool_calls {
        blocks.push(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
            cache_control: None,
        });
    }

    ContentPayload::Blocks(blocks)
}

fn build_tool_result_content(message: &Message) -> ContentPayload {
    let tool_use_id = message.tool_call_id().unwrap_or_else(|| {
        panic!("Tool message missing tool_call_id required by Claude tool_result payload")
    });
    let content = flatten_content(message);
    ContentPayload::Blocks(vec![ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        content,
        cache_control: None,
    }])
}

/// Apply Claude cache strategy to request payload sections.
///
/// Returns top-level `cache_control` when automatic caching is selected.
/// Explicit mode mutates blocks in-place and returns `None`.
pub fn apply_cache_strategy(
    system: &mut Option<SystemPayload>,
    messages: &mut [MessagePayload],
    tools: &mut Option<Vec<ToolPayload>>,
    cache: ClaudePromptCache,
) -> Result<Option<CacheControlPayload>, String> {
    let default_ttl = cache.ttl;
    match cache.strategy {
        ClaudePromptCacheStrategy::Automatic => {
            let automatic = maybe_automatic_cache_control(
                system.as_ref(),
                messages,
                tools.as_ref(),
                default_ttl,
                None,
            )?;
            Ok(automatic.map(|(cache_control, _)| cache_control))
        }
        ClaudePromptCacheStrategy::Explicit(breakpoints) => {
            let applied =
                apply_explicit_breakpoints(system, messages, tools, breakpoints, default_ttl)?;
            validate_mixed_ttl_ordering(&applied)?;
            Ok(None)
        }
        ClaudePromptCacheStrategy::AutomaticAndExplicit(breakpoints) => {
            let mut applied =
                apply_explicit_breakpoints(system, messages, tools, breakpoints, default_ttl)?;
            let automatic = maybe_automatic_cache_control(
                system.as_ref(),
                messages,
                tools.as_ref(),
                default_ttl,
                Some(breakpoints),
            )?;
            if let Some((cache_control, automatic_breakpoint)) = automatic {
                applied.push(automatic_breakpoint);
                validate_mixed_ttl_ordering(&applied)?;
                return Ok(Some(cache_control));
            }
            validate_mixed_ttl_ordering(&applied)?;
            Ok(None)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PromptPosition {
    section: u8,
    major: usize,
    minor: usize,
}

impl PromptPosition {
    const fn tool(index: usize) -> Self {
        Self {
            section: 0,
            major: index,
            minor: 0,
        }
    }

    const fn system(index: usize) -> Self {
        Self {
            section: 1,
            major: index,
            minor: 0,
        }
    }

    const fn message(message_index: usize, block_index: usize) -> Self {
        Self {
            section: 2,
            major: message_index,
            minor: block_index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AppliedBreakpoint {
    position: PromptPosition,
    ttl: ClaudePromptCacheTtl,
}

fn apply_explicit_breakpoints(
    system: &mut Option<SystemPayload>,
    messages: &mut [MessagePayload],
    tools: &mut Option<Vec<ToolPayload>>,
    breakpoints: ClaudeExplicitCacheBreakpoints,
    default_ttl: ClaudePromptCacheTtl,
) -> Result<Vec<AppliedBreakpoint>, String> {
    let mut applied = Vec::new();
    for breakpoint in breakpoints.iter() {
        let ttl = breakpoint.effective_ttl(default_ttl);
        let cache_control = cache_control_payload_for_ttl(ttl);
        let position =
            mark_target_cache_control(system, messages, tools, breakpoint.target, &cache_control)?;
        applied.push(AppliedBreakpoint { position, ttl });
    }
    Ok(applied)
}

#[allow(clippy::ref_option)]
fn maybe_automatic_cache_control(
    system: Option<&SystemPayload>,
    messages: &[MessagePayload],
    tools: Option<&Vec<ToolPayload>>,
    ttl: ClaudePromptCacheTtl,
    explicit_breakpoints: Option<ClaudeExplicitCacheBreakpoints>,
) -> Result<Option<(CacheControlPayload, AppliedBreakpoint)>, String> {
    let Some((position, existing_control)) = find_last_cacheable_block(system, messages, tools)
    else {
        // No cacheable block exists: skip automatic caching as Anthropic does.
        return Ok(None);
    };

    if let Some(existing_control) = existing_control {
        if cache_control_matches_ttl(&existing_control, ttl) {
            // Anthropic behavior: automatic is a no-op when last block already has same TTL.
            return Ok(None);
        }
        return Err(
            "Claude automatic cache_control conflicts with explicit cache_control on the last cacheable block".to_string(),
        );
    }

    if let Some(explicit_breakpoints) = explicit_breakpoints
        && explicit_breakpoints.is_full()
    {
        return Err(
            "Claude supports at most 4 cache breakpoints; automatic caching needs one free slot"
                .to_string(),
        );
    }

    let cache_control = cache_control_payload_for_ttl(ttl);
    Ok(Some((cache_control, AppliedBreakpoint { position, ttl })))
}

#[allow(clippy::ref_option)]
fn find_last_cacheable_block(
    system: Option<&SystemPayload>,
    messages: &[MessagePayload],
    tools: Option<&Vec<ToolPayload>>,
) -> Option<(PromptPosition, Option<CacheControlPayload>)> {
    for (message_index, message) in messages.iter().enumerate().rev() {
        match &message.content {
            ContentPayload::Text(text) => {
                if !text.is_empty() {
                    return Some((PromptPosition::message(message_index, 0), None));
                }
            }
            ContentPayload::Blocks(blocks) => {
                if let Some(block_index) = last_cacheable_block_index(blocks) {
                    return Some((
                        PromptPosition::message(message_index, block_index),
                        block_cache_control(&blocks[block_index]).cloned(),
                    ));
                }
            }
        }
    }

    if let Some(system) = system {
        match system {
            SystemPayload::Text(text) => {
                if !text.is_empty() {
                    return Some((PromptPosition::system(0), None));
                }
            }
            SystemPayload::Blocks(blocks) => {
                if let Some(system_index) = blocks.iter().rposition(|block| !block.text.is_empty())
                {
                    return Some((
                        PromptPosition::system(system_index),
                        blocks[system_index].cache_control.clone(),
                    ));
                }
            }
        }
    }

    if let Some(tools) = tools
        && let Some((tool_index, tool)) = tools
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, tool)| custom_tool(tool).map(|custom| (index, custom)))
    {
        return Some((PromptPosition::tool(tool_index), tool.cache_control.clone()));
    }

    None
}

const fn cache_control_payload_for_ttl(ttl: ClaudePromptCacheTtl) -> CacheControlPayload {
    CacheControlPayload {
        kind: "ephemeral",
        ttl: match ttl {
            ClaudePromptCacheTtl::FiveMinutes => None,
            ClaudePromptCacheTtl::OneHour => Some(ClaudePromptCacheTtl::OneHour.as_str()),
        },
    }
}

fn cache_control_matches_ttl(
    cache_control: &CacheControlPayload,
    ttl: ClaudePromptCacheTtl,
) -> bool {
    let expected = cache_control_payload_for_ttl(ttl);
    cache_control.kind == expected.kind && cache_control.ttl == expected.ttl
}

fn validate_mixed_ttl_ordering(applied: &[AppliedBreakpoint]) -> Result<(), String> {
    if applied.is_empty() {
        return Ok(());
    }
    let mut ordered = applied.to_vec();
    ordered.sort_by_key(|breakpoint| breakpoint.position);
    let mut seen_five_minute = false;
    for breakpoint in ordered {
        match breakpoint.ttl {
            ClaudePromptCacheTtl::OneHour => {
                if seen_five_minute {
                    return Err(
                        "Claude mixed TTL ordering requires 1h cache breakpoints to appear before 5m breakpoints".to_string(),
                    );
                }
            }
            ClaudePromptCacheTtl::FiveMinutes => {
                seen_five_minute = true;
            }
        }
    }
    Ok(())
}

fn mark_target_cache_control(
    system: &mut Option<SystemPayload>,
    messages: &mut [MessagePayload],
    tools: &mut Option<Vec<ToolPayload>>,
    target: ClaudeCacheBreakpointTarget,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    match target {
        ClaudeCacheBreakpointTarget::LastTool => mark_last_tool_cache_control(tools, cache_control),
        ClaudeCacheBreakpointTarget::Tool(index) => {
            mark_tool_cache_control(tools, index, cache_control)
        }
        ClaudeCacheBreakpointTarget::LastSystem => {
            mark_last_system_cache_control(system, cache_control)
        }
        ClaudeCacheBreakpointTarget::System(index) => {
            mark_system_cache_control(system, index, cache_control)
        }
        ClaudeCacheBreakpointTarget::LastMessage => {
            mark_last_message_cache_control(messages, cache_control)
        }
        ClaudeCacheBreakpointTarget::Message {
            message_index,
            block_index,
        } => mark_message_cache_control(messages, message_index, block_index, cache_control),
    }
}

fn mark_last_tool_cache_control(
    tools: &mut Option<Vec<ToolPayload>>,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let tools = tools.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for tools but tools are empty".to_string()
    })?;
    let (index, tool) = tools
        .iter_mut()
        .enumerate()
        .rev()
        .find_map(|(index, tool)| custom_tool_mut(tool).map(|custom| (index, custom)))
        .ok_or_else(|| {
            "Claude explicit cache breakpoint requested for tools but no custom tool exists"
                .to_string()
        })?;
    set_cache_control(
        &mut tool.cache_control,
        cache_control,
        "tool breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::tool(index))
}

fn mark_tool_cache_control(
    tools: &mut Option<Vec<ToolPayload>>,
    index: usize,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let tools = tools.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for tools but tools are empty".to_string()
    })?;
    let tool = tools
        .get_mut(index)
        .ok_or_else(|| format!("Claude explicit tool breakpoint index {index} is out of range"))?;
    let tool = custom_tool_mut(tool).ok_or_else(|| {
        format!("Claude explicit tool breakpoint index {index} targets a native tool")
    })?;
    set_cache_control(
        &mut tool.cache_control,
        cache_control,
        "tool breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::tool(index))
}

fn mark_last_system_cache_control(
    system: &mut Option<SystemPayload>,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let system_payload = system.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for system but no system prompt exists"
            .to_string()
    })?;
    let blocks = ensure_system_blocks(system_payload);
    let index = blocks
        .iter()
        .rposition(|block| !block.text.is_empty())
        .ok_or_else(|| {
            "Claude explicit cache breakpoint requested for system but all system text blocks are empty"
                .to_string()
        })?;
    set_cache_control(
        &mut blocks[index].cache_control,
        cache_control,
        "system breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::system(index))
}

fn mark_system_cache_control(
    system: &mut Option<SystemPayload>,
    index: usize,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let system_payload = system.as_mut().ok_or_else(|| {
        "Claude explicit cache breakpoint requested for system but no system prompt exists"
            .to_string()
    })?;
    let blocks = ensure_system_blocks(system_payload);
    let block = blocks.get_mut(index).ok_or_else(|| {
        format!("Claude explicit system breakpoint index {index} is out of range")
    })?;
    if block.text.is_empty() {
        return Err(format!(
            "Claude explicit system breakpoint index {index} targets an empty text block"
        ));
    }
    set_cache_control(
        &mut block.cache_control,
        cache_control,
        "system breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::system(index))
}

fn ensure_system_blocks(system: &mut SystemPayload) -> &mut Vec<SystemTextBlock> {
    if let SystemPayload::Text(text) = system {
        *system = SystemPayload::Blocks(vec![SystemTextBlock {
            kind: "text",
            text: core::mem::take(text),
            cache_control: None,
        }]);
    }
    match system {
        SystemPayload::Blocks(blocks) => blocks,
        SystemPayload::Text(_) => panic!("System payload conversion to blocks failed"),
    }
}

fn mark_last_message_cache_control(
    messages: &mut [MessagePayload],
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    for (message_index, message) in messages.iter_mut().enumerate().rev() {
        match &mut message.content {
            ContentPayload::Text(text) => {
                if text.is_empty() {
                    continue;
                }
                let text = core::mem::take(text);
                message.content = ContentPayload::Blocks(vec![ContentBlock::Text {
                    text,
                    cache_control: Some(cache_control.clone()),
                }]);
                return Ok(PromptPosition::message(message_index, 0));
            }
            ContentPayload::Blocks(blocks) => {
                // last_cacheable_block_index already excluded thinking blocks,
                // so this slot is always present.
                if let Some(index) = last_cacheable_block_index(blocks)
                    && let Some(slot) = block_cache_control_mut(&mut blocks[index])
                {
                    set_cache_control(
                        slot,
                        cache_control,
                        "message breakpoint already has conflicting cache_control",
                    )?;
                    return Ok(PromptPosition::message(message_index, index));
                }
            }
        }
    }
    Err("Claude explicit cache breakpoint requested for messages but no cacheable message block exists".to_string())
}

fn mark_message_cache_control(
    messages: &mut [MessagePayload],
    message_index: usize,
    block_index: usize,
    cache_control: &CacheControlPayload,
) -> Result<PromptPosition, String> {
    let message = messages.get_mut(message_index).ok_or_else(|| {
        format!("Claude explicit message breakpoint message_index {message_index} is out of range")
    })?;
    if let ContentPayload::Text(text) = &mut message.content {
        if block_index != 0 {
            return Err(format!(
                "Claude explicit message breakpoint block_index {block_index} is invalid for text-only message at index {message_index}"
            ));
        }
        if text.is_empty() {
            return Err(format!(
                "Claude explicit message breakpoint targets empty text message at index {message_index}"
            ));
        }
        let text = core::mem::take(text);
        message.content = ContentPayload::Blocks(vec![ContentBlock::Text {
            text,
            cache_control: None,
        }]);
    }

    let blocks = match &mut message.content {
        ContentPayload::Blocks(blocks) => blocks,
        ContentPayload::Text(_) => panic!("message content conversion to blocks failed"),
    };
    let block = blocks.get_mut(block_index).ok_or_else(|| {
        format!(
            "Claude explicit message breakpoint block_index {block_index} is out of range for message index {message_index}"
        )
    })?;
    if !is_cacheable_block(block) {
        return Err(format!(
            "Claude explicit message breakpoint targets a non-cacheable block at message index {message_index}, block index {block_index}"
        ));
    }
    let slot = block_cache_control_mut(block).ok_or_else(|| {
        format!(
            "Claude explicit message breakpoint targets a thinking block at message index {message_index}, block index {block_index}, which cannot carry cache_control"
        )
    })?;
    set_cache_control(
        slot,
        cache_control,
        "message breakpoint already has conflicting cache_control",
    )?;
    Ok(PromptPosition::message(message_index, block_index))
}

fn last_cacheable_block_index(blocks: &[ContentBlock]) -> Option<usize> {
    blocks.iter().rposition(|block| match block {
        ContentBlock::Text { text, .. } => !text.is_empty(),
        ContentBlock::Image { .. }
        | ContentBlock::Document { .. }
        | ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. } => true,
        // Anthropic defines no cache_control on thinking blocks.
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => false,
    })
}

const fn is_cacheable_block(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Text { text, .. } => !text.is_empty(),
        ContentBlock::Image { .. }
        | ContentBlock::Document { .. }
        | ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. } => true,
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => false,
    }
}

/// Returns `None` for blocks that have no `cache_control` field at all.
///
/// Callers reach this only after [`is_cacheable_block`], so a `None` here means
/// a caller skipped that check rather than a user error.
const fn block_cache_control_mut(
    block: &mut ContentBlock,
) -> Option<&mut Option<CacheControlPayload>> {
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::Document { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => Some(cache_control),
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => None,
    }
}

const fn block_cache_control(block: &ContentBlock) -> Option<&CacheControlPayload> {
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::Document { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => cache_control.as_ref(),
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => None,
    }
}

fn set_cache_control(
    slot: &mut Option<CacheControlPayload>,
    expected: &CacheControlPayload,
    mismatch_error: &str,
) -> Result<(), String> {
    if let Some(current) = slot {
        if current.kind != expected.kind || current.ttl != expected.ttl {
            return Err(mismatch_error.to_string());
        }
        return Ok(());
    }
    *slot = Some(expected.clone());
    Ok(())
}

/// Converts a typed attachment into a Claude media source.
async fn parse_media_source(attachment: &Attachment) -> Result<MediaSource, String> {
    let url = attachment.url();
    let media_type = attachment.media_type().as_ref();
    match url.scheme() {
        "data" => {
            let after_data = url
                .as_str()
                .strip_prefix("data:")
                .ok_or_else(|| "attachment data URL is malformed".to_string())?;
            let (header, data) = after_data
                .split_once(',')
                .ok_or_else(|| "attachment data URL is missing its payload".to_string())?;
            let encoded_media_type = header
                .strip_suffix(";base64")
                .ok_or_else(|| "attachment data URL must use base64 encoding".to_string())?;
            if encoded_media_type != media_type {
                return Err(format!(
                    "attachment MIME type '{media_type}' does not match data URL MIME type '{encoded_media_type}'"
                ));
            }
            Ok(MediaSource::Base64 {
                media_type: media_type.to_string(),
                data: data.to_string(),
            })
        }
        // wasm32 has no filesystem, and CI builds this crate for it with -D warnings.
        #[cfg(target_arch = "wasm32")]
        "file" => {
            Err("Claude attachments from file:// URLs are not supported on wasm32".to_string())
        }
        #[cfg(not(target_arch = "wasm32"))]
        "file" => {
            let path = url
                .to_file_path()
                .map_err(|()| "attachment file URL could not be converted to a path".to_string())?;
            let data = async_fs::read(&path).await.map_err(|error| {
                format!("failed to read attachment '{}': {error}", path.display())
            })?;
            Ok(MediaSource::Base64 {
                media_type: media_type.to_string(),
                data: base64::engine::general_purpose::STANDARD.encode(data),
            })
        }
        "http" | "https" => Ok(MediaSource::Url {
            url: url.as_str().to_string(),
        }),
        scheme => Err(format!(
            "Claude does not support attachment URL scheme '{scheme}'"
        )),
    }
}

/// Flatten message content.
fn flatten_content(message: &Message) -> String {
    message.content().to_owned()
}

const fn custom_tool(tool: &ToolPayload) -> Option<&CustomToolPayload> {
    match tool {
        ToolPayload::Custom(tool) => Some(tool),
        ToolPayload::Native(_) => None,
    }
}

const fn custom_tool_mut(tool: &mut ToolPayload) -> Option<&mut CustomToolPayload> {
    match tool {
        ToolPayload::Custom(tool) => Some(tool),
        ToolPayload::Native(_) => None,
    }
}

/// Convert aither tool definitions to Claude format.
pub fn convert_tools(definitions: &[ToolDefinition]) -> Vec<ToolPayload> {
    definitions
        .iter()
        .map(|tool| {
            ToolPayload::Custom(CustomToolPayload {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.arguments_openai_schema(),
                cache_control: None,
            })
        })
        .collect()
}

pub fn convert_native_tools(tools: &ClaudeNativeTools) -> Vec<ToolPayload> {
    let mut payload = Vec::new();
    if tools.web_search {
        payload.push(ToolPayload::Native(NativeToolPayload {
            kind: "web_search_20260209",
            name: "web_search",
            max_characters: None,
        }));
    }
    if tools.web_fetch {
        payload.push(ToolPayload::Native(NativeToolPayload {
            kind: "web_fetch_20260209",
            name: "web_fetch",
            max_characters: None,
        }));
    }
    if tools.code_execution {
        payload.push(ToolPayload::Native(NativeToolPayload {
            kind: "code_execution_20260120",
            name: "code_execution",
            max_characters: None,
        }));
    }
    if tools.bash {
        payload.push(ToolPayload::Native(NativeToolPayload {
            kind: "bash_20250124",
            name: "bash",
            max_characters: None,
        }));
    }
    if let Some(text_editor) = tools.text_editor {
        payload.push(ToolPayload::Native(NativeToolPayload {
            kind: "text_editor_20250728",
            name: "str_replace_based_edit_tool",
            max_characters: text_editor.max_characters,
        }));
    }
    payload
}

pub fn filter_tool_definitions(
    definitions: Vec<ToolDefinition>,
    choice: &ToolChoice,
) -> Vec<ToolDefinition> {
    match choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => definitions
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => definitions,
    }
}

pub fn tool_choice_payload(
    choice: &ToolChoice,
    has_tools: bool,
    parallel_tool_calls: Option<bool>,
) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    // Anthropic states the negative: the wire flag disables parallelism, while
    // the portable parameter enables it.
    let disable_parallel_tool_use = parallel_tool_calls.map(|enabled| !enabled);
    match choice {
        ToolChoice::None => Some(ToolChoicePayload::None),
        ToolChoice::Auto => Some(ToolChoicePayload::Auto {
            disable_parallel_tool_use,
        }),
        ToolChoice::Required => Some(ToolChoicePayload::Any {
            disable_parallel_tool_use,
        }),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Tool {
            name: name.clone(),
            disable_parallel_tool_use,
        }),
    }
}

#[cfg(test)]
#[allow(clippy::match_wildcard_for_single_variants)]
mod tests {
    use super::*;
    use aither_core::llm::{
        Attachment, ReasoningState, ToolCall,
        model::{
            ClaudeCacheBreakpointTarget, ClaudeExplicitCacheBreakpoint,
            ClaudeExplicitCacheBreakpoints, ClaudePromptCache, ClaudePromptCacheStrategy,
            ClaudePromptCacheTtl, Parameters, ToolChoice,
        },
    };
    #[tokio::test]
    async fn assistant_tool_calls_are_encoded_as_tool_use_blocks() {
        let messages = vec![Message::assistant_with_tool_calls(
            "Working on it",
            vec![ToolCall {
                reasoning_state: None,
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                arguments: serde_json::json!({"q":"rust"}),
            }],
        )];
        let (_, encoded) = to_claude_messages(&messages)
            .await
            .expect("encode Claude messages");
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0].role, "assistant");
        match &encoded[0].content {
            ContentPayload::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(blocks[0], ContentBlock::Text { .. }));
                assert!(matches!(blocks[1], ContentBlock::ToolUse { .. }));
            }
            other @ ContentPayload::Text(_) => {
                panic!("expected assistant blocks payload, got: {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn tool_message_is_encoded_as_tool_result_block() {
        let messages = vec![Message::tool("call_9", "{\"ok\":true}")];
        let (_, encoded) = to_claude_messages(&messages)
            .await
            .expect("encode Claude messages");
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0].role, "user");
        match &encoded[0].content {
            ContentPayload::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        assert_eq!(tool_use_id, "call_9");
                        assert_eq!(content, "{\"ok\":true}");
                    }
                    other => panic!("expected tool_result block, got: {other:?}"),
                }
            }
            other @ ContentPayload::Text(_) => {
                panic!("expected user blocks payload, got: {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn image_and_pdf_attachments_use_distinct_blocks() {
        let image = Attachment::new(
            "data:image/png;base64,AA==".parse().expect("image URL"),
            "image/png".parse().expect("image MIME"),
        );
        let pdf = Attachment::new(
            "https://platform.claude.com/docs/en/build-with-claude/pdf-support/sample.pdf"
                .parse()
                .expect("PDF URL"),
            "application/pdf".parse().expect("PDF MIME"),
        );
        let messages = vec![Message::user("inspect").with_attachments([image, pdf])];
        let (_, encoded) = to_claude_messages(&messages)
            .await
            .expect("encode Claude attachments");
        let ContentPayload::Blocks(blocks) = &encoded[0].content else {
            panic!("expected Claude content blocks");
        };
        assert!(matches!(blocks[0], ContentBlock::Image { .. }));
        assert!(matches!(blocks[1], ContentBlock::Document { .. }));
        assert!(matches!(blocks[2], ContentBlock::Text { .. }));
    }

    #[tokio::test]
    async fn audio_attachment_fails_instead_of_becoming_an_image() {
        let audio = Attachment::new(
            "data:audio/wav;base64,AA==".parse().expect("audio URL"),
            "audio/wav".parse().expect("audio MIME"),
        );
        let messages = vec![Message::user("listen").with_attachment(audio)];
        let error = to_claude_messages(&messages)
            .await
            .expect_err("Claude audio input must fail");
        assert!(error.contains("does not support attachment MIME type"));
    }

    #[test]
    fn required_tool_choice_maps_to_any() {
        let payload = tool_choice_payload(&ToolChoice::Required, true, None)
            .expect("required should create payload");
        let json = serde_json::to_value(payload).expect("serialize tool choice");
        assert_eq!(json["type"], "any");
    }

    #[test]
    fn exact_tool_choice_maps_to_named_tool() {
        let payload = tool_choice_payload(&ToolChoice::Exact("search".to_string()), true, None)
            .expect("exact should create payload");
        let json = serde_json::to_value(payload).expect("serialize tool choice");
        assert_eq!(json["type"], "tool");
        assert_eq!(json["name"], "search");
    }

    fn tool_choice_json(choice: &ToolChoice, parallel: Option<bool>) -> serde_json::Value {
        let payload = tool_choice_payload(choice, true, parallel).expect("tool choice payload");
        serde_json::to_value(payload).expect("serialize tool choice")
    }

    /// Anthropic states the flag negatively, so the portable "enable parallel"
    /// parameter has to invert on its way to the wire.
    #[test]
    fn parallel_tool_calls_inverts_into_disable_parallel_tool_use() {
        let serialized = tool_choice_json(&ToolChoice::Auto, Some(false));
        assert_eq!(serialized["disable_parallel_tool_use"], true);

        let parallel = tool_choice_json(&ToolChoice::Required, Some(true));
        assert_eq!(parallel["disable_parallel_tool_use"], false);
    }

    /// Unset must stay off the wire entirely so Anthropic's own default applies.
    #[test]
    fn unset_parallel_tool_calls_is_omitted() {
        for choice in [
            ToolChoice::Auto,
            ToolChoice::Required,
            ToolChoice::Exact("search".to_string()),
        ] {
            let serialized = tool_choice_json(&choice, None);
            assert_eq!(
                serialized.get("disable_parallel_tool_use"),
                None,
                "{choice:?} leaked a parallelism hint"
            );
        }
    }

    #[test]
    fn exact_tool_choice_carries_the_parallelism_hint() {
        let serialized = tool_choice_json(&ToolChoice::Exact("search".to_string()), Some(false));
        assert_eq!(serialized["type"], "tool");
        assert_eq!(serialized["name"], "search");
        assert_eq!(serialized["disable_parallel_tool_use"], true);
    }

    /// Anthropic expresses "do not call tools" as a choice, not as an absent
    /// tool list, and that variant takes no parallelism hint.
    #[test]
    fn none_tool_choice_maps_to_the_none_variant() {
        let serialized = tool_choice_json(&ToolChoice::None, Some(false));
        assert_eq!(serialized["type"], "none");
        assert_eq!(serialized.get("disable_parallel_tool_use"), None);
    }

    /// The round trip that CLA-2 made impossible: state out of the response
    /// parser must come back as a replayable block, signature intact.
    #[test]
    fn thinking_state_round_trips_into_the_assistant_turn() {
        let state = ReasoningState::new(
            PROVIDER_NAME,
            serde_json::json!({
                "type": "thinking",
                "thinking": "step one",
                "signature": "sig-abc",
            })
            .to_string(),
        );
        let message = Message::assistant_with_reasoning(
            "answer",
            vec![ToolCall::new("call_1", "search", serde_json::json!({}))],
            vec![state],
        );

        let value = serde_json::to_value(build_assistant_content(&message))
            .expect("serialize assistant content");
        let blocks = value.as_array().expect("block form");

        // Anthropic requires the turn to begin with its thinking blocks.
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "step one");
        assert_eq!(blocks[0]["signature"], "sig-abc");
        assert!(blocks.iter().any(|block| block["type"] == "tool_use"));
    }

    #[test]
    fn redacted_thinking_round_trips() {
        let state = ReasoningState::new(
            PROVIDER_NAME,
            serde_json::json!({"type": "redacted_thinking", "data": "cipher"}).to_string(),
        );
        let message = Message::assistant_with_reasoning("", Vec::new(), vec![state]);
        let value = serde_json::to_value(build_assistant_content(&message))
            .expect("serialize assistant content");
        assert_eq!(value[0]["type"], "redacted_thinking");
        assert_eq!(value[0]["data"], "cipher");
    }

    /// Thinking blocks are tied to the model that produced them, so state from
    /// another provider is stripped rather than replayed.
    #[test]
    fn foreign_reasoning_state_is_dropped() {
        let message = Message::assistant_with_reasoning(
            "answer",
            Vec::new(),
            vec![ReasoningState::new(
                "openai",
                serde_json::json!({"type": "reasoning", "id": "rs_1"}).to_string(),
            )],
        );
        assert!(replayed_thinking_blocks(&message).is_empty());
    }

    #[test]
    fn tool_choice_is_absent_without_tools() {
        assert!(tool_choice_payload(&ToolChoice::Auto, false, Some(false)).is_none());
    }

    fn snapshot(params: &Parameters) -> ParameterSnapshot {
        ParameterSnapshot::from(params)
    }

    /// Anthropic's effort ladder, verbatim. `minimal` is absent from it.
    #[test]
    fn effort_maps_onto_the_anthropic_ladder() {
        for (effort, expected) in [
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::XHigh, "xhigh"),
            (ReasoningEffort::Max, "max"),
        ] {
            let config =
                build_output_config(&snapshot(&Parameters::default().reasoning_effort(effort)))
                    .expect("effort supported by Claude")
                    .expect("output config");
            let value = serde_json::to_value(config).expect("serialize output config");
            assert_eq!(value["effort"], expected);
        }
    }

    #[test]
    fn minimal_effort_is_rejected() {
        let error = build_output_config(&snapshot(
            &Parameters::default().reasoning_effort(ReasoningEffort::Minimal),
        ))
        .expect_err("Claude has no minimal effort");
        assert!(error.to_string().contains("no 'minimal' level"));
    }

    /// `None` is a thinking mode on Claude, not an effort level, so it must
    /// produce `thinking: {"type":"disabled"}` and no `output_config`.
    #[test]
    fn none_effort_disables_thinking_rather_than_setting_an_effort() {
        let snap = snapshot(&Parameters::default().reasoning_effort(ReasoningEffort::None));

        let thinking = build_thinking(&snap).expect("thinking payload");
        let value = serde_json::to_value(thinking).expect("serialize thinking");
        assert_eq!(value["type"], "disabled");

        assert!(
            build_output_config(&snap)
                .expect("None is supported")
                .is_none()
        );
    }

    /// Adaptive is already the model's own mode, so the only reason to send a
    /// thinking config is to ask for the text — otherwise stay off the wire.
    #[test]
    fn thinking_is_only_sent_when_the_caller_wants_the_text() {
        assert!(build_thinking(&snapshot(&Parameters::default())).is_none());

        let thinking = build_thinking(&snapshot(&Parameters::default().include_reasoning(true)))
            .expect("thinking payload");
        let value = serde_json::to_value(thinking).expect("serialize thinking");
        assert_eq!(value["type"], "adaptive");
        assert_eq!(value["display"], "summarized");
    }

    /// An effort without a request for the text means "think harder, don't show
    /// me" — which is `display: "omitted"`, and also lets the server skip
    /// streaming thinking tokens.
    #[test]
    fn effort_without_visible_text_asks_for_omitted_display() {
        let thinking = build_thinking(&snapshot(
            &Parameters::default().reasoning_effort(ReasoningEffort::High),
        ))
        .expect("thinking payload");
        let value = serde_json::to_value(thinking).expect("serialize thinking");
        assert_eq!(value["type"], "adaptive");
        assert_eq!(value["display"], "omitted");
    }

    /// The deprecated manual mode must be unrepresentable, not merely unused:
    /// it returns a 400 on every model after the 4.6 generation.
    #[test]
    fn thinking_never_serializes_a_token_budget() {
        let thinking = build_thinking(&snapshot(&Parameters::default().include_reasoning(true)))
            .expect("thinking payload");
        let value = serde_json::to_value(thinking).expect("serialize thinking");
        assert_ne!(value["type"], "enabled");
        assert_eq!(value.get("budget_tokens"), None);
    }

    #[test]
    fn parameter_snapshot_preserves_claude_cache_setting() {
        let params = Parameters::default()
            .claude_prompt_cache(ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour));
        let snapshot = ParameterSnapshot::from(&params);
        assert_eq!(
            snapshot.cache,
            Some(ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour))
        );
    }

    #[test]
    fn cache_control_payload_serializes_expected_shape() {
        let one_hour =
            CacheControlPayload::from(ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour));
        let one_hour_json = serde_json::to_value(one_hour).expect("serialize one hour payload");
        assert_eq!(one_hour_json["type"], "ephemeral");
        assert_eq!(one_hour_json["ttl"], "1h");

        let short =
            CacheControlPayload::from(ClaudePromptCache::new(ClaudePromptCacheTtl::FiveMinutes));
        let short_json = serde_json::to_value(short).expect("serialize five minute payload");
        assert_eq!(short_json["type"], "ephemeral");
        assert!(short_json.get("ttl").is_none());
    }

    #[tokio::test]
    async fn multiple_system_messages_are_preserved_as_system_blocks() {
        let messages = vec![
            Message::system("Instruction A"),
            Message::system("Instruction B"),
            Message::user("Hello"),
        ];
        let (system, encoded) = to_claude_messages(&messages)
            .await
            .expect("encode Claude messages");
        assert_eq!(encoded.len(), 1);
        let system = system.expect("system payload should exist");
        match system {
            SystemPayload::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].text, "Instruction A");
                assert_eq!(blocks[1].text, "Instruction B");
            }
            other @ SystemPayload::Text(_) => {
                panic!("expected system blocks payload, got: {other:?}")
            }
        }
    }

    #[test]
    fn automatic_cache_strategy_returns_top_level_cache_control() {
        let mut system = Some(SystemPayload::Text("You are helpful".to_string()));
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("Hello".to_string()),
        }];
        let mut tools = None;
        let cache = ClaudePromptCache::automatic(ClaudePromptCacheTtl::FiveMinutes);
        let top_level = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect("automatic cache strategy should succeed");
        let top_level = top_level.expect("automatic strategy should return top-level cache");
        assert_eq!(top_level.kind, "ephemeral");
        assert!(top_level.ttl.is_none());
    }

    #[test]
    fn explicit_message_breakpoint_marks_last_message_block() {
        let mut system = None;
        let mut messages = vec![MessagePayload {
            role: "assistant",
            content: ContentPayload::Blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "search".to_string(),
                input: serde_json::json!({"q":"mars"}),
                cache_control: None,
            }]),
        }];
        let mut tools = None;
        let cache = ClaudePromptCache::new(ClaudePromptCacheTtl::OneHour).with_strategy(
            ClaudePromptCacheStrategy::Explicit(ClaudeExplicitCacheBreakpoints::messages_only()),
        );
        let top_level = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect("explicit strategy should succeed");
        assert!(top_level.is_none());

        match &messages[0].content {
            ContentPayload::Blocks(blocks) => match &blocks[0] {
                ContentBlock::ToolUse { cache_control, .. } => {
                    let cache_control =
                        cache_control.as_ref().expect("cache control should be set");
                    assert_eq!(cache_control.kind, "ephemeral");
                    assert_eq!(cache_control.ttl, Some("1h"));
                }
                other => panic!("expected tool_use block, got: {other:?}"),
            },
            other @ ContentPayload::Text(_) => {
                panic!("expected block payload, got: {other:?}")
            }
        }
    }

    #[test]
    fn explicit_breakpoint_target_out_of_range_fails_fast() {
        let mut system = None;
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("hello".to_string()),
        }];
        let mut tools = None;
        let cache = ClaudePromptCache::explicit(
            ClaudePromptCacheTtl::FiveMinutes,
            ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
                ClaudeCacheBreakpointTarget::Message {
                    message_index: 0,
                    block_index: 1,
                },
            )),
        );
        let err = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect_err("explicit strategy with out-of-range block index must fail");
        assert!(err.contains("block_index"));
    }

    #[test]
    fn automatic_and_explicit_strategy_sets_both_levels() {
        let mut system = Some(SystemPayload::Text("You are helpful".to_string()));
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("Tell me about Mars".to_string()),
        }];
        let mut tools = Some(vec![ToolPayload::Custom(CustomToolPayload {
            name: "search".to_string(),
            description: "search docs".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
            cache_control: None,
        })]);

        let cache =
            ClaudePromptCache::automatic_with_explicit(ClaudePromptCacheTtl::FiveMinutes, {
                ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
                    ClaudeCacheBreakpointTarget::LastTool,
                ))
                .with_second(ClaudeExplicitCacheBreakpoint::new(
                    ClaudeCacheBreakpointTarget::LastSystem,
                ))
            });
        let top_level = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect("combined strategy should succeed");
        assert!(top_level.is_some());

        let tools = tools.expect("tools should stay present");
        assert!(
            tools
                .last()
                .and_then(custom_tool)
                .and_then(|tool| tool.cache_control.as_ref())
                .is_some()
        );
    }

    #[test]
    fn mixed_ttl_requires_one_hour_before_five_minutes() {
        let mut system = Some(SystemPayload::Text("System".to_string()));
        let mut messages = vec![MessagePayload {
            role: "user",
            content: ContentPayload::Text("Hello".to_string()),
        }];
        let mut tools = Some(vec![ToolPayload::Custom(CustomToolPayload {
            name: "search".to_string(),
            description: "search docs".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
            cache_control: None,
        })]);

        let breakpoints = ClaudeExplicitCacheBreakpoints::new(
            ClaudeExplicitCacheBreakpoint::new(ClaudeCacheBreakpointTarget::LastTool)
                .with_ttl(ClaudePromptCacheTtl::FiveMinutes),
        )
        .with_second(
            ClaudeExplicitCacheBreakpoint::new(ClaudeCacheBreakpointTarget::LastMessage)
                .with_ttl(ClaudePromptCacheTtl::OneHour),
        );
        let cache = ClaudePromptCache::explicit(ClaudePromptCacheTtl::FiveMinutes, breakpoints);
        let err = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect_err("5m before 1h should fail mixed TTL ordering validation");
        assert!(err.contains("1h"));
    }

    #[test]
    fn automatic_with_full_explicit_slots_fails_when_no_overlap() {
        let mut system = Some(SystemPayload::Blocks(vec![
            SystemTextBlock {
                kind: "text",
                text: "S0".to_string(),
                cache_control: None,
            },
            SystemTextBlock {
                kind: "text",
                text: "S1".to_string(),
                cache_control: None,
            },
        ]));
        let mut messages = vec![
            MessagePayload {
                role: "assistant",
                content: ContentPayload::Blocks(vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q":"mars"}),
                    cache_control: None,
                }]),
            },
            MessagePayload {
                role: "user",
                content: ContentPayload::Text("Final user message".to_string()),
            },
        ];
        let mut tools = Some(vec![ToolPayload::Custom(CustomToolPayload {
            name: "search".to_string(),
            description: "search docs".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
            cache_control: None,
        })]);

        let breakpoints = ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::Tool(0),
        ))
        .with_second(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::System(0),
        ))
        .with_third(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::System(1),
        ))
        .with_fourth(ClaudeExplicitCacheBreakpoint::new(
            ClaudeCacheBreakpointTarget::Message {
                message_index: 0,
                block_index: 0,
            },
        ));
        let cache = ClaudePromptCache::automatic_with_explicit(
            ClaudePromptCacheTtl::FiveMinutes,
            breakpoints,
        );
        let err = apply_cache_strategy(&mut system, &mut messages, &mut tools, cache)
            .expect_err("automatic caching should fail when all 4 explicit slots are occupied");
        assert!(err.contains("at most 4 cache breakpoints"));
    }
}
