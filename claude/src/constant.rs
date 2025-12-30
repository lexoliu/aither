//! Claude API constants and model identifiers.

/// Base URL for the Anthropic Messages API.
pub const CLAUDE_BASE_URL: &str = "https://api.anthropic.com";

/// Required API version header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

// =============================================================================
// Model Identifiers
// =============================================================================

/// Claude Sonnet 4.5 - Best model for real-world agents and coding.
pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5";

/// Claude Sonnet 4.5 with specific version date.
pub const CLAUDE_SONNET_4_5_20250929: &str = "claude-sonnet-4-5-20250929";

/// Claude Sonnet 4.0 - High-performance model with extended thinking.
pub const CLAUDE_SONNET_4_0: &str = "claude-sonnet-4-0";

/// Claude Opus 4.1 - Most capable model.
pub const CLAUDE_OPUS_4_1: &str = "claude-opus-4-1-20250805";

/// Claude Opus 4.0 - Most capable model.
pub const CLAUDE_OPUS_4_0: &str = "claude-opus-4-0";

/// Claude Haiku 4.5 - Hybrid model, capable of near-instant responses and extended thinking.
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

/// Claude 3.5 Haiku - Fastest and most compact model for near-instant responsiveness.
pub const CLAUDE_3_5_HAIKU_LATEST: &str = "claude-3-5-haiku-latest";

/// Claude 3.7 Sonnet - High-performance model with early extended thinking.
pub const CLAUDE_3_7_SONNET_LATEST: &str = "claude-3-7-sonnet-latest";

/// Claude 3 Opus - Excels at writing and complex tasks.
pub const CLAUDE_3_OPUS_LATEST: &str = "claude-3-opus-latest";

// =============================================================================
// Defaults
// =============================================================================

/// Default model for chat completions.
pub const DEFAULT_MODEL: &str = CLAUDE_SONNET_4_5;

/// Default maximum tokens for responses.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Maximum tool calling iterations to prevent infinite loops.
pub const MAX_TOOL_ITERATIONS: usize = 8;
