//! Structured snapshots of the currently assembled context window.

use aither_core::llm::Message;
use serde::{Deserialize, Serialize};

/// Lifecycle phase of the assembled context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextWindowPhase {
    /// Usage is below the compaction threshold.
    Stable,
    /// The window should be rebuilt before the next request.
    ///
    /// A staged system-block rewrite is waiting for the next compaction to
    /// fold it into the cacheable prefix.
    ReassemblyDue,
    /// A compaction is in flight alongside the run.
    Compacting,
    /// Usage reached the pause threshold and the run must wait for a handoff.
    Exhausted,
}

/// Metrics describing the currently assembled context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindowMetrics {
    /// Fraction of the calibrated context window currently in use.
    pub usage_fraction: f32,
    /// Usable context window in tokens.
    pub context_window: u32,
    /// Estimated prompt tokens currently held in the window.
    pub estimated_tokens: usize,
    /// Characters per token, calibrated against the provider's own counts.
    pub chars_per_token: f64,
    /// Number of persistent system blocks in the stable prefix.
    pub system_block_count: usize,
    /// Number of conversation messages.
    pub recent_message_count: usize,
    /// Number of durable system alerts that reassembly can prune.
    pub recent_system_message_count: usize,
    /// Whether a handoff document is present.
    pub has_handoff: bool,
    /// Whether a system block rewrite is waiting for reassembly.
    pub has_staged_system_updates: bool,
    /// How many times the cacheable prefix has been rewritten this session.
    pub prompt_cache_epoch: u32,
}

/// Fully assembled LLM request window plus lifecycle metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindowSnapshot {
    /// Lifecycle phase derived from the current policy.
    pub phase: ContextWindowPhase,
    /// Metrics for observability and persistence.
    pub metrics: ContextWindowMetrics,
    /// Exact message list that would be sent to the model for the next turn.
    pub messages: Vec<Message>,
}
