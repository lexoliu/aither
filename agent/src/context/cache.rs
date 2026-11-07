//! Prompt cache control for the assembled window.
//!
//! Every provider's prompt cache keys on an unbroken prefix, but they differ
//! in how that prefix is nominated. OpenAI-compatible endpoints want a stable
//! routing key so successive requests from one session land on the same
//! shard. Claude wants explicit breakpoints naming the blocks worth
//! persisting. Both are derived from the same two facts the agent already
//! knows: the window layout, and whether the cacheable part of it changed
//! since the previous request.
//!
//! The second fact is why this type exists rather than a free function.
//! Reassembly and compaction deliberately rewrite the stable prefix, which
//! makes everything cached under the old key dead weight. [`PromptCache`]
//! notices the change through a fingerprint and bumps an *epoch* that is
//! folded into the cache key, so a rewritten window asks for a new shard
//! instead of repeatedly missing on the old one.

use std::time::Duration;

use aither_core::llm::Message;
use aither_core::llm::model::Parameters;
use aither_core::llm::model::{
    CacheOptions, ClaudeCacheBreakpointTarget, ClaudeExplicitCacheBreakpoint,
    ClaudeExplicitCacheBreakpoints, ClaudePromptCache, ClaudePromptCacheTtl,
};

/// How long the conversation's prompt cache stays warm.
///
/// This is what an idle gap is measured against: reassembly rewrites the
/// cacheable prefix, so it is only free once the provider's copy can no
/// longer be read anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLifetime {
    /// The cache expires after this long without a hit.
    Limited(Duration),
    /// The cache never expires; rewriting the prefix is never free.
    Permanent,
}

/// Derives per-request cache controls for one agent session.
#[derive(Debug, Clone)]
pub struct PromptCache {
    /// Stable identity for this session's cache shard.
    session_key: String,
    /// Incremented whenever the cacheable prefix is rewritten.
    epoch: u32,
    /// Fingerprint of the stable prefix as of the last request.
    fingerprint: Option<u64>,
    /// How long Claude should retain the cached prefix.
    ttl: ClaudePromptCacheTtl,
    /// How long the conversation's cache stays warm, across providers.
    lifetime: CacheLifetime,
}

impl PromptCache {
    /// Creates a cache controller for a session identified by `session_key`.
    ///
    /// Agent runs are long and their prefixes are large, so the one-hour
    /// retention is the default: a five-minute TTL expires during ordinary
    /// think time and turns the next request into a full re-read.
    #[must_use]
    pub fn new(session_key: impl Into<String>) -> Self {
        Self {
            session_key: session_key.into(),
            epoch: 0,
            fingerprint: None,
            ttl: ClaudePromptCacheTtl::OneHour,
            lifetime: CacheLifetime::Limited(Duration::from_secs(3600)),
        }
    }

    /// Overrides the Claude cache retention.
    ///
    /// The cache lifetime follows the retention, since Claude's advertised
    /// TTL is exactly how long the provider promises to keep the prefix.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: ClaudePromptCacheTtl) -> Self {
        self.ttl = ttl;
        self.lifetime = CacheLifetime::Limited(match ttl {
            ClaudePromptCacheTtl::FiveMinutes => Duration::from_secs(300),
            ClaudePromptCacheTtl::OneHour => Duration::from_secs(3600),
        });
        self
    }

    /// Overrides how long the conversation's cache is trusted to stay warm.
    ///
    /// A host whose provider retains prompts longer than the breakpoints ask
    /// for (or forever) widens the idle-reassembly gap here; a lifetime of
    /// [`CacheLifetime::Permanent`] means an idle gap never justifies a
    /// reassembly.
    #[must_use]
    pub const fn with_lifetime(mut self, lifetime: CacheLifetime) -> Self {
        self.lifetime = lifetime;
        self
    }

    /// How long the conversation's cache stays warm.
    #[must_use]
    pub const fn lifetime(&self) -> CacheLifetime {
        self.lifetime
    }

    /// Resumes a session's key at `epoch`, as read back from
    /// [`PromptCache::epoch`] on the agent being replaced.
    ///
    /// A rebuilt agent keyed by the same session identity and epoch asks the
    /// provider for the same cache shard its predecessor warmed, instead of
    /// restarting at epoch zero.
    #[must_use]
    pub const fn with_epoch(mut self, epoch: u32) -> Self {
        self.epoch = epoch;
        self
    }

    /// The current cache key, which changes only when the prefix is rewritten.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}:{}", self.session_key, self.epoch)
    }

    /// How many times the cacheable prefix has been rewritten.
    #[must_use]
    pub const fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Records the stable prefix fingerprint for the request about to be sent.
    ///
    /// Returns `true` when the prefix changed, which means the previous
    /// epoch's cache is now unreachable.
    pub fn observe_prefix(&mut self, fingerprint: u64) -> bool {
        match self.fingerprint {
            Some(previous) if previous == fingerprint => false,
            Some(_) => {
                self.epoch += 1;
                self.fingerprint = Some(fingerprint);
                tracing::debug!(epoch = self.epoch, "prompt cache prefix rewritten");
                true
            }
            None => {
                self.fingerprint = Some(fingerprint);
                false
            }
        }
    }

    /// Applies cache controls to `parameters` for a request.
    ///
    /// `messages` is the assembled request, `has_tools` says whether tool
    /// definitions are attached, and `mid_conversation_system` whether the
    /// model keeps later system messages in the conversation; all three decide
    /// where a Claude breakpoint can legally be placed.
    #[must_use]
    pub fn apply(
        &self,
        mut parameters: Parameters,
        messages: &[Message],
        has_tools: bool,
        mid_conversation_system: bool,
    ) -> Parameters {
        parameters.cache = CacheOptions {
            openai: Some(aither_core::llm::model::OpenAIPromptCache {
                key: Some(self.key()),
                retention: Some(aither_core::llm::model::OpenAIPromptCacheRetention::Hours24),
            }),
            claude: Some(ClaudePromptCache::explicit(
                self.ttl,
                Self::breakpoints(messages, has_tools, mid_conversation_system),
            )),
            gemini: None,
        };
        parameters
    }

    /// Chooses the Claude breakpoints for an assembled request.
    ///
    /// The subtlety is the last message: it is the live tail, rebuilt from
    /// context sources on every single request, so caching *it* would write a
    /// new entry each turn and read none of them back. The breakpoint goes on
    /// the message before it — the newest byte that will still be there next
    /// time.
    fn breakpoints(
        messages: &[Message],
        has_tools: bool,
        mid_conversation_system: bool,
    ) -> ClaudeExplicitCacheBreakpoints {
        // Tool definitions precede the system prompt in Claude's cacheable
        // order, so a breakpoint there protects the whole tool surface.
        let breakpoints = if has_tools {
            ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
                ClaudeCacheBreakpointTarget::LastTool,
            ))
            .with_second(ClaudeExplicitCacheBreakpoint::new(
                ClaudeCacheBreakpointTarget::LastSystem,
            ))
        } else {
            ClaudeExplicitCacheBreakpoints::new(ClaudeExplicitCacheBreakpoint::new(
                ClaudeCacheBreakpointTarget::LastSystem,
            ))
        };
        let Some(index) = stable_conversation_end(messages)
            .and_then(|end| claude_message_index(messages, end, mid_conversation_system))
        else {
            return breakpoints;
        };
        let breakpoint = ClaudeExplicitCacheBreakpoint::new(ClaudeCacheBreakpointTarget::Message {
            message_index: index,
            block_index: 0,
        });
        if has_tools {
            breakpoints.with_third(breakpoint)
        } else {
            breakpoints.with_second(breakpoint)
        }
    }
}

/// Index of the last conversation message that will still be present, byte
/// for byte, in the next request.
///
/// Returns `None` when there is nothing but the prefix and the live tail.
pub(crate) const fn stable_conversation_end(messages: &[Message]) -> Option<usize> {
    // The final message is the live tail; the one before it is the newest
    // append-only item.
    messages.len().checked_sub(2)
}

/// Position, in the message list Claude is sent, of `messages[index]`.
///
/// Claude lifts the leading system messages into the top-level `system`. A
/// model that takes mid-conversation system messages keeps every later one in
/// place; any other folds all of them into `system`, so none counts. Returns
/// `None` when the message itself is lifted out.
fn claude_message_index(
    messages: &[Message],
    index: usize,
    mid_conversation_system: bool,
) -> Option<usize> {
    let is_system = |message: &Message| message.role() == aither_core::llm::Role::System;
    let lifted = if mid_conversation_system {
        messages
            .iter()
            .take_while(|message| is_system(message))
            .count()
    } else {
        if is_system(&messages[index]) {
            return None;
        }
        messages[..index]
            .iter()
            .filter(|message| is_system(message))
            .count()
    };
    index.checked_sub(lifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::model::ClaudePromptCacheStrategy;

    fn claude_strategy(parameters: &Parameters) -> ClaudePromptCacheStrategy {
        parameters
            .cache
            .claude
            .expect("claude cache options are set")
            .strategy
    }

    #[test]
    fn the_key_is_stable_until_the_prefix_is_rewritten() {
        let mut cache = PromptCache::new("session-1");
        assert!(!cache.observe_prefix(7));
        let first = cache.key();
        assert!(!cache.observe_prefix(7));
        assert_eq!(cache.key(), first);

        assert!(cache.observe_prefix(9));
        assert_ne!(cache.key(), first);
        assert_eq!(cache.epoch(), 1);
    }

    #[test]
    fn a_rebuilt_agent_keeps_the_sessions_key_and_rotates_with_the_epoch() {
        let mut first = PromptCache::new("session-1");
        first.observe_prefix(1);
        first.observe_prefix(2);
        let rebuilt = PromptCache::new("session-1").with_epoch(first.epoch());
        assert_eq!(rebuilt.key(), first.key());
        assert_eq!(rebuilt.key(), "session-1:1");
        assert_ne!(PromptCache::new("session-1").key(), rebuilt.key());
    }

    #[test]
    fn requests_carry_cache_options_for_every_provider_shape() {
        let cache = PromptCache::new("session-1");
        let messages = vec![
            Message::system("prefix"),
            Message::user("hello"),
            Message::system("live tail"),
        ];
        let parameters = cache.apply(Parameters::default(), &messages, true, true);

        assert!(!parameters.cache.is_empty());
        assert_eq!(
            parameters
                .cache
                .openai
                .as_ref()
                .and_then(|openai| openai.key.as_deref()),
            Some("session-1:0")
        );
        let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude_strategy(&parameters) else {
            panic!("expected explicit claude breakpoints");
        };
        assert_eq!(
            breakpoints.first.target,
            ClaudeCacheBreakpointTarget::LastTool
        );
        assert_eq!(
            breakpoints.second.map(|point| point.target),
            Some(ClaudeCacheBreakpointTarget::LastSystem)
        );
    }

    /// The live tail changes every request; caching it would write an entry
    /// per turn and never read one back.
    #[test]
    fn the_message_breakpoint_skips_the_live_tail() {
        let cache = PromptCache::new("session-1");
        let messages = vec![
            Message::system("prefix"),
            Message::user("hello"),
            Message::assistant("hi"),
            Message::system("live tail"),
        ];
        let parameters = cache.apply(Parameters::default(), &messages, false, true);
        let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude_strategy(&parameters) else {
            panic!("expected explicit claude breakpoints");
        };
        assert_eq!(
            breakpoints.second.map(|point| point.target),
            Some(ClaudeCacheBreakpointTarget::Message {
                message_index: 1,
                block_index: 0,
            }),
            "breakpoint must land on the last stable message, not the tail"
        );
    }

    /// The index is into the list Claude is sent, which lacks the leading
    /// system prefix.
    #[test]
    fn the_message_breakpoint_indexes_the_claude_message_list() {
        let cache = PromptCache::new("session-1");
        let messages = vec![
            Message::system("prefix"),
            Message::user("hello"),
            Message::system("live tail"),
        ];
        let parameters = cache.apply(Parameters::default(), &messages, false, true);
        let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude_strategy(&parameters) else {
            panic!("expected explicit claude breakpoints");
        };
        assert_eq!(
            breakpoints.second.map(|point| point.target),
            Some(ClaudeCacheBreakpointTarget::Message {
                message_index: 0,
                block_index: 0,
            })
        );
    }

    /// A model without mid-conversation system messages folds the tail into
    /// `system`, so the Claude list holds only the conversation.
    #[test]
    fn without_the_ability_no_system_message_counts() {
        let cache = PromptCache::new("session-1");
        let messages = vec![
            Message::system("prefix"),
            Message::user("hello"),
            Message::assistant("hi"),
            Message::system("live tail"),
        ];
        let parameters = cache.apply(Parameters::default(), &messages, false, false);
        let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude_strategy(&parameters) else {
            panic!("expected explicit claude breakpoints");
        };
        assert_eq!(
            breakpoints.second.map(|point| point.target),
            Some(ClaudeCacheBreakpointTarget::Message {
                message_index: 1,
                block_index: 0,
            })
        );
    }

    #[test]
    fn a_stable_system_prefix_gets_no_message_breakpoint() {
        let cache = PromptCache::new("session-1");
        let messages = vec![
            Message::system("prefix"),
            Message::system("more prefix"),
            Message::system("live tail"),
        ];
        let parameters = cache.apply(Parameters::default(), &messages, false, true);
        let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude_strategy(&parameters) else {
            panic!("expected explicit claude breakpoints");
        };
        assert!(breakpoints.second.is_none());
    }

    #[test]
    fn with_ttl_keeps_the_lifetime_coherent() {
        let cache = PromptCache::new("s").with_ttl(ClaudePromptCacheTtl::FiveMinutes);
        assert_eq!(
            cache.lifetime(),
            CacheLifetime::Limited(Duration::from_secs(300))
        );
    }

    #[test]
    fn a_prefix_only_request_gets_no_message_breakpoint() {
        let cache = PromptCache::new("session-1");
        let messages = vec![Message::system("prefix")];
        let parameters = cache.apply(Parameters::default(), &messages, false, true);
        let ClaudePromptCacheStrategy::Explicit(breakpoints) = claude_strategy(&parameters) else {
            panic!("expected explicit claude breakpoints");
        };
        assert!(breakpoints.second.is_none());
    }
}
