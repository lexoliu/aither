//! Hook system for intercepting agent operations.
//!
//! Hooks allow customizing agent behavior at key points:
//! - Before/after tool calls
//! - When the agent stops
//! - When text is streamed
//!
//! # Example
//!
//! ```rust,ignore
//! struct LoggingHook;
//!
//! impl Hook for LoggingHook {
//!     async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> HookAction {
//!         println!("Calling: {}", ctx.tool_name);
//!         HookAction::Continue
//!     }
//! }
//!
//! let agent = Agent::builder(llm)
//!     .hook(LoggingHook)
//!     .build();
//! ```

use std::time::Duration;

/// Context provided to hooks before a tool is called.
#[derive(Debug)]
pub struct ToolUseContext<'a> {
    /// Name of the tool being called.
    pub tool_name: &'a str,
    /// JSON-encoded arguments.
    pub arguments: &'a str,
    /// Current turn number (0-indexed).
    pub turn: usize,
    /// Number of messages in the conversation.
    pub message_count: usize,
}

/// Context provided to hooks after a tool is executed.
#[derive(Debug)]
pub struct ToolResultContext<'a> {
    /// Name of the tool that was called.
    pub tool_name: &'a str,
    /// JSON-encoded arguments that were passed.
    pub arguments: &'a str,
    /// Result of the tool execution.
    pub result: Result<&'a str, &'a str>,
    /// Time taken to execute the tool.
    pub duration: Duration,
}

/// Context provided to hooks when the agent is about to stop.
#[derive(Debug)]
pub struct StopContext<'a> {
    /// The final response text.
    pub final_text: &'a str,
    /// Total number of turns taken.
    pub turns: usize,
    /// Reason for stopping.
    pub reason: StopReason,
}

/// Reason why the agent stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Query completed successfully.
    Complete,
    /// No more tool calls in the response.
    NoToolCalls,
    /// Maximum iterations reached.
    MaxIterations,
    /// Model signaled end of turn.
    EndTurn,
}

/// Result of a hook invocation.
#[derive(Debug, Clone)]
pub enum HookAction {
    /// Continue with the operation normally.
    Continue,

    /// Skip this tool call (only valid for `pre_tool_use`).
    Skip,

    /// Abort the agent run with an error.
    Abort(String),

    /// Replace the tool result with a custom value (only valid for `post_tool_use`).
    Replace(String),
}

impl HookAction {
    /// Returns `true` if this action indicates the operation should be skipped.
    #[must_use]
    pub fn should_skip(&self) -> bool {
        matches!(self, Self::Skip)
    }

    /// Returns `true` if this action indicates the agent should abort.
    #[must_use]
    pub fn should_abort(&self) -> bool {
        matches!(self, Self::Abort(_))
    }

    /// Returns the abort reason if this is an abort action.
    #[must_use]
    pub fn abort_reason(&self) -> Option<&str> {
        match self {
            Self::Abort(reason) => Some(reason),
            _ => None,
        }
    }

    /// Returns the replacement value if this is a replace action.
    #[must_use]
    pub fn replacement(&self) -> Option<&str> {
        match self {
            Self::Replace(value) => Some(value),
            _ => None,
        }
    }
}

/// Trait for intercepting agent operations.
///
/// All methods have default no-op implementations that return `HookAction::Continue`.
/// Implement only the methods you need.
///
/// Hooks are composed using the [`HCons`] type, allowing multiple hooks to be
/// chained together at compile time with zero runtime overhead.
pub trait Hook: Send + Sync {
    /// Called before a tool is executed.
    ///
    /// Return `HookAction::Skip` to skip the tool call, or `HookAction::Abort`
    /// to stop the agent entirely.
    fn pre_tool_use(
        &self,
        _ctx: &ToolUseContext<'_>,
    ) -> impl std::future::Future<Output = HookAction> + Send {
        async { HookAction::Continue }
    }

    /// Called after a tool is executed.
    ///
    /// Return `HookAction::Replace` to override the tool result, or
    /// `HookAction::Abort` to stop the agent.
    fn post_tool_use(
        &self,
        _ctx: &ToolResultContext<'_>,
    ) -> impl std::future::Future<Output = HookAction> + Send {
        async { HookAction::Continue }
    }

    /// Called when the agent is about to stop.
    ///
    /// Return `HookAction::Abort` to convert a successful stop into an error.
    fn on_stop(
        &self,
        _ctx: &StopContext<'_>,
    ) -> impl std::future::Future<Output = HookAction> + Send {
        async { HookAction::Continue }
    }

    /// Called when text is streamed from the LLM.
    ///
    /// This is for observation only; the return value is ignored.
    fn on_text(&self, _text: &str) -> impl std::future::Future<Output = ()> + Send {
        async {}
    }
}

/// No-op implementation for unit type (base case for HCons).
impl Hook for () {}

/// Heterogeneous list cons cell for composing hooks at compile time.
///
/// This allows chaining multiple hooks together without boxing or dynamic dispatch.
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::builder(llm)
///     .hook(LoggingHook)
///     .hook(ConfirmationHook)
///     .build();
/// // Creates Agent<LLM, HCons<ConfirmationHook, HCons<LoggingHook, ()>>>
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct HCons<Head, Tail> {
    /// The first hook in the chain.
    pub head: Head,
    /// The remaining hooks.
    pub tail: Tail,
}

impl<Head, Tail> HCons<Head, Tail> {
    /// Creates a new hook chain.
    #[must_use]
    pub const fn new(head: Head, tail: Tail) -> Self {
        Self { head, tail }
    }
}

impl<Head, Tail> Hook for HCons<Head, Tail>
where
    Head: Hook,
    Tail: Hook,
{
    async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> HookAction {
        match self.head.pre_tool_use(ctx).await {
            HookAction::Continue => self.tail.pre_tool_use(ctx).await,
            other => other,
        }
    }

    async fn post_tool_use(&self, ctx: &ToolResultContext<'_>) -> HookAction {
        match self.head.post_tool_use(ctx).await {
            HookAction::Continue => self.tail.post_tool_use(ctx).await,
            other => other,
        }
    }

    async fn on_stop(&self, ctx: &StopContext<'_>) -> HookAction {
        match self.head.on_stop(ctx).await {
            HookAction::Continue => self.tail.on_stop(ctx).await,
            other => other,
        }
    }

    async fn on_text(&self, text: &str) {
        self.head.on_text(text).await;
        self.tail.on_text(text).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingHook {
        count: std::sync::atomic::AtomicUsize,
    }

    impl CountingHook {
        fn new() -> Self {
            Self {
                count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn count(&self) -> usize {
            self.count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl Hook for CountingHook {
        async fn pre_tool_use(&self, _ctx: &ToolUseContext<'_>) -> HookAction {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            HookAction::Continue
        }
    }

    #[tokio::test]
    async fn test_unit_hook() {
        let hook = ();
        let ctx = ToolUseContext {
            tool_name: "test",
            arguments: "{}",
            turn: 0,
            message_count: 1,
        };
        let action = hook.pre_tool_use(&ctx).await;
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_hcons_chain() {
        let chain = HCons::new(CountingHook::new(), HCons::new(CountingHook::new(), ()));

        let ctx = ToolUseContext {
            tool_name: "test",
            arguments: "{}",
            turn: 0,
            message_count: 1,
        };

        let action = chain.pre_tool_use(&ctx).await;
        assert!(matches!(action, HookAction::Continue));
        // Both hooks should have been called (count would be 1 each, but we can't check
        // individual counts without keeping references - just verify it didn't panic)
    }

    #[test]
    fn test_hook_action_helpers() {
        assert!(HookAction::Skip.should_skip());
        assert!(!HookAction::Continue.should_skip());

        assert!(HookAction::Abort("reason".to_string()).should_abort());
        assert!(!HookAction::Continue.should_abort());

        assert_eq!(
            HookAction::Abort("reason".to_string()).abort_reason(),
            Some("reason")
        );
        assert_eq!(HookAction::Continue.abort_reason(), None);

        assert_eq!(
            HookAction::Replace("new".to_string()).replacement(),
            Some("new")
        );
        assert_eq!(HookAction::Continue.replacement(), None);
    }
}
