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
//!     async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> PreToolAction {
//!         println!("Calling: {}", ctx.tool_name);
//!         PreToolAction::Allow
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

/// Action to take before a tool is executed.
#[derive(Debug, Clone)]
pub enum PreToolAction {
    /// Allow the tool to execute.
    Allow,
    /// Deny the tool execution. The reason is sent to the LLM as an error.
    Deny(String),
    /// Abort the agent run entirely with an error.
    Abort(String),
}

/// Action to take after a tool is executed.
#[derive(Debug, Clone)]
pub enum PostToolAction {
    /// Keep the original tool result.
    Keep,
    /// Replace the tool result with a custom value.
    Replace(String),
    /// Abort the agent run entirely with an error.
    Abort(String),
}

/// Trait for intercepting agent operations.
///
/// All methods have default no-op implementations.
/// Implement only the methods you need.
///
/// Hooks are composed using the [`HCons`] type, allowing multiple hooks to be
/// chained together at compile time with zero runtime overhead.
pub trait Hook: Send + Sync {
    /// Called before a tool is executed.
    ///
    /// Return `PreToolAction::Deny` to reject the tool call (reason sent to LLM),
    /// or `PreToolAction::Abort` to stop the agent entirely.
    fn pre_tool_use(
        &self,
        _ctx: &ToolUseContext<'_>,
    ) -> impl std::future::Future<Output = PreToolAction> + Send {
        async { PreToolAction::Allow }
    }

    /// Called after a tool is executed.
    ///
    /// Return `PostToolAction::Replace` to override the tool result, or
    /// `PostToolAction::Abort` to stop the agent.
    fn post_tool_use(
        &self,
        _ctx: &ToolResultContext<'_>,
    ) -> impl std::future::Future<Output = PostToolAction> + Send {
        async { PostToolAction::Keep }
    }

    /// Called when the agent is about to stop.
    ///
    /// Return `Some(error)` to convert a successful stop into an error.
    fn on_stop(
        &self,
        _ctx: &StopContext<'_>,
    ) -> impl std::future::Future<Output = Option<String>> + Send {
        async { None }
    }

    /// Called when text is streamed from the LLM.
    ///
    /// This is for observation only.
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
    async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> PreToolAction {
        match self.head.pre_tool_use(ctx).await {
            PreToolAction::Allow => self.tail.pre_tool_use(ctx).await,
            other => other,
        }
    }

    async fn post_tool_use(&self, ctx: &ToolResultContext<'_>) -> PostToolAction {
        match self.head.post_tool_use(ctx).await {
            PostToolAction::Keep => self.tail.post_tool_use(ctx).await,
            other => other,
        }
    }

    async fn on_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        if let Some(err) = self.head.on_stop(ctx).await {
            return Some(err);
        }
        self.tail.on_stop(ctx).await
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
    }

    impl Hook for CountingHook {
        async fn pre_tool_use(&self, _ctx: &ToolUseContext<'_>) -> PreToolAction {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            PreToolAction::Allow
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
        assert!(matches!(action, PreToolAction::Allow));
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
        assert!(matches!(action, PreToolAction::Allow));
    }
}
