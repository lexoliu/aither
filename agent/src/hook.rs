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

use aither_core::llm::{CallIdentity, ToolResult};

use crate::external_hook::{InjectedContext, LifecyclePayload};
use crate::{ContextCheckpoint, context_window::ContextWindowSnapshot};

/// Context provided to hooks before a tool is called.
#[derive(Debug)]
pub struct ToolUseContext<'a> {
    /// The id of this call and the agent making it.
    pub call: &'a CallIdentity,
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
    /// Structured result of the tool execution.
    pub result: &'a ToolResult,
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

/// Context provided to hooks at a safe iteration boundary.
///
/// This boundary happens after the agent has fully processed the turn (its
/// tool results, or the alerts that object to it stopping) and updated its
/// internal context, but before it starts the next model request.
#[derive(Debug)]
pub struct TurnBoundaryContext<'a> {
    /// Assistant text the turn produced.
    pub assistant_text: &'a str,
    /// Current turn number (1-indexed within the agent loop).
    pub turn: usize,
    /// Number of recent conversation messages after processing the tool batch.
    pub message_count: usize,
    /// The working document as the model ends its turn; `None` after a
    /// turn that ran tools, and for an agent that keeps no document.
    pub document: Option<&'a crate::working_document::DocumentReview>,
}

/// What a [`StopHook`] decides as the model ends its turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopDecision {
    /// The run ends, unless a context source's alert keeps it going.
    End,
    /// The run goes on to another model request.
    Hold {
        /// What the model is told, as a system reminder; `None` when the
        /// alerts already raised say enough, such as the working document's.
        reminder: Option<String>,
    },
}

/// Decides whether the run ends when the model ends its turn: the
/// embedder's stop rule.
///
/// It is asked at every turn the model ends without tool calls, after the
/// context sources raised their alerts and the working document was
/// reviewed ([`TurnBoundaryContext::document`]). Without one, a run ends
/// whenever the model ends its turn and no source objects, whatever the
/// working document holds.
pub trait StopHook: Send + Sync + 'static {
    /// Decides for the turn `ctx` describes.
    fn on_turn_end(
        &self,
        ctx: &TurnBoundaryContext<'_>,
    ) -> impl std::future::Future<Output = StopDecision> + Send;
}

type Decide<'a> = std::pin::Pin<Box<dyn std::future::Future<Output = StopDecision> + Send + 'a>>;

/// Object-safe twin of [`StopHook`]; boxing lives inside [`AnyStopHook`].
trait StopHookImpl: Send + Sync {
    fn on_turn_end<'a>(&'a self, ctx: &'a TurnBoundaryContext<'a>) -> Decide<'a>;
}

impl<T: StopHook> StopHookImpl for T {
    fn on_turn_end<'a>(&'a self, ctx: &'a TurnBoundaryContext<'a>) -> Decide<'a> {
        Box::pin(StopHook::on_turn_end(self, ctx))
    }
}

/// A shared [`StopHook`], carried by the agent and handed to the subagents
/// it builds.
#[derive(Clone)]
pub struct AnyStopHook(std::sync::Arc<dyn StopHookImpl>);

impl AnyStopHook {
    /// Shares `hook`.
    #[must_use]
    pub fn new(hook: impl StopHook) -> Self {
        Self(std::sync::Arc::new(hook))
    }

    /// See [`StopHook::on_turn_end`].
    pub fn on_turn_end<'a>(
        &'a self,
        ctx: &'a TurnBoundaryContext<'a>,
    ) -> impl std::future::Future<Output = StopDecision> + Send + 'a {
        self.0.on_turn_end(ctx)
    }
}

impl std::fmt::Debug for AnyStopHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyStopHook").finish_non_exhaustive()
    }
}

/// Reason why a checkpoint is being emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointReason {
    /// A full tool/result cycle was committed and the next model request has not started yet.
    TurnBoundary,
    /// The run is about to stop with the current context state.
    Stop,
}

/// Structured checkpoint payload emitted from the runtime.
#[derive(Debug)]
pub struct CheckpointContext<'a> {
    /// Why this checkpoint was emitted.
    pub reason: CheckpointReason,
    /// Assistant text produced in the current turn before this checkpoint.
    pub assistant_text: &'a str,
    /// Current turn number (1-indexed within the agent loop).
    pub turn: usize,
    /// Number of recent conversation messages currently in memory.
    pub message_count: usize,
    /// Structured runtime context after the latest turn mutations.
    pub context: &'a ContextCheckpoint,
    /// Hash of the currently exposed tool surface.
    pub tool_surface_hash: &'a str,
    /// Whether background work is still active.
    pub has_background_tasks: bool,
    /// Structured snapshot of the currently assembled context window.
    pub window: &'a ContextWindowSnapshot,
    /// What the conversation holds of the working document, for
    /// [`AgentCheckpoint::document`](crate::AgentCheckpoint::document).
    pub document: Option<&'a crate::working_document::DocumentCheckpoint>,
}

/// Reason why the agent stopped.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Query completed successfully.
    Complete,
    /// No more tool calls in the response.
    NoToolCalls,
    /// Maximum iterations reached.
    MaxIterations,
    /// Model signaled end of turn.
    EndTurn,
    /// Every call of the model's last turn was refused for re-reading output
    /// that had not changed while background work runs: another request
    /// would only poll again, so the run waits for that work to report.
    AwaitingBackground,
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
    Replace(ToolResult),
    /// Abort the agent run entirely with an error.
    Abort(String),
}

/// Action to take at a safe turn boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnBoundaryAction {
    /// Continue with the next model request.
    Continue,
    /// End the current run cleanly after the processed tool batch.
    EndTurn,
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

    /// Called before the run goes on to another model request: after a full
    /// tool/result cycle is committed into agent memory, and after a turn
    /// without tool calls that a context source's alert or the
    /// [`StopHook`] keeps running.
    fn on_turn_boundary(
        &self,
        _ctx: &TurnBoundaryContext<'_>,
    ) -> impl std::future::Future<Output = TurnBoundaryAction> + Send {
        async { TurnBoundaryAction::Continue }
    }

    /// Called when the runtime emits a structured checkpoint.
    ///
    /// Return `Some(error)` to fail the run immediately.
    fn on_checkpoint(
        &self,
        _ctx: &CheckpointContext<'_>,
    ) -> impl std::future::Future<Output = Option<String>> + Send {
        async { None }
    }

    /// Called when a lifecycle event of the session happens.
    ///
    /// For observation: the event cannot be blocked or changed. A hook that
    /// has context to add keeps it and hands it over from
    /// [`take_injected_context`](Self::take_injected_context).
    fn on_lifecycle<P: LifecyclePayload>(
        &self,
        _payload: &P,
    ) -> impl std::future::Future<Output = ()> + Send {
        async {}
    }

    /// Hands over the context lifecycle events injected since the last call.
    ///
    /// The agent calls it where it collects system alerts and delivers each
    /// piece once, as a system alert.
    fn take_injected_context(&self) -> Vec<InjectedContext> {
        Vec::new()
    }
}

/// No-op implementation for unit type (base case for `HCons`).
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

    async fn on_turn_boundary(&self, ctx: &TurnBoundaryContext<'_>) -> TurnBoundaryAction {
        match self.head.on_turn_boundary(ctx).await {
            TurnBoundaryAction::Continue => self.tail.on_turn_boundary(ctx).await,
            other @ TurnBoundaryAction::EndTurn => other,
        }
    }

    async fn on_checkpoint(&self, ctx: &CheckpointContext<'_>) -> Option<String> {
        if let Some(err) = self.head.on_checkpoint(ctx).await {
            return Some(err);
        }
        self.tail.on_checkpoint(ctx).await
    }

    async fn on_lifecycle<P: LifecyclePayload>(&self, payload: &P) {
        futures::join!(
            self.head.on_lifecycle(payload),
            self.tail.on_lifecycle(payload)
        );
    }

    fn take_injected_context(&self) -> Vec<InjectedContext> {
        let mut context = self.head.take_injected_context();
        context.extend(self.tail.take_injected_context());
        context
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
        fn pre_tool_use(
            &self,
            _ctx: &ToolUseContext<'_>,
        ) -> impl std::future::Future<Output = PreToolAction> + Send {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::future::ready(PreToolAction::Allow)
        }
    }

    fn test_call() -> CallIdentity {
        CallIdentity::new(
            aither_core::llm::ToolCallId::new("call_1"),
            aither_core::llm::AgentIdentity::Main,
        )
    }

    #[tokio::test]
    async fn test_unit_hook() {
        let hook = ();
        let call = test_call();
        let ctx = ToolUseContext {
            call: &call,
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

        let call = test_call();
        let ctx = ToolUseContext {
            call: &call,
            tool_name: "test",
            arguments: "{}",
            turn: 0,
            message_count: 1,
        };

        let action = chain.pre_tool_use(&ctx).await;
        assert!(matches!(action, PreToolAction::Allow));
    }
}
