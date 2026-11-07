//! The native `compact` tool.
//!
//! Only the harness used to decide when to compact, so compaction landed
//! wherever usage crossed the threshold, often mid-task. The model knows when
//! its work is at a clean point; this tool lets it ask for compaction there.
//!
//! The tool does not compact. It raises a request on a [`CompactSignal`] it
//! shares with its agent and returns at once; the agent starts the
//! compaction at the next request boundary, through the same state and the
//! same fork and apply path automatic compaction uses, so the tool call and
//! its result are in the window the compaction summarises and the turn never
//! waits.
//!
//! A call is worth a compaction only when none is in flight, something was
//! done since the last one landed, and the context holds at least the
//! [`ContextPolicy`](crate::ContextPolicy) compaction floor; otherwise the
//! call starts nothing and says which reason applied.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

use aither_core::llm::ToolContext;
use aither_core::llm::tool::{Tool, ToolResult};
use askama::Template;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The name the tool is registered under.
pub const COMPACT_TOOL_NAME: &str = "compact";

/// No compaction is requested or running.
const IDLE: u8 = 0;
/// The model asked for a compaction the agent has not started yet.
const REQUESTED: u8 = 1;
/// A compaction is running or waiting to be applied.
const RUNNING: u8 = 2;

/// What a `compact` call found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactRequest {
    /// The request was recorded; the agent starts the compaction.
    Started,
    /// A compaction is already requested or running; nothing was recorded.
    InFlight,
    /// A compaction completed and nothing was done since; nothing was recorded.
    Settled,
    /// The context holds fewer than `floor` tokens; nothing was recorded.
    BelowFloor {
        /// The compaction floor the context is under, in tokens.
        floor: usize,
    },
}

/// The compaction state an agent shares with its `compact` tool.
///
/// The agent publishes what the valve needs at each request boundary; the
/// tool reads it and records at most one request.
#[derive(Debug, Default)]
pub struct CompactSignal {
    state: AtomicU8,
    /// Tokens the context held at the last request boundary.
    tokens: AtomicUsize,
    /// The compaction floor of the agent's policy.
    floor: AtomicUsize,
    /// Whether a compaction landed and no action has followed it.
    settled: AtomicBool,
}

impl CompactSignal {
    /// Records a request unless the valve refuses it, and says what was found.
    ///
    /// The reasons are checked in order: a compaction in flight, a compaction
    /// completed with nothing done since, a context under the floor.
    pub(crate) fn request(&self) -> CompactRequest {
        if self.state.load(Ordering::Acquire) != IDLE {
            return CompactRequest::InFlight;
        }
        if self.settled.load(Ordering::Acquire) {
            return CompactRequest::Settled;
        }
        let floor = self.floor.load(Ordering::Acquire);
        if self.tokens.load(Ordering::Acquire) < floor {
            return CompactRequest::BelowFloor { floor };
        }
        match self
            .state
            .compare_exchange(IDLE, REQUESTED, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => CompactRequest::Started,
            Err(_) => CompactRequest::InFlight,
        }
    }

    /// Takes the pending request, returning whether there was one. The
    /// compaction it asks for counts as running from here.
    pub(crate) fn take_request(&self) -> bool {
        self.state
            .compare_exchange(REQUESTED, RUNNING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Brings the valve in line with the agent at a request boundary: a
    /// compaction is running while one is in flight or waiting to be applied
    /// and idle once none is, a request not yet taken is kept, and the
    /// context holds `tokens` against the policy's `floor`.
    pub(crate) fn sync(&self, in_flight: bool, tokens: usize, floor: usize) {
        self.tokens.store(tokens, Ordering::Release);
        self.floor.store(floor, Ordering::Release);
        if in_flight {
            self.state.store(RUNNING, Ordering::Release);
        } else {
            let _ = self
                .state
                .compare_exchange(RUNNING, IDLE, Ordering::AcqRel, Ordering::Acquire);
        }
    }

    /// Records that a compaction landed: the next call is refused until
    /// [`acted`](Self::acted).
    pub(crate) fn landed(&self) {
        self.settled.store(true, Ordering::Release);
    }

    /// Records that the agent did something other than compact.
    pub(crate) fn acted(&self) {
        self.settled.store(false, Ordering::Release);
    }
}

#[derive(Template)]
#[template(path = "compact_description.txt", escape = "none")]
struct CompactDescription;

/// Starts compaction of the conversation so far.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CompactArgs {}

/// The `compact` tool, bound to the agent it was registered on.
#[derive(Debug, Clone)]
pub struct CompactTool {
    signal: std::sync::Arc<CompactSignal>,
    /// Renders the result of each call.
    notices: crate::AnyNotices,
}

impl CompactTool {
    /// Creates the tool over the `signal` its agent reads, its results
    /// rendered through `notices`.
    #[must_use]
    pub(crate) const fn new(
        signal: std::sync::Arc<CompactSignal>,
        notices: crate::AnyNotices,
    ) -> Self {
        Self { signal, notices }
    }
}

impl Tool for CompactTool {
    fn name(&self) -> Cow<'static, str> {
        COMPACT_TOOL_NAME.into()
    }

    fn description(&self) -> Cow<'static, str> {
        CompactDescription
            .render()
            .expect("compact description template must render")
            .into()
    }

    type Arguments = CompactArgs;
    type Res = ToolResult;

    #[allow(clippy::unused_async_trait_impl, reason = "the trait method is async")]
    async fn call(
        &self,
        _args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let request = self.signal.request();
        let text = self.notices.compact(&request);
        // A refusal is a typed tool error, so an embedder tells it from a
        // start without reading the wording; the model reads the same text.
        Ok(match request {
            CompactRequest::Started => ToolResult::text(text),
            CompactRequest::InFlight
            | CompactRequest::Settled
            | CompactRequest::BelowFloor { .. } => ToolResult::error(text),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_recorded_once_until_the_compaction_ends() {
        let signal = CompactSignal::default();
        assert_eq!(signal.request(), CompactRequest::Started);
        assert_eq!(signal.request(), CompactRequest::InFlight);
        assert!(signal.take_request());
        assert!(!signal.take_request());
        assert_eq!(signal.request(), CompactRequest::InFlight);
        signal.sync(true, 0, 0);
        assert_eq!(signal.request(), CompactRequest::InFlight);
        signal.sync(false, 0, 0);
        assert_eq!(signal.request(), CompactRequest::Started);
    }

    #[test]
    fn syncing_idle_keeps_a_request_not_yet_taken() {
        let signal = CompactSignal::default();
        assert_eq!(signal.request(), CompactRequest::Started);
        signal.sync(false, 0, 0);
        assert!(signal.take_request());
    }

    #[test]
    fn a_landed_compaction_refuses_calls_until_the_agent_acts() {
        let signal = CompactSignal::default();
        signal.landed();
        assert_eq!(signal.request(), CompactRequest::Settled);
        signal.acted();
        assert_eq!(signal.request(), CompactRequest::Started);
    }

    #[test]
    fn a_context_under_the_floor_refuses_calls_and_names_it() {
        let signal = CompactSignal::default();
        signal.sync(false, 19_999, 20_000);
        assert_eq!(
            signal.request(),
            CompactRequest::BelowFloor { floor: 20_000 }
        );
        signal.sync(false, 20_000, 20_000);
        assert_eq!(signal.request(), CompactRequest::Started);
    }

    #[test]
    fn the_description_asks_for_a_clean_point_and_tidy_notes() {
        let tool = CompactTool::new(std::sync::Arc::default(), crate::AnyNotices::default());
        let description = tool.description();
        for needle in ["clean point", "topic", "notes", "verbatim"] {
            assert!(description.contains(needle), "{needle}: {description}");
        }
    }
}
