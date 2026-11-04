//! The native `compact` tool.
//!
//! Only the harness used to decide when to compact, so compaction landed
//! wherever usage crossed the threshold, often mid-task. The model knows when
//! its work is at a clean point; this tool lets it ask for compaction there.
//!
//! The tool does not compact. It raises a request on a [`CompactSignal`] it
//! shares with its agent and returns at once; the agent starts the
//! compaction at the next request boundary, through the same fork and apply
//! path automatic compaction uses, so the tool call and its result are in the
//! window the compaction summarises and the turn never waits.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU8, Ordering};

use aither_core::llm::ToolContext;
use aither_core::llm::tool::{Tool, ToolResult};
use askama::Template;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
}

/// The compaction state an agent shares with its `compact` tool.
#[derive(Debug, Default)]
pub struct CompactSignal {
    state: AtomicU8,
}

impl CompactSignal {
    /// Records a request unless a compaction is already requested or running.
    pub(crate) fn request(&self) -> CompactRequest {
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

    /// Brings the state in line with the agent: a compaction is running while
    /// one is in flight or waiting to be applied, and idle once none is. A
    /// request not yet taken is kept.
    pub(crate) fn sync(&self, in_flight: bool) {
        if in_flight {
            self.state.store(RUNNING, Ordering::Release);
        } else {
            let _ = self
                .state
                .compare_exchange(RUNNING, IDLE, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

#[derive(Template)]
#[template(path = "compact_started.txt", escape = "none")]
struct CompactStarted;

#[derive(Template)]
#[template(path = "compact_in_flight.txt", escape = "none")]
struct CompactInFlight;

/// Starts compaction of the conversation so far.
///
/// Call it at a clean point of your work, after saving what the summary
/// cannot carry. It returns at once and the work continues; the summary
/// replaces the older conversation at a later step. A call while a
/// compaction is already in flight starts nothing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CompactArgs {}

/// The `compact` tool, bound to the agent it was registered on.
#[derive(Debug, Clone)]
pub struct CompactTool {
    signal: std::sync::Arc<CompactSignal>,
}

impl CompactTool {
    /// Creates the tool over the `signal` its agent reads.
    #[must_use]
    pub(crate) const fn new(signal: std::sync::Arc<CompactSignal>) -> Self {
        Self { signal }
    }
}

impl Tool for CompactTool {
    fn name(&self) -> Cow<'static, str> {
        "compact".into()
    }

    type Arguments = CompactArgs;
    type Res = ToolResult;

    #[allow(clippy::unused_async_trait_impl, reason = "the trait method is async")]
    async fn call(
        &self,
        _args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let text = match self.signal.request() {
            CompactRequest::Started => CompactStarted.render(),
            CompactRequest::InFlight => CompactInFlight.render(),
        }
        .expect("compact result template must render");
        Ok(ToolResult::text(text))
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
        signal.sync(true);
        assert_eq!(signal.request(), CompactRequest::InFlight);
        signal.sync(false);
        assert_eq!(signal.request(), CompactRequest::Started);
    }

    #[test]
    fn syncing_idle_keeps_a_request_not_yet_taken() {
        let signal = CompactSignal::default();
        assert_eq!(signal.request(), CompactRequest::Started);
        signal.sync(false);
        assert!(signal.take_request());
    }
}
