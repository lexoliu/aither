//! Idle compaction driven through the public `TerminalAgentBuilder` only, on an
//! injected clock, the way a downstream host exercises it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::context::CacheLifetime;
use crate::sandbox::TerminalTool;
use crate::sandbox::permission::NoopPermissionHandler;
use crate::{Agent, ManualClock, TerminalAgentBuilder};
use aither_core::LanguageModel;
use aither_core::llm::{Event, LLMRequest, model::Profile as ModelProfile};
use executor_core::tokio::TokioGlobal;
use futures_core::Stream;
use tempfile::tempdir;

use crate::test_support;

const HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

#[derive(Debug)]
struct MockError;

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mock error")
    }
}

impl std::error::Error for MockError {}

/// Replies "ok" to the run, then the handoff document to a compaction, and
/// counts the requests it received.
#[derive(Clone, Default)]
struct ScriptedLlm {
    requests: Arc<AtomicUsize>,
}

impl ScriptedLlm {
    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl LanguageModel for ScriptedLlm {
    type Error = MockError;

    fn respond(
        &self,
        _request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let seen = self.requests.fetch_add(1, Ordering::SeqCst);
        let text = if seen == 0 { "ok" } else { HANDOFF_JSON };
        futures_lite::stream::iter([Ok(Event::Text(text.to_string()))])
    }

    fn profile(
        &self,
    ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
        std::future::ready(Ok(ModelProfile::new(
            "mock",
            "test",
            "mock-model",
            "mock model",
            1_000_000,
        )))
    }
}

/// Builds an agent through the public builder and runs one turn on it.
async fn agent_after_one_turn(
    llm: ScriptedLlm,
    lifetime: Option<CacheLifetime>,
    clock: &ManualClock,
) -> Agent<ScriptedLlm> {
    let dir = tempdir().expect("temp dir");
    let terminal = TerminalTool::new_exact(dir.path(), NoopPermissionHandler, TokioGlobal)
        .await
        .expect("terminal tool initializes");
    let tools = tempdir().expect("tool dir");
    let cache = tempdir().expect("cache dir");
    let binary = test_support::compile_baseline_fixture(tools.path()).await;
    let distribution = test_support::fixture_distribution(&binary, "1.0.0").await;
    let mut builder = TerminalAgentBuilder::new(llm, terminal)
        .initialize_tools(&distribution, cache.path())
        .await
        .expect("tools prepare")
        .clock(clock.clone());
    if let Some(lifetime) = lifetime {
        builder = builder.cache_lifetime(lifetime);
    }
    let mut agent = builder.build();
    {
        let events = agent.run("hello", []);
        futures_lite::pin!(events);
        while futures_lite::StreamExt::next(&mut events).await.is_some() {}
    }
    agent
}

#[tokio::test(flavor = "multi_thread")]
async fn a_limited_lifetime_schedules_one_idle_compaction_on_the_injected_clock() {
    let llm = ScriptedLlm::default();
    let clock = ManualClock::new();
    let lifetime = CacheLifetime::Limited(Duration::from_secs(600));
    let mut agent = agent_after_one_turn(llm.clone(), Some(lifetime), &clock).await;
    assert_eq!(agent.cache_lifetime(), lifetime);
    assert_eq!(llm.requests(), 1);

    // Nothing is due while the clock stands still, and calling is a no-op.
    let due_in = agent
        .idle_compaction_due_in()
        .expect("a compaction is scheduled");
    assert!(due_in > Duration::ZERO && due_in < Duration::from_secs(600));
    assert!(agent.compact_when_idle().await.expect("no-op").is_none());
    assert_eq!(llm.requests(), 1);

    // Just short of the due point: still not due.
    clock.advance(due_in.saturating_sub(Duration::from_secs(1)));
    assert!(agent.compact_when_idle().await.expect("no-op").is_none());
    assert_eq!(llm.requests(), 1);

    // At the due point: exactly one compaction.
    clock.advance(Duration::from_secs(1));
    assert_eq!(agent.idle_compaction_due_in(), Some(Duration::ZERO));
    agent
        .compact_when_idle()
        .await
        .expect("the compaction succeeds")
        .expect("the compaction ran");
    assert_eq!(llm.requests(), 2);

    // Still idle, later in the same gap: nothing more.
    clock.advance(Duration::from_secs(30));
    assert!(agent.compact_when_idle().await.expect("no-op").is_none());
    assert_eq!(llm.requests(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_permanent_lifetime_schedules_no_idle_compaction() {
    let llm = ScriptedLlm::default();
    let clock = ManualClock::new();
    let mut agent = agent_after_one_turn(llm.clone(), Some(CacheLifetime::Permanent), &clock).await;
    assert_eq!(agent.cache_lifetime(), CacheLifetime::Permanent);

    clock.advance(Duration::from_secs(10 * 86_400));
    assert_eq!(agent.idle_compaction_due_in(), None);
    assert!(agent.compact_when_idle().await.expect("no-op").is_none());
    assert_eq!(llm.requests(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_default_lifetime_is_one_hour() {
    let clock = ManualClock::new();
    let agent = agent_after_one_turn(ScriptedLlm::default(), None, &clock).await;
    assert_eq!(
        agent.cache_lifetime(),
        CacheLifetime::Limited(Duration::from_secs(3600))
    );
}
