//! Ends a provider response stream that the machine's sleep or network loss
//! has made unreachable.

use core::pin::Pin;

use aither_core::llm::Event;
use futures_core::Stream;

use crate::AgentError;

/// A provider response stream as the agent consumes it.
pub type EventStream<'a> = Pin<Box<dyn Stream<Item = Result<Event, AgentError>> + Send + 'a>>;

/// The connectivity source an agent watches while a response is in flight;
/// empty unless the builder was given one.
#[derive(Debug, Clone, Default)]
pub struct Watch {
    #[cfg(feature = "connectivity")]
    source: Option<aither_connectivity::Connectivity>,
}

impl Watch {
    #[cfg(feature = "connectivity")]
    pub const fn new(source: aither_connectivity::Connectivity) -> Self {
        Self {
            source: Some(source),
        }
    }

    /// Wraps `stream` so a wake or an offline change ends it.
    #[cfg_attr(not(feature = "connectivity"), allow(clippy::unused_self))]
    pub fn guard<'a>(&self, stream: EventStream<'a>) -> EventStream<'a> {
        #[cfg(feature = "connectivity")]
        if let Some(source) = &self.source {
            return abandon_on_wake_or_offline(source, stream);
        }
        stream
    }
}

#[cfg(feature = "connectivity")]
fn abandon_on_wake_or_offline<'a>(
    source: &aither_connectivity::Connectivity,
    mut inner: EventStream<'a>,
) -> EventStream<'a> {
    use aither_connectivity::{Change, ChangeKind};
    use aither_core::llm::Failure;
    use futures_lite::{StreamExt, future};

    enum Step {
        Event(Option<Result<Event, AgentError>>),
        Change(Option<Change>),
    }

    // Subscribed now, before the request is sent, so a change between the two
    // is not missed.
    let mut changes = Box::pin(source.subscribe());
    Box::pin(async_stream::stream! {
        loop {
            // Changes first, so a busy stream cannot delay abandoning it.
            let step = future::or(
                async { Step::Change(changes.next().await) },
                async { Step::Event(inner.next().await) },
            )
            .await;
            let reason = match step {
                Step::Event(Some(item)) => {
                    yield item;
                    continue;
                }
                Step::Event(None) => break,
                Step::Change(Some(Change { kind: ChangeKind::Woke, .. })) => {
                    "the machine woke from sleep"
                }
                Step::Change(Some(Change { kind: ChangeKind::Offline, .. })) => {
                    "the network went offline"
                }
                Step::Change(Some(_)) => continue,
                // The monitor stopped; the stream is no longer watched.
                Step::Change(None) => {
                    while let Some(item) = inner.next().await {
                        yield item;
                    }
                    break;
                }
            };
            yield Err(AgentError::Llm {
                message: format!("the response stream was abandoned because {reason}"),
                failure: Failure::network(),
            });
            break;
        }
    })
}

#[cfg(all(test, feature = "connectivity"))]
mod tests {
    use aither_connectivity::Connectivity;
    use aither_core::{
        LanguageModel,
        llm::{Classify, Failure, FailureKind, LLMRequest, model::Profile as ModelProfile},
    };
    use futures_lite::{StreamExt, future};
    use std::ops::ControlFlow;

    use super::*;
    use crate::Agent;

    #[derive(Debug)]
    struct StallError;

    impl core::fmt::Display for StallError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("stalled")
        }
    }

    impl std::error::Error for StallError {}

    impl Classify for StallError {
        fn failure(&self) -> Failure {
            Failure::permanent()
        }
    }

    /// Streams one chunk of text and then goes silent, like a socket that died
    /// while the machine slept.
    #[derive(Clone)]
    struct StallingLlm;

    impl LanguageModel for StallingLlm {
        type Error = StallError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            futures_lite::stream::once(Ok(Event::text("partial")))
                .chain(futures_lite::stream::pending())
        }

        fn profile(
            &self,
        ) -> impl core::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
            core::future::ready(Ok(ModelProfile::new(
                "mock", "test", "stall", "stall", 200_000,
            )))
        }
    }

    /// Runs one prompt against the stalling model, injects `change` once the
    /// first text arrives, and returns how the run ended.
    async fn run_with(
        change: impl FnOnce(&aither_connectivity::Controller),
    ) -> Result<(), AgentError> {
        let (connectivity, controller) = Connectivity::manual(true);
        let mut agent = Agent::builder(StallingLlm)
            .connectivity(connectivity)
            .build();
        let controller = &controller;
        agent
            .run("hello", std::iter::empty())
            .drive(Some(change), |mut change, timed| async move {
                if matches!(timed.event, crate::AgentEvent::Text(_))
                    && let Some(change) = change.take()
                {
                    change(controller);
                }
                ControlFlow::<std::convert::Infallible, _>::Continue(change)
            })
            .await
            .map(|_| ())
    }

    #[tokio::test]
    async fn waking_mid_stream_ends_the_attempt_with_a_network_failure() {
        let error = run_with(aither_connectivity::Controller::wake)
            .await
            .expect_err("the attempt is abandoned");
        let AgentError::Llm { message, failure } = &error else {
            panic!("expected an LLM error, got {error}");
        };
        assert_eq!(failure.kind, FailureKind::Network);
        assert!(message.contains("woke"), "{message}");
    }

    #[tokio::test]
    async fn going_offline_mid_stream_ends_the_attempt_with_a_network_failure() {
        let error = run_with(|controller| controller.set_online(false))
            .await
            .expect_err("the attempt is abandoned");
        let AgentError::Llm { message, failure } = &error else {
            panic!("expected an LLM error, got {error}");
        };
        assert_eq!(failure.kind, FailureKind::Network);
        assert!(message.contains("offline"), "{message}");
    }

    #[tokio::test]
    async fn sleeping_leaves_the_stream_alone() {
        let (connectivity, controller) = Connectivity::manual(true);
        let stalled = Watch::new(connectivity).guard(Box::pin(
            futures_lite::stream::once(Ok(Event::text("partial")))
                .chain(futures_lite::stream::pending()),
        ));
        futures_lite::pin!(stalled);
        assert!(stalled.next().await.is_some());
        controller.sleep();
        assert!(future::poll_once(stalled.next()).await.is_none());
    }
}
