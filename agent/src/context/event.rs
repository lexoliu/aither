//! Events: things that happened, each told to the model once.
//!
//! A command finishing, processes killed with a foreground command, a job
//! killed for its memory: an event is history, not a condition, so it is
//! never merged with a later one and never restated. An [`EventSource`]
//! hands its events to the agent as a request is assembled; each becomes one
//! alert in the conversation. The alert stays until a request has carried
//! it, so a reassembly or compaction before that keeps it, and drops it
//! after, never writing it again.

use std::future::Future;
use std::pin::Pin;

use super::source::Block;

/// Supplies events: each delivered to the model once.
///
/// Events never start a run of the agent; they reach the model with the
/// next request.
///
/// # Example
///
/// ```
/// use aither_agent::{Block, EventSource};
///
/// struct Deploys(async_channel::Receiver<String>);
///
/// impl EventSource for Deploys {
///     type Event = String;
///
///     async fn take(&mut self) -> Vec<String> {
///         std::iter::from_fn(|| self.0.try_recv().ok()).collect()
///     }
///
///     fn announce(&self, version: &String) -> Block {
///         Block::new("deployed", version.clone())
///     }
/// }
/// ```
pub trait EventSource: Send + Sync + 'static {
    /// One event.
    type Event: Send + 'static;

    /// The events raised since the last call, oldest first.
    ///
    /// Called as each request is assembled. It must not block.
    fn take(&mut self) -> impl Future<Output = Vec<Self::Event>> + Send;

    /// The alert telling the model of `event`.
    fn announce(&self, event: &Self::Event) -> Block;
}

/// Object-safe twin of [`EventSource`].
trait EventSourceImpl: Send + Sync {
    fn deliver<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Vec<Block>> + Send + 'a>>;
}

impl<S: EventSource> EventSourceImpl for S {
    fn deliver<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Vec<Block>> + Send + 'a>> {
        Box::pin(async move {
            let events = self.take().await;
            events.iter().map(|event| self.announce(event)).collect()
        })
    }
}

/// A type-erased [`EventSource`], so an agent can hold a heterogeneous set.
pub struct AnyEventSource {
    inner: Box<dyn EventSourceImpl>,
}

impl std::fmt::Debug for AnyEventSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnyEventSource")
            .finish_non_exhaustive()
    }
}

impl AnyEventSource {
    /// Erases the concrete type of `source`.
    #[must_use]
    pub fn new<S: EventSource>(source: S) -> Self {
        Self {
            inner: Box::new(source),
        }
    }

    /// The alerts for the events raised since the last call, oldest first.
    pub(crate) fn deliver(&mut self) -> impl Future<Output = Vec<Block>> + Send + '_ {
        self.inner.deliver()
    }
}
