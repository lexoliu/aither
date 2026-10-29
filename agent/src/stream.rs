//! Streaming agent execution.
//!
//! Provides a `Stream` implementation for observing agent execution
//! in real-time with true streaming of text chunks.

use std::pin::Pin;
use std::task::{Context, Poll};

use crate::error::AgentError;
use crate::event::TimedEvent;
use futures_core::Stream;

/// Stream of agent events during execution.
///
/// This stream yields `TimedEvent`s as they happen, including
/// individual text chunks for real-time display.
///
/// # Example
///
/// ```rust,ignore
/// use futures::StreamExt;
///
/// let mut stream = agent.run("Do something");
/// while let Some(event) = stream.next().await {
///     let event = event?;
///     match event.event {
///         AgentEvent::Text(t) => print!("{}", t),
///         AgentEvent::Complete { .. } => break,
///         _ => {}
///     }
/// }
/// ```
pub struct AgentStream<S> {
    inner: Pin<Box<S>>,
}

impl<S> AgentStream<S>
where
    S: Stream<Item = Result<TimedEvent, AgentError>>,
{
    /// Creates a new agent stream wrapping an inner stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }
}

impl<S> Stream for AgentStream<S>
where
    S: Stream<Item = Result<TimedEvent, AgentError>>,
{
    type Item = Result<TimedEvent, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl<S> std::fmt::Debug for AgentStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentStream").finish()
    }
}
