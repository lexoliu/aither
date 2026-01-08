//! Streaming agent execution.
//!
//! Provides a `Stream` implementation for observing agent execution
//! in real-time.

use std::pin::Pin;
use std::task::{Context, Poll};

use aither_core::LanguageModel;
use futures_core::Stream;

use crate::agent::Agent;
use crate::error::AgentError;
use crate::event::AgentEvent;
use crate::hook::Hook;

/// Stream of agent events during execution.
///
/// This is returned by [`Agent::run`] and yields `AgentEvent`s as the
/// agent processes a prompt.
///
/// # Example
///
/// ```rust,ignore
/// use futures::StreamExt;
///
/// let mut stream = agent.run("Do something");
/// while let Some(event) = stream.next().await {
///     match event? {
///         AgentEvent::Text(t) => print!("{}", t),
///         AgentEvent::Complete { .. } => break,
///         _ => {}
///     }
/// }
/// ```
pub struct AgentStream<'a, Advanced, Balanced, Fast, H> {
    agent: &'a mut Agent<Advanced, Balanced, Fast, H>,
    prompt: String,
    state: StreamState,
}

/// Internal state for the stream.
#[derive(Debug)]
enum StreamState {
    /// Initial state - not yet started.
    Initial,
    /// Running the agent loop.
    Running,
    /// Completed successfully or with error.
    Done,
}

impl<'a, Advanced, Balanced, Fast, H> AgentStream<'a, Advanced, Balanced, Fast, H>
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    /// Creates a new agent stream.
    pub(crate) fn new(agent: &'a mut Agent<Advanced, Balanced, Fast, H>, prompt: String) -> Self {
        Self {
            agent,
            prompt,
            state: StreamState::Initial,
        }
    }

    /// Collects all events and returns the final result.
    ///
    /// This is equivalent to consuming the stream and extracting
    /// the final text from the `Complete` event.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent fails.
    pub async fn collect(mut self) -> Result<String, AgentError> {
        self.agent.query(&self.prompt).await
    }
}

impl<Advanced, Balanced, Fast, H> Stream for AgentStream<'_, Advanced, Balanced, Fast, H>
where
    Advanced: LanguageModel + Unpin,
    Balanced: LanguageModel + Unpin,
    Fast: LanguageModel + Unpin,
    H: Hook + Unpin,
{
    type Item = Result<AgentEvent, AgentError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        match this.state {
            StreamState::Initial => {
                this.state = StreamState::Running;
                // For now, we don't implement true streaming
                // This is a placeholder that will be enhanced later
                Poll::Pending
            }
            StreamState::Running => {
                // Streaming not fully implemented yet
                // Return Pending to indicate async work needed
                Poll::Pending
            }
            StreamState::Done => Poll::Ready(None),
        }
    }
}

/// A simple blocking stream adapter for testing.
///
/// This runs the agent to completion and yields events.
#[cfg(test)]
pub struct BlockingStream<'a, Advanced, Balanced, Fast, H> {
    agent: &'a mut Agent<Advanced, Balanced, Fast, H>,
    prompt: String,
    completed: bool,
    result: Option<Result<String, AgentError>>,
}

#[cfg(test)]
impl<'a, Advanced, Balanced, Fast, H> BlockingStream<'a, Advanced, Balanced, Fast, H>
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    pub fn new(agent: &'a mut Agent<Advanced, Balanced, Fast, H>, prompt: String) -> Self {
        Self {
            agent,
            prompt,
            completed: false,
            result: None,
        }
    }

    pub async fn run(&mut self) -> Option<Result<AgentEvent, AgentError>> {
        if self.completed {
            return None;
        }

        if self.result.is_none() {
            self.result = Some(self.agent.query(&self.prompt).await);
        }

        self.completed = true;

        match self.result.take() {
            Some(Ok(text)) => Some(Ok(AgentEvent::complete(text, 1))),
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }
}
