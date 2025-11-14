use crate::{Error, Result as CoreResult};
use alloc::{
    collections::VecDeque,
    string::{String, ToString},
    vec::Vec,
};
use core::{
    future::Future,
    mem,
    pin::Pin,
    task::{Context, Poll},
};
use futures_core::Stream;
use pin_project_lite::pin_project;

use super::LLMResponse;

/// Batched delta emitted by providers when streaming responses.
///
/// Providers can push both user-visible text chunks and reasoning summaries.
/// Construct using [`ResponseChunk::default`] and [`ResponseChunk::push_text`]/[`ResponseChunk::push_reasoning`].
#[derive(Debug, Default)]
pub struct ResponseChunk {
    text: Vec<String>,
    reasoning: Vec<String>,
}

impl ResponseChunk {
    /// Creates a response chunk containing text.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        let mut chunk = Self::default();
        chunk.push_text(text);
        chunk
    }

    /// Pushes a visible text segment if it is non-empty.
    pub fn push_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        if !text.is_empty() {
            self.text.push(text);
        }
    }

    /// Pushes a reasoning entry if it is non-empty.
    pub fn push_reasoning(&mut self, reasoning: impl Into<String>) {
        let reasoning = reasoning.into();
        if !reasoning.is_empty() {
            self.reasoning.push(reasoning);
        }
    }

    fn take(self) -> (Vec<String>, Vec<String>) {
        (self.text, self.reasoning)
    }

    const fn is_empty(&self) -> bool {
        self.text.is_empty() && self.reasoning.is_empty()
    }
}

pin_project! {
    /// [`LLMResponse`] implementation that multiplexes visible text and reasoning steps.
    ///
    /// Providers feed a base stream of [`ResponseChunk`]s. This adapter exposes the text stream via
    /// [`Stream`] while [`LLMResponse::poll_reasoning_next`] yields reasoning summaries.
    pub struct ReasoningStream<S, E> {
        #[pin]
        inner: S,
        text_buffer: VecDeque<String>,
        reasoning_buffer: VecDeque<CoreResult>,
        collected: String,
        stream_error: Option<E>,
        reasoning_error: Option<String>,
        finished: bool,
    }
}

impl<S, E> ReasoningStream<S, E>
where
    S: Stream<Item = Result<ResponseChunk, E>>,
    E: core::error::Error,
{
    /// Creates a new reasoning stream adapter.
    #[must_use]
    pub const fn new(stream: S) -> Self {
        Self {
            inner: stream,
            text_buffer: VecDeque::new(),
            reasoning_buffer: VecDeque::new(),
            collected: String::new(),
            stream_error: None,
            reasoning_error: None,
            finished: false,
        }
    }

    fn poll_driver(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut this = self.project();
        loop {
            if *this.finished || this.stream_error.is_some() {
                return Poll::Ready(());
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    if chunk.is_empty() {
                        continue;
                    }
                    // SAFETY: project() returns short-lived references.
                    let (texts, reasoning) = chunk.take();
                    for text in texts {
                        this.collected.push_str(&text);
                        this.text_buffer.push_back(text);
                    }
                    for reason in reasoning {
                        this.reasoning_buffer.push_back(Ok(reason));
                    }
                    return Poll::Ready(());
                }
                Poll::Ready(Some(Err(err))) => {
                    *this.reasoning_error = Some(err.to_string());
                    *this.stream_error = Some(err);
                    return Poll::Ready(());
                }
                Poll::Ready(None) => {
                    *this.finished = true;
                    return Poll::Ready(());
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<S, E> Stream for ReasoningStream<S, E>
where
    S: Stream<Item = Result<ResponseChunk, E>>,
    E: core::error::Error,
{
    type Item = Result<String, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let this = self.as_mut().project();
            if let Some(text) = this.text_buffer.pop_front() {
                return Poll::Ready(Some(Ok(text)));
            }

            if let Some(err) = this.stream_error.take() {
                return Poll::Ready(Some(Err(err)));
            }

            if *this.finished {
                return Poll::Ready(None);
            }

            match self.as_mut().poll_driver(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(()) => (),
            }
        }
    }
}

impl<S, E> LLMResponse for ReasoningStream<S, E>
where
    S: Stream<Item = Result<ResponseChunk, E>> + Send,
    E: core::error::Error + Send + Sync + 'static,
{
    type Error = E;

    fn poll_reasoning_next(self: Pin<&mut Self>) -> Poll<Option<CoreResult>> {
        let this = self.project();
        if let Some(reason) = this.reasoning_buffer.pop_front() {
            return Poll::Ready(Some(reason));
        }

        if let Some(msg) = this.reasoning_error.take() {
            return Poll::Ready(Some(Err(Error::msg(msg))));
        }

        if *this.finished {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

pin_project! {
    /// Future returned when awaiting a [`ReasoningStream`].
    pub struct ReasoningFuture<S, E> {
        #[pin]
        response: Option<ReasoningStream<S, E>>,
    }
}

impl<S, E> Future for ReasoningFuture<S, E>
where
    S: Stream<Item = Result<ResponseChunk, E>> + Send,
    E: core::error::Error,
{
    type Output = Result<String, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        let mut response = this.response.as_mut().as_pin_mut().map_or_else(
            || panic!("response future already completed"),
            |stream| stream,
        );

        loop {
            match response.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(_))) => {}
                Poll::Ready(Some(Err(err))) => {
                    this.response.set(None);
                    return Poll::Ready(Err(err));
                }
                Poll::Ready(None) => {
                    let projection = response.as_mut().project();
                    let collected = mem::take(projection.collected);
                    this.response.set(None);
                    return Poll::Ready(Ok(collected));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<S, E> IntoFuture for ReasoningStream<S, E>
where
    S: Stream<Item = Result<ResponseChunk, E>> + Send,
    E: core::error::Error,
{
    type Output = Result<String, E>;
    type IntoFuture = ReasoningFuture<S, E>;

    fn into_future(self) -> Self::IntoFuture {
        ReasoningFuture {
            response: Some(self),
        }
    }
}
