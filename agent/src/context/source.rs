//! Composable providers of per-request context.
//!
//! Working documents, running background jobs and active skills all follow
//! the same shape: read some live state, render it into the tail of the next
//! request, and occasionally insist on something durable — a nag that a
//! checklist has gone stale, or a refusal to let the run end with unchecked
//! items. Hard-coding each of them into the agent loop made that loop the
//! place every new kind of ambient context had to be threaded through.
//! A [`ContextSource`] is that contract, so the loop knows only the trait and
//! hosts register whatever else they need with
//! [`AgentBuilder::context_source`](crate::AgentBuilder::context_source).
//!
//! Every source renders into the *tail* of the request, never into the middle
//! of it. The prefix a provider has cached stays byte-identical as the
//! conversation grows, and content that changes on every single request is
//! the last thing the model reads.

use std::future::Future;
use std::pin::Pin;

use serde::Serialize;

use super::window::serialize_xml;

/// A single piece of rendered context, identified by its XML tag.
///
/// The tag is both the delimiter the model sees and the identity the agent
/// uses to replace a block it rendered earlier, so it must be stable for a
/// given source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    tag: String,
    content: String,
}

impl Block {
    /// Creates a block whose `content` is already rendered.
    #[must_use]
    pub fn new(tag: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            content: content.into(),
        }
    }

    /// Creates a block by serializing `value` into an XML element named `tag`.
    #[must_use]
    pub fn xml<T: Serialize>(tag: impl Into<String>, value: &T) -> Self {
        let tag = tag.into();
        let content = serialize_xml(&tag, value);
        Self { tag, content }
    }

    /// The block's stable identity.
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// The rendered content sent to the model.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// The rendered length in characters, for window accounting.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.content.len()
    }

    /// Whether the block renders to nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Supplies live context to every request the agent sends.
///
/// Implementations are polled at a request boundary, so they may read files
/// or query registries; they must not block.
///
/// # Example
///
/// ```
/// use aither_agent::{Block, ContextSource};
///
/// struct Clock;
///
/// impl ContextSource for Clock {
///     async fn render(&self) -> Option<Block> {
///         Some(Block::new("now", "2025-01-01T00:00:00Z"))
///     }
/// }
/// ```
pub trait ContextSource: Send + Sync + 'static {
    /// Renders this source's contribution to the next request's tail.
    ///
    /// Returning `None` omits the source entirely — the right answer when
    /// there is nothing to say, since an empty block still costs tokens and
    /// still invites the model to reason about it.
    fn render(&self) -> impl Future<Output = Option<Block>> + Send;

    /// Reports the block that was actually sent.
    ///
    /// Called once per request, after the tail has been assembled, which is
    /// the only point at which a source can know what the model has seen and
    /// therefore what counts as a change next time.
    fn commit(&mut self, block: &Block) -> impl Future<Output = ()> + Send {
        let _ = block;
        async {}
    }

    /// Produces a durable alert after a turn that executed tools.
    ///
    /// Unlike [`render`](Self::render), the returned block is appended to the
    /// conversation and stays there: it survives into later requests instead
    /// of being regenerated, which is what makes it usable for one-shot
    /// corrections.
    fn on_tool_turn_end(&mut self) -> impl Future<Output = Option<Block>> + Send {
        async { None }
    }

    /// Produces a durable alert when the model is about to end the run.
    ///
    /// Returning `Some` forces another turn, so a source may use this to
    /// refuse an early finish — for instance while a checklist still has
    /// unchecked items.
    fn on_final_turn(&mut self) -> impl Future<Output = Option<Block>> + Send {
        async { None }
    }
}

/// Object-safe twin of [`ContextSource`].
trait ContextSourceImpl: Send + Sync {
    fn render<'a>(&'a self) -> Pin<Box<dyn Future<Output = Option<Block>> + Send + 'a>>;

    fn commit<'a>(&'a mut self, block: &'a Block) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    fn on_tool_turn_end<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Option<Block>> + Send + 'a>>;

    fn on_final_turn<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Option<Block>> + Send + 'a>>;
}

impl<T: ContextSource> ContextSourceImpl for T {
    fn render<'a>(&'a self) -> Pin<Box<dyn Future<Output = Option<Block>> + Send + 'a>> {
        Box::pin(ContextSource::render(self))
    }

    fn commit<'a>(&'a mut self, block: &'a Block) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(ContextSource::commit(self, block))
    }

    fn on_tool_turn_end<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Option<Block>> + Send + 'a>> {
        Box::pin(ContextSource::on_tool_turn_end(self))
    }

    fn on_final_turn<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Option<Block>> + Send + 'a>> {
        Box::pin(ContextSource::on_final_turn(self))
    }
}

/// A type-erased [`ContextSource`], so an agent can hold a heterogeneous set.
///
/// Boxing lives entirely inside this wrapper: the methods it exposes return
/// `impl Future` exactly like the trait they stand in for.
pub struct AnyContextSource {
    inner: Box<dyn ContextSourceImpl>,
}

impl std::fmt::Debug for AnyContextSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("AnyContextSource").finish()
    }
}

impl AnyContextSource {
    /// Erases the concrete type of `source`.
    #[must_use]
    pub fn new<T: ContextSource>(source: T) -> Self {
        Self {
            inner: Box::new(source),
        }
    }

    /// See [`ContextSource::render`].
    pub fn render(&self) -> impl Future<Output = Option<Block>> + Send + '_ {
        self.inner.render()
    }

    /// See [`ContextSource::commit`].
    pub fn commit<'a>(&'a mut self, block: &'a Block) -> impl Future<Output = ()> + Send + 'a {
        self.inner.commit(block)
    }

    /// See [`ContextSource::on_tool_turn_end`].
    pub fn on_tool_turn_end(&mut self) -> impl Future<Output = Option<Block>> + Send + '_ {
        self.inner.on_tool_turn_end()
    }

    /// See [`ContextSource::on_final_turn`].
    pub fn on_final_turn(&mut self) -> impl Future<Output = Option<Block>> + Send + '_ {
        self.inner.on_final_turn()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Counter {
        committed: Vec<String>,
    }

    impl ContextSource for Counter {
        fn render(&self) -> impl Future<Output = Option<Block>> + Send {
            std::future::ready(Some(Block::new(
                "counter",
                self.committed.len().to_string(),
            )))
        }

        async fn commit(&mut self, block: &Block) {
            self.committed.push(block.content().to_string());
        }

        fn on_tool_turn_end(&mut self) -> impl Future<Output = Option<Block>> + Send {
            std::future::ready(Some(Block::new("counter_alert", "tool turn observed")))
        }
    }

    #[tokio::test]
    async fn erased_source_forwards_every_method() {
        let mut source = AnyContextSource::new(Counter::default());
        let block = source.render().await.expect("counter renders");
        assert_eq!(block.content(), "0");
        source.commit(&block).await;
        let next = source.render().await.expect("counter renders");
        assert_eq!(next.content(), "1");
        assert_eq!(
            source.on_tool_turn_end().await.map(|b| b.tag().to_string()),
            Some("counter_alert".to_string())
        );
        assert!(source.on_final_turn().await.is_none());
    }

    #[test]
    fn xml_blocks_wrap_their_payload_in_the_tag() {
        #[derive(Serialize)]
        struct Payload {
            #[serde(rename = "$text")]
            text: &'static str,
        }

        let block = Block::xml("tasks_document", &Payload { text: "- [ ] work" });
        assert_eq!(block.tag(), "tasks_document");
        assert!(block.content().starts_with("<tasks_document>"));
        assert!(block.content().contains("- [ ] work"));
    }
}
