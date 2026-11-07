//! State: keyed conditions the model is told about only when they change.
//!
//! Memory pressure, machine load, the free space of a volume: each is a
//! condition with one current value, and only the latest value matters. A
//! [`StateSource`] reports those values; the agent keeps, per key, the value
//! the model last saw, and writes an alert only as a request is assembled
//! and only when the current value differs from it. Nothing is written while
//! the agent is idle, so a condition that rises and settles back between two
//! requests reaches the model as nothing at all.
//!
//! A state alert is restated rather than kept: every reassembly, a cold
//! cache's or a compaction's, removes the earlier alerts, and the next
//! request writes the current value again only when it is not the normal
//! one, which the model assumes until told otherwise.

use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;

use super::source::Block;

/// A state whose value differs from the one the model last saw, as the
/// model is to be told it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateChange<'a, K, V> {
    key: &'a K,
    seen: Option<&'a V>,
    now: &'a V,
}

impl<'a, K, V> StateChange<'a, K, V> {
    /// The state that changed.
    #[must_use]
    pub const fn key(&self) -> &'a K {
        self.key
    }

    /// The value the model last saw; `None` when it was never told, or
    /// nothing it was told is still in the context, so it assumes the
    /// normal value.
    #[must_use]
    pub const fn seen(&self) -> Option<&'a V> {
        self.seen
    }

    /// The value now.
    #[must_use]
    pub const fn now(&self) -> &'a V {
        self.now
    }
}

/// Supplies state: keyed conditions with a latest value.
///
/// The agent reads the source only as a request is assembled. It keeps the
/// latest value of every key and the value the model last saw, and appends
/// the alert [`announce`](Self::announce) writes only when the two differ.
/// A return to the value the model already saw writes nothing. After a
/// reassembly the model has seen nothing, so a key is restated only when
/// its value is not the normal one.
///
/// State never starts a run of the agent.
///
/// # Example
///
/// ```
/// use aither_agent::{Block, StateChange, StateSource};
///
/// struct SleepMode(bool);
///
/// impl StateSource for SleepMode {
///     type Key = ();
///     type Value = bool;
///
///     async fn poll(&mut self) -> Vec<((), bool)> {
///         vec![((), self.0)]
///     }
///
///     fn is_normal(&self, (): &(), asleep: &bool) -> bool {
///         !asleep
///     }
///
///     fn announce(&self, change: &StateChange<'_, (), bool>) -> Block {
///         let text = if *change.now() { "The host sleeps." } else { "The host is awake." };
///         Block::new("sleep_mode", text)
///     }
/// }
/// ```
pub trait StateSource: Send + Sync + 'static {
    /// The identity of one state, such as a volume.
    type Key: Eq + Hash + Clone + Send + Sync + 'static;

    /// A state's value. Two values are the same state when they are equal.
    type Value: PartialEq + Clone + Send + Sync + 'static;

    /// The latest value of every state that may have changed since the
    /// last call; a key left out keeps the value it had.
    ///
    /// Called only as a request is assembled, never while the agent is
    /// idle, so a source may drain whatever it collected meanwhile. It must
    /// not block.
    fn poll(&mut self) -> impl Future<Output = Vec<(Self::Key, Self::Value)>> + Send;

    /// Whether `value` is the normal value of the state `key`: the one the
    /// model assumes while it has been told nothing.
    fn is_normal(&self, key: &Self::Key, value: &Self::Value) -> bool;

    /// The alert telling the model of `change`, appended to the
    /// conversation once and removed at the next reassembly.
    fn announce(&self, change: &StateChange<'_, Self::Key, Self::Value>) -> Block;
}

/// One state: its latest value and the one the model last saw.
struct Entry<K, V> {
    key: K,
    now: V,
    seen: Option<V>,
}

/// A [`StateSource`] with the values the model has seen.
struct Tracked<S: StateSource> {
    source: S,
    /// In the order the keys first appeared, so alerts come in a stable
    /// order.
    entries: Vec<Entry<S::Key, S::Value>>,
}

impl<S: StateSource> Tracked<S> {
    fn record(&mut self, key: S::Key, value: S::Value) {
        match self.entries.iter_mut().find(|entry| entry.key == key) {
            Some(entry) => entry.now = value,
            None => self.entries.push(Entry {
                key,
                now: value,
                seen: None,
            }),
        }
    }

    /// The alerts for every state the model does not know as it stands.
    fn alerts(&mut self, reassembled: bool) -> Vec<Block> {
        let mut alerts = Vec::new();
        for entry in &mut self.entries {
            if reassembled {
                entry.seen = None;
            }
            let known = entry.seen.as_ref().map_or_else(
                || self.source.is_normal(&entry.key, &entry.now),
                |seen| *seen == entry.now,
            );
            if known {
                continue;
            }
            alerts.push(self.source.announce(&StateChange {
                key: &entry.key,
                seen: entry.seen.as_ref(),
                now: &entry.now,
            }));
            entry.seen = Some(entry.now.clone());
        }
        alerts
    }
}

/// Object-safe twin of [`StateSource`], with the agent's bookkeeping.
trait StateSourceImpl: Send + Sync {
    fn on_request<'a>(
        &'a mut self,
        reassembled: bool,
    ) -> Pin<Box<dyn Future<Output = Vec<Block>> + Send + 'a>>;
}

impl<S: StateSource> StateSourceImpl for Tracked<S> {
    fn on_request<'a>(
        &'a mut self,
        reassembled: bool,
    ) -> Pin<Box<dyn Future<Output = Vec<Block>> + Send + 'a>> {
        Box::pin(async move {
            for (key, value) in self.source.poll().await {
                self.record(key, value);
            }
            self.alerts(reassembled)
        })
    }
}

/// A type-erased [`StateSource`] together with the values the model has
/// seen, so an agent can hold a heterogeneous set.
pub struct AnyStateSource {
    inner: Box<dyn StateSourceImpl>,
}

impl std::fmt::Debug for AnyStateSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnyStateSource")
            .finish_non_exhaustive()
    }
}

impl AnyStateSource {
    /// Erases the concrete type of `source`; the model has seen nothing of
    /// it yet.
    #[must_use]
    pub fn new<S: StateSource>(source: S) -> Self {
        Self {
            inner: Box::new(Tracked {
                source,
                entries: Vec::new(),
            }),
        }
    }

    /// Reads the source and returns the alerts for the states whose value
    /// differs from what the model last saw, which it then has seen.
    ///
    /// `reassembled` is `true` when the window was rewritten since the
    /// previous request, which removed every earlier state alert.
    pub(crate) fn on_request(
        &mut self,
        reassembled: bool,
    ) -> impl Future<Output = Vec<Block>> + Send + '_ {
        self.inner.on_request(reassembled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disk whose free-space level a test sets; `0` is normal.
    struct Disk {
        levels: async_channel::Receiver<u8>,
    }

    impl StateSource for Disk {
        type Key = &'static str;
        type Value = u8;

        fn poll(&mut self) -> impl Future<Output = Vec<(&'static str, u8)>> + Send {
            std::future::ready(
                std::iter::from_fn(|| self.levels.try_recv().ok())
                    .map(|level| ("/work", level))
                    .collect(),
            )
        }

        fn is_normal(&self, _: &&'static str, level: &u8) -> bool {
            *level == 0
        }

        fn announce(&self, change: &StateChange<'_, &'static str, u8>) -> Block {
            Block::new(
                "disk",
                format!("{} {:?} -> {}", change.key(), change.seen(), change.now()),
            )
        }
    }

    fn disk() -> (async_channel::Sender<u8>, AnyStateSource) {
        let (sender, levels) = async_channel::unbounded();
        (sender, AnyStateSource::new(Disk { levels }))
    }

    fn texts(blocks: &[Block]) -> Vec<&str> {
        blocks.iter().map(Block::content).collect()
    }

    const NONE: [&str; 0] = [];

    /// The normal value is what the model assumes: nothing is said of it,
    /// a change from it is said once, and only the latest value counts.
    #[tokio::test]
    async fn only_a_value_the_model_does_not_know_is_announced() {
        let (levels, mut source) = disk();
        levels.send(0).await.expect("open");
        assert_eq!(texts(&source.on_request(false).await), NONE);

        levels.send(1).await.expect("open");
        levels.send(2).await.expect("open");
        assert_eq!(texts(&source.on_request(false).await), ["/work None -> 2"]);
        assert_eq!(texts(&source.on_request(false).await), NONE, "said once");

        levels.send(0).await.expect("open");
        assert_eq!(
            texts(&source.on_request(false).await),
            ["/work Some(2) -> 0"]
        );
    }

    /// A value that changes and comes back between two requests is no
    /// news.
    #[tokio::test]
    async fn a_return_to_the_value_the_model_saw_says_nothing() {
        let (levels, mut source) = disk();
        levels.send(2).await.expect("open");
        assert_eq!(source.on_request(false).await.len(), 1);
        levels.send(0).await.expect("open");
        levels.send(2).await.expect("open");
        assert_eq!(texts(&source.on_request(false).await), NONE);
    }

    /// After a reassembly the model knows nothing: a value that is not
    /// normal is restated, a normal one is not.
    #[tokio::test]
    async fn a_reassembly_restates_only_a_value_that_is_not_normal() {
        let (levels, mut source) = disk();
        levels.send(2).await.expect("open");
        assert_eq!(source.on_request(false).await.len(), 1);
        assert_eq!(texts(&source.on_request(true).await), ["/work None -> 2"]);

        levels.send(0).await.expect("open");
        assert_eq!(source.on_request(false).await.len(), 1);
        assert_eq!(texts(&source.on_request(true).await), NONE);
    }
}
