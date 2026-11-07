//! What a row's blocker still waits on, as the host knows it.

use std::pin::Pin;
use std::sync::Arc;

/// Decides whether a row's blocker still waits on something live.
///
/// A row names its blocker through [`DocumentRow::blocked_on`]; the host
/// resolves it: `true` while it names something still open — an unanswered
/// question's request id, or a free-text reason the host cannot check.
/// `false` means the blocker is dead — its question was answered or
/// withdrawn — and its row is stale. An error says nothing either way: the
/// blocker keeps holding its row.
///
/// [`DocumentRow::blocked_on`]: super::DocumentRow::blocked_on
pub trait BlockerResolver: Send + Sync + 'static {
    /// Whether `blocker` still names something the run waits on.
    fn is_open(&self, blocker: &str) -> impl Future<Output = Result<bool, BlockerError>> + Send;
}

/// Why a blocker could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BlockerError {
    /// The host's record of what the blocker names could not be read.
    #[error("the blocker {blocker} cannot be resolved: {reason}")]
    Unavailable {
        /// The blocker, as the row names it.
        blocker: String,
        /// What went wrong.
        reason: String,
    },
}

type IsOpen<'a> = Pin<Box<dyn Future<Output = Result<bool, BlockerError>> + Send + 'a>>;

/// Object-safe twin of [`BlockerResolver`]; boxing lives inside
/// [`AnyBlockerResolver`].
trait BlockerResolverImpl: Send + Sync {
    fn is_open<'a>(&'a self, blocker: &'a str) -> IsOpen<'a>;
}

impl<T: BlockerResolver> BlockerResolverImpl for T {
    fn is_open<'a>(&'a self, blocker: &'a str) -> IsOpen<'a> {
        Box::pin(BlockerResolver::is_open(self, blocker))
    }
}

/// A type-erased [`BlockerResolver`] carried by the agent.
#[derive(Clone)]
pub struct AnyBlockerResolver(Arc<dyn BlockerResolverImpl>);

impl AnyBlockerResolver {
    /// Wraps `resolver`.
    #[must_use]
    pub fn new(resolver: impl BlockerResolver) -> Self {
        Self(Arc::new(resolver))
    }

    /// Whether `blocker` still names something the run waits on.
    pub fn is_open<'a>(
        &'a self,
        blocker: &'a str,
    ) -> impl Future<Output = Result<bool, BlockerError>> + Send + 'a {
        self.0.is_open(blocker)
    }
}

impl std::fmt::Debug for AnyBlockerResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyBlockerResolver").finish_non_exhaustive()
    }
}
