//! Where the issues and pull requests `tasks.md` names live.
//!
//! The harness knows no forge: the host implements [`WorkTracker`] over its
//! own issue and pull-request backends, and the tasks document asks it what
//! state an item is in and what hangs under a root.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;

use super::reference::{WorkKind, WorkRef};

/// How many items a lookup fetches at once.
const FETCH_CONCURRENCY: usize = 8;

/// The state of an issue or pull request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkState {
    /// Open: an issue still to do, a pull request not yet landed.
    Open,
    /// Closed without landing.
    Closed,
    /// A pull request that landed.
    Merged,
}

impl WorkState {
    /// Whether the item is finished, so a row naming it is stale.
    #[must_use]
    pub const fn is_done(self) -> bool {
        matches!(self, Self::Closed | Self::Merged)
    }
}

impl fmt::Display for WorkState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Open => "open",
            Self::Closed => "closed",
            Self::Merged => "merged",
        })
    }
}

/// An item related to another, in compact form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedWork {
    /// The related item.
    pub reference: WorkRef,
    /// Its title.
    pub title: String,
    /// Its state.
    pub state: WorkState,
}

/// One comment on an item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkComment {
    /// Who wrote it, when the backend records authors.
    pub author: Option<String>,
    /// Its text.
    pub body: String,
}

/// An issue or pull request in full.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkItem {
    /// The item.
    pub reference: WorkRef,
    /// Its title.
    pub title: String,
    /// Its state.
    pub state: WorkState,
    /// Its body.
    pub body: String,
    /// Its comments, oldest first.
    pub comments: Vec<WorkComment>,
    /// Its sub-issues.
    pub sub_issues: Vec<RelatedWork>,
    /// The items it is blocked by.
    pub blocked_by: Vec<RelatedWork>,
}

impl WorkItem {
    /// Whether an item it is blocked by is still open.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.blocked_by
            .iter()
            .any(|blocker| blocker.state == WorkState::Open)
    }
}

/// Why an item could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkTrackerError {
    /// The item does not exist.
    #[error("{0} does not exist")]
    NotFound(WorkRef),
    /// The tracker cannot read the item: not this session's repository, no
    /// backend for its kind, or the backend failed.
    #[error("{reference} cannot be read: {reason}")]
    Unavailable {
        /// The item.
        reference: WorkRef,
        /// What went wrong, for the model to read.
        reason: String,
    },
}

/// Reads the issues and pull requests `tasks.md` names.
pub trait WorkTracker: Send + Sync + 'static {
    /// The item `reference` names, with its relations.
    fn item(
        &self,
        reference: &WorkRef,
    ) -> impl Future<Output = Result<WorkItem, WorkTrackerError>> + Send;
}

/// Object-safe twin of [`WorkTracker`]; boxing lives inside
/// [`AnyWorkTracker`].
trait WorkTrackerImpl: Send + Sync {
    fn item<'a>(
        &'a self,
        reference: &'a WorkRef,
    ) -> Pin<Box<dyn Future<Output = Result<WorkItem, WorkTrackerError>> + Send + 'a>>;
}

impl<T: WorkTracker> WorkTrackerImpl for T {
    fn item<'a>(
        &'a self,
        reference: &'a WorkRef,
    ) -> Pin<Box<dyn Future<Output = Result<WorkItem, WorkTrackerError>> + Send + 'a>> {
        Box::pin(WorkTracker::item(self, reference))
    }
}

/// A type-erased [`WorkTracker`] carried by the agent.
#[derive(Clone)]
pub struct AnyWorkTracker(Arc<dyn WorkTrackerImpl>);

impl AnyWorkTracker {
    /// Wraps `tracker`.
    #[must_use]
    pub fn new(tracker: impl WorkTracker) -> Self {
        Self(Arc::new(tracker))
    }

    /// See [`WorkTracker::item`].
    pub fn item<'a>(
        &'a self,
        reference: &'a WorkRef,
    ) -> impl Future<Output = Result<WorkItem, WorkTrackerError>> + Send + 'a {
        self.0.item(reference)
    }

    /// Fetches every one of `references`, at most [`FETCH_CONCURRENCY`] at
    /// a time, in the order given.
    pub(crate) async fn items(
        &self,
        references: Vec<WorkRef>,
    ) -> Vec<(WorkRef, Result<WorkItem, WorkTrackerError>)> {
        futures::stream::iter(references)
            .map(|reference| async move {
                let item = self.item(&reference).await;
                (reference, item)
            })
            .buffered(FETCH_CONCURRENCY)
            .collect()
            .await
    }

    /// Fetches each of `references` once, keyed by reference.
    pub(crate) async fn lookup(
        &self,
        references: impl IntoIterator<Item = WorkRef>,
    ) -> HashMap<WorkRef, Result<WorkItem, WorkTrackerError>> {
        let mut unique = Vec::new();
        for reference in references {
            if !unique.contains(&reference) {
                unique.push(reference);
            }
        }
        self.items(unique).await.into_iter().collect()
    }

    /// Resolves each of `roots` into itself, its sub-issues and the items it
    /// is blocked by, recursively, each item once.
    pub(crate) async fn trees(&self, roots: Vec<WorkRef>) -> IssueTrees {
        let mut trees = IssueTrees::default();
        let mut visited: HashSet<WorkRef> = roots.iter().cloned().collect();
        let mut frontier = roots;
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for (reference, item) in self.items(std::mem::take(&mut frontier)).await {
                match item {
                    Ok(item) => {
                        for related in item.sub_issues.iter().chain(&item.blocked_by) {
                            if visited.insert(related.reference.clone()) {
                                next.push(related.reference.clone());
                            }
                        }
                        trees.items.push(item);
                    }
                    Err(error) => {
                        tracing::warn!(%reference, %error, "an issue tree item could not be read");
                        trees.unresolved.push(error);
                    }
                }
            }
            frontier = next;
        }
        trees
    }
}

impl fmt::Debug for AnyWorkTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyWorkTracker").finish_non_exhaustive()
    }
}

/// The items under some roots, in the order they were reached.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IssueTrees {
    /// Every item read.
    pub items: Vec<WorkItem>,
    /// Every item that could not be read, with the reason.
    pub unresolved: Vec<WorkTrackerError>,
}

impl IssueTrees {
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty() && self.unresolved.is_empty()
    }
}

/// The kind word a model reads for `reference`.
pub const fn kind_label(reference: &WorkRef) -> &'static str {
    match reference.kind() {
        WorkKind::Issue => "Issue",
        WorkKind::PullRequest => "Pull request",
    }
}

#[cfg(test)]
pub mod test_tracker {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// A tracker over a fixed set of items, whose states a test can change.
    #[derive(Clone, Default)]
    pub struct FixedTracker {
        items: Arc<Mutex<HashMap<WorkRef, WorkItem>>>,
    }

    impl FixedTracker {
        pub fn with(self, item: WorkItem) -> Self {
            self.items
                .lock()
                .expect("tracker lock")
                .insert(item.reference.clone(), item);
            self
        }

        pub fn set_state(&self, reference: &WorkRef, state: WorkState) {
            self.items
                .lock()
                .expect("tracker lock")
                .get_mut(reference)
                .expect("known item")
                .state = state;
        }
    }

    impl WorkTracker for FixedTracker {
        fn item(
            &self,
            reference: &WorkRef,
        ) -> impl Future<Output = Result<WorkItem, WorkTrackerError>> + Send {
            let found = self
                .items
                .lock()
                .expect("tracker lock")
                .get(reference)
                .cloned()
                .ok_or_else(|| WorkTrackerError::NotFound(reference.clone()));
            std::future::ready(found)
        }
    }

    pub fn related(reference: WorkRef, state: WorkState) -> RelatedWork {
        RelatedWork {
            title: format!("title of {reference}"),
            reference,
            state,
        }
    }

    pub fn item(reference: WorkRef, state: WorkState) -> WorkItem {
        WorkItem {
            title: format!("title of {reference}"),
            body: format!("body of {reference}"),
            reference,
            state,
            comments: Vec::new(),
            sub_issues: Vec::new(),
            blocked_by: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_tracker::{FixedTracker, item, related};
    use super::*;

    #[tokio::test]
    async fn a_tree_reaches_sub_issues_and_blockers_recursively_once_each() {
        let mut root = item(WorkRef::issue(1), WorkState::Open);
        root.sub_issues = vec![related(WorkRef::issue(2), WorkState::Open)];
        root.blocked_by = vec![related(WorkRef::issue(3), WorkState::Open)];
        let mut child = item(WorkRef::issue(2), WorkState::Open);
        child.sub_issues = vec![
            related(WorkRef::issue(4), WorkState::Closed),
            related(WorkRef::issue(1), WorkState::Open),
        ];
        let tracker = AnyWorkTracker::new(
            FixedTracker::default()
                .with(root)
                .with(child)
                .with(item(WorkRef::issue(3), WorkState::Open))
                .with(item(WorkRef::issue(4), WorkState::Closed)),
        );

        let trees = tracker.trees(vec![WorkRef::issue(1)]).await;
        let reached: Vec<_> = trees
            .items
            .iter()
            .map(|item| item.reference.number())
            .collect();
        assert_eq!(reached, vec![1, 2, 3, 4]);
        assert_eq!(trees.unresolved, Vec::new());
    }

    #[tokio::test]
    async fn an_unreadable_item_is_listed_with_its_reason() {
        let tracker = AnyWorkTracker::new(FixedTracker::default());
        let trees = tracker.trees(vec![WorkRef::issue(9)]).await;
        assert_eq!(trees.items, Vec::new());
        assert_eq!(
            trees.unresolved,
            vec![WorkTrackerError::NotFound(WorkRef::issue(9))]
        );
    }
}
