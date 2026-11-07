//! Runtime-mutable filesystem grants for [`TerminalTool`].
//!
//! A [`TerminalGrants`] handle is cheap to clone and shares one store with
//! every clone — a host keeps the handle it got from
//! [`TerminalTool::grants`], adds grants when a workspace request is
//! approved, and every sandbox launched afterwards carries them.
//!
//! A command or shell snapshots the grants once, when it launches: a
//! command already running (and a persistent shell already running) keeps
//! the grants it started with — see [`TerminalGrants::add_readable`].

use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;

/// The readable and writable path grants one terminal sandbox launches
/// with. Immutable once handed out: snapshots are replaced, never edited.
#[derive(Debug, Clone, Default)]
pub struct GrantSet {
    /// Paths sandboxed commands may read but not write.
    pub readable: Vec<PathBuf>,
    /// Paths sandboxed commands may write to (and therefore also read).
    pub writable: Vec<PathBuf>,
    /// The access to the common git directory of a writable linked worktree.
    pub git_dir: GitDirAccess,
}

/// The access a terminal has to the common git directory of a linked
/// worktree it may write.
///
/// That directory is the main repository's, so only the session's own
/// terminal writes it; a subagent's terminal reads it, and commits only
/// through a commit grant's repository of its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GitDirAccess {
    /// Read, write and execute: the session's own terminal.
    #[default]
    ReadWrite,
    /// Read only: a subagent's terminal.
    ReadOnly,
}

impl GitDirAccess {
    /// The access to the common git directory of a worktree the terminal
    /// can write.
    #[must_use]
    pub fn writable_worktree_access(self) -> heel::Access {
        match self {
            Self::ReadWrite => heel::Access::WRITE | heel::Access::EXEC,
            Self::ReadOnly => heel::Access::READ,
        }
    }
}

#[derive(Debug, Clone)]
enum GrantsInner {
    /// The terminal's own store.
    Own(Arc<ArcSwap<GrantSet>>),
    /// A subagent's view: its parent's readable side plus its own grants.
    /// Readable grants added to the parent later reach the subagent;
    /// writable grants added to the parent never become writable here.
    Derived {
        parent: Box<TerminalGrants>,
        grants: Arc<ArcSwap<GrantSet>>,
    },
}

/// Shared, runtime-mutable grants for a [`crate::TerminalTool`].
///
/// Cloning the handle keeps the same underlying store: every clone,
/// including the clones a `TerminalTool` makes for its own bookkeeping and
/// for subagents (see [`Self::derived`]), sees grants added through any
/// handle.
#[derive(Debug, Clone)]
pub struct TerminalGrants {
    inner: GrantsInner,
}

impl Default for TerminalGrants {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalGrants {
    /// An empty grant store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: GrantsInner::Own(Arc::new(ArcSwap::from_pointee(GrantSet::default()))),
        }
    }

    /// A subagent's view of a parent terminal's grants: the snapshot merges
    /// the parent's readable list (kept live) with `own`'s extra readable
    /// and writable lists. Writable grants on the parent are never
    /// writable through this view.
    #[must_use]
    pub fn derived(
        parent: &Self,
        extra_readable: Vec<PathBuf>,
        own_writable: Vec<PathBuf>,
    ) -> Self {
        Self {
            inner: GrantsInner::Derived {
                parent: Box::new(parent.clone()),
                grants: Arc::new(ArcSwap::from_pointee(GrantSet {
                    readable: extra_readable,
                    writable: own_writable,
                    git_dir: GitDirAccess::ReadOnly,
                })),
            },
        }
    }

    const fn own_store(&self) -> &Arc<ArcSwap<GrantSet>> {
        match &self.inner {
            GrantsInner::Own(store) | GrantsInner::Derived { grants: store, .. } => store,
        }
    }

    fn update_own(&self, mutate: impl Fn(&mut GrantSet)) {
        let store = self.own_store();
        store.rcu(|current| {
            let mut next = (**current).clone();
            mutate(&mut next);
            next
        });
    }

    /// Adds `path` as a readable grant. Commands and shells launched after
    /// this call read it; commands and persistent shells already running
    /// keep the grants they launched with. Dedupes; a path already
    /// writable stays writable and is not also listed readable.
    pub fn add_readable(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| {
            if !set.writable.contains(&path) && !set.readable.contains(&path) {
                set.readable.push(path.clone());
            }
        });
    }

    /// Adds `path` as a writable grant, with the same "from the next
    /// launch on" semantics as [`Self::add_readable`]. A writable grant
    /// subsumes a readable one for the same path.
    pub fn add_writable(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| {
            set.readable.retain(|entry| *entry != path);
            if !set.writable.contains(&path) {
                set.writable.push(path.clone());
            }
        });
    }

    /// Takes back the readable grant of `path`, with the same "from the next
    /// launch on" semantics as [`Self::add_readable`]: commands and shells
    /// already running keep the grants they launched with. A writable grant
    /// of the same path is a grant of its own and stays. On a subagent's
    /// view this removes only its own grant, never one its parent holds.
    pub fn remove_readable(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| set.readable.retain(|entry| *entry != path));
    }

    /// Takes back the writable grant of `path`, with the same "from the next
    /// launch on" semantics as [`Self::add_readable`]. Nothing of it is left
    /// readable: read access stays only through a readable grant of the
    /// path.
    pub fn remove_writable(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| set.writable.retain(|entry| *entry != path));
    }

    /// One consistent view of the grants, taken at launch time.
    ///
    /// For a derived handle this merges the parent's current readable
    /// list with this handle's own lists, so readable grants added to the
    /// parent between the derivation and this snapshot are included.
    #[must_use]
    pub fn snapshot(&self) -> Arc<GrantSet> {
        match &self.inner {
            GrantsInner::Own(store) => store.load_full(),
            GrantsInner::Derived {
                parent,
                grants: own,
            } => {
                let parent_set = parent.snapshot();
                let own_set = own.load_full();
                Arc::new(GrantSet {
                    readable: parent_set
                        .readable
                        .iter()
                        .cloned()
                        .chain(own_set.readable.iter().cloned())
                        .collect(),
                    writable: own_set.writable.clone(),
                    git_dir: GitDirAccess::ReadOnly,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_grant_takes_it_back_from_the_next_snapshot() {
        let grants = TerminalGrants::new();
        grants.add_readable("/data/readable");
        grants.add_writable("/data/writable");
        let launched = grants.snapshot();

        grants.remove_readable("/data/readable");
        grants.remove_writable("/data/writable");

        let next = grants.snapshot();
        assert!(next.readable.is_empty(), "{next:?}");
        assert!(next.writable.is_empty(), "{next:?}");
        assert_eq!(launched.readable, [PathBuf::from("/data/readable")]);
        assert_eq!(launched.writable, [PathBuf::from("/data/writable")]);
    }

    #[test]
    fn removing_one_kind_of_grant_leaves_the_other() {
        let grants = TerminalGrants::new();
        grants.add_writable("/data/shared");
        grants.remove_readable("/data/shared");
        assert_eq!(grants.snapshot().writable, [PathBuf::from("/data/shared")]);

        grants.add_readable("/data/other");
        grants.remove_writable("/data/other");
        assert_eq!(grants.snapshot().readable, [PathBuf::from("/data/other")]);
    }

    #[test]
    fn a_subagent_removes_only_its_own_grants() {
        let parent = TerminalGrants::new();
        parent.add_readable("/data/parent");
        let child = TerminalGrants::derived(&parent, vec!["/data/child".into()], Vec::new());

        child.remove_readable("/data/parent");
        child.remove_readable("/data/child");

        assert_eq!(child.snapshot().readable, [PathBuf::from("/data/parent")]);
        assert_eq!(parent.snapshot().readable, [PathBuf::from("/data/parent")]);
    }
}
