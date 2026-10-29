//! What a subagent's terminal is allowed to write.
//!
//! A subagent works in a space of its own inside its launcher's space and can
//! read, but not write, everything its launcher can write. The launcher widens
//! that with a [`SubagentGrant`]: directories, usually linked worktrees, the
//! subagent may write, and the worktrees it may commit in.
//!
//! A linked worktree keeps its git data in the main repository, outside the
//! worktree, so a worktree grant alone leaves `git commit` failing. A commit
//! grant adds write access to that worktree's own git directory, to the
//! loose-object directories and to one branch namespace,
//! `refs/heads/subagent/<id>/`, that the worktree's branch lives in. The
//! repository configuration, the packs, the object store's `info` directory
//! and every other ref stay read-only, so nothing the subagent writes can
//! redirect or reconfigure the host-side [`CommitLedger`] that lists its
//! commits.

use std::io;
use std::path::{Path, PathBuf};

use futures::future::try_join_all;
use serde::{Serialize, Serializer};

/// What a launcher grants a subagent beyond its own space.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubagentGrant {
    worktrees: Vec<WorktreeGrant>,
}

/// One directory a subagent may write, and whether it may commit there.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WorktreeGrant {
    path: PathBuf,
    commit: bool,
}

impl SubagentGrant {
    /// A grant of nothing beyond the subagent's own space.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lets the subagent write the directory `path`, usually a linked worktree
    /// the launcher created. Its git data stays read-only, so the subagent
    /// cannot commit there.
    #[must_use]
    pub fn with_worktree(mut self, path: impl Into<PathBuf>) -> Self {
        self.worktrees.push(WorktreeGrant {
            path: path.into(),
            commit: false,
        });
        self
    }

    /// Lets the subagent write the linked worktree at `path` and commit in it.
    ///
    /// The worktree's branch must lie in a namespace of its own,
    /// `refs/heads/subagent/<id>/<name>`, holding no other branch when the
    /// subagent launches: the commit grant covers that namespace and no other
    /// ref.
    #[must_use]
    pub fn with_committing_worktree(mut self, path: impl Into<PathBuf>) -> Self {
        self.worktrees.push(WorktreeGrant {
            path: path.into(),
            commit: true,
        });
        self
    }
}

/// Why a [`SubagentGrant`] cannot be given.
#[derive(Debug, thiserror::Error)]
pub enum SubagentGrantError {
    /// The launcher cannot write the path, so it cannot hand it on.
    #[error("{} is not writable by the agent launching the subagent", .0.display())]
    OutsideAuthority(PathBuf),
    /// The launcher is a subagent that may not commit in the worktree, so it
    /// cannot hand that right on.
    #[error(
        "the agent launching the subagent may not commit in {}, so it cannot grant commit there",
        .0.display()
    )]
    CommitOutsideAuthority(PathBuf),
    /// A path of the grant could not be inspected.
    #[error("cannot inspect {}: {source}", .path.display())]
    Inspect {
        /// The path.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
    /// The granted path holds a repository checkout, whose git data sits inside
    /// it and so would be writable whatever the commit grant says.
    #[error(
        "{} is a repository checkout, not a linked worktree: its git data would be writable without a commit grant",
        .0.display()
    )]
    RepositoryCheckout(PathBuf),
    /// A linked worktree's `.git` file or its git directory is not what git
    /// writes for a linked worktree of the repository it names.
    #[error("malformed git data of worktree {}: {reason}", .path.display())]
    MalformedGitData {
        /// The worktree.
        path: PathBuf,
        /// What is wrong.
        reason: &'static str,
    },
    /// A commit grant names a path that is not a linked worktree.
    #[error("{} is not a linked worktree, so there is nothing to commit in", .0.display())]
    NotAWorktree(PathBuf),
    /// Git data the commit authorization depends on overlaps a path some
    /// subagent can write, so a subagent could rewrite it.
    #[error(
        "the git data {} of worktree {} overlaps {}, which a subagent can write",
        .git_data.display(),
        .worktree.display(),
        .writable.display()
    )]
    GitDataExposed {
        /// The worktree.
        worktree: PathBuf,
        /// Its git directory or repository.
        git_data: PathBuf,
        /// The writable path it overlaps.
        writable: PathBuf,
    },
    /// The worktree's branch does not lie in a subagent branch namespace.
    #[error(
        "worktree {} must be on a branch `refs/heads/subagent/<id>/<name>` to be granted commit; its HEAD is `{head}`",
        .worktree.display()
    )]
    BranchOutsideNamespace {
        /// The worktree.
        worktree: PathBuf,
        /// What its `HEAD` holds.
        head: String,
    },
    /// The branch namespace of a committing worktree holds other refs, so it
    /// is not that worktree's own.
    #[error(
        "the branch namespace `{namespace}` of worktree {} also holds `{other}`; each committing worktree needs a namespace of its own",
        .worktree.display()
    )]
    SharedNamespace {
        /// The worktree.
        worktree: PathBuf,
        /// The namespace, `refs/heads/subagent/<id>/`.
        namespace: String,
        /// A ref in it that is not the worktree's branch.
        other: String,
    },
    /// The repository stores refs in a reftable, which has no per-namespace
    /// directory a grant could cover.
    #[error(
        "the repository {} stores its refs in a reftable, which a commit grant cannot narrow to one branch namespace",
        .0.display()
    )]
    ReftableRepository(PathBuf),
    /// The refs of a committing worktree could not be read.
    #[error(transparent)]
    Ledger(#[from] CommitLedgerError),
    /// The terminal runs in a container that does not say which host
    /// directory it mounts, so a subagent's paths have no container spelling.
    #[error(
        "the container runtime declares no workspace mount, so a subagent's paths cannot be confined in it"
    )]
    NoContainerMount,
    /// A path the subagent works in is not inside the container's workspace
    /// mount.
    #[error("{} has no path inside the container: {source}", .path.display())]
    OutsideContainerMount {
        /// The host path.
        path: PathBuf,
        /// Why it cannot be mapped.
        source: crate::MountRootError,
    },
    /// The subagent's own space could not be created.
    #[error("cannot create the subagent space {}: {source}", .path.display())]
    Space {
        /// The directory.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
}

/// What a launching terminal may hand on.
#[derive(Debug, Clone, Copy)]
pub struct Launcher<'a> {
    /// The canonical paths the launcher can write, its working directory
    /// included.
    pub writable: &'a [PathBuf],
    /// The worktrees a subagent launcher may commit in. `None` for the main
    /// agent, whose commit grants are read from the repository and need its
    /// git data inside [`Self::writable`].
    pub committing: Option<&'a [CommittingWorktree]>,
    /// The directory every subagent space of the session lives under.
    pub spaces_root: &'a Path,
}

/// What the main agent's terminal has handed to subagents so far.
///
/// Every subagent can write only paths recorded here (or inside a space
/// under the spaces root), so git data that overlaps none of them cannot have
/// been rewritten by a subagent. A worktree granted commit once keeps the
/// git data resolved then, whatever its `.git` file says later.
#[derive(Debug, Default)]
pub struct LaunchHistory {
    /// Every directory handed out with a plain write grant.
    handed_out: Vec<PathBuf>,
    /// Every worktree granted commit, with the git data resolved for it.
    committed: Vec<CommittingWorktree>,
}

/// The paths a [`SubagentGrant`] resolved to.
#[derive(Debug)]
pub struct ResolvedGrant {
    pub writable: Vec<PathBuf>,
    pub committing: Vec<CommittingWorktree>,
}

/// A linked worktree a subagent may commit in, with the git data the grant
/// covers. Everything here was resolved when the grant was first given and is
/// never re-read from paths a subagent can write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittingWorktree {
    /// The worktree.
    worktree: PathBuf,
    /// Its per-worktree git directory: index, `HEAD`, locks.
    git_dir: PathBuf,
    /// The repository's shared git directory.
    common_dir: PathBuf,
    /// The `<id>` of the worktree's branch namespace `refs/heads/subagent/<id>/`.
    namespace: String,
}

/// The two-hex-digit loose-object directories git writes new objects into.
fn fan_out_dirs(objects: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    (0..=u8::MAX).map(move |byte| objects.join(hex::encode([byte])))
}

/// Whether one path lies inside the other.
fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

fn inside_any(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| path.starts_with(root))
}

impl CommittingWorktree {
    /// `refs/heads/subagent/<id>/`, as git spells the namespace.
    fn ref_prefix(&self) -> String {
        format!("refs/heads/subagent/{}/", self.namespace)
    }

    fn namespace_dir(&self, root: &Path) -> PathBuf {
        root.join("refs")
            .join("heads")
            .join("subagent")
            .join(&self.namespace)
    }

    /// The directories a commit writes: the worktree's git directory, the
    /// loose-object directories, and the branch namespace with its reflogs.
    pub fn writable(&self) -> Vec<PathBuf> {
        let mut paths = vec![
            self.git_dir.clone(),
            self.namespace_dir(&self.common_dir),
            self.namespace_dir(&self.common_dir.join("logs")),
        ];
        paths.extend(fan_out_dirs(&self.common_dir.join("objects")));
        paths
    }

    /// Creates the granted directories git would otherwise create on demand
    /// in a directory the subagent cannot write.
    async fn prepare(&self) -> Result<(), SubagentGrantError> {
        try_join_all(self.writable().into_iter().map(|path| async move {
            async_fs::create_dir_all(&path)
                .await
                .map_err(|source| SubagentGrantError::Inspect { path, source })
        }))
        .await
        .map(drop)
    }

    /// Reads the git data of the linked worktree at `worktree` from the
    /// repository, for a commit grant the main agent gives.
    async fn discover(
        worktree: &Path,
        git: Option<WorktreeGit>,
        exposed: impl Iterator<Item = &Path>,
    ) -> Result<Self, SubagentGrantError> {
        let malformed = |reason| SubagentGrantError::MalformedGitData {
            path: worktree.to_path_buf(),
            reason,
        };
        let WorktreeGit {
            git_dir,
            common_dir,
        } = git.ok_or_else(|| SubagentGrantError::NotAWorktree(worktree.to_path_buf()))?;

        // The repository must list this worktree back: its git directory is
        // one of the repository's worktree entries, and that entry names this
        // worktree's `.git` file.
        if git_dir.parent() != Some(common_dir.join("worktrees").as_path()) {
            return Err(malformed(
                "its git directory is not an entry of the repository's `worktrees`",
            ));
        }
        let back = read_file(&git_dir.join("gitdir")).await?;
        let back = canonical(&git_dir.join(back.trim())).await?;
        if back != canonical(&worktree.join(".git")).await? {
            return Err(malformed(
                "the repository's worktree entry names another worktree",
            ));
        }
        if async_fs::metadata(common_dir.join("reftable"))
            .await
            .is_ok()
        {
            return Err(SubagentGrantError::ReftableRepository(common_dir));
        }
        for writable in exposed {
            for git_data in [&git_dir, &common_dir] {
                if overlaps(writable, git_data) {
                    return Err(SubagentGrantError::GitDataExposed {
                        worktree: worktree.to_path_buf(),
                        git_data: git_data.clone(),
                        writable: writable.to_path_buf(),
                    });
                }
            }
        }

        let head = read_file(&git_dir.join("HEAD")).await?;
        let head = head.trim();
        let namespace = head
            .strip_prefix("ref: refs/heads/subagent/")
            .and_then(|rest| rest.split_once('/'))
            .filter(|(id, name)| !name.is_empty() && !id.is_empty() && *id != "." && *id != "..")
            .map(|(id, _)| id.to_string())
            .ok_or_else(|| SubagentGrantError::BranchOutsideNamespace {
                worktree: worktree.to_path_buf(),
                head: head.to_string(),
            })?;
        let branch = head.trim_start_matches("ref: ").to_string();

        let worktree = Self {
            worktree: worktree.to_path_buf(),
            git_dir,
            common_dir,
            namespace,
        };
        // The namespace is the worktree's own: nothing else lives in it, so
        // every ref the subagent moves there is its own work.
        if let Some(other) = worktree
            .namespace_refs()
            .await?
            .into_iter()
            .map(|(_, name)| name)
            .find(|name| *name != branch)
        {
            return Err(SubagentGrantError::SharedNamespace {
                namespace: worktree.ref_prefix(),
                worktree: worktree.worktree,
                other,
            });
        }
        Ok(worktree)
    }

    /// Every ref in the namespace, as `(object id, ref name)`, read from the
    /// repository.
    async fn namespace_refs(&self) -> Result<Vec<(String, String)>, CommitLedgerError> {
        let prefix = self.ref_prefix();
        let output = git(
            &self.common_dir,
            &[
                "for-each-ref",
                "--format=%(objectname)%00%(refname)",
                prefix.as_str(),
            ],
        )
        .await?;
        output
            .lines()
            .map(|line| {
                line.split_once('\0')
                    .and_then(|(id, name)| Some((object_id(id)?.to_string(), name.to_string())))
                    .ok_or_else(|| CommitLedgerError::Output {
                        repository: self.common_dir.clone(),
                        line: line.to_string(),
                    })
            })
            .collect()
    }

    /// The commits the worktree's refs point at now: every ref in the branch
    /// namespace and a detached `HEAD`.
    async fn tips(&self) -> Result<Vec<String>, CommitLedgerError> {
        let mut tips: Vec<String> = self
            .namespace_refs()
            .await?
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        // `HEAD` is read as plain data: a detached commit id is a tip, a
        // symbolic ref names a branch that is either in the namespace or not
        // the subagent's to move.
        let head_path = self.git_dir.join("HEAD");
        match async_fs::read_to_string(&head_path).await {
            Ok(head) => tips.extend(object_id(head.trim()).map(str::to_string)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(CommitLedgerError::Read {
                    path: head_path,
                    source,
                });
            }
        }
        tips.sort_unstable();
        tips.dedup();
        Ok(tips)
    }

    /// The commits reachable from `now` and not from `before`, oldest first.
    async fn commits_between(
        &self,
        before: &[String],
        now: &[String],
    ) -> Result<Vec<SubagentCommit>, CommitLedgerError> {
        if now.is_empty() {
            return Ok(Vec::new());
        }
        let mut args = vec![
            "log",
            "--no-show-signature",
            "--no-decorate",
            "--no-color",
            "--reverse",
            "--format=%H%x00%s",
        ];
        args.extend(now.iter().map(String::as_str));
        args.push("--not");
        args.extend(before.iter().map(String::as_str));
        let output = git(&self.common_dir, &args).await?;
        output
            .lines()
            .map(|line| {
                line.split_once('\0')
                    .and_then(|(id, subject)| {
                        Some(SubagentCommit {
                            worktree: self.worktree.clone(),
                            commit: object_id(id)?.to_string(),
                            subject: subject.to_string(),
                        })
                    })
                    .ok_or_else(|| CommitLedgerError::Output {
                        repository: self.common_dir.clone(),
                        line: line.to_string(),
                    })
            })
            .collect()
    }
}

/// `text` when it is a full hexadecimal object id, SHA-1 or SHA-256.
fn object_id(text: &str) -> Option<&str> {
    (matches!(text.len(), 40 | 64) && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then_some(text)
}

/// The git data a linked worktree's `.git` file names.
struct WorktreeGit {
    /// The per-worktree git directory.
    git_dir: PathBuf,
    /// The repository's shared git directory.
    common_dir: PathBuf,
}

pub async fn canonical(path: &Path) -> Result<PathBuf, SubagentGrantError> {
    async_fs::canonicalize(path)
        .await
        .map_err(|source| SubagentGrantError::Inspect {
            path: path.to_path_buf(),
            source,
        })
}

async fn read_file(path: &Path) -> Result<String, SubagentGrantError> {
    async_fs::read_to_string(path)
        .await
        .map_err(|source| SubagentGrantError::Inspect {
            path: path.to_path_buf(),
            source,
        })
}

/// Reads the git data the `.git` file of `worktree` names.
///
/// `None` when `worktree` is not a git working tree at all; an error when it is
/// a repository checkout rather than a linked worktree.
async fn read_worktree_git(worktree: &Path) -> Result<Option<WorktreeGit>, SubagentGrantError> {
    let dot_git = worktree.join(".git");
    let metadata = match async_fs::metadata(&dot_git).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SubagentGrantError::Inspect {
                path: dot_git,
                source,
            });
        }
    };
    if metadata.is_dir() {
        return Err(SubagentGrantError::RepositoryCheckout(
            worktree.to_path_buf(),
        ));
    }
    let malformed = |reason| SubagentGrantError::MalformedGitData {
        path: worktree.to_path_buf(),
        reason,
    };
    let content = read_file(&dot_git).await?;
    let git_dir = content
        .trim()
        .strip_prefix("gitdir:")
        .map(str::trim)
        .ok_or_else(|| malformed("`.git` has no `gitdir:` line"))?;
    let git_dir = canonical(&worktree.join(git_dir)).await?;
    let common_dir = match async_fs::read_to_string(git_dir.join("commondir")).await {
        Ok(relative) => canonical(&git_dir.join(relative.trim())).await?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(malformed("the git directory has no `commondir`"));
        }
        Err(source) => {
            return Err(SubagentGrantError::Inspect {
                path: git_dir.join("commondir"),
                source,
            });
        }
    };
    Ok(Some(WorktreeGit {
        git_dir,
        common_dir,
    }))
}

impl CommittingWorktree {
    /// The committing worktree `launcher` may hand on at `path`: one it holds
    /// itself when it is a subagent; for the main agent, the one granted
    /// before or, the first time, the one read from the repository.
    async fn granted(
        path: &Path,
        git: Option<WorktreeGit>,
        launcher: Launcher<'_>,
        history: &LaunchHistory,
        granted: &[PathBuf],
    ) -> Result<Self, SubagentGrantError> {
        if let Some(held) = launcher.committing {
            return held
                .iter()
                .find(|held| held.worktree == path)
                .cloned()
                .ok_or_else(|| SubagentGrantError::CommitOutsideAuthority(path.to_path_buf()));
        }
        if let Some(held) = history.committed.iter().find(|held| held.worktree == path) {
            return Ok(held.clone());
        }
        Self::discover(
            path,
            git,
            history
                .handed_out
                .iter()
                .chain(granted)
                .map(PathBuf::as_path)
                .chain([launcher.spaces_root]),
        )
        .await
    }
}

impl SubagentGrant {
    /// Resolves the grant against what `launcher` may hand on.
    ///
    /// Every path the subagent will be able to write, git data included, must
    /// lie inside what the launcher can write; a subagent launcher hands on
    /// only the commit grants it holds itself. The main agent's grants are
    /// recorded in `history`.
    pub(crate) async fn resolve(
        &self,
        launcher: Launcher<'_>,
        history: &mut LaunchHistory,
    ) -> Result<ResolvedGrant, SubagentGrantError> {
        let mut paths = Vec::with_capacity(self.worktrees.len());
        for grant in &self.worktrees {
            let path = canonical(&grant.path).await?;
            if !inside_any(&path, launcher.writable) {
                return Err(SubagentGrantError::OutsideAuthority(path));
            }
            paths.push((path, grant.commit));
        }
        let granted: Vec<PathBuf> = paths.iter().map(|(path, _)| path.clone()).collect();

        let mut writable = Vec::new();
        let mut committing: Vec<CommittingWorktree> = Vec::new();
        for (path, commit) in &paths {
            let git = read_worktree_git(path).await?;
            // A worktree's git data inside another granted path would be
            // writable whatever the commit grant says.
            if let Some(git) = &git
                && let Some(exposing) = granted.iter().find(|granted| {
                    overlaps(granted, &git.git_dir) || overlaps(granted, &git.common_dir)
                })
            {
                return Err(SubagentGrantError::GitDataExposed {
                    worktree: path.clone(),
                    git_data: git.common_dir.clone(),
                    writable: exposing.clone(),
                });
            }
            if !commit {
                // A plain write grant never reaches into git data some
                // subagent's commits are listed from.
                if let Some(held) = history
                    .committed
                    .iter()
                    .find(|held| overlaps(path, &held.common_dir) || overlaps(path, &held.git_dir))
                {
                    return Err(SubagentGrantError::GitDataExposed {
                        worktree: held.worktree.clone(),
                        git_data: held.common_dir.clone(),
                        writable: path.clone(),
                    });
                }
                writable.push(path.clone());
                continue;
            }

            let worktree =
                CommittingWorktree::granted(path, git, launcher, history, &granted).await?;
            if let Some(part) = worktree
                .writable()
                .into_iter()
                .find(|part| !inside_any(part, launcher.writable))
            {
                return Err(SubagentGrantError::OutsideAuthority(part));
            }
            if let Some(shared) = committing.iter().find(|other| {
                other.common_dir == worktree.common_dir && other.namespace == worktree.namespace
            }) {
                return Err(SubagentGrantError::SharedNamespace {
                    worktree: worktree.worktree.clone(),
                    namespace: worktree.ref_prefix(),
                    other: shared.worktree.display().to_string(),
                });
            }
            worktree.prepare().await?;
            writable.push(path.clone());
            committing.push(worktree);
        }

        if launcher.committing.is_none() {
            history.handed_out.extend(
                paths
                    .iter()
                    .filter(|(_, commit)| !commit)
                    .map(|(path, _)| path.clone()),
            );
            for worktree in &committing {
                if !history.committed.contains(worktree) {
                    history.committed.push(worktree.clone());
                }
            }
        }
        writable.extend(committing.iter().flat_map(CommittingWorktree::writable));
        Ok(ResolvedGrant {
            writable,
            committing,
        })
    }
}

/// A commit a subagent made.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubagentCommit {
    /// The worktree it was made in.
    pub worktree: PathBuf,
    /// The commit id.
    pub commit: String,
    /// The first line of the commit message.
    pub subject: String,
}

/// Why the commits of a subagent could not be listed.
#[derive(Debug, thiserror::Error)]
pub enum CommitLedgerError {
    /// `git` could not be run.
    #[error("cannot run git on {}: {source}", .repository.display())]
    Spawn {
        /// The repository.
        repository: PathBuf,
        /// The failure.
        source: io::Error,
    },
    /// `git` failed.
    #[error("git failed on {}: {stderr}", .repository.display())]
    Git {
        /// The repository.
        repository: PathBuf,
        /// What git printed.
        stderr: String,
    },
    /// `git` printed a line that is not what the query asked for.
    #[error("unexpected git output on {}: {line:?}", .repository.display())]
    Output {
        /// The repository.
        repository: PathBuf,
        /// The line.
        line: String,
    },
    /// A worktree's `HEAD` could not be read.
    #[error("cannot read {}: {source}", .path.display())]
    Read {
        /// The file.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
}

/// Where a ledger's worktrees stood at one moment: the commits their branch
/// namespaces and detached heads pointed at.
#[derive(Debug, Clone)]
pub struct CommitMark {
    tips: Vec<Vec<String>>,
}

/// Lists the commits a subagent holding a commit grant made.
///
/// It reads only the repository each grant was resolved against, through
/// paths fixed when the grant was given: never a worktree's `.git` file, and
/// never configuration a subagent can write.
#[derive(Debug, Clone)]
pub struct CommitLedger {
    worktrees: Vec<CommittingWorktree>,
}

impl CommitLedger {
    /// The ledger of `worktrees`; `None` when there are none.
    pub(crate) fn new(worktrees: &[CommittingWorktree]) -> Option<Self> {
        (!worktrees.is_empty()).then(|| Self {
            worktrees: worktrees.to_vec(),
        })
    }

    /// Records where every worktree stands now. A branch not yet born simply
    /// has no tip.
    ///
    /// # Errors
    /// Fails when git cannot list a worktree's refs.
    pub async fn mark(&self) -> Result<CommitMark, CommitLedgerError> {
        let tips = try_join_all(self.worktrees.iter().map(CommittingWorktree::tips)).await?;
        Ok(CommitMark { tips })
    }

    /// The commits made since `mark`, oldest first per worktree: everything
    /// the worktree's branch namespace or detached head reaches now and did
    /// not reach then, whatever the subagent did with `HEAD` in between.
    ///
    /// # Errors
    /// Fails when git cannot list a worktree's refs or commits.
    pub async fn since(&self, mark: &CommitMark) -> Result<Vec<SubagentCommit>, CommitLedgerError> {
        let listed = try_join_all(self.worktrees.iter().zip(&mark.tips).map(
            |(worktree, before)| async move {
                let now = worktree.tips().await?;
                worktree.commits_between(before, &now).await
            },
        ))
        .await?;
        Ok(listed.into_iter().flatten().collect())
    }
}

/// The commits listed for one run of a subagent holding a commit grant, or
/// why they could not be listed. Serializes as the list, or as
/// `{"error": "..."}`.
#[derive(Debug)]
pub struct CommitListing(pub Result<Vec<SubagentCommit>, CommitLedgerError>);

impl Serialize for CommitListing {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Failure {
            error: String,
        }
        match &self.0 {
            Ok(commits) => commits.serialize(serializer),
            Err(error) => Failure {
                error: error.to_string(),
            }
            .serialize(serializer),
        }
    }
}

/// The null device, which stands in for the global git configuration.
#[cfg(windows)]
const NULL_DEVICE: &str = "NUL";
#[cfg(not(windows))]
const NULL_DEVICE: &str = "/dev/null";

/// Runs a read-only git query against the repository at `common_dir`, on the
/// host and outside any sandbox.
///
/// Git is pointed at the repository directly, never through a worktree, so
/// no `.git` file, `commondir` or per-worktree configuration a subagent wrote
/// is consulted. The only configuration read is the repository's own, which
/// a commit grant never makes writable; system and global configuration and
/// every inherited `GIT_*` variable are dropped, and signature display and
/// lazy fetching, the paths by which a query could start another program,
/// are off.
async fn git(common_dir: &Path, args: &[&str]) -> Result<String, CommitLedgerError> {
    let mut command = async_process::Command::new("git");
    for (key, _) in std::env::vars_os() {
        if key.to_str().is_some_and(|key| key.starts_with("GIT_")) {
            command.env_remove(key);
        }
    }
    let output = command
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", NULL_DEVICE)
        .env("GIT_NO_LAZY_FETCH", "1")
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .current_dir(common_dir)
        .arg("--git-dir")
        .arg(common_dir)
        .args([
            "-c",
            "log.showSignature=false",
            "-c",
            "core.fsmonitor=false",
        ])
        .args(args)
        .stdin(async_process::Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| CommitLedgerError::Spawn {
            repository: common_dir.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(CommitLedgerError::Git {
            repository: common_dir.to_path_buf(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Runs git in `dir` with a fixed identity and no user configuration.
    pub fn git_in(dir: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", NULL_DEVICE)
            .args(["-c", "user.name=test", "-c", "user.email=test@example.com"])
            .args(["-c", "commit.gpgSign=false"])
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// A repository with one commit and a linked worktree on the branch
    /// `subagent/<namespace>/work`, both inside `root`; returns the worktree.
    pub fn repository_with_linked_worktree(root: &Path, namespace: &str) -> PathBuf {
        let repository = root.join("repository");
        std::fs::create_dir_all(&repository).unwrap();
        git_in(&repository, &["init", "--initial-branch=main"]);
        git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
        let worktree = root.join(namespace);
        let branch = format!("subagent/{namespace}/work");
        git_in(
            &repository,
            &[
                "worktree",
                "add",
                "-b",
                branch.as_str(),
                worktree.to_str().unwrap(),
            ],
        );
        worktree
    }

    async fn main_launch(
        root: &Path,
        grant: &SubagentGrant,
        history: &mut LaunchHistory,
    ) -> Result<ResolvedGrant, SubagentGrantError> {
        let writable = [std::fs::canonicalize(root).unwrap()];
        let spaces_root = writable[0].join("subagent-spaces");
        grant
            .resolve(
                Launcher {
                    writable: &writable,
                    committing: None,
                    spaces_root: &spaces_root,
                },
                history,
            )
            .await
    }

    #[tokio::test]
    async fn a_commit_grant_covers_one_branch_namespace_and_no_repository_configuration() {
        let dir = tempfile::tempdir().unwrap();
        let worktree = repository_with_linked_worktree(dir.path(), "alpha");
        let resolved = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut LaunchHistory::default(),
        )
        .await
        .expect("a worktree on a namespaced branch can be granted commit");
        let common = std::fs::canonicalize(dir.path().join("repository/.git")).unwrap();
        let writable = &resolved.writable;
        assert!(writable.contains(&common.join("refs/heads/subagent/alpha")));
        assert!(writable.contains(&common.join("objects/ab")));
        for forbidden in [
            common.clone(),
            common.join("config"),
            common.join("objects"),
            common.join("objects/info"),
            common.join("objects/pack"),
            common.join("refs/heads"),
            common.join("refs/heads/main"),
            common.join("packed-refs"),
            common.join("hooks"),
        ] {
            assert!(
                !writable.iter().any(|path| forbidden.starts_with(path)),
                "{} must stay read-only",
                forbidden.display()
            );
        }
    }

    #[tokio::test]
    async fn a_branch_outside_a_subagent_namespace_cannot_be_granted_commit() {
        let dir = tempfile::tempdir().unwrap();
        let repository = dir.path().join("repository");
        std::fs::create_dir_all(&repository).unwrap();
        git_in(&repository, &["init", "--initial-branch=main"]);
        git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
        let worktree = dir.path().join("feature");
        git_in(
            &repository,
            &[
                "worktree",
                "add",
                "-b",
                "feature",
                worktree.to_str().unwrap(),
            ],
        );
        let error = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut LaunchHistory::default(),
        )
        .await
        .expect_err("a plain branch is not a subagent namespace");
        assert!(
            matches!(error, SubagentGrantError::BranchOutsideNamespace { .. }),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_namespace_holding_another_branch_is_not_the_worktrees_own() {
        let dir = tempfile::tempdir().unwrap();
        let worktree = repository_with_linked_worktree(dir.path(), "beta");
        git_in(
            &dir.path().join("repository"),
            &["branch", "subagent/beta/other", "main"],
        );
        let error = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut LaunchHistory::default(),
        )
        .await
        .expect_err("a shared namespace is refused");
        assert!(
            matches!(error, SubagentGrantError::SharedNamespace { .. }),
            "{error}"
        );
    }

    /// A subagent with a commit grant rewrites the worktree's `.git` file and
    /// its git directory's `commondir` to a repository of its own whose
    /// configuration runs a program for every signed commit `git log` shows.
    /// The ledger keeps reading the real repository and runs nothing.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_ledger_ignores_a_redirected_git_directory_and_runs_no_program() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let worktree = repository_with_linked_worktree(dir.path(), "gamma");
        let resolved = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut LaunchHistory::default(),
        )
        .await
        .unwrap();
        let ledger = CommitLedger::new(&resolved.committing).expect("one committing worktree");
        let mark = ledger.mark().await.unwrap();
        git_in(&worktree, &["commit", "--allow-empty", "-m", "real work"]);

        // The decoy: a repository whose config runs `program` to show a
        // signature, with a commit carrying a `gpgsig` header.
        let decoy = dir.path().join("decoy");
        std::fs::create_dir_all(&decoy).unwrap();
        git_in(&decoy, &["init", "--initial-branch=main"]);
        let marker = dir.path().join("ran");
        let program = dir.path().join("gpg-program");
        std::fs::write(
            &program,
            format!("#!/bin/sh\ntouch '{}'\nexit 1\n", marker.display()),
        )
        .unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        git_in(&decoy, &["config", "log.showSignature", "true"]);
        git_in(
            &decoy,
            &["config", "gpg.program", program.to_str().unwrap()],
        );
        let tree = git_in(&decoy, &["mktree"]);
        let signed = format!(
            "tree {tree}\nauthor a <a@a> 0 +0000\ncommitter a <a@a> 0 +0000\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n -----END PGP SIGNATURE-----\n\nsigned decoy\n"
        );
        let object = dir.path().join("signed-commit");
        std::fs::write(&object, signed).unwrap();
        let signed = git_in(
            &decoy,
            &[
                "hash-object",
                "-t",
                "commit",
                "-w",
                object.to_str().unwrap(),
            ],
        );
        git_in(&decoy, &["update-ref", "HEAD", signed.as_str()]);
        let decoy_git = std::fs::canonicalize(decoy.join(".git")).unwrap();
        let worktree_git = &resolved.committing[0].git_dir;
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", decoy_git.display()),
        )
        .unwrap();
        std::fs::write(
            worktree_git.join("commondir"),
            format!("{}\n", decoy_git.display()),
        )
        .unwrap();

        let commits = ledger
            .since(&mark)
            .await
            .expect("the real repository is read");
        assert_eq!(commits.len(), 1, "{commits:?}");
        assert_eq!(commits[0].subject, "real work");
        assert!(
            !marker.exists(),
            "the ledger ran a program the subagent chose"
        );
    }

    /// Commits made on another branch of the namespace, or left behind by
    /// resetting the worktree's branch and detaching `HEAD`, are still listed;
    /// a branch moved back to where it was lists nothing.
    #[tokio::test]
    async fn the_ledger_lists_commits_whatever_head_does() {
        let dir = tempfile::tempdir().unwrap();
        let worktree = repository_with_linked_worktree(dir.path(), "delta");
        let resolved = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut LaunchHistory::default(),
        )
        .await
        .unwrap();
        let ledger = CommitLedger::new(&resolved.committing).unwrap();
        let mark = ledger.mark().await.unwrap();

        git_in(&worktree, &["switch", "-c", "subagent/delta/side"]);
        git_in(
            &worktree,
            &["commit", "--allow-empty", "-m", "on a side branch"],
        );
        git_in(&worktree, &["switch", "subagent/delta/work"]);
        git_in(
            &worktree,
            &["commit", "--allow-empty", "-m", "detached later"],
        );
        git_in(&worktree, &["switch", "--detach"]);
        git_in(&worktree, &["branch", "-f", "subagent/delta/work", "main"]);

        let mut subjects: Vec<String> = ledger
            .since(&mark)
            .await
            .unwrap()
            .into_iter()
            .map(|commit| commit.subject)
            .collect();
        subjects.sort();
        assert_eq!(subjects, ["detached later", "on a side branch"]);
    }

    #[tokio::test]
    async fn a_plain_grant_cannot_reach_git_data_commits_are_listed_from() {
        let dir = tempfile::tempdir().unwrap();
        let worktree = repository_with_linked_worktree(dir.path(), "epsilon");
        let mut history = LaunchHistory::default();
        main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut history,
        )
        .await
        .unwrap();
        let objects_info = dir.path().join("repository/.git/objects/info");
        std::fs::create_dir_all(&objects_info).unwrap();
        let error = main_launch(
            dir.path(),
            &SubagentGrant::new().with_worktree(&objects_info),
            &mut history,
        )
        .await
        .expect_err("the object store's info directory stays read-only");
        assert!(
            matches!(error, SubagentGrantError::GitDataExposed { .. }),
            "{error}"
        );
    }

    #[tokio::test]
    async fn git_data_a_subagent_could_write_is_not_trusted_for_a_commit_grant() {
        let dir = tempfile::tempdir().unwrap();
        // The repository itself lies inside a directory handed out earlier.
        let shared = dir.path().join("shared");
        std::fs::create_dir_all(&shared).unwrap();
        let worktree = repository_with_linked_worktree(&shared, "zeta");
        let mut history = LaunchHistory::default();
        main_launch(
            dir.path(),
            &SubagentGrant::new().with_worktree(&shared),
            &mut history,
        )
        .await
        .unwrap();
        let error = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut history,
        )
        .await
        .expect_err("a subagent could have rewritten that repository");
        assert!(
            matches!(error, SubagentGrantError::GitDataExposed { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_failed_listing_serializes_as_an_error() {
        let failed = CommitListing(Err(CommitLedgerError::Git {
            repository: PathBuf::from("/repo"),
            stderr: "boom".to_string(),
        }));
        let value = serde_json::to_value(&failed).unwrap();
        assert!(
            value["error"]
                .as_str()
                .is_some_and(|error| error.contains("boom")),
            "{value}"
        );
        let listed = CommitListing(Ok(Vec::new()));
        assert_eq!(
            serde_json::to_value(&listed).unwrap(),
            serde_json::json!([])
        );
    }
}
