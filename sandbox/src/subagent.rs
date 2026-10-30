//! What a subagent's terminal is allowed to write.
//!
//! A subagent works in a space of its own inside its launcher's space and can
//! read, but not write, everything its launcher can write. The launcher widens
//! that with a [`SubagentGrant`]: directories, usually linked worktrees, the
//! subagent may write, and the worktrees it may commit in.
//!
//! A linked worktree keeps its git data in the main repository, outside the
//! worktree, so a worktree grant alone leaves `git commit` failing. A commit
//! grant gives the worktree a repository of its own, created by the host when
//! the grant is first given, that the worktree's `.git` file names: it holds
//! the worktree's `HEAD`, index, refs and new objects, reads the main object
//! store as a read-only alternate and includes the main repository's
//! configuration. It is the only path the grant makes writable, so nothing a
//! subagent writes can change what the main repository reads. The commits
//! reach the main repository only through the [`CommitLedger`]'s import, which
//! packs them inside the subagent's sandbox and verifies the pack on the host
//! before any ref moves.

mod host_git;
mod import;
mod repository;
mod sandboxed;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::future::join_all;

pub use repository::Spelling;
use sandboxed::Bounds;
pub use sandboxed::ExportError;
pub use sandboxed::{SandboxedExit, SandboxedGitImpl, SandboxedRun, SubagentGit};

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
    /// The repository of a committing worktree could not be created.
    #[error("cannot create the subagent repository at {}: {source}", .path.display())]
    Repository {
        /// The path being written.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
    /// A path the repository of a committing worktree must spell out is not
    /// text on one line.
    #[error("{} cannot be written into a subagent repository", .0.display())]
    UnrepresentablePath(PathBuf),
    /// A path handed out with a plain grant contains the committing
    /// worktree, so a subagent holding that grant could replace the worktree
    /// and make the host write its `.git` file elsewhere.
    #[error(
        "{} contains worktree {}, which is granted commit, and is writable by a subagent",
        .writable.display(),
        .worktree.display()
    )]
    WorktreeExposed {
        /// The worktree.
        worktree: PathBuf,
        /// The writable path that contains it.
        writable: PathBuf,
    },
    /// The git on the host is older than a commit grant works with.
    #[error("a commit grant needs git 2.29 or later; the host has `{found}`")]
    GitTooOld {
        /// What `git --version` printed.
        found: String,
    },
}

/// What a launching terminal may hand on.
#[derive(Clone, Copy)]
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
    /// How the subagents' backend spells host paths.
    pub(crate) spelling: Spelling<'a>,
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
    /// Its entry in the repository's `worktrees`: git's own files for it,
    /// `HEAD` mirrored by the import. Nothing in it is granted.
    git_dir: PathBuf,
    /// The repository's shared git directory.
    common_dir: PathBuf,
    /// The repository the host created for the worktree, inside its entry.
    /// The one path the commit grant makes writable.
    repository: PathBuf,
    /// The `<id>` of the worktree's branch namespace `refs/heads/subagent/<id>/`.
    namespace: String,
}

/// The most bytes a subagent's sandbox may send the host while the commits it
/// made are imported: the pack, and every answer to a question about its
/// repository.
///
/// The budget is an operator setting. Every clone is a handle on the same
/// value, so a change through any of them applies to the next import of every
/// terminal that holds one. There is no ceiling, and `0` means the bytes are
/// not bounded at all.
#[derive(Debug, Clone)]
pub struct ImportBudget(Arc<AtomicU64>);

impl ImportBudget {
    /// A budget of `bytes`; `0` is unbounded.
    #[must_use]
    pub fn new(bytes: u64) -> Self {
        Self(Arc::new(AtomicU64::new(bytes)))
    }

    /// Changes the budget for every handle; `0` is unbounded.
    pub fn set(&self, bytes: u64) {
        self.0.store(bytes, Ordering::Relaxed);
    }

    /// The budget now, `0` being unbounded.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// The byte bound of an import that starts now.
    fn limit(&self) -> Option<u64> {
        Some(self.get()).filter(|bytes| *bytes != 0)
    }
}

/// What bounds one import of a subagent's commits.
#[derive(Debug, Clone)]
pub struct ImportLimits {
    /// The most bytes of pack, or of any answer, the subagent's sandbox may
    /// send the host.
    pub bytes: ImportBudget,
    /// How long each piece of sandboxed work of an import may take: every
    /// question to the subagent's repository, the pack, and the tidying after.
    pub time: Duration,
}

impl ImportLimits {
    /// The bounds an import that starts now runs under.
    fn bounds(&self) -> Bounds {
        Bounds {
            bytes: self.bytes.limit(),
            time: self.time,
        }
    }
}

/// Whether one path lies inside the other.
fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

fn inside_any(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| path.starts_with(root))
}

impl CommittingWorktree {
    /// The worktree.
    pub(crate) fn worktree(&self) -> &Path {
        &self.worktree
    }

    /// Its entry in the repository's `worktrees`: the host-owned git
    /// directory a main agent's `git` in the worktree resolves to.
    pub(crate) fn git_dir(&self) -> &Path {
        &self.git_dir
    }

    /// `refs/heads/subagent/<id>/`, as git spells the namespace.
    fn ref_prefix(&self) -> String {
        format!("refs/heads/subagent/{}/", self.namespace)
    }

    /// The directories a commit writes: the worktree's repository, and
    /// nothing else.
    pub fn writable(&self) -> Vec<PathBuf> {
        vec![self.repository.clone()]
    }

    fn layout(&self) -> repository::Layout<'_> {
        repository::Layout {
            worktree: &self.worktree,
            entry: &self.git_dir,
            common: &self.common_dir,
            repository: &self.repository,
            namespace: &self.namespace,
        }
    }

    /// Gives the worktree its repository, if it has none yet, and points the
    /// worktree's `.git` file at it.
    async fn prepare(&self, spelling: Spelling<'_>) -> Result<(), SubagentGrantError> {
        repository::ensure(self.layout(), spelling).await
    }

    /// Reads the git data of the linked worktree at `worktree` from the
    /// repository, for a commit grant the main agent gives.
    ///
    /// `exposed` are the paths a subagent can write that must not overlap the
    /// worktree's git data; `plain` those among them that were handed out
    /// whole, which must not contain the worktree either.
    async fn discover(
        worktree: &Path,
        git: Option<WorktreeGit>,
        exposed: impl Iterator<Item = &Path>,
        plain: impl Iterator<Item = &Path>,
    ) -> Result<Self, SubagentGrantError> {
        let malformed = |reason| SubagentGrantError::MalformedGitData {
            path: worktree.to_path_buf(),
            reason,
        };
        let WorktreeGit {
            git_dir,
            common_dir,
            converted,
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
        // The host writes the worktree's `.git` file; a subagent that could
        // swap the worktree for a symlink first would redirect that write.
        for writable in plain {
            if worktree.starts_with(writable) {
                return Err(SubagentGrantError::WorktreeExposed {
                    worktree: worktree.to_path_buf(),
                    writable: writable.to_path_buf(),
                });
            }
        }

        let repository = git_dir.join(repository::REPOSITORY_DIR);
        if converted {
            // The namespace was fixed when the repository was created, in a
            // file only the host writes; the repository itself says what a
            // subagent made it say.
            let namespace = read_file(&git_dir.join(repository::NAMESPACE_FILE)).await?;
            let namespace = namespace.trim();
            if namespace.is_empty()
                || namespace.contains('/')
                || namespace == "."
                || namespace == ".."
            {
                return Err(malformed("its recorded branch namespace is not valid"));
            }
            return Ok(Self {
                worktree: worktree.to_path_buf(),
                git_dir,
                common_dir,
                repository,
                namespace: namespace.to_string(),
            });
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
            repository,
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
        let output = host_git::git(
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
    /// namespace and a detached `HEAD`. Both live in the main repository,
    /// where only the host writes.
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
        let output = host_git::git(&self.common_dir, &args).await?;
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
    /// Whether the `.git` file names the worktree's own repository inside
    /// that directory, as a commit grant leaves it, rather than the directory
    /// itself.
    converted: bool,
}

/// `path` resolved to its canonical form, without the Windows verbatim
/// (`\\?\`) prefix, which git and other tools given these paths cannot read.
pub async fn canonical(path: &Path) -> Result<PathBuf, SubagentGrantError> {
    async_fs::canonicalize(path)
        .await
        .map(|canonical| dunce::simplified(&canonical).to_path_buf())
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

/// The most bytes of a worktree's `.git` file the host reads: a `gitdir:` line
/// holds one path.
const DOT_GIT_LIMIT: u64 = 16 * 1024;

/// What a worktree's `.git` is.
enum DotGit {
    Missing,
    Directory,
    /// A regular file and what it holds.
    File(String),
}

/// Reads the worktree's `.git`, which a subagent can write, without ever
/// following a symlink in its place, waiting on a FIFO, or reading more than
/// [`DOT_GIT_LIMIT`] bytes.
async fn read_dot_git(dot_git: &Path) -> io::Result<DotGit> {
    use std::io::Read as _;

    let path = dot_git.to_path_buf();
    blocking::unblock(move || {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = match options.open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(DotGit::Missing),
            Err(error) => return Err(error),
        };
        let metadata = file.metadata()?;
        if metadata.is_dir() {
            return Ok(DotGit::Directory);
        }
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        let mut content = Vec::new();
        file.take(DOT_GIT_LIMIT + 1).read_to_end(&mut content)?;
        if u64::try_from(content.len()).map_or(true, |len| len > DOT_GIT_LIMIT) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "longer than a gitdir line can be",
            ));
        }
        String::from_utf8(content)
            .map(DotGit::File)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    })
    .await
}

/// `path` resolved like [`canonical`], except that its last component may be
/// missing: the repository of a worktree that was deleted, to be created
/// again.
async fn canonical_with_missing_leaf(path: &Path) -> Result<PathBuf, SubagentGrantError> {
    match canonical(path).await {
        Err(SubagentGrantError::Inspect { source, .. })
            if source.kind() == io::ErrorKind::NotFound =>
        {
            match (path.parent(), path.file_name()) {
                (Some(parent), Some(name)) => Ok(canonical(parent).await?.join(name)),
                _ => canonical(path).await,
            }
        }
        other => other,
    }
}

/// Reads the git data the `.git` file of `worktree` names.
///
/// `None` when `worktree` is not a git working tree at all; an error when it is
/// a repository checkout rather than a linked worktree.
async fn read_worktree_git(worktree: &Path) -> Result<Option<WorktreeGit>, SubagentGrantError> {
    let dot_git = worktree.join(".git");
    let content = match read_dot_git(&dot_git).await {
        Ok(DotGit::Missing) => return Ok(None),
        Ok(DotGit::Directory) => {
            return Err(SubagentGrantError::RepositoryCheckout(
                worktree.to_path_buf(),
            ));
        }
        Ok(DotGit::File(content)) => content,
        Err(source) => {
            return Err(SubagentGrantError::Inspect {
                path: dot_git,
                source,
            });
        }
    };
    let malformed = |reason| SubagentGrantError::MalformedGitData {
        path: worktree.to_path_buf(),
        reason,
    };
    let git_dir = content
        .trim()
        .strip_prefix("gitdir:")
        .map(str::trim)
        .ok_or_else(|| malformed("`.git` has no `gitdir:` line"))?;
    let named = canonical_with_missing_leaf(&worktree.join(git_dir)).await?;
    // A worktree granted commit names its own repository, which sits inside
    // its entry in the main repository and has no `commondir` of its own.
    let (git_dir, converted) = match async_fs::metadata(named.join("commondir")).await {
        Ok(_) => (named, false),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match named
                .parent()
                .filter(|_| named.file_name() == Some(repository::REPOSITORY_DIR.as_ref()))
            {
                Some(entry) => (entry.to_path_buf(), true),
                None => return Err(malformed("the git directory has no `commondir`")),
            }
        }
        Err(source) => {
            return Err(SubagentGrantError::Inspect {
                path: named.join("commondir"),
                source,
            });
        }
    };
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
        converted,
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
        plain: &[PathBuf],
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
            history
                .handed_out
                .iter()
                .chain(plain)
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
    #[allow(clippy::too_many_lines)]
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
        let plain: Vec<PathBuf> = paths
            .iter()
            .filter(|(_, commit)| !commit)
            .map(|(path, _)| path.clone())
            .collect();

        let mut writable = Vec::new();
        let mut committing: Vec<CommittingWorktree> = Vec::new();
        for (path, commit) in &paths {
            // A worktree already granted commit has its git data resolved;
            // its `.git`, which a subagent may have replaced, is not read
            // again.
            let known = commit
                .then(|| {
                    launcher
                        .committing
                        .unwrap_or(&history.committed)
                        .iter()
                        .find(|held| held.worktree == *path)
                })
                .flatten();
            let git = if known.is_some() {
                None
            } else {
                read_worktree_git(path).await?
            };
            // A worktree's git data inside another granted path would be
            // writable whatever the commit grant says.
            let data = known
                .map(|held| (&held.git_dir, &held.common_dir))
                .or_else(|| git.as_ref().map(|git| (&git.git_dir, &git.common_dir)));
            if let Some((git_dir, common_dir)) = data
                && let Some(exposing) = granted
                    .iter()
                    .find(|granted| overlaps(granted, git_dir) || overlaps(granted, common_dir))
            {
                return Err(SubagentGrantError::GitDataExposed {
                    worktree: path.clone(),
                    git_data: common_dir.clone(),
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
                // Nor does it contain a committing worktree, which a subagent
                // could then replace under the host's `.git` file.
                if let Some(held) = history
                    .committed
                    .iter()
                    .find(|held| held.worktree.starts_with(path))
                {
                    return Err(SubagentGrantError::WorktreeExposed {
                        worktree: held.worktree.clone(),
                        writable: path.clone(),
                    });
                }
                writable.push(path.clone());
                continue;
            }

            if launcher.committing.is_none() {
                host_git::require_supported_version().await?;
            }
            let worktree =
                CommittingWorktree::granted(path, git, launcher, history, &granted, &plain).await?;
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
            if launcher.committing.is_none() {
                worktree.prepare(launcher.spelling).await?;
            }
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// A subagent's repository reported a ref or object id that is malformed
    /// or lies outside the subagent's branch namespace.
    #[error("the subagent repository {} reported {line:?}, which is not a ref of its namespace. {}", .repository.display(), unimported(.repository))]
    Report {
        /// The subagent's repository.
        repository: PathBuf,
        /// What it reported.
        line: String,
    },
    /// Git could not run in the subagent's sandbox, failed there, or did not
    /// finish in time.
    #[error("cannot export from the subagent repository {}: {source}. {}", .repository.display(), unimported(.repository))]
    Export {
        /// The subagent's repository.
        repository: PathBuf,
        /// Why.
        source: ExportError,
    },
    /// The pack the subagent's sandbox sent was refused: an object that is
    /// not what it claims to be, a broken link, or more than the import
    /// allows.
    #[error("the pack from the subagent repository {} was refused: {stderr}. {}", .repository.display(), unimported(.repository))]
    Rejected {
        /// The subagent's repository.
        repository: PathBuf,
        /// What git said.
        stderr: String,
    },
    /// A tip the subagent reported is missing after indexing, is no commit,
    /// or reaches objects that are not there.
    #[error("the subagent repository {} reported commits that did not arrive whole: {detail}. {}", .repository.display(), unimported(.repository))]
    Incomplete {
        /// The subagent's repository.
        repository: PathBuf,
        /// The object or the git message that says what is missing.
        detail: String,
    },
    /// A ref of the main repository moved while the import was under way.
    #[error("`{reference}` moved in the main repository while the commits of {} were imported", .repository.display())]
    Moved {
        /// The subagent's repository.
        repository: PathBuf,
        /// The ref.
        reference: String,
    },
    /// The directory a pack is indexed in could not be set up, or the indexed
    /// pack could not be moved into the main store.
    #[error("cannot use {} to import a pack: {source}", .path.display())]
    Quarantine {
        /// The path.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
}

/// What becomes of commits a repository holds that could not be imported.
fn unimported(repository: &Path) -> String {
    format!(
        "Its commits that were not imported stay in {}; deleting that directory discards them, and granting the worktree again gives it a fresh repository",
        repository.display()
    )
}

/// Where a ledger's worktrees stood at one moment: the commits their branch
/// namespaces and detached heads pointed at.
#[derive(Debug, Clone)]
pub struct CommitMark {
    tips: Vec<Vec<String>>,
    setbacks: Arc<[CommitLedgerError]>,
}

impl CommitMark {
    /// What could not be done to the subagent's repositories before the run.
    ///
    /// A repository the host cannot refresh or import never keeps a
    /// subagent from running; it keeps its commits in itself, out of the main
    /// repository, and the run is told here. The commits that failed to
    /// import are listed with the next run's when a later import succeeds.
    #[must_use]
    pub fn setbacks(&self) -> &[CommitLedgerError] {
        &self.setbacks
    }
}

/// Lists the commits a subagent holding a commit grant made, and brings them
/// into the main repository.
///
/// The main repository's refs, and so the listing, change only in the import
/// that [`Self::since`] runs; it reads nothing a subagent wrote, and the
/// subagent's own git runs only inside the subagent's sandbox.
#[derive(Debug, Clone)]
pub struct CommitLedger {
    worktrees: Vec<CommittingWorktree>,
    git: SubagentGit,
    limits: ImportLimits,
}

impl CommitLedger {
    /// The ledger of `worktrees`; `None` when there are none.
    pub(crate) fn new(
        worktrees: &[CommittingWorktree],
        git: SubagentGit,
        limits: ImportLimits,
    ) -> Option<Self> {
        (!worktrees.is_empty()).then(|| Self {
            worktrees: worktrees.to_vec(),
            git,
            limits,
        })
    }

    /// Starts a run: refreshes each worktree's view of the main repository's
    /// refs inside the subagent's sandbox, imports what its repository
    /// already holds, then records where every worktree stands. A branch not
    /// yet born simply has no tip.
    ///
    /// What the subagent's repository does or holds never keeps a run from
    /// starting. A repository that cannot be refreshed or imported (a stalled
    /// or refused one) is left as it is, the run starts from what the main
    /// repository holds, and the failure is in [`CommitMark::setbacks`] for the
    /// run's report; its commits stay in that repository, which the error
    /// names. The mark's tips are always the main repository's, so a later
    /// import that succeeds lists the commits an earlier holder left too,
    /// never dropping them.
    ///
    /// # Errors
    /// Fails when git cannot list the main repository's refs.
    pub async fn mark(&self) -> Result<CommitMark, CommitLedgerError> {
        // Every worktree is carried through before an error is looked at, so
        // none is dropped partway.
        let prepared = join_all(self.worktrees.iter().map(|worktree| async move {
            refresh(&self.git, worktree, &self.limits).await?;
            // Whatever an earlier holder of the repository left is imported
            // now, so it counts as before this run, not as made in it.
            worktree.import(&self.git, &self.limits).await
        }))
        .await;
        let setbacks: Vec<CommitLedgerError> =
            prepared.into_iter().filter_map(Result::err).collect();
        for setback in &setbacks {
            tracing::warn!(%setback, "a subagent repository could not be prepared for a run");
        }
        let tips = join_all(self.worktrees.iter().map(CommittingWorktree::tips))
            .await
            .into_iter()
            .collect::<Result<_, _>>()?;
        Ok(CommitMark {
            tips,
            setbacks: setbacks.into(),
        })
    }

    /// The commits made since `mark`, oldest first per worktree: everything
    /// the worktree's branch namespace or detached head reaches now and did
    /// not reach then, whatever the subagent did with `HEAD` in between.
    ///
    /// The subagent's commits are imported into the main repository first.
    /// Every worktree's import runs to its end before an error is returned.
    ///
    /// # Errors
    /// Fails when a subagent's commits cannot be imported or git cannot list
    /// a worktree's refs or commits.
    pub async fn since(&self, mark: &CommitMark) -> Result<Vec<SubagentCommit>, CommitLedgerError> {
        let listed = join_all(self.worktrees.iter().zip(&mark.tips).map(
            |(worktree, before)| async move {
                worktree.import(&self.git, &self.limits).await?;
                let now = worktree.tips().await?;
                worktree.commits_between(before, &now).await
            },
        ))
        .await;
        let mut commits = Vec::new();
        for listed in listed {
            commits.extend(listed?);
        }
        Ok(commits)
    }
}

/// Refreshes the subagent repository's snapshot of the main repository's refs
/// with a fetch that runs in the subagent's sandbox, so whatever it does stays
/// inside what the subagent can write.
async fn refresh(
    git: &SubagentGit,
    worktree: &CommittingWorktree,
    limits: &ImportLimits,
) -> Result<(), CommitLedgerError> {
    git.succeed(
        &worktree.repository,
        &[
            "fetch",
            "--quiet",
            "--prune",
            "--no-write-fetch-head",
            "--no-recurse-submodules",
            repository::MAIN_REMOTE,
        ],
        limits.bounds(),
    )
    .await
    .map(drop)
    .map_err(|source| CommitLedgerError::Export {
        repository: worktree.repository.clone(),
        source,
    })
}

/// The null device, which stands in for the global git configuration.
#[cfg(windows)]
const NULL_DEVICE: &str = "NUL";
#[cfg(not(windows))]
const NULL_DEVICE: &str = "/dev/null";

#[cfg(test)]
pub mod tests;
