//! The host's import of a subagent's commits into the main repository.
//!
//! Nothing the subagent wrote is opened by the host. Git running inside the
//! subagent's sandbox, as the subagent, says what the subagent's refs are and
//! packs what the main repository lacks; the host feeds that pack to its own
//! `git index-pack --strict` in a quarantine directory only it writes, checks
//! that every tip arrived whole, moves the pack into the main store and moves
//! the refs in one transaction. The quarantine is removed whatever happens, and
//! no ref moves unless everything the refs name is in the main store.

use std::path::Path;

use askama::Template;
use futures::{StreamExt as _, TryStreamExt as _};

use super::host_git::{HostGit, Quarantine, input};
use super::sandboxed::{Bounds, ExportError, PIPE_DEPTH, SandboxedRun, SubagentGit, within};
use super::{CommitLedgerError, CommittingWorktree, ImportLimits, object_id};

/// The prefix of the directory a pack is indexed in. It is not git's own
/// quarantine prefix: git's receive-pack uses that, and nothing here removes a
/// directory it did not create.
const QUARANTINE_PREFIX: &str = "aither-import-";

/// How many reported ref names are checked at once.
const REF_CHECKS_IN_FLIGHT: usize = 8;

/// Where a `HEAD` points.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Head {
    /// A branch.
    Symbolic(String),
    /// A commit.
    Detached(String),
}

/// What the subagent's repository says about itself.
struct Report {
    /// `(ref name, object id)`, every ref of the worktree's namespace.
    refs: Vec<(String, String)>,
    head: Head,
}

impl Report {
    /// Every commit the report names, once.
    fn tips(&self) -> Vec<String> {
        let mut tips: Vec<String> = self.refs.iter().map(|(_, id)| id.clone()).collect();
        if let Head::Detached(id) = &self.head {
            tips.push(id.clone());
        }
        tips.sort_unstable();
        tips.dedup();
        tips
    }
}

#[derive(Template)]
#[template(path = "subagent_object_ids", escape = "none")]
struct ObjectIds<'a> {
    ids: &'a [String],
}

#[derive(Template)]
#[template(path = "subagent_revisions", escape = "none")]
struct Revisions<'a> {
    want: &'a [String],
    have: &'a [String],
}

struct RefUpdate<'a> {
    name: &'a str,
    new: &'a str,
    old: &'a str,
}

struct RefCreate<'a> {
    name: &'a str,
    new: &'a str,
}

struct RefDelete<'a> {
    name: &'a str,
    old: &'a str,
}

#[derive(Template)]
#[template(path = "subagent_ref_transaction", escape = "none")]
struct RefTransaction<'a> {
    updates: Vec<RefUpdate<'a>>,
    creates: Vec<RefCreate<'a>>,
    deletes: Vec<RefDelete<'a>>,
    /// The commit the worktree's entry is to be detached at, and the ref or
    /// object id it must still resolve to - a HEAD that moved between the
    /// report and the import is not overwritten.
    detached: Option<(&'a str, &'a str)>,
}

impl RefTransaction<'_> {
    const fn is_empty(&self) -> bool {
        self.updates.is_empty()
            && self.creates.is_empty()
            && self.deletes.is_empty()
            && self.detached.is_none()
    }

    /// Whether the transaction moves `name`.
    fn touches(&self, name: &str) -> bool {
        self.updates.iter().any(|update| update.name == name)
            || self.creates.iter().any(|create| create.name == name)
            || self.deletes.iter().any(|delete| delete.name == name)
    }
}

/// A directory only the host writes, in which a pack is indexed before the
/// main store sees any of it.
struct PackQuarantine {
    directory: tempfile::TempDir,
}

impl PackQuarantine {
    async fn create(objects: &Path) -> Result<Self, CommitLedgerError> {
        let under = objects.to_path_buf();
        let directory = blocking::unblock(move || {
            tempfile::Builder::new()
                .prefix(QUARANTINE_PREFIX)
                .tempdir_in(&under)
        })
        .await
        .map_err(|source| CommitLedgerError::Quarantine {
            path: objects.to_path_buf(),
            source,
        })?;
        let pack = directory.path().join("pack");
        async_fs::create_dir(&pack)
            .await
            .map_err(|source| CommitLedgerError::Quarantine { path: pack, source })?;
        Ok(Self { directory })
    }

    fn path(&self) -> &Path {
        self.directory.path()
    }

    /// Moves the indexed pack into the main store, its index last: git finds
    /// a pack by its index, so the pack is whole when it can be found.
    async fn publish(&self, objects: &Path) -> Result<(), CommitLedgerError> {
        let from = self.path().join("pack");
        let to = objects.join("pack");
        let error = |path: &Path| {
            let path = path.to_path_buf();
            move |source| CommitLedgerError::Quarantine { path, source }
        };
        let mut files = Vec::new();
        let mut entries = async_fs::read_dir(&from).await.map_err(error(&from))?;
        while let Some(entry) = entries.next().await {
            files.push(entry.map_err(error(&from))?.file_name());
        }
        // Index last; anything git left half-written is not a pack file.
        files.retain(|name| name.to_string_lossy().starts_with("pack-"));
        files.sort_by_key(|name| name.to_string_lossy().ends_with(".idx"));
        let mut moved = Vec::new();
        for name in files {
            let target = to.join(&name);
            if let Err(source) = async_fs::rename(from.join(&name), &target).await {
                for path in moved {
                    let _ = async_fs::remove_file(path).await;
                }
                return Err(CommitLedgerError::Quarantine {
                    path: target,
                    source,
                });
            }
            moved.push(target);
        }
        Ok(())
    }
}

impl CommittingWorktree {
    /// Brings the commits the subagent made into the main repository.
    ///
    /// The main repository changes only here, and only once every tip the
    /// subagent reports is in the main store, connected to it, and the refs
    /// can move from the values they had when the report was read.
    pub(super) async fn import(
        &self,
        git: &SubagentGit,
        limits: &ImportLimits,
    ) -> Result<(), CommitLedgerError> {
        // The bounds in force when the import starts hold for all of it.
        let bounds = limits.bounds();
        let report = self.report(git, bounds).await?;
        let current = self.namespace_refs().await?;
        let entry_head = self.entry_head().await?;
        let tips = report.tips();
        let missing = self.missing_from_main(&tips).await?;
        if missing.is_empty() && Self::unchanged(&report, &current, &entry_head) {
            return Ok(());
        }

        let have = self.main_tips().await?;
        let objects = self.common_dir.join("objects");
        let quarantine = if missing.is_empty() {
            None
        } else {
            Some(
                self.indexed_pack(git, bounds, &objects, &missing, &have)
                    .await?,
            )
        };
        self.verify(&tips, &missing, &have, quarantine.as_ref(), &objects)
            .await?;
        if let Some(quarantine) = &quarantine {
            quarantine.publish(&objects).await?;
        }
        self.move_refs(&report, &current, &entry_head).await?;
        drop(quarantine);

        if !missing.is_empty() {
            // What the main store has now needs no second copy.
            if let Err(error) = git
                .succeed(&self.repository, &["prune-packed"], bounds)
                .await
            {
                tracing::warn!(
                    repository = %self.repository.display(),
                    %error,
                    "cannot drop the imported objects from the subagent's repository"
                );
            }
        }
        Ok(())
    }

    fn export_error(&self, source: ExportError) -> CommitLedgerError {
        CommitLedgerError::Export {
            repository: self.repository.clone(),
            source,
        }
    }

    fn report_error(&self, line: impl Into<String>) -> CommitLedgerError {
        CommitLedgerError::Report {
            repository: self.repository.clone(),
            line: line.into(),
        }
    }

    /// Asks the subagent's repository, from inside its sandbox, what its refs
    /// are, and checks every answer.
    async fn report(&self, git: &SubagentGit, bounds: Bounds) -> Result<Report, CommitLedgerError> {
        let hex_length = self.hex_length().await?;
        let prefix = self.ref_prefix();
        let listing = git
            .succeed(
                &self.repository,
                &[
                    "for-each-ref",
                    "--format=%(objectname) %(refname)",
                    prefix.as_str(),
                ],
                bounds,
            )
            .await
            .map_err(|source| self.export_error(source))?;
        let listing = String::from_utf8_lossy(&listing);
        let mut refs = Vec::new();
        for line in listing.lines() {
            let (id, name) = line
                .split_once(' ')
                .filter(|(id, name)| {
                    id.len() == hex_length
                        && object_id(id).is_some()
                        && name.starts_with(&prefix)
                        && name.len() > prefix.len()
                })
                .ok_or_else(|| self.report_error(line))?;
            refs.push((name.to_string(), id.to_string()));
        }

        let head = git
            .collect(&self.repository, &["symbolic-ref", "-q", "HEAD"], bounds)
            .await
            .map_err(|source| self.export_error(source))?;
        let head = match head.exit.code {
            Some(0) => {
                let target = String::from_utf8_lossy(&head.stdout).trim().to_string();
                if !target.starts_with(&prefix) || target.len() == prefix.len() {
                    return Err(self.report_error(target));
                }
                Head::Symbolic(target)
            }
            // Exit 1 is how `-q` says HEAD is not a symbolic ref.
            Some(1) => {
                let id = git
                    .succeed(&self.repository, &["rev-parse", "--verify", "HEAD"], bounds)
                    .await
                    .map_err(|source| self.export_error(source))?;
                let id = String::from_utf8_lossy(&id).trim().to_string();
                if id.len() != hex_length || object_id(&id).is_none() {
                    return Err(self.report_error(id));
                }
                Head::Detached(id)
            }
            _ => return Err(self.export_error(head.exit.into_failure())),
        };

        // A name that git itself refuses, or that is not one line, is not one
        // the host moves a ref to.
        let names: Vec<String> = refs
            .iter()
            .map(|(name, _)| name.clone())
            .chain(match &head {
                Head::Symbolic(name) => Some(name.clone()),
                Head::Detached(_) => None,
            })
            .collect();
        // The names are as many as the subagent made them, so checking them is
        // bounded like everything else it can make large.
        within(
            bounds.time,
            futures::stream::iter(names)
                .map(|name: String| async move {
                    let output =
                        HostGit::in_repository(&self.common_dir, &["check-ref-format", &name])
                            .output(input(Vec::new()))
                            .await?;
                    if output.status.success() {
                        Ok(())
                    } else {
                        Err(self.report_error(name))
                    }
                })
                .buffer_unordered(REF_CHECKS_IN_FLIGHT)
                .try_collect::<()>(),
        )
        .await
        .ok_or_else(|| self.export_error(ExportError::TimedOut(bounds.time)))??;
        refs.sort_unstable();
        Ok(Report { refs, head })
    }

    /// The length of an object id in the main repository's hash.
    async fn hex_length(&self) -> Result<usize, CommitLedgerError> {
        let format =
            super::host_git::git(&self.common_dir, &["rev-parse", "--show-object-format"]).await?;
        match format.trim() {
            "sha1" => Ok(40),
            "sha256" => Ok(64),
            other => Err(CommitLedgerError::Output {
                repository: self.common_dir.clone(),
                line: other.to_string(),
            }),
        }
    }

    /// Where the worktree's entry in the main repository says `HEAD` is. Only
    /// the host writes it.
    async fn entry_head(&self) -> Result<Head, CommitLedgerError> {
        let path = self.git_dir.join("HEAD");
        let head =
            async_fs::read_to_string(&path)
                .await
                .map_err(|source| CommitLedgerError::Read {
                    path: path.clone(),
                    source,
                })?;
        let head = head.trim();
        head.strip_prefix("ref: ").map_or_else(
            || {
                if object_id(head).is_some() {
                    Ok(Head::Detached(head.to_string()))
                } else {
                    Err(CommitLedgerError::Output {
                        repository: self.git_dir.clone(),
                        line: head.to_string(),
                    })
                }
            },
            |branch| Ok(Head::Symbolic(branch.to_string())),
        )
    }

    /// Whether the main repository already says what the subagent's does.
    fn unchanged(report: &Report, current: &[(String, String)], entry: &Head) -> bool {
        let mut current: Vec<(&str, &str)> = current
            .iter()
            .map(|(id, name)| (name.as_str(), id.as_str()))
            .collect();
        current.sort_unstable();
        let reported: Vec<(&str, &str)> = report
            .refs
            .iter()
            .map(|(name, id)| (name.as_str(), id.as_str()))
            .collect();
        current == reported && report.head == *entry
    }

    /// The reported commits the main store does not hold.
    async fn missing_from_main(&self, tips: &[String]) -> Result<Vec<String>, CommitLedgerError> {
        if tips.is_empty() {
            return Ok(Vec::new());
        }
        let checked = HostGit::in_repository(&self.common_dir, &["cat-file", "--batch-check"])
            .succeed(render(&ObjectIds { ids: tips }).into_bytes())
            .await?;
        let checked = String::from_utf8_lossy(&checked.stdout);
        checked
            .lines()
            .zip(tips)
            .filter_map(|(line, tip)| {
                let mut parts = line.split(' ');
                match (parts.next(), parts.next()) {
                    (Some(id), Some("missing")) if id == tip => Some(Ok(tip.clone())),
                    (Some(id), Some(_)) if id == tip => None,
                    _ => Some(Err(CommitLedgerError::Output {
                        repository: self.common_dir.clone(),
                        line: line.to_string(),
                    })),
                }
            })
            .collect()
    }

    /// Every object id a ref of the main repository names: what the subagent
    /// need not send.
    async fn main_tips(&self) -> Result<Vec<String>, CommitLedgerError> {
        let listing = super::host_git::git(
            &self.common_dir,
            &["for-each-ref", "--format=%(objectname)"],
        )
        .await?;
        let mut have: Vec<String> = listing.lines().map(str::to_string).collect();
        have.sort_unstable();
        have.dedup();
        Ok(have)
    }

    /// Has the subagent's sandbox pack what the main store lacks and indexes
    /// the pack in a quarantine, as it arrives.
    async fn indexed_pack(
        &self,
        git: &SubagentGit,
        bounds: Bounds,
        objects: &Path,
        want: &[String],
        have: &[String],
    ) -> Result<PackQuarantine, CommitLedgerError> {
        let quarantine = PackQuarantine::create(objects).await?;
        let view = Quarantine {
            objects: quarantine.path(),
            alternate: objects,
        };
        let max_input = bounds
            .bytes
            .map(|bytes| format!("--max-input-size={bytes}"));
        let (pack_tx, pack_rx) = async_channel::bounded(PIPE_DEPTH);
        let export = git.run(SandboxedRun {
            repository: &self.repository,
            args: &["pack-objects", "--revs", "--stdout", "--local"],
            stdin: render(&Revisions { want, have }).into_bytes(),
            stdout: pack_tx,
        });
        let mut index_args = vec!["index-pack", "--stdin", "--strict"];
        index_args.extend(max_input.as_deref());
        let indexer = HostGit::in_repository(&self.common_dir, &index_args).through(view);
        let index = indexer.output(pack_rx);
        let (export, index) = within(bounds.time, futures::future::join(export, index))
            .await
            .ok_or_else(|| self.export_error(ExportError::TimedOut(bounds.time)))?;

        // A sandboxed git that failed is the better explanation than the
        // truncated pack the host then refused; one that merely found nobody
        // reading explains nothing.
        match export {
            Err(ExportError::OutputClosed) => {}
            Err(source) => return Err(self.export_error(source)),
            Ok(exit) if !exit.success() => return Err(self.export_error(exit.into_failure())),
            Ok(_) => {}
        }
        let index = index?;
        if !index.status.success() {
            return Err(CommitLedgerError::Rejected {
                repository: self.repository.clone(),
                stderr: String::from_utf8_lossy(&index.stderr).trim().to_string(),
            });
        }
        Ok(quarantine)
    }

    /// Checks, in the view the import will leave behind, that every reported
    /// tip is a commit and everything reachable from the new ones is present.
    async fn verify(
        &self,
        tips: &[String],
        want: &[String],
        have: &[String],
        quarantine: Option<&PackQuarantine>,
        objects: &Path,
    ) -> Result<(), CommitLedgerError> {
        let host = |args: &'static [&'static str]| {
            let git = HostGit::in_repository(&self.common_dir, args);
            match quarantine {
                Some(quarantine) => git.through(Quarantine {
                    objects: quarantine.path(),
                    alternate: objects,
                }),
                None => git,
            }
        };
        let incomplete = |detail: String| CommitLedgerError::Incomplete {
            repository: self.repository.clone(),
            detail,
        };
        if tips.is_empty() {
            return Ok(());
        }
        let checked = host(&["cat-file", "--batch-check"])
            .succeed(render(&ObjectIds { ids: tips }).into_bytes())
            .await?;
        let checked = String::from_utf8_lossy(&checked.stdout);
        for (line, tip) in checked.lines().zip(tips) {
            let mut parts = line.split(' ');
            if (parts.next(), parts.next()) != (Some(tip.as_str()), Some("commit")) {
                return Err(incomplete(line.to_string()));
            }
        }
        if want.is_empty() {
            return Ok(());
        }
        let walked = host(&["rev-list", "--objects", "--quiet", "--stdin"])
            .output(input(render(&Revisions { want, have }).into_bytes()))
            .await?;
        if walked.status.success() {
            Ok(())
        } else {
            Err(incomplete(
                String::from_utf8_lossy(&walked.stderr).trim().to_string(),
            ))
        }
    }

    /// Moves the namespace refs, and the entry's `HEAD`, to what the subagent
    /// reported, in one transaction that fails if any ref is no longer where
    /// the report was read from.
    async fn move_refs(
        &self,
        report: &Report,
        current: &[(String, String)],
        entry_head: &Head,
    ) -> Result<(), CommitLedgerError> {
        let old_of = |name: &str| {
            current
                .iter()
                .find(|(_, current)| current == name)
                .map(|(id, _)| id.as_str())
        };
        let mut transaction = RefTransaction {
            updates: Vec::new(),
            creates: Vec::new(),
            deletes: Vec::new(),
            detached: None,
        };
        for (name, new) in &report.refs {
            match old_of(name) {
                Some(old) if old == new => {}
                Some(old) => transaction.updates.push(RefUpdate { name, new, old }),
                None => transaction.creates.push(RefCreate { name, new }),
            }
        }
        for (old, name) in current {
            if !report.refs.iter().any(|(reported, _)| reported == name) {
                transaction.deletes.push(RefDelete { name, old });
            }
        }
        // A symbolic HEAD cannot move inside the transaction: symref-update
        // on a worktree entry lands on a per-worktree ref rather than its
        // HEAD file, so a symbolic move follows the committed refs through
        // symbolic-ref, retargeted only while HEAD still says what the
        // report read.
        let mut symbolic_after = None;
        if report.head != *entry_head {
            match &report.head {
                // The value the entry's HEAD held at the report goes into
                // the update as its expected old value.
                Head::Detached(id) => {
                    transaction.detached = Some((
                        id,
                        match entry_head {
                            Head::Detached(oid) | Head::Symbolic(oid) => oid,
                        },
                    ));
                }
                Head::Symbolic(branch) => symbolic_after = Some(branch),
            }
        }
        // Updating the ref the entry's HEAD points at counts as a second
        // update of HEAD, which update-ref refuses; a detached HEAD then
        // moves in a follow-up transaction of its own. One that fails
        // leaves the entry's HEAD behind — recoverable, since the next
        // import reads the difference and tries again.
        let head_after = if let Head::Symbolic(referent) = entry_head
            && transaction.touches(referent)
        {
            Some(RefTransaction {
                updates: Vec::new(),
                creates: Vec::new(),
                deletes: Vec::new(),
                detached: transaction.detached.take(),
            })
        } else {
            None
        };
        for next in [Some(&transaction), head_after.as_ref()]
            .into_iter()
            .flatten()
        {
            if next.is_empty() {
                continue;
            }
            // Once it is under way it is left to finish: killed, it would
            // leave the lock files of the refs it was moving behind.
            let applied = HostGit::in_repository(&self.git_dir, &["update-ref", "--stdin"])
                .finish_when_dropped()
                .output(input(render(next).into_bytes()))
                .await?;
            if !applied.status.success() {
                let now = self.namespace_refs().await?;
                return Err(match moved_ref(current, &now) {
                    Some(reference) => CommitLedgerError::Moved {
                        repository: self.repository.clone(),
                        reference,
                    },
                    None => CommitLedgerError::Git {
                        repository: self.common_dir.clone(),
                        stderr: String::from_utf8_lossy(&applied.stderr).trim().to_string(),
                    },
                });
            }
        }
        if let Some(target) = symbolic_after {
            self.mirror_symbolic_head(target, entry_head).await?;
        }
        Ok(())
    }

    /// Points the entry's `HEAD` at `target` with `symbolic-ref`, the one
    /// write that reaches a worktree entry's HEAD file, refusing while it
    /// no longer names the branch the report found.
    async fn mirror_symbolic_head(
        &self,
        target: &str,
        entry_head: &Head,
    ) -> Result<(), CommitLedgerError> {
        let now = HostGit::in_repository(&self.git_dir, &["symbolic-ref", "-q", "HEAD"])
            .output(input(Vec::new()))
            .await?;
        if let Head::Symbolic(old) = entry_head {
            let current = String::from_utf8_lossy(&now.stdout);
            if !now.status.success() || current.trim() != *old {
                return Err(CommitLedgerError::Moved {
                    repository: self.git_dir.clone(),
                    reference: "HEAD".to_string(),
                });
            }
        }
        let moved = HostGit::in_repository(&self.git_dir, &["symbolic-ref", "HEAD", target])
            .output(input(Vec::new()))
            .await?;
        if !moved.status.success() {
            return Err(CommitLedgerError::Git {
                repository: self.git_dir.clone(),
                stderr: String::from_utf8_lossy(&moved.stderr).trim().to_string(),
            });
        }
        Ok(())
    }
}

/// The first ref whose value differs between two readings of a namespace.
fn moved_ref(before: &[(String, String)], after: &[(String, String)]) -> Option<String> {
    before
        .iter()
        .find(|entry| !after.contains(entry))
        .or_else(|| after.iter().find(|entry| !before.contains(entry)))
        .map(|(_, name)| name.clone())
}

fn render(template: &impl Template) -> String {
    template
        .render()
        .expect("the import templates render any input")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_detached_head_move_carries_its_expected_value() {
        let transaction = RefTransaction {
            updates: Vec::new(),
            creates: Vec::new(),
            deletes: Vec::new(),
            detached: Some(("new-oid", "refs/heads/old")),
        };
        let rendered = render(&transaction);
        assert!(
            rendered.contains("option no-deref\nupdate HEAD new-oid refs/heads/old"),
            "{rendered}"
        );
    }
}
