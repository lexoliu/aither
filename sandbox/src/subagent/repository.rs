//! The repository of one committing worktree.
//!
//! The host creates it, once, when the worktree is first granted commit, and
//! the worktree's `.git` file then names it. It holds the worktree's `HEAD`,
//! index, refs and new objects; the main object store is its read-only
//! alternate and the main repository's configuration is included in its own.
//! Every path in it is relative, so the same files work under a container's
//! workspace mount.

use std::fmt;
use std::io;
use std::path::{Component, Path, PathBuf};

use askama::Template;
use futures_lite::io::AsyncWriteExt as _;

use super::host_git::{HostGit, input};
use super::{CommitLedgerError, SubagentGrantError};

/// The directory of a worktree's repository inside the worktree's entry in
/// the main repository.
pub(super) const REPOSITORY_DIR: &str = "subagent";

/// The file in the worktree's entry that records the branch namespace the
/// host granted. The repository is the subagent's to write, so what it says
/// of its own namespace is never believed.
pub(super) const NAMESPACE_FILE: &str = "subagent-namespace";

/// The name the main repository goes by in a worktree's repository.
pub(super) const MAIN_REMOTE: &str = "main-repository";

/// Where the pieces of one committing worktree live. Every path is canonical.
#[derive(Clone, Copy)]
pub(super) struct Layout<'a> {
    pub worktree: &'a Path,
    /// The worktree's entry in the main repository.
    pub entry: &'a Path,
    /// The main repository's shared git directory.
    pub common: &'a Path,
    /// The worktree's repository.
    pub repository: &'a Path,
    pub namespace: &'a str,
}

/// How the subagent's backend spells host paths: itself for the local
/// sandbox, through the workspace mount for a container.
#[derive(Clone, Copy)]
pub struct Spelling<'a> {
    mount: Option<&'a crate::MountRoot>,
}

impl<'a> Spelling<'a> {
    /// Host paths are spelled as they are.
    pub const fn host() -> Self {
        Self { mount: None }
    }

    /// Host paths are spelled as `mount` maps them.
    pub const fn through(mount: &'a crate::MountRoot) -> Self {
        Self { mount: Some(mount) }
    }

    fn text(self, path: &Path) -> Result<String, SubagentGrantError> {
        self.mount.map_or_else(
            || text(path),
            |mount| {
                mount
                    .container_path(path)
                    .map(|path| path.as_str().to_string())
                    .map_err(|source| SubagentGrantError::OutsideContainerMount {
                        path: path.to_path_buf(),
                        source,
                    })
            },
        )
    }
}

/// A value written into a git configuration file, quoted so that no
/// character in it ends or alters the line.
struct ConfigValue(String);

impl ConfigValue {
    fn new(value: &str) -> Result<Self, SubagentGrantError> {
        if value.contains(['\n', '\0']) {
            return Err(SubagentGrantError::UnrepresentablePath(PathBuf::from(
                value,
            )));
        }
        Ok(Self(value.to_string()))
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"")?;
        for character in self.0.chars() {
            if matches!(character, '"' | '\\') {
                f.write_str("\\")?;
            }
            write!(f, "{character}")?;
        }
        f.write_str("\"")
    }
}

/// One `extensions.*` key of the main repository.
struct Extension {
    key: String,
    value: ConfigValue,
}

#[derive(Template)]
#[template(path = "subagent_repository.gitconfig", escape = "none")]
struct RepositoryConfig {
    format_version: u32,
    hooks_path: ConfigValue,
    extensions: Vec<Extension>,
    include_path: ConfigValue,
    url: ConfigValue,
    positive_refspec: ConfigValue,
    negative_refspec: ConfigValue,
}

#[derive(Template)]
#[template(path = "subagent_head", escape = "none")]
struct Head<'a> {
    branch: &'a str,
}

#[derive(Template)]
#[template(path = "subagent_gitfile", escape = "none")]
struct GitFile {
    repository: String,
}

#[derive(Template)]
#[template(path = "subagent_alternates", escape = "none")]
struct Alternates {
    objects: String,
}

/// The refspec that keeps a fetch from the main repository away from the
/// worktree's own branch namespace.
fn negative_refspec(namespace: &str) -> String {
    format!("^refs/heads/subagent/{namespace}/*")
}

/// A path spelled the way git's own files take one: `/` between components
/// on every platform — a gitdir, alternates or include line asks git for it
/// back, and git reads `/` as a separator everywhere. On Unix the separator
/// already is `/` — and where it is not, a `\` is an ordinary file name
/// character nothing should rewrite.
fn text(path: &Path) -> Result<String, SubagentGrantError> {
    path.to_str()
        .filter(|text| !text.contains(['\n', '\0']))
        .map(|text| text.replace(std::path::MAIN_SEPARATOR, "/"))
        .ok_or_else(|| SubagentGrantError::UnrepresentablePath(path.to_path_buf()))
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> SubagentGrantError + '_ {
    move |source| SubagentGrantError::Repository {
        path: path.to_path_buf(),
        source,
    }
}

/// The relative path that leads from the directory `from` to `to`, both
/// canonical.
pub(super) fn relative(from: &Path, to: &Path) -> PathBuf {
    let mut from = from.components().peekable();
    let mut to = to.components().peekable();
    while let (Some(a), Some(b)) = (from.peek(), to.peek()) {
        if a != b {
            break;
        }
        from.next();
        to.next();
    }
    let mut path = PathBuf::new();
    for component in from {
        if matches!(component, Component::Normal(_)) {
            path.push("..");
        }
    }
    path.extend(to);
    path
}

/// Makes sure the worktree has its repository and names it in its `.git`
/// file.
pub(super) async fn ensure(
    layout: Layout<'_>,
    spelling: Spelling<'_>,
) -> Result<(), SubagentGrantError> {
    if async_fs::symlink_metadata(layout.repository).await.is_err() {
        create(layout, spelling).await?;
    }
    point_worktree_at_repository(layout).await
}

/// Builds the repository next to its final place and renames it into it, so
/// that it either exists whole or not at all.
async fn create(layout: Layout<'_>, spelling: Spelling<'_>) -> Result<(), SubagentGrantError> {
    let staging = layout.entry.join(format!(
        "{REPOSITORY_DIR}.{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    async_fs::create_dir(&staging)
        .await
        .map_err(io_error(&staging))?;
    let built = async {
        populate(layout, spelling, &staging).await?;
        async_fs::rename(&staging, layout.repository)
            .await
            .map_err(io_error(layout.repository))
    }
    .await;
    if built.is_err() {
        // Nothing but the host has touched it; failing to remove it only
        // leaves a directory the next creation does not use.
        let _ = async_fs::remove_dir_all(&staging).await;
    }
    built
}

async fn populate(
    layout: Layout<'_>,
    spelling: Spelling<'_>,
    staging: &Path,
) -> Result<(), SubagentGrantError> {
    let write = |relative: &str, content: String| {
        let path = staging.join(relative);
        async move {
            async_fs::write(&path, content)
                .await
                .map_err(io_error(&path))
        }
    };
    for directory in [
        "objects/info",
        "objects/pack",
        "refs/heads",
        "refs/tags",
        "info",
    ] {
        let path = staging.join(directory);
        async_fs::create_dir_all(&path)
            .await
            .map_err(io_error(&path))?;
    }

    let head = async_fs::read_to_string(layout.entry.join("HEAD"))
        .await
        .map_err(io_error(&layout.entry.join("HEAD")))?;
    let branch =
        head.trim()
            .strip_prefix("ref: ")
            .ok_or_else(|| SubagentGrantError::MalformedGitData {
                path: layout.worktree.to_path_buf(),
                reason: "the worktree's HEAD is not a branch",
            })?;
    write(
        "HEAD",
        Head { branch }.render().expect("the HEAD template renders"),
    )
    .await?;

    // The alternate is relative to the final place of the object directory.
    let objects = relative(
        &layout.repository.join("objects"),
        &layout.common.join("objects"),
    );
    write(
        "objects/info/alternates",
        Alternates {
            objects: text(&objects)?,
        }
        .render()
        .expect("the alternates template renders"),
    )
    .await?;

    write("config", config(layout, spelling).await?).await?;

    // What the main agent staged and excluded carries over.
    for (source, target) in [
        (layout.entry.join("index"), "index"),
        (layout.common.join("info/exclude"), "info/exclude"),
        (layout.common.join("info/attributes"), "info/attributes"),
        (
            layout.entry.join("info/sparse-checkout"),
            "info/sparse-checkout",
        ),
        (layout.common.join("shallow"), "shallow"),
    ] {
        match async_fs::copy(&source, staging.join(target)).await {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source_error) => return Err(io_error(&source)(source_error)),
        }
    }

    // The worktree's branch, then a snapshot of the main refs. No object
    // moves: they are all reachable through the alternate.
    let common = text(layout.common)?;
    let branch_tip = HostGit::in_repository(
        layout.common,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{branch}^{{commit}}"),
        ],
    )
    .output(input(Vec::new()))
    .await?;
    if branch_tip.status.success() {
        let tip = String::from_utf8_lossy(&branch_tip.stdout)
            .trim()
            .to_string();
        HostGit::in_repository(staging, &["update-ref", branch, &tip])
            .succeed(Vec::new())
            .await?;
    }
    let negative = negative_refspec(layout.namespace);
    HostGit::in_repository(
        staging,
        &[
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "--no-recurse-submodules",
            &common,
            "+refs/*:refs/*",
            &negative,
        ],
    )
    .succeed(Vec::new())
    .await?;

    let namespace = layout.entry.join(NAMESPACE_FILE);
    async_fs::write(&namespace, layout.namespace)
        .await
        .map_err(io_error(&namespace))
}

/// Renders the configuration of the repository: the main repository's own
/// format, then its whole configuration, then what makes the repository a
/// worktree's.
async fn config(layout: Layout<'_>, spelling: Spelling<'_>) -> Result<String, SubagentGrantError> {
    // The format keys are not honored through an include.
    let keys = HostGit::in_repository(
        layout.common,
        &[
            "config",
            "-z",
            "--get-regexp",
            r"^(core\.repositoryformatversion|extensions\.)",
        ],
    )
    .output(input(Vec::new()))
    .await?;
    let mut format_version = 0;
    let mut extensions = Vec::new();
    for entry in keys
        .stdout
        .split(|byte| *byte == 0)
        .filter(|e| !e.is_empty())
    {
        let entry = String::from_utf8_lossy(entry);
        let (key, value) = entry.split_once('\n').unwrap_or((&entry, ""));
        if key == "core.repositoryformatversion" {
            format_version = value.parse().map_err(|_| {
                SubagentGrantError::from(CommitLedgerError::Output {
                    repository: layout.common.to_path_buf(),
                    line: entry.to_string(),
                })
            })?;
        } else if let Some(extension) = key.strip_prefix("extensions.")
            && extension != "refstorage"
        {
            if !extension.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
                return Err(CommitLedgerError::Output {
                    repository: layout.common.to_path_buf(),
                    line: entry.to_string(),
                }
                .into());
            }
            extensions.push(Extension {
                key: extension.to_string(),
                value: ConfigValue::new(value)?,
            });
        }
    }

    // Hooks run in the worktree; the main repository's configuration, which
    // follows, wins when it names hooks of its own.
    let hooks = relative(layout.worktree, &layout.common.join("hooks"));
    let include = relative(layout.repository, &layout.common.join("config"));
    Ok(RepositoryConfig {
        format_version,
        hooks_path: ConfigValue::new(&text(&hooks)?)?,
        extensions,
        include_path: ConfigValue::new(&text(&include)?)?,
        url: ConfigValue::new(&spelling.text(layout.common)?)?,
        positive_refspec: ConfigValue::new("+refs/*:refs/*")?,
        negative_refspec: ConfigValue::new(&negative_refspec(layout.namespace))?,
    }
    .render()
    .expect("the repository configuration template renders"))
}

/// Replaces the worktree's `.git` file with one that names the repository.
///
/// The new file is written next to the old one and renamed over it: a rename
/// replaces a symlink without following it and fails on a directory, so
/// nothing a subagent swapped into the path can make the host write
/// elsewhere.
async fn point_worktree_at_repository(layout: Layout<'_>) -> Result<(), SubagentGrantError> {
    let repository = relative(layout.worktree, layout.repository);
    let wanted = GitFile {
        repository: text(&repository)?,
    }
    .render()
    .expect("the gitfile template renders");
    let dot_git = layout.worktree.join(".git");
    // Whatever else is there (a symlink, a FIFO, something too long to be a
    // gitdir line) is replaced, never followed or waited on.
    if matches!(
        super::read_dot_git(&dot_git).await,
        Ok(super::DotGit::File(current)) if current == wanted
    ) {
        return Ok(());
    }
    let staged = layout
        .worktree
        .join(format!(".git.{}.tmp", uuid::Uuid::new_v4().simple()));
    let written = async {
        let mut file = async_fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged)
            .await?;
        file.write_all(wanted.as_bytes()).await?;
        file.flush().await?;
        drop(file);
        async_fs::rename(&staged, &dot_git).await
    }
    .await;
    if let Err(source) = written {
        let _ = async_fs::remove_file(&staged).await;
        return Err(io_error(&dot_git)(source));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_path_climbs_to_the_common_ancestor_and_descends() {
        assert_eq!(
            relative(
                Path::new("/r/.git/worktrees/w/subagent/objects"),
                Path::new("/r/.git/objects")
            ),
            Path::new("../../../../objects")
        );
        assert_eq!(
            relative(Path::new("/r/w"), Path::new("/r/.git/worktrees/w/subagent")),
            Path::new("../.git/worktrees/w/subagent")
        );
        assert_eq!(relative(Path::new("/a"), Path::new("/a")), Path::new(""));
    }

    #[test]
    fn a_config_value_is_quoted_whatever_it_holds() {
        assert_eq!(
            ConfigValue::new(r#"a "b" \c ; #d"#).unwrap().to_string(),
            r#""a \"b\" \\c ; #d""#
        );
        assert!(ConfigValue::new("one\ntwo").is_err());
    }
}
