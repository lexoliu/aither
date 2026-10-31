//! Tests of the commit grant and the import, with real git repositories in
//! temporary directories. The subagent is played by git run in its worktree,
//! and its sandbox by [`TestGit`], which runs git unsandboxed and can lie.

// Style lints that cost readability in test fixtures.
#![allow(
    clippy::assert_is_empty,
    clippy::cast_possible_truncation,
    clippy::cloned_ref_to_slice_refs,
    clippy::format_push_string,
    clippy::literal_string_with_formatting_args,
    clippy::similar_names,
    clippy::significant_drop_tightening,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unused_self,
    clippy::write_with_newline
)]

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use super::sandboxed::within;
use super::*;

/// A fixed moment for every commit, so that commits made in the same second
/// are still the same commits on every run.
const MOMENT: &str = "1700000000 +0000";

/// Runs git in `dir` with a fixed identity and no user configuration.
pub fn git_in(dir: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", NULL_DEVICE)
        .env("GIT_AUTHOR_DATE", MOMENT)
        .env("GIT_COMMITTER_DATE", MOMENT)
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

/// Whether git, run in `dir`, exits successfully.
fn git_succeeds(dir: &Path, args: &[&str]) -> bool {
    std::process::Command::new("git")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", NULL_DEVICE)
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git runs")
        .status
        .success()
}

/// A repository with one commit and a linked worktree on the branch
/// `subagent/<namespace>/work`, both inside `root`; returns the worktree.
pub fn repository_with_linked_worktree(root: &Path, namespace: &str) -> PathBuf {
    let repository = root.join("repository");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(&repository, &["init", "--initial-branch=main"]);
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    add_worktree(&repository, root, namespace)
}

/// Adds the worktree `<parent>/<namespace>` on the branch
/// `subagent/<namespace>/work` to `repository`.
fn add_worktree(repository: &Path, parent: &Path, namespace: &str) -> PathBuf {
    let worktree = parent.join(namespace);
    let branch = format!("subagent/{namespace}/work");
    git_in(
        repository,
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
    let writable = [dunce::canonicalize(root).unwrap()];
    let spaces_root = writable[0].join("subagent-spaces");
    grant
        .resolve(
            Launcher {
                writable: &writable,
                committing: None,
                spaces_root: &spaces_root,
                spelling: Spelling::host(),
            },
            history,
        )
        .await
}

/// What the sandbox's git does for one command that is not just run.
enum Reply {
    /// Prints these bytes and exits with this code instead of running.
    Print { code: i32, stdout: Vec<u8> },
    /// Never finishes (a long `sleep` stands in for it).
    Hang,
}

/// Plays a subagent's sandbox: git run unsandboxed against the subagent's
/// repository, or a lie told in its place.
#[derive(Default)]
pub struct TestGit {
    replies: Mutex<Vec<(String, Reply)>>,
    hooks: Mutex<Vec<(String, Box<dyn Fn() + Send + Sync>)>>,
    ran: Mutex<Vec<Vec<String>>>,
    hung: Mutex<Vec<u32>>,
}

impl TestGit {
    fn new() -> Arc<Self> {
        Arc::default()
    }

    fn replying(self: Arc<Self>, command: &str, reply: Reply) -> Arc<Self> {
        self.replies.lock().unwrap().push((command.into(), reply));
        self
    }

    fn before(
        self: Arc<Self>,
        command: &str,
        hook: impl Fn() + Send + Sync + 'static,
    ) -> Arc<Self> {
        self.hooks
            .lock()
            .unwrap()
            .push((command.into(), Box::new(hook)));
        self
    }

    /// Every command run so far.
    fn ran(&self) -> Vec<Vec<String>> {
        self.ran.lock().unwrap().clone()
    }

    fn ran_command(&self, command: &str) -> bool {
        self.ran().iter().any(|args| args[0] == command)
    }

    fn git(self: &Arc<Self>) -> SubagentGit {
        SubagentGit::new(self.clone())
    }
}

impl SandboxedGitImpl for TestGit {
    fn run<'a>(
        &'a self,
        run: SandboxedRun<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<SandboxedExit, ExportError>> + Send + 'a>> {
        Box::pin(async move {
            let args: Vec<String> = run.args.iter().map(ToString::to_string).collect();
            self.ran.lock().unwrap().push(args.clone());
            let command = args[0].as_str();
            for (hooked, hook) in self.hooks.lock().unwrap().iter() {
                if hooked == command {
                    hook();
                }
            }
            let reply = {
                let replies = self.replies.lock().unwrap();
                replies
                    .iter()
                    .find(|(replied, _)| replied == command)
                    .map(|(_, reply)| match reply {
                        Reply::Print { code, stdout } => Reply::Print {
                            code: *code,
                            stdout: stdout.clone(),
                        },
                        Reply::Hang => Reply::Hang,
                    })
            };
            match reply {
                Some(Reply::Print { code, stdout }) => {
                    for chunk in stdout.chunks(4096) {
                        run.stdout
                            .send(chunk.to_vec())
                            .await
                            .map_err(|_| ExportError::OutputClosed)?;
                    }
                    return Ok(SandboxedExit {
                        code: Some(code),
                        stderr: String::new(),
                    });
                }
                Some(Reply::Hang) => {
                    let mut child = async_process::Command::new("sleep")
                        .arg("60")
                        .kill_on_drop(true)
                        .spawn()?;
                    self.hung.lock().unwrap().push(child.id());
                    let status = child.status().await?;
                    return Ok(SandboxedExit {
                        code: status.code(),
                        stderr: String::new(),
                    });
                }
                None => {}
            }

            let mut command = async_process::Command::new("git");
            command
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_CONFIG_GLOBAL", NULL_DEVICE)
                .arg("--git-dir")
                .arg(run.repository)
                .args(&args)
                .stdin(async_process::Stdio::piped())
                .stdout(async_process::Stdio::piped())
                .stderr(async_process::Stdio::piped())
                .kill_on_drop(true);
            let mut child = command.spawn()?;
            let mut stdin = child.stdin.take().expect("piped");
            let mut stdout = child.stdout.take().expect("piped");
            let stdin_bytes = run.stdin;
            let feed = async move {
                use futures_lite::io::AsyncWriteExt as _;
                let _ = stdin.write_all(&stdin_bytes).await;
            };
            let forward = async {
                use futures_lite::io::AsyncReadExt as _;
                let mut buffer = vec![0_u8; 8192];
                loop {
                    let read = stdout.read(&mut buffer).await?;
                    if read == 0 {
                        return Ok::<_, ExportError>(());
                    }
                    run.stdout
                        .send(buffer[..read].to_vec())
                        .await
                        .map_err(|_| ExportError::OutputClosed)?;
                }
            };
            let (forwarded, ()) = futures::join!(forward, feed);
            forwarded?;
            let output = child.output().await?;
            Ok(SandboxedExit {
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        })
    }
}

/// A repository, its subagent worktree and a ledger over them.
struct Fixture {
    dir: tempfile::TempDir,
    /// The worktree, as the subagent sees it.
    worktree: PathBuf,
    /// The main repository's git directory.
    common: PathBuf,
    /// The worktree's entry in it.
    entry: PathBuf,
    /// The subagent's repository.
    subagent: PathBuf,
    resolved: ResolvedGrant,
    stub: Arc<TestGit>,
    ledger: CommitLedger,
}

const ONE_MIB: u64 = 1024 * 1024;

fn limits() -> ImportLimits {
    ImportLimits {
        bytes: ImportBudget::new(256 * ONE_MIB),
        time: Duration::from_secs(60),
    }
}

impl Fixture {
    async fn new(namespace: &str) -> Self {
        Self::with(namespace, TestGit::new(), limits()).await
    }

    async fn with(namespace: &str, stub: Arc<TestGit>, limits: ImportLimits) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let worktree = repository_with_linked_worktree(dir.path(), namespace);
        Self::granted(dir, worktree, stub, limits).await
    }

    async fn granted(
        dir: tempfile::TempDir,
        worktree: PathBuf,
        stub: Arc<TestGit>,
        limits: ImportLimits,
    ) -> Self {
        let resolved = main_launch(
            dir.path(),
            &SubagentGrant::new().with_committing_worktree(&worktree),
            &mut LaunchHistory::default(),
        )
        .await
        .expect("a worktree on a namespaced branch can be granted commit");
        let committing = &resolved.committing[0];
        let ledger = CommitLedger::new(&resolved.committing, stub.git(), limits)
            .expect("one committing worktree");
        Self {
            worktree: committing.worktree.clone(),
            common: committing.common_dir.clone(),
            entry: committing.git_dir.clone(),
            subagent: committing.repository.clone(),
            dir,
            resolved,
            stub,
            ledger,
        }
    }

    /// Commits as the subagent does, in its worktree.
    fn commit(&self, subject: &str) -> String {
        git_in(&self.worktree, &["commit", "--allow-empty", "-m", subject]);
        git_in(&self.worktree, &["rev-parse", "HEAD"])
    }

    /// Whether the main repository holds the object.
    fn main_has(&self, id: &str) -> bool {
        git_succeeds(&self.common, &["cat-file", "-e", id])
    }

    fn main(&self, args: &[&str]) -> String {
        git_in(&self.common, args)
    }

    fn pack_files(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.common.join("objects/pack"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// The directories in the main object store that are not loose-object
    /// fan-outs, `pack` or `info`: where a quarantine would be.
    fn stray_object_directories(&self) -> Vec<String> {
        std::fs::read_dir(self.common.join("objects"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "pack" && name != "info" && name.len() != 2)
            .collect()
    }

    fn branch(&self, namespace: &str) -> String {
        format!("refs/heads/subagent/{namespace}/work")
    }
}

/// Every file under `root` with its content, except below `except`.
fn contents(root: &Path, except: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(directory: &Path, root: &Path, except: &Path, into: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.starts_with(except) {
                continue;
            }
            if path.is_dir() {
                walk(&path, root, except, into);
            } else {
                into.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut into = BTreeMap::new();
    walk(root, root, except, &mut into);
    into
}

// ---- 1. the grant's layout ------------------------------------------------

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_commit_grant_writes_the_workstrees_own_repository_and_nothing_in_the_main_one() {
    let dir = tempfile::tempdir().unwrap();
    let worktree = repository_with_linked_worktree(dir.path(), "alpha");
    let common = dunce::canonicalize(dir.path().join("repository/.git")).unwrap();
    let entry = common.join("worktrees/alpha");
    let head_before = std::fs::read(entry.join("HEAD")).unwrap();
    let index_before = std::fs::read(entry.join("index")).unwrap();

    let resolved = main_launch(
        dir.path(),
        &SubagentGrant::new().with_committing_worktree(&worktree),
        &mut LaunchHistory::default(),
    )
    .await
    .expect("a worktree on a namespaced branch can be granted commit");

    let subagent = entry.join("subagent");
    assert_eq!(
        resolved.committing[0].writable(),
        [subagent.clone()],
        "the commit grant makes the worktree's repository the one writable path"
    );
    let granted: Vec<&PathBuf> = resolved
        .writable
        .iter()
        .filter(|path| path.starts_with(&common))
        .collect();
    assert_eq!(granted, [&subagent], "nothing else of the main repository");
    assert_eq!(
        std::fs::read_to_string(worktree.join(".git")).unwrap(),
        "gitdir: ../repository/.git/worktrees/alpha/subagent\n",
        "the worktree's .git file names the repository, relatively"
    );
    assert_eq!(std::fs::read(entry.join("HEAD")).unwrap(), head_before);
    assert_eq!(std::fs::read(entry.join("index")).unwrap(), index_before);
    let config = std::fs::read_to_string(subagent.join("config")).unwrap();
    assert!(config.contains("repositoryformatversion = 0"), "{config}");
    assert!(config.contains("path = \"../../../config\""), "{config}");
    assert!(config.contains("[remote \"main-repository\"]"), "{config}");
    assert_eq!(
        std::fs::read_to_string(subagent.join("objects/info/alternates")).unwrap(),
        "../../../../objects\n"
    );
    assert_eq!(
        dunce::canonicalize(git_in(&worktree, &["rev-parse", "--absolute-git-dir"])).unwrap(),
        subagent,
        "git in the worktree uses the subagent's repository"
    );
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_sha256_repository_works_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let repository = dir.path().join("repository");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(
        &repository,
        &["init", "--initial-branch=main", "--object-format=sha256"],
    );
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    let worktree = add_worktree(&repository, dir.path(), "sha");
    let fixture = Fixture::granted(dir, worktree, TestGit::new(), limits()).await;
    let config = std::fs::read_to_string(fixture.subagent.join("config")).unwrap();
    assert!(config.contains("objectformat"), "{config}");
    assert_eq!(
        git_in(&fixture.worktree, &["rev-parse", "--show-object-format"]),
        "sha256"
    );

    let mark = fixture.ledger.mark().await.unwrap();
    let commit = fixture.commit("on sha256");
    assert_eq!(commit.len(), 64);
    assert!(!fixture.main_has(&commit));
    let commits = fixture.ledger.since(&mark).await.expect("the import works");
    assert_eq!(commits.len(), 1, "{commits:?}");
    assert_eq!(commits[0].commit, commit);
    assert!(fixture.main_has(&commit));
}

// ---- 2. a commit lands only in the subagent's repository -------------------

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_commit_lands_only_in_the_subagents_repository_and_the_main_one_stays_sound() {
    let fixture = Fixture::new("beta").await;
    let before = contents(&fixture.common, &fixture.subagent);
    let commit = fixture.commit("private work");
    assert_eq!(
        contents(&fixture.common, &fixture.subagent),
        before,
        "the main repository, its refs and the worktree's entry are as they were"
    );
    assert!(!fixture.main_has(&commit));
    assert!(git_succeeds(
        &fixture.worktree,
        &["cat-file", "-e", commit.as_str()]
    ));

    // While the commit exists only in the subagent's repository, the main
    // agent's repository works as it always does.
    let other = fixture.dir.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    git_in(&other, &["init", "--initial-branch=main"]);
    git_in(&other, &["commit", "--allow-empty", "-m", "elsewhere"]);
    git_in(
        &fixture.common,
        &[
            "fetch",
            other.to_str().unwrap(),
            "main:refs/remotes/other/main",
        ],
    );
    fixture.main(&["repack", "-a", "-d"]);
    fixture.main(&["log", "--all", "--oneline"]);
    fixture.main(&["fsck", "--strict"]);
    fixture.main(&["gc"]);
}

/// The control for the test above: a ref of the main repository that names an
/// object only another repository has is what made the main agent's fetch,
/// repack, gc and fsck fail when a subagent's refs lived there.
// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_ref_naming_an_object_the_main_store_lacks_breaks_the_main_repository() {
    let fixture = Fixture::new("gamma").await;
    let commit = fixture.commit("private work");
    let reference = fixture.common.join(fixture.branch("gamma"));
    std::fs::write(&reference, format!("{commit}\n")).unwrap();
    for args in [&["repack", "-a", "-d"][..], &["fsck", "--strict"]] {
        assert!(
            !git_succeeds(&fixture.common, args),
            "git {args:?} must fail when a ref names an object only the subagent has"
        );
    }
}

// ---- 3. git elsewhere is untouched -----------------------------------------

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn git_elsewhere_is_untouched_by_the_redirected_worktree() {
    let fixture = Fixture::new("delta").await;
    fixture.commit("private work");
    let other = fixture.dir.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    git_in(&other, &["init", "--initial-branch=main"]);
    git_in(&other, &["commit", "--allow-empty", "-m", "elsewhere"]);
    assert!(git_in(&other, &["log", "--oneline"]).ends_with("elsewhere"));
    let clone = fixture.dir.path().join("clone");
    git_in(
        fixture.dir.path(),
        &["clone", other.to_str().unwrap(), clone.to_str().unwrap()],
    );
    assert!(git_in(&clone, &["log", "--oneline"]).ends_with("elsewhere"));

    // A submodule added in the worktree keeps its git directory in the
    // subagent's repository.
    git_in(
        &fixture.worktree,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            other.to_str().unwrap(),
            "sub",
        ],
    );
    let sub_git = std::fs::read_to_string(fixture.worktree.join("sub/.git")).unwrap();
    assert!(
        sub_git.contains("subagent/modules/sub"),
        "the submodule's git directory is in the subagent's repository: {sub_git}"
    );
    git_in(&fixture.worktree, &["commit", "-m", "add the submodule"]);
}

// ---- 4. since imports -------------------------------------------------------

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn since_imports_the_commits_and_a_second_since_runs_no_pack_export() {
    let fixture = Fixture::new("epsilon").await;
    let mark = fixture.ledger.mark().await.unwrap();
    let first = fixture.commit("first");
    let second = fixture.commit("second");
    assert!(!fixture.main_has(&first));

    let commits = fixture.ledger.since(&mark).await.expect("the import works");
    assert_eq!(
        commits
            .iter()
            .map(|commit| commit.subject.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert!(fixture.main_has(&first) && fixture.main_has(&second));
    assert_eq!(
        fixture.main(&["rev-parse", &fixture.branch("epsilon")]),
        second,
        "the namespace ref moved"
    );
    fixture.main(&["fsck", "--strict", "--no-dangling"]);
    assert_eq!(
        git_in(&fixture.subagent, &["count-objects", "-v"])
            .lines()
            .find(|line| line.starts_with("count:"))
            .unwrap(),
        "count: 0",
        "the subagent's repository keeps no loose copy of an imported object"
    );

    let packs = fixture.pack_files();
    let exports = fixture
        .stub
        .ran()
        .iter()
        .filter(|args| args[0] == "pack-objects")
        .count();
    let again = fixture
        .ledger
        .since(&mark)
        .await
        .expect("nothing to import");
    assert_eq!(again.len(), 2);
    assert_eq!(fixture.pack_files(), packs, "a second since adds no pack");
    assert_eq!(
        fixture
            .stub
            .ran()
            .iter()
            .filter(|args| args[0] == "pack-objects")
            .count(),
        exports,
        "a second since runs no pack export"
    );
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn merging_the_subagents_work_keeps_its_objects_after_its_repository_is_removed() {
    let fixture = Fixture::new("zeta").await;
    let mark = fixture.ledger.mark().await.unwrap();
    let commit = fixture.commit("to merge");
    fixture.ledger.since(&mark).await.unwrap();
    fixture.main(&["update-ref", "refs/heads/main", &commit]);
    std::fs::remove_dir_all(&fixture.subagent).unwrap();
    fixture.main(&["fsck", "--strict", "--no-dangling"]);
    fixture.main(&["gc", "--prune=now"]);
    assert!(fixture.main_has(&commit));
    assert!(!fixture.main(&["log", "--oneline", "main", "-1"]).is_empty());
}

// ---- 5-8. what the import refuses -------------------------------------------

/// The subagent plants `evil` under the id of `good`, then commits a tree
/// that names that id.
fn forge_object(fixture: &Fixture) {
    use std::io::Write as _;
    let good = fixture.dir.path().join("good.txt");
    std::fs::write(&good, "good\n").unwrap();
    let id = git_in(&fixture.worktree, &["hash-object", good.to_str().unwrap()]);
    let path = fixture
        .subagent
        .join("objects")
        .join(&id[..2])
        .join(&id[2..]);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"blob 5\0evil\n").unwrap();
    std::fs::write(&path, encoder.finish().unwrap()).unwrap();
    git_in(
        &fixture.worktree,
        &[
            "update-index",
            "--add",
            "--cacheinfo",
            &format!("100644,{id},good.txt"),
        ],
    );
    git_in(&fixture.worktree, &["commit", "-m", "forged"]);
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_forged_object_under_a_predicted_id_is_rejected_and_leaves_nothing_behind() {
    let fixture = Fixture::new("eta").await;
    let mark = fixture.ledger.mark().await.unwrap();
    forge_object(&fixture);
    let packs = fixture.pack_files();
    let refs = fixture.main(&["for-each-ref"]);

    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the forged object cannot be imported");
    assert!(
        matches!(error, CommitLedgerError::Rejected { .. }),
        "{error}"
    );
    assert_eq!(fixture.pack_files(), packs, "no pack and no tmp_pack_*");
    assert_eq!(fixture.main(&["for-each-ref"]), refs, "no ref moved");
    assert!(fixture.stray_object_directories().is_empty());
}

/// A pack of one blob, which holds none of the commits a report names.
fn unrelated_pack(directory: &Path) -> Vec<u8> {
    use std::io::Write as _;
    git_in(directory, &["init", "--quiet"]);
    let blob = directory.join("blob.txt");
    std::fs::write(&blob, "unrelated\n").unwrap();
    let id = git_in(directory, &["hash-object", "-w", blob.to_str().unwrap()]);
    let mut child = std::process::Command::new("git")
        .args(["pack-objects", "--stdout"])
        .current_dir(directory)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{id}").unwrap();
    child.wait_with_output().unwrap().stdout
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_pack_that_omits_a_reported_tip_is_incomplete_and_moves_no_ref() {
    let scratch = tempfile::tempdir().unwrap();
    let pack = unrelated_pack(scratch.path());
    let stub = TestGit::new().replying(
        "pack-objects",
        Reply::Print {
            code: 0,
            stdout: pack,
        },
    );
    let fixture = Fixture::with("theta", stub, limits()).await;
    let mark = fixture.ledger.mark().await.unwrap();
    let commit = fixture.commit("lost in transit");
    let refs = fixture.main(&["for-each-ref"]);

    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the tip never arrived");
    assert!(
        matches!(&error, CommitLedgerError::Incomplete { detail, .. } if detail.contains(&commit)),
        "{error}"
    );
    assert_eq!(fixture.main(&["for-each-ref"]), refs);
    assert!(!fixture.main_has(&commit));
    assert!(fixture.stray_object_directories().is_empty());
}

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_report_naming_a_ref_outside_the_namespace_is_refused_before_anything_runs() {
    let id = "a".repeat(40);
    let lies: [(&str, Reply); 3] = [
        (
            "for-each-ref",
            Reply::Print {
                code: 0,
                stdout: format!("{id} refs/heads/main\n").into_bytes(),
            },
        ),
        (
            "for-each-ref",
            Reply::Print {
                code: 0,
                stdout: b"not-an-id refs/heads/subagent/iota/work\n".to_vec(),
            },
        ),
        (
            "symbolic-ref",
            Reply::Print {
                code: 0,
                stdout: b"refs/heads/main\n".to_vec(),
            },
        ),
    ];
    for (command, reply) in lies {
        let stub = TestGit::new().replying(command, reply);
        let fixture = Fixture::with("iota", stub, limits()).await;
        let refs = fixture.main(&["for-each-ref"]);
        // The run start imports what the repository holds, so it meets the
        // lie; it does not keep the run from starting, but says so.
        let mark = fixture
            .ledger
            .mark()
            .await
            .expect("a lying repository does not keep a run from starting");
        assert!(
            matches!(mark.setbacks(), [CommitLedgerError::Report { .. }]),
            "{:?}",
            mark.setbacks()
        );
        let error = fixture
            .ledger
            .since(&mark)
            .await
            .expect_err("the report is still a lie");
        assert!(matches!(error, CommitLedgerError::Report { .. }), "{error}");
        assert_eq!(fixture.main(&["for-each-ref"]), refs);
        assert!(
            !fixture.stub.ran_command("pack-objects"),
            "nothing runs after a bad report"
        );
    }
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_pack_over_the_byte_budget_is_rejected_and_nothing_is_left_behind() {
    let tight = ImportLimits {
        bytes: ImportBudget::new(2048),
        time: Duration::from_secs(60),
    };
    let fixture = Fixture::with("kappa", TestGit::new(), tight).await;
    let mark = fixture.ledger.mark().await.unwrap();
    // Incompressible content, so the pack is as large as the file.
    let noise: Vec<u8> = (0_u32..100_000)
        .map(|n| (n.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    std::fs::write(fixture.worktree.join("noise.bin"), noise).unwrap();
    git_in(&fixture.worktree, &["add", "noise.bin"]);
    git_in(&fixture.worktree, &["commit", "-m", "heavy"]);
    let packs = fixture.pack_files();

    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the pack is over the budget");
    assert!(
        matches!(&error, CommitLedgerError::Rejected { stderr, .. } if stderr.contains("exceeds")),
        "{error}"
    );
    assert_eq!(fixture.pack_files(), packs);
    assert!(fixture.stray_object_directories().is_empty());
}

// ---- 9. a detached HEAD ------------------------------------------------------

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn detached_work_is_listed_mirrored_into_the_entry_and_kept_by_gc() {
    let fixture = Fixture::new("lambda").await;
    let mark = fixture.ledger.mark().await.unwrap();
    git_in(&fixture.worktree, &["switch", "--detach"]);
    let detached = fixture.commit("detached work");
    let commits = fixture.ledger.since(&mark).await.expect("the import works");
    assert_eq!(commits.len(), 1, "{commits:?}");
    assert_eq!(commits[0].commit, detached);
    assert_eq!(
        std::fs::read_to_string(fixture.entry.join("HEAD"))
            .unwrap()
            .trim(),
        detached,
        "the entry's HEAD holds the detached commit"
    );
    fixture.main(&["gc", "--prune=now"]);
    assert!(fixture.main_has(&detached), "gc keeps the detached commit");

    git_in(&fixture.worktree, &["switch", "subagent/lambda/work"]);
    fixture.ledger.since(&mark).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(fixture.entry.join("HEAD"))
            .unwrap()
            .trim(),
        "ref: refs/heads/subagent/lambda/work",
        "re-attaching restores the symbolic HEAD"
    );
}

// ---- 10. forged graph, nested alternate -------------------------------------

/// A commit-graph file that lists `commit` with `tree` and no parent.
fn graph_hiding_the_parent(commit: &str, tree: &str) -> Vec<u8> {
    let commit = hex::decode(commit).unwrap();
    let tree = hex::decode(tree).unwrap();
    let mut fanout = [0_u32; 256];
    for slot in fanout.iter_mut().skip(usize::from(commit[0])) {
        *slot = 1;
    }
    let mut data = Vec::new();
    data.extend_from_slice(b"CGPH");
    data.extend_from_slice(&[1, 1, 3, 0]);
    let header = 8 + 4 * 12;
    let oidf = header;
    let oidl = oidf + 256 * 4;
    let cdat = oidl + 20;
    let end = cdat + 36;
    for (id, offset) in [
        (b"OIDF", oidf),
        (b"OIDL", oidl),
        (b"CDAT", cdat),
        (b"\0\0\0\0", end),
    ] {
        data.extend_from_slice(id);
        data.extend_from_slice(&u64::try_from(offset).unwrap().to_be_bytes());
    }
    for slot in fanout {
        data.extend_from_slice(&slot.to_be_bytes());
    }
    data.extend_from_slice(&commit);
    data.extend_from_slice(&tree);
    data.extend_from_slice(&0x7000_0000_u32.to_be_bytes());
    data.extend_from_slice(&0x7000_0000_u32.to_be_bytes());
    // Generation 1 in the high bits, then the commit time.
    data.extend_from_slice(&(1_u32 << 2).to_be_bytes());
    data.extend_from_slice(&1_700_000_000_u32.to_be_bytes());
    // Git does not check the trailing checksum when it loads the file.
    data.extend_from_slice(&[0; 20]);
    data
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_forged_commit_graph_and_a_nested_alternate_change_nothing_the_main_repository_reads() {
    let fixture = Fixture::new("mu").await;
    let mark = fixture.ledger.mark().await.unwrap();
    let parent = fixture.commit("parent only the subagent has");
    let child = fixture.commit("child");
    let tree = git_in(&fixture.worktree, &["rev-parse", "HEAD^{tree}"]);
    let before = (
        fixture.main(&["log", "--all", "--oneline"]),
        fixture.main(&["rev-list", "--all"]),
    );

    // A graph that says the child has no parent, and an alternate naming a
    // directory of the subagent's choosing.
    let info = fixture.subagent.join("objects/info");
    std::fs::write(
        info.join("commit-graph"),
        graph_hiding_the_parent(&child, &tree),
    )
    .unwrap();
    let extra = fixture.dir.path().join("extra-objects");
    std::fs::create_dir_all(&extra).unwrap();
    let alternates = info.join("alternates");
    let mut content = std::fs::read_to_string(&alternates).unwrap();
    content.push_str(&format!("{}\n", extra.display()));
    std::fs::write(&alternates, content).unwrap();
    assert_eq!(
        (
            fixture.main(&["log", "--all", "--oneline"]),
            fixture.main(&["rev-list", "--all"]),
        ),
        before,
        "the main repository reads nothing of the subagent's"
    );

    let refs = fixture.main(&["for-each-ref"]);
    match fixture.ledger.since(&mark).await {
        Ok(_) => {
            assert!(fixture.main_has(&parent) && fixture.main_has(&child));
            fixture.main(&["fsck", "--strict", "--no-dangling"]);
        }
        Err(CommitLedgerError::Rejected { .. } | CommitLedgerError::Incomplete { .. }) => {
            assert_eq!(fixture.main(&["for-each-ref"]), refs, "no partial ref move");
            assert!(!fixture.main_has(&child));
        }
        Err(other) => panic!("a forged graph must end in Rejected or Incomplete: {other}"),
    }
}

// ---- 11. compare and swap ---------------------------------------------------

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_ref_that_moved_in_the_main_repository_during_the_import_is_reported_and_left_alone() {
    let common = Arc::new(Mutex::new(None::<(PathBuf, String)>));
    let hook_target = common.clone();
    let stub = TestGit::new().before("pack-objects", move || {
        let guard = hook_target.lock().unwrap();
        let (common, reference) = guard.as_ref().expect("set up");
        let moved = git_in(common, &["commit-tree", "-m", "moved", "HEAD^{tree}"]);
        git_in(common, &["update-ref", reference, &moved]);
    });
    let fixture = Fixture::with("nu", stub, limits()).await;
    *common.lock().unwrap() = Some((fixture.common.clone(), fixture.branch("nu")));
    let mark = fixture.ledger.mark().await.unwrap();
    fixture.commit("raced");

    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the ref moved under the import");
    assert!(matches!(error, CommitLedgerError::Moved { .. }), "{error}");
    fixture.main(&["fsck", "--strict", "--no-dangling"]);
    assert!(fixture.stray_object_directories().is_empty());
}

// ---- 12. cancellation and several repositories -----------------------------

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn dropping_since_mid_index_leaves_no_quarantine() {
    let stub = TestGit::new().replying("pack-objects", Reply::Hang);
    let fixture = Fixture::with("xi", stub, limits()).await;
    let mark = fixture.ledger.mark().await.unwrap();
    fixture.commit("never imported");
    let outcome =
        futures_lite::future::or(async { Some(fixture.ledger.since(&mark).await) }, async {
            async_io::Timer::after(Duration::from_millis(1500)).await;
            None
        })
        .await;
    assert!(outcome.is_none(), "the export hangs");
    assert!(
        fixture.stub.ran_command("pack-objects"),
        "the import reached the export"
    );
    assert!(fixture.stray_object_directories().is_empty());
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn one_rejected_repository_does_not_stop_the_others_and_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let repository = dir.path().join("repository");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(&repository, &["init", "--initial-branch=main"]);
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    let first = add_worktree(&repository, dir.path(), "good");
    let second = add_worktree(&repository, dir.path(), "bad");
    let resolved = main_launch(
        dir.path(),
        &SubagentGrant::new()
            .with_committing_worktree(&first)
            .with_committing_worktree(&second),
        &mut LaunchHistory::default(),
    )
    .await
    .unwrap();
    let stub = TestGit::new();
    let ledger = CommitLedger::new(&resolved.committing, stub.git(), limits()).unwrap();
    let mark = ledger.mark().await.unwrap();

    git_in(&first, &["commit", "--allow-empty", "-m", "fine"]);
    let fine = git_in(&first, &["rev-parse", "HEAD"]);
    // The second worktree's commit names a blob that is not in its pack.
    let bad_repository = resolved
        .committing
        .iter()
        .find(|worktree| worktree.namespace == "bad")
        .unwrap()
        .repository
        .clone();
    let missing_blob = "b".repeat(40);
    let tree = {
        use std::io::Write as _;
        let mut child = std::process::Command::new("git")
            .args(["mktree", "--missing"])
            .current_dir(&second)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        write!(
            child.stdin.take().unwrap(),
            "100644 blob {missing_blob}\tmissing.txt\n"
        )
        .unwrap();
        String::from_utf8(child.wait_with_output().unwrap().stdout)
            .unwrap()
            .trim()
            .to_string()
    };
    let broken = git_in(&second, &["commit-tree", "-m", "broken", &tree]);
    git_in(
        &second,
        &["update-ref", "refs/heads/subagent/bad/work", &broken],
    );

    let error = ledger
        .since(&mark)
        .await
        .expect_err("one repository is broken");
    assert!(
        error.to_string().contains(bad_repository.to_str().unwrap()),
        "the error names the failing repository: {error}"
    );
    assert!(
        git_succeeds(
            &dunce::canonicalize(repository.join(".git")).unwrap(),
            &["cat-file", "-e", &fine]
        ),
        "the other repository's import completed"
    );
}

// ---- 13. relocation ---------------------------------------------------------

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn moving_the_whole_tree_keeps_the_worktree_working() {
    let fixture = Fixture::new("omicron").await;
    fixture.commit("before the move");
    let moved = fixture.dir.path().join("moved");
    std::fs::create_dir_all(&moved).unwrap();
    for name in ["repository", "omicron"] {
        std::fs::rename(fixture.dir.path().join(name), moved.join(name)).unwrap();
    }
    let worktree = moved.join("omicron");
    assert!(git_in(&worktree, &["log", "--oneline"]).contains("before the move"));
    git_in(
        &worktree,
        &["commit", "--allow-empty", "-m", "after the move"],
    );
    git_in(&worktree, &["status", "--short"]);
}

// ---- 14. re-grants ----------------------------------------------------------

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_converted_worktree_is_granted_again_with_a_fresh_history_and_reuses_its_repository() {
    let fixture = Fixture::new("pi").await;
    let commit = fixture.commit("kept across grants");
    let again = main_launch(
        fixture.dir.path(),
        &SubagentGrant::new().with_committing_worktree(&fixture.worktree),
        &mut LaunchHistory::default(),
    )
    .await
    .expect("a converted worktree can be granted again");
    assert_eq!(again.committing, fixture.resolved.committing);
    assert_eq!(
        git_in(&fixture.worktree, &["rev-parse", "HEAD"]),
        commit,
        "the repository was reused as it stood"
    );
}

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_converted_worktree_that_names_another_worktrees_repository_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let first = repository_with_linked_worktree(dir.path(), "rho");
    let repository = dir.path().join("repository");
    let second = add_worktree(&repository, dir.path(), "sigma");
    main_launch(
        dir.path(),
        &SubagentGrant::new()
            .with_committing_worktree(&first)
            .with_committing_worktree(&second),
        &mut LaunchHistory::default(),
    )
    .await
    .unwrap();
    // The second worktree's .git file now names the first one's repository.
    let first_git = std::fs::read_to_string(first.join(".git")).unwrap();
    std::fs::write(second.join(".git"), first_git).unwrap();
    let error = main_launch(
        dir.path(),
        &SubagentGrant::new().with_committing_worktree(&second),
        &mut LaunchHistory::default(),
    )
    .await
    .expect_err("the entry names another worktree");
    assert!(
        matches!(error, SubagentGrantError::MalformedGitData { .. }),
        "{error}"
    );
}

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_commit_grant_under_a_plain_granted_ancestor_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let repository = dir.path().join("repository");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(&repository, &["init", "--initial-branch=main"]);
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    let shared = dir.path().join("shared");
    std::fs::create_dir_all(&shared).unwrap();
    let worktree = add_worktree(&repository, &shared, "tau");

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
    .expect_err("a subagent could swap the worktree for a symlink");
    assert!(
        matches!(error, SubagentGrantError::WorktreeExposed { .. }),
        "{error}"
    );
    // Neither order lets a plain grant of the ancestor in afterwards.
    let committed = tempfile::tempdir().unwrap();
    let inner = repository_with_linked_worktree(committed.path(), "upsilon");
    let mut history = LaunchHistory::default();
    main_launch(
        committed.path(),
        &SubagentGrant::new().with_committing_worktree(&inner),
        &mut history,
    )
    .await
    .unwrap();
    let error = main_launch(
        committed.path(),
        &SubagentGrant::new().with_worktree(committed.path()),
        &mut history,
    )
    .await
    .expect_err("the committed worktree lies inside");
    assert!(
        matches!(
            error,
            SubagentGrantError::WorktreeExposed { .. } | SubagentGrantError::GitDataExposed { .. }
        ),
        "{error}"
    );
}

// ---- 15. the exporter's time bound ------------------------------------------

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn an_exporter_that_never_finishes_is_killed_at_the_time_bound() {
    let stub = TestGit::new().replying("pack-objects", Reply::Hang);
    let quick = ImportLimits {
        bytes: ImportBudget::new(256 * ONE_MIB),
        time: Duration::from_millis(800),
    };
    let fixture = Fixture::with("phi", stub, quick).await;
    let mark = fixture.ledger.mark().await.unwrap();
    fixture.commit("stuck");
    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the export never ends");
    assert!(
        matches!(
            &error,
            CommitLedgerError::Export {
                source: ExportError::TimedOut(_),
                ..
            }
        ),
        "{error}"
    );
    let pids = fixture.stub.hung.lock().unwrap().clone();
    assert!(!pids.is_empty());
    for pid in pids {
        let mut gone = false;
        for _ in 0..40 {
            // A killed process nobody has reaped yet is a zombie: gone.
            let state = std::process::Command::new("ps")
                .args(["-o", "stat=", "-p", &pid.to_string()])
                .output()
                .unwrap();
            let state = String::from_utf8_lossy(&state.stdout);
            if state.trim().is_empty() || state.trim().starts_with('Z') {
                gone = true;
                break;
            }
            async_io::Timer::after(Duration::from_millis(100)).await;
        }
        assert!(gone, "the exporter's process {pid} is gone");
    }
    assert!(fixture.stray_object_directories().is_empty());
}

// ---- the #227 guarantees that still hold -------------------------------------

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

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_namespace_holding_another_branch_is_not_the_worktrees_own() {
    let dir = tempfile::tempdir().unwrap();
    let worktree = repository_with_linked_worktree(dir.path(), "beta2");
    git_in(
        &dir.path().join("repository"),
        &["branch", "subagent/beta2/other", "main"],
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

/// A subagent rewrites its repository's configuration to run a program for
/// every signed commit `git log` shows, plants such a commit, and points its
/// worktree's `.git` file at a decoy. The host reads the main repository only,
/// with its own configuration, and runs nothing.
#[cfg(unix)]
#[tokio::test]
async fn the_ledger_reads_only_the_main_repository_and_runs_no_program_the_subagent_chose() {
    use std::os::unix::fs::PermissionsExt as _;

    let fixture = Fixture::new("gamma2").await;
    let mark = fixture.ledger.mark().await.unwrap();
    git_in(
        &fixture.worktree,
        &["commit", "--allow-empty", "-m", "real work"],
    );

    let marker = fixture.dir.path().join("ran");
    let program = fixture.dir.path().join("gpg-program");
    std::fs::write(
        &program,
        format!("#!/bin/sh\ntouch '{}'\nexit 1\n", marker.display()),
    )
    .unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    // A signed commit in the subagent's repository, on its branch.
    let tree = git_in(&fixture.worktree, &["rev-parse", "HEAD^{tree}"]);
    let parent = git_in(&fixture.worktree, &["rev-parse", "HEAD"]);
    let signed = format!(
        "tree {tree}\nparent {parent}\nauthor a <a@a> 0 +0000\ncommitter a <a@a> 0 +0000\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n -----END PGP SIGNATURE-----\n\nsigned decoy\n"
    );
    let object = fixture.dir.path().join("signed-commit");
    std::fs::write(&object, signed).unwrap();
    let signed = git_in(
        fixture.dir.path(),
        &[
            "--git-dir",
            fixture.subagent.to_str().unwrap(),
            "hash-object",
            "-t",
            "commit",
            "-w",
            object.to_str().unwrap(),
        ],
    );
    git_in(
        fixture.dir.path(),
        &[
            "--git-dir",
            fixture.subagent.to_str().unwrap(),
            "-c",
            "core.fsmonitor=false",
            "update-ref",
            &fixture.branch("gamma2"),
            &signed,
        ],
    );
    let mut config = std::fs::read_to_string(fixture.subagent.join("config")).unwrap();
    config.push_str(&format!(
        "[log]\n\tshowSignature = true\n[gpg]\n\tprogram = {}\n[core]\n\tfsmonitor = {}\n",
        program.display(),
        program.display()
    ));
    std::fs::write(fixture.subagent.join("config"), config).unwrap();

    let decoy = fixture.dir.path().join("decoy");
    std::fs::create_dir_all(&decoy).unwrap();
    git_in(&decoy, &["init", "--initial-branch=main"]);
    std::fs::write(
        fixture.worktree.join(".git"),
        format!("gitdir: {}\n", decoy.join(".git").display()),
    )
    .unwrap();

    let commits = fixture
        .ledger
        .since(&mark)
        .await
        .expect("the real repository is read");
    let subjects: Vec<&str> = commits.iter().map(|c| c.subject.as_str()).collect();
    assert_eq!(subjects, ["real work", "signed decoy"]);
    assert!(
        !marker.exists(),
        "the ledger ran a program the subagent chose"
    );
}

/// Commits made on another branch of the namespace, or left behind by
/// resetting the worktree's branch and detaching `HEAD`, are still listed;
/// a branch moved back to where it was lists nothing.
// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn the_ledger_lists_commits_whatever_head_does() {
    let fixture = Fixture::new("delta2").await;
    let worktree = &fixture.worktree;
    let mark = fixture.ledger.mark().await.unwrap();

    git_in(worktree, &["switch", "-c", "subagent/delta2/side"]);
    git_in(
        worktree,
        &["commit", "--allow-empty", "-m", "on a side branch"],
    );
    git_in(worktree, &["switch", "subagent/delta2/work"]);
    git_in(
        worktree,
        &["commit", "--allow-empty", "-m", "detached later"],
    );
    git_in(worktree, &["switch", "--detach"]);
    git_in(worktree, &["branch", "-f", "subagent/delta2/work", "main"]);

    let mut subjects: Vec<String> = fixture
        .ledger
        .since(&mark)
        .await
        .unwrap()
        .into_iter()
        .map(|commit| commit.subject)
        .collect();
    subjects.sort();
    assert_eq!(subjects, ["detached later", "on a side branch"]);
}

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn a_plain_grant_cannot_reach_git_data_commits_are_listed_from() {
    let dir = tempfile::tempdir().unwrap();
    let worktree = repository_with_linked_worktree(dir.path(), "epsilon2");
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

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn git_data_a_subagent_could_write_is_not_trusted_for_a_commit_grant() {
    let dir = tempfile::tempdir().unwrap();
    // The repository itself lies inside a directory handed out earlier.
    let shared = dir.path().join("shared");
    std::fs::create_dir_all(&shared).unwrap();
    let worktree = repository_with_linked_worktree(&shared, "zeta2");
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

// ---- the review of the third implementation ----------------------------------

/// `future`'s output, or a panic if it has not finished by `seconds`: a test
/// of something that must not hang fails instead of hanging.
async fn finishes_within<T>(seconds: u64, future: impl Future<Output = T>) -> T {
    within(Duration::from_secs(seconds), future)
        .await
        .unwrap_or_else(|| panic!("did not finish within {seconds} seconds"))
}

fn quick(millis: u64) -> ImportLimits {
    ImportLimits {
        bytes: ImportBudget::new(256 * ONE_MIB),
        time: Duration::from_millis(millis),
    }
}

#[cfg(unix)]
fn make_fifo(path: &Path) {
    let status = std::process::Command::new("mkfifo")
        .arg(path)
        .status()
        .expect("mkfifo runs");
    assert!(status.success());
}

/// Whether the process is gone, a killed one nobody has reaped yet counting
/// as gone.
async fn process_is_gone(pid: u32) -> bool {
    for _ in 0..50 {
        let state = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        let state = String::from_utf8_lossy(&state.stdout);
        if state.trim().is_empty() || state.trim().starts_with('Z') {
            return true;
        }
        async_io::Timer::after(Duration::from_millis(100)).await;
    }
    false
}

// A commit grant is refused where no shell can carry its shim.
#[cfg(unix)]
#[tokio::test]
async fn every_question_to_the_subagents_repository_is_bounded_in_time_and_its_process_killed() {
    // Each sandboxed command of an import, in the order the import meets them.
    let hangs = [
        ("for-each-ref", None),
        ("symbolic-ref", None),
        (
            "rev-parse",
            Some((
                "symbolic-ref",
                Reply::Print {
                    code: 1,
                    stdout: Vec::new(),
                },
            )),
        ),
    ];
    for (command, before) in hangs {
        let mut stub = TestGit::new();
        if let Some((earlier, reply)) = before {
            stub = stub.replying(earlier, reply);
        }
        let stub = stub.replying(command, Reply::Hang);
        let fixture = Fixture::with("chi", stub, quick(700)).await;
        let mark = finishes_within(20, fixture.ledger.mark())
            .await
            .expect("a stalled repository does not keep a run from starting");
        assert!(
            matches!(
                mark.setbacks(),
                [CommitLedgerError::Export {
                    source: ExportError::TimedOut(_),
                    ..
                }]
            ),
            "{command}: {:?}",
            mark.setbacks()
        );
        let error = finishes_within(20, fixture.ledger.since(&mark))
            .await
            .expect_err("the question never gets an answer");
        assert!(
            matches!(
                &error,
                CommitLedgerError::Export {
                    source: ExportError::TimedOut(_),
                    ..
                }
            ),
            "{command}: {error}"
        );
        assert!(
            error
                .to_string()
                .contains(fixture.subagent.to_str().unwrap()),
            "the error names the repository whose commits stay: {error}"
        );
        let pids = fixture.stub.hung.lock().unwrap().clone();
        assert!(!pids.is_empty(), "{command} was asked");
        for pid in pids {
            assert!(process_is_gone(pid).await, "{command}: process {pid} lives");
        }
    }
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn the_tidying_after_an_import_is_bounded_in_time() {
    let stub = TestGit::new().replying("prune-packed", Reply::Hang);
    let fixture = Fixture::with("psi", stub, quick(700)).await;
    let mark = fixture.ledger.mark().await.unwrap();
    let commit = fixture.commit("imported before the tidying hangs");
    let listed = finishes_within(20, fixture.ledger.since(&mark))
        .await
        .expect("the commits arrived; only the tidying stalled");
    assert_eq!(listed.len(), 1);
    assert!(fixture.main_has(&commit));
    let pids = fixture.stub.hung.lock().unwrap().clone();
    for pid in pids {
        assert!(process_is_gone(pid).await, "process {pid} lives");
    }
}

#[tokio::test]
#[cfg(unix)]
async fn a_fifo_planted_in_the_subagents_repository_cannot_hang_the_main_agent() {
    // One planted as a ref file, one behind a config include: every git in
    // the repository reads both.
    for plant in ["packed-refs", "include"] {
        let fixture = Fixture::with("omega", TestGit::new(), quick(700)).await;
        let mark = finishes_within(20, fixture.ledger.mark()).await.unwrap();
        fixture.commit("while the fifo waits");
        if plant == "packed-refs" {
            make_fifo(&fixture.subagent.join("packed-refs"));
        } else {
            make_fifo(&fixture.subagent.join("hang"));
            let config = fixture.subagent.join("config");
            let mut text = std::fs::read_to_string(&config).unwrap();
            text.push_str("[include]\n\tpath = hang\n");
            std::fs::write(&config, text).unwrap();
        }
        let error = finishes_within(20, fixture.ledger.since(&mark))
            .await
            .expect_err("the repository cannot be read");
        assert!(
            matches!(
                &error,
                CommitLedgerError::Export {
                    source: ExportError::TimedOut(_),
                    ..
                }
            ),
            "{plant}: {error}"
        );
        let next = finishes_within(20, fixture.ledger.mark())
            .await
            .expect("the next run starts all the same");
        assert_eq!(next.setbacks().len(), 1, "{plant}");
    }
}

/// A repository with its worktree four directories below the common ancestor,
/// where the relative hooks path of the subagent's repository resolves, from
/// the directory the host runs git in, into a directory a subagent may be
/// granted.
fn deep_layout(root: &Path) -> (PathBuf, PathBuf) {
    let repository = root.join("r");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(&repository, &["init", "--initial-branch=main"]);
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    let parent = root.join("a/b/c");
    std::fs::create_dir_all(&parent).unwrap();
    let worktree = add_worktree(&repository, &parent, "wt");
    (repository, worktree)
}

#[cfg(unix)]
fn planted_hook(directory: &Path, marker: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::create_dir_all(directory).unwrap();
    let hook = directory.join("reference-transaction");
    std::fs::write(&hook, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[tokio::test]
#[cfg(unix)]
async fn host_git_runs_no_hook_a_subagent_could_have_placed() {
    let dir = tempfile::tempdir().unwrap();
    let (repository, worktree) = deep_layout(dir.path());
    let marker = dir.path().join("hook-ran");
    // Where `../../../../r/.git/hooks`, spelled for the worktree, lands when
    // it is resolved against the repository's creation directory instead.
    planted_hook(&repository.join("r/.git/hooks"), &marker);
    // The main repository's own hooks are off for the import as well.
    let own_marker = dir.path().join("own-hook-ran");
    planted_hook(&repository.join(".git/hooks"), &own_marker);

    let resolved = main_launch(
        dir.path(),
        &SubagentGrant::new().with_committing_worktree(&worktree),
        &mut LaunchHistory::default(),
    )
    .await
    .expect("the grant is given");
    assert!(
        !marker.exists(),
        "creating the repository ran a hook from a path resolved against the wrong directory"
    );

    let ledger = CommitLedger::new(&resolved.committing, TestGit::new().git(), limits()).unwrap();
    let mark = ledger.mark().await.unwrap();
    git_in(&worktree, &["commit", "--allow-empty", "-m", "imported"]);
    // The subagent's own commit ran the main repository's hooks, as its
    // configuration says; that is not the import.
    assert!(own_marker.exists());
    std::fs::remove_file(&own_marker).unwrap();
    let listed = ledger.since(&mark).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert!(!marker.exists(), "the import ran a placed hook");
    assert!(!own_marker.exists(), "the import ran the repository's hook");
}

/// Plants a commit the import refuses, as the subagent.
async fn poisoned() -> (Fixture, CommitMark) {
    let fixture = Fixture::new("poisoned").await;
    let mark = fixture.ledger.mark().await.unwrap();
    forge_object(&fixture);
    (fixture, mark)
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_repository_whose_commits_cannot_be_imported_does_not_lock_the_worktree() {
    let (fixture, mark) = poisoned().await;
    let refs = fixture.main(&["for-each-ref"]);
    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the forged commit is refused");
    let message = error.to_string();
    assert!(
        message.contains(fixture.subagent.to_str().unwrap())
            && message.contains("deleting that directory discards them"),
        "the error says where the commits stay and how to discard them: {message}"
    );

    // The next subagent on the worktree starts all the same, from what the
    // main repository holds, and the main agent is told what stays behind.
    let next = fixture
        .ledger
        .mark()
        .await
        .expect("an earlier failed import does not keep the worktree from running");
    assert!(
        matches!(next.setbacks(), [CommitLedgerError::Rejected { .. }]),
        "{:?}",
        next.setbacks()
    );
    assert_eq!(
        fixture.main(&["for-each-ref"]),
        refs,
        "nothing was dropped or moved"
    );
    assert!(
        fixture.subagent.exists(),
        "the unimported commits are still in place"
    );

    // Deleting the repository and granting the worktree again gives it a
    // fresh one, and its commits import.
    std::fs::remove_dir_all(&fixture.subagent).unwrap();
    let resolved = main_launch(
        fixture.dir.path(),
        &SubagentGrant::new().with_committing_worktree(&fixture.worktree),
        &mut LaunchHistory::default(),
    )
    .await
    .expect("a worktree whose repository was deleted is granted again");
    let ledger = CommitLedger::new(&resolved.committing, fixture.stub.git(), limits()).unwrap();
    let fresh = ledger.mark().await.unwrap();
    assert!(fresh.setbacks().is_empty(), "{:?}", fresh.setbacks());
    let commit = fixture.commit("after the reset");
    let listed = ledger.since(&fresh).await.unwrap();
    assert_eq!(
        listed.iter().map(|c| c.commit.as_str()).collect::<Vec<_>>(),
        [commit.as_str()]
    );
    assert!(fixture.main_has(&commit));
}

/// A symbolic HEAD move cannot ride the refs transaction — symref-update
/// cannot rewrite a worktree entry's HEAD — so it follows the committed
/// refs through symbolic-ref, which takes HEAD's lock like any ref write:
/// the refs are already committed when the lock refuses the mirror, and
/// the next import heals it.
// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_locked_head_stops_a_symbolic_mirror_but_not_the_refs() {
    let fixture = Fixture::new("locked").await;
    let mark = fixture.ledger.mark().await.unwrap();
    // A branch inside the granted namespace moves the report's HEAD while
    // leaving the entry's referent untouched.
    git_in(
        &fixture.worktree,
        &["checkout", "-b", "subagent/locked/other"],
    );
    let commit = fixture.commit("moved");
    let lock = fixture.entry.join("HEAD.lock");
    std::fs::write(&lock, "").unwrap();
    fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the locked HEAD refuses the mirror");
    assert_eq!(
        fixture.main(&["rev-parse", "refs/heads/subagent/locked/other"]),
        commit,
        "the namespace refs committed, the HEAD mirror alone is behind"
    );
    assert_eq!(
        std::fs::read_to_string(fixture.entry.join("HEAD"))
            .unwrap()
            .trim(),
        "ref: refs/heads/subagent/locked/work",
        "the entry's HEAD stayed where it was"
    );

    std::fs::remove_file(&lock).unwrap();
    let listed = fixture.ledger.since(&mark).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].commit, commit);
    assert_eq!(
        std::fs::read_to_string(fixture.entry.join("HEAD"))
            .unwrap()
            .trim(),
        "ref: refs/heads/subagent/locked/other"
    );
}

/// When an import also moves the ref the entry's HEAD points at, HEAD
/// moves in a follow-up transaction — updating its referent already
/// counts as updating it. A lock on it still freezes the whole import:
/// the referent update itself needs HEAD's lock. Removing it recovers.
// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_locked_head_blocks_an_import_moving_its_referent_too() {
    let fixture = Fixture::new("referent").await;
    let mark = fixture.ledger.mark().await.unwrap();
    let commit = fixture.commit("moved");
    // Detaching changes the entry's HEAD while the import also updates
    // its referent: HEAD cannot ride this transaction, but its lock still
    // covers the referent update.
    git_in(&fixture.worktree, &["checkout", "--detach"]);
    let refs = fixture.main(&["for-each-ref"]);
    let lock = fixture.entry.join("HEAD.lock");
    std::fs::write(&lock, "").unwrap();
    fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("the locked HEAD refuses the import");
    assert_eq!(
        fixture.main(&["for-each-ref"]),
        refs,
        "the namespace refs did not move either"
    );

    std::fs::remove_file(&lock).unwrap();
    let listed = fixture.ledger.since(&mark).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].commit, commit);
    assert_eq!(
        std::fs::read_to_string(fixture.entry.join("HEAD"))
            .unwrap()
            .trim(),
        commit,
        "the entry's HEAD was detached at the commit once the lock went"
    );
}

#[tokio::test]
#[cfg(unix)]
async fn one_worktree_that_cannot_be_prepared_does_not_hold_back_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let repository = dir.path().join("repository");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(&repository, &["init", "--initial-branch=main"]);
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    let good = add_worktree(&repository, dir.path(), "kept");
    let bad = add_worktree(&repository, dir.path(), "stuck");
    let resolved = main_launch(
        dir.path(),
        &SubagentGrant::new()
            .with_committing_worktree(&good)
            .with_committing_worktree(&bad),
        &mut LaunchHistory::default(),
    )
    .await
    .unwrap();
    git_in(
        &good,
        &["commit", "--allow-empty", "-m", "left by an earlier holder"],
    );
    let left = git_in(&good, &["rev-parse", "HEAD"]);
    let stuck = resolved
        .committing
        .iter()
        .find(|worktree| worktree.namespace == "stuck")
        .unwrap();
    make_fifo(&stuck.repository.join("packed-refs"));

    let ledger = CommitLedger::new(&resolved.committing, TestGit::new().git(), quick(700)).unwrap();
    let mark = finishes_within(20, ledger.mark()).await.unwrap();
    assert_eq!(mark.setbacks().len(), 1);
    assert!(
        mark.setbacks()[0]
            .to_string()
            .contains(stuck.repository.to_str().unwrap())
    );
    assert!(
        git_succeeds(&stuck.common_dir, &["cat-file", "-e", &left]),
        "the other worktree was carried through its import"
    );
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn an_import_never_removes_a_directory_it_did_not_create() {
    let fixture = Fixture::new("sweep").await;
    let objects = fixture.common.join("objects");
    // Git's own receive-pack quarantine, and another import's, long idle.
    let mut left = Vec::new();
    for name in ["tmp_objdir-incoming-abc123", "aither-import-someone-else"] {
        let directory = objects.join(name);
        std::fs::create_dir_all(directory.join("pack")).unwrap();
        std::fs::File::open(&directory)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - Duration::from_secs(3600))
            .unwrap();
        left.push(directory);
    }
    let mark = fixture.ledger.mark().await.unwrap();
    fixture.commit("imported past them");
    fixture.ledger.since(&mark).await.unwrap();
    for directory in left {
        assert!(directory.exists(), "{} was removed", directory.display());
    }
}

#[tokio::test]
#[cfg(unix)]
async fn killing_the_update_of_the_refs_would_leave_locks_but_letting_it_finish_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let repository = dir.path().join("repository");
    std::fs::create_dir_all(&repository).unwrap();
    git_in(&repository, &["init", "--initial-branch=main"]);
    git_in(&repository, &["commit", "--allow-empty", "-m", "initial"]);
    let git_dir = repository.join(".git");
    let head = git_in(&repository, &["rev-parse", "HEAD"]);

    for (to_completion, branch) in [(false, "killed"), (true, "finished")] {
        let lock = git_dir.join(format!("refs/heads/{branch}.lock"));
        let (commands, input) = async_channel::bounded(1);
        // The transaction is prepared, holding the lock, and not committed.
        commands
            .send(format!("start\ncreate refs/heads/{branch} {head}\nprepare\n").into_bytes())
            .await
            .unwrap();
        let mut update = host_git::HostGit::in_repository(&git_dir, &["update-ref", "--stdin"]);
        if to_completion {
            update = update.finish_when_dropped();
        }
        // The future is dropped as soon as the lock exists, as a cancelled
        // import drops it.
        let outcome = futures_lite::future::or(
            async {
                let _ = update.output(input).await;
                false
            },
            async {
                for _ in 0..100 {
                    if lock.exists() {
                        return true;
                    }
                    async_io::Timer::after(Duration::from_millis(50)).await;
                }
                false
            },
        )
        .await;
        assert!(outcome, "git took the lock");
        drop(commands);

        let mut released = false;
        for _ in 0..60 {
            if !lock.exists() {
                released = true;
                break;
            }
            async_io::Timer::after(Duration::from_millis(100)).await;
        }
        assert_eq!(
            released, to_completion,
            "{branch}: a killed git leaves its lock, one left to finish removes it"
        );
    }
}

#[test]
fn a_budget_handle_changes_the_setting_for_every_clone() {
    let budget = ImportBudget::new(10);
    let other = budget.clone();
    other.set(0);
    assert_eq!(budget.get(), 0);
    assert_eq!(budget.limit(), None, "zero is unbounded");
    budget.set(7);
    assert_eq!(other.limit(), Some(7));
}

/// A commit holding a file of incompressible bytes.
fn commit_noise(fixture: &Fixture) {
    let noise: Vec<u8> = (0_u32..100_000)
        .map(|n| (n.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    std::fs::write(fixture.worktree.join("noise.bin"), noise).unwrap();
    git_in(&fixture.worktree, &["add", "noise.bin"]);
    git_in(&fixture.worktree, &["commit", "-m", "heavy"]);
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn a_byte_budget_of_zero_bounds_nothing() {
    let limits = ImportLimits {
        bytes: ImportBudget::new(0),
        time: Duration::from_secs(60),
    };
    let fixture = Fixture::with("zero", TestGit::new(), limits).await;
    let mark = fixture.ledger.mark().await.unwrap();
    commit_noise(&fixture);
    let listed = fixture
        .ledger
        .since(&mark)
        .await
        .expect("no byte bound refuses nothing, reports and pack alike");
    assert_eq!(listed.len(), 1);
}

// Sandboxed git runs only off Windows: a package's ancestors carry
// traverse alone, and git resolves its paths through a read on each.
#[cfg(unix)]
#[tokio::test]
async fn the_byte_budget_is_read_at_each_import() {
    let budget = ImportBudget::new(2048);
    let limits = ImportLimits {
        bytes: budget.clone(),
        time: Duration::from_secs(60),
    };
    let fixture = Fixture::with("live", TestGit::new(), limits).await;
    let mark = fixture.ledger.mark().await.unwrap();
    commit_noise(&fixture);
    let error = fixture
        .ledger
        .since(&mark)
        .await
        .expect_err("over the budget");
    assert!(
        matches!(error, CommitLedgerError::Rejected { .. }),
        "{error}"
    );
    // The operator raises it; the ledger already running sees that.
    budget.set(256 * ONE_MIB);
    let listed = fixture.ledger.since(&mark).await.unwrap();
    assert_eq!(listed.len(), 1);
}

#[tokio::test]
#[cfg(unix)]
async fn a_gitfile_the_subagent_replaced_is_never_followed_or_waited_on() {
    let fixture = Fixture::new("gitfile").await;
    let dot_git = fixture.worktree.join(".git");
    let grant = SubagentGrant::new().with_committing_worktree(&fixture.worktree);
    let hostile: [(&str, Box<dyn Fn()>); 3] = [
        (
            "a fifo",
            Box::new(|| {
                std::fs::remove_file(&dot_git).unwrap();
                make_fifo(&dot_git);
            }),
        ),
        (
            "a symlink to an endless file",
            Box::new(|| {
                std::fs::remove_file(&dot_git).unwrap();
                std::os::unix::fs::symlink("/dev/zero", &dot_git).unwrap();
            }),
        ),
        (
            "a file too long for a gitdir line",
            Box::new(|| {
                std::fs::remove_file(&dot_git).unwrap();
                std::fs::write(&dot_git, vec![b'x'; 1024 * 1024]).unwrap();
            }),
        ),
    ];
    for (what, replace) in hostile {
        replace();
        // A fresh history has to read the file to learn where the data is.
        let fresh = finishes_within(
            20,
            main_launch(fixture.dir.path(), &grant, &mut LaunchHistory::default()),
        )
        .await;
        assert!(
            matches!(fresh, Err(SubagentGrantError::Inspect { .. })),
            "{what}: {fresh:?}"
        );
        // A history that holds the layout does not read it, and the file
        // naming the repository is written back.
        let mut history = LaunchHistory::default();
        history
            .committed
            .push(fixture.resolved.committing[0].clone());
        finishes_within(20, main_launch(fixture.dir.path(), &grant, &mut history))
            .await
            .unwrap_or_else(|error| panic!("{what}: {error}"));
        git_in(&fixture.worktree, &["status", "--short"]);
    }
}
