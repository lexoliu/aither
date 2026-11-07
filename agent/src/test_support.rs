//! Fixtures shared by the crate's tests.

/// The `CPython` version the baseline fixture's `uv` installs and reports.
const BASELINE_PYTHON_VERSION: &str = "3.13.1";

/// An output store in `dir`, leasing from the default allocator.
pub async fn output_store(dir: impl Into<std::path::PathBuf>) -> aither_sandbox::OutputStore {
    aither_sandbox::OutputStore::new(dir, aither_core::AnyIdAllocator::default())
        .await
        .expect("the output directory can be created")
}

/// Drives `run` to its end without looking at its events.
pub async fn finish_run(
    run: crate::Run<impl std::future::Future<Output = Result<(), crate::AgentError>>>,
) -> Result<(), crate::AgentError> {
    run.drive((), std::future::pending(), |(), _| async {
        std::ops::ControlFlow::<std::convert::Infallible, _>::Continue(())
    })
    .await
    .map(|_| ())
}

/// A temporary directory that can hold prepared tool runtimes: the managed
/// interpreter tree is read-only, which a plain `TempDir` cannot delete.
pub struct TempDir(tempfile::TempDir);

impl TempDir {
    pub fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }
}

impl std::ops::Deref for TempDir {
    type Target = tempfile::TempDir;

    fn deref(&self) -> &tempfile::TempDir {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut pending = vec![self.0.path().to_path_buf()];
            while let Some(path) = pending.pop() {
                let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                    continue;
                };
                if metadata.is_symlink() {
                    continue;
                }
                let mode = metadata.permissions().mode() | 0o200;
                let _ =
                    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode | 0o100));
                if metadata.is_dir()
                    && let Ok(entries) = std::fs::read_dir(&path)
                {
                    pending.extend(entries.flatten().map(|entry| entry.path()));
                }
            }
        }
    }
}

/// Where tool distributions download and prepare their runtimes.
pub struct ToolCache {
    path: std::path::PathBuf,
    _temporary: Option<TempDir>,
}

impl ToolCache {
    /// Environment variable naming a directory that outlives the test
    /// process; CI restores it from the Actions cache so a pinned download
    /// reaches the network only once.
    pub const SHARED_ENVIRONMENT: &str = "AITHER_TOOL_CACHE";

    /// A cache of its own, deleted on drop.
    pub fn temporary() -> Self {
        let directory = TempDir::new();
        Self {
            path: directory.path().to_path_buf(),
            _temporary: Some(directory),
        }
    }

    /// The directory named by [`Self::SHARED_ENVIRONMENT`], which is shared by
    /// every test process and never deleted, or a temporary one when unset.
    /// Its runtimes and artifacts are content-addressed and verified on reuse,
    /// and concurrent preparers take a lock per runtime.
    pub fn shared() -> Self {
        std::env::var_os(Self::SHARED_ENVIRONMENT).map_or_else(Self::temporary, |path| Self {
            path: path.into(),
            _temporary: None,
        })
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// Compiles the baseline-tool fixture, which reports version 1.0.0 for
/// every command, into `source`.
///
/// `rustc` runs in `source` on relative paths, so no absolute path is embedded
/// and the same toolchain yields the same bytes wherever `source` is. The
/// fixture's digest is part of a prepared runtime's cache key, so this is what
/// lets a restored `AITHER_TOOL_CACHE` serve the runtime again.
pub async fn compile_baseline_fixture(source: &std::path::Path) -> std::path::PathBuf {
    let fixture = std::path::Path::new("baseline_tool.rs");
    async_fs::write(
        source.join(fixture),
        include_bytes!("../templates/test_baseline_tool.rs"),
    )
    .await
    .unwrap();
    let output =
        std::path::Path::new("baseline-tool").with_extension(std::env::consts::EXE_EXTENSION);
    let path = source.join(&output);
    let compiled = async_process::Command::new("rustc")
        .current_dir(source)
        .env(
            "BASELINE_ALIASES",
            aither_sandbox::ManagedTool::Coreutils.aliases().join("\n"),
        )
        .env("BASELINE_PYTHON_VERSION", BASELINE_PYTHON_VERSION)
        .env(
            "BASELINE_COMMANDS",
            aither_sandbox::ManagedTool::ALL
                .map(aither_sandbox::ManagedTool::command)
                .join("\n"),
        )
        .arg("--crate-name")
        .arg("baseline_tool")
        .arg(fixture)
        .arg("-o")
        .arg(&output)
        .kill_on_drop(true)
        .output()
        .await
        .unwrap();
    assert!(compiled.status.success(), "{compiled:?}");
    path
}

/// Pinned artifacts serving `binary` as each of `tools`, at `version`.
async fn fixture_artifacts(
    binary: &std::path::Path,
    version: &str,
    tools: &[aither_sandbox::ManagedTool],
) -> Vec<aither_sandbox::ToolArtifact> {
    use aither_sandbox::{ToolArtifact, ToolArtifactFormat, ToolTarget};
    use sha2::Digest as _;
    let bytes = async_fs::read(binary).await.unwrap();
    tools
        .iter()
        .map(|&tool| ToolArtifact {
            tool,
            version: version.into(),
            target: ToolTarget::native().unwrap(),
            url: aither_sandbox::ToolArtifactLocation::Download(
                url::Url::from_file_path(binary).unwrap(),
            ),
            sha256: hex::encode(sha2::Sha256::digest(&bytes)),
            size: bytes.len() as u64,
            encoding: ToolArtifactFormat::Executable,
        })
        .collect()
}

fn distribution_of(
    artifacts: &[serde_json::Value],
    python: &[serde_json::Value],
) -> aither_sandbox::ToolDistribution {
    aither_sandbox::ToolDistribution::from_json(
        &serde_json::to_vec(&serde_json::json!({ "artifacts": artifacts, "python": python }))
            .unwrap(),
    )
    .unwrap()
}

/// The interpreter pin of the fixture `uv`, which installs a copy of `binary`.
async fn fixture_python(binary: &std::path::Path) -> Vec<serde_json::Value> {
    use sha2::Digest as _;
    let target = aither_sandbox::ToolTarget::native().unwrap();
    if !target.provides_python() {
        return Vec::new();
    }
    let bytes = async_fs::read(binary).await.unwrap();
    vec![
        serde_json::to_value(aither_sandbox::PythonPin {
            version: BASELINE_PYTHON_VERSION.into(),
            target,
            interpreter_sha256: hex::encode(sha2::Sha256::digest(&bytes)),
        })
        .unwrap(),
    ]
}

/// A trusted manifest serving `binary` for every managed tool, pinned to `version`.
pub async fn fixture_distribution(
    binary: &std::path::Path,
    version: &str,
) -> aither_sandbox::ToolDistribution {
    let native = aither_sandbox::ToolTarget::native().unwrap();
    let artifacts = fixture_artifacts(binary, version, native.required_tools()).await;
    distribution_of(
        &artifacts
            .iter()
            .map(|artifact| serde_json::to_value(artifact).unwrap())
            .collect::<Vec<_>>(),
        &fixture_python(binary).await,
    )
}

/// The distribution of the real baseline tools: the repository's
/// hash-pinned release artifacts and interpreter pins
/// (`patch/third-party-tools.json`, downloaded and verified on first use like
/// any distribution), with `binary` standing in for `apply_patch`, `aither-sed`
/// and `bash`, which are only published with a release.
pub async fn real_distribution(binary: &std::path::Path) -> aither_sandbox::ToolDistribution {
    let pinned =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../patch/third-party-tools.json");
    let pinned: serde_json::Value =
        serde_json::from_slice(&async_fs::read(pinned).await.unwrap()).unwrap();
    let mut artifacts = pinned["artifacts"].as_array().unwrap().clone();
    artifacts.extend(
        fixture_artifacts(
            binary,
            "1.0.0",
            &[
                aither_sandbox::ManagedTool::ApplyPatch,
                aither_sandbox::ManagedTool::Sed,
                aither_sandbox::ManagedTool::Bash,
            ],
        )
        .await
        .iter()
        .map(|artifact| serde_json::to_value(artifact).unwrap()),
    );
    distribution_of(&artifacts, pinned["python"].as_array().unwrap())
}

/// Baseline tools prepared as a terminal prepares them, kept alive with
/// their directories.
pub struct PreparedFixture {
    /// The prepared tools.
    pub tools: aither_sandbox::PreparedTools<std::path::PathBuf>,
    _source: tempfile::TempDir,
    _cache: TempDir,
}

/// Prepares the stub baseline tools.
pub async fn prepared_fixture() -> PreparedFixture {
    let source = tempfile::tempdir().unwrap();
    let cache = TempDir::new();
    let binary = compile_baseline_fixture(source.path()).await;
    let bundle = fixture_distribution(&binary, "1.0.0")
        .await
        .load(
            &aither_sandbox::ToolTarget::native().unwrap(),
            &cache.path().join("artifacts"),
        )
        .await
        .unwrap();
    let tools = bundle
        .prepare_local(&cache.path().join("runtimes"))
        .await
        .unwrap();
    PreparedFixture {
        tools,
        _source: source,
        _cache: cache,
    }
}
