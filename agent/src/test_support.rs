//! Fixtures shared by the crate's tests.

/// Compiles the baseline-tool fixture, which reports version 1.0.0 for
/// every command, into `source`.
pub async fn compile_baseline_fixture(source: &std::path::Path) -> std::path::PathBuf {
    let fixture = source.join("baseline_tool.rs");
    async_fs::write(
        &fixture,
        include_bytes!("../templates/test_baseline_tool.rs"),
    )
    .await
    .unwrap();
    let path = source
        .join("baseline-tool")
        .with_extension(std::env::consts::EXE_EXTENSION);
    let compiled = async_process::Command::new("rustc")
        .arg("--crate-name")
        .arg("baseline_tool")
        .arg(&fixture)
        .arg("-o")
        .arg(&path)
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
            encoding: ToolArtifactFormat::Executable,
        })
        .collect()
}

fn distribution_of(artifacts: &[serde_json::Value]) -> aither_sandbox::ToolDistribution {
    aither_sandbox::ToolDistribution::from_json(
        &serde_json::to_vec(&serde_json::json!({ "artifacts": artifacts })).unwrap(),
    )
    .unwrap()
}

/// A trusted manifest serving `binary` for every managed tool, pinned to `version`.
pub async fn fixture_distribution(
    binary: &std::path::Path,
    version: &str,
) -> aither_sandbox::ToolDistribution {
    let artifacts = fixture_artifacts(binary, version, &aither_sandbox::ManagedTool::ALL).await;
    distribution_of(
        &artifacts
            .iter()
            .map(|artifact| serde_json::to_value(artifact).unwrap())
            .collect::<Vec<_>>(),
    )
}

/// The distribution of the real baseline tools: the repository's
/// hash-pinned `ripgrep`, `ast-grep` and `jq` release artifacts
/// (`patch/third-party-tools.json`, downloaded and verified on first use like
/// any distribution), with `binary` standing in for `apply_patch`, which is
/// only published with a release.
pub async fn real_distribution(binary: &std::path::Path) -> aither_sandbox::ToolDistribution {
    let pinned =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../patch/third-party-tools.json");
    let pinned: serde_json::Value =
        serde_json::from_slice(&async_fs::read(pinned).await.unwrap()).unwrap();
    let mut artifacts = pinned["artifacts"].as_array().unwrap().clone();
    artifacts.extend(
        fixture_artifacts(binary, "1.0.0", &[aither_sandbox::ManagedTool::ApplyPatch])
            .await
            .iter()
            .map(|artifact| serde_json::to_value(artifact).unwrap()),
    );
    distribution_of(&artifacts)
}

/// Baseline tools prepared as a terminal prepares them, kept alive with
/// their directories.
pub struct PreparedFixture {
    /// The prepared tools.
    pub tools: aither_sandbox::PreparedTools<std::path::PathBuf>,
    _source: tempfile::TempDir,
    _cache: tempfile::TempDir,
}

/// Prepares the stub baseline tools.
pub async fn prepared_fixture() -> PreparedFixture {
    let source = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
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
