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

/// A trusted manifest serving `binary` for every managed tool, pinned to `version`.
pub async fn fixture_distribution(
    binary: &std::path::Path,
    version: &str,
) -> aither_sandbox::ToolDistribution {
    use aither_sandbox::{
        ManagedTool, ToolArtifact, ToolArtifactFormat, ToolDistribution, ToolTarget,
    };
    use sha2::Digest as _;
    #[derive(serde::Serialize)]
    struct Manifest {
        artifacts: Vec<ToolArtifact>,
    }
    let bytes = async_fs::read(binary).await.unwrap();
    let artifacts = ManagedTool::ALL
        .into_iter()
        .map(|tool| ToolArtifact {
            tool,
            version: version.into(),
            target: ToolTarget::native().unwrap(),
            url: aither_sandbox::ToolArtifactLocation::Download(
                url::Url::from_file_path(binary).unwrap(),
            ),
            sha256: hex::encode(sha2::Sha256::digest(&bytes)),
            encoding: ToolArtifactFormat::Executable,
        })
        .collect();
    ToolDistribution::from_json(&serde_json::to_vec(&Manifest { artifacts }).unwrap()).unwrap()
}
