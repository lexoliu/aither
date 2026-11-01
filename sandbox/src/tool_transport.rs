use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;

use askama::Template;
use futures_lite::io::AsyncWriteExt;

use crate::ContainerPath;
use crate::shell_session::{ContainerShellRuntime, SshServer};
use crate::tool_runtime::{
    ImageToolDefect, ImageTools, PreparedTools, ToolBundle, ToolRuntimeError, ToolTarget,
};

/// Trusted initialization transport, separate from agent command execution.
pub trait ToolTransport: Send + Sync {
    /// Runs an initialization script and optionally streams a verified tool archive.
    fn run(
        &self,
        script: &str,
        input: Option<Vec<u8>>,
    ) -> impl Future<Output = Result<Output, ToolRuntimeError>> + Send;
}

/// Tool initialization over an already authorized SSH destination.
#[derive(Debug, Clone)]
pub struct SshToolTransport {
    server: SshServer,
}

impl SshToolTransport {
    /// Binds provisioning to the configured SSH server, not a model-provided target.
    #[must_use]
    pub const fn new(server: SshServer) -> Self {
        Self { server }
    }
}

impl ToolTransport for SshToolTransport {
    async fn run(&self, script: &str, input: Option<Vec<u8>>) -> Result<Output, ToolRuntimeError> {
        let mut command = async_process::Command::new("ssh");
        command
            .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--"])
            .arg(&self.server.target)
            .arg(shell_words::join(["sh", "-c", script]));
        run_command(command, input).await
    }
}

/// Tool initialization using the configured container executor.
#[derive(Debug, Clone)]
pub struct ContainerToolTransport {
    runtime: ContainerShellRuntime,
}

impl ContainerToolTransport {
    /// Binds initialization to an existing container runtime.
    #[must_use]
    pub const fn new(runtime: ContainerShellRuntime) -> Self {
        Self { runtime }
    }
}

impl ToolTransport for ContainerToolTransport {
    async fn run(&self, script: &str, input: Option<Vec<u8>>) -> Result<Output, ToolRuntimeError> {
        let exec = self.runtime.exec();
        let (kill_tx, kill_rx) = async_channel::bounded(1);
        let (input_tx, input_rx) = async_channel::unbounded();
        if let Some(input) = input {
            input_tx
                .try_send(input)
                .map_err(|_| ToolRuntimeError::InitializationInterrupted)?;
        }
        drop(input_tx);
        let execution = exec.exec_boxed(
            self.runtime.container_id(),
            script,
            "/",
            kill_rx,
            input_rx,
            crate::ExecLifetime::Foreground,
        );
        let result = futures_lite::future::or(
            async { execution.await.map_err(ToolRuntimeError::Container) },
            async {
                async_io::Timer::after(Duration::from_secs(120)).await;
                let _ = kill_tx.try_send(());
                Err(ToolRuntimeError::InitializationTimeout)
            },
        )
        .await?;
        match result {
            crate::ContainerExecOutcome::Completed { output, .. } => Ok(output),
            crate::ContainerExecOutcome::Killed => Err(ToolRuntimeError::InitializationInterrupted),
        }
    }
}

async fn run_command(
    mut command: async_process::Command,
    input: Option<Vec<u8>>,
) -> Result<Output, ToolRuntimeError> {
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or(ToolRuntimeError::InitializationInterrupted)?;
    futures_lite::future::or(
        async {
            let send = async {
                if let Some(bytes) = input {
                    stdin.write_all(&bytes).await?;
                }
                drop(stdin);
                Ok::<(), ToolRuntimeError>(())
            };
            let (transferred, output) = futures_lite::future::zip(send, child.output()).await;
            transferred?;
            Ok(output?)
        },
        async {
            async_io::Timer::after(Duration::from_secs(120)).await;
            Err(ToolRuntimeError::InitializationTimeout)
        },
    )
    .await
}

/// Platform information from the target execution environment.
#[derive(Debug, Clone)]
pub struct ToolEnvironmentInfo {
    /// Target platform, not the coordinator's platform.
    pub target: ToolTarget,
    /// Target-owned cache directory outside agent workspaces.
    pub cache_root: PathBuf,
}

/// Probes a trusted backend during initialization, never during model commands.
///
/// # Errors
/// Rejects probe failure, malformed output or unsupported target platforms.
pub async fn probe_tool_environment(
    transport: &impl ToolTransport,
) -> Result<ToolEnvironmentInfo, ToolRuntimeError> {
    let output = transport
        .run(include_str!("../templates/tool_probe.sh"), None)
        .await?;
    let output = successful_output(output)?;
    let mut lines = output.lines();
    let os = lines.next().ok_or(ToolRuntimeError::InvalidProbe)?;
    let arch = lines.next().ok_or(ToolRuntimeError::InvalidProbe)?;
    let home = lines.next().ok_or(ToolRuntimeError::InvalidProbe)?;
    if lines.next().is_some() || !home.starts_with('/') {
        return Err(ToolRuntimeError::InvalidProbe);
    }
    let target = ToolTarget::from_probe(os, arch)?;
    Ok(ToolEnvironmentInfo {
        target,
        cache_root: Path::new(home).join(".cache/aither/tools"),
    })
}

#[derive(Debug)]
struct InstallTool {
    command: String,
    sha256: String,
}

#[derive(Template)]
#[template(path = "tool_install.sh", escape = "none")]
struct ToolInstallTemplate<'a> {
    root: std::borrow::Cow<'a, str>,
    cache_key: String,
    tools: Vec<InstallTool>,
    macos: bool,
    probe_only: bool,
}

#[derive(Template)]
#[template(path = "tool_verify.sh", escape = "none")]
struct ToolVerifyTemplate<'a> {
    directory: std::borrow::Cow<'a, str>,
    tools: Vec<&'static str>,
}

/// The defect `tool_verify.sh` reports on stderr, or `None` for a failure it
/// did not diagnose (a probe command error, for example).
fn image_defect(stderr: &str) -> Option<ImageToolDefect> {
    stderr.lines().find_map(|line| {
        line.strip_prefix("aither-image:")
            .and_then(|marker| match marker.split_once(':') {
                None => {
                    (marker == "missing-directory").then_some(ImageToolDefect::MissingDirectory)
                }
                Some(("missing-binary", command)) => {
                    Some(ImageToolDefect::MissingBinary(command.to_owned()))
                }
                _ => None,
            })
    })
}

impl ImageTools {
    /// Verifies the baseline tools baked into the container image at
    /// [`Self::container_directory`]: every pinned command present as a
    /// regular file and reporting its pinned version.
    ///
    /// The image is expected to contain a directory produced by
    /// [`crate::tool_distribution::ToolDistribution::stage_image_tools`]; a
    /// missing directory or binary fails with
    /// [`ToolRuntimeError::InvalidImageTools`] and a wrong version
    /// with [`ToolRuntimeError::VersionMismatch`], never a fallback to a host
    /// download or mount.
    pub(crate) async fn verify(
        &self,
        transport: &impl ToolTransport,
    ) -> Result<PreparedTools<ContainerPath>, ToolRuntimeError> {
        let directory = self.container_directory()?;
        let script = ToolVerifyTemplate {
            directory: shell_words::quote(directory.as_str()),
            tools: self.tools.iter().map(|tool| tool.tool.command()).collect(),
        }
        .render()?;
        let output = transport.run(&script, None).await?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if let Some(defect) = image_defect(&stderr) {
                return Err(ToolRuntimeError::InvalidImageTools { directory, defect });
            }
            return Err(ToolRuntimeError::InitializationCommand {
                status: output.status.code(),
                stderr: stderr.into_owned(),
            });
        }
        let output = String::from_utf8(output.stdout)?;
        let catalog = self.catalog();
        catalog.verify_versions(&output)?;
        Ok(PreparedTools {
            bin_dir: self.container_bin_dir()?,
            catalog,
        })
    }
}

impl ToolBundle {
    /// Installs verified artifacts through a backend's independent initialization transport.
    ///
    /// # Errors
    /// Rejects failed transfers, a tampered cache or invalid activation output.
    pub async fn prepare_remote<F, Fut>(
        &self,
        transport: &impl ToolTransport,
        root: &Path,
        authorize: F,
    ) -> Result<PreparedTools<PathBuf>, ToolRuntimeError>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<(), ToolRuntimeError>> + Send,
    {
        let root_text = root.to_str().ok_or(ToolRuntimeError::InvalidProbe)?;
        let mut template = ToolInstallTemplate {
            root: shell_words::quote(root_text),
            cache_key: self.cache_key()?,
            tools: self
                .binaries
                .iter()
                .map(|binary| InstallTool {
                    command: binary.tool.command().into(),
                    sha256: hex::encode(binary.sha256),
                })
                .collect(),
            macos: self.target().os() == "macos",
            probe_only: true,
        };
        let probe = transport.run(&template.render()?, None).await?;
        let output = if probe.status.code() == Some(3) {
            authorize().await?;
            template.probe_only = false;
            let bundle = self.clone();
            let archive = blocking::unblock(move || bundle.archive()).await?;
            transport.run(&template.render()?, Some(archive)).await?
        } else {
            probe
        };
        let path = successful_output(output)?;
        let mut lines = path.lines();
        let bin_dir = lines.next().ok_or(ToolRuntimeError::InvalidProbe)?;
        if lines.next().is_some() || !bin_dir.starts_with('/') {
            return Err(ToolRuntimeError::InvalidProbe);
        }
        Ok(PreparedTools {
            bin_dir: bin_dir.into(),
            catalog: self.catalog(),
        })
    }

    fn archive(&self) -> Result<Vec<u8>, ToolRuntimeError> {
        let mut archive = tar::Builder::new(Vec::new());
        for binary in self.binaries.iter() {
            let mut header = tar::Header::new_gnu();
            header.set_size(binary.bytes.len() as u64);
            header.set_mode(0o555);
            header.set_cksum();
            archive.append_data(
                &mut header,
                Path::new("bin").join(binary.tool.command()),
                binary.bytes.as_ref(),
            )?;
        }
        Ok(archive.into_inner()?)
    }
}

fn successful_output(output: Output) -> Result<String, ToolRuntimeError> {
    if !output.status.success() {
        return Err(ToolRuntimeError::InitializationCommand {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(String::from_utf8(output.stdout)?)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The `heel run` wrapper the verify script puts around each version probe.
    /// Host runs drop it: the staged binaries are the host's own scripts.
    const HEEL_RUN_PREFIX: &str =
        "heel run --network deny --working-dir \"$scratch\" --executable \"$directory/bin\" -- ";

    /// Runs the verify script on the host as a container would inside the
    /// image: the image's `/opt/aither/tools` root is replaced by a directory
    /// populated via [`ToolDistribution::stage_image_tools`], so the real
    /// presence and version checks run against the staged bytes.
    struct StagedImageTransport {
        /// Host directory standing in for the image's `/opt/aither/tools`.
        image_root: PathBuf,
    }

    impl ToolTransport for StagedImageTransport {
        async fn run(
            &self,
            script: &str,
            input: Option<Vec<u8>>,
        ) -> Result<Output, ToolRuntimeError> {
            assert!(input.is_none());
            let script = script
                .replace("/opt/aither/tools", &self.image_root.to_string_lossy())
                .replace(HEEL_RUN_PREFIX, "");
            let mut command = async_process::Command::new("sh");
            command.args(["-c", &script]);
            run_command(command, input).await
        }
    }

    fn image_transport(root: &tempfile::TempDir) -> StagedImageTransport {
        StagedImageTransport {
            image_root: root.path().to_path_buf(),
        }
    }

    /// A file-backed `script_distribution`, the directory holding its artifact
    /// files (keep alive while `load` reads them) and its pinned `ImageTools`.
    fn script_fixture(
        version: &str,
        url: impl Fn(&Path) -> crate::tool_distribution::ToolArtifactLocation,
    ) -> (
        tempfile::TempDir,
        crate::tool_distribution::ToolDistribution,
        ImageTools,
    ) {
        use crate::tool_distribution::tests::script_distribution;
        let files = tempfile::tempdir().unwrap();
        let target = ToolTarget::native().unwrap();
        let distribution = script_distribution(&target, version, files.path(), url);
        let image = distribution.image_tools(&target).unwrap();
        (files, distribution, image)
    }

    #[tokio::test]
    async fn image_verification_accepts_a_staged_directory() {
        use crate::tool_distribution::tests::file_artifact;
        let (files, distribution, image) = script_fixture("1.0.0", file_artifact);
        let out = tempfile::tempdir().unwrap();
        let target = ToolTarget::native().unwrap();
        distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        let tools = image.verify(&image_transport(&out)).await.unwrap();
        assert_eq!(
            tools.bin_dir().as_str(),
            format!("{}/bin", image.container_directory().unwrap())
        );
    }

    #[tokio::test]
    async fn image_verification_names_a_missing_binary() {
        use crate::tool_distribution::tests::file_artifact;
        let (files, distribution, image) = script_fixture("1.0.0", file_artifact);
        let out = tempfile::tempdir().unwrap();
        let target = ToolTarget::native().unwrap();
        let staged = distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        std::fs::remove_file(staged.join("bin").join("rg")).unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::InvalidImageTools { directory, defect }) => {
                assert_eq!(
                    directory.as_str(),
                    image.container_directory().unwrap().as_str()
                );
                assert_eq!(defect, ImageToolDefect::MissingBinary("rg".to_owned()));
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_verification_names_a_wrong_version() {
        use crate::tool_distribution::tests::{file_artifact, script_tool};
        let (files, distribution, image) = script_fixture("1.0.0", file_artifact);
        let out = tempfile::tempdir().unwrap();
        let target = ToolTarget::native().unwrap();
        let staged = distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        // A binary reporting a version other than the pin: present, regular,
        // but rejected by the version probe.
        let binary = staged.join("bin").join("jq");
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&binary).unwrap().permissions().mode();
            std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(mode | 0o200))
                .unwrap();
        }
        std::fs::write(&binary, script_tool("jq", "2.0.0")).unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::VersionMismatch(command)) => assert_eq!(command, "jq"),
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_verification_names_an_absent_directory() {
        use crate::tool_distribution::tests::file_artifact;
        let (_files, _, image) = script_fixture("1.0.0", file_artifact);
        let out = tempfile::tempdir().unwrap();
        let target = ToolTarget::native().unwrap();
        // Pinned to another version the same executables stage under a
        // different cache key, so this image's directory is absent.
        let (other_files, other, _) = script_fixture("2.0.0", file_artifact);
        other
            .stage_image_tools(&target, &other_files.path().join("cache"), out.path())
            .await
            .unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::InvalidImageTools { directory, defect }) => {
                assert_eq!(
                    directory.as_str(),
                    image.container_directory().unwrap().as_str()
                );
                assert_eq!(defect, ImageToolDefect::MissingDirectory);
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_verification_downloads_nothing() {
        use crate::tool_distribution::tests::{file_artifact, unreachable_artifact};
        let (files, distribution, _) = script_fixture("1.0.0", file_artifact);
        let out = tempfile::tempdir().unwrap();
        let target = ToolTarget::native().unwrap();
        distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        // Identical pins behind unreachable artifact URLs: `image_tools` and
        // `verify` never touch the network or the artifact cache.
        let (_, _, image) = script_fixture("1.0.0", unreachable_artifact);
        image.verify(&image_transport(&out)).await.unwrap();
    }

    struct LocalTransport;
    impl ToolTransport for LocalTransport {
        async fn run(
            &self,
            script: &str,
            input: Option<Vec<u8>>,
        ) -> Result<Output, ToolRuntimeError> {
            let mut command = async_process::Command::new("sh");
            command.args(["-c", script]);
            run_command(command, input).await
        }
    }

    #[tokio::test]
    async fn initialization_script_installs_and_warm_cache_skips_transfer() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        let first = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await
            .unwrap();
        let second = bundle
            .prepare_remote(&LocalTransport, root.path(), || async {
                Err(ToolRuntimeError::InstallDenied(
                    "warm cache must not ask".into(),
                ))
            })
            .await
            .unwrap();
        assert_eq!(first.bin_dir(), second.bin_dir());
        for tool in crate::ManagedTool::ALL {
            assert!(first.bin_dir().join(tool.command()).is_file());
        }
    }

    #[tokio::test]
    async fn denied_remote_install_never_activates_tools() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        let result = bundle
            .prepare_remote(&LocalTransport, root.path(), || async {
                Err(ToolRuntimeError::InstallDenied("test server".into()))
            })
            .await;
        assert!(matches!(result, Err(ToolRuntimeError::InstallDenied(_))));
        let activation = root.path().join(bundle.cache_key().unwrap());
        assert_eq!(
            async_fs::symlink_metadata(activation)
                .await
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::NotFound
        );
    }

    #[tokio::test]
    async fn concurrent_remote_activation_is_atomic() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        let (first, second) = futures_lite::future::zip(
            bundle.prepare_remote(&LocalTransport, root.path(), || async { Ok(()) }),
            bundle.prepare_remote(&LocalTransport, root.path(), || async { Ok(()) }),
        )
        .await;
        assert_eq!(first.unwrap().bin_dir(), second.unwrap().bin_dir());
    }
}
