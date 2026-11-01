use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;

use askama::Template;
use futures_lite::io::AsyncWriteExt;

use crate::ContainerPath;
use crate::shell_session::{ContainerShellRuntime, SshServer};
use crate::tool_runtime::{
    ImageToolDefect, PreparedTools, ToolBundle, ToolRuntimeError, ToolTarget,
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
    tools: Vec<InstallTool>,
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
                Some(("sha256-mismatch", command)) => {
                    Some(ImageToolDefect::Sha256Mismatch(command.to_owned()))
                }
                _ => None,
            })
    })
}

impl ToolBundle {
    /// Verifies the baseline tools baked into the container image at
    /// [`Self::container_directory`]: every executable present with its pinned
    /// SHA-256 and reporting its pinned version.
    ///
    /// The image is expected to contain a directory produced by
    /// [`Self::stage`]; a missing or mismatched directory fails with
    /// [`ToolRuntimeError::InvalidImageTools`], never a fallback to a host
    /// download or mount.
    pub(crate) async fn verify_container_image(
        &self,
        transport: &impl ToolTransport,
    ) -> Result<PreparedTools<ContainerPath>, ToolRuntimeError> {
        let directory = self.container_directory()?;
        let script = ToolVerifyTemplate {
            directory: shell_words::quote(directory.as_str()),
            tools: self
                .binaries
                .iter()
                .map(|binary| InstallTool {
                    command: binary.tool.command().into(),
                    sha256: hex::encode(binary.sha256),
                })
                .collect(),
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

    /// macOS lacks `sha256sum`; `shasum -a 256` prints the same `<hash>  <file>` format.
    const SHA256SUM: &str = if cfg!(target_os = "macos") {
        "shasum -a 256"
    } else {
        "sha256sum"
    };

    /// The `heel run` wrapper the verify script puts around each version probe.
    /// Host runs drop it: the staged binaries are the host's own scripts.
    const HEEL_RUN_PREFIX: &str =
        "heel run --network deny --working-dir \"$scratch\" --executable \"$directory/bin\" -- ";

    /// Runs the verify script on the host as a container would inside the
    /// image: the image's `/opt/aither/tools` root is replaced by a directory
    /// populated via [`ToolBundle::stage`], so the real SHA-256 and version
    /// checks run against the staged bytes.
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
                .replace(HEEL_RUN_PREFIX, "")
                .replace("sha256sum", SHA256SUM);
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

    #[tokio::test]
    async fn image_verification_accepts_a_staged_directory() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        bundle.stage(root.path()).await.unwrap();
        let tools = bundle
            .verify_container_image(&image_transport(&root))
            .await
            .unwrap();
        assert_eq!(
            tools.bin_dir().as_str(),
            format!("{}/bin", bundle.container_directory().unwrap())
        );
    }

    #[tokio::test]
    async fn image_verification_names_a_missing_binary() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        let staged = bundle.stage(root.path()).await.unwrap();
        std::fs::remove_file(staged.join("bin").join("rg")).unwrap();
        match bundle.verify_container_image(&image_transport(&root)).await {
            Err(ToolRuntimeError::InvalidImageTools { directory, defect }) => {
                assert_eq!(
                    directory.as_str(),
                    bundle.container_directory().unwrap().as_str()
                );
                assert_eq!(defect, ImageToolDefect::MissingBinary("rg".to_owned()));
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_verification_names_a_tampered_binary() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        let staged = bundle.stage(root.path()).await.unwrap();
        let binary = staged.join("bin").join("jq");
        let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions = std::fs::Permissions::from_mode(permissions.mode() | 0o200);
        }
        #[cfg(not(unix))]
        permissions.set_readonly(false);
        std::fs::set_permissions(&binary, permissions).unwrap();
        std::fs::write(&binary, b"tampered").unwrap();
        match bundle.verify_container_image(&image_transport(&root)).await {
            Err(ToolRuntimeError::InvalidImageTools { directory, defect }) => {
                assert_eq!(
                    directory.as_str(),
                    bundle.container_directory().unwrap().as_str()
                );
                assert_eq!(defect, ImageToolDefect::Sha256Mismatch("jq".to_owned()));
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_verification_names_an_absent_directory() {
        let root = tempfile::tempdir().unwrap();
        let bundle = crate::tool_runtime::tests::bundle();
        // A bundle pinned to another version stages under a different cache
        // key, so this bundle's directory is absent even though tools exist.
        crate::tool_runtime::tests::bundle_pinned("2.0.0")
            .stage(root.path())
            .await
            .unwrap();
        match bundle.verify_container_image(&image_transport(&root)).await {
            Err(ToolRuntimeError::InvalidImageTools { directory, defect }) => {
                assert_eq!(
                    directory.as_str(),
                    bundle.container_directory().unwrap().as_str()
                );
                assert_eq!(defect, ImageToolDefect::MissingDirectory);
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
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
