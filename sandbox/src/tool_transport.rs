use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;

use askama::Template;
use futures_lite::io::AsyncWriteExt;

use crate::ContainerPath;
use crate::shell_session::{ContainerShellRuntime, SshServer};
use crate::tool_python::PythonScript;
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

/// Initialization on the coordinator's own machine.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalToolTransport;

impl ToolTransport for LocalToolTransport {
    async fn run(&self, script: &str, input: Option<Vec<u8>>) -> Result<Output, ToolRuntimeError> {
        let mut command = async_process::Command::new("sh");
        command.args(["-c", script]);
        run_command(command, input).await
    }
}

/// Exit status `clt_check.sh` uses for a missing Command Line Tools install.
const CLT_MISSING_STATUS: i32 = 4;

/// Requires the Command Line Tools on a macOS target.
///
/// The check reads `xcode-select -p` and looks for `usr/bin/git` in that
/// directory; it never runs `/usr/bin/git`, whose shim opens an install dialog.
/// Other targets pass untouched.
///
/// # Errors
/// Fails with [`ToolRuntimeError::CommandLineToolsMissing`] when the tools are
/// absent, or with the transport's error when the check cannot run.
pub async fn ensure_command_line_tools(
    transport: &impl ToolTransport,
    target: &ToolTarget,
) -> Result<(), ToolRuntimeError> {
    if target.os() != "macos" {
        return Ok(());
    }
    let output = transport
        .run(include_str!("../templates/clt_check.sh"), None)
        .await?;
    match output.status.code() {
        Some(0) => Ok(()),
        Some(CLT_MISSING_STATUS) => Err(ToolRuntimeError::CommandLineToolsMissing),
        status => Err(ToolRuntimeError::InitializationCommand {
            status,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }),
    }
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
    /// Commands installed as relative links to `command`, space separated.
    aliases: String,
}

#[derive(Template)]
#[template(path = "tool_install.sh", escape = "none")]
struct ToolInstallTemplate<'a> {
    root: std::borrow::Cow<'a, str>,
    cache_key: String,
    tools: Vec<InstallTool>,
    python: Option<PythonScript>,
    macos: bool,
    probe_only: bool,
}

#[derive(Template)]
#[template(path = "tool_verify.sh", escape = "none")]
struct ToolVerifyTemplate<'a> {
    directory: std::borrow::Cow<'a, str>,
    tools: Vec<VerifyTool>,
    python: Option<PythonScript>,
}

struct VerifyTool {
    command: &'static str,
    /// Commands that must be relative links to `command`, space separated.
    aliases: String,
}

/// The defect `tool_verify.sh` reports on stderr, or `None` for a failure it
/// did not diagnose (a probe command error, for example).
fn image_defect(stderr: &str) -> Option<ImageToolDefect> {
    stderr.lines().find_map(|line| {
        line.strip_prefix("aither-image:")
            .and_then(|marker| match marker.split_once(':') {
                None => match marker {
                    "missing-directory" => Some(ImageToolDefect::MissingDirectory),
                    "invalid-python" => Some(ImageToolDefect::InvalidPython),
                    _ => None,
                },
                Some(("missing-binary", command)) => {
                    Some(ImageToolDefect::MissingBinary(command.to_owned()))
                }
                Some(("invalid-alias", alias)) => {
                    Some(ImageToolDefect::InvalidAlias(alias.to_owned()))
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
            tools: self
                .tools
                .iter()
                .map(|tool| VerifyTool {
                    command: tool.tool.command(),
                    aliases: tool.tool.aliases().join(" "),
                })
                .collect(),
            python: self
                .python
                .as_ref()
                .map(|python| python.script(&self.target))
                .transpose()?,
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
            uv_dir: None,
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
                    aliases: binary.tool.aliases().join(" "),
                })
                .collect(),
            python: self
                .python
                .as_ref()
                .map(|python| python.script(self.target()))
                .transpose()?,
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
            // The install script creates it beside the runtimes.
            uv_dir: self
                .python
                .is_some()
                .then(|| root.join(crate::tool_runtime::UV_STATE_DIRECTORY)),
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
            for alias in binary.tool.aliases() {
                let mut header = tar::Header::new_gnu();
                header.set_entry_type(tar::EntryType::Symlink);
                header.set_size(0);
                header.set_mode(0o777);
                archive.append_link(
                    &mut header,
                    Path::new("bin").join(alias),
                    binary.tool.command(),
                )?;
            }
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
        let out = crate::tool_runtime::tests::unlocking_tempdir();
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
        let out = crate::tool_runtime::tests::unlocking_tempdir();
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
        let out = crate::tool_runtime::tests::unlocking_tempdir();
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
        std::fs::write(&binary, script_tool(crate::ManagedTool::Jq, "2.0.0")).unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::VersionMismatch(command)) => assert_eq!(command, "jq"),
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_verification_names_an_absent_directory() {
        use crate::tool_distribution::tests::file_artifact;
        let (_files, _, image) = script_fixture("1.0.0", file_artifact);
        let out = crate::tool_runtime::tests::unlocking_tempdir();
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
        let out = crate::tool_runtime::tests::unlocking_tempdir();
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
        let root = crate::tool_runtime::tests::unlocking_tempdir();
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
        for &tool in crate::tool_runtime::tests::native_tools() {
            assert!(first.bin_dir().join(tool.command()).is_file());
        }
        for alias in crate::tool_runtime::COREUTILS_ALIASES {
            assert_eq!(
                std::fs::read_link(first.bin_dir().join(alias)).unwrap(),
                Path::new("coreutils"),
                "{alias}"
            );
        }
    }

    #[tokio::test]
    async fn remote_install_rejects_a_tampered_alias_in_the_cache() {
        let root = crate::tool_runtime::tests::unlocking_tempdir();
        let bundle = crate::tool_runtime::tests::bundle();
        let installed = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await
            .unwrap();
        let alias = installed.bin_dir().join("date");
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(installed.bin_dir(), std::fs::Permissions::from_mode(0o755))
                .unwrap();
        }
        std::fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink("rg", &alias).unwrap();
        let result = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await;
        assert!(
            matches!(
                result,
                Err(ToolRuntimeError::InitializationCommand { ref stderr, .. })
                    if stderr.contains("tool cache integrity failure")
            ),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn image_verification_rejects_a_tampered_alias() {
        use crate::tool_distribution::tests::file_artifact;
        let (files, distribution, image) = script_fixture("1.0.0", file_artifact);
        let out = crate::tool_runtime::tests::unlocking_tempdir();
        let target = ToolTarget::native().unwrap();
        let staged = distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        image.verify(&image_transport(&out)).await.unwrap();
        let alias = staged.join("bin").join("date");
        std::fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink("rg", &alias).unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::InvalidImageTools { defect, .. }) => {
                assert_eq!(defect, ImageToolDefect::InvalidAlias("date".to_owned()));
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
        std::fs::remove_file(&alias).unwrap();
        std::fs::write(&alias, b"#!/bin/sh\n").unwrap();
        assert!(matches!(
            image.verify(&image_transport(&out)).await,
            Err(ToolRuntimeError::InvalidImageTools {
                defect: ImageToolDefect::InvalidAlias(_),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn remote_install_places_the_managed_python_read_only() {
        use std::os::unix::fs::PermissionsExt;
        let root = crate::tool_runtime::tests::unlocking_tempdir();
        let bundle = crate::tool_runtime::tests::bundle();
        let prepared = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await
            .unwrap();
        let request = bundle
            .python
            .as_ref()
            .unwrap()
            .request(bundle.target())
            .unwrap();
        let link = format!(".python/{request}/bin/python3.13");
        let bin = prepared.bin_dir();
        assert_eq!(
            std::fs::read_link(bin.join("python3")).unwrap(),
            Path::new(&link)
        );
        let interpreter = bin.join(&link);
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&interpreter), 0o555);
        // The interpreter tree, its directories included, has no write bit.
        for directory in [
            bin.join(".python"),
            interpreter.parent().unwrap().to_path_buf(),
        ] {
            assert_eq!(mode(&directory) & 0o222, 0, "{}", directory.display());
        }
        assert_eq!(mode(bin), 0o555);
        // Only the tree was kept of what uv wrote.
        assert!(!bin.join(".python-install").exists());
        // The shared uv directory sits beside the runtimes, writable by the user only.
        assert_eq!(prepared.uv_dir(), Some(&root.path().join("uv")));
        assert_eq!(mode(&root.path().join("uv")), 0o700);
        let output = std::process::Command::new(bin.join("python3"))
            .arg("--version")
            .output()
            .unwrap();
        assert_eq!(String::from_utf8(output.stdout).unwrap(), "Python 3.13.1\n");
    }

    #[tokio::test]
    async fn remote_cache_with_a_tampered_python_is_rejected() {
        use std::os::unix::fs::PermissionsExt;
        let root = crate::tool_runtime::tests::unlocking_tempdir();
        let bundle = crate::tool_runtime::tests::bundle();
        let installed = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await
            .unwrap();
        let python = installed.bin_dir().join("python3");
        let interpreter = installed
            .bin_dir()
            .join(std::fs::read_link(&python).unwrap());
        let rejected = |result: Result<PreparedTools<PathBuf>, ToolRuntimeError>| {
            matches!(
                result,
                Err(ToolRuntimeError::InitializationCommand { ref stderr, .. })
                    if stderr.contains("tool cache integrity failure")
            )
        };
        // The interpreter's bytes change.
        let parent = interpreter.parent().unwrap().to_path_buf();
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(&interpreter, b"#!/bin/sh\necho Python 9.9.9\n").unwrap();
        std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o555)).unwrap();
        let result = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await;
        assert!(rejected(result), "a replaced interpreter was accepted");
        // The link points elsewhere.
        std::fs::set_permissions(installed.bin_dir(), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        std::fs::remove_file(&python).unwrap();
        std::os::unix::fs::symlink("../elsewhere", &python).unwrap();
        let result = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await;
        assert!(rejected(result), "a retargeted python3 was accepted");
    }

    #[tokio::test]
    async fn remote_install_stops_on_a_failed_or_unacceptable_python() {
        for (name, extra) in [
            ("uv fails", "echo 'no route' >&2; exit 3"),
            ("absolute link", r#"ln -s /etc/passwd "$dir/bin/escape""#),
            (
                "link out of the tree",
                r#"ln -s ../../../outside "$dir/lib""#,
            ),
            ("named pipe", r#"mkfifo "$dir/pipe""#),
        ] {
            let root = crate::tool_runtime::tests::unlocking_tempdir();
            let bundle = crate::tool_runtime::tests::bundle_with_uv(
                crate::tool_runtime::tests::fixture_uv("1.0.0", extra),
            );
            let result = bundle
                .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
                .await;
            assert!(
                matches!(
                    result,
                    Err(ToolRuntimeError::InitializationCommand { ref stderr, .. })
                        if stderr.contains("python installation failed")
                ),
                "{name}: {result:?}"
            );
            // Nothing is activated and the stage is gone again.
            let names = std::fs::read_dir(root.path())
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            assert_eq!(names, [std::ffi::OsString::from("uv")], "{name}");
        }
    }

    #[tokio::test]
    async fn remote_install_rejects_an_interpreter_that_is_not_the_pinned_one() {
        let root = crate::tool_runtime::tests::unlocking_tempdir();
        let bundle =
            crate::tool_runtime::tests::bundle_with_uv(crate::tool_runtime::tests::fixture_uv(
                "1.0.0",
                r#"printf 'other' > "$dir/bin/python3.13""#,
            ));
        let result = bundle
            .prepare_remote(&LocalTransport, root.path(), || async { Ok(()) })
            .await;
        assert!(
            matches!(
                result,
                Err(ToolRuntimeError::InitializationCommand { ref stderr, .. })
                    if stderr.contains("python installation integrity failure")
            ),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn image_verification_checks_the_python_link_and_version() {
        use crate::tool_distribution::tests::file_artifact;
        let (files, distribution, image) = script_fixture("1.0.0", file_artifact);
        let out = crate::tool_runtime::tests::unlocking_tempdir();
        let target = ToolTarget::native().unwrap();
        let staged = distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        let tools = image.verify(&image_transport(&out)).await.unwrap();
        assert!(tools.catalog().render().unwrap().contains("python3 3.13.1"));
        // An interpreter reporting another version is a version mismatch.
        let link = std::fs::read_link(staged.join("bin/python3")).unwrap();
        let interpreter = staged.join("bin").join(&link);
        crate::tool_python::unlock_tree(&staged.join("bin/.python")).unwrap();
        std::fs::write(&interpreter, b"#!/bin/sh\necho Python 3.12.0\n").unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::VersionMismatch(command)) => assert_eq!(command, "python3"),
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
        // A python3 that is not the link to the pinned tree is named.
        std::fs::remove_file(staged.join("bin/python3")).unwrap();
        std::os::unix::fs::symlink("../elsewhere", staged.join("bin/python3")).unwrap();
        match image.verify(&image_transport(&out)).await {
            Err(ToolRuntimeError::InvalidImageTools { defect, .. }) => {
                assert_eq!(defect, ImageToolDefect::InvalidPython);
            }
            other => panic!("expected InvalidImageTools, got {other:?}"),
        }
        std::fs::remove_file(staged.join("bin/python3")).unwrap();
        assert!(matches!(
            image.verify(&image_transport(&out)).await,
            Err(ToolRuntimeError::InvalidImageTools {
                defect: ImageToolDefect::InvalidPython,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn denied_remote_install_never_activates_tools() {
        let root = crate::tool_runtime::tests::unlocking_tempdir();
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
        let root = crate::tool_runtime::tests::unlocking_tempdir();
        let bundle = crate::tool_runtime::tests::bundle();
        let (first, second) = futures_lite::future::zip(
            bundle.prepare_remote(&LocalTransport, root.path(), || async { Ok(()) }),
            bundle.prepare_remote(&LocalTransport, root.path(), || async { Ok(()) }),
        )
        .await;
        assert_eq!(first.unwrap().bin_dir(), second.unwrap().bin_dir());
    }

    /// Answers the probe with fixed stdout, as an SSH host would.
    struct CannedTransport(&'static str);

    impl ToolTransport for CannedTransport {
        fn run(
            &self,
            _script: &str,
            _input: Option<Vec<u8>>,
        ) -> impl Future<Output = Result<Output, ToolRuntimeError>> + Send {
            use std::os::unix::process::ExitStatusExt;
            std::future::ready(Ok(Output {
                status: std::process::ExitStatus::from_raw(0),
                stdout: self.0.as_bytes().to_vec(),
                stderr: Vec::new(),
            }))
        }
    }

    #[tokio::test]
    async fn probe_parses_a_macos_host() {
        let info = probe_tool_environment(&CannedTransport("Darwin\narm64\n/Users/dev\n"))
            .await
            .unwrap();
        assert_eq!(info.target.os(), "macos");
        assert_eq!(info.target.arch(), "aarch64");
        assert_eq!(
            info.cache_root,
            PathBuf::from("/Users/dev/.cache/aither/tools")
        );
    }

    #[tokio::test]
    async fn clt_check_skips_other_targets() {
        struct Unreachable;
        impl ToolTransport for Unreachable {
            async fn run(
                &self,
                _script: &str,
                _input: Option<Vec<u8>>,
            ) -> Result<Output, ToolRuntimeError> {
                panic!("the check must not run on a non-macOS target");
            }
        }
        let target = ToolTarget::from_probe("Linux", "x86_64").unwrap();
        ensure_command_line_tools(&Unreachable, &target)
            .await
            .unwrap();
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn clt_check_passes_with_the_installed_tools() {
        ensure_command_line_tools(&LocalToolTransport, &ToolTarget::native().unwrap())
            .await
            .unwrap();
    }

    /// Runs the check with `DEVELOPER_DIR` set for that child only.
    #[cfg(target_os = "macos")]
    struct DeveloperDirTransport(PathBuf);

    #[cfg(target_os = "macos")]
    impl ToolTransport for DeveloperDirTransport {
        async fn run(
            &self,
            script: &str,
            input: Option<Vec<u8>>,
        ) -> Result<Output, ToolRuntimeError> {
            let mut command = async_process::Command::new("sh");
            command.args(["-c", script]).env("DEVELOPER_DIR", &self.0);
            run_command(command, input).await
        }
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn clt_check_fails_on_an_empty_developer_dir() {
        let empty = tempfile::tempdir().unwrap();
        let error = ensure_command_line_tools(
            &DeveloperDirTransport(empty.path().to_path_buf()),
            &ToolTarget::native().unwrap(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, ToolRuntimeError::CommandLineToolsMissing));
        assert!(error.to_string().contains("xcode-select --install"));
    }
}
