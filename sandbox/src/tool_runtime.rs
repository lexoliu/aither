use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use askama::Template;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ContainerPath;

/// Commands guaranteed by a prepared execution environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedTool {
    /// Contextual multi-file editing.
    ApplyPatch,
    /// Parallel text and file search.
    Ripgrep,
    /// Syntax-aware code search.
    AstGrep,
    /// JSON selection and transformation.
    Jq,
}

impl ManagedTool {
    /// The complete baseline, in stable prompt order.
    pub const ALL: [Self; 4] = [Self::ApplyPatch, Self::Ripgrep, Self::AstGrep, Self::Jq];

    /// Name used by the model in every execution environment.
    #[must_use]
    pub const fn command(self) -> &'static str {
        match self {
            Self::ApplyPatch => "apply_patch",
            Self::Ripgrep => "rg",
            Self::AstGrep => "ast-grep",
            Self::Jq => "jq",
        }
    }

    fn executable_name(self, target: &ToolTarget) -> String {
        executable_name(self.command(), target)
    }
}

/// Programs the harness runs itself on the host.
///
/// Unlike a [`ManagedTool`], a host program is never placed on a sandbox or
/// container `PATH` and never appears in the [`ToolCatalog`]; only the
/// harness receives its verified path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostProgram {
    /// The `search` daemon of repo-search.
    RepoSearch,
}

impl HostProgram {
    /// Every host program a distribution can provide.
    pub const ALL: [Self; 1] = [Self::RepoSearch];

    /// Name of the program's executable.
    #[must_use]
    pub const fn command(self) -> &'static str {
        match self {
            Self::RepoSearch => "search",
        }
    }

    pub(crate) fn executable_name(self, target: &ToolTarget) -> String {
        executable_name(self.command(), target)
    }
}

/// Either kind of distributed program, for diagnostics shared by both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedTool {
    /// A sandbox command.
    Command(ManagedTool),
    /// A host program.
    Host(HostProgram),
}

impl From<ManagedTool> for DistributedTool {
    fn from(tool: ManagedTool) -> Self {
        Self::Command(tool)
    }
}

impl From<HostProgram> for DistributedTool {
    fn from(program: HostProgram) -> Self {
        Self::Host(program)
    }
}

fn executable_name(command: &str, target: &ToolTarget) -> String {
    if target.os == "windows" {
        format!("{command}.exe")
    } else {
        command.to_owned()
    }
}

/// Operating system and architecture selected during initialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolTarget {
    os: String,
    arch: String,
}

impl ToolTarget {
    /// Resolves the platform of the harness machine.
    ///
    /// # Errors
    /// Returns an error rather than selecting artifacts for an unsupported target.
    pub fn native() -> Result<Self, ToolRuntimeError> {
        Self::from_probe(std::env::consts::OS, std::env::consts::ARCH)
    }

    /// Normalizes an execution backend's platform probe.
    ///
    /// # Errors
    /// Returns an error for an unsupported operating system or architecture.
    pub fn from_probe(os: &str, arch: &str) -> Result<Self, ToolRuntimeError> {
        let normalized_os = match os.trim() {
            "Linux" | "linux" => "linux",
            "Darwin" | "macos" => "macos",
            "Windows_NT" | "windows" => "windows",
            _ => {
                return Err(ToolRuntimeError::UnsupportedTarget {
                    os: os.into(),
                    arch: arch.into(),
                });
            }
        };
        let normalized_arch = match arch.trim() {
            "arm64" | "aarch64" => "aarch64",
            "amd64" | "x86_64" => "x86_64",
            _ => {
                return Err(ToolRuntimeError::UnsupportedTarget {
                    os: os.into(),
                    arch: arch.into(),
                });
            }
        };
        Ok(Self {
            os: normalized_os.into(),
            arch: normalized_arch.into(),
        })
    }

    /// Normalized operating system name.
    #[must_use]
    pub fn os(&self) -> &str {
        &self.os
    }

    /// Normalized machine architecture.
    #[must_use]
    pub fn arch(&self) -> &str {
        &self.arch
    }
}

/// An executable whose bytes match the distribution's trusted digest.
#[derive(Debug, Clone)]
pub struct ToolBinary {
    pub(crate) tool: ManagedTool,
    version: semver::Version,
    pub(crate) sha256: [u8; 32],
    pub(crate) bytes: Arc<[u8]>,
}

impl ToolBinary {
    /// Verifies executable bytes before they can enter a prepared environment.
    ///
    /// # Errors
    /// Rejects an invalid version, oversized artifact or mismatched SHA-256.
    pub fn verify(
        tool: ManagedTool,
        version: &str,
        expected: &[u8; 32],
        bytes: Vec<u8>,
    ) -> Result<Self, ToolRuntimeError> {
        let (version, bytes) = verify_executable(tool.into(), version, expected, bytes)?;
        Ok(Self {
            tool,
            version,
            sha256: *expected,
            bytes,
        })
    }
}

/// Checks the version, size and digest every distributed executable must satisfy.
fn verify_executable(
    tool: DistributedTool,
    version: &str,
    expected: &[u8; 32],
    bytes: Vec<u8>,
) -> Result<(semver::Version, Arc<[u8]>), ToolRuntimeError> {
    let version = semver::Version::parse(version)?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(ToolRuntimeError::ArtifactTooLarge(tool));
    }
    if digest(&bytes) != *expected {
        return Err(ToolRuntimeError::Integrity { tool });
    }
    Ok((version, bytes.into()))
}

/// A verified host program for one platform.
///
/// This type has no way into a [`ToolBundle`], [`PreparedTools`] or
/// [`ToolCatalog`]: a host program cannot reach a sandbox `PATH` or the
/// model's command list.
#[derive(Debug, Clone)]
pub struct HostProgramBinary {
    program: HostProgram,
    target: ToolTarget,
    version: semver::Version,
    sha256: [u8; 32],
    bytes: Arc<[u8]>,
}

impl HostProgramBinary {
    /// Verifies executable bytes against the distribution's trusted digest.
    ///
    /// # Errors
    /// Rejects an invalid version, oversized artifact or mismatched SHA-256.
    pub fn verify(
        program: HostProgram,
        target: ToolTarget,
        version: &str,
        expected: &[u8; 32],
        bytes: Vec<u8>,
    ) -> Result<Self, ToolRuntimeError> {
        let (version, bytes) = verify_executable(program.into(), version, expected, bytes)?;
        Ok(Self {
            program,
            target,
            version,
            sha256: *expected,
            bytes,
        })
    }

    /// The program this executable implements.
    #[must_use]
    pub const fn program(&self) -> HostProgram {
        self.program
    }

    /// The pinned version of the program.
    #[must_use]
    pub const fn version(&self) -> &semver::Version {
        &self.version
    }

    /// Writes the executable to `<root>/<sha256>/<command>`, read-only, or
    /// verifies the copy already there, and returns its path.
    ///
    /// The directory holds nothing else, is not shared with any sandbox
    /// `PATH` directory and is replaced when its contents are invalid.
    ///
    /// # Errors
    /// Fails when this is not the host's platform or on I/O failure.
    pub async fn prepare_local(&self, root: &Path) -> Result<HostProgramPath, ToolRuntimeError> {
        if self.target != ToolTarget::native()? {
            return Err(ToolRuntimeError::TargetMismatch);
        }
        let binary = self.clone();
        let root = root.to_path_buf();
        blocking::unblock(move || binary.prepare_local_sync(&root)).await
    }

    fn prepare_local_sync(&self, root: &Path) -> Result<HostProgramPath, ToolRuntimeError> {
        std::fs::create_dir_all(root)?;
        let directory = std::fs::canonicalize(root)?.join(hex::encode(self.sha256));
        let path = directory.join(self.program.executable_name(&self.target));
        if self.is_prepared(&path)? {
            return Ok(HostProgramPath(path));
        }
        std::fs::create_dir_all(&directory)?;
        let mut stage = tempfile::NamedTempFile::new_in(&directory)?;
        std::io::Write::write_all(&mut stage, &self.bytes)?;
        drop(stage.persist(&path).map_err(|error| error.error)?);
        // After the rename: persisting may reset the file's attributes on
        // Windows, which would clear a read-only bit set on the staged file.
        set_executable_readonly(&path)?;
        if self.is_prepared(&path)? {
            Ok(HostProgramPath(path))
        } else {
            Err(ToolRuntimeError::InvalidCache(path))
        }
    }

    /// Whether `path` is a regular read-only executable with the trusted digest.
    fn is_prepared(&self, path: &Path) -> Result<bool, ToolRuntimeError> {
        let metadata = match std::fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) => {
                return match cache_entry_error(path, error) {
                    ToolRuntimeError::InvalidCache(_) => Ok(false),
                    error => Err(error),
                };
            }
        };
        if !metadata.file_type().is_file() || digest(&std::fs::read(path)?) != self.sha256 {
            return Ok(false);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata.permissions().mode();
            Ok(mode & 0o222 == 0 && mode & 0o111 != 0)
        }
        #[cfg(not(unix))]
        Ok(metadata.permissions().readonly())
    }
}

/// The verified path of a prepared host program, for the harness to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostProgramPath(PathBuf);

impl HostProgramPath {
    /// The read-only executable on the host.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }
}

/// Complete, immutable baseline for one execution platform.
#[derive(Debug, Clone)]
pub struct ToolBundle {
    target: ToolTarget,
    pub(crate) binaries: Arc<[ToolBinary]>,
}

impl ToolBundle {
    /// Constructs a complete baseline from verified binaries.
    ///
    /// # Errors
    /// Rejects missing or duplicate commands; partial bundles cannot become ready.
    pub fn new(
        target: ToolTarget,
        mut binaries: Vec<ToolBinary>,
    ) -> Result<Self, ToolRuntimeError> {
        binaries.sort_by_key(|binary| binary.tool);
        for pair in binaries.windows(2) {
            if pair[0].tool == pair[1].tool {
                return Err(ToolRuntimeError::DuplicateTool(pair[0].tool));
            }
        }
        for tool in ManagedTool::ALL {
            if !binaries.iter().any(|binary| binary.tool == tool) {
                return Err(ToolRuntimeError::MissingTool(tool));
            }
        }
        Ok(Self {
            target,
            binaries: binaries.into(),
        })
    }

    /// The target platform; never inferred from the coordinator for a remote backend.
    #[must_use]
    pub const fn target(&self) -> &ToolTarget {
        &self.target
    }

    /// Stable operational metadata shared by every prepared backend.
    #[must_use]
    pub fn catalog(&self) -> ToolCatalog {
        ToolCatalog {
            tools: self
                .binaries
                .iter()
                .map(|binary| ToolInfo {
                    command: binary.tool.command(),
                    version: binary.version.clone(),
                })
                .collect(),
        }
    }

    #[cfg(test)]
    fn binary(&self, tool: ManagedTool) -> &ToolBinary {
        self.binaries
            .iter()
            .find(|binary| binary.tool == tool)
            .expect("bundle is complete")
    }

    pub(crate) fn cache_key(&self) -> Result<String, ToolRuntimeError> {
        #[derive(Serialize)]
        struct Identity<'a> {
            target: &'a ToolTarget,
            tools: Vec<(&'a str, &'a semver::Version, &'a [u8; 32])>,
        }
        let identity = Identity {
            target: &self.target,
            tools: self
                .binaries
                .iter()
                .map(|binary| (binary.tool.command(), &binary.version, &binary.sha256))
                .collect(),
        };
        Ok(hex::encode(digest(&serde_json::to_vec(&identity)?)))
    }

    /// Prepares or verifies a content-addressed directory outside agent workspaces.
    ///
    /// # Errors
    /// Returns errors for target mismatch, I/O failure or tampered cached content.
    pub async fn prepare_local(
        &self,
        root: &Path,
    ) -> Result<PreparedTools<PathBuf>, ToolRuntimeError> {
        if self.target != ToolTarget::native()? {
            return Err(ToolRuntimeError::TargetMismatch);
        }
        let bundle = self.clone();
        let root = root.to_path_buf();
        blocking::unblock(move || bundle.prepare_local_sync(&root)).await
    }

    #[doc = "Prepares target executables on the coordinator and describes their read-only container mount."]
    #[doc = "# Errors"]
    #[doc = "Rejects invalid cached content, failed writes or invalid bundle identity."]
    #[doc = "# Panics"]
    #[doc = "Panics if the internal prepared directory lacks its required parent."]
    pub async fn prepare_container_mount(
        &self,
        root: &Path,
    ) -> Result<(crate::MountSpec, PreparedTools<PathBuf>), ToolRuntimeError> {
        let bundle = self.clone();
        let root = root.to_path_buf();
        let source = blocking::unblock(move || bundle.prepare_local_sync(&root)).await?;
        let destination = self.container_directory()?;
        let host_directory = source
            .bin_dir
            .parent()
            .expect("prepared binaries have a parent directory")
            .to_path_buf();
        Ok((
            crate::MountSpec::read_only(host_directory, destination),
            source,
        ))
    }

    /// Where the prepared runtime is mounted inside a container.
    pub(crate) fn container_directory(&self) -> Result<ContainerPath, ToolRuntimeError> {
        Ok(ContainerPath::root()
            .join("opt/aither/tools")
            .join(self.cache_key()?))
    }

    /// The executable directory inside a container, which goes on its `PATH`.
    pub(crate) fn container_bin_dir(&self) -> Result<ContainerPath, ToolRuntimeError> {
        Ok(self.container_directory()?.join("bin"))
    }

    /// Returns the verified runtime at `<root>/<cache key>`, staging it when it is
    /// missing and replacing it when its contents are invalid.
    ///
    /// Verifying a valid runtime takes no lock. Creation and replacement run under an
    /// exclusive lock on `<root>/.<cache key>.lock`, so concurrent preparers never
    /// discard a runtime another preparer has just activated.
    fn prepare_local_sync(&self, root: &Path) -> Result<PreparedTools<PathBuf>, ToolRuntimeError> {
        std::fs::create_dir_all(root)?;
        let root = std::fs::canonicalize(root)?;
        let key = self.cache_key()?;
        let destination = root.join(&key);
        if let Some(prepared) = self.reusable(&destination)? {
            return Ok(prepared);
        }
        let lock_path = root.join(format!(".{key}.lock"));
        let mut lock = fd_lock::RwLock::new(
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&lock_path)?,
        );
        let _guard = lock.write()?;
        // A preparer holding the lock before us may have activated a valid runtime.
        if let Some(prepared) = self.reusable(&destination)? {
            return Ok(prepared);
        }
        let stage = self.stage(&root)?;
        let discarded = if destination.try_exists()? {
            let discarded = tempfile::Builder::new()
                .prefix(".invalid-")
                .tempdir_in(&root)?;
            std::fs::rename(&destination, discarded.path().join("runtime"))?;
            Some(discarded)
        } else {
            None
        };
        match std::fs::rename(stage.path(), &destination) {
            Ok(()) => {}
            Err(error) if destination.try_exists()? => {
                tracing::debug!(%error, "another initializer activated the tool bundle");
            }
            Err(error) => return Err(error.into()),
        }
        if let Some(discarded) = discarded {
            discarded.close()?;
        }
        self.verify_prepared(&destination)
    }

    /// Verifies an existing runtime; `None` when it is absent or must be replaced.
    fn reusable(
        &self,
        destination: &Path,
    ) -> Result<Option<PreparedTools<PathBuf>>, ToolRuntimeError> {
        if !destination.try_exists()? {
            return Ok(None);
        }
        match self.verify_prepared(destination) {
            Ok(prepared) => Ok(Some(prepared)),
            Err(
                error @ (ToolRuntimeError::InvalidCache(_) | ToolRuntimeError::Integrity { .. }),
            ) => {
                tracing::warn!(
                    path = %destination.display(),
                    %error,
                    "prepared tool runtime is invalid and will be replaced"
                );
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Writes every executable into a fresh `.initializing-` directory under `root`.
    fn stage(&self, root: &Path) -> Result<tempfile::TempDir, ToolRuntimeError> {
        let stage = tempfile::Builder::new()
            .prefix(".initializing-")
            .tempdir_in(root)?;
        let bin = stage.path().join("bin");
        std::fs::create_dir(&bin)?;
        for binary in self.binaries.iter() {
            let path = bin.join(binary.tool.executable_name(&self.target));
            std::fs::write(&path, &binary.bytes)?;
            set_executable_readonly(&path)?;
        }
        Ok(stage)
    }

    /// Checks that every executable under `destination` is a regular, read-only
    /// executable file with the trusted digest.
    fn verify_prepared(
        &self,
        destination: &Path,
    ) -> Result<PreparedTools<PathBuf>, ToolRuntimeError> {
        let bin_dir = destination.join("bin");
        for binary in self.binaries.iter() {
            let path = bin_dir.join(binary.tool.executable_name(&self.target));
            let metadata = std::fs::symlink_metadata(&path)
                .map_err(|error| cache_entry_error(&path, error))?;
            if !metadata.file_type().is_file() {
                return Err(ToolRuntimeError::InvalidCache(path));
            }
            let bytes = std::fs::read(&path).map_err(|error| cache_entry_error(&path, error))?;
            if digest(&bytes) != binary.sha256 {
                return Err(ToolRuntimeError::Integrity {
                    tool: binary.tool.into(),
                });
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = metadata.permissions().mode();
                if mode & 0o222 != 0 || mode & 0o111 == 0 {
                    return Err(ToolRuntimeError::InvalidCache(path));
                }
            }
            #[cfg(not(unix))]
            if !metadata.permissions().readonly() {
                return Err(ToolRuntimeError::InvalidCache(path));
            }
        }
        Ok(PreparedTools {
            bin_dir,
            catalog: self.catalog(),
        })
    }
}

/// Computes the SHA-256 digest used to validate artifact bytes.
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Names the cache entry when it is missing or its parent is not a directory.
fn cache_entry_error(path: &Path, error: std::io::Error) -> ToolRuntimeError {
    match error.kind() {
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory => {
            ToolRuntimeError::InvalidCache(path.to_path_buf())
        }
        _ => ToolRuntimeError::Io(error),
    }
}

fn set_executable_readonly(path: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o555))
    }
    #[cfg(not(unix))]
    {
        let mut permissions = std::fs::metadata(path)?.permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(path, permissions)
    }
}

/// Tool directory available only after successful initialization.
///
/// `D` is the path type of the environment the tools run in: a host
/// [`PathBuf`] for the local sandbox and SSH, a [`ContainerPath`] inside a
/// container.
#[derive(Debug, Clone)]
pub struct PreparedTools<D> {
    pub(crate) bin_dir: D,
    pub(crate) catalog: ToolCatalog,
}

impl<D> PreparedTools<D> {
    /// Read-only executable directory to expose to the sandbox.
    #[must_use]
    pub const fn bin_dir(&self) -> &D {
        &self.bin_dir
    }

    /// Verified commands to describe to the model before its first request.
    #[must_use]
    pub const fn catalog(&self) -> &ToolCatalog {
        &self.catalog
    }
}

impl PreparedTools<PathBuf> {
    pub(crate) async fn verify_local_versions<E: executor_core::Executor + Clone + 'static>(
        &self,
        executor: E,
    ) -> Result<(), ToolRuntimeError> {
        use futures::{StreamExt, TryStreamExt};
        let scratch = blocking::unblock(tempfile::tempdir).await?;
        let config = heel::SandboxConfig::builder()
            .working_dir(scratch.path())
            .grant(&self.bin_dir, heel::Access::READ | heel::Access::EXEC)
            .build();
        let sandbox = std::sync::Arc::new(
            heel::Sandbox::with_config_and_executor(config, executor)
                .await
                .map_err(|error| ToolRuntimeError::SshBootstrap(error.to_string()))?,
        );
        let directory = self.bin_dir.clone();
        futures::stream::iter(self.catalog.tools.clone())
            .map(|info| {
                let sandbox = sandbox.clone();
                let program = directory
                    .join(info.command)
                    .with_extension(std::env::consts::EXE_EXTENSION);
                async move {
                    let program = program
                        .to_str()
                        .ok_or_else(|| ToolRuntimeError::InvalidCache(program.clone()))?;
                    let command = sandbox.command(program).arg("--version");
                    let output = futures_lite::future::or(
                        async {
                            command
                                .output()
                                .await
                                .map_err(|error| ToolRuntimeError::SshBootstrap(error.to_string()))
                        },
                        async {
                            async_io::Timer::after(std::time::Duration::from_secs(30)).await;
                            Err(ToolRuntimeError::InitializationTimeout)
                        },
                    )
                    .await?;
                    if !output.status.success() {
                        return Err(ToolRuntimeError::VersionMismatch(info.command.into()));
                    }
                    info.verify_version(&String::from_utf8(output.stdout)?)
                }
            })
            .buffer_unordered(ManagedTool::ALL.len())
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }

    #[doc = "Rejects writable host grants that expose the managed executable directory."]
    #[doc = "# Errors"]
    #[doc = "Returns path resolution errors or a writable-directory conflict."]
    pub async fn ensure_readonly<'a>(
        &self,
        paths: impl IntoIterator<Item = &'a PathBuf> + Send,
    ) -> Result<(), ToolRuntimeError> {
        use futures::{StreamExt, TryStreamExt};
        let paths = paths.into_iter().cloned().collect::<Vec<_>>();
        let bin_dir = async_fs::canonicalize(&self.bin_dir).await?;
        futures::stream::iter(paths)
            .map(|path| {
                let bin_dir = &bin_dir;
                async move {
                    let canonical = async_fs::canonicalize(path).await?;
                    if bin_dir.starts_with(&canonical) || canonical.starts_with(bin_dir) {
                        return Err(ToolRuntimeError::WritableToolDirectory(canonical));
                    }
                    Ok(())
                }
            })
            .buffer_unordered(4)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }

    /// Prepends the prepared directory while retaining the backend's own PATH.
    ///
    /// # Errors
    /// Returns an error when a path cannot be represented in the platform's PATH format.
    pub fn prepend_path(&self, existing: Option<&OsStr>) -> Result<OsString, ToolRuntimeError> {
        let existing = existing.map(std::env::split_paths).into_iter().flatten();
        Ok(std::env::join_paths(
            std::iter::once(self.bin_dir.clone()).chain(existing),
        )?)
    }
}

#[derive(Template)]
#[template(path = "tool_path.sh", escape = "none")]
struct ToolPathTemplate<'a> {
    bin_dir: std::borrow::Cow<'a, str>,
    script: &'a str,
}

impl PreparedTools<PathBuf> {
    /// The executable directory as text a shell script can name.
    pub(crate) fn script_bin_dir(&self) -> Result<&str, ToolRuntimeError> {
        self.bin_dir
            .to_str()
            .ok_or_else(|| ToolRuntimeError::InvalidCache(self.bin_dir.clone()))
    }

    pub(crate) fn wrap_script(&self, script: &str) -> Result<String, ToolRuntimeError> {
        render_tool_path(self.script_bin_dir()?, script)
    }
}

impl PreparedTools<ContainerPath> {
    pub(crate) fn wrap_script(&self, script: &str) -> Result<String, ToolRuntimeError> {
        render_tool_path(self.bin_dir.as_str(), script)
    }
}

/// Prepends `bin_dir` to a POSIX shell script's `PATH`.
fn render_tool_path(bin_dir: &str, script: &str) -> Result<String, ToolRuntimeError> {
    Ok(ToolPathTemplate {
        bin_dir: shell_words::quote(bin_dir),
        script,
    }
    .render()?)
}

/// Prepared command versions, in stable order independent of cache paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCatalog {
    tools: Vec<ToolInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolInfo {
    command: &'static str,
    version: semver::Version,
}

impl ToolCatalog {
    pub(crate) fn verify_versions(&self, output: &str) -> Result<(), ToolRuntimeError> {
        let reported = output.split_terminator('\0').collect::<Vec<_>>();
        if reported.len() != self.tools.len() {
            return Err(ToolRuntimeError::InvalidProbe);
        }
        for (info, output) in self.tools.iter().zip(reported) {
            info.verify_version(output)?;
        }
        Ok(())
    }
}

impl ToolInfo {
    fn verify_version(&self, output: &str) -> Result<(), ToolRuntimeError> {
        let reported = output
            .lines()
            .next()
            .and_then(|line| {
                line.strip_prefix("jq-")
                    .or_else(|| line.split_whitespace().nth(1))
            })
            .and_then(|version| semver::Version::parse(version).ok());
        if reported.as_ref() != Some(&self.version) {
            return Err(ToolRuntimeError::VersionMismatch(self.command.into()));
        }
        Ok(())
    }
}

#[derive(Template)]
#[template(path = "prepared_tools.txt", escape = "none")]
struct ToolCatalogTemplate<'a> {
    tools: &'a [ToolInfo],
}

impl ToolCatalog {
    /// Renders concise operational instructions without a command-discovery phase.
    ///
    /// # Errors
    /// Returns a template rendering error.
    pub fn render(&self) -> Result<String, ToolRuntimeError> {
        Ok(ToolCatalogTemplate { tools: &self.tools }.render()?)
    }
}

/// A part of a registry's configuration that preparation is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedEnvironment {
    /// The backend that runs commands on this machine or in its container.
    HomeBackend,
    /// The container the home tools were mounted into.
    Container,
    /// The set of SSH servers and their targets.
    SshServers,
}

impl std::fmt::Display for PreparedEnvironment {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::HomeBackend => "the home execution backend",
            Self::Container => "the container runtime",
            Self::SshServers => "the SSH server set",
        })
    }
}

/// Fail-closed errors from tool initialization and shell integration.
#[derive(Debug, thiserror::Error)]
pub enum ToolRuntimeError {
    /// Configured container has no execution transport.
    #[error("container runtime is not configured")]
    MissingContainerRuntime,
    /// Environments resolved different command versions.
    #[error("prepared tool catalogs differ between environments")]
    CatalogMismatch,
    /// SSH runtime bootstrap failed.
    #[error("SSH runtime initialization failed: {0}")]
    SshBootstrap(String),
    /// Installation was not authorized.
    #[error("managed tool installation denied for {0}")]
    InstallDenied(String),
    /// Application authorizer failed.
    #[error("tool installation authorization failed: {0}")]
    Authorization(crate::SshAuthorizationError),
    /// Initialization transport was cancelled.
    #[error("tool initialization interrupted")]
    InitializationInterrupted,
    /// Initialization exceeded its bounded deadline.
    #[error("tool initialization timed out")]
    InitializationTimeout,
    /// Backend platform/path probe was malformed.
    #[error("invalid tool environment probe output")]
    InvalidProbe,
    /// Trusted initialization command failed.
    #[error("tool initialization command failed ({status:?}): {stderr}")]
    InitializationCommand {
        /// Process exit status, if available.
        status: Option<i32>,
        /// Backend diagnostic, without artifact contents.
        stderr: String,
    },
    /// Container initialization transport failed.
    #[error("container tool initialization failed: {0}")]
    Container(String),
    /// Backend output is not valid UTF-8.
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    /// Initialization has not prepared the requested execution environment.
    #[error("tool runtime is not initialized for {0}")]
    NotInitialized(String),
    /// The registry's configuration changed after its environments were prepared.
    #[error("{0} changed after tool preparation")]
    EnvironmentChanged(PreparedEnvironment),
    /// Agent write permissions would expose managed executable contents.
    #[error("managed tool directory overlaps a writable sandbox path: {0}")]
    WritableToolDirectory(PathBuf),
    /// No artifact set supports the probed platform.
    #[error("unsupported tool target: {os}/{arch}")]
    UnsupportedTarget {
        /// Probed operating system.
        os: String,
        /// Probed architecture.
        arch: String,
    },
    /// The executable cannot run or does not report the pinned version.
    #[error("prepared tool failed its version probe: {0}")]
    VersionMismatch(String),
    #[doc = "Container mount metadata could not be parsed."]
    #[error(transparent)]
    MountInfo(#[from] procfs_core::ProcError),
    #[doc = "The tools directory is not protected by a read-only container mount."]
    #[error("managed tools require a read-only container mount: {0}")]
    MissingReadonlyMount(ContainerPath),
    /// Required command missing from a bundle.
    #[error("required tool is missing: {0:?}")]
    MissingTool(ManagedTool),
    /// A bundle declares the same command twice.
    #[error("duplicate managed tool: {0:?}")]
    DuplicateTool(ManagedTool),
    /// The distribution has no artifact of a host program for the target.
    #[error("host program `{}` has no artifact for {os}/{arch}", .program.command())]
    MissingHostProgram {
        /// The requested program.
        program: HostProgram,
        /// Operating system of the requested target.
        os: String,
        /// Architecture of the requested target.
        arch: String,
    },
    /// The distribution has several artifacts of a host program for the target.
    #[error("host program `{}` has several artifacts for {os}/{arch}", .program.command())]
    DuplicateHostProgram {
        /// The requested program.
        program: HostProgram,
        /// Operating system of the requested target.
        os: String,
        /// Architecture of the requested target.
        arch: String,
    },
    /// Bytes do not match the trusted artifact manifest.
    #[error("tool artifact failed SHA-256 verification: {tool:?}")]
    Integrity {
        /// Program whose artifact failed verification.
        tool: DistributedTool,
    },
    /// An artifact exceeds the executable-size limit.
    #[error("tool artifact exceeds the size limit: {0:?}")]
    ArtifactTooLarge(DistributedTool),
    /// A cached command is not a regular read-only executable.
    #[error("invalid prepared tool cache entry: {0}")]
    InvalidCache(PathBuf),
    /// Host installation cannot use another platform's binaries.
    #[error("tool artifact platform does not match the execution environment")]
    TargetMismatch,
    /// Filesystem preparation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Manifest version is not semantic versioning.
    #[error(transparent)]
    Version(#[from] semver::Error),
    /// PATH could not be assembled.
    #[error(transparent)]
    Path(#[from] std::env::JoinPathsError),
    /// Manifest encoding failed.
    #[error(transparent)]
    Manifest(#[from] serde_json::Error),
    /// Artifact source is neither a local file nor credential-free HTTPS.
    #[error("{0:?} requires a local file or credential-free HTTPS artifact source")]
    ArtifactSource(DistributedTool),
    /// Distribution returned a non-success HTTP status.
    #[error("tool artifact download failed with HTTP status {0}")]
    DownloadStatus(u16),
    /// Artifact transfer failed.
    #[error(transparent)]
    Download(#[from] zenwave::Error),
    /// Artifact checksum is not a SHA-256 hexadecimal value.
    #[error(transparent)]
    Digest(#[from] hex::FromHexError),
    /// Executable archive entry is missing, ambiguous, oversized or not regular.
    #[error("invalid executable archive member")]
    InvalidArchive,
    /// ZIP decoding failed.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
    /// Operational prompt rendering failed.
    #[error(transparent)]
    Template(#[from] askama::Error),
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn artifact_digest_matches_known_sha256_vector() {
        assert_eq!(
            hex::encode(digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    pub fn bundle() -> ToolBundle {
        bundle_pinned("1.0.0")
    }

    /// A bundle pinned to `version` whose commands all report version 1.0.0,
    /// so any other pin fails the version probe while passing integrity checks.
    pub fn bundle_pinned(version: &str) -> ToolBundle {
        let binaries = ManagedTool::ALL
            .into_iter()
            .map(|tool| {
                let bytes =
                    format!("#!/bin/sh\nprintf '%s\\n' '{} 1.0.0'\n", tool.command()).into_bytes();
                ToolBinary::verify(tool, version, &digest(&bytes), bytes).unwrap()
            })
            .collect();
        ToolBundle::new(ToolTarget::native().unwrap(), binaries).unwrap()
    }

    /// Tools as a container sees them once the bundle is mounted read-only.
    pub fn container_tools(bundle: &ToolBundle) -> PreparedTools<ContainerPath> {
        PreparedTools {
            bin_dir: bundle.container_bin_dir().unwrap(),
            catalog: bundle.catalog(),
        }
    }

    // Runs on every host: Windows must still produce POSIX container paths.
    #[test]
    fn container_tool_paths_are_posix_on_every_host() {
        let bundle = bundle();
        let key = bundle.cache_key().unwrap();
        assert_eq!(
            bundle.container_directory().unwrap().as_str(),
            format!("/opt/aither/tools/{key}")
        );
        assert_eq!(
            bundle.container_bin_dir().unwrap().as_str(),
            format!("/opt/aither/tools/{key}/bin")
        );
        let script = container_tools(&bundle)
            .wrap_script("rg --version")
            .unwrap();
        assert!(
            script.starts_with(&format!("PATH=/opt/aither/tools/{key}/bin")),
            "{script}"
        );
    }

    #[tokio::test]
    async fn container_mount_targets_the_posix_tool_directory() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let (mount, host) = bundle.prepare_container_mount(cache.path()).await.unwrap();
        assert_eq!(
            mount.container_path.as_str(),
            format!("/opt/aither/tools/{}", bundle.cache_key().unwrap())
        );
        assert_eq!(Some(mount.host_path.as_path()), host.bin_dir().parent());
        assert!(mount.access.read_only());
    }

    #[test]
    fn missing_baseline_tool_is_an_initialization_error() {
        let mut binaries = bundle().binaries.to_vec();
        binaries.pop();
        assert!(matches!(
            ToolBundle::new(ToolTarget::native().unwrap(), binaries),
            Err(ToolRuntimeError::MissingTool(ManagedTool::Jq))
        ));
    }

    #[test]
    fn incorrect_digest_is_rejected_before_installation() {
        assert!(matches!(
            ToolBinary::verify(ManagedTool::Ripgrep, "1.0.0", &[0; 32], b"corrupt".to_vec()),
            Err(ToolRuntimeError::Integrity {
                tool: DistributedTool::Command(ManagedTool::Ripgrep)
            })
        ));
    }

    #[tokio::test]
    async fn initialize_outside_workspace_and_reuse_verified_content() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let first = bundle.prepare_local(cache.path()).await.unwrap();
        let second = bundle.prepare_local(cache.path()).await.unwrap();
        assert_eq!(first.bin_dir(), second.bin_dir());
        assert_eq!(first.catalog(), second.catalog());
        for tool in ManagedTool::ALL {
            let path = first.bin_dir().join(tool.executable_name(&bundle.target));
            assert_eq!(
                async_fs::read(path).await.unwrap().as_slice(),
                bundle.binary(tool).bytes.as_ref()
            );
        }
        let existing = [
            cache.path().join("system tools"),
            cache.path().join("project tools"),
        ];
        let original_path = std::env::join_paths(&existing).unwrap();
        let path = first.prepend_path(Some(&original_path)).unwrap();
        let entries = std::env::split_paths(&path).collect::<Vec<_>>();
        assert_eq!(entries[0], *first.bin_dir());
        assert_eq!(entries[1..], existing);
    }

    #[tokio::test]
    async fn concurrent_initialization_never_exposes_partial_tools() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let (left, right) = futures_lite::future::zip(
            bundle.prepare_local(cache.path()),
            bundle.prepare_local(cache.path()),
        )
        .await;
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.bin_dir(), right.bin_dir());
        for tool in ManagedTool::ALL {
            assert!(
                left.bin_dir()
                    .join(tool.executable_name(&bundle.target))
                    .is_file()
            );
        }
    }

    #[tokio::test]
    async fn managed_tools_cannot_overlap_a_writable_grant() {
        let cache = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let ready = bundle().prepare_local(cache.path()).await.unwrap();
        ready
            .ensure_readonly([&workspace.path().to_path_buf()])
            .await
            .unwrap();
        let result = ready.ensure_readonly([&cache.path().to_path_buf()]).await;
        assert!(matches!(
            result,
            Err(ToolRuntimeError::WritableToolDirectory(_))
        ));
    }

    #[tokio::test]
    async fn writable_grant_to_one_managed_executable_is_rejected() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let executable = ready
            .bin_dir()
            .join(ManagedTool::Ripgrep.executable_name(&bundle.target));
        assert!(matches!(
            ready.ensure_readonly([&executable]).await,
            Err(ToolRuntimeError::WritableToolDirectory(_))
        ));
    }

    fn assert_prepared(bundle: &ToolBundle, prepared: &PreparedTools<PathBuf>) {
        for tool in ManagedTool::ALL {
            let path = prepared
                .bin_dir()
                .join(tool.executable_name(&bundle.target));
            assert_eq!(
                std::fs::read(&path).unwrap().as_slice(),
                bundle.binary(tool).bytes.as_ref(),
                "{} differs from the bundle",
                path.display()
            );
        }
        bundle
            .verify_prepared(prepared.bin_dir().parent().unwrap())
            .unwrap();
    }

    /// Entries left in the cache root other than the runtime and its lock file.
    fn leftovers(root: &Path, bundle: &ToolBundle) -> Vec<OsString> {
        let key = bundle.cache_key().unwrap();
        let lock = format!(".{key}.lock");
        std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| *name != *key && *name != *lock)
            .collect()
    }

    fn make_writable(path: &Path) {
        let mut permissions = std::fs::metadata(path).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(0o755);
        }
        #[cfg(not(unix))]
        permissions.set_readonly(false);
        std::fs::set_permissions(path, permissions).unwrap();
    }

    #[test]
    fn missing_executable_is_reported_with_its_path() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let destination = cache.path().join(bundle.cache_key().unwrap());
        std::fs::create_dir_all(destination.join("bin")).unwrap();
        let first = destination
            .join("bin")
            .join(bundle.binaries[0].tool.executable_name(&bundle.target));
        assert!(matches!(
            bundle.verify_prepared(&destination),
            Err(ToolRuntimeError::InvalidCache(path)) if path == first
        ));
    }

    #[tokio::test]
    async fn empty_runtime_skeleton_is_replaced() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        std::fs::create_dir_all(cache.path().join(bundle.cache_key().unwrap()).join("bin"))
            .unwrap();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        assert_prepared(&bundle, &ready);
        assert_eq!(leftovers(cache.path(), &bundle), Vec::<OsString>::new());
    }

    #[tokio::test]
    async fn tampered_executable_is_replaced() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let path = ready
            .bin_dir()
            .join(ManagedTool::Ripgrep.executable_name(&bundle.target));
        make_writable(&path);
        std::fs::write(&path, b"tampered").unwrap();
        assert!(matches!(
            bundle.verify_prepared(ready.bin_dir().parent().unwrap()),
            Err(ToolRuntimeError::Integrity {
                tool: DistributedTool::Command(ManagedTool::Ripgrep)
            })
        ));
        let repaired = bundle.prepare_local(cache.path()).await.unwrap();
        assert_eq!(repaired.bin_dir(), ready.bin_dir());
        assert_prepared(&bundle, &repaired);
        assert_eq!(leftovers(cache.path(), &bundle), Vec::<OsString>::new());
    }

    #[tokio::test]
    async fn writable_executable_is_replaced() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let path = ready
            .bin_dir()
            .join(ManagedTool::Jq.executable_name(&bundle.target));
        make_writable(&path);
        assert!(matches!(
            bundle.verify_prepared(ready.bin_dir().parent().unwrap()),
            Err(ToolRuntimeError::InvalidCache(invalid)) if invalid == path
        ));
        let repaired = bundle.prepare_local(cache.path()).await.unwrap();
        assert_prepared(&bundle, &repaired);
    }

    #[tokio::test]
    async fn valid_runtime_is_reused_untouched() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let snapshot = |prepared: &PreparedTools<PathBuf>| {
            let mut entries = vec![prepared.bin_dir().parent().unwrap().to_path_buf()];
            entries.push(prepared.bin_dir().clone());
            entries.extend(ManagedTool::ALL.into_iter().map(|tool| {
                prepared
                    .bin_dir()
                    .join(tool.executable_name(&bundle.target))
            }));
            entries
                .into_iter()
                .map(|path| {
                    let metadata = std::fs::symlink_metadata(&path).unwrap();
                    #[cfg(unix)]
                    let inode = std::os::unix::fs::MetadataExt::ino(&metadata);
                    #[cfg(not(unix))]
                    let inode = 0;
                    (path, inode, metadata.modified().unwrap())
                })
                .collect::<Vec<_>>()
        };
        let before = snapshot(&ready);
        let reused = bundle.prepare_local(cache.path()).await.unwrap();
        assert_eq!(before, snapshot(&reused));
    }

    #[tokio::test]
    async fn concurrent_preparers_repair_one_damaged_runtime() {
        let cache = tempfile::tempdir().unwrap();
        let bundle = bundle();
        std::fs::create_dir_all(cache.path().join(bundle.cache_key().unwrap()).join("bin"))
            .unwrap();
        let results =
            futures::future::join_all((0..8).map(|_| bundle.prepare_local(cache.path()))).await;
        for result in results {
            assert_prepared(&bundle, &result.unwrap());
        }
        assert_eq!(leftovers(cache.path(), &bundle), Vec::<OsString>::new());
    }

    #[test]
    fn prompt_catalog_is_operational_and_stable() {
        let catalog = bundle().catalog();
        let first = catalog.render().unwrap();
        assert_eq!(first, catalog.render().unwrap());
        assert!(first.contains("ast-grep run"));
        assert!(first.contains("rg -n"));
        assert!(first.contains("*** Begin Patch"));
        assert!(first.contains("wait"));
        assert!(!first.contains("command -v"));
        assert!(!first.contains("which "));
    }

    struct RuntimeProbe;

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct RuntimeProbeArgs {}

    impl heel::IpcCommand for RuntimeProbe {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "runtime_probe".into()
        }
        type Args = RuntimeProbeArgs;
        type Response = String;
        fn handle(
            &self,
            _args: RuntimeProbeArgs,
        ) -> impl Future<Output = Result<String, heel::IpcError>> {
            std::future::ready(Ok("runtime_probe_ok".into()))
        }
    }

    #[tokio::test]
    async fn prepared_path_retains_heel_ipc_commands() {
        let cache = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let ready = bundle().prepare_local(cache.path()).await.unwrap();
        let sandbox_path = heel::default_sandbox_path();
        let path = ready.prepend_path(Some(OsStr::new(&sandbox_path))).unwrap();
        let config = heel::SandboxConfig::builder()
            .network(heel::DenyAll)
            .working_dir(workspace.path())
            .grants([heel::Grant::new(ready.bin_dir(), heel::Access::EXEC)])
            .ipc(heel::IpcRouter::new().register(RuntimeProbe))
            .security(heel::SecurityConfig::interactive())
            .build();
        let sandbox =
            heel::Sandbox::with_config_and_executor(config, executor_core::tokio::TokioGlobal)
                .await
                .unwrap();
        #[cfg(not(windows))]
        let command = sandbox.command("sh").args(["-c", "runtime_probe"]);
        #[cfg(windows)]
        let command = sandbox
            .command("cmd.exe")
            .args(["/D", "/S", "/C", "runtime_probe"]);
        let output = command
            .env("PATH", path.to_str().unwrap())
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "prepared PATH removed IPC commands: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("runtime_probe_ok"));
    }

    #[test]
    fn unknown_remote_platform_is_not_a_native_fallback() {
        assert!(matches!(
            ToolTarget::from_probe("FreeBSD", "riscv64"),
            Err(ToolRuntimeError::UnsupportedTarget { .. })
        ));
    }
}
