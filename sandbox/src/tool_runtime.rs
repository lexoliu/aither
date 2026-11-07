use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use askama::Template;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ContainerPath;
use crate::tool_python::{PYTHON_COMMAND, unlock_tree};
pub use crate::tool_python::{PythonPin, PythonRuntime};

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
    /// GNU bash, the shell every terminal script runs in on Unix.
    Bash,
    /// The uutils coreutils multicall binary, reached through [`COREUTILS_ALIASES`].
    Coreutils,
    /// `xargs` from uutils findutils.
    Xargs,
    /// Fast file finder.
    Fd,
    /// `aither-sed`, the `sed` of terminal scripts, which only prints line
    /// ranges. Its manifest id is its executable's name, as
    /// `patch/built-tools.json` names it.
    #[serde(rename = "aither-sed")]
    Sed,
    /// The `uv` Python package and project manager, which installs the
    /// managed `python3` ([`PythonRuntime`]) when the tools are prepared.
    Uv,
}

/// The coreutils utilities exposed as relative symlinks `bin/<name> -> coreutils`,
/// through [`ManagedTool::aliases`].
///
/// Excludes the `[` and `more` entries of `coreutils --list` (not utilities
/// of GNU coreutils) and every name the managed set replaces with a modern
/// tool (`grep`, `sed`, `awk`, `find`), none of which coreutils provides.
/// `stdbuf` is left out too: the musl builds omit it (it needs a cdylib), and
/// the set is the same on every platform. Staging checks each name against the `coreutils --list` of the pinned
/// binary, so a name the binary does not provide is never shadowed.
pub const COREUTILS_ALIASES: &[&str] = &[
    "arch",
    "b2sum",
    "base32",
    "base64",
    "basename",
    "basenc",
    "cat",
    "chgrp",
    "chmod",
    "chown",
    "chroot",
    "cksum",
    "comm",
    "cp",
    "csplit",
    "cut",
    "date",
    "dd",
    "df",
    "dir",
    "dircolors",
    "dirname",
    "du",
    "echo",
    "env",
    "expand",
    "expr",
    "factor",
    "false",
    "fmt",
    "fold",
    "groups",
    "head",
    "hostid",
    "hostname",
    "id",
    "install",
    "join",
    "kill",
    "link",
    "ln",
    "logname",
    "ls",
    "md5sum",
    "mkdir",
    "mkfifo",
    "mknod",
    "mktemp",
    "mv",
    "nice",
    "nl",
    "nohup",
    "nproc",
    "numfmt",
    "od",
    "paste",
    "pathchk",
    "pinky",
    "pr",
    "printenv",
    "printf",
    "ptx",
    "pwd",
    "readlink",
    "realpath",
    "rm",
    "rmdir",
    "seq",
    "sha1sum",
    "sha224sum",
    "sha256sum",
    "sha384sum",
    "sha512sum",
    "shred",
    "shuf",
    "sleep",
    "sort",
    "split",
    "stat",
    "stty",
    "sum",
    "sync",
    "tac",
    "tail",
    "tee",
    "test",
    "timeout",
    "touch",
    "tr",
    "true",
    "truncate",
    "tsort",
    "tty",
    "uname",
    "unexpand",
    "uniq",
    "unlink",
    "uptime",
    "users",
    "vdir",
    "wc",
    "who",
    "whoami",
    "yes",
];

impl ManagedTool {
    /// Every managed tool, in stable prompt order. The tools a platform must
    /// provide are [`ToolTarget::required_tools`].
    pub const ALL: [Self; 10] = [
        Self::ApplyPatch,
        Self::Ripgrep,
        Self::AstGrep,
        Self::Jq,
        Self::Bash,
        Self::Coreutils,
        Self::Xargs,
        Self::Fd,
        Self::Sed,
        Self::Uv,
    ];

    /// The tools with an artifact on every platform, Windows included.
    const PORTABLE: [Self; 4] = [Self::ApplyPatch, Self::Ripgrep, Self::AstGrep, Self::Jq];

    /// Name used by the model in every execution environment.
    #[must_use]
    pub const fn command(self) -> &'static str {
        match self {
            Self::ApplyPatch => "apply_patch",
            Self::Ripgrep => "rg",
            Self::AstGrep => "ast-grep",
            Self::Jq => "jq",
            Self::Bash => "bash",
            Self::Coreutils => "coreutils",
            Self::Xargs => "xargs",
            Self::Fd => "fd",
            Self::Sed => "aither-sed",
            Self::Uv => "uv",
        }
    }

    /// Commands exposed as relative symlinks `bin/<alias> -> <command>` to
    /// this tool's executable; empty for every tool but `coreutils`.
    #[must_use]
    pub const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Coreutils => COREUTILS_ALIASES,
            _ => &[],
        }
    }

    pub(crate) fn executable_name(self, target: &ToolTarget) -> String {
        executable_name(self.command(), target)
    }

    /// The version this tool's `--version` output reports.
    ///
    /// The banners differ: `jq-1.8.2`, `coreutils 0.12.0 (multi-call binary)`,
    /// `fd 10.5.0`, `xargs 0.11.0`, `uv 0.13.7 (316c827e0 2026-11-03 aarch64-apple-darwin)`,
    /// `GNU bash, version 5.3.20(1)-release (...)`;
    /// the others report `<name> <version>`.
    fn reported_version(self, output: &str) -> Option<semver::Version> {
        let line = output.lines().next()?;
        let second_word = || line.split_whitespace().nth(1);
        let version = match self {
            // The baseline test fixtures report `jq <version>`.
            Self::Jq => line.strip_prefix("jq-").or_else(second_word)?,
            Self::Bash => line.strip_prefix("GNU bash, version ")?.split_once('(')?.0,
            Self::Coreutils
            | Self::Xargs
            | Self::Fd
            | Self::Sed
            | Self::Uv
            | Self::ApplyPatch
            | Self::Ripgrep
            | Self::AstGrep => second_word()?,
        };
        semver::Version::parse(version).ok()
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

    /// The tools every environment of this platform must provide: Windows
    /// has no `bash`, `coreutils`, `xargs`, `fd`, `aither-sed` or `uv` artifacts, every other
    /// platform the complete set.
    #[must_use]
    pub fn required_tools(&self) -> &'static [ManagedTool] {
        if self.os == "windows" {
            &ManagedTool::PORTABLE
        } else {
            &ManagedTool::ALL
        }
    }

    /// Whether the environment of this platform carries the managed
    /// `python3`, which exactly the platforms that provide `uv` do.
    #[must_use]
    pub fn provides_python(&self) -> bool {
        self.required_tools().contains(&ManagedTool::Uv)
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

impl HostProgramPath {
    /// The prepared executable at `<root>/<sha256>/<command>`, when it is a
    /// regular read-only executable with the trusted digest.
    pub(crate) fn cached(
        root: &Path,
        program: HostProgram,
        target: &ToolTarget,
        sha256: &[u8; 32],
    ) -> Result<Option<Self>, ToolRuntimeError> {
        let path = Self::location(root, program, target, sha256)?;
        Ok(Self::is_prepared(&path, sha256)?.then_some(Self(path)))
    }

    /// Writes `bytes` to `<root>/<sha256>/<command>`, read-only, replacing
    /// whatever is there. The directory holds nothing else and is not shared
    /// with any sandbox `PATH` directory.
    pub(crate) fn write(
        root: &Path,
        program: HostProgram,
        target: &ToolTarget,
        sha256: &[u8; 32],
        bytes: &[u8],
    ) -> Result<Self, ToolRuntimeError> {
        let path = Self::location(root, program, target, sha256)?;
        let directory = path.parent().ok_or(ToolRuntimeError::InvalidArchive)?;
        std::fs::create_dir_all(directory)?;
        let mut stage = tempfile::NamedTempFile::new_in(directory)?;
        std::io::Write::write_all(&mut stage, bytes)?;
        drop(stage.persist(&path).map_err(|error| error.error)?);
        // After the rename: persisting may reset the file's attributes on
        // Windows, which would clear a read-only bit set on the staged file.
        set_executable_readonly(&path)?;
        if Self::is_prepared(&path, sha256)? {
            Ok(Self(path))
        } else {
            Err(ToolRuntimeError::InvalidCache(path))
        }
    }

    fn location(
        root: &Path,
        program: HostProgram,
        target: &ToolTarget,
        sha256: &[u8; 32],
    ) -> Result<PathBuf, ToolRuntimeError> {
        std::fs::create_dir_all(root)?;
        Ok(std::fs::canonicalize(root)?
            .join(hex::encode(sha256))
            .join(program.executable_name(target)))
    }

    /// Whether `path` is a regular read-only executable with the trusted digest.
    fn is_prepared(path: &Path, sha256: &[u8; 32]) -> Result<bool, ToolRuntimeError> {
        let metadata = match std::fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) => {
                return match cache_entry_error(path, error) {
                    ToolRuntimeError::InvalidCache(_) => Ok(false),
                    error => Err(error),
                };
            }
        };
        if !metadata.file_type().is_file() || digest(&std::fs::read(path)?) != *sha256 {
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

/// Complete, immutable baseline for one execution platform.
#[derive(Debug, Clone)]
pub struct ToolBundle {
    target: ToolTarget,
    pub(crate) binaries: Arc<[ToolBinary]>,
    pub(crate) python: Option<PythonRuntime>,
}

impl ToolBundle {
    /// Constructs a complete baseline from verified binaries and, on a
    /// platform that provides one, the interpreter its `uv` installs.
    ///
    /// # Errors
    /// Rejects missing or duplicate commands, and a missing or unexpected
    /// interpreter pin; partial bundles cannot become ready.
    pub fn new(
        target: ToolTarget,
        mut binaries: Vec<ToolBinary>,
        python: Option<PythonRuntime>,
    ) -> Result<Self, ToolRuntimeError> {
        binaries.sort_by_key(|binary| binary.tool);
        for pair in binaries.windows(2) {
            if pair[0].tool == pair[1].tool {
                return Err(ToolRuntimeError::DuplicateTool(pair[0].tool));
            }
        }
        let required = target.required_tools();
        if let Some(binary) = binaries
            .iter()
            .find(|binary| !required.contains(&binary.tool))
        {
            return Err(ToolRuntimeError::UnexpectedTool(binary.tool));
        }
        for &tool in required {
            if !binaries.iter().any(|binary| binary.tool == tool) {
                return Err(ToolRuntimeError::MissingTool(tool));
            }
        }
        check_python(&target, python.as_ref())?;
        Ok(Self {
            target,
            binaries: binaries.into(),
            python,
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
                .map(|binary| ToolInfo::tool(binary.tool, binary.version.clone()))
                .chain(self.python.as_ref().map(ToolInfo::python))
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
            python: Option<(&'a semver::Version, &'a [u8; 32])>,
        }
        let identity = Identity {
            target: &self.target,
            tools: self
                .binaries
                .iter()
                .map(|binary| (binary.tool.command(), &binary.version, &binary.sha256))
                .collect(),
            python: self
                .python
                .as_ref()
                .map(|python| (python.version(), python.interpreter_sha256())),
        };
        Ok(hex::encode(digest(&serde_json::to_vec(&identity)?)))
    }

    /// Prepares or verifies a content-addressed directory outside agent workspaces.
    ///
    /// Beside the content-addressed runtimes, `<root>/uv` is created as the
    /// shared writable directory of `uv`'s cache and project Python
    /// installs; it is separate from the read-only managed interpreter.
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
        blocking::unblock(move || {
            let key = bundle.cache_key()?;
            let mut prepared = bundle.prepare_local_sync(&root, &key, None)?;
            if bundle.python.is_some() {
                let uv_dir = std::fs::canonicalize(&root)?.join(UV_STATE_DIRECTORY);
                std::fs::create_dir_all(&uv_dir)?;
                set_private(&uv_dir)?;
                prepared.uv_dir = Some(uv_dir);
            }
            Ok(prepared)
        })
        .await
    }

    /// Returns the verified runtime at `<root>/<key>`, staging it when it is
    /// missing and replacing it when its contents are invalid.
    ///
    /// Verifying a valid runtime takes no lock. Creation and replacement run under an
    /// exclusive lock on `<root>/.<key>.lock`, so concurrent preparers never
    /// discard a runtime another preparer has just activated.
    ///
    /// `installer` is the `uv` that installs the interpreter when this bundle's
    /// own cannot run on the building machine (an image for another platform).
    pub(crate) fn prepare_local_sync(
        &self,
        root: &Path,
        key: &str,
        installer: Option<&[u8]>,
    ) -> Result<PreparedTools<PathBuf>, ToolRuntimeError> {
        std::fs::create_dir_all(root)?;
        let root = std::fs::canonicalize(root)?;
        let destination = root.join(key);
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
        let stage = self.stage_scratch(&root, installer)?;
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
                self.unlock_python(stage.path())?;
            }
            Err(error) => {
                self.unlock_python(stage.path())?;
                return Err(error.into());
            }
        }
        if let Some(discarded) = discarded {
            self.unlock_python(&discarded.path().join("runtime"))?;
            discarded.close()?;
        }
        self.verify_prepared(&destination)
    }

    /// Opens up the read-only managed interpreter tree under `runtime` so the
    /// runtime can be deleted. A bundle without Python has no such tree, and
    /// its platform may have no permission modes to change.
    fn unlock_python(&self, runtime: &Path) -> Result<(), ToolRuntimeError> {
        if self.python.is_some() {
            unlock_tree(runtime)?;
        }
        Ok(())
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
                error @ (ToolRuntimeError::InvalidCache(_)
                | ToolRuntimeError::Integrity { .. }
                | ToolRuntimeError::PythonIntegrity),
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
    ///
    /// The managed interpreter is installed last, and the finished tree is
    /// read-only.
    fn stage_scratch(
        &self,
        root: &Path,
        installer: Option<&[u8]>,
    ) -> Result<tempfile::TempDir, ToolRuntimeError> {
        let stage = tempfile::Builder::new()
            .prefix(".initializing-")
            .tempdir_in(root)?;
        let bin = stage.path().join("bin");
        std::fs::create_dir(&bin)?;
        for binary in self.binaries.iter() {
            let path = bin.join(binary.tool.executable_name(&self.target));
            std::fs::write(&path, &binary.bytes)?;
            set_executable_readonly(&path)?;
            if !binary.tool.aliases().is_empty() {
                // A foreign target's executable cannot run here; an image
                // build for one stages the aliases unchecked.
                if self.target == ToolTarget::native()? {
                    ensure_aliases_provided(binary.tool, &path)?;
                }
                for alias in binary.tool.aliases() {
                    link_alias(binary.tool, &bin.join(alias))?;
                }
            }
        }
        if let Some(python) = &self.python {
            let own = || {
                self.binaries
                    .iter()
                    .find(|binary| binary.tool == ManagedTool::Uv)
                    .map(|binary| binary.bytes.as_ref())
            };
            // This bundle's `uv` runs only on its own platform.
            let installer = match installer {
                Some(installer) => installer,
                None if self.target == ToolTarget::native()? => {
                    own().ok_or(ToolRuntimeError::MissingTool(ManagedTool::Uv))?
                }
                None => return Err(ToolRuntimeError::PythonInstallerUnavailable),
            };
            python.stage(&self.target, stage.path(), installer)?;
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
            for alias in binary.tool.aliases() {
                let alias = bin_dir.join(alias);
                let target =
                    std::fs::read_link(&alias).map_err(|error| alias_entry_error(&alias, error))?;
                if target != Path::new(binary.tool.command()) {
                    return Err(ToolRuntimeError::InvalidCache(alias));
                }
            }
        }
        if let Some(python) = &self.python {
            python.verify(&self.target, &bin_dir)?;
        }
        Ok(PreparedTools {
            bin_dir,
            uv_dir: None,
            catalog: self.catalog(),
        })
    }
}

/// Name of the shared, writable directory for `uv`'s cache and project
/// Python installs, beside the content-addressed runtimes.
pub const UV_STATE_DIRECTORY: &str = "uv";

/// One baseline command a container image must carry, without its bytes.
#[derive(Debug, Clone)]
pub struct ImageTool {
    /// Program the image must provide.
    pub tool: ManagedTool,
    /// Version the executable must report to its version probe.
    pub version: semver::Version,
    /// SHA-256 of the artifact archive the executable came from.
    pub archive_sha256: [u8; 32],
}

/// The baseline tools a container image must carry, pinned without bytes.
///
/// Unlike a [`ToolBundle`], this carries no executable content and is built
/// without touching the network or the artifact cache: the image build is
/// responsible for placing exactly these commands, at these versions, under
/// [`ImageTools::container_directory`].
#[derive(Debug, Clone)]
pub struct ImageTools {
    pub(crate) target: ToolTarget,
    pub(crate) tools: Vec<ImageTool>,
    pub(crate) python: Option<PythonRuntime>,
}

impl ImageTools {
    pub(crate) const fn new(
        target: ToolTarget,
        tools: Vec<ImageTool>,
        python: Option<PythonRuntime>,
    ) -> Self {
        Self {
            target,
            tools,
            python,
        }
    }

    /// Stable identity of the pinned image tools, for directory naming.
    pub(crate) fn cache_key(&self) -> Result<String, ToolRuntimeError> {
        #[derive(Serialize)]
        struct Identity<'a> {
            target: &'a ToolTarget,
            tools: Vec<(&'a str, &'a semver::Version, &'a [u8; 32])>,
            python: Option<(&'a semver::Version, &'a [u8; 32])>,
        }
        let identity = Identity {
            target: &self.target,
            tools: self
                .tools
                .iter()
                .map(|tool| (tool.tool.command(), &tool.version, &tool.archive_sha256))
                .collect(),
            python: self
                .python
                .as_ref()
                .map(|python| (python.version(), python.interpreter_sha256())),
        };
        Ok(hex::encode(digest(&serde_json::to_vec(&identity)?)))
    }

    /// Where the image carries these tools: `/opt/aither/tools/<cache key>`.
    ///
    /// # Errors
    /// Returns an error when the identity cannot be hashed.
    pub fn container_directory(&self) -> Result<ContainerPath, ToolRuntimeError> {
        Ok(ContainerPath::root()
            .join("opt/aither/tools")
            .join(self.cache_key()?))
    }

    /// The executable directory inside a container, which goes on its `PATH`.
    pub(crate) fn container_bin_dir(&self) -> Result<ContainerPath, ToolRuntimeError> {
        Ok(self.container_directory()?.join("bin"))
    }

    /// Stable operational metadata shared by every prepared backend.
    #[must_use]
    pub fn catalog(&self) -> ToolCatalog {
        ToolCatalog {
            tools: self
                .tools
                .iter()
                .map(|tool| ToolInfo::tool(tool.tool, tool.version.clone()))
                .chain(self.python.as_ref().map(ToolInfo::python))
                .collect(),
        }
    }
}

/// Rejects a missing interpreter pin where `target` provides one, and a pin
/// where it does not.
pub fn check_python(
    target: &ToolTarget,
    python: Option<&PythonRuntime>,
) -> Result<(), ToolRuntimeError> {
    match (target.provides_python(), python) {
        (true, None) => Err(ToolRuntimeError::MissingPython),
        (false, Some(_)) => Err(ToolRuntimeError::UnexpectedPython),
        _ => Ok(()),
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

/// Maps what `read_link` reports for a missing or non-link alias to an
/// invalid cache entry; any other failure stays an I/O error.
fn alias_entry_error(path: &Path, error: std::io::Error) -> ToolRuntimeError {
    match error.kind() {
        // `read_link` fails with `InvalidInput` (EINVAL) on a regular file.
        std::io::ErrorKind::NotFound
        | std::io::ErrorKind::NotADirectory
        | std::io::ErrorKind::InvalidInput => ToolRuntimeError::InvalidCache(path.to_path_buf()),
        _ => ToolRuntimeError::Io(error),
    }
}

/// Creates `alias` as a relative symlink to its sibling `tool` executable.
fn link_alias(tool: ManagedTool, alias: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(tool.command(), alias)
    }
    #[cfg(not(unix))]
    {
        let _ = (tool, alias);
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

/// Waits between the attempts to run a freshly written executable; the whole
/// schedule totals 315 ms, long past the instant a child forked during the
/// write needs to exec and drop its inherited write descriptor.
const BUSY_BACKOFF: [std::time::Duration; 6] = [
    std::time::Duration::from_millis(5),
    std::time::Duration::from_millis(10),
    std::time::Duration::from_millis(20),
    std::time::Duration::from_millis(40),
    std::time::Duration::from_millis(80),
    std::time::Duration::from_millis(160),
];

/// Runs `command`, an executable this process just wrote, to completion.
///
/// A child forked by another thread while the file was open for writing holds
/// the write descriptor until it execs, and executing the file meanwhile fails
/// with `ETXTBSY`. That failure is retried with a short bounded backoff; any
/// other error, or a file still busy afterwards, is returned unchanged.
pub fn run_installed(command: &mut std::process::Command) -> std::io::Result<std::process::Output> {
    retry_busy(|| command.output())
}

fn retry_busy<T>(mut attempt: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    for delay in BUSY_BACKOFF {
        match attempt() {
            Err(error) if error.kind() == std::io::ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(delay);
            }
            result => return result,
        }
    }
    attempt()
}

/// Fails unless every alias of `tool` is a command its executable at `path`
/// reports in `--list`, so the managed directory never shadows a command the
/// pinned binary lacks.
fn ensure_aliases_provided(tool: ManagedTool, path: &Path) -> Result<(), ToolRuntimeError> {
    let output = run_installed(std::process::Command::new(path).arg("--list"))?;
    if !output.status.success() {
        return Err(ToolRuntimeError::AliasList(tool.command().into()));
    }
    let listed = String::from_utf8(output.stdout)?;
    let listed = listed.lines().collect::<std::collections::HashSet<_>>();
    if let Some(alias) = tool
        .aliases()
        .iter()
        .find(|alias| !listed.contains(**alias))
    {
        return Err(ToolRuntimeError::UnprovidedAlias {
            tool,
            alias: (*alias).into(),
        });
    }
    Ok(())
}

/// Makes `path` accessible to its owner only.
fn set_private(path: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
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
    /// The shared writable directory of `uv`'s cache and project Python
    /// installs, when the environment provides one.
    pub(crate) uv_dir: Option<D>,
    pub(crate) catalog: ToolCatalog,
}

impl<D> PreparedTools<D> {
    /// Read-only executable directory to expose to the sandbox.
    #[must_use]
    pub const fn bin_dir(&self) -> &D {
        &self.bin_dir
    }

    /// The writable directory shared by every session for `uv`'s cache
    /// (`UV_CACHE_DIR`) and project Python installs (`UV_PYTHON_INSTALL_DIR`),
    /// kept apart from the read-only managed interpreter and from the
    /// per-session `HOME`. `None` where the environment has no such directory:
    /// Windows, which has no managed `uv`, and container images, whose
    /// writable state belongs to the container.
    #[must_use]
    pub const fn uv_dir(&self) -> Option<&D> {
        self.uv_dir.as_ref()
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
        heel_binary: heel::HeelBinary,
    ) -> Result<(), ToolRuntimeError> {
        use futures::{StreamExt, TryStreamExt};
        let scratch = blocking::unblock(tempfile::tempdir).await?;
        let config = heel::SandboxConfig::builder()
            .heel_binary(heel_binary)
            .working_dir(
                std::fs::canonicalize(scratch.path())
                    .unwrap_or_else(|_| scratch.path().to_path_buf()),
            )
            .grant(
                heel::ResolvedPath::new(std::fs::canonicalize(&self.bin_dir)?)
                    .map_err(|error| ToolRuntimeError::SshBootstrap(error.to_string()))?,
                heel::Access::READ | heel::Access::EXEC,
            )
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
    uv: Option<UvEnvironmentText<'a>>,
    script: &'a str,
}

/// The quoted values of the `uv` variables a script exports.
struct UvEnvironmentText<'a> {
    cache: std::borrow::Cow<'a, str>,
    python_installs: std::borrow::Cow<'a, str>,
    python_preference: &'static str,
}

/// `UV_PYTHON_PREFERENCE`: a project without its own Python request gets the
/// managed `python3` from `PATH` rather than whichever interpreter `uv` last
/// installed for another project. A `.python-version` or `.venv` that no
/// `PATH` interpreter satisfies is still installed or found by `uv`.
pub const UV_PYTHON_PREFERENCE: &str = "system";

/// `uv`'s cache and project Python installs, both under the shared
/// [`PreparedTools::uv_dir`].
pub struct UvEnvironment {
    pub(crate) cache: PathBuf,
    pub(crate) python_installs: PathBuf,
}

impl PreparedTools<PathBuf> {
    /// The executable directory as text a shell script can name.
    pub(crate) fn script_bin_dir(&self) -> Result<&str, ToolRuntimeError> {
        self.bin_dir
            .to_str()
            .ok_or_else(|| ToolRuntimeError::InvalidCache(self.bin_dir.clone()))
    }

    /// Where `uv` keeps its cache (`UV_CACHE_DIR`) and the Pythons it installs
    /// for projects (`UV_PYTHON_INSTALL_DIR`), when there is a shared directory.
    pub(crate) fn uv_environment(&self) -> Option<UvEnvironment> {
        self.uv_dir.as_ref().map(|dir| UvEnvironment {
            cache: dir.join("cache"),
            python_installs: dir.join("python"),
        })
    }

    pub(crate) fn wrap_script(&self, script: &str) -> Result<String, ToolRuntimeError> {
        let text = |path: &Path| {
            path.to_str()
                .map(|text| shell_words::quote(text).into_owned())
                .ok_or_else(|| ToolRuntimeError::InvalidCache(path.to_path_buf()))
        };
        let uv = self
            .uv_environment()
            .map(|environment| {
                Ok::<_, ToolRuntimeError>(UvEnvironmentText {
                    cache: text(&environment.cache)?.into(),
                    python_installs: text(&environment.python_installs)?.into(),
                    python_preference: UV_PYTHON_PREFERENCE,
                })
            })
            .transpose()?;
        render_tool_path(self.script_bin_dir()?, uv, script)
    }

    /// The managed bash executable terminal scripts run in.
    pub(crate) fn shell(&self) -> Result<String, ToolRuntimeError> {
        self.bin_dir
            .join(ManagedTool::Bash.command())
            .into_os_string()
            .into_string()
            .map_err(|path| ToolRuntimeError::InvalidCache(path.into()))
    }
}

impl PreparedTools<ContainerPath> {
    pub(crate) fn wrap_script(&self, script: &str) -> Result<String, ToolRuntimeError> {
        render_tool_path(self.bin_dir.as_str(), None, script)
    }

    /// A command line running `script` in the managed bash of the container.
    pub(crate) fn shell_command(&self, script: &str) -> Result<String, ToolRuntimeError> {
        #[derive(Template)]
        #[template(path = "managed_shell.sh", escape = "none")]
        struct ManagedShellTemplate<'a> {
            shell: std::borrow::Cow<'a, str>,
            arguments: String,
        }
        Ok(ManagedShellTemplate {
            shell: shell_words::quote(self.bin_dir.join(ManagedTool::Bash.command()).as_str()),
            arguments: shell_words::join(shell_arguments(script)),
        }
        .render()?)
    }
}

/// Host commands a script must not call, each with what to use instead.
const BLOCKED_COMMANDS: [(&str, &str); 3] = [
    ("grep", "`rg`"),
    ("awk", "`python3` or `jq`"),
    ("find", "`fd`"),
];

/// The host command the managed shell replaces with a function running
/// [`ManagedTool::Sed`], which prints line ranges and refuses the rest.
const SED_FUNCTION: &str = "sed";

/// Whether `command` is one the managed shell refuses or replaces in a
/// script, so only a shell can run it.
pub fn is_blocked_command(command: &str) -> bool {
    command == SED_FUNCTION || BLOCKED_COMMANDS.iter().any(|(name, _)| *name == command)
}

#[derive(Template)]
#[template(path = "blocked_command.txt", escape = "none")]
struct BlockedCommandTemplate<'a> {
    name: &'a str,
    replacement: &'a str,
}

struct BlockedCommandFunction {
    name: &'static str,
    message: String,
}

#[derive(Template)]
#[template(path = "blocked_commands.sh", escape = "none")]
struct BlockedCommandsTemplate<'a> {
    commands: Vec<BlockedCommandFunction>,
    sed: &'a str,
    sed_tool: &'a str,
    script: &'a str,
}

/// `script` behind shell functions that make the blocked commands fail with a
/// message naming their replacement, and a `sed` that runs the managed
/// `aither-sed`. They are not exported, so only the script's own shell sees
/// them: child processes still reach the host programs.
fn with_blocked_commands(script: &str) -> String {
    BlockedCommandsTemplate {
        script,
        sed: SED_FUNCTION,
        sed_tool: ManagedTool::Sed.command(),
        commands: BLOCKED_COMMANDS
            .iter()
            .map(|&(name, replacement)| {
                let message = BlockedCommandTemplate { name, replacement }
                    .render()
                    .expect("the blocked command message is a static template");
                BlockedCommandFunction {
                    name,
                    message: shell_words::quote(&message).into_owned(),
                }
            })
            .collect(),
    }
    .render()
    .expect("the blocked commands preamble is a static template")
}

/// The arguments with which the managed bash runs `script`: no startup file
/// is read, so a script behaves the same on every host, and the blocked
/// commands are defined ahead of it.
pub fn shell_arguments(script: &str) -> Vec<String> {
    [
        "--noprofile",
        "--norc",
        "-c",
        &with_blocked_commands(script),
    ]
    .map(Into::into)
    .into()
}

/// Prepends `bin_dir` to a POSIX shell script's `PATH`.
fn render_tool_path(
    bin_dir: &str,
    uv: Option<UvEnvironmentText<'_>>,
    script: &str,
) -> Result<String, ToolRuntimeError> {
    Ok(ToolPathTemplate {
        bin_dir: shell_words::quote(bin_dir),
        uv,
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
    source: CatalogCommand,
    command: &'static str,
    version: semver::Version,
}

/// What reports the version of a catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogCommand {
    /// A managed tool, by its own banner.
    Tool(ManagedTool),
    /// The managed `python3`, which `uv` installs.
    Python,
}

impl ToolInfo {
    const fn tool(tool: ManagedTool, version: semver::Version) -> Self {
        Self {
            source: CatalogCommand::Tool(tool),
            command: tool.command(),
            version,
        }
    }

    fn python(python: &PythonRuntime) -> Self {
        Self {
            source: CatalogCommand::Python,
            command: PYTHON_COMMAND,
            version: python.version().clone(),
        }
    }
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
        let reported = match self.source {
            CatalogCommand::Tool(tool) => tool.reported_version(output),
            CatalogCommand::Python => PythonRuntime::reported_version(output),
        };
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

impl ToolCatalogTemplate<'_> {
    fn has(&self, command: &str) -> bool {
        self.tools.iter().any(|tool| tool.command == command)
    }
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

#[derive(Template)]
#[template(path = "command_line_tools_missing.txt", escape = "none")]
struct CommandLineToolsMissingNotice;

/// Fail-closed errors from tool initialization and shell integration.
#[derive(Debug, thiserror::Error)]
pub enum ToolRuntimeError {
    /// Configured container has no execution transport.
    #[error("container runtime is not configured")]
    MissingContainerRuntime,
    /// A macOS host has no Command Line Tools to provide `git`.
    #[error("{}", CommandLineToolsMissingNotice)]
    CommandLineToolsMissing,
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
    /// The container image does not carry the verified baseline tools the bundle pins.
    #[error("container image lacks verified baseline tools at {directory}: {defect}")]
    InvalidImageTools {
        /// Expected `/opt/aither/tools/<cache key>` directory.
        directory: ContainerPath,
        /// Which image verification check failed.
        defect: ImageToolDefect,
    },
    /// Required command missing from a bundle.
    #[error("required tool is missing: {0:?}")]
    MissingTool(ManagedTool),
    /// A bundle declares the same command twice.
    #[error("duplicate managed tool: {0:?}")]
    DuplicateTool(ManagedTool),
    /// A bundle carries a tool its platform does not provide.
    #[error("tool is not provided on this platform: {0:?}")]
    UnexpectedTool(ManagedTool),
    /// The platform provides a managed `python3` but nothing pins its interpreter.
    #[error("the interpreter pin of the managed python3 is missing")]
    MissingPython,
    /// The distribution pins several interpreters for one target.
    #[error("the distribution pins several interpreters for one target")]
    DuplicatePython,
    /// An interpreter is pinned for a platform that has no managed `uv`.
    #[error("python3 is not provided on this platform")]
    UnexpectedPython,
    /// `uv python install` failed while preparing the managed interpreter.
    #[error("installing the managed python3 failed ({status:?}): {stderr}")]
    PythonInstall {
        /// Process exit status, if available.
        status: Option<i32>,
        /// What `uv` reported.
        stderr: String,
    },
    /// The installed interpreter tree holds an entry that is neither a
    /// directory, a regular file nor a link inside the tree.
    #[error("the managed python3 tree holds an unacceptable entry: {0}")]
    InvalidPythonTree(PathBuf),
    /// The installed interpreter executable does not have the pinned digest.
    #[error("the managed python3 interpreter failed SHA-256 verification")]
    PythonIntegrity,
    /// A bundle for another platform has no `uv` that runs here to install its interpreter.
    #[error("the managed python3 of another platform needs an installer that runs on this machine")]
    PythonInstallerUnavailable,
    /// A multicall tool's `--list` failed.
    #[error("`{0} --list` failed")]
    AliasList(String),
    /// An alias names a command its multicall binary does not provide.
    #[error("{tool:?} does not provide the aliased command `{alias}`")]
    UnprovidedAlias {
        /// The multicall tool.
        tool: ManagedTool,
        /// The command it lacks.
        alias: String,
    },
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

/// The check inside [`ToolRuntimeError::InvalidImageTools`] that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageToolDefect {
    /// The `<cache key>/bin` directory does not exist in the image.
    MissingDirectory,
    /// A pinned executable is absent or is not a regular file.
    MissingBinary(String),
    /// An alias is absent or is not a symlink to its multicall executable.
    InvalidAlias(String),
    /// `python3` is absent or is not a link to the interpreter of the pinned tree.
    InvalidPython,
}

impl std::fmt::Display for ImageToolDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDirectory => write!(f, "missing bin directory"),
            Self::MissingBinary(command) => {
                write!(f, "missing or non-regular executable `{command}`")
            }
            Self::InvalidAlias(alias) => {
                write!(f, "missing or mislinked alias `{alias}`")
            }
            Self::InvalidPython => write!(f, "missing or mislinked python3"),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[cfg(unix)]
    fn run_in_shell(script: &str) -> std::process::Output {
        std::process::Command::new("bash")
            .args(shell_arguments(script))
            .output()
            .expect("bash should run")
    }

    #[cfg(unix)]
    #[test]
    fn blocked_commands_fail_with_their_replacement() {
        for (name, replacement) in [("grep", "`rg`"), ("awk", "`jq`"), ("find", "`fd`")] {
            let output = run_in_shell(&format!("{name} x"));
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success(), "{name} should fail");
            assert!(output.stdout.is_empty(), "{name} should print to stderr");
            assert!(
                stderr.contains(&format!("`{name}` is not available")),
                "{name}: {stderr}"
            );
            assert!(stderr.contains(replacement), "{name}: {stderr}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn sed_runs_the_managed_aither_sed_with_its_arguments() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().expect("tempdir");
        let managed = directory.path().join(ManagedTool::Sed.command());
        std::fs::write(&managed, "#!/bin/sh\nprintf 'managed:%s\\n' \"$*\"\n").expect("write");
        std::fs::set_permissions(&managed, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let path = format!(
            "{}:{}",
            directory.path().display(),
            std::env::var("PATH").expect("PATH")
        );
        let output = std::process::Command::new("bash")
            .args(shell_arguments("sed -n '2,3p' file"))
            .env("PATH", path)
            .output()
            .expect("bash should run");
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "managed:-n 2,3p file\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn blocked_commands_stay_out_of_child_processes() {
        let output = run_in_shell("bash -c \"printf 'a\\n' | sed 's/a/b/'\"");
        assert!(output.status.success(), "{output:?}");
        assert_eq!(String::from_utf8_lossy(&output.stdout), "b\n");
    }

    #[cfg(unix)]
    #[test]
    fn make_recipes_reach_the_host_sed() {
        let directory = tempfile::tempdir().expect("tempdir");
        let makefile = directory.path().join("Makefile");
        std::fs::write(&makefile, "all:\n\t@echo a | sed 's/a/b/'\n").expect("write Makefile");
        let output = run_in_shell(&format!(
            "make -f {}",
            shell_words::quote(&makefile.to_string_lossy())
        ));
        assert!(output.status.success(), "{output:?}");
        assert_eq!(String::from_utf8_lossy(&output.stdout), "b\n");
    }

    #[test]
    fn artifact_digest_matches_known_sha256_vector() {
        assert_eq!(
            hex::encode(digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// A temporary directory that can hold prepared runtimes: their managed
    /// interpreter tree is read-only, which a plain `TempDir` cannot delete.
    pub struct UnlockingTempDir(tempfile::TempDir);

    impl std::ops::Deref for UnlockingTempDir {
        type Target = tempfile::TempDir;

        fn deref(&self) -> &tempfile::TempDir {
            &self.0
        }
    }

    impl Drop for UnlockingTempDir {
        fn drop(&mut self) {
            let _ = unlock_tree(self.0.path());
        }
    }

    pub fn unlocking_tempdir() -> UnlockingTempDir {
        UnlockingTempDir(tempfile::tempdir().unwrap())
    }

    /// The `CPython` version the fixture `uv` installs.
    pub const FIXTURE_PYTHON_VERSION: &str = "3.13.1";

    /// The interpreter the fixture `uv` installs: a script reporting
    /// `version`, which runs anything else under the host's `python3` so
    /// tests of Python scripts execute real code.
    pub fn fixture_interpreter(version: &str) -> Vec<u8> {
        format!(
            "#!/bin/sh\ncase \"$1\" in --version) printf '%s\\n' 'Python {version}'; exit 0;; esac\n\
             PATH=/usr/bin:/bin exec python3 \"$@\"\n"
        )
        .into_bytes()
    }

    /// The manifest pin matching what the fixture `uv` installs on `target`.
    pub fn fixture_python_pin(target: &ToolTarget) -> PythonPin {
        PythonPin {
            version: FIXTURE_PYTHON_VERSION.into(),
            target: target.clone(),
            interpreter_sha256: hex::encode(digest(&fixture_interpreter(FIXTURE_PYTHON_VERSION))),
        }
    }

    /// The native target's interpreter pin for an interpreter with `digest`,
    /// as manifest JSON; empty where the platform has no managed `python3`.
    pub fn python_pins_for(digest: [u8; 32], version: &str) -> Vec<PythonPin> {
        let target = ToolTarget::native().unwrap();
        target
            .provides_python()
            .then(|| PythonPin {
                version: version.into(),
                target,
                interpreter_sha256: hex::encode(digest),
            })
            .into_iter()
            .collect()
    }

    /// The interpreter pin matching what the fixture `uv` installs.
    pub fn fixture_python() -> PythonRuntime {
        fixture_python_pin(&ToolTarget::native().unwrap())
            .runtime()
            .unwrap()
    }

    pub fn bundle() -> ToolBundle {
        bundle_pinned("1.0.0")
    }

    /// A bundle pinned to `version` whose commands all report version 1.0.0,
    /// so any other pin fails the version probe while passing integrity checks.
    pub fn bundle_pinned(version: &str) -> ToolBundle {
        let binaries = native_tools()
            .iter()
            .map(|&tool| {
                let bytes = script_tool(tool, "1.0.0");
                ToolBinary::verify(tool, version, &digest(&bytes), bytes).unwrap()
            })
            .collect();
        let target = ToolTarget::native().unwrap();
        let python = target.provides_python().then(fixture_python);
        ToolBundle::new(target, binaries, python).unwrap()
    }

    /// The tools the test host's platform must provide.
    pub fn native_tools() -> &'static [ManagedTool] {
        ToolTarget::native().unwrap().required_tools()
    }

    /// A script executable that reports `<command> <version>` to `--version`
    /// and, as a multicall tool, lists its aliases to `--list`.
    ///
    /// Invoked through one of its aliases it runs the host's command of that
    /// name, as the real multicall binary would.
    ///
    /// The `bash` fixture reports its banner to `--version` and otherwise
    /// runs the host's bash, with `AITHER_FIXTURE_BASH` set to the pinned
    /// version so a script can tell it ran through the managed shell.
    pub fn script_tool(tool: ManagedTool, version: &str) -> Vec<u8> {
        if tool == ManagedTool::Bash {
            return format!(
                "#!/bin/sh\ncase \"$1\" in --version) printf '%s\\n' 'GNU bash, version {version}(1)-release (fixture)'; exit 0;; esac\n\
                 AITHER_FIXTURE_BASH={version}\nexport AITHER_FIXTURE_BASH\nexec /bin/bash \"$@\"\n"
            )
            .into_bytes();
        }
        if tool == ManagedTool::Uv {
            return fixture_uv(version, "");
        }
        let multicall = if tool.aliases().is_empty() {
            String::new()
        } else {
            format!(
                "case \"${{0##*/}}\" in {}) ;; *) PATH=/usr/bin:/bin exec \"${{0##*/}}\" \"$@\";; esac\n\
                 case \"$1\" in --list) printf '%s\\n' {}; exit 0;; esac\n",
                tool.command(),
                tool.aliases().join(" ")
            )
        };
        format!(
            "#!/bin/sh\n{multicall}printf '%s\\n' '{} {version}'\n",
            tool.command()
        )
        .into_bytes()
    }

    /// A `uv` that reports `uv <version>` and, like the real one, installs
    /// `python install <request>` as `$UV_PYTHON_INSTALL_DIR/<request>` with
    /// its housekeeping entries (an absolute minor-version link, `.lock`,
    /// `.temp`). `extra` is shell run last, with the installed tree in `$dir`.
    pub fn fixture_uv(version: &str, extra: &str) -> Vec<u8> {
        let interpreter = String::from_utf8(fixture_interpreter(FIXTURE_PYTHON_VERSION)).unwrap();
        r#"#!/bin/sh
PATH=/usr/bin:/bin
case "$1" in
--version) printf '%s\n' 'uv @VERSION@'; exit 0;;
python)
    for request; do :; done
    dir="$UV_PYTHON_INSTALL_DIR/$request"
    mkdir -p "$dir/bin" "$UV_PYTHON_INSTALL_DIR/.temp"
    cat > "$dir/bin/python3.13" <<'INTERPRETER'
@INTERPRETER@INTERPRETER
    chmod 755 "$dir/bin/python3.13"
    ln -s python3.13 "$dir/bin/python3"
    ln -s "$dir" "$UV_PYTHON_INSTALL_DIR/cpython-3.13-minor"
    : > "$UV_PYTHON_INSTALL_DIR/.lock"
    @EXTRA@
    exit 0;;
esac
printf '%s\n' 'uv @VERSION@'
"#
        .replace("@VERSION@", version)
        .replace("@INTERPRETER@", &interpreter)
        .replace("@EXTRA@", extra)
        .into_bytes()
    }

    /// The `ImageTools` counterpart of [`bundle`]: the same target and pins,
    /// keyed by the digests of the script bytes a fixture artifact would hold.
    pub fn image_tools() -> ImageTools {
        image_tools_pinned("1.0.0")
    }

    /// [`image_tools`] pinned to another version, for a different cache key.
    pub fn image_tools_pinned(version: &str) -> ImageTools {
        let tools = native_tools()
            .iter()
            .map(|&tool| ImageTool {
                tool,
                version: semver::Version::parse(version).unwrap(),
                archive_sha256: digest(&script_tool(tool, version)),
            })
            .collect();
        let target = ToolTarget::native().unwrap();
        let python = target.provides_python().then(fixture_python);
        ImageTools::new(target, tools, python)
    }

    /// Tools as a container sees them once baked into its image.
    pub fn container_tools(tools: &ImageTools) -> PreparedTools<ContainerPath> {
        PreparedTools {
            bin_dir: tools.container_bin_dir().unwrap(),
            uv_dir: None,
            catalog: tools.catalog(),
        }
    }

    // Runs on every host: Windows must still produce POSIX container paths.
    #[test]
    fn container_tool_paths_are_posix_on_every_host() {
        let tools = image_tools();
        let key = tools.cache_key().unwrap();
        assert_eq!(
            tools.container_directory().unwrap().as_str(),
            format!("/opt/aither/tools/{key}")
        );
        assert_eq!(
            tools.container_bin_dir().unwrap().as_str(),
            format!("/opt/aither/tools/{key}/bin")
        );
        let script = container_tools(&tools).wrap_script("rg --version").unwrap();
        assert!(
            script.starts_with(&format!("PATH=/opt/aither/tools/{key}/bin")),
            "{script}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wrapped_scripts_export_the_shared_uv_directory() {
        let cache = unlocking_tempdir();
        let ready = bundle().prepare_local(cache.path()).await.unwrap();
        let uv = ready.uv_dir().unwrap().clone();
        let script = ready
            .wrap_script(
                r#"printf '%s\n' "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_PREFERENCE""#,
            )
            .unwrap();
        let output = std::process::Command::new("sh")
            .args(["-c", &script])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!(
                "{}\n{}\nsystem\n",
                uv.join("cache").display(),
                uv.join("python").display()
            )
        );
        // A container image has no such directory to export.
        let container = container_tools(&image_tools()).wrap_script("true").unwrap();
        assert!(!container.contains("UV_"), "{container}");
    }

    #[test]
    fn missing_baseline_tool_is_an_initialization_error() {
        let mut binaries = bundle().binaries.to_vec();
        let removed = binaries.pop().unwrap().tool;
        assert!(matches!(
            ToolBundle::new(ToolTarget::native().unwrap(), binaries, Some(fixture_python())),
            Err(ToolRuntimeError::MissingTool(missing)) if missing == removed
        ));
    }

    fn target(os: &str) -> ToolTarget {
        ToolTarget::from_probe(os, "x86_64").unwrap()
    }

    #[test]
    fn required_tools_depend_on_the_platform() {
        for os in ["Darwin", "Linux"] {
            assert_eq!(target(os).required_tools(), ManagedTool::ALL, "{os}");
        }
        let windows = target("Windows_NT").required_tools();
        assert_eq!(
            windows,
            [
                ManagedTool::ApplyPatch,
                ManagedTool::Ripgrep,
                ManagedTool::AstGrep,
                ManagedTool::Jq,
            ]
        );
        assert!(!windows.contains(&ManagedTool::Coreutils));
        assert!(!windows.contains(&ManagedTool::Bash));
    }

    fn script_binaries(tools: &[ManagedTool]) -> Vec<ToolBinary> {
        tools
            .iter()
            .map(|&tool| {
                let bytes = script_tool(tool, "1.0.0");
                ToolBinary::verify(tool, "1.0.0", &digest(&bytes), bytes).unwrap()
            })
            .collect()
    }

    #[test]
    fn windows_bundle_needs_only_the_portable_tools() {
        let windows = target("Windows_NT");
        ToolBundle::new(
            windows.clone(),
            script_binaries(windows.required_tools()),
            None,
        )
        .unwrap();
        assert!(matches!(
            ToolBundle::new(windows.clone(), script_binaries(&ManagedTool::ALL), None),
            Err(ToolRuntimeError::UnexpectedTool(ManagedTool::Bash))
        ));
        // Windows has no managed `uv`, so no interpreter pin either.
        assert!(matches!(
            ToolBundle::new(
                windows.clone(),
                script_binaries(windows.required_tools()),
                Some(fixture_python())
            ),
            Err(ToolRuntimeError::UnexpectedPython)
        ));
        assert!(!windows.provides_python());
    }

    #[test]
    fn a_platform_with_uv_needs_its_interpreter_pinned() {
        let linux = target("Linux");
        assert!(linux.provides_python());
        assert!(matches!(
            ToolBundle::new(linux.clone(), script_binaries(&ManagedTool::ALL), None),
            Err(ToolRuntimeError::MissingPython)
        ));
        ToolBundle::new(
            linux,
            script_binaries(&ManagedTool::ALL),
            Some(fixture_python()),
        )
        .unwrap();
    }

    #[test]
    fn unix_bundle_without_coreutils_is_incomplete() {
        let linux = target("Linux");
        let tools = ManagedTool::ALL
            .into_iter()
            .filter(|tool| *tool != ManagedTool::Coreutils)
            .collect::<Vec<_>>();
        assert!(matches!(
            ToolBundle::new(linux, script_binaries(&tools), Some(fixture_python())),
            Err(ToolRuntimeError::MissingTool(ManagedTool::Coreutils))
        ));
    }

    #[test]
    fn each_tool_parses_its_own_version_banner() {
        let version = |text| semver::Version::parse(text).unwrap();
        let cases = [
            (ManagedTool::Jq, "jq-1.8.2\n", "1.8.2"),
            (
                ManagedTool::Coreutils,
                "coreutils 0.12.0 (multi-call binary)\n",
                "0.12.0",
            ),
            (ManagedTool::Xargs, "xargs 0.11.0\n", "0.11.0"),
            (
                ManagedTool::Bash,
                "GNU bash, version 5.3.20(1)-release (aarch64-apple-darwin25.0.0)\nCopyright (C) 2025 Free Software Foundation, Inc.\n",
                "5.3.20",
            ),
            (ManagedTool::Fd, "fd 10.5.0\n", "10.5.0"),
            (ManagedTool::Sed, "aither-sed 0.13.0\n", "0.13.0"),
            (
                ManagedTool::Ripgrep,
                "ripgrep 15.2.0\nfeatures:+pcre2\n",
                "15.2.0",
            ),
            (ManagedTool::AstGrep, "ast-grep 0.45.3\n", "0.45.3"),
            (
                ManagedTool::Uv,
                "uv 0.13.7 (316c827e0 2026-11-03 aarch64-apple-darwin)\n",
                "0.13.7",
            ),
        ];
        for (tool, output, expected) in cases {
            assert_eq!(
                tool.reported_version(output),
                Some(version(expected)),
                "{tool:?}"
            );
        }
        // A banner in another tool's format is not accepted.
        assert_eq!(ManagedTool::Fd.reported_version("fd-10.5.0\n"), None);
        assert_eq!(ManagedTool::Coreutils.reported_version("coreutils\n"), None);
        assert_eq!(ManagedTool::Xargs.reported_version(""), None);
        assert_eq!(ManagedTool::Bash.reported_version("bash 5.3.20\n"), None);
        assert_eq!(
            ManagedTool::Bash.reported_version("GNU bash, version 5.3.20\n"),
            None
        );
        assert_eq!(ManagedTool::Uv.reported_version("uv\n"), None);
        assert_eq!(ManagedTool::Uv.reported_version("uv-0.13.7\n"), None);
    }

    #[cfg(unix)]
    fn alias_path(prepared: &PreparedTools<PathBuf>, alias: &str) -> PathBuf {
        prepared.bin_dir().join(alias)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn coreutils_aliases_are_relative_links_to_the_multicall_binary() {
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let prepared = bundle.prepare_local(cache.path()).await.unwrap();
        assert_ne!(COREUTILS_ALIASES.len(), 0);
        for alias in COREUTILS_ALIASES {
            assert_eq!(
                std::fs::read_link(alias_path(&prepared, alias)).unwrap(),
                Path::new("coreutils"),
                "{alias}"
            );
        }
        for excluded in ["grep", "sed", "awk", "find", "xargs", "fd", "[", "stdbuf"] {
            assert!(!COREUTILS_ALIASES.contains(&excluded), "{excluded}");
            assert!(!alias_path(&prepared, excluded).is_symlink(), "{excluded}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn tampered_aliases_are_rejected_and_repaired() {
        type Tamper = fn(&Path, &Path);
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let prepared = bundle.prepare_local(cache.path()).await.unwrap();
        let runtime = prepared.bin_dir().parent().unwrap().to_path_buf();
        let bin = prepared.bin_dir().clone();
        let alias = bin.join("date");
        // Each tampering replaces the alias, given its path and the bin directory.
        let tamper: [(&str, Tamper); 4] = [
            ("retargeted link", |alias, _| {
                std::os::unix::fs::symlink("rg", alias).unwrap();
            }),
            ("absolute link", |alias, bin| {
                std::os::unix::fs::symlink(bin.join("coreutils"), alias).unwrap();
            }),
            ("regular file", |alias, _| {
                std::fs::write(alias, b"#!/bin/sh\n").unwrap();
            }),
            ("missing", |_, _| {}),
        ];
        for (name, apply) in tamper {
            std::fs::remove_file(&alias).ok();
            std::os::unix::fs::symlink("coreutils", &alias).unwrap();
            bundle.verify_prepared(&runtime).unwrap();
            std::fs::remove_file(&alias).unwrap();
            apply(&alias, &bin);
            assert!(
                matches!(
                    bundle.verify_prepared(&runtime),
                    Err(ToolRuntimeError::InvalidCache(path)) if path == alias
                ),
                "{name} was accepted"
            );
        }
        std::fs::remove_file(&alias).ok();
        std::os::unix::fs::symlink("rg", &alias).unwrap();
        let repaired = bundle.prepare_local(cache.path()).await.unwrap();
        assert_prepared(&bundle, &repaired);
        assert_eq!(
            std::fs::read_link(bin.join("date")).unwrap(),
            Path::new("coreutils")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn alias_the_binary_does_not_provide_is_a_staging_error() {
        let cache = unlocking_tempdir();
        let mut binaries = script_binaries(&ManagedTool::ALL);
        let limited = b"#!/bin/sh\ncase \"$1\" in --list) printf 'cat\\n'; exit 0;; esac\nprintf '%s\\n' 'coreutils 1.0.0'\n".to_vec();
        let position = binaries
            .iter()
            .position(|binary| binary.tool == ManagedTool::Coreutils)
            .unwrap();
        binaries[position] =
            ToolBinary::verify(ManagedTool::Coreutils, "1.0.0", &digest(&limited), limited)
                .unwrap();
        let bundle = ToolBundle::new(
            ToolTarget::native().unwrap(),
            binaries,
            Some(fixture_python()),
        )
        .unwrap();
        assert!(matches!(
            bundle.prepare_local(cache.path()).await,
            Err(ToolRuntimeError::UnprovidedAlias {
                tool: ManagedTool::Coreutils,
                ..
            })
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
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let first = bundle.prepare_local(cache.path()).await.unwrap();
        let second = bundle.prepare_local(cache.path()).await.unwrap();
        assert_eq!(first.bin_dir(), second.bin_dir());
        assert_eq!(first.catalog(), second.catalog());
        for &tool in native_tools() {
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
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let (left, right) = futures_lite::future::zip(
            bundle.prepare_local(cache.path()),
            bundle.prepare_local(cache.path()),
        )
        .await;
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.bin_dir(), right.bin_dir());
        for &tool in native_tools() {
            assert!(
                left.bin_dir()
                    .join(tool.executable_name(&bundle.target))
                    .is_file()
            );
        }
    }

    #[tokio::test]
    async fn managed_tools_cannot_overlap_a_writable_grant() {
        let cache = unlocking_tempdir();
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
        let cache = unlocking_tempdir();
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
        for &tool in native_tools() {
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

    /// Entries left in the cache root other than the runtime, its lock file
    /// and the shared `uv` directory.
    fn leftovers(root: &Path, bundle: &ToolBundle) -> Vec<OsString> {
        let key = bundle.cache_key().unwrap();
        let lock = format!(".{key}.lock");
        std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| *name != *key && *name != *lock && *name != *UV_STATE_DIRECTORY)
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
        let cache = unlocking_tempdir();
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
        let cache = unlocking_tempdir();
        let bundle = bundle();
        std::fs::create_dir_all(cache.path().join(bundle.cache_key().unwrap()).join("bin"))
            .unwrap();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        assert_prepared(&bundle, &ready);
        assert_eq!(leftovers(cache.path(), &bundle), Vec::<OsString>::new());
    }

    #[tokio::test]
    async fn tampered_executable_is_replaced() {
        let cache = unlocking_tempdir();
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
        let cache = unlocking_tempdir();
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
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let snapshot = |prepared: &PreparedTools<PathBuf>| {
            let mut entries = vec![prepared.bin_dir().parent().unwrap().to_path_buf()];
            entries.push(prepared.bin_dir().clone());
            entries.extend(native_tools().iter().map(|tool| {
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
        let cache = unlocking_tempdir();
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

    /// A bundle whose `uv` is `uv`, otherwise the fixture bundle.
    pub fn bundle_with_uv(uv: Vec<u8>) -> ToolBundle {
        let mut binaries = bundle().binaries.to_vec();
        let position = binaries
            .iter()
            .position(|binary| binary.tool == ManagedTool::Uv)
            .unwrap();
        binaries[position] =
            ToolBinary::verify(ManagedTool::Uv, "1.0.0", &digest(&uv), uv).unwrap();
        ToolBundle::new(
            ToolTarget::native().unwrap(),
            binaries,
            Some(fixture_python()),
        )
        .unwrap()
    }

    /// Every entry under `path` that is not a link, with its mode.
    #[cfg(unix)]
    fn modes_under(path: &Path) -> Vec<(PathBuf, u32)> {
        use std::os::unix::fs::PermissionsExt;
        let mut found = Vec::new();
        let mut pending = vec![path.to_path_buf()];
        while let Some(path) = pending.pop() {
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            if metadata.is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                pending.extend(
                    std::fs::read_dir(&path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path()),
                );
            }
            found.push((path, metadata.permissions().mode()));
        }
        found
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_managed_python_is_installed_read_only_and_linked() {
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let prepared = bundle.prepare_local(cache.path()).await.unwrap();
        let bin = prepared.bin_dir();
        let request = bundle
            .python
            .as_ref()
            .unwrap()
            .request(bundle.target())
            .unwrap();
        let link = format!(".python/{request}/bin/python3.13");
        assert_eq!(
            std::fs::read_link(bin.join("python3")).unwrap(),
            Path::new(&link)
        );
        // The link reaches the pinned interpreter, which runs.
        let output = std::process::Command::new(bin.join("python3"))
            .arg("--version")
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("Python {FIXTURE_PYTHON_VERSION}\n")
        );
        // Everything uv wrote besides the tree itself is gone.
        let python_dir = bin.join(".python");
        let installed = std::fs::read_dir(&python_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(installed, [OsString::from(&request)]);
        assert!(!bin.parent().unwrap().join(".python-install").exists());
        assert!(!bin.join(".python-install").exists());
        // The tree is read-only throughout; only the shared uv directory is writable.
        let modes = modes_under(&python_dir);
        assert!(modes.len() > 3);
        for (path, mode) in modes {
            assert_eq!(mode & 0o222, 0, "{} is writable", path.display());
        }
        let uv = prepared.uv_dir().unwrap();
        assert!(uv.is_dir() && !uv.starts_with(bin));
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::symlink_metadata(uv).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700);
        }
        let environment = prepared.uv_environment().unwrap();
        assert_eq!(environment.cache, uv.join("cache"));
        assert_eq!(environment.python_installs, uv.join("python"));
        assert!(!environment.python_installs.starts_with(&python_dir));
    }

    #[cfg(unix)]
    #[test]
    fn the_python_version_and_interpreter_enter_the_cache_key() {
        let base = bundle();
        let key = base.cache_key().unwrap();
        let pinned = |version: &str, interpreter: [u8; 32]| {
            let mut bundle = base.clone();
            bundle.python = Some(
                PythonPin {
                    version: version.into(),
                    target: ToolTarget::native().unwrap(),
                    interpreter_sha256: hex::encode(interpreter),
                }
                .runtime()
                .unwrap(),
            );
            bundle.cache_key().unwrap()
        };
        let digest_of = digest(&fixture_interpreter(FIXTURE_PYTHON_VERSION));
        assert_eq!(pinned(FIXTURE_PYTHON_VERSION, digest_of), key);
        assert_ne!(pinned("3.13.2", digest_of), key);
        assert_ne!(pinned(FIXTURE_PYTHON_VERSION, [7; 32]), key);
        let image = image_tools();
        let image_key = image.cache_key().unwrap();
        let mut other = image;
        other.python = Some(
            PythonPin {
                version: "3.13.2".into(),
                target: ToolTarget::native().unwrap(),
                interpreter_sha256: hex::encode(digest_of),
            }
            .runtime()
            .unwrap(),
        );
        assert_ne!(other.cache_key().unwrap(), image_key);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn tampering_with_the_python_link_or_interpreter_is_rejected_and_repaired() {
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let runtime = ready.bin_dir().parent().unwrap().to_path_buf();
        let python = ready.bin_dir().join("python3");
        let target = std::fs::read_link(&python).unwrap();
        let interpreter = ready.bin_dir().join(&target);

        std::fs::remove_file(&python).unwrap();
        std::os::unix::fs::symlink("../elsewhere/python3.13", &python).unwrap();
        assert!(matches!(
            bundle.verify_prepared(&runtime),
            Err(ToolRuntimeError::InvalidCache(path)) if path == python
        ));
        std::fs::remove_file(&python).unwrap();
        assert!(matches!(
            bundle.verify_prepared(&runtime),
            Err(ToolRuntimeError::InvalidCache(path)) if path == python
        ));
        std::os::unix::fs::symlink(&target, &python).unwrap();
        bundle.verify_prepared(&runtime).unwrap();

        // An interpreter that is not the pinned bytes.
        let parent = interpreter.parent().unwrap();
        make_writable(parent);
        make_writable(&interpreter);
        std::fs::write(&interpreter, b"#!/bin/sh\necho Python 9.9.9\n").unwrap();
        set_executable_readonly(&interpreter).unwrap();
        assert!(matches!(
            bundle.verify_prepared(&runtime),
            Err(ToolRuntimeError::PythonIntegrity)
        ));
        let repaired = bundle.prepare_local(cache.path()).await.unwrap();
        assert_prepared(&bundle, &repaired);
        assert_eq!(
            std::fs::read(&interpreter).unwrap(),
            fixture_interpreter(FIXTURE_PYTHON_VERSION)
        );
        assert_eq!(leftovers(cache.path(), &bundle), Vec::<OsString>::new());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_writable_interpreter_is_rejected() {
        let cache = unlocking_tempdir();
        let bundle = bundle();
        let ready = bundle.prepare_local(cache.path()).await.unwrap();
        let interpreter = ready
            .bin_dir()
            .join(std::fs::read_link(ready.bin_dir().join("python3")).unwrap());
        make_writable(interpreter.parent().unwrap());
        make_writable(&interpreter);
        assert!(matches!(
            bundle.verify_prepared(ready.bin_dir().parent().unwrap()),
            Err(ToolRuntimeError::InvalidCache(path)) if path == interpreter
        ));
    }

    /// Prepares a bundle whose `uv` runs `extra` after installing, and
    /// returns the staging error together with what the cache root kept.
    #[cfg(unix)]
    async fn stage_with(extra: &str) -> (ToolRuntimeError, Vec<OsString>) {
        let cache = unlocking_tempdir();
        let bundle = bundle_with_uv(fixture_uv("1.0.0", extra));
        let error = bundle.prepare_local(cache.path()).await.unwrap_err();
        (error, leftovers(cache.path(), &bundle))
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_python_tree_with_an_unacceptable_entry_is_not_staged() {
        for (name, extra) in [
            ("absolute link", r#"ln -s /etc/passwd "$dir/bin/escape""#),
            (
                "link out of the tree",
                r#"ln -s ../../../outside "$dir/lib""#,
            ),
            ("named pipe", r#"mkfifo "$dir/pipe""#),
        ] {
            let (error, kept) = stage_with(extra).await;
            assert!(
                matches!(error, ToolRuntimeError::InvalidPythonTree(_)),
                "{name}: {error:?}"
            );
            // The rejected stage was deleted again, read-only or not.
            assert!(kept.is_empty(), "{name} left {kept:?}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_installed_interpreter_that_is_not_the_pinned_one_is_not_staged() {
        let (error, kept) = stage_with(r#"printf 'other' > "$dir/bin/python3.13""#).await;
        assert!(
            matches!(error, ToolRuntimeError::PythonIntegrity),
            "{error:?}"
        );
        assert!(kept.is_empty(), "{kept:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_python_install_reports_what_uv_said() {
        let (error, kept) = stage_with("echo 'no network' >&2; exit 3").await;
        match error {
            ToolRuntimeError::PythonInstall { status, stderr } => {
                assert_eq!(status, Some(3));
                assert!(stderr.contains("no network"), "{stderr}");
            }
            error => panic!("expected PythonInstall, got {error:?}"),
        }
        assert!(kept.is_empty(), "{kept:?}");
    }

    #[cfg(unix)]
    #[test]
    fn the_python_installer_runs_with_a_cleared_environment() {
        // `UV_PYTHON_INSTALL_MIRROR` and friends must not redirect the download.
        let cache = unlocking_tempdir();
        // SAFETY: the installer inherits only the proxy and certificate
        // variables, and no other test reads this one.
        unsafe { std::env::set_var("UV_PYTHON_INSTALL_MIRROR", "https://example.invalid") };
        let bundle = bundle_with_uv(fixture_uv(
            "1.0.0",
            r#"[ -z "${UV_PYTHON_INSTALL_MIRROR+x}" ] || exit 9"#,
        ));
        let result = bundle.prepare_local_sync(cache.path(), &bundle.cache_key().unwrap(), None);
        unsafe { std::env::remove_var("UV_PYTHON_INSTALL_MIRROR") };
        result.unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn another_platforms_interpreter_needs_an_installer_that_runs_here() {
        let native = ToolTarget::native().unwrap();
        let foreign = ToolTarget::from_probe(
            "Linux",
            if native.arch() == "aarch64" {
                "x86_64"
            } else {
                "aarch64"
            },
        )
        .unwrap();
        let binaries = script_binaries(&ManagedTool::ALL);
        let bundle = ToolBundle::new(foreign.clone(), binaries, Some(fixture_python())).unwrap();
        let root = unlocking_tempdir();
        let key = bundle.cache_key().unwrap();
        assert!(matches!(
            bundle.prepare_local_sync(root.path(), &key, None),
            Err(ToolRuntimeError::PythonInstallerUnavailable)
        ));
        let installer = script_tool(ManagedTool::Uv, "1.0.0");
        let prepared = bundle
            .prepare_local_sync(root.path(), &key, Some(&installer))
            .unwrap();
        let request = bundle.python.as_ref().unwrap().request(&foreign).unwrap();
        assert!(
            request.ends_with(&format!("linux-{}-gnu", foreign.arch())),
            "{request}"
        );
        assert_eq!(
            std::fs::read_link(prepared.bin_dir().join("python3")).unwrap(),
            Path::new(&format!(".python/{request}/bin/python3.13"))
        );
    }

    #[test]
    fn python_requests_pin_the_platform_and_the_libc() {
        let python = fixture_python();
        let request =
            |os: &str, arch: &str| python.request(&ToolTarget::from_probe(os, arch).unwrap());
        assert_eq!(
            request("Darwin", "arm64").unwrap(),
            "cpython-3.13.1-macos-aarch64-none"
        );
        assert_eq!(
            request("Linux", "x86_64").unwrap(),
            "cpython-3.13.1-linux-x86_64-gnu"
        );
        assert!(matches!(
            request("Windows_NT", "x86_64"),
            Err(ToolRuntimeError::UnsupportedTarget { .. })
        ));
    }

    #[test]
    fn python_pins_are_validated() {
        let pin = |version: &str, digest: &str| PythonPin {
            version: version.into(),
            target: ToolTarget::native().unwrap(),
            interpreter_sha256: digest.into(),
        };
        let good = "00".repeat(32);
        pin("3.13.18", &good).runtime().unwrap();
        assert!(pin("3.13", &good).runtime().is_err());
        assert!(pin("3.13.18", "abc").runtime().is_err());
        assert!(pin("3.13.18", &"g".repeat(64)).runtime().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn every_catalog_entry_checks_its_own_version() {
        let image = image_tools();
        let catalog = image.catalog();
        let banner = |tool: ManagedTool| match tool {
            ManagedTool::Jq => "jq-1.0.0".to_owned(),
            ManagedTool::Bash => "GNU bash, version 1.0.0(1)-release (fixture)".to_owned(),
            tool => format!("{} 1.0.0", tool.command()),
        };
        let reports = |python: &str| {
            let mut output = String::new();
            for tool in native_tools() {
                output.push_str(&banner(*tool));
                output.push('\0');
            }
            output.push_str(python);
            output.push('\0');
            output
        };
        catalog.verify_versions(&reports("Python 3.13.1")).unwrap();
        for wrong in ["Python 3.12.0", "python 3.13.1", "uv 3.13.1", ""] {
            assert!(
                matches!(
                    catalog.verify_versions(&reports(wrong)),
                    Err(ToolRuntimeError::VersionMismatch(command)) if command == "python3"
                ),
                "{wrong:?}"
            );
        }
    }

    #[test]
    fn prompt_catalog_is_operational_and_stable() {
        let catalog = bundle().catalog();
        let first = catalog.render().unwrap();
        assert_eq!(first, catalog.render().unwrap());
        for command in native_tools() {
            assert!(first.contains(command.command()), "{command:?}");
        }
        if ToolTarget::native().unwrap().provides_python() {
            assert!(first.contains(&format!("python3 {FIXTURE_PYTHON_VERSION}")));
            assert!(first.contains("`.python-version`"));
        }
        assert!(first.contains("ast-grep run"));
        assert!(first.contains("*** Begin Patch"));
        assert!(first.contains("--help"));
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
            _context: heel::IpcContext,
        ) -> impl Future<Output = Result<String, heel::IpcError>> {
            std::future::ready(Ok("runtime_probe_ok".into()))
        }
    }

    #[tokio::test]
    async fn prepared_path_retains_heel_ipc_commands() {
        let cache = unlocking_tempdir();
        let workspace = tempfile::tempdir().unwrap();
        let ready = bundle().prepare_local(cache.path()).await.unwrap();
        let sandbox_path = heel::default_sandbox_path();
        let path = ready.prepend_path(Some(OsStr::new(&sandbox_path))).unwrap();
        let config = heel::SandboxConfig::builder()
            .network(heel::DenyAll)
            .working_dir(
                std::fs::canonicalize(workspace.path())
                    .unwrap_or_else(|_| workspace.path().to_path_buf()),
            )
            .grants([heel::Grant::new(
                heel::ResolvedPath::new(
                    std::fs::canonicalize(ready.bin_dir())
                        .unwrap_or_else(|_| ready.bin_dir().clone()),
                )
                .unwrap(),
                heel::Access::EXEC,
            )])
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

    #[test]
    fn busy_retry_returns_other_errors_after_one_attempt() {
        let mut attempts = 0;
        let error = retry_busy(|| -> std::io::Result<()> {
            attempts += 1;
            Err(std::io::ErrorKind::NotFound.into())
        })
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(attempts, 1);
    }

    #[test]
    fn busy_retry_gives_up_after_the_bound() {
        let mut attempts = 0;
        let error = retry_busy(|| -> std::io::Result<()> {
            attempts += 1;
            Err(std::io::ErrorKind::ExecutableFileBusy.into())
        })
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::ExecutableFileBusy);
        assert_eq!(attempts, BUSY_BACKOFF.len() + 1);
    }

    /// Linux refuses to exec a file while any descriptor open for writing
    /// refers to it, which is the state a forked child leaves behind until it
    /// execs; macOS does not, so this reproduction only runs on Linux.
    #[cfg(target_os = "linux")]
    #[test]
    fn running_an_executable_waits_for_its_write_descriptor_to_close() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tool");
        let mut writer = std::fs::File::create(&path).unwrap();
        writer.write_all(b"#!/bin/sh\necho ready\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();

        // Without the retry the exec fails while the descriptor is open.
        assert_eq!(
            std::process::Command::new(&path)
                .output()
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::ExecutableFileBusy
        );

        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(60));
            drop(writer);
        });
        let output = run_installed(&mut std::process::Command::new(&path)).unwrap();
        release.join().unwrap();
        assert_eq!(output.stdout, b"ready\n");
    }
}
