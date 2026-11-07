use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use futures::{StreamExt, TryStreamExt, stream};
use futures_lite::io::AsyncReadExt;
use serde::{Deserialize, Serialize};
use url::Url;
use zenwave::Client;

use crate::tool_runtime::{
    DistributedTool, HostProgram, HostProgramPath, ImageTool, ImageTools, ManagedTool, PythonPin,
    PythonRuntime, ToolBinary, ToolBundle, ToolRuntimeError, ToolTarget, check_python, digest,
};

/// A kind of program a manifest artifact can provide.
trait Distributed: Copy + Send + Sync + Into<DistributedTool> {}

impl<P: Copy + Send + Sync + Into<DistributedTool>> Distributed for P {}

/// Largest xz decoder memory a trusted artifact may demand, in KiB.
const XZ_MEMORY_LIMIT_KIB: u32 = 256 * 1024;

/// Encoding of an artifact, with exactly one executable selected from archives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case")]
pub enum ToolArtifactFormat {
    /// Uncompressed executable.
    Executable,
    /// Gzip-compressed tar archive.
    TarGzip {
        /// Regular-file member containing the executable.
        executable: PathBuf,
    },
    /// Xz-compressed tar archive.
    TarXz {
        /// Regular-file member containing the executable.
        executable: PathBuf,
    },
    /// Zip archive.
    Zip {
        /// Regular-file member containing the executable.
        executable: PathBuf,
    },
}

/// A remote artifact URL or a path relative to its packaged manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolArtifactLocation {
    /// A credential-free HTTPS URL or absolute local file URL.
    Download(Url),
    /// A portable artifact path beneath the manifest's installation directory.
    Bundled(PathBuf),
}

impl ToolArtifactLocation {
    const fn download_url(&self, tool: DistributedTool) -> Result<&Url, ToolRuntimeError> {
        match self {
            Self::Download(url) => Ok(url),
            Self::Bundled(_) => Err(ToolRuntimeError::ArtifactSource(tool)),
        }
    }
}

/// A version-pinned artifact from a trusted distribution manifest.
///
/// `P` is the kind of program it provides: a sandbox command
/// ([`ToolArtifact`]) or a host program ([`HostProgramArtifact`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedArtifact<P> {
    /// Program provided by this artifact.
    pub tool: P,
    /// Semantic version of the program.
    pub version: String,
    /// Platform for which the executable was built.
    pub target: ToolTarget,
    /// Absolute local file URL or HTTPS distribution URL.
    pub url: ToolArtifactLocation,
    /// SHA-256 of the complete artifact, before extraction.
    pub sha256: String,
    /// Length in bytes of the complete artifact, before extraction.
    pub size: u64,
    /// Artifact encoding and executable member.
    pub encoding: ToolArtifactFormat,
}

/// A pinned artifact of a sandbox command.
pub type ToolArtifact = PinnedArtifact<ManagedTool>;

/// A pinned artifact of a host program.
///
/// Besides the archive digest it pins the digest of the executable inside, so
/// a prepared copy is verified without reading the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostProgramArtifact {
    /// The pinned archive.
    #[serde(flatten)]
    pub artifact: PinnedArtifact<HostProgram>,
    /// SHA-256 of the extracted executable.
    pub executable_sha256: String,
}

/// Pinned artifact catalog used by every execution backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDistribution {
    artifacts: Vec<ToolArtifact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    host_programs: Vec<HostProgramArtifact>,
    /// The interpreter the pinned `uv` installs, per platform that has one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    python: Vec<PythonPin>,
}

impl ToolDistribution {
    /// Loads a trusted, version-pinned manifest.
    ///
    /// # Errors
    /// Rejects malformed manifests or unsupported source protocols.
    pub fn from_json(json: &[u8]) -> Result<Self, ToolRuntimeError> {
        let distribution: Self = serde_json::from_slice(json)?;
        distribution.validate()?;
        Ok(distribution)
    }

    /// Loads a packaged manifest and resolves contained artifacts before validation.
    ///
    /// # Errors
    /// Rejects unreadable manifests, escaping paths and invalid artifact metadata.
    pub async fn from_path(path: &Path) -> Result<Self, ToolRuntimeError> {
        let path = async_fs::canonicalize(path).await?;
        let root = path.parent().ok_or(ToolRuntimeError::InvalidArchive)?;
        let mut distribution: Self = serde_json::from_slice(&async_fs::read(&path).await?)?;
        for artifact in &mut distribution.artifacts {
            resolve_bundled(artifact, root)?;
        }
        for artifact in &mut distribution.host_programs {
            resolve_bundled(&mut artifact.artifact, root)?;
        }
        distribution.validate()?;
        Ok(distribution)
    }

    fn validate(&self) -> Result<(), ToolRuntimeError> {
        for artifact in &self.artifacts {
            validate_artifact(artifact)?;
        }
        for artifact in &self.host_programs {
            validate_artifact(&artifact.artifact)?;
            hex::decode_to_slice(&artifact.executable_sha256, &mut [0; 32])?;
        }
        for pin in &self.python {
            pin.runtime()?;
        }
        Ok(())
    }

    /// The interpreter pinned for `target`: present exactly on the platforms
    /// that provide `uv`. A pin for a platform that does not is not selected.
    ///
    /// # Errors
    /// Rejects a missing or duplicate pin for a platform that needs one.
    fn select_python(
        &self,
        target: &ToolTarget,
    ) -> Result<Option<PythonRuntime>, ToolRuntimeError> {
        let mut pins = self
            .python
            .iter()
            .filter(|pin| &pin.target == target && target.provides_python());
        let runtime = pins.next().map(PythonPin::runtime).transpose()?;
        if pins.next().is_some() {
            return Err(ToolRuntimeError::DuplicatePython);
        }
        check_python(target, runtime.as_ref())?;
        Ok(runtime)
    }

    /// Packages hash-verified executables and a relocatable manifest for installation.
    ///
    /// # Errors
    /// Fails on invalid artifacts, tampered existing content or manifest write failure.
    pub async fn package_into(&self, directory: &Path) -> Result<PathBuf, ToolRuntimeError> {
        self.validate()?;
        async_fs::create_dir_all(directory.join("artifacts")).await?;
        let (artifacts, host_programs) = futures::future::try_join(
            package_artifacts(&self.artifacts, directory, ManagedTool::ALL.len()),
            package_artifacts(
                &self
                    .host_programs
                    .iter()
                    .map(|program| program.artifact.clone())
                    .collect::<Vec<_>>(),
                directory,
                HostProgram::ALL.len(),
            ),
        )
        .await?;
        // Packaged artifacts are the bare executable, whose digest the
        // manifest pinned; packaging must have found exactly that.
        let host_programs = host_programs
            .into_iter()
            .map(|artifact| {
                if !self.host_programs.iter().any(|pinned| {
                    pinned.artifact.tool == artifact.tool
                        && pinned.artifact.target == artifact.target
                        && pinned.executable_sha256 == artifact.sha256
                }) {
                    return Err(ToolRuntimeError::Integrity {
                        tool: artifact.tool.into(),
                    });
                }
                Ok(HostProgramArtifact {
                    executable_sha256: artifact.sha256.clone(),
                    artifact,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = directory.join("distribution.json");
        let content = serde_json::to_vec_pretty(&Self {
            artifacts,
            host_programs,
            python: self.python.clone(),
        })?;
        let destination = manifest.clone();
        let directory = directory.to_path_buf();
        blocking::unblock(move || {
            use std::io::Write;
            let mut stage = tempfile::NamedTempFile::new_in(directory)?;
            stage.write_all(&content)?;
            stage.persist(destination).map_err(|error| error.error)?;
            Ok::<(), ToolRuntimeError>(())
        })
        .await?;
        Ok(manifest)
    }

    /// The artifacts pinned for `target`, exactly one per tool the target
    /// requires; pins of other tools for it are not selected.
    ///
    /// # Errors
    /// Rejects missing or duplicate required tools for the target.
    fn select_artifacts(&self, target: &ToolTarget) -> Result<Vec<ToolArtifact>, ToolRuntimeError> {
        let required = target.required_tools();
        let selected = self
            .artifacts
            .iter()
            .filter(|artifact| &artifact.target == target && required.contains(&artifact.tool))
            .cloned()
            .collect::<Vec<_>>();
        for &tool in required {
            match selected
                .iter()
                .filter(|artifact| artifact.tool == tool)
                .count()
            {
                0 => return Err(ToolRuntimeError::MissingTool(tool)),
                1 => {}
                _ => return Err(ToolRuntimeError::DuplicateTool(tool)),
            }
        }
        Ok(selected)
    }

    /// The tools a container image must carry for `target`, pinned without bytes.
    ///
    /// Never touches the network or the filesystem: each entry carries the
    /// command, the version it must report and the digest of its pinned
    /// archive, which is all the container backend needs to verify the image.
    ///
    /// # Errors
    /// Rejects missing or duplicate tools for the target, or a manifest pin
    /// whose version or digest is malformed.
    pub fn image_tools(&self, target: &ToolTarget) -> Result<ImageTools, ToolRuntimeError> {
        let tools = self
            .select_artifacts(target)?
            .into_iter()
            .map(|artifact| {
                let mut archive_sha256 = [0; 32];
                hex::decode_to_slice(&artifact.sha256, &mut archive_sha256)?;
                Ok(ImageTool {
                    tool: artifact.tool,
                    version: artifact.version.parse()?,
                    archive_sha256,
                })
            })
            .collect::<Result<Vec<_>, ToolRuntimeError>>()?;
        Ok(ImageTools::new(
            target.clone(),
            tools,
            self.select_python(target)?,
        ))
    }

    /// Stages the executables an image build must place at
    /// [`ImageTools::container_directory`]: `<out>/<image cache key>/bin`,
    /// returning that `<out>/<key>` directory.
    ///
    /// The bundle is downloaded and digest-verified through [`Self::load`]:
    /// downloads are expected on the image-build machine, never on the host
    /// that runs the container.
    ///
    /// # Errors
    /// Rejects missing/duplicate tools, failed downloads, oversized artifacts,
    /// digest mismatch or failed writes.
    ///
    /// # Panics
    /// Panics if the staged directory lacks its required parent.
    pub async fn stage_image_tools(
        &self,
        target: &ToolTarget,
        artifact_cache: &Path,
        out: &Path,
    ) -> Result<PathBuf, ToolRuntimeError> {
        let key = self.image_tools(target)?.cache_key()?;
        let bundle = self.load(target, artifact_cache).await?;
        // The interpreter of an image for another platform is still installed
        // by a `uv` that runs here.
        let installer = if target.provides_python() && *target != ToolTarget::native()? {
            Some(self.load_native_installer(artifact_cache).await?)
        } else {
            None
        };
        let out = out.to_path_buf();
        blocking::unblock(move || {
            let prepared = bundle.prepare_local_sync(&out, &key, installer.as_deref())?;
            Ok(prepared
                .bin_dir()
                .parent()
                .expect("prepared binaries have a parent directory")
                .to_path_buf())
        })
        .await
    }

    /// Loads one complete target bundle concurrently, before shell execution.
    ///
    /// # Errors
    /// Rejects missing/duplicate tools, failed downloads, oversized artifacts,
    /// digest mismatch or unsafe/non-regular archive members.
    pub async fn load(
        &self,
        target: &ToolTarget,
        cache_root: &Path,
    ) -> Result<ToolBundle, ToolRuntimeError> {
        let selected = self.select_artifacts(target)?;
        async_fs::create_dir_all(cache_root).await?;
        let binaries = stream::iter(selected)
            .map(|artifact| {
                let cache_root = cache_root.to_path_buf();
                async move {
                    let bytes = load_artifact(&artifact, &cache_root).await?;
                    ToolBinary::verify(artifact.tool, &artifact.version, &digest(&bytes), bytes)
                }
            })
            .buffer_unordered(ManagedTool::ALL.len())
            .try_collect::<Vec<_>>()
            .await?;
        ToolBundle::new(target.clone(), binaries, self.select_python(target)?)
    }

    /// The bytes of the `uv` pinned for the machine running this build, which
    /// installs the interpreter of an image for another platform.
    async fn load_native_installer(
        &self,
        artifact_cache: &Path,
    ) -> Result<Vec<u8>, ToolRuntimeError> {
        let native = ToolTarget::native()?;
        let artifact = self
            .artifacts
            .iter()
            .find(|artifact| artifact.tool == ManagedTool::Uv && artifact.target == native)
            .ok_or(ToolRuntimeError::MissingTool(ManagedTool::Uv))?;
        async_fs::create_dir_all(artifact_cache).await?;
        load_artifact(artifact, artifact_cache).await
    }

    /// Prepares one host program for the host's platform and returns the path
    /// of its read-only executable, with the same pinning, download and
    /// verification as the sandbox commands.
    ///
    /// Archives are cached under `<cache_root>/artifacts` and executables
    /// under `<cache_root>/bin/<executable sha256>/<command>`. When that
    /// executable already exists with the pinned digest, the archive is not
    /// read. The path has no way into a [`ToolBundle`], so the program never
    /// reaches a sandbox `PATH` or the command catalog.
    ///
    /// # Errors
    /// Fails with [`ToolRuntimeError::MissingHostProgram`] naming the program
    /// and target when the distribution has no artifact for it,
    /// [`ToolRuntimeError::TargetMismatch`] when `target` is not the host's
    /// platform, [`ToolRuntimeError::Integrity`] when the extracted executable
    /// does not have the pinned digest, and on a failed download, archive
    /// digest mismatch, unsafe archive member or I/O failure.
    pub async fn prepare_host_program(
        &self,
        program: HostProgram,
        target: &ToolTarget,
        cache_root: &Path,
    ) -> Result<HostProgramPath, ToolRuntimeError> {
        let mut selected = self
            .host_programs
            .iter()
            .filter(|pinned| pinned.artifact.tool == program && &pinned.artifact.target == target);
        let (os, arch) = (target.os().to_owned(), target.arch().to_owned());
        let pinned = selected
            .next()
            .ok_or_else(|| ToolRuntimeError::MissingHostProgram {
                program,
                os: os.clone(),
                arch: arch.clone(),
            })?;
        if selected.next().is_some() {
            return Err(ToolRuntimeError::DuplicateHostProgram { program, os, arch });
        }
        if *target != ToolTarget::native()? {
            return Err(ToolRuntimeError::TargetMismatch);
        }
        let mut executable_sha256 = [0; 32];
        hex::decode_to_slice(&pinned.executable_sha256, &mut executable_sha256)?;
        let bin = cache_root.join("bin");
        let cached = {
            let (bin, target) = (bin.clone(), target.clone());
            blocking::unblock(move || {
                HostProgramPath::cached(&bin, program, &target, &executable_sha256)
            })
            .await?
        };
        if let Some(path) = cached {
            return Ok(path);
        }
        let artifacts = cache_root.join("artifacts");
        async_fs::create_dir_all(&artifacts).await?;
        let bytes = load_artifact(&pinned.artifact, &artifacts).await?;
        let target = target.clone();
        blocking::unblock(move || {
            if digest(&bytes) != executable_sha256 {
                return Err(ToolRuntimeError::Integrity {
                    tool: program.into(),
                });
            }
            HostProgramPath::write(&bin, program, &target, &executable_sha256, &bytes)
        })
        .await
    }
}

fn resolve_bundled<P: Distributed>(
    artifact: &mut PinnedArtifact<P>,
    root: &Path,
) -> Result<(), ToolRuntimeError> {
    if let ToolArtifactLocation::Bundled(relative) = &artifact.url {
        validate_member(relative)?;
        let url = Url::from_file_path(root.join(relative))
            .map_err(|()| ToolRuntimeError::ArtifactSource(artifact.tool.into()))?;
        artifact.url = ToolArtifactLocation::Download(url);
    }
    Ok(())
}

fn validate_artifact<P: Distributed>(artifact: &PinnedArtifact<P>) -> Result<(), ToolRuntimeError> {
    let tool = artifact.tool.into();
    let url = artifact.url.download_url(tool)?;
    match url.scheme() {
        "https" | "file" => {}
        _ => return Err(ToolRuntimeError::ArtifactSource(tool)),
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ToolRuntimeError::ArtifactSource(tool));
    }
    hex::decode_to_slice(&artifact.sha256, &mut [0; 32])?;
    let _ = semver::Version::parse(&artifact.version)?;
    Ok(())
}

/// Stores every artifact's verified executable beside a relocatable manifest.
async fn package_artifacts<P: Distributed>(
    artifacts: &[PinnedArtifact<P>],
    directory: &Path,
    concurrency: usize,
) -> Result<Vec<PinnedArtifact<P>>, ToolRuntimeError> {
    stream::iter(artifacts.to_vec())
        .map(|mut artifact| {
            let directory = directory.to_path_buf();
            async move {
                let tool = artifact.tool.into();
                let bytes = load_artifact(&artifact, &directory.join("artifacts")).await?;
                let sha256 = digest(&bytes);
                let size = bytes.len() as u64;
                let name = hex::encode(sha256);
                let path = directory.join(&name);
                blocking::unblock(move || {
                    use std::io::Write;
                    if path.try_exists()? {
                        if digest(&std::fs::read(&path)?) != sha256 {
                            return Err(ToolRuntimeError::Integrity { tool });
                        }
                    } else {
                        let mut stage = tempfile::NamedTempFile::new_in(&directory)?;
                        stage.write_all(&bytes)?;
                        match stage.persist_noclobber(&path) {
                            Ok(_) => {}
                            Err(error)
                                if error.error.kind() == std::io::ErrorKind::AlreadyExists =>
                            {
                                if digest(&std::fs::read(&path)?) != sha256 {
                                    return Err(ToolRuntimeError::Integrity { tool });
                                }
                            }
                            Err(error) => return Err(error.error.into()),
                        }
                    }
                    Ok::<(), ToolRuntimeError>(())
                })
                .await?;
                artifact.url = ToolArtifactLocation::Bundled(name.clone().into());
                artifact.sha256 = name;
                artifact.size = size;
                artifact.encoding = ToolArtifactFormat::Executable;
                Ok::<_, ToolRuntimeError>(artifact)
            }
        })
        .buffer_unordered(concurrency)
        .try_collect::<Vec<_>>()
        .await
}

/// Fetches an artifact through the cache and returns its extracted executable.
async fn load_artifact<P: Distributed>(
    artifact: &PinnedArtifact<P>,
    cache_root: &Path,
) -> Result<Vec<u8>, ToolRuntimeError> {
    let tool = artifact.tool.into();
    let mut expected = [0; 32];
    hex::decode_to_slice(&artifact.sha256, &mut expected)?;
    let path = cache_root.join(hex::encode(expected));
    let encoded = match read_artifact_file(&path, tool, artifact.size).await {
        Ok(encoded) => encoded,
        Err(ToolRuntimeError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            let encoded = download_artifact(tool, &artifact.url, artifact.size).await?;
            if digest(&encoded) != expected {
                return Err(ToolRuntimeError::Integrity { tool });
            }
            let destination = path.clone();
            let root = cache_root.to_path_buf();
            blocking::unblock(move || {
                use std::io::Write;
                let mut stage = tempfile::NamedTempFile::new_in(root)?;
                stage.write_all(&encoded)?;
                match stage.persist_noclobber(destination) {
                    Ok(_) => Ok::<(), ToolRuntimeError>(()),
                    Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                    Err(error) => Err(error.error.into()),
                }
            })
            .await?;
            read_artifact_file(&path, tool, artifact.size).await?
        }
        Err(error) => return Err(error),
    };
    if digest(&encoded) != expected {
        return Err(ToolRuntimeError::Integrity { tool });
    }
    let encoding = artifact.encoding.clone();
    blocking::unblock(move || extract_executable(&encoding, &encoded)).await
}

/// Reads a file the manifest pins at `size` bytes, rejecting any other length
/// before the content is hashed.
async fn read_artifact_file(
    path: &Path,
    tool: DistributedTool,
    size: u64,
) -> Result<Vec<u8>, ToolRuntimeError> {
    let file = async_fs::File::open(path).await?;
    let mut bytes = Vec::new();
    file.take(size.saturating_add(1))
        .read_to_end(&mut bytes)
        .await?;
    check_size(tool, size, &bytes)?;
    Ok(bytes)
}

/// Rejects `bytes` unless it is exactly the pinned `size`; a read bounded to
/// `size + 1` bytes that overran reports that longer length.
const fn check_size(
    tool: DistributedTool,
    size: u64,
    bytes: &[u8],
) -> Result<(), ToolRuntimeError> {
    let actual = bytes.len() as u64;
    if actual == size {
        Ok(())
    } else {
        Err(ToolRuntimeError::ArtifactSize {
            tool,
            expected: size,
            actual,
        })
    }
}

async fn download_artifact(
    tool: DistributedTool,
    location: &ToolArtifactLocation,
    size: u64,
) -> Result<Vec<u8>, ToolRuntimeError> {
    let url = location.download_url(tool)?;
    let encoded = match url.scheme() {
        "file" => {
            let path = url
                .to_file_path()
                .map_err(|()| ToolRuntimeError::ArtifactSource(tool))?;
            read_artifact_file(&path, tool, size).await?
        }
        "https" => download_with_retry(url.as_str(), size, DOWNLOAD_RETRY).await?,
        _ => return Err(ToolRuntimeError::ArtifactSource(tool)),
    };
    check_size(tool, size, &encoded)?;
    Ok(encoded)
}

/// Retry schedule for transient artifact download failures.
#[derive(Clone, Copy, Debug)]
struct DownloadRetry {
    /// Total attempts, the first included.
    attempts: u32,
    /// Wait before the second attempt; each later wait doubles.
    initial_backoff: Duration,
    /// Deadline of a single attempt.
    attempt_timeout: Duration,
}

/// Four attempts with 1 s, 2 s and 4 s between them and 60 s per attempt: the
/// worst case is 4 * 60 s + 7 s = 247 s, and the common CI failure (a dropped
/// or timed-out connection setup) recovers within seconds.
const DOWNLOAD_RETRY: DownloadRetry = DownloadRetry {
    attempts: 4,
    initial_backoff: Duration::from_secs(1),
    attempt_timeout: Duration::from_secs(60),
};

/// Fetches `url` once, bounded by the attempt deadline.
async fn download_once(
    url: &str,
    size: u64,
    timeout: Duration,
) -> Result<Vec<u8>, ToolRuntimeError> {
    futures_lite::future::or(
        async {
            let mut client = zenwave::client();
            let response = client.get(url)?.await?;
            if !response.status().is_success() {
                return Err(ToolRuntimeError::DownloadStatus(response.status().as_u16()));
            }
            let mut bytes = Vec::new();
            response
                .into_body()
                .into_reader()
                .take(size.saturating_add(1))
                .read_to_end(&mut bytes)
                .await?;
            Ok(bytes)
        },
        async {
            async_io::Timer::after(timeout).await;
            Err(ToolRuntimeError::Download(zenwave::Error::Timeout))
        },
    )
    .await
}

/// Whether an HTTP status is worth another attempt: a server fault or throttling.
fn is_transient_status(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

/// Whether a failed download attempt may succeed when repeated. Only the
/// attempt's own failures reach this: connection setup, the deadline, a body
/// read cut off mid-transfer, and the HTTP status. A TLS, redirect, request or
/// proxy error is permanent.
fn is_transient(error: &ToolRuntimeError) -> bool {
    match error {
        ToolRuntimeError::DownloadStatus(status) => is_transient_status(*status),
        ToolRuntimeError::Io(_) => true,
        ToolRuntimeError::Download(error) => match error {
            zenwave::Error::Transport(_) | zenwave::Error::Timeout => true,
            zenwave::Error::Http { status, .. } => is_transient_status(status.as_u16()),
            _ => false,
        },
        _ => false,
    }
}

/// Downloads `url`, retrying transient failures with exponential backoff and
/// failing with the last error once the attempts are spent.
async fn download_with_retry(
    url: &str,
    size: u64,
    retry: DownloadRetry,
) -> Result<Vec<u8>, ToolRuntimeError> {
    let mut backoff = retry.initial_backoff;
    let mut attempt = 1;
    loop {
        match download_once(url, size, retry.attempt_timeout).await {
            Err(error) if attempt < retry.attempts && is_transient(&error) => {
                tracing::warn!(
                    %error,
                    attempt,
                    attempts = retry.attempts,
                    retry_in = ?backoff,
                    "tool artifact download failed; retrying"
                );
                async_io::Timer::after(backoff).await;
                backoff *= 2;
                attempt += 1;
            }
            result => return result,
        }
    }
}

/// Reads an archive member in full: the archive's pinned hash fixes its content.
fn read_executable(mut reader: impl Read) -> Result<Vec<u8>, ToolRuntimeError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn validate_member(path: &Path) -> Result<(), ToolRuntimeError> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ToolRuntimeError::InvalidArchive);
    }
    Ok(())
}

/// Reads the one regular member named `executable` from a decompressed tar stream.
fn select_tar_member(reader: impl Read, executable: &Path) -> Result<Vec<u8>, ToolRuntimeError> {
    let mut archive = tar::Archive::new(reader);
    let mut found = None;
    for entry in archive.entries()? {
        let entry = entry?;
        if entry.path()?.as_ref() != executable {
            continue;
        }
        if found.is_some() || !entry.header().entry_type().is_file() {
            return Err(ToolRuntimeError::InvalidArchive);
        }
        found = Some(read_executable(entry)?);
    }
    found.ok_or(ToolRuntimeError::InvalidArchive)
}

fn extract_executable(
    encoding: &ToolArtifactFormat,
    encoded: &[u8],
) -> Result<Vec<u8>, ToolRuntimeError> {
    match encoding {
        ToolArtifactFormat::Executable => Ok(encoded.to_vec()),
        ToolArtifactFormat::TarGzip { executable } => {
            validate_member(executable)?;
            select_tar_member(flate2::read::GzDecoder::new(encoded), executable)
        }
        ToolArtifactFormat::TarXz { executable } => {
            validate_member(executable)?;
            select_tar_member(
                lzma_rust2::XzReader::new_mem_limit(encoded, false, XZ_MEMORY_LIMIT_KIB),
                executable,
            )
        }
        ToolArtifactFormat::Zip { executable } => {
            validate_member(executable)?;
            let name = executable
                .to_str()
                .ok_or(ToolRuntimeError::InvalidArchive)?;
            let mut archive = zip::ZipArchive::new(Cursor::new(encoded))?;
            let mut found = None;
            for index in 0..archive.len() {
                let entry = archive.by_index(index)?;
                if entry.name()? != name {
                    continue;
                }
                if found.is_some() || !entry.is_file() || entry.is_symlink() {
                    return Err(ToolRuntimeError::InvalidArchive);
                }
                found = Some(read_executable(entry)?);
            }
            found.ok_or(ToolRuntimeError::InvalidArchive)
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub use crate::tool_runtime::tests::{
        UnlockingTempDir, fixture_python_pin, native_tools, script_tool, unlocking_tempdir,
    };

    /// A distribution pinning one `script_tool` executable per command for
    /// `target`, each artifact located by `url` — file URLs for a fixture that
    /// `load` can read, unreachable URLs where nothing may be downloaded.
    ///
    /// The files are written under `dir`; keep `dir` alive while `load` runs.
    pub fn script_distribution(
        target: &ToolTarget,
        version: &str,
        dir: &Path,
        url: impl Fn(&Path) -> ToolArtifactLocation,
    ) -> ToolDistribution {
        let artifacts = target
            .required_tools()
            .iter()
            .map(|&tool| {
                let bytes = script_tool(tool, version);
                let path = dir.join(tool.command());
                std::fs::write(&path, &bytes).unwrap();
                ToolArtifact {
                    tool,
                    version: version.into(),
                    target: target.clone(),
                    url: url(&path),
                    sha256: hex::encode(digest(&bytes)),
                    size: bytes.len() as u64,
                    encoding: ToolArtifactFormat::Executable,
                }
            })
            .collect();
        ToolDistribution {
            artifacts,
            host_programs: Vec::new(),
            python: python_pins(target),
        }
    }

    /// The interpreter pins of `target`: one where it provides `python3`.
    pub fn python_pins(target: &ToolTarget) -> Vec<PythonPin> {
        target
            .provides_python()
            .then(|| fixture_python_pin(target))
            .into_iter()
            .collect()
    }

    /// File-backed artifacts for `script_distribution` fixtures that `load` reads.
    pub fn file_artifact(path: &Path) -> ToolArtifactLocation {
        ToolArtifactLocation::Download(Url::from_file_path(path).unwrap())
    }

    /// An artifact URL that can never be fetched, for fixtures that must
    /// prove nothing is downloaded.
    pub fn unreachable_artifact(_path: &Path) -> ToolArtifactLocation {
        ToolArtifactLocation::Download(Url::parse("http://127.0.0.1:9/tool").unwrap())
    }

    /// The repository's pinned third-party tools; `apply_patch`, `aither-sed` and `bash` are added by a release.
    fn pinned_distribution() -> ToolDistribution {
        ToolDistribution::from_json(include_bytes!("../../patch/third-party-tools.json")).unwrap()
    }

    /// Every tool aither builds is named in the release manifest by its
    /// `patch/built-tools.json` entry, so each name must be a `ManagedTool`.
    #[test]
    fn every_built_tool_is_a_managed_tool() {
        let list: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../patch/built-tools.json")).unwrap();
        for entry in list["tools"].as_array().unwrap() {
            let name = entry["tool"].clone();
            serde_json::from_value::<ManagedTool>(name.clone())
                .unwrap_or_else(|error| panic!("{name} is not a managed tool: {error}"));
        }
    }

    #[test]
    fn pinned_tools_cover_the_required_set_of_every_platform() {
        let pinned = pinned_distribution();
        let mut targets = pinned
            .artifacts
            .iter()
            .map(|artifact| artifact.target.clone())
            .collect::<Vec<_>>();
        targets.sort_by_key(|target| (target.os().to_owned(), target.arch().to_owned()));
        targets.dedup();
        assert_eq!(targets.len(), 5);
        for target in targets {
            for &tool in target.required_tools() {
                let count = pinned
                    .artifacts
                    .iter()
                    .filter(|artifact| artifact.target == target && artifact.tool == tool)
                    .count();
                // A release supplies `apply_patch`, `aither-sed` and `bash`.
                let expected = usize::from(!matches!(
                    tool,
                    ManagedTool::ApplyPatch | ManagedTool::Sed | ManagedTool::Bash
                ));
                assert_eq!(
                    count,
                    expected,
                    "{tool:?} for {}/{}",
                    target.os(),
                    target.arch()
                );
            }
            let pinned_coreutils = pinned.artifacts.iter().any(|artifact| {
                artifact.target == target && artifact.tool == ManagedTool::Coreutils
            });
            assert_eq!(pinned_coreutils, target.os() != "windows");
        }
    }

    #[test]
    fn a_windows_target_selects_only_its_portable_pins() {
        let windows = ToolTarget::from_probe("Windows_NT", "x86_64").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut distribution = script_distribution(&windows, "1.0.0", dir.path(), file_artifact);
        // A pin for a tool the platform does not provide is ignored, not required.
        distribution.artifacts.push(ToolArtifact {
            tool: ManagedTool::Coreutils,
            ..distribution.artifacts[0].clone()
        });
        let selected = distribution.select_artifacts(&windows).unwrap();
        assert_eq!(selected.len(), windows.required_tools().len());
        assert!(
            selected
                .iter()
                .all(|artifact| artifact.tool != ManagedTool::Coreutils)
        );
        distribution
            .artifacts
            .retain(|artifact| artifact.tool != ManagedTool::Jq);
        assert!(matches!(
            distribution.select_artifacts(&windows),
            Err(ToolRuntimeError::MissingTool(ManagedTool::Jq))
        ));
    }

    #[test]
    fn the_pinned_uv_and_python_cover_exactly_the_unix_targets() {
        let pinned = pinned_distribution();
        let unix = [
            ("macos", "aarch64"),
            ("linux", "aarch64"),
            ("linux", "x86_64"),
        ];
        for (os, arch) in unix {
            let target = ToolTarget::from_probe(os, arch).unwrap();
            let uv = pinned
                .artifacts
                .iter()
                .filter(|artifact| artifact.target == target && artifact.tool == ManagedTool::Uv)
                .collect::<Vec<_>>();
            assert_eq!(uv.len(), 1, "{os}/{arch}");
            assert_eq!(uv[0].version, "0.13.7");
            let ToolArtifactLocation::Download(url) = &uv[0].url else {
                panic!("pinned artifacts are downloads");
            };
            assert!(
                url.path()
                    .starts_with("/astral-sh/uv/releases/download/0.13.7/"),
                "{os}/{arch}"
            );
            let python = pinned.select_python(&target).unwrap().unwrap();
            assert_eq!(python.version(), &semver::Version::new(3, 13, 18));
            // The pinned request is what `uv python list` offers for the target.
            let request = python.request(&target).unwrap();
            assert!(request.starts_with("cpython-3.13.18-"), "{request}");
            assert!(
                request.ends_with(if os == "macos" { "-none" } else { "-gnu" }),
                "{request}"
            );
        }
        let digests = pinned
            .python
            .iter()
            .map(|pin| pin.interpreter_sha256.clone())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(digests.len(), pinned.python.len());
        assert_eq!(pinned.python.len(), unix.len());
        let windows = ToolTarget::from_probe("Windows_NT", "x86_64").unwrap();
        assert!(pinned.select_python(&windows).unwrap().is_none());
        assert!(
            !pinned
                .artifacts
                .iter()
                .any(|artifact| artifact.tool == ManagedTool::Uv && artifact.target == windows)
        );
    }

    #[test]
    fn a_missing_or_duplicate_python_pin_is_an_error_where_uv_is_provided() {
        let linux = ToolTarget::from_probe("Linux", "x86_64").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut distribution = script_distribution(&linux, "1.0.0", dir.path(), file_artifact);
        distribution.select_python(&linux).unwrap().unwrap();
        distribution.python.push(distribution.python[0].clone());
        assert!(matches!(
            distribution.select_python(&linux),
            Err(ToolRuntimeError::DuplicatePython)
        ));
        distribution.python.clear();
        assert!(matches!(
            distribution.select_python(&linux),
            Err(ToolRuntimeError::MissingPython)
        ));
        // A pin for a platform without `uv` is not selected.
        let windows = ToolTarget::from_probe("Windows_NT", "x86_64").unwrap();
        distribution.python.push(fixture_python_pin(&windows));
        assert!(distribution.select_python(&windows).unwrap().is_none());
    }

    #[test]
    fn a_malformed_python_pin_is_rejected_when_the_manifest_loads() {
        let manifest = |version: &str, digest: &str| {
            format!(
                r#"{{"artifacts":[],"python":[{{"version":"{version}",
                "target":{{"os":"linux","arch":"x86_64"}},"interpreter_sha256":"{digest}"}}]}}"#
            )
        };
        ToolDistribution::from_json(manifest("3.13.18", &"0".repeat(64)).as_bytes()).unwrap();
        for (version, digest) in [
            ("3.13", "0".repeat(64)),
            ("three", "0".repeat(64)),
            ("3.13.18", "0".repeat(63)),
            ("3.13.18", "z".repeat(64)),
        ] {
            assert!(
                ToolDistribution::from_json(manifest(version, &digest).as_bytes()).is_err(),
                "{version} {digest}"
            );
        }
    }

    #[test]
    fn image_tools_carry_the_python_pin_and_catalog_entry() {
        let target = ToolTarget::native().unwrap();
        if !target.provides_python() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let distribution = script_distribution(&target, "1.0.0", dir.path(), file_artifact);
        let image = distribution.image_tools(&target).unwrap();
        assert_eq!(
            image
                .python
                .as_ref()
                .map(|python| python.version().to_string()),
            Some(crate::tool_runtime::tests::FIXTURE_PYTHON_VERSION.to_owned())
        );
        let mut without = distribution;
        without.python[0].version = "3.13.2".into();
        assert_ne!(
            without.image_tools(&target).unwrap().cache_key().unwrap(),
            image.cache_key().unwrap()
        );
    }

    #[tokio::test]
    async fn stage_image_tools_produces_the_directory_layout_the_image_verifies() {
        let files = tempfile::tempdir().unwrap();
        let out = unlocking_tempdir();
        let target = ToolTarget::native().unwrap();
        let distribution = script_distribution(&target, "1.0.0", files.path(), file_artifact);
        let staged = distribution
            .stage_image_tools(&target, &files.path().join("cache"), out.path())
            .await
            .unwrap();
        let image = distribution.image_tools(&target).unwrap();
        assert_eq!(
            staged,
            std::fs::canonicalize(out.path())
                .unwrap()
                .join(image.cache_key().unwrap())
        );
        for &tool in native_tools() {
            assert!(
                staged
                    .join("bin")
                    .join(tool.executable_name(&target))
                    .is_file()
            );
        }
    }

    #[tokio::test]
    async fn packaged_distribution_remains_valid_after_relocation() {
        let source = tempfile::tempdir().unwrap();
        let installation = tempfile::tempdir().unwrap();
        let relocated = tempfile::tempdir().unwrap();
        let bytes = b"verified artifact fixture";
        let binary = source.path().join("tool");
        async_fs::write(&binary, bytes).await.unwrap();
        let distribution = ToolDistribution {
            artifacts: native_tools()
                .iter()
                .map(|&tool| ToolArtifact {
                    tool,
                    version: "1.0.0".into(),
                    target: ToolTarget::native().unwrap(),
                    url: ToolArtifactLocation::Download(Url::from_file_path(&binary).unwrap()),
                    sha256: hex::encode(digest(bytes)),
                    size: bytes.len() as u64,
                    encoding: ToolArtifactFormat::Executable,
                })
                .collect(),
            host_programs: Vec::new(),
            python: python_pins(&ToolTarget::native().unwrap()),
        };
        let manifest = distribution
            .package_into(installation.path())
            .await
            .unwrap();
        async_fs::copy(&manifest, relocated.path().join("distribution.json"))
            .await
            .unwrap();
        let name = hex::encode(digest(bytes));
        async_fs::copy(
            installation.path().join(&name),
            relocated.path().join(&name),
        )
        .await
        .unwrap();
        let loaded = ToolDistribution::from_path(&relocated.path().join("distribution.json"))
            .await
            .unwrap();
        let bundle = loaded
            .load(&ToolTarget::native().unwrap(), &source.path().join("cache"))
            .await
            .unwrap();
        assert_eq!(
            bundle.catalog(),
            distribution
                .load(
                    &ToolTarget::native().unwrap(),
                    &source.path().join("original-cache")
                )
                .await
                .unwrap()
                .catalog()
        );
        async_fs::write(relocated.path().join(name), b"verified artifact fixtur3")
            .await
            .unwrap();
        assert!(matches!(
            loaded
                .load(
                    &ToolTarget::native().unwrap(),
                    &source.path().join("fresh-cache")
                )
                .await,
            Err(ToolRuntimeError::Integrity { .. })
        ));
    }

    #[tokio::test]
    async fn bundled_artifacts_cannot_escape_the_installation() {
        let root = tempfile::tempdir().unwrap();
        let distribution = ToolDistribution {
            artifacts: vec![ToolArtifact {
                tool: ManagedTool::ApplyPatch,
                version: "1.0.0".into(),
                target: ToolTarget::native().unwrap(),
                url: ToolArtifactLocation::Bundled("../outside".into()),
                sha256: "00".repeat(32),
                size: 1,
                encoding: ToolArtifactFormat::Executable,
            }],
            host_programs: Vec::new(),
            python: Vec::new(),
        };
        let path = root.path().join("distribution.json");
        async_fs::write(&path, serde_json::to_vec(&distribution).unwrap())
            .await
            .unwrap();
        assert!(matches!(
            ToolDistribution::from_path(&path).await,
            Err(ToolRuntimeError::InvalidArchive)
        ));
    }

    #[test]
    fn artifact_credentials_are_rejected_without_echoing_them() {
        let manifest = ToolDistribution {
            artifacts: vec![ToolArtifact {
                tool: ManagedTool::Ripgrep,
                version: "1.0.0".into(),
                target: ToolTarget::native().unwrap(),
                url: ToolArtifactLocation::Download(
                    Url::parse("https://user:secret@example.com/tool").unwrap(),
                ),
                sha256: "00".repeat(32),
                size: 1,
                encoding: ToolArtifactFormat::Executable,
            }],
            host_programs: Vec::new(),
            python: Vec::new(),
        };
        let error =
            ToolDistribution::from_json(&serde_json::to_vec(&manifest).unwrap()).unwrap_err();
        assert!(matches!(
            error,
            ToolRuntimeError::ArtifactSource(DistributedTool::Command(ManagedTool::Ripgrep))
        ));
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn rejects_path_traversal_before_archive_processing() {
        let encoding = ToolArtifactFormat::TarGzip {
            executable: "../rg".into(),
        };
        assert!(matches!(
            extract_executable(&encoding, b""),
            Err(ToolRuntimeError::InvalidArchive)
        ));
    }

    #[test]
    fn only_reads_the_regular_selected_member() {
        let mut tar = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(7);
        header.set_mode(0o755);
        header.set_cksum();
        tar.append_data(&mut header, "bin/rg", Cursor::new(b"content"))
            .unwrap();
        let contents = tar.into_inner().unwrap();
        let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut gzip, &contents).unwrap();
        let encoded = gzip.finish().unwrap();
        let encoding = ToolArtifactFormat::TarGzip {
            executable: "bin/rg".into(),
        };
        assert_eq!(extract_executable(&encoding, &encoded).unwrap(), b"content");
    }

    #[test]
    fn rejects_symlink_members() {
        let mut tar = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_size(0);
        header.set_mode(0o755);
        header.set_link_name("/etc/passwd").unwrap();
        header.set_cksum();
        tar.append_data(&mut header, "rg", Cursor::new(b""))
            .unwrap();
        let contents = tar.into_inner().unwrap();
        let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut gzip, &contents).unwrap();
        let encoded = gzip.finish().unwrap();
        let encoding = ToolArtifactFormat::TarGzip {
            executable: "rg".into(),
        };
        assert!(matches!(
            extract_executable(&encoding, &encoded),
            Err(ToolRuntimeError::InvalidArchive)
        ));
    }

    fn tar_xz(member: &str, content: &[u8]) -> Vec<u8> {
        let mut tar = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        tar.append_data(&mut header, member, Cursor::new(content))
            .unwrap();
        let contents = tar.into_inner().unwrap();
        let mut xz =
            lzma_rust2::XzWriter::new(Vec::new(), lzma_rust2::XzOptions::with_preset(1)).unwrap();
        std::io::Write::write_all(&mut xz, &contents).unwrap();
        xz.finish().unwrap()
    }

    #[test]
    fn extracts_the_executable_from_a_tar_xz_artifact() {
        let encoded = tar_xz("repo-search-x/search", b"daemon");
        let encoding = ToolArtifactFormat::TarXz {
            executable: "repo-search-x/search".into(),
        };
        assert_eq!(extract_executable(&encoding, &encoded).unwrap(), b"daemon");
        let missing = ToolArtifactFormat::TarXz {
            executable: "other".into(),
        };
        assert!(matches!(
            extract_executable(&missing, &encoded),
            Err(ToolRuntimeError::InvalidArchive)
        ));
        assert!(extract_executable(&encoding, b"not xz").is_err());
    }

    /// A distribution of fixture sandbox commands plus a tar.xz `search` host program.
    struct HostFixture {
        directory: UnlockingTempDir,
        distribution: ToolDistribution,
        executable: &'static [u8],
        archive: PathBuf,
    }

    impl HostFixture {
        async fn new() -> Self {
            let directory = unlocking_tempdir();
            let target = ToolTarget::native().unwrap();
            let executable: &'static [u8] = b"host program fixture";
            let archive = directory.path().join("search.tar.xz");
            let encoded = tar_xz("repo-search/search", executable);
            async_fs::write(&archive, &encoded).await.unwrap();
            let fixture_tools =
                script_distribution(&target, "1.0.0", directory.path(), file_artifact);
            let distribution = ToolDistribution {
                artifacts: fixture_tools.artifacts,
                python: fixture_tools.python,
                host_programs: vec![HostProgramArtifact {
                    artifact: PinnedArtifact {
                        tool: HostProgram::RepoSearch,
                        version: "0.0.3".into(),
                        target,
                        url: ToolArtifactLocation::Download(Url::from_file_path(&archive).unwrap()),
                        sha256: hex::encode(digest(&encoded)),
                        size: encoded.len() as u64,
                        encoding: ToolArtifactFormat::TarXz {
                            executable: "repo-search/search".into(),
                        },
                    },
                    executable_sha256: hex::encode(digest(executable)),
                }],
            };
            Self {
                directory,
                distribution,
                executable,
                archive,
            }
        }
    }

    #[tokio::test]
    async fn host_program_is_returned_as_a_host_path_and_never_enters_sandbox_tools() {
        let fixture = HostFixture::new().await;
        let target = ToolTarget::native().unwrap();
        let cache = fixture.directory.path().join("cache");
        let host = fixture.directory.path().join("host");
        let prepared = fixture
            .distribution
            .prepare_host_program(HostProgram::RepoSearch, &target, &host)
            .await
            .unwrap();
        assert_eq!(std::fs::read(prepared.path()).unwrap(), fixture.executable);
        assert_eq!(
            prepared.path().file_name().unwrap(),
            HostProgram::RepoSearch.executable_name(&target).as_str()
        );
        // Preparing again reuses the verified copy.
        assert_eq!(
            fixture
                .distribution
                .prepare_host_program(HostProgram::RepoSearch, &target, &host)
                .await
                .unwrap(),
            prepared
        );

        let bundle = fixture.distribution.load(&target, &cache).await.unwrap();
        let without_host_programs = ToolDistribution {
            host_programs: Vec::new(),
            ..fixture.distribution.clone()
        };
        let baseline = without_host_programs
            .load(&target, &fixture.directory.path().join("baseline-cache"))
            .await
            .unwrap();
        assert_eq!(bundle.catalog(), baseline.catalog());
        assert_eq!(
            bundle.catalog().render().unwrap(),
            baseline.catalog().render().unwrap()
        );
        let local = bundle
            .prepare_local(&fixture.directory.path().join("local"))
            .await
            .unwrap();
        let staged = fixture
            .distribution
            .stage_image_tools(&target, &cache, &fixture.directory.path().join("container"))
            .await
            .unwrap();
        let staged_bin = staged.join("bin");
        for bin_dir in [local.bin_dir().clone(), staged_bin.clone()] {
            let names = std::fs::read_dir(&bin_dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            assert!(!names.iter().any(|name| name == "search"));
            assert!(!prepared.path().starts_with(&bin_dir));
        }
        assert!(
            !local
                .prepend_path(None)
                .unwrap()
                .to_string_lossy()
                .contains(prepared.path().parent().unwrap().to_str().unwrap())
        );
        assert!(!staged_bin.join("search").exists());
        assert!(!prepared.path().starts_with(&staged));
    }

    #[tokio::test]
    async fn tampered_host_program_artifact_is_rejected() {
        let fixture = HostFixture::new().await;
        let target = ToolTarget::native().unwrap();
        let cache = fixture.directory.path().join("cache");
        let mut tampered = async_fs::read(&fixture.archive).await.unwrap();
        *tampered.last_mut().unwrap() ^= 1;
        async_fs::write(&fixture.archive, tampered).await.unwrap();
        assert!(matches!(
            fixture
                .distribution
                .prepare_host_program(HostProgram::RepoSearch, &target, &cache)
                .await,
            Err(ToolRuntimeError::Integrity {
                tool: DistributedTool::Host(HostProgram::RepoSearch)
            })
        ));
    }

    #[tokio::test]
    async fn pinned_executable_digest_mismatch_is_rejected() {
        let mut fixture = HostFixture::new().await;
        let pin = hex::encode(digest(b"another executable"));
        fixture.distribution.host_programs[0].executable_sha256 = pin.clone();
        let cache = fixture.directory.path().join("cache");
        assert!(matches!(
            fixture
                .distribution
                .prepare_host_program(
                    HostProgram::RepoSearch,
                    &ToolTarget::native().unwrap(),
                    &cache
                )
                .await,
            Err(ToolRuntimeError::Integrity {
                tool: DistributedTool::Host(HostProgram::RepoSearch)
            })
        ));
        assert!(!cache.join("bin").join(pin).exists());
    }

    #[tokio::test]
    async fn prepared_host_program_does_not_touch_the_archive() {
        let fixture = HostFixture::new().await;
        let target = ToolTarget::native().unwrap();
        let cache = fixture.directory.path().join("cache");
        let first = fixture
            .distribution
            .prepare_host_program(HostProgram::RepoSearch, &target, &cache)
            .await
            .unwrap();
        async_fs::remove_file(&fixture.archive).await.unwrap();
        async_fs::remove_dir_all(cache.join("artifacts"))
            .await
            .unwrap();
        let second = fixture
            .distribution
            .prepare_host_program(HostProgram::RepoSearch, &target, &cache)
            .await
            .unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn tampered_cached_host_program_is_replaced_from_the_archive() {
        let fixture = HostFixture::new().await;
        let target = ToolTarget::native().unwrap();
        let cache = fixture.directory.path().join("cache");
        let prepared = fixture
            .distribution
            .prepare_host_program(HostProgram::RepoSearch, &target, &cache)
            .await
            .unwrap();
        let mut permissions = std::fs::metadata(prepared.path()).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(prepared.path(), permissions).unwrap();
        std::fs::write(prepared.path(), b"tampered").unwrap();
        let repaired = fixture
            .distribution
            .prepare_host_program(HostProgram::RepoSearch, &target, &cache)
            .await
            .unwrap();
        assert_eq!(repaired, prepared);
        assert_eq!(std::fs::read(repaired.path()).unwrap(), fixture.executable);
    }

    #[test]
    fn manifest_rejects_a_missing_or_malformed_executable_digest() {
        let manifest = |entry: &str| {
            format!(
                r#"{{"artifacts":[],"host_programs":[{{"tool":"repo_search","version":"0.0.3",
                "target":{{"os":"linux","arch":"x86_64"}},"url":"https://example.com/search.tar.xz",
                "sha256":"{}","size":1,"encoding":{{"format":"executable"}}{entry}}}]}}"#,
                "0".repeat(64)
            )
        };
        assert!(ToolDistribution::from_json(manifest("").as_bytes()).is_err());
        for malformed in ["abc", &"g".repeat(64), &"0".repeat(63)] {
            let entry = format!(r#","executable_sha256":"{malformed}""#);
            assert!(ToolDistribution::from_json(manifest(&entry).as_bytes()).is_err());
        }
        let entry = format!(r#","executable_sha256":"{}""#, "0".repeat(64));
        ToolDistribution::from_json(manifest(&entry).as_bytes()).unwrap();
    }

    #[tokio::test]
    async fn host_program_without_an_artifact_for_the_target_names_both() {
        let fixture = HostFixture::new().await;
        let error = fixture
            .distribution
            .prepare_host_program(
                HostProgram::RepoSearch,
                &ToolTarget::from_probe("Windows_NT", "arm64").unwrap(),
                &fixture.directory.path().join("cache"),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            ToolRuntimeError::MissingHostProgram {
                program: HostProgram::RepoSearch,
                ..
            }
        ));
        assert_eq!(
            error.to_string(),
            "host program `search` has no artifact for windows/aarch64"
        );
    }

    #[test]
    fn third_party_manifest_pins_repo_search_for_its_published_targets() {
        let distribution =
            ToolDistribution::from_json(include_bytes!("../../patch/third-party-tools.json"))
                .unwrap();
        let mut targets = distribution
            .host_programs
            .iter()
            .map(|pinned| {
                assert_eq!(pinned.executable_sha256.len(), 64);
                let artifact = &pinned.artifact;
                assert_eq!(artifact.tool, HostProgram::RepoSearch);
                assert_eq!(artifact.version, "0.0.3");
                assert!(matches!(
                    artifact.encoding,
                    ToolArtifactFormat::TarXz { .. }
                ));
                (artifact.target.os(), artifact.target.arch())
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        assert_eq!(
            targets,
            [
                ("linux", "aarch64"),
                ("linux", "x86_64"),
                ("macos", "aarch64")
            ]
        );
    }

    #[tokio::test]
    async fn packaged_distribution_keeps_host_programs() {
        let fixture = HostFixture::new().await;
        let installation = tempfile::tempdir().unwrap();
        let manifest = fixture
            .distribution
            .package_into(installation.path())
            .await
            .unwrap();
        let packaged = ToolDistribution::from_path(&manifest).await.unwrap();
        let prepared = packaged
            .prepare_host_program(
                HostProgram::RepoSearch,
                &ToolTarget::native().unwrap(),
                &fixture.directory.path().join("host"),
            )
            .await
            .unwrap();
        assert_eq!(std::fs::read(prepared.path()).unwrap(), fixture.executable);
    }

    type Scripted = Vec<Option<(u16, &'static [u8])>>;

    /// Answers one scripted response per connection, in order (`None` drops the
    /// connection), and resolves to the number of connections served.
    async fn serve(responses: Scripted) -> (String, tokio::task::JoinHandle<usize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/artifact", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let mut served = 0;
            for response in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                served += 1;
                let Some((status, body)) = response else {
                    drop(socket);
                    continue;
                };
                let mut request = [0; 4096];
                let _ = socket.read(&mut request).await.unwrap();
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
                socket.shutdown().await.unwrap();
            }
            served
        });
        (url, handle)
    }

    const FAST_RETRY: DownloadRetry = DownloadRetry {
        attempts: 4,
        initial_backoff: Duration::from_millis(5),
        attempt_timeout: Duration::from_secs(10),
    };

    #[tokio::test]
    async fn dropped_connections_and_server_faults_are_retried_until_the_bytes_arrive() {
        let (url, server) = serve(vec![
            None,
            Some((503, b"busy")),
            Some((429, b"")),
            Some((200, b"payload")),
        ])
        .await;
        let bytes = download_with_retry(&url, 7, FAST_RETRY).await.unwrap();
        assert_eq!(bytes, b"payload");
        assert_eq!(server.await.unwrap(), 4);
    }

    #[tokio::test]
    async fn a_permanent_status_is_tried_exactly_once() {
        let (url, server) = serve(vec![Some((404, b"missing"))]).await;
        let error = download_with_retry(&url, 7, FAST_RETRY).await.unwrap_err();
        assert!(error.to_string().contains("404"), "{error}");
        assert!(!is_transient(&error));
        assert_eq!(server.await.unwrap(), 1);
    }

    #[tokio::test]
    async fn transient_failures_end_after_the_last_attempt_with_the_last_error() {
        let (url, server) = serve(vec![
            None,
            Some((503, b"")),
            Some((503, b"")),
            Some((502, b"")),
        ])
        .await;
        let error = download_with_retry(&url, 7, FAST_RETRY).await.unwrap_err();
        assert!(error.to_string().contains("502"), "{error}");
        assert_eq!(server.await.unwrap(), 4);
    }

    #[tokio::test]
    async fn a_download_longer_than_its_pinned_size_is_rejected_before_hashing() {
        let (url, server) = serve(vec![Some((200, b"payload and more"))]).await;
        let tool = DistributedTool::Command(ManagedTool::Ripgrep);
        // The body read stops one byte past the pin instead of buffering it all.
        let bytes = download_with_retry(&url, 7, FAST_RETRY).await.unwrap();
        assert_eq!(bytes.len(), 8);
        let error = check_size(tool, 7, &bytes).unwrap_err();
        assert!(
            matches!(
                error,
                ToolRuntimeError::ArtifactSize {
                    tool: DistributedTool::Command(ManagedTool::Ripgrep),
                    expected: 7,
                    actual: 8,
                }
            ),
            "{error:?}"
        );
        assert_eq!(server.await.unwrap(), 1);
    }

    #[tokio::test]
    async fn a_cached_file_of_the_wrong_size_is_rejected_before_hashing() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("artifact");
        let tool = DistributedTool::Command(ManagedTool::Ripgrep);
        async_fs::write(&path, b"payload").await.unwrap();
        assert_eq!(
            read_artifact_file(&path, tool, 7).await.unwrap(),
            b"payload"
        );
        for pinned in [6, 8] {
            let error = read_artifact_file(&path, tool, pinned).await.unwrap_err();
            assert!(
                matches!(error, ToolRuntimeError::ArtifactSize { expected, .. } if expected == pinned),
                "{error:?}"
            );
        }
    }

    #[test]
    #[allow(
        clippy::manual_assert_eq,
        reason = "assert_eq! would print 64 MiB of bytes on failure"
    )]
    fn an_executable_over_64_mib_extracts_from_a_hash_verified_archive() {
        let executable = vec![7_u8; 64 * 1024 * 1024 + 1];
        let encoded = tar_xz("repo-search-x/search", &executable);
        let encoding = ToolArtifactFormat::TarXz {
            executable: "repo-search-x/search".into(),
        };
        let extracted = extract_executable(&encoding, &encoded).unwrap();
        assert!(extracted == executable);
    }
}
