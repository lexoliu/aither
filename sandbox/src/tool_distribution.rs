use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use futures::{StreamExt, TryStreamExt, stream};
use futures_lite::io::AsyncReadExt;
use serde::{Deserialize, Serialize};
use url::Url;
use zenwave::Client;

use crate::tool_runtime::{
    DistributedTool, HostProgram, HostProgramBinary, ManagedTool, ToolBinary, ToolBundle,
    ToolRuntimeError, ToolTarget, digest,
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
    /// Artifact encoding and executable member.
    pub encoding: ToolArtifactFormat,
}

/// A pinned artifact of a sandbox command.
pub type ToolArtifact = PinnedArtifact<ManagedTool>;

/// A pinned artifact of a host program.
pub type HostProgramArtifact = PinnedArtifact<HostProgram>;

/// Pinned artifact catalog used by every execution backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDistribution {
    artifacts: Vec<ToolArtifact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    host_programs: Vec<HostProgramArtifact>,
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
            resolve_bundled(artifact, root)?;
        }
        distribution.validate()?;
        Ok(distribution)
    }

    fn validate(&self) -> Result<(), ToolRuntimeError> {
        for artifact in &self.artifacts {
            validate_artifact(artifact)?;
        }
        for artifact in &self.host_programs {
            validate_artifact(artifact)?;
        }
        Ok(())
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
            package_artifacts(&self.host_programs, directory, HostProgram::ALL.len()),
        )
        .await?;
        let manifest = directory.join("distribution.json");
        let content = serde_json::to_vec_pretty(&Self {
            artifacts,
            host_programs,
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
        let selected = self
            .artifacts
            .iter()
            .filter(|artifact| &artifact.target == target)
            .cloned()
            .collect::<Vec<_>>();
        for tool in ManagedTool::ALL {
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
        ToolBundle::new(target.clone(), binaries)
    }

    /// Loads one host program for `target`, with the same pinning, download,
    /// verification and cache as the sandbox commands.
    ///
    /// The result has no way into a [`ToolBundle`], so the program never
    /// reaches a sandbox `PATH` or the command catalog.
    ///
    /// # Errors
    /// Fails with [`ToolRuntimeError::MissingHostProgram`] naming the program
    /// and target when the distribution has no artifact for it, and on a
    /// failed download, digest mismatch or unsafe archive member.
    pub async fn load_host_program(
        &self,
        program: HostProgram,
        target: &ToolTarget,
        cache_root: &Path,
    ) -> Result<HostProgramBinary, ToolRuntimeError> {
        let mut selected = self
            .host_programs
            .iter()
            .filter(|artifact| artifact.tool == program && &artifact.target == target);
        let (os, arch) = (target.os().to_owned(), target.arch().to_owned());
        let artifact = selected
            .next()
            .ok_or_else(|| ToolRuntimeError::MissingHostProgram {
                program,
                os: os.clone(),
                arch: arch.clone(),
            })?;
        if selected.next().is_some() {
            return Err(ToolRuntimeError::DuplicateHostProgram { program, os, arch });
        }
        async_fs::create_dir_all(cache_root).await?;
        let bytes = load_artifact(artifact, cache_root).await?;
        HostProgramBinary::verify(
            program,
            target.clone(),
            &artifact.version,
            &digest(&bytes),
            bytes,
        )
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
    let encoded = match read_artifact_file(&path, tool).await {
        Ok(encoded) => encoded,
        Err(ToolRuntimeError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            let encoded = download_artifact(tool, &artifact.url).await?;
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
            read_artifact_file(&path, tool).await?
        }
        Err(error) => return Err(error),
    };
    if digest(&encoded) != expected {
        return Err(ToolRuntimeError::Integrity { tool });
    }
    let encoding = artifact.encoding.clone();
    blocking::unblock(move || extract_executable(&encoding, &encoded)).await
}

async fn read_artifact_file(
    path: &Path,
    tool: DistributedTool,
) -> Result<Vec<u8>, ToolRuntimeError> {
    let file = async_fs::File::open(path).await?;
    let mut bytes = Vec::new();
    file.take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(ToolRuntimeError::ArtifactTooLarge(tool));
    }
    Ok(bytes)
}

async fn download_artifact(
    tool: DistributedTool,
    location: &ToolArtifactLocation,
) -> Result<Vec<u8>, ToolRuntimeError> {
    let url = location.download_url(tool)?;
    let encoded = match url.scheme() {
        "file" => {
            let path = url
                .to_file_path()
                .map_err(|()| ToolRuntimeError::ArtifactSource(tool))?;
            read_artifact_file(&path, tool).await?
        }
        "https" => {
            futures_lite::future::or(
                async {
                    let mut client = zenwave::client();
                    let response = client.get(url.as_str())?.await?;
                    if !response.status().is_success() {
                        return Err(ToolRuntimeError::DownloadStatus(response.status().as_u16()));
                    }
                    let mut bytes = Vec::new();
                    response
                        .into_body()
                        .into_reader()
                        .take(64 * 1024 * 1024 + 1)
                        .read_to_end(&mut bytes)
                        .await?;
                    Ok(bytes)
                },
                async {
                    async_io::Timer::after(Duration::from_secs(60)).await;
                    Err(ToolRuntimeError::Download(zenwave::Error::Timeout))
                },
            )
            .await?
        }
        _ => return Err(ToolRuntimeError::ArtifactSource(tool)),
    };
    if encoded.len() > 64 * 1024 * 1024 {
        return Err(ToolRuntimeError::ArtifactTooLarge(tool));
    }
    Ok(encoded)
}

fn read_executable(reader: impl Read) -> Result<Vec<u8>, ToolRuntimeError> {
    let mut bytes = Vec::new();
    reader.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(ToolRuntimeError::InvalidArchive);
    }
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn packaged_distribution_remains_valid_after_relocation() {
        let source = tempfile::tempdir().unwrap();
        let installation = tempfile::tempdir().unwrap();
        let relocated = tempfile::tempdir().unwrap();
        let bytes = b"verified artifact fixture";
        let binary = source.path().join("tool");
        async_fs::write(&binary, bytes).await.unwrap();
        let distribution = ToolDistribution {
            artifacts: ManagedTool::ALL
                .into_iter()
                .map(|tool| ToolArtifact {
                    tool,
                    version: "1.0.0".into(),
                    target: ToolTarget::native().unwrap(),
                    url: ToolArtifactLocation::Download(Url::from_file_path(&binary).unwrap()),
                    sha256: hex::encode(digest(bytes)),
                    encoding: ToolArtifactFormat::Executable,
                })
                .collect(),
            host_programs: Vec::new(),
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
        async_fs::write(relocated.path().join(name), b"tampered")
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
                encoding: ToolArtifactFormat::Executable,
            }],
            host_programs: Vec::new(),
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
                encoding: ToolArtifactFormat::Executable,
            }],
            host_programs: Vec::new(),
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
        directory: tempfile::TempDir,
        distribution: ToolDistribution,
        executable: &'static [u8],
        archive: PathBuf,
    }

    impl HostFixture {
        async fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let target = ToolTarget::native().unwrap();
            let command = b"baseline command fixture";
            let command_path = directory.path().join("command");
            async_fs::write(&command_path, command).await.unwrap();
            let executable: &'static [u8] = b"host program fixture";
            let archive = directory.path().join("search.tar.xz");
            let encoded = tar_xz("repo-search/search", executable);
            async_fs::write(&archive, &encoded).await.unwrap();
            let distribution = ToolDistribution {
                artifacts: ManagedTool::ALL
                    .into_iter()
                    .map(|tool| ToolArtifact {
                        tool,
                        version: "1.0.0".into(),
                        target: target.clone(),
                        url: ToolArtifactLocation::Download(
                            Url::from_file_path(&command_path).unwrap(),
                        ),
                        sha256: hex::encode(digest(command)),
                        encoding: ToolArtifactFormat::Executable,
                    })
                    .collect(),
                host_programs: vec![HostProgramArtifact {
                    tool: HostProgram::RepoSearch,
                    version: "0.0.3".into(),
                    target,
                    url: ToolArtifactLocation::Download(Url::from_file_path(&archive).unwrap()),
                    sha256: hex::encode(digest(&encoded)),
                    encoding: ToolArtifactFormat::TarXz {
                        executable: "repo-search/search".into(),
                    },
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
        let program = fixture
            .distribution
            .load_host_program(HostProgram::RepoSearch, &target, &cache)
            .await
            .unwrap();
        assert_eq!(program.program(), HostProgram::RepoSearch);
        assert_eq!(program.version().to_string(), "0.0.3");
        let prepared = program
            .prepare_local(&fixture.directory.path().join("host"))
            .await
            .unwrap();
        assert_eq!(std::fs::read(prepared.path()).unwrap(), fixture.executable);
        assert_eq!(prepared.path().file_name().unwrap(), "search");
        // Preparing again reuses the verified copy.
        assert_eq!(
            program
                .prepare_local(&fixture.directory.path().join("host"))
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
        let (mount, container) = bundle
            .prepare_container_mount(&fixture.directory.path().join("container"))
            .await
            .unwrap();
        for tools in [&local, &container] {
            let names = std::fs::read_dir(tools.bin_dir())
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            assert!(!names.iter().any(|name| name == "search"));
            assert!(!prepared.path().starts_with(tools.bin_dir()));
            assert!(
                !tools
                    .prepend_path(None)
                    .unwrap()
                    .to_string_lossy()
                    .contains(prepared.path().parent().unwrap().to_str().unwrap())
            );
        }
        assert!(!mount.host_path.join("bin").join("search").exists());
        assert!(!prepared.path().starts_with(mount.host_path));
    }

    #[tokio::test]
    async fn tampered_host_program_artifact_is_rejected() {
        let fixture = HostFixture::new().await;
        let target = ToolTarget::native().unwrap();
        let cache = fixture.directory.path().join("cache");
        async_fs::write(&fixture.archive, tar_xz("repo-search/search", b"tampered"))
            .await
            .unwrap();
        assert!(matches!(
            fixture
                .distribution
                .load_host_program(HostProgram::RepoSearch, &target, &cache)
                .await,
            Err(ToolRuntimeError::Integrity {
                tool: DistributedTool::Host(HostProgram::RepoSearch)
            })
        ));
    }

    #[tokio::test]
    async fn host_program_without_an_artifact_for_the_target_names_both() {
        let fixture = HostFixture::new().await;
        let error = fixture
            .distribution
            .load_host_program(
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
            .map(|artifact| {
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
        let program = packaged
            .load_host_program(
                HostProgram::RepoSearch,
                &ToolTarget::native().unwrap(),
                &fixture.directory.path().join("cache"),
            )
            .await
            .unwrap();
        let prepared = program
            .prepare_local(&fixture.directory.path().join("host"))
            .await
            .unwrap();
        assert_eq!(std::fs::read(prepared.path()).unwrap(), fixture.executable);
    }
}
