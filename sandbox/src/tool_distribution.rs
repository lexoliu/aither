use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use futures::{StreamExt, TryStreamExt, stream};
use futures_lite::io::AsyncReadExt;
use serde::{Deserialize, Serialize};
use url::Url;
use zenwave::Client;

use crate::tool_runtime::{
    ManagedTool, ToolBinary, ToolBundle, ToolRuntimeError, ToolTarget, digest,
};

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
    const fn download_url(&self, tool: ManagedTool) -> Result<&Url, ToolRuntimeError> {
        match self {
            Self::Download(url) => Ok(url),
            Self::Bundled(_) => Err(ToolRuntimeError::ArtifactSource(tool)),
        }
    }
}

/// A version-pinned artifact from a trusted distribution manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolArtifact {
    /// Command provided by this artifact.
    pub tool: ManagedTool,
    /// Semantic version of the command.
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

/// Pinned artifact catalog used by every execution backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDistribution {
    artifacts: Vec<ToolArtifact>,
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
            if let ToolArtifactLocation::Bundled(relative) = &artifact.url {
                validate_member(relative)?;
                let url = Url::from_file_path(root.join(relative))
                    .map_err(|()| ToolRuntimeError::ArtifactSource(artifact.tool))?;
                artifact.url = ToolArtifactLocation::Download(url);
            }
        }
        distribution.validate()?;
        Ok(distribution)
    }

    fn validate(&self) -> Result<(), ToolRuntimeError> {
        for artifact in &self.artifacts {
            let url = artifact.url.download_url(artifact.tool)?;
            match url.scheme() {
                "https" | "file" => {}
                _ => return Err(ToolRuntimeError::ArtifactSource(artifact.tool)),
            }
            if !url.username().is_empty() || url.password().is_some() {
                return Err(ToolRuntimeError::ArtifactSource(artifact.tool));
            }
            hex::decode_to_slice(&artifact.sha256, &mut [0; 32])?;
            let _ = semver::Version::parse(&artifact.version)?;
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
        let artifacts = stream::iter(self.artifacts.clone())
            .map(|mut artifact| {
                let directory = directory.to_path_buf();
                async move {
                    let binary = load_artifact(&artifact, &directory.join("artifacts")).await?;
                    let name = hex::encode(binary.sha256);
                    let path = directory.join(&name);
                    let bytes = binary.bytes;
                    blocking::unblock(move || {
                        use std::io::Write;
                        if path.try_exists()? {
                            if digest(&std::fs::read(&path)?) != digest(&bytes) {
                                return Err(ToolRuntimeError::Integrity {
                                    tool: artifact.tool,
                                });
                            }
                        } else {
                            let mut stage = tempfile::NamedTempFile::new_in(&directory)?;
                            stage.write_all(&bytes)?;
                            match stage.persist_noclobber(&path) {
                                Ok(_) => {}
                                Err(error)
                                    if error.error.kind() == std::io::ErrorKind::AlreadyExists =>
                                {
                                    if digest(&std::fs::read(&path)?) != digest(&bytes) {
                                        return Err(ToolRuntimeError::Integrity {
                                            tool: artifact.tool,
                                        });
                                    }
                                }
                                Err(error) => return Err(error.error.into()),
                            }
                        }
                        Ok::<(), ToolRuntimeError>(())
                    })
                    .await?;
                    artifact.url = ToolArtifactLocation::Bundled(name.into());
                    artifact.sha256 = hex::encode(binary.sha256);
                    artifact.encoding = ToolArtifactFormat::Executable;
                    Ok::<_, ToolRuntimeError>(artifact)
                }
            })
            .buffer_unordered(ManagedTool::ALL.len())
            .try_collect::<Vec<_>>()
            .await?;
        let manifest = directory.join("distribution.json");
        let content = serde_json::to_vec_pretty(&Self { artifacts })?;
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
                async move { load_artifact(&artifact, &cache_root).await }
            })
            .buffer_unordered(ManagedTool::ALL.len())
            .try_collect::<Vec<_>>()
            .await?;
        ToolBundle::new(target.clone(), binaries)
    }
}

async fn load_artifact(
    artifact: &ToolArtifact,
    cache_root: &Path,
) -> Result<ToolBinary, ToolRuntimeError> {
    let mut expected = [0; 32];
    hex::decode_to_slice(&artifact.sha256, &mut expected)?;
    let path = cache_root.join(hex::encode(expected));
    let encoded = match read_artifact_file(&path, artifact.tool).await {
        Ok(encoded) => encoded,
        Err(ToolRuntimeError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            let encoded = download_artifact(artifact).await?;
            if digest(&encoded) != expected {
                return Err(ToolRuntimeError::Integrity {
                    tool: artifact.tool,
                });
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
            read_artifact_file(&path, artifact.tool).await?
        }
        Err(error) => return Err(error),
    };
    if digest(&encoded) != expected {
        return Err(ToolRuntimeError::Integrity {
            tool: artifact.tool,
        });
    }
    let artifact = artifact.clone();
    blocking::unblock(move || {
        let bytes = extract_executable(&artifact.encoding, &encoded)?;
        ToolBinary::verify(artifact.tool, &artifact.version, &digest(&bytes), bytes)
    })
    .await
}

async fn read_artifact_file(path: &Path, tool: ManagedTool) -> Result<Vec<u8>, ToolRuntimeError> {
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

async fn download_artifact(artifact: &ToolArtifact) -> Result<Vec<u8>, ToolRuntimeError> {
    let url = artifact.url.download_url(artifact.tool)?;
    let encoded = match url.scheme() {
        "file" => {
            let path = url
                .to_file_path()
                .map_err(|()| ToolRuntimeError::ArtifactSource(artifact.tool))?;
            read_artifact_file(&path, artifact.tool).await?
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
        _ => return Err(ToolRuntimeError::ArtifactSource(artifact.tool)),
    };
    if encoded.len() > 64 * 1024 * 1024 {
        return Err(ToolRuntimeError::ArtifactTooLarge(artifact.tool));
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

fn extract_executable(
    encoding: &ToolArtifactFormat,
    encoded: &[u8],
) -> Result<Vec<u8>, ToolRuntimeError> {
    match encoding {
        ToolArtifactFormat::Executable => Ok(encoded.to_vec()),
        ToolArtifactFormat::TarGzip { executable } => {
            validate_member(executable)?;
            let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(encoded));
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
        };
        let error =
            ToolDistribution::from_json(&serde_json::to_vec(&manifest).unwrap()).unwrap_err();
        assert!(matches!(
            error,
            ToolRuntimeError::ArtifactSource(ManagedTool::Ripgrep)
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
}
