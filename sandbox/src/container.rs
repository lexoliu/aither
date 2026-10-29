//! Typed container runtime specifications shared by host integrations.
//!
//! This module defines type-safe container image, mount, and launch specs so
//! application code can construct container requests without relying on
//! positional arguments or ad-hoc string conventions.

use std::path::{Component, Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_path::{Utf8UnixPath, Utf8UnixPathBuf};

/// An absolute POSIX path inside a container.
///
/// Container paths are joined with `/` whatever the host platform is, so a
/// host [`Path`] can never stand in for one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "String", into = "String")]
#[schemars(with = "String")]
pub struct ContainerPath(Utf8UnixPathBuf);

/// A string that is not an absolute POSIX path.
#[derive(Debug, thiserror::Error)]
#[error("container path must be absolute: {0}")]
pub struct RelativeContainerPath(pub String);

impl ContainerPath {
    /// The container's filesystem root, `/`.
    #[must_use]
    pub fn root() -> Self {
        Self(Utf8UnixPathBuf::from("/"))
    }

    /// Parses an absolute POSIX path.
    ///
    /// # Errors
    /// Returns [`RelativeContainerPath`] when `path` does not start at `/`.
    pub fn new(path: impl Into<String>) -> Result<Self, RelativeContainerPath> {
        let path = Utf8UnixPathBuf::from(path.into());
        if path.is_absolute() {
            Ok(Self(path))
        } else {
            Err(RelativeContainerPath(path.into_string()))
        }
    }

    /// Appends `path` with POSIX separators; an absolute `path` replaces this one,
    /// so the result is always absolute.
    #[must_use]
    pub fn join(&self, path: impl AsRef<str>) -> Self {
        Self(self.0.join(path.as_ref()))
    }

    /// The containing directory, or `None` at the root.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        self.0.parent().map(|parent| Self(parent.to_path_buf()))
    }

    /// The path as POSIX text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The path as a POSIX path.
    #[must_use]
    pub fn as_unix_path(&self) -> &Utf8UnixPath {
        &self.0
    }
}

impl std::fmt::Display for ContainerPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<String> for ContainerPath {
    type Error = RelativeContainerPath;

    fn try_from(path: String) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

impl From<ContainerPath> for String {
    fn from(path: ContainerPath) -> Self {
        path.0.into_string()
    }
}

/// Supported Docker-compatible runtime kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContainerRuntimeKind {
    /// `OrbStack`, a macOS-native Docker-compatible runtime.
    OrbStack,
    /// Podman, the daemonless Docker-compatible runtime.
    Podman,
    /// Docker Engine itself.
    Docker,
}

impl std::fmt::Display for ContainerRuntimeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrbStack => write!(f, "OrbStack"),
            Self::Podman => write!(f, "Podman"),
            Self::Docker => write!(f, "Docker"),
        }
    }
}

/// Runtime selection order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimePreference {
    /// Runtimes to try, most preferred first.
    pub order: Vec<ContainerRuntimeKind>,
}

impl Default for RuntimePreference {
    fn default() -> Self {
        Self {
            order: vec![
                ContainerRuntimeKind::OrbStack,
                ContainerRuntimeKind::Podman,
                ContainerRuntimeKind::Docker,
            ],
        }
    }
}

/// Filesystem mount access policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MountAccess {
    /// The container may read the mount but not write to it.
    ReadOnly,
    /// The container may read and write the mount.
    ReadWrite,
}

impl MountAccess {
    /// Whether this policy forbids writes.
    #[must_use]
    pub const fn read_only(self) -> bool {
        matches!(self, Self::ReadOnly)
    }
}

/// A host path mounted into a container path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MountSpec {
    /// Path on the host to expose.
    pub host_path: PathBuf,
    /// Path inside the container where it appears.
    pub container_path: ContainerPath,
    /// Whether the container may write to it.
    pub access: MountAccess,
}

/// Declarative host/container root pair used to create validated child mounts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MountRoot {
    /// Base directory on the host.
    pub host_root: PathBuf,
    /// Directory the host root maps to inside the container.
    pub container_root: ContainerPath,
}

/// Mount-root composition error.
#[derive(Debug, thiserror::Error)]
pub enum MountRootError {
    /// The relative path was empty, which would mount the root itself.
    #[error("relative path must not be empty")]
    /// Relative path was empty.
    EmptyRelativePath,
    /// An absolute path was given where a path relative to the root was expected.
    #[error("relative path must not be absolute: {0}")]
    /// Relative path was absolute.
    AbsoluteRelativePath(PathBuf),
    /// The path contained `..`, which would escape the mount root.
    #[error("relative path must not contain parent traversal: {0}")]
    /// Relative path contained parent traversal.
    ParentTraversal(PathBuf),
    /// A component is not UTF-8, so it has no container path spelling.
    #[error("relative path must be valid UTF-8: {0}")]
    /// Relative path contained a non-UTF-8 component.
    NonUtf8(PathBuf),
    /// A host path that is not beneath the mount root.
    #[error("{} is not beneath the mount root {}", .path.display(), .root.display())]
    OutsideRoot {
        /// The host path.
        path: PathBuf,
        /// The mount root's host directory.
        root: PathBuf,
    },
}

impl MountRoot {
    /// Pairs a host directory with the container path it maps to.
    #[must_use]
    pub fn new(host_root: impl Into<PathBuf>, container_root: ContainerPath) -> Self {
        Self {
            host_root: host_root.into(),
            container_root,
        }
    }

    /// Derives a mount for a path beneath this root.
    ///
    /// # Errors
    ///
    /// Returns [`MountRootError`] if `relative` is empty, absolute, or contains
    /// a `..` component — any of which would place the mount outside the root —
    /// or if a component is not UTF-8.
    pub fn child(
        &self,
        relative: impl AsRef<Path>,
        access: MountAccess,
    ) -> Result<MountSpec, MountRootError> {
        let relative = relative.as_ref();
        let container_path = Self::container_child(&self.container_root, relative)?;
        let host_path = self.host_root.join(relative);
        Ok(match access {
            MountAccess::ReadOnly => MountSpec::read_only(host_path, container_path),
            MountAccess::ReadWrite => MountSpec::read_write(host_path, container_path),
        })
    }

    /// Where the host path `host`, the root itself or a path beneath it,
    /// appears inside the container.
    ///
    /// # Errors
    ///
    /// Returns [`MountRootError::OutsideRoot`] if `host` is not beneath the
    /// root, and the conditions of [`Self::child`] for what lies beneath it.
    pub fn container_path(&self, host: &Path) -> Result<ContainerPath, MountRootError> {
        let relative =
            host.strip_prefix(&self.host_root)
                .map_err(|_| MountRootError::OutsideRoot {
                    path: host.to_path_buf(),
                    root: self.host_root.clone(),
                })?;
        if relative.as_os_str().is_empty() {
            return Ok(self.container_root.clone());
        }
        Self::container_child(&self.container_root, relative)
    }

    /// Derives a read-only mount beneath this root.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::child`].
    pub fn read_only_child(&self, relative: impl AsRef<Path>) -> Result<MountSpec, MountRootError> {
        self.child(relative, MountAccess::ReadOnly)
    }

    /// Derives a writable mount beneath this root.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::child`].
    pub fn read_write_child(
        &self,
        relative: impl AsRef<Path>,
    ) -> Result<MountSpec, MountRootError> {
        self.child(relative, MountAccess::ReadWrite)
    }

    /// Joins the host-relative `path` onto `root` one component at a time, so
    /// the container side uses POSIX separators whatever the host spells.
    fn container_child(root: &ContainerPath, path: &Path) -> Result<ContainerPath, MountRootError> {
        let mut container_path = root.clone();
        let mut named = false;
        for component in path.components() {
            match component {
                Component::Normal(name) => {
                    let name = name
                        .to_str()
                        .ok_or_else(|| MountRootError::NonUtf8(path.to_path_buf()))?;
                    container_path = container_path.join(name);
                    named = true;
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    return Err(MountRootError::ParentTraversal(path.to_path_buf()));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(MountRootError::AbsoluteRelativePath(path.to_path_buf()));
                }
            }
        }
        if named {
            Ok(container_path)
        } else {
            Err(MountRootError::EmptyRelativePath)
        }
    }
}

impl MountSpec {
    /// Mounts a host path into the container without write access.
    #[must_use]
    pub fn read_only(host_path: impl Into<PathBuf>, container_path: ContainerPath) -> Self {
        Self {
            host_path: host_path.into(),
            container_path,
            access: MountAccess::ReadOnly,
        }
    }

    /// Mounts a host path into the container with write access.
    #[must_use]
    pub fn read_write(host_path: impl Into<PathBuf>, container_path: ContainerPath) -> Self {
        Self {
            host_path: host_path.into(),
            container_path,
            access: MountAccess::ReadWrite,
        }
    }
}

/// Image build specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ContainerImageSpec {
    /// Tag to apply to the built image.
    pub tag: String,
    /// Directory used as the build context.
    pub context_dir: PathBuf,
    /// Dockerfile to build, if not the default one in the context directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dockerfile: Option<PathBuf>,
}

impl ContainerImageSpec {
    /// Describes an image built from `context_dir` and tagged `tag`.
    #[must_use]
    pub fn new(tag: impl Into<String>, context_dir: impl Into<PathBuf>) -> Self {
        Self {
            tag: tag.into(),
            context_dir: context_dir.into(),
            dockerfile: None,
        }
    }

    /// Builds from a specific Dockerfile rather than the context default.
    #[must_use]
    pub fn with_dockerfile(mut self, dockerfile: impl Into<PathBuf>) -> Self {
        self.dockerfile = Some(dockerfile.into());
        self
    }
}

/// Container launch specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ContainerLaunchSpec {
    /// Name to give the running container.
    pub name: String,
    /// Image to run.
    pub image: String,
    /// Directory the container treats as its workspace.
    pub workspace: PathBuf,
    /// Additional host paths to expose.
    #[serde(default)]
    pub mounts: Vec<MountSpec>,
    /// Environment entries, each formatted as `KEY=value`.
    #[serde(default)]
    pub env: Vec<String>,
}

impl ContainerLaunchSpec {
    /// Describes a container to launch from `image` in `workspace`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        image: impl Into<String>,
        workspace: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            workspace: workspace.into(),
            mounts: Vec::new(),
            env: Vec::new(),
        }
    }

    /// Exposes an additional host path inside the container.
    #[must_use]
    pub fn with_mount(mut self, mount: MountSpec) -> Self {
        self.mounts.push(mount);
        self
    }

    /// Adds an environment entry, formatted as `KEY=value`.
    #[must_use]
    pub fn with_env(mut self, value: impl Into<String>) -> Self {
        self.env.push(value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ContainerPath, MountRoot, MountRootError, MountSpec};

    fn container_path(path: &str) -> ContainerPath {
        ContainerPath::new(path).expect("test container paths are absolute")
    }

    #[test]
    fn mount_root_maps_relative_path() {
        let root = MountRoot::new("/host/.may", container_path("/root/.may"));
        let mount = root
            .read_only_child("skills")
            .expect("relative child mount should be valid");
        assert_eq!(
            mount.host_path,
            std::path::PathBuf::from("/host/.may/skills")
        );
        assert_eq!(mount.container_path.as_str(), "/root/.may/skills");
    }

    #[test]
    fn mount_root_joins_nested_children_with_posix_separators() {
        let root = MountRoot::new("/host/.may", container_path("/root/.may"));
        let relative = std::path::Path::new("skills").join("python");
        let mount = root
            .read_write_child(relative)
            .expect("nested child mount should be valid");
        assert_eq!(mount.container_path.as_str(), "/root/.may/skills/python");
    }

    #[test]
    fn mount_root_rejects_parent_traversal() {
        let root = MountRoot::new("/host/.may", container_path("/root/.may"));
        let error = root
            .read_write_child("../secrets")
            .expect_err("parent traversal must fail");
        assert!(matches!(error, MountRootError::ParentTraversal(_)));
    }

    #[test]
    fn mount_root_rejects_empty_and_rooted_children() {
        let root = MountRoot::new("/host/.may", container_path("/root/.may"));
        assert!(matches!(
            root.read_only_child("."),
            Err(MountRootError::EmptyRelativePath)
        ));
        assert!(matches!(
            root.read_only_child("/etc"),
            Err(MountRootError::AbsoluteRelativePath(_))
        ));
    }

    #[test]
    fn container_paths_are_absolute_posix_paths() {
        assert!(ContainerPath::new("opt/aither").is_err());
        let path = ContainerPath::root().join("opt").join("aither/tools");
        assert_eq!(path.as_str(), "/opt/aither/tools");
        assert_eq!(path.parent().unwrap().as_str(), "/opt/aither");
        assert_eq!(path.join("/etc").as_str(), "/etc");
    }

    #[test]
    fn mount_spec_serializes_container_path_as_text_and_rejects_relative() {
        let mount = MountSpec::read_only("/host/tools", container_path("/opt/aither/tools/key"));
        let json = serde_json::to_value(&mount).unwrap();
        assert_eq!(json["container_path"], "/opt/aither/tools/key");
        assert_eq!(serde_json::from_value::<MountSpec>(json).unwrap(), mount);
        let relative = serde_json::json!({
            "host_path": "/host/tools",
            "container_path": "opt/aither/tools",
            "access": "read_only",
        });
        assert!(serde_json::from_value::<MountSpec>(relative).is_err());
    }
}
