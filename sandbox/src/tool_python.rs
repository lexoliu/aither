//! The `CPython` that the pinned `uv` installs into a prepared tool tree.
//!
//! A prepared tree carries the interpreter beside the tools, under
//! `bin/.python/<request>/`, and exposes it as the relative link
//! `bin/python3`. Keeping the tree inside `bin` means the one executable
//! grant every backend already gives the tools directory covers the whole
//! interpreter. The tree is installed once, by the verified `uv` outside any
//! sandbox, and is read-only afterwards; the shared writable state `uv` needs
//! at run time lives elsewhere (`PreparedTools::uv_dir`).

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::tool_runtime::{ToolRuntimeError, ToolTarget, digest};

/// Name of the command that runs the managed interpreter.
pub const PYTHON_COMMAND: &str = "python3";

/// Directory inside `bin` that holds the installed interpreter trees.
pub const PYTHON_DIRECTORY: &str = ".python";

/// Proxy and certificate settings the installer inherits; everything else,
/// including every `UV_*` setting that could redirect the download, is dropped.
const INSTALLER_NETWORK_ENVIRONMENT: [&str; 10] = [
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
];

/// A manifest pin of the interpreter `uv` installs on one target.
///
/// The version pins what is installed; the digest pins the interpreter
/// executable it yields, which preparation verifies after installation and
/// every reuse verifies again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonPin {
    /// `CPython` version, such as `3.13.18`.
    pub version: String,
    /// Platform the interpreter was built for.
    pub target: ToolTarget,
    /// SHA-256 of the interpreter executable (`bin/python3.<minor>`).
    pub interpreter_sha256: String,
}

/// What a shell script needs to install or check the managed interpreter.
#[derive(Debug)]
pub struct PythonScript {
    /// The `uv python install` request, also the installed directory's name.
    pub(crate) request: String,
    /// Where `bin/python3` points, relative to `bin`.
    pub(crate) link_target: String,
    /// Hex SHA-256 of the interpreter executable.
    pub(crate) interpreter_sha256: String,
    /// Names of the environment variables the installer inherits, space separated.
    pub(crate) network_environment: String,
}

/// The validated interpreter pin of one target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonRuntime {
    version: semver::Version,
    interpreter_sha256: [u8; 32],
}

impl PythonPin {
    pub(crate) fn runtime(&self) -> Result<PythonRuntime, ToolRuntimeError> {
        let mut interpreter_sha256 = [0; 32];
        hex::decode_to_slice(&self.interpreter_sha256, &mut interpreter_sha256)?;
        Ok(PythonRuntime {
            version: self.version.parse()?,
            interpreter_sha256,
        })
    }
}

impl PythonRuntime {
    /// The `CPython` version the pinned `uv` installs.
    #[must_use]
    pub const fn version(&self) -> &semver::Version {
        &self.version
    }

    pub(crate) const fn interpreter_sha256(&self) -> &[u8; 32] {
        &self.interpreter_sha256
    }

    /// The version `python3 --version` reports: `Python 3.13.18`.
    pub(crate) fn reported_version(output: &str) -> Option<semver::Version> {
        let mut words = output.lines().next()?.split_whitespace();
        (words.next()? == "Python")
            .then(|| words.next())
            .flatten()
            .and_then(|version| semver::Version::parse(version).ok())
    }

    /// The `uv python install` request, which also names the installed
    /// directory: `cpython-3.13.18-macos-aarch64-none`.
    ///
    /// Linux requests the glibc build explicitly, so the interpreter digest
    /// does not depend on the libc of the host `uv` happens to run on.
    pub(crate) fn request(&self, target: &ToolTarget) -> Result<String, ToolRuntimeError> {
        let (os, libc) = match target.os() {
            "macos" => ("macos", "none"),
            "linux" => ("linux", "gnu"),
            os => {
                return Err(ToolRuntimeError::UnsupportedTarget {
                    os: os.into(),
                    arch: target.arch().into(),
                });
            }
        };
        Ok(format!(
            "cpython-{}-{os}-{}-{libc}",
            self.version,
            target.arch()
        ))
    }

    /// The interpreter executable's file name: `python3.13`.
    fn interpreter_name(&self) -> String {
        format!("python{}.{}", self.version.major, self.version.minor)
    }

    /// Where `bin/python3` points, relative to `bin`.
    fn link_target(&self, target: &ToolTarget) -> Result<PathBuf, ToolRuntimeError> {
        Ok(Path::new(PYTHON_DIRECTORY)
            .join(self.request(target)?)
            .join("bin")
            .join(self.interpreter_name()))
    }

    /// The values the install and image-verification scripts need.
    pub(crate) fn script(&self, target: &ToolTarget) -> Result<PythonScript, ToolRuntimeError> {
        let request = self.request(target)?;
        let link_target = format!(
            "{PYTHON_DIRECTORY}/{request}/bin/{}",
            self.interpreter_name()
        );
        Ok(PythonScript {
            request,
            link_target,
            interpreter_sha256: hex::encode(self.interpreter_sha256),
            network_environment: INSTALLER_NETWORK_ENVIRONMENT.join(" "),
        })
    }

    /// Installs the interpreter with `installer`, the verified bytes of the
    /// `uv` pinned for the build machine, into `<stage>/bin`.
    ///
    /// The installer runs from a scratch directory that is removed again, with
    /// a cleared environment, no configuration file and no cache. Afterwards
    /// the whole tree is checked, its interpreter digest verified, and the
    /// tree made read-only.
    pub(crate) fn stage(
        &self,
        target: &ToolTarget,
        stage: &Path,
        installer: &[u8],
    ) -> Result<(), ToolRuntimeError> {
        let request = self.request(target)?;
        let scratch = stage.join(".python-install");
        std::fs::create_dir(&scratch)?;
        let uv = scratch.join("uv");
        std::fs::write(&uv, installer)?;
        set_mode(&uv, 0o500)?;
        let install_dir = scratch.join("python");
        std::fs::create_dir(&install_dir)?;
        let output = std::process::Command::new(&uv)
            .args(["python", "install", "--no-bin", "--no-config", "--no-cache"])
            .arg(&request)
            .env_clear()
            .envs(
                INSTALLER_NETWORK_ENVIRONMENT
                    .iter()
                    .filter_map(|name| Some((*name, std::env::var_os(name)?))),
            )
            .env("UV_PYTHON_INSTALL_DIR", &install_dir)
            .output()?;
        if !output.status.success() {
            return Err(ToolRuntimeError::PythonInstall {
                status: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        let unpacked = install_dir.join(&request);
        if !std::fs::symlink_metadata(&unpacked)?.file_type().is_dir() {
            return Err(ToolRuntimeError::InvalidPythonTree(unpacked));
        }
        let bin = stage.join("bin");
        let python = bin.join(PYTHON_DIRECTORY);
        std::fs::create_dir(&python)?;
        let tree = python.join(&request);
        std::fs::rename(&unpacked, &tree)?;
        std::fs::remove_dir_all(&scratch)?;
        verify_tree(&tree)?;
        link_python(&self.link_target(target)?, &bin.join(PYTHON_COMMAND))?;
        lock_tree(&python)?;
        // A locked tree cannot be deleted, so a rejected one is opened again.
        self.verify(target, &bin).inspect_err(|_| {
            let _ = unlock_tree(&python);
        })
    }

    /// Checks that `<bin>/python3` is the expected relative link and that the
    /// interpreter it reaches is a read-only executable with the pinned digest.
    pub(crate) fn verify(&self, target: &ToolTarget, bin: &Path) -> Result<(), ToolRuntimeError> {
        let link = bin.join(PYTHON_COMMAND);
        let expected = self.link_target(target)?;
        let actual = std::fs::read_link(&link).map_err(|error| link_error(&link, error))?;
        if actual != expected {
            return Err(ToolRuntimeError::InvalidCache(link));
        }
        let interpreter = bin.join(&expected);
        let metadata = std::fs::symlink_metadata(&interpreter)
            .map_err(|error| entry_error(&interpreter, error))?;
        if !metadata.file_type().is_file() {
            return Err(ToolRuntimeError::InvalidCache(interpreter));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata.permissions().mode();
            if mode & 0o222 != 0 || mode & 0o111 == 0 {
                return Err(ToolRuntimeError::InvalidCache(interpreter));
            }
        }
        if digest(&std::fs::read(&interpreter)?) != self.interpreter_sha256 {
            return Err(ToolRuntimeError::PythonIntegrity);
        }
        Ok(())
    }
}

fn entry_error(path: &Path, error: std::io::Error) -> ToolRuntimeError {
    match error.kind() {
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory => {
            ToolRuntimeError::InvalidCache(path.to_path_buf())
        }
        _ => ToolRuntimeError::Io(error),
    }
}

fn link_error(path: &Path, error: std::io::Error) -> ToolRuntimeError {
    match error.kind() {
        std::io::ErrorKind::InvalidInput => ToolRuntimeError::InvalidCache(path.to_path_buf()),
        _ => entry_error(path, error),
    }
}

fn link_python(target: &Path, link: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(unix))]
    {
        let _ = (target, link);
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

fn set_mode(path: &Path, mode: u32) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

/// Checks that `root` holds only directories, regular files and relative
/// symlinks that stay inside `root`.
pub fn verify_tree(root: &Path) -> Result<(), ToolRuntimeError> {
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                directories.push(path);
            } else if kind.is_symlink() {
                let target = std::fs::read_link(&path)?;
                let parent = path.parent().unwrap_or(root);
                if !stays_inside(root, parent, &target) {
                    return Err(ToolRuntimeError::InvalidPythonTree(path));
                }
            } else if !kind.is_file() {
                return Err(ToolRuntimeError::InvalidPythonTree(path));
            }
        }
    }
    Ok(())
}

/// Whether the relative link `target`, placed in `parent`, resolves to a path
/// under `root`, judged lexically.
fn stays_inside(root: &Path, parent: &Path, target: &Path) -> bool {
    let Ok(relative) = parent.strip_prefix(root) else {
        return false;
    };
    let mut depth = relative.components().count();
    for component in target.components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir => {
                let Some(shallower) = depth.checked_sub(1) else {
                    return false;
                };
                depth = shallower;
            }
            Component::RootDir | Component::Prefix(_) => return false,
        }
    }
    true
}

/// Removes every write bit under `root`, links excluded.
fn lock_tree(root: &Path) -> Result<(), ToolRuntimeError> {
    change_modes(root, |mode| mode & !0o222)
}

/// Restores the owner's write bit under `root` so the tree can be deleted.
pub fn unlock_tree(root: &Path) -> Result<(), ToolRuntimeError> {
    change_modes(root, |mode| mode | 0o200)
}

#[cfg(unix)]
fn change_modes(root: &Path, change: fn(u32) -> u32) -> Result<(), ToolRuntimeError> {
    use std::os::unix::fs::PermissionsExt;
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        let mode = metadata.permissions().mode();
        if metadata.is_dir() {
            // Directories open up first so their entries can be listed and changed.
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode | 0o700))?;
            for entry in std::fs::read_dir(&path)? {
                pending.push(entry?.path());
            }
        }
        // Changing a child needs only search permission on its parent.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(change(mode)))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn change_modes(_root: &Path, _change: fn(u32) -> u32) -> Result<(), ToolRuntimeError> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_banner_is_parsed() {
        let version = |text| semver::Version::parse(text).unwrap();
        assert_eq!(
            PythonRuntime::reported_version("Python 3.13.18\n"),
            Some(version("3.13.18"))
        );
        for rejected in [
            "",
            "Python\n",
            "python 3.13.18\n",
            "uv 0.13.7\n",
            "Python 3.13\n",
        ] {
            assert_eq!(
                PythonRuntime::reported_version(rejected),
                None,
                "{rejected:?}"
            );
        }
    }

    #[test]
    fn links_must_stay_inside_the_tree() {
        let root = Path::new("/tree");
        let inside =
            |parent: &str, target: &str| stays_inside(root, &root.join(parent), Path::new(target));
        assert!(inside("bin", "python3.13"));
        assert!(inside("share/terminfo/3", "../x/xterm"));
        assert!(inside("bin", "../lib/libpython3.so"));
        assert!(!inside("bin", "../../outside"));
        assert!(!inside("", "../outside"));
        assert!(!inside("bin", "/usr/bin/python3"));
    }
}
