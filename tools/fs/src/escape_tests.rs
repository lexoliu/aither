use std::{
    fs, io,
    path::{Path, PathBuf},
};

use super::{FileSystem, LocalFileSystem};

/// A fresh `<base>/root` plus a sibling `<base>/outside`, removed on drop.
struct Sandbox {
    base: PathBuf,
    root: PathBuf,
    outside: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!("aither-fs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let root = base.join("root");
        let outside = base.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        Self {
            base: fs::canonicalize(&base).unwrap(),
            root: fs::canonicalize(&root).unwrap(),
            outside: fs::canonicalize(&outside).unwrap(),
        }
    }

    fn fs(&self) -> LocalFileSystem {
        LocalFileSystem::new(&self.root).unwrap()
    }

    fn outside_entries(&self) -> Vec<String> {
        fs::read_dir(&self.outside)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn assert_denied(result: io::Result<()>) {
    let err = result.expect_err("path must be rejected");
    assert_eq!(err.kind(), io::ErrorKind::PermissionDenied, "{err}");
}

#[tokio::test]
async fn dotdot_write_is_rejected_and_creates_nothing() {
    let sb = Sandbox::new("dotdot");
    let fs = sb.fs();
    assert_denied(
        fs.write_file(Path::new("../outside/x.txt"), "x".into())
            .await,
    );
    assert_denied(
        fs.write_file(Path::new("../newdir/x.txt"), "x".into())
            .await,
    );
    assert_denied(
        fs.write_file(Path::new("fresh/../../escaped.txt"), "x".into())
            .await,
    );
    assert_denied(fs.create_dir(Path::new("../newdir")).await);
    assert!(sb.outside_entries().is_empty());
    assert!(!sb.base.join("newdir").exists());
    assert!(!sb.base.join("escaped.txt").exists());
    assert!(
        !sb.root.join("fresh").exists(),
        "rejected path left a directory"
    );
}

#[tokio::test]
async fn absolute_path_outside_is_rejected_and_creates_nothing() {
    let sb = Sandbox::new("absolute");
    let fs = sb.fs();
    let target = sb.outside.join("deep/x.txt");
    assert_denied(fs.write_file(&target, "x".into()).await);
    assert_denied(fs.append_file(&target, "x".into()).await);
    assert!(sb.outside_entries().is_empty());
}

#[tokio::test]
async fn absolute_path_inside_is_allowed() {
    let sb = Sandbox::new("absolute-inside");
    let fs = sb.fs();
    fs.write_file(&sb.root.join("a/b.txt"), "ok".into())
        .await
        .unwrap();
    assert_eq!(fs::read_to_string(sb.root.join("a/b.txt")).unwrap(), "ok");
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_directory_write_is_rejected_and_creates_nothing() {
    let sb = Sandbox::new("symlink-dir");
    std::os::unix::fs::symlink(&sb.outside, sb.root.join("link")).unwrap();
    let fs = sb.fs();
    assert_denied(fs.write_file(Path::new("link/new/x.txt"), "x".into()).await);
    assert_denied(fs.append_file(Path::new("link/x.txt"), "x".into()).await);
    assert_denied(fs.create_dir(Path::new("link/sub")).await);
    assert!(sb.outside_entries().is_empty());
}

#[cfg(unix)]
#[tokio::test]
async fn dangling_symlink_leaf_is_rejected_and_creates_nothing() {
    let sb = Sandbox::new("symlink-dangling");
    std::os::unix::fs::symlink(sb.outside.join("target.txt"), sb.root.join("link")).unwrap();
    let fs = sb.fs();
    assert!(fs.write_file(Path::new("link"), "x".into()).await.is_err());
    assert!(fs.append_file(Path::new("link"), "x".into()).await.is_err());
    assert!(sb.outside_entries().is_empty());
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_file_read_is_rejected() {
    let sb = Sandbox::new("symlink-read");
    fs::write(sb.outside.join("secret.txt"), "secret").unwrap();
    std::os::unix::fs::symlink(sb.outside.join("secret.txt"), sb.root.join("link")).unwrap();
    let err = sb.fs().read_file(Path::new("link")).await.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
}

#[tokio::test]
async fn nested_new_paths_inside_root_are_created() {
    let sb = Sandbox::new("inside");
    let fs = sb.fs();
    fs.write_file(Path::new("a/b/./c.txt"), "ok".into())
        .await
        .unwrap();
    fs.create_dir(Path::new("d/e")).await.unwrap();
    assert_eq!(fs::read_to_string(sb.root.join("a/b/c.txt")).unwrap(), "ok");
    assert!(sb.root.join("d/e").is_dir());
}

#[test]
fn glob_cannot_leave_the_root() {
    let sb = Sandbox::new("glob");
    fs::write(sb.outside.join("secret.txt"), "secret").unwrap();
    fs::write(sb.root.join("mine.txt"), "mine").unwrap();
    let fs = sb.fs();
    assert_eq!(fs.glob("*.txt").unwrap(), ["mine.txt"]);
    assert_eq!(
        fs.glob("../outside/*").unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    let absolute = format!("{}/*", sb.outside.display());
    assert_eq!(
        fs.glob(&absolute).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&sb.outside, sb.root.join("link")).unwrap();
        assert_eq!(fs.glob("link/*").unwrap(), Vec::<String>::new());
    }
}
