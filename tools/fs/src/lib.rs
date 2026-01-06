use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt, fs,
    future::Future,
    io,
    path::{Component, Path, PathBuf},
    sync::{Arc, RwLock},
};

use aither_core::llm::{Tool, tool::json};
use anyhow::{Result, anyhow};
use async_fs::{
    OpenOptions, create_dir_all, read_dir, read_to_string, remove_dir, remove_file, write,
};
use futures_lite::{AsyncWriteExt, StreamExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Abstract filesystem interface for agent tools.
///
/// Implementors can provide virtual or real filesystem access.
pub trait FileSystem: Send + Sync + 'static {
    fn read_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<String>> + Send + 'a;
    fn write_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a;
    fn append_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a;
    fn remove_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a;
    fn list_dir<'a>(
        &'a self,
        dir: &'a Path,
    ) -> impl Future<Output = io::Result<Vec<DirEntry>>> + Send + 'a;
    fn create_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a;
    fn remove_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a;
}

/// Filesystem operation. Set "operation" to one of: read, write, append, delete, list.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum FsOperation {
    /// Read a file's content.
    Read { path: String },
    /// Write content to a file (creates or overwrites).
    Write { path: String, content: String },
    /// Append content to a file.
    Append { path: String, content: String },
    /// Delete a file.
    Delete { path: String },
    /// List directory contents.
    List { path: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
}

#[derive(Clone)]
pub struct FileSystemTool<FS> {
    filesystem: FS,
    allow_writes: bool,
    name: String,
    description: String,
}

impl FileSystemTool<LocalFileSystem> {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::try_new(root).expect("failed to initialize filesystem tool")
    }

    pub fn try_new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let fs = LocalFileSystem::new(root)?;
        Ok(Self::with_filesystem(fs))
    }

    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        Self::try_read_only(root).expect("failed to initialize filesystem tool")
    }

    pub fn try_read_only(root: impl Into<PathBuf>) -> io::Result<Self> {
        let fs = LocalFileSystem::read_only(root)?;
        Ok(Self::with_filesystem(fs).allow_writes(false))
    }
}

impl FileSystemTool<InMemoryFileSystem> {
    pub fn in_memory() -> Self {
        Self::with_filesystem(InMemoryFileSystem::new())
    }
}

impl<FS: FileSystem> FileSystemTool<FS> {
    pub fn with_filesystem(fs: FS) -> Self {
        Self {
            filesystem: fs,
            allow_writes: true,
            name: "filesystem".into(),
            description: "Reads and writes files relative to the mounted workspace.".into(),
        }
    }

    pub fn allow_writes(mut self, allow: bool) -> Self {
        self.allow_writes = allow;
        self
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.allow_writes {
            Ok(())
        } else {
            Err(anyhow!("Filesystem tool is read-only"))
        }
    }
}

impl<FS: FileSystem> Tool for FileSystemTool<FS> {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = FsOperation;

    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        let response = match arguments {
            FsOperation::Read { path } => self
                .filesystem
                .read_file(Path::new(&path))
                .await
                .map_err(anyhow::Error::new)?,
            FsOperation::Write { path, content } => {
                self.ensure_writable()?;
                let bytes = content.len();
                self.filesystem
                    .write_file(Path::new(&path), content)
                    .await
                    .map_err(anyhow::Error::new)?;
                format!("Successfully wrote {bytes} bytes to {path}")
            }
            FsOperation::Append { path, content } => {
                self.ensure_writable()?;
                let bytes = content.len();
                self.filesystem
                    .append_file(Path::new(&path), content)
                    .await
                    .map_err(anyhow::Error::new)?;
                format!("Successfully appended {bytes} bytes to {path}")
            }
            FsOperation::Delete { path } => {
                self.ensure_writable()?;
                self.filesystem
                    .remove_file(Path::new(&path))
                    .await
                    .map_err(anyhow::Error::new)?;
                format!("Deleted {path}")
            }
            FsOperation::List { path } => {
                let listing = self
                    .filesystem
                    .list_dir(path.as_deref().map_or(Path::new(""), Path::new))
                    .await
                    .map_err(anyhow::Error::new)?;
                json(&listing)
            }
        };

        Ok(response)
    }
}

#[derive(Debug, Clone)]
pub struct LocalFileSystem {
    root: PathBuf,
    canonical_root: PathBuf,
    allow_writes: bool,
}

impl LocalFileSystem {
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        Self::build(root, true)
    }

    pub fn read_only(root: impl Into<PathBuf>) -> io::Result<Self> {
        Self::build(root, false)
    }

    fn build(root: impl Into<PathBuf>, allow_writes: bool) -> io::Result<Self> {
        let mut root = root.into();
        if root.as_os_str().is_empty() {
            root = PathBuf::from(".");
        }
        fs::create_dir_all(&root)?;
        let canonical_root = fs::canonicalize(&root)?;
        Ok(Self {
            root: canonical_root.clone(),
            canonical_root,
            allow_writes,
        })
    }

    async fn resolve(&self, relative: &Path, create_parent: bool) -> io::Result<PathBuf> {
        let candidate = if relative.as_os_str().is_empty() {
            self.root.clone()
        } else {
            self.root.join(relative)
        };

        if create_parent && let Some(parent) = candidate.parent() {
            create_dir_all(parent).await?;
        }

        let canonical = if candidate.exists() {
            candidate.canonicalize()?
        } else {
            candidate.clone()
        };

        if !canonical.starts_with(&self.canonical_root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "Access to '{}' escapes filesystem root {}",
                    canonical.display(),
                    self.canonical_root.display()
                ),
            ));
        }

        Ok(canonical)
    }

    fn ensure_writable(&self) -> io::Result<()> {
        if self.allow_writes {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Filesystem is read-only",
            ))
        }
    }
}

impl FileSystem for LocalFileSystem {
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        let target = self.resolve(path, false).await?;
        read_to_string(&target).await
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(path, true).await?;
        write(&target, contents).await
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(path, true).await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target)
            .await?;
        file.write_all(contents.as_bytes()).await
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(path, false).await?;
        remove_file(&target).await
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        let target = self.resolve(dir, false).await?;
        let mut reader = read_dir(&target).await?;
        let mut entries = Vec::new();
        while let Some(entry) = reader.next().await {
            if let Ok(entry) = entry {
                entries.push(DirEntry {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    is_dir: entry.path().is_dir(),
                });
            }
        }
        Ok(entries)
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(dir, true).await?;
        create_dir_all(&target).await
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(dir, false).await?;
        remove_dir(&target).await
    }
}

#[derive(Debug, Clone)]
pub struct InMemoryFileSystem {
    state: Arc<RwLock<MemoryState>>,
}

#[derive(Debug, Default)]
struct MemoryState {
    entries: BTreeMap<PathBuf, MemoryEntry>,
}

#[derive(Debug, Clone)]
enum MemoryEntry {
    File(String),
    Directory,
}

impl InMemoryFileSystem {
    pub fn new() -> Self {
        let mut entries = BTreeMap::new();
        entries.insert(PathBuf::from("/"), MemoryEntry::Directory);
        Self {
            state: Arc::new(RwLock::new(MemoryState { entries })),
        }
    }

    fn normalize(path: &Path) -> io::Result<PathBuf> {
        let mut normalized = PathBuf::from("/");
        for component in path.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(part) => normalized.push(part),
                Component::ParentDir => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "Parent directory traversal is not allowed",
                    ));
                }
                Component::RootDir => {}
                Component::Prefix(_) => {}
            }
        }
        Ok(normalized)
    }

    fn ensure_dir_present(
        entries: &mut BTreeMap<PathBuf, MemoryEntry>,
        dir: &Path,
    ) -> io::Result<()> {
        if !entries.contains_key(dir) {
            entries.insert(dir.to_path_buf(), MemoryEntry::Directory);
        }
        Ok(())
    }

    fn list_children(entries: &BTreeMap<PathBuf, MemoryEntry>, dir: &Path) -> Vec<DirEntry> {
        let mut result = Vec::new();
        for (path, entry) in entries.iter() {
            if let Some(parent) = path.parent()
                && parent == dir
            {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "/".into());
                let is_dir = matches!(entry, MemoryEntry::Directory);
                if name != "/" {
                    result.push(DirEntry { name, is_dir });
                }
            }
        }
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }
}

impl Default for InMemoryFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystem for InMemoryFileSystem {
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        let path = Self::normalize(path)?;
        let guard = self.state.read().unwrap();
        match guard.entries.get(&path) {
            Some(MemoryEntry::File(contents)) => Ok(contents.clone()),
            Some(MemoryEntry::Directory) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot read a directory",
            )),
            None => Err(io::Error::new(io::ErrorKind::NotFound, "File not found")),
        }
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        let path = Self::normalize(path)?;
        let mut guard = self.state.write().unwrap();
        if let Some(parent) = path.parent() {
            Self::ensure_dir_present(&mut guard.entries, parent)?;
        }
        guard.entries.insert(path, MemoryEntry::File(contents));
        Ok(())
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        let path = Self::normalize(path)?;
        let mut guard = self.state.write().unwrap();
        if let Some(entry) = guard.entries.get_mut(&path) {
            match entry {
                MemoryEntry::File(existing) => existing.push_str(&contents),
                MemoryEntry::Directory => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Cannot append to a directory",
                    ));
                }
            }
        } else {
            if let Some(parent) = path.parent() {
                Self::ensure_dir_present(&mut guard.entries, parent)?;
            }
            guard.entries.insert(path, MemoryEntry::File(contents));
        }
        Ok(())
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        let path = Self::normalize(path)?;
        let mut guard = self.state.write().unwrap();
        match guard.entries.remove(&path) {
            Some(MemoryEntry::File(_)) => Ok(()),
            Some(MemoryEntry::Directory) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Use remove_dir for directories",
            )),
            None => Err(io::Error::new(io::ErrorKind::NotFound, "File not found")),
        }
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        let dir = Self::normalize(dir)?;
        let guard = self.state.read().unwrap();
        if !matches!(guard.entries.get(&dir), Some(MemoryEntry::Directory)) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Directory does not exist",
            ));
        }
        Ok(Self::list_children(&guard.entries, &dir))
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        let dir = Self::normalize(dir)?;
        let mut guard = self.state.write().unwrap();
        if let Some(parent) = dir.parent() {
            Self::ensure_dir_present(&mut guard.entries, parent)?;
        }
        guard.entries.insert(dir, MemoryEntry::Directory);
        Ok(())
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        let dir = Self::normalize(dir)?;
        if dir == Path::new("/") {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Cannot remove root directory",
            ));
        }
        let mut guard = self.state.write().unwrap();
        let mut to_remove = Vec::new();
        for path in guard.entries.keys() {
            if path.starts_with(&dir) {
                to_remove.push(path.clone());
            }
        }
        if to_remove.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Directory does not exist",
            ));
        }
        for path in to_remove {
            guard.entries.remove(&path);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct MountFileSystem<FS> {
    mounts: Arc<RwLock<Vec<MountPoint<FS>>>>,
}

#[derive(Clone)]
struct MountPoint<FS> {
    prefix: PathBuf,
    fs: Arc<FS>,
}

impl<FS> MountFileSystem<FS>
where
    FS: FileSystem,
{
    pub fn new() -> Self {
        Self {
            mounts: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn mount(self, prefix: impl Into<PathBuf>, fs: FS) -> Self {
        let mut guard = self.mounts.write().unwrap();
        guard.push(MountPoint {
            prefix: normalize_mount(prefix.into()),
            fs: Arc::new(fs),
        });
        drop(guard);
        self
    }

    fn route(&self, path: &Path) -> io::Result<(Arc<FS>, PathBuf)> {
        let normalized = normalize_mount(path.to_path_buf());
        let guard = self.mounts.read().unwrap();
        let mut best: Option<&MountPoint<_>> = None;
        for mount in guard.iter() {
            if normalized.starts_with(&mount.prefix) {
                match best {
                    Some(current)
                        if current.prefix.components().count()
                            >= mount.prefix.components().count() => {}
                    _ => best = Some(mount),
                }
            }
        }
        let mount =
            best.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "No mount for path"))?;
        let relative = normalized
            .strip_prefix(&mount.prefix)
            .unwrap_or_else(|_| Path::new(""))
            .to_path_buf();
        Ok((Arc::clone(&mount.fs), relative))
    }
}

impl<FS> Default for MountFileSystem<FS>
where
    FS: FileSystem,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<FS> fmt::Debug for MountFileSystem<FS>
where
    FS: FileSystem,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = self.mounts.read().unwrap();
        let prefixes: Vec<PathBuf> = guard.iter().map(|mount| mount.prefix.clone()).collect();
        f.debug_struct("MountFileSystem")
            .field("mounts", &prefixes)
            .finish()
    }
}

impl<FS> FileSystem for MountFileSystem<FS>
where
    FS: FileSystem,
{
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().read_file(&relative).await
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().write_file(&relative, contents).await
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().append_file(&relative, contents).await
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().remove_file(&relative).await
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        let (fs, relative) = self.route(dir)?;
        fs.as_ref().list_dir(&relative).await
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        let (fs, relative) = self.route(dir)?;
        fs.as_ref().create_dir(&relative).await
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        let (fs, relative) = self.route(dir)?;
        fs.as_ref().remove_dir(&relative).await
    }
}

fn normalize_mount(path: PathBuf) -> PathBuf {
    if path.as_os_str().is_empty() {
        return PathBuf::from("/");
    }
    let mut normalized = PathBuf::from("/");
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                normalized.pop();
            }
            Component::RootDir => {}
            Component::Prefix(_) => {}
        }
    }
    normalized
}

#[derive(Debug, Clone, Copy)]
pub struct FilePermissions {
    pub read: bool,
    pub write: bool,
    pub list: bool,
    pub delete: bool,
}

impl FilePermissions {
    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            list: true,
            delete: false,
        }
    }

    pub const fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            list: true,
            delete: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermissionedFileSystem<FS> {
    inner: FS,
    permissions: FilePermissions,
}

impl<FS> PermissionedFileSystem<FS> {
    pub fn new(inner: FS, permissions: FilePermissions) -> Self {
        Self { inner, permissions }
    }

    fn require(&self, allowed: bool, action: &str) -> io::Result<()> {
        if allowed {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Operation '{action}' is not permitted"),
            ))
        }
    }
}

impl<FS> FileSystem for PermissionedFileSystem<FS>
where
    FS: FileSystem,
{
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        self.require(self.permissions.read, "read")?;
        self.inner.read_file(path).await
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        self.require(self.permissions.write, "write")?;
        self.inner.write_file(path, contents).await
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        self.require(self.permissions.write, "append")?;
        self.inner.append_file(path, contents).await
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        self.require(self.permissions.delete, "remove_file")?;
        self.inner.remove_file(path).await
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        self.require(self.permissions.list, "list_dir")?;
        self.inner.list_dir(dir).await
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        self.require(self.permissions.write, "create_dir")?;
        self.inner.create_dir(dir).await
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        self.require(self.permissions.delete, "remove_dir")?;
        self.inner.remove_dir(dir).await
    }
}

#[derive(Debug, Clone)]
pub struct FsHook {
    pub operation: FsHookOperation,
}

#[derive(Debug, Clone)]
pub enum FsHookOperation {
    Read { path: PathBuf },
    Write { path: PathBuf, bytes: usize },
    Append { path: PathBuf, bytes: usize },
    RemoveFile { path: PathBuf },
    List { path: PathBuf },
    CreateDir { path: PathBuf },
    RemoveDir { path: PathBuf },
}

#[derive(Clone)]
pub struct HookedFileSystem<FS> {
    inner: FS,
    hook: Arc<dyn Fn(FsHook) + Send + Sync>,
}

impl<FS> HookedFileSystem<FS> {
    pub fn new(inner: FS, hook: impl Fn(FsHook) + Send + Sync + 'static) -> Self {
        Self {
            inner,
            hook: Arc::new(hook),
        }
    }

    fn emit(&self, operation: FsHookOperation) {
        (self.hook)(FsHook { operation });
    }
}

impl<FS> FileSystem for HookedFileSystem<FS>
where
    FS: FileSystem,
{
    fn read_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<String>> + Send + 'a {
        self.emit(FsHookOperation::Read {
            path: path.to_path_buf(),
        });
        self.inner.read_file(path)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::Write {
            path: path.to_path_buf(),
            bytes: contents.len(),
        });
        self.inner.write_file(path, contents)
    }

    fn append_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::Append {
            path: path.to_path_buf(),
            bytes: contents.len(),
        });
        self.inner.append_file(path, contents)
    }

    fn remove_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::RemoveFile {
            path: path.to_path_buf(),
        });
        self.inner.remove_file(path)
    }

    fn list_dir<'a>(
        &'a self,
        dir: &'a Path,
    ) -> impl Future<Output = io::Result<Vec<DirEntry>>> + Send + 'a {
        self.emit(FsHookOperation::List {
            path: dir.to_path_buf(),
        });
        self.inner.list_dir(dir)
    }

    fn create_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::CreateDir {
            path: dir.to_path_buf(),
        });
        self.inner.create_dir(dir)
    }

    fn remove_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::RemoveDir {
            path: dir.to_path_buf(),
        });
        self.inner.remove_dir(dir)
    }
}
