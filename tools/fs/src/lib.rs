use std::{
    borrow::Cow,
    fs::{self, OpenOptions},
    path::PathBuf,
};

use aither_core::llm::{Tool, tool::json};
use anyhow::{Context, Result, anyhow, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct FileSystemTool {
    root: PathBuf,
    canonical_root: PathBuf,
    allow_writes: bool,
    name: String,
    description: String,
}

impl FileSystemTool {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::builder(root).allow_writes(true)
    }

    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        Self::builder(root).allow_writes(false)
    }

    fn builder(root: impl Into<PathBuf>) -> Self {
        let mut root = root.into();
        let _ = fs::create_dir_all(&root);
        let canonical_root = root.canonicalize().unwrap_or_else(|_| root.clone());
        root = canonical_root.clone();

        Self {
            root,
            canonical_root,
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

    fn resolve(&self, relative: Option<&str>, create: bool) -> Result<PathBuf> {
        let candidate = match relative {
            Some(path) if !path.is_empty() => self.root.join(path),
            _ => self.root.clone(),
        };

        if create && let Some(parent) = candidate.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Unable to create parent directories for {}",
                    candidate.display()
                )
            })?;
        }

        let canonical = candidate.canonicalize().or_else(|err| {
            if create && !candidate.exists() {
                Ok(candidate.clone())
            } else {
                Err(err)
            }
        })?;

        if !canonical.starts_with(&self.canonical_root) {
            bail!(
                "Access to '{}' is blocked because it escapes the mounted root '{}'",
                canonical.display(),
                self.canonical_root.display()
            );
        }

        Ok(canonical)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum FsOperation {
    Read { path: String },
    Write { path: String, contents: String },
    Append { path: String, contents: String },
    List { path: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
}

impl Tool for FileSystemTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.description.clone())
    }

    type Arguments = FsOperation;

    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        let response = match arguments {
            FsOperation::Read { path } => {
                let target = self.resolve(Some(&path), false)?;
                fs::read_to_string(&target)
                    .with_context(|| format!("failed to read {}", target.display()))?
            }
            FsOperation::Write { path, contents } => {
                self.ensure_writable()?;
                let target = self.resolve(Some(&path), true)?;
                fs::write(&target, contents)
                    .with_context(|| format!("failed to write {}", target.display()))?;
                format!("Wrote {}", target.display())
            }
            FsOperation::Append { path, contents } => {
                self.ensure_writable()?;
                let target = self.resolve(Some(&path), true)?;
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&target)
                    .with_context(|| format!("failed to open {}", target.display()))?;
                use std::io::Write;
                file.write_all(contents.as_bytes())
                    .with_context(|| format!("failed to append {}", target.display()))?;
                format!("Appended {}", target.display())
            }
            FsOperation::List { path } => {
                let target = self.resolve(path.as_deref(), false)?;
                let entries = fs::read_dir(&target)
                    .with_context(|| format!("failed to list {}", target.display()))?
                    .filter_map(|entry| entry.ok())
                    .map(|entry| DirEntry {
                        name: entry.file_name().to_string_lossy().into_owned(),
                        is_dir: entry.path().is_dir(),
                    })
                    .collect::<Vec<_>>();
                json(&entries)
            }
        };

        Ok(response)
    }
}

impl FileSystemTool {
    fn ensure_writable(&self) -> Result<()> {
        if self.allow_writes {
            Ok(())
        } else {
            Err(anyhow!("Filesystem tool is read-only"))
        }
    }
}
