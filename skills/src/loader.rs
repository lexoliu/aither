//! Skill loader for loading skills from the filesystem.

use std::path::{Path, PathBuf};

use async_fs as afs;
use futures_lite::stream::StreamExt;

use crate::{Skill, SkillError};

/// Loads skills from filesystem directories.
///
/// # Example
///
/// ```rust,ignore
/// let loader = SkillLoader::new()
///     .add_path("~/.aither/skills")
///     .add_path("./.aither/skills");
///
/// let skills = loader.load_all().await?;
/// ```
#[derive(Debug, Default)]
pub struct SkillLoader {
    paths: Vec<PathBuf>,
}

impl SkillLoader {
    /// Create a new skill loader with no paths.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a path to search for skills.
    ///
    /// Each path should be a directory containing skill subdirectories.
    /// The structure expected is:
    /// ```text
    /// path/
    /// └── skill-name/
    ///     └── SKILL.md
    /// ```
    #[must_use]
    pub fn add_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.paths.push(path.into());
        self
    }

    /// Load all skills from all configured paths.
    ///
    /// Skills are loaded from subdirectories of each path.
    /// Each subdirectory must contain a `SKILL.md` file.
    ///
    /// # Errors
    ///
    /// Returns an error if a skill directory cannot be read or parsed.
    pub async fn load_all(&self) -> Result<Vec<Skill>, SkillError> {
        let mut skills = Vec::new();

        for base_path in &self.paths {
            if !path_exists_async(base_path).await? {
                continue;
            }

            let mut entries =
                afs::read_dir(base_path)
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: base_path.clone(),
                        source,
                    })?;

            while let Some(entry) =
                entries
                    .try_next()
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: base_path.clone(),
                        source,
                    })?
            {
                let path = entry.path();
                let file_type = entry
                    .file_type()
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: path.clone(),
                        source,
                    })?;

                if file_type.is_dir() {
                    let skill = Self::load_from_dir(&path).await?;
                    skills.push(skill);
                }
            }
        }

        Ok(skills)
    }

    /// Load a specific skill by name.
    ///
    /// Searches all configured paths for a skill with the given name.
    ///
    /// # Errors
    ///
    /// Returns `SkillError::NotFound` if no skill with the given name exists,
    /// or other errors if the skill cannot be parsed.
    pub async fn load(&self, name: &str) -> Result<Skill, SkillError> {
        for base_path in &self.paths {
            let skill_path = base_path.join(name);
            if path_exists_async(&skill_path).await? {
                return Self::load_from_dir(&skill_path).await;
            }
        }

        Err(SkillError::NotFound {
            name: name.to_string(),
        })
    }

    /// Load a skill from a specific directory.
    async fn load_from_dir(dir: &Path) -> Result<Skill, SkillError> {
        let skill_file = dir.join("SKILL.md");

        if !path_exists_async(&skill_file).await? {
            return Err(SkillError::InvalidStructure {
                path: dir.to_path_buf(),
                reason: "missing SKILL.md file".to_string(),
            });
        }

        let content =
            afs::read_to_string(&skill_file)
                .await
                .map_err(|source| SkillError::ReadFile {
                    path: skill_file,
                    source,
                })?;

        let mut skill = Skill::parse(&content)?;

        // Load additional resources from the directory
        Self::load_resources(dir, &mut skill).await?;

        Ok(skill)
    }

    /// Load additional resources (templates, scripts) from the skill directory.
    async fn load_resources(dir: &Path, skill: &mut Skill) -> Result<(), SkillError> {
        // Load templates
        let templates_dir = dir.join("templates");
        if path_exists_async(&templates_dir).await? {
            Self::load_dir_contents(&templates_dir, "templates/", skill).await?;
        }

        // Load scripts
        let scripts_dir = dir.join("scripts");
        if path_exists_async(&scripts_dir).await? {
            Self::load_dir_contents(&scripts_dir, "scripts/", skill).await?;
        }

        Ok(())
    }

    /// Load all files from a directory into the skill's resources.
    async fn load_dir_contents(
        dir: &Path,
        prefix: &str,
        skill: &mut Skill,
    ) -> Result<(), SkillError> {
        let mut entries = afs::read_dir(dir)
            .await
            .map_err(|source| SkillError::ReadFile {
                path: dir.to_path_buf(),
                source,
            })?;

        while let Some(entry) = entries
            .try_next()
            .await
            .map_err(|source| SkillError::ReadFile {
                path: dir.to_path_buf(),
                source,
            })?
        {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .await
                .map_err(|source| SkillError::ReadFile {
                    path: path.clone(),
                    source,
                })?;

            if !file_type.is_file() {
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let content =
                    afs::read_to_string(&path)
                        .await
                        .map_err(|source| SkillError::ReadFile {
                            path: path.clone(),
                            source,
                        })?;
                let key = format!("{prefix}{name}");
                skill.resources.insert(key, content);
            }
        }

        Ok(())
    }
}

async fn path_exists_async(path: &Path) -> Result<bool, SkillError> {
    match afs::metadata(path).await {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(SkillError::ReadFile {
            path: path.to_path_buf(),
            source: err,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::future::block_on;
    use std::fs;
    use std::path::Path;
    use tempfile::{TempDir, tempdir};

    fn minimal_skill_content(name: &str, description: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n\nInstructions here.\n")
    }

    fn create_skill_dir(base: &Path, name: &str) -> PathBuf {
        let skill_dir = base.join(name);
        fs::create_dir(&skill_dir).expect("create skill directory");
        skill_dir
    }

    fn create_skill_with_content(base: &Path, name: &str, content: &str) -> PathBuf {
        let skill_dir = create_skill_dir(base, name);
        fs::write(skill_dir.join("SKILL.md"), content).expect("write SKILL.md");
        skill_dir
    }

    fn prepare_loader() -> (TempDir, SkillLoader) {
        let dir = tempdir().expect("create temp dir");
        let loader = SkillLoader::new().add_path(dir.path());
        (dir, loader)
    }

    #[test]
    fn test_load_skill_from_dir() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(
            dir.path(),
            "test-skill",
            &minimal_skill_content("test-skill", "A test skill"),
        );

        let skills = block_on(loader.load_all()).expect("load skill");

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "test-skill");
    }

    #[test]
    fn test_load_skill_by_name() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(
            dir.path(),
            "my-skill",
            &minimal_skill_content("my-skill", "My skill"),
        );

        let skill = block_on(loader.load("my-skill")).expect("load skill by name");

        assert_eq!(skill.name, "my-skill");
    }

    #[test]
    fn test_load_skill_not_found() {
        let (_dir, loader) = prepare_loader();
        let result = block_on(loader.load("nonexistent"));

        assert!(matches!(result, Err(SkillError::NotFound { .. })));
    }

    #[test]
    fn test_load_resources() {
        let (dir, loader) = prepare_loader();
        let skill_dir = create_skill_with_content(
            dir.path(),
            "with-resources",
            &minimal_skill_content("with-resources", "Skill with resources"),
        );

        // Create templates directory
        let templates_dir = skill_dir.join("templates");
        fs::create_dir(&templates_dir).expect("create templates dir");
        fs::write(templates_dir.join("review.md"), "# Review Template").expect("write template");

        let skill = block_on(loader.load("with-resources")).expect("load skill with resources");

        assert_eq!(
            skill.resources.get("templates/review.md"),
            Some(&"# Review Template".to_string())
        );
    }

    #[test]
    fn test_load_all_fails_on_missing_skill_file() {
        let (dir, loader) = prepare_loader();
        create_skill_dir(dir.path(), "missing-markdown");

        let result = block_on(loader.load_all());
        assert!(matches!(result, Err(SkillError::InvalidStructure { .. })));
    }

    #[test]
    fn test_load_all_fails_on_malformed_frontmatter() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(dir.path(), "broken", "name: broken");

        let result = block_on(loader.load_all());
        assert!(matches!(result, Err(SkillError::MissingFrontmatter)));
    }

    #[test]
    fn test_load_all_fails_fast_when_any_skill_is_invalid() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(
            dir.path(),
            "valid-skill",
            &minimal_skill_content("valid-skill", "A valid skill"),
        );
        create_skill_dir(dir.path(), "invalid-skill");

        let result = block_on(loader.load_all());
        assert!(matches!(result, Err(SkillError::InvalidStructure { .. })));
    }
}
