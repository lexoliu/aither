//! Skill loader for loading skills from the filesystem.

use std::fs;
use std::path::{Path, PathBuf};

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
/// let skills = loader.load_all()?;
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
    pub fn load_all(&self) -> Result<Vec<Skill>, SkillError> {
        let mut skills = Vec::new();

        for base_path in &self.paths {
            if !base_path.exists() {
                continue;
            }

            let entries = fs::read_dir(base_path).map_err(|source| SkillError::ReadFile {
                path: base_path.clone(),
                source,
            })?;

            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(skill) = Self::load_from_dir(&path) {
                        skills.push(skill);
                    }
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
    pub fn load(&self, name: &str) -> Result<Skill, SkillError> {
        for base_path in &self.paths {
            let skill_path = base_path.join(name);
            if skill_path.exists() {
                return Self::load_from_dir(&skill_path);
            }
        }

        Err(SkillError::NotFound {
            name: name.to_string(),
        })
    }

    /// Load a skill from a specific directory.
    fn load_from_dir(dir: &Path) -> Result<Skill, SkillError> {
        let skill_file = dir.join("SKILL.md");

        if !skill_file.exists() {
            return Err(SkillError::InvalidStructure {
                path: dir.to_path_buf(),
                reason: "missing SKILL.md file".to_string(),
            });
        }

        let content = fs::read_to_string(&skill_file).map_err(|source| SkillError::ReadFile {
            path: skill_file,
            source,
        })?;

        let mut skill = Skill::parse(&content)?;

        // Load additional resources from the directory
        Self::load_resources(dir, &mut skill)?;

        Ok(skill)
    }

    /// Load additional resources (templates, scripts) from the skill directory.
    fn load_resources(dir: &Path, skill: &mut Skill) -> Result<(), SkillError> {
        // Load templates
        let templates_dir = dir.join("templates");
        if templates_dir.exists() {
            Self::load_dir_contents(&templates_dir, "templates/", skill)?;
        }

        // Load scripts
        let scripts_dir = dir.join("scripts");
        if scripts_dir.exists() {
            Self::load_dir_contents(&scripts_dir, "scripts/", skill)?;
        }

        Ok(())
    }

    /// Load all files from a directory into the skill's resources.
    fn load_dir_contents(
        dir: &Path,
        prefix: &str,
        skill: &mut Skill,
    ) -> Result<(), SkillError> {
        let entries = fs::read_dir(dir).map_err(|source| SkillError::ReadFile {
            path: dir.to_path_buf(),
            source,
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let content =
                        fs::read_to_string(&path).map_err(|source| SkillError::ReadFile {
                            path: path.clone(),
                            source,
                        })?;
                    let key = format!("{prefix}{name}");
                    skill.resources.insert(key, content);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_load_skill_from_dir() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir(&skill_dir).unwrap();

        let skill_content = r#"---
name: test-skill
description: A test skill
triggers:
  - "test"
---

# Test Skill

Instructions here.
"#;

        fs::write(skill_dir.join("SKILL.md"), skill_content).unwrap();

        let loader = SkillLoader::new().add_path(dir.path());
        let skills = loader.load_all().unwrap();

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "test-skill");
    }

    #[test]
    fn test_load_skill_by_name() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("my-skill");
        fs::create_dir(&skill_dir).unwrap();

        let skill_content = r#"---
name: my-skill
description: My skill
---

Content.
"#;

        fs::write(skill_dir.join("SKILL.md"), skill_content).unwrap();

        let loader = SkillLoader::new().add_path(dir.path());
        let skill = loader.load("my-skill").unwrap();

        assert_eq!(skill.name, "my-skill");
    }

    #[test]
    fn test_load_skill_not_found() {
        let dir = tempdir().unwrap();
        let loader = SkillLoader::new().add_path(dir.path());
        let result = loader.load("nonexistent");

        assert!(matches!(result, Err(SkillError::NotFound { .. })));
    }

    #[test]
    fn test_load_resources() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("with-resources");
        fs::create_dir(&skill_dir).unwrap();

        let skill_content = r#"---
name: with-resources
description: Skill with resources
---

Content.
"#;

        fs::write(skill_dir.join("SKILL.md"), skill_content).unwrap();

        // Create templates directory
        let templates_dir = skill_dir.join("templates");
        fs::create_dir(&templates_dir).unwrap();
        fs::write(templates_dir.join("review.md"), "# Review Template").unwrap();

        let loader = SkillLoader::new().add_path(dir.path());
        let skill = loader.load("with-resources").unwrap();

        assert_eq!(skill.resources.get("templates/review.md"), Some(&"# Review Template".to_string()));
    }
}
