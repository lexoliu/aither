//! Error types for the skills system.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur when working with skills.
#[derive(Debug, Error)]
pub enum SkillError {
    /// Failed to read a skill file.
    #[error("failed to read skill file at {path}: {source}")]
    ReadFile {
        /// Path to the file that couldn't be read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse skill frontmatter.
    #[error("failed to parse skill frontmatter: {0}")]
    ParseFrontmatter(#[from] serde_yaml::Error),

    /// Skill file is missing required frontmatter.
    #[error("skill file missing frontmatter delimiter '---'")]
    MissingFrontmatter,

    /// Skill not found in registry.
    #[error("skill '{name}' not found")]
    NotFound {
        /// Name of the skill that wasn't found.
        name: String,
    },

    /// Invalid skill directory structure.
    #[error("invalid skill directory at {path}: {reason}")]
    InvalidStructure {
        /// Path to the invalid directory.
        path: PathBuf,
        /// Reason why it's invalid.
        reason: String,
    },
}
