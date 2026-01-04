//! Skill types and frontmatter parsing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::SkillError;

/// A loaded skill with parsed instructions and metadata.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique name of the skill.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Trigger phrases that activate this skill.
    pub triggers: Vec<String>,
    /// The markdown instructions (everything after frontmatter).
    pub instructions: String,
    /// Optional list of allowed tools (None = all tools allowed).
    pub allowed_tools: Option<Vec<String>>,
    /// Additional resources loaded from the skill directory.
    pub resources: HashMap<String, String>,
}

/// YAML frontmatter parsed from SKILL.md files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillFrontmatter {
    /// Unique name of the skill.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Trigger phrases that activate this skill.
    #[serde(default)]
    pub triggers: Vec<String>,
    /// Optional list of allowed tools.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

impl Skill {
    /// Parse a skill from SKILL.md content.
    ///
    /// The file format is:
    /// ```markdown
    /// ---
    /// name: skill-name
    /// description: What this skill does
    /// triggers:
    ///   - "trigger phrase"
    /// tools:
    ///   - Read
    ///   - Grep
    /// ---
    ///
    /// # Instructions
    ///
    /// The rest of the file is markdown instructions...
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `SkillError::MissingFrontmatter` if the content doesn't have
    /// valid YAML frontmatter, or `SkillError::ParseFrontmatter` if the YAML
    /// cannot be parsed.
    pub fn parse(content: &str) -> Result<Self, SkillError> {
        let content = content.trim();

        // Must start with ---
        if !content.starts_with("---") {
            return Err(SkillError::MissingFrontmatter);
        }

        // Find the closing ---
        let after_first = &content[3..];
        let end_idx = after_first
            .find("\n---")
            .ok_or(SkillError::MissingFrontmatter)?;

        let yaml_content = &after_first[..end_idx];
        let instructions = after_first[end_idx + 4..].trim().to_string();

        let frontmatter: SkillFrontmatter = serde_yaml::from_str(yaml_content)?;

        Ok(Self {
            name: frontmatter.name,
            description: frontmatter.description,
            triggers: frontmatter.triggers,
            instructions,
            allowed_tools: frontmatter.tools,
            resources: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_skill() {
        let content = r#"---
name: code-review
description: Reviews code for security and best practices
triggers:
  - "review"
  - "security audit"
tools:
  - Read
  - Grep
---

# Code Review Skill

When reviewing code, follow these steps:
1. First scan for security vulnerabilities
2. Check for common anti-patterns
"#;

        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name, "code-review");
        assert_eq!(
            skill.description,
            "Reviews code for security and best practices"
        );
        assert_eq!(skill.triggers, vec!["review", "security audit"]);
        assert_eq!(
            skill.allowed_tools,
            Some(vec!["Read".to_string(), "Grep".to_string()])
        );
        assert!(skill.instructions.contains("Code Review Skill"));
        assert!(skill.instructions.contains("security vulnerabilities"));
    }

    #[test]
    fn test_parse_skill_no_tools() {
        let content = r#"---
name: simple
description: A simple skill
---

Instructions here.
"#;

        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name, "simple");
        assert!(skill.triggers.is_empty());
        assert!(skill.allowed_tools.is_none());
    }

    #[test]
    fn test_parse_missing_frontmatter() {
        let content = "# No Frontmatter\n\nJust content.";
        let result = Skill::parse(content);
        assert!(matches!(result, Err(SkillError::MissingFrontmatter)));
    }

    #[test]
    fn test_parse_unclosed_frontmatter() {
        let content = "---\nname: test\n\nNo closing delimiter";
        let result = Skill::parse(content);
        assert!(matches!(result, Err(SkillError::MissingFrontmatter)));
    }
}
