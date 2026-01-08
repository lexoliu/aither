//! File-based subagent definitions.
//!
//! Subagents can be defined in markdown files with the format:
//!
//! ```markdown
//! # subagent-id
//!
//! One sentence description.
//!
//! ---
//!
//! System prompt content here...
//! ```
//!
//! The first line after `#` is the subagent ID.
//! The line before `---` is the description.
//! Everything after `---` is the system prompt.

use std::path::Path;

/// A subagent definition loaded from a file.
#[derive(Debug, Clone)]
pub struct SubagentDefinition {
    /// Unique identifier (e.g., "explore", "plan").
    pub id: String,
    /// One-sentence description for the tool listing.
    pub description: String,
    /// System prompt for the subagent.
    pub system_prompt: String,
    /// Maximum iterations (default: 20).
    pub max_iterations: usize,
}

impl SubagentDefinition {
    /// Parse a subagent definition from markdown content.
    ///
    /// Format:
    /// ```markdown
    /// # id
    ///
    /// Description sentence.
    ///
    /// ---
    ///
    /// System prompt...
    /// ```
    pub fn parse(content: &str) -> Option<Self> {
        let content = content.trim();

        // Find the ID (first # heading)
        let id = content
            .lines()
            .find(|l| l.starts_with("# "))?
            .trim_start_matches("# ")
            .trim()
            .to_string();

        // Split by ---
        let parts: Vec<&str> = content.splitn(2, "\n---").collect();
        if parts.len() != 2 {
            return None;
        }

        let header = parts[0];
        let system_prompt = parts[1].trim().to_string();

        // Extract description (non-empty line after the # heading, before ---)
        let description = header
            .lines()
            .skip_while(|l| l.starts_with('#') || l.is_empty())
            .find(|l| !l.is_empty())?
            .trim()
            .to_string();

        Some(Self {
            id,
            description,
            system_prompt,
            max_iterations: 20,
        })
    }

    /// Load a subagent definition from a file.
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Option<Self>> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::parse(&content))
    }

    /// Load all subagent definitions from a directory.
    ///
    /// Looks for `.md` files in the directory.
    pub fn load_from_dir(dir: impl AsRef<Path>) -> std::io::Result<Vec<Self>> {
        let mut definitions = Vec::new();

        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e),
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "md").unwrap_or(false) {
                if let Ok(Some(def)) = Self::from_file(&path) {
                    definitions.push(def);
                }
            }
        }

        Ok(definitions)
    }
}

/// Default subagent definitions embedded in the binary.
pub fn builtin_subagents() -> Vec<SubagentDefinition> {
    let mut defs = Vec::new();

    if let Some(def) = SubagentDefinition::parse(include_str!("prompts/subagents/explore.md")) {
        defs.push(def);
    }
    if let Some(def) = SubagentDefinition::parse(include_str!("prompts/subagents/plan.md")) {
        defs.push(def);
    }

    defs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subagent_definition() {
        let content = r#"# test-agent

This is a test agent.

---

You are a test agent. Do test things.
"#;

        let def = SubagentDefinition::parse(content).unwrap();
        assert_eq!(def.id, "test-agent");
        assert_eq!(def.description, "This is a test agent.");
        assert!(def.system_prompt.contains("You are a test agent"));
    }

    #[test]
    fn builtin_subagents_load() {
        let defs = builtin_subagents();
        assert!(!defs.is_empty(), "should have builtin subagents");

        // Check that explore is present
        let explore = defs.iter().find(|d| d.id == "explore");
        assert!(explore.is_some(), "should have explore subagent");
        let explore = explore.unwrap();
        assert!(!explore.description.is_empty());
        assert!(!explore.system_prompt.is_empty());

        // Check that plan is present
        let plan = defs.iter().find(|d| d.id == "plan");
        assert!(plan.is_some(), "should have plan subagent");
    }
}
