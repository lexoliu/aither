//! File-based subagent definitions.
//!
//! Subagents can be defined in markdown files with YAML frontmatter:
//!
//! ```markdown
//! ---
//! name: subagent-id
//! description: One sentence description of when to use this subagent.
//! model: balanced
//! ---
//!
//! # Subagent Title
//!
//! System prompt content here...
//! ```
//!
//! The `name` field is the subagent ID used for tool registration.
//! The `description` field is shown in tool listings.
//! The optional `model` field pins the subagent to a model tier
//! (`advanced`, `balanced` or `fast`); when it is absent and the caller
//! does not select a model either, the fast model routes the prompt.
//! Everything after the frontmatter is the system prompt.

use std::path::Path;

use async_fs as fs;
use futures_lite::stream::StreamExt;
use serde::Deserialize;

use crate::model_group::ModelTier;

const CURRENT_SUBAGENT_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
struct SubagentFrontmatter {
    version: u32,
    name: String,
    description: String,
    #[serde(default)]
    model: Option<ModelTier>,
    #[serde(default = "default_max_iterations")]
    max_iterations: usize,
}

const fn default_max_iterations() -> usize {
    20
}

/// A subagent definition loaded from a file.
#[derive(Debug, Clone)]
pub struct SubagentDefinition {
    /// Unique identifier (e.g., "explore", "plan").
    pub id: String,
    /// One-sentence description for the tool listing.
    pub description: String,
    /// System prompt for the subagent.
    pub system_prompt: String,
    /// Model tier the definition declares, if any.
    ///
    /// A caller-provided model takes precedence; when neither is set the
    /// fast model routes the prompt to a configured tier.
    pub model: Option<ModelTier>,
    /// Maximum iterations (default: 20).
    pub max_iterations: usize,
}

impl SubagentDefinition {
    fn split_frontmatter(content: &str) -> Result<(&str, &str), String> {
        let content = content.trim();
        let Some(after_open) = content.strip_prefix("---") else {
            return Err("subagent definition must start with YAML frontmatter".to_string());
        };
        let after_open = after_open.trim_start_matches(['\r', '\n']);
        let Some(close_idx) = after_open.find("\n---") else {
            return Err("subagent definition frontmatter is not terminated".to_string());
        };
        let frontmatter = &after_open[..close_idx];
        let system_prompt = after_open[close_idx + 4..].trim();
        if system_prompt.is_empty() {
            return Err("subagent definition system prompt must not be empty".to_string());
        }
        Ok((frontmatter, system_prompt))
    }

    fn parse_frontmatter(frontmatter: &str) -> Result<SubagentFrontmatter, String> {
        let parsed: SubagentFrontmatter =
            serde_yaml::from_str(frontmatter).map_err(|error| error.to_string())?;
        if parsed.version != CURRENT_SUBAGENT_VERSION {
            let mut message = String::new();
            message.push_str("unsupported subagent definition version: ");
            message.push_str(parsed.version.to_string().as_str());
            return Err(message);
        }
        if parsed.name.trim().is_empty() {
            return Err("subagent definition name must not be empty".to_string());
        }
        if parsed.description.trim().is_empty() {
            return Err("subagent definition description must not be empty".to_string());
        }
        if parsed.max_iterations == 0 {
            return Err("subagent definition max_iterations must be > 0".to_string());
        }
        Ok(parsed)
    }

    fn invalid_data_error(message: String) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message)
    }

    /// Parse a subagent definition from markdown content with YAML frontmatter.
    ///
    /// Format:
    /// ```markdown
    /// ---
    /// name: subagent-id
    /// description: Description of when to use this subagent.
    /// ---
    ///
    /// System prompt content...
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when frontmatter is missing, invalid, or uses an unsupported version.
    pub fn parse(content: &str) -> Result<Self, String> {
        let (frontmatter, system_prompt) = Self::split_frontmatter(content)?;
        let parsed = Self::parse_frontmatter(frontmatter)?;
        Ok(Self {
            id: parsed.name,
            description: parsed.description,
            system_prompt: system_prompt.to_string(),
            model: parsed.model,
            max_iterations: parsed.max_iterations,
        })
    }

    /// Load a subagent definition from a file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed.
    pub async fn from_file_async(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let content = fs::read_to_string(path).await?;
        Self::parse(&content).map_err(Self::invalid_data_error)
    }

    /// Load all subagent definitions from a directory asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be read or any definition cannot be parsed.
    pub async fn load_from_dir_async(dir: impl AsRef<Path>) -> std::io::Result<Vec<Self>> {
        let mut definitions = Vec::new();
        let dir = dir.as_ref();

        let mut entries = match fs::read_dir(dir).await {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e),
        };

        while let Some(entry) = entries.try_next().await? {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "md") {
                definitions.push(Self::from_file_async(&path).await?);
            }
        }

        Ok(definitions)
    }
}

/// A bundled subagent, started by the `subagent` subcommand of its name.
///
/// Its prompt, the model tier it runs on and what it may write are fixed:
/// every one but [`General`](Self::General) writes only its own scratch
/// directory, and only `General` leaves its tier to routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinSubagent {
    /// Investigates a codebase; fast tier, read-only.
    Explore,
    /// Researches a topic on the web; fast tier, read-only.
    Research,
    /// Audits code for defects; advanced tier, read-only.
    Audit,
    /// Reviews a change; advanced tier, read-only.
    Review,
    /// Plans an implementation; advanced tier, read-only.
    Plan,
    /// Carries out a task, code changes included, in the worktrees it is
    /// granted; the fast model routes its prompt to a tier.
    General,
}

impl BuiltinSubagent {
    /// Every bundled subagent.
    pub const ALL: [Self; 6] = [
        Self::Explore,
        Self::Research,
        Self::Audit,
        Self::Review,
        Self::Plan,
        Self::General,
    ];

    /// The subcommand that starts it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Research => "research",
            Self::Audit => "audit",
            Self::Review => "review",
            Self::Plan => "plan",
            Self::General => "general",
        }
    }

    /// The tier it runs on; `None` for [`General`](Self::General), whose
    /// prompt the fast model routes to a tier.
    #[must_use]
    pub const fn tier(self) -> Option<ModelTier> {
        match self {
            Self::Explore | Self::Research => Some(ModelTier::Fast),
            Self::Audit | Self::Review | Self::Plan => Some(ModelTier::Advanced),
            Self::General => None,
        }
    }

    const fn source(self) -> &'static str {
        match self {
            Self::Explore => include_str!("prompts/subagents/explore.md"),
            Self::Research => include_str!("prompts/subagents/research.md"),
            Self::Audit => include_str!("prompts/subagents/audit.md"),
            Self::Review => include_str!("prompts/subagents/review.md"),
            Self::Plan => include_str!("prompts/subagents/plan.md"),
            Self::General => include_str!("prompts/subagents/general.md"),
        }
    }

    /// Its bundled definition.
    ///
    /// # Panics
    ///
    /// Panics if the bundled definition is invalid.
    #[must_use]
    pub fn definition(self) -> SubagentDefinition {
        SubagentDefinition::parse(self.source())
            .unwrap_or_else(|error| panic!("bundled subagent '{}': {error}", self.name()))
    }
}

/// The definitions of every bundled subagent, in [`BuiltinSubagent::ALL`]
/// order.
///
/// # Panics
///
/// Panics if a bundled subagent definition is invalid.
#[must_use]
pub fn builtin_subagents() -> Vec<SubagentDefinition> {
    BuiltinSubagent::ALL
        .into_iter()
        .map(BuiltinSubagent::definition)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subagent_definition() {
        let content = r"---
version: 1
name: test-agent
description: This is a test agent for exploring and testing.
model: fast
max_iterations: 42
---

# Test Agent

You are a test agent. Do test things.
";

        let def = SubagentDefinition::parse(content).unwrap();
        assert_eq!(def.id, "test-agent");
        assert_eq!(
            def.description,
            "This is a test agent for exploring and testing."
        );
        assert_eq!(def.model, Some(ModelTier::Fast));
        assert_eq!(def.max_iterations, 42);
        assert!(def.system_prompt.contains("You are a test agent"));

        let without_model =
            SubagentDefinition::parse(&content.replace("model: fast\n", "")).unwrap();
        assert_eq!(without_model.model, None);
    }

    #[test]
    fn every_bundled_subagent_loads_under_its_subcommand_name() {
        for subagent in BuiltinSubagent::ALL {
            let def = subagent.definition();
            assert_eq!(def.id, subagent.name());
            assert_ne!(def.description, "");
            assert_ne!(def.system_prompt, "");
            assert_eq!(
                def.model,
                None,
                "'{}' takes its tier from `BuiltinSubagent::tier`",
                subagent.name()
            );
        }
        assert_eq!(builtin_subagents().len(), BuiltinSubagent::ALL.len());
    }
}
