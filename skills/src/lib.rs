//! Aither Skills System
//!
//! Skills are external plugins that teach agents specialized workflows.
//! They're folders with instructions and resources that agents load dynamically.
//!
//! # Structure
//!
//! Skills are stored on disk with the following structure:
//!
//! ```text
//! ~/.aither/skills/
//! └── code-review/
//!     ├── SKILL.md           # Frontmatter + instructions
//!     ├── templates/         # Optional templates
//!     └── scripts/           # Optional scripts
//! ```
//!
//! # SKILL.md Format
//!
//! ```markdown
//! ---
//! name: code-review
//! description: Reviews code for security and best practices
//! triggers:
//!   - "review"
//!   - "security audit"
//! tools:
//!   - Read
//!   - Grep
//! ---
//!
//! # Code Review Skill
//!
//! When reviewing code, follow these steps:
//! 1. First scan for security vulnerabilities...
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use aither_skills::{SkillLoader, SkillRegistry};
//!
//! // Load skills from filesystem
//! let loader = SkillLoader::new()
//!     .add_path("~/.aither/skills")
//!     .add_path("./.aither/skills");
//!
//! let registry = SkillRegistry::from_loader(&loader)?;
//!
//! // Find relevant skills
//! let skills = registry.match_prompt("review this code for security issues");
//!
//! // Get skill by name
//! if let Some(skill) = registry.get("code-review") {
//!     println!("{}", skill.instructions);
//! }
//! ```

mod error;
mod loader;
mod matcher;
mod registry;
mod skill;

pub use error::SkillError;
pub use loader::SkillLoader;
pub use matcher::{MatchResult, SkillMatcher};
pub use registry::SkillRegistry;
pub use skill::{Skill, SkillFrontmatter};
