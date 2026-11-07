//! Built-in context sources.
//!
//! These cover the ambient state every terminal-first agent has: the
//! terminals still running behind it and the skills activated for the run.
//! The working document has a source of its own,
//! [`WorkingDocument`](crate::WorkingDocument).
//! Each one used to be a special case inside the agent loop, with its own
//! field on `Agent`, its own repopulation call and its own alert path. As
//! [`ContextSource`] implementations they are ordinary registrations with
//! nothing privileged about them.

#[cfg(feature = "skills")]
use std::future::Future;

use askama::Template;

use super::source::{Block, ContextSource};

#[derive(Template)]
#[template(path = "system_reminder.txt", escape = "none")]
struct SystemReminderTemplate {
    content: String,
}

/// Renders a typed template into a block under `tag`.
///
/// # Panics
///
/// Panics if the template fails to render, which is a bug in the template.
fn block(tag: &str, template: &impl Template) -> Block {
    Block::new(
        tag,
        template
            .render()
            .expect("context block template must render"),
    )
}

/// Surfaces the terminals still running behind the agent.
#[derive(Debug)]
pub struct BackgroundJobs {
    registry: aither_sandbox::JobRegistry,
}

impl BackgroundJobs {
    /// Reports the jobs tracked by `registry`.
    #[must_use]
    pub const fn new(registry: aither_sandbox::JobRegistry) -> Self {
        Self { registry }
    }
}

impl ContextSource for BackgroundJobs {
    async fn render(&self) -> Option<Block> {
        let running = self.registry.format_running_jobs().await;
        if running.is_empty() {
            return None;
        }
        Some(block(
            "system_reminder",
            &SystemReminderTemplate {
                content: format!(
                    "Running background terminals:\n{running}Use terminal_read for incremental output, read redirected output files via terminal commands like head/tail/rg/cat when needed, terminal_input for stdin, and terminal_kill to stop tasks."
                ),
            },
        ))
    }
}

#[cfg(feature = "skills")]
#[derive(Template)]
#[template(path = "skill_instruction.txt", escape = "none")]
struct SkillInstructionTemplate<'a> {
    name: &'a str,
    description: &'a str,
    instructions: &'a str,
    allowed_tools: Option<&'a [String]>,
    resource_paths: Option<&'a [String]>,
}

/// Surfaces the instructions of the skills activated for the current run.
#[cfg(feature = "skills")]
#[derive(Debug, Default)]
pub struct ActiveSkills {
    skills: Vec<aither_skills::Skill>,
}

#[cfg(feature = "skills")]
impl ActiveSkills {
    /// Creates an empty set of active skills.
    #[must_use]
    pub const fn new() -> Self {
        Self { skills: Vec::new() }
    }

    /// Replaces the activated skills.
    pub fn set(&mut self, skills: Vec<aither_skills::Skill>) {
        self.skills = skills;
    }

    /// The activated skills.
    #[must_use]
    pub fn skills(&self) -> &[aither_skills::Skill] {
        &self.skills
    }

    /// Whether any skill is activated.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

#[cfg(feature = "skills")]
impl ContextSource for ActiveSkills {
    fn render(&self) -> impl Future<Output = Option<Block>> + Send {
        std::future::ready(self.render_skills())
    }
}

#[cfg(feature = "skills")]
impl ActiveSkills {
    /// Renders every activated skill into a single block.
    fn render_skills(&self) -> Option<Block> {
        if self.skills.is_empty() {
            return None;
        }
        let mut rendered = String::new();
        for skill in &self.skills {
            let mut resource_paths = skill.resources.keys().cloned().collect::<Vec<_>>();
            resource_paths.sort();
            let instruction = SkillInstructionTemplate {
                name: &skill.name,
                description: &skill.description,
                instructions: &skill.instructions,
                allowed_tools: skill.allowed_tools.as_deref(),
                resource_paths: (!resource_paths.is_empty()).then_some(resource_paths.as_slice()),
            }
            .render()
            .expect("skill instruction template must render");
            if !rendered.is_empty() {
                rendered.push('\n');
            }
            rendered.push_str(&instruction);
        }
        Some(Block::new("skill_instructions", rendered))
    }
}
