//! Built-in context sources.
//!
//! These three cover the ambient state every terminal-first agent has: the
//! checklist it is working through, the terminals still running behind it,
//! and the skills activated for the run. Each one used to be a special case
//! inside the agent loop, with its own field on `Agent`, its own
//! repopulation call and its own alert path. As [`ContextSource`]
//! implementations they are ordinary registrations with nothing privileged
//! about them, and the state each one needs to remember between turns —
//! previous file contents, how many turns have passed without a change —
//! belongs to the source instead of the agent.

use std::future::Future;
use std::path::PathBuf;

use serde::Serialize;

use super::source::{Block, ContextSource};
use crate::working_docs::{self, WorkingDocsSnapshot};

#[derive(Serialize)]
struct TasksXml {
    #[serde(skip_serializing_if = "Option::is_none")]
    tasks: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recent_changes: Option<String>,
}

#[derive(Serialize)]
struct TasksStale {
    #[serde(rename = "$text")]
    content: String,
}

#[derive(Serialize)]
struct SystemReminder {
    #[serde(rename = "$text")]
    content: String,
}

/// Surfaces `tasks.md` and nags when it stops being maintained.
///
/// The document is the canonical task list: it outlives compaction, which the
/// conversation does not, so a model that stops updating it loses the only
/// durable record of its own plan. Staleness is measured in tool turns rather
/// than wall-clock time — a turn that ran tools without touching the file is
/// the actual signal — and the counter resets on any content change, as well
/// as while the checklist has no unchecked items, because a finished
/// checklist is not stale. The alert fires once per stale episode.
#[derive(Debug)]
pub struct TasksDocument {
    /// Sandbox root holding `tasks.md`.
    sandbox_dir: PathBuf,
    /// Tool turns without a change before alerting.
    stale_after_turns: usize,
    /// Snapshot rendered into the previous request, for the change summary.
    rendered: Option<WorkingDocsSnapshot>,
    /// Content observed at the previous tool turn (`None` = file absent).
    last_content: Option<String>,
    /// Whether a baseline observation exists yet.
    seen: bool,
    /// Consecutive tool turns without a content change.
    stale_turns: usize,
    /// Whether the current stale episode already alerted.
    alerted: bool,
}

impl TasksDocument {
    /// Watches `tasks.md` in `sandbox_dir`, alerting after `stale_after_turns`.
    #[must_use]
    pub fn new(sandbox_dir: impl Into<PathBuf>, stale_after_turns: usize) -> Self {
        Self {
            sandbox_dir: sandbox_dir.into(),
            stale_after_turns,
            rendered: None,
            last_content: None,
            seen: false,
            stale_turns: 0,
            alerted: false,
        }
    }
}

impl ContextSource for TasksDocument {
    async fn render(&self) -> Option<Block> {
        let docs = working_docs::read_snapshot(&self.sandbox_dir).await;
        let recent_changes = self
            .rendered
            .as_ref()
            .and_then(|previous| working_docs::tasks_md_diff(previous, &docs));
        let tasks = docs.tasks_md;
        if tasks.is_none() && recent_changes.is_none() {
            return None;
        }
        Some(Block::xml(
            "tasks_document",
            &TasksXml {
                tasks,
                recent_changes,
            },
        ))
    }

    async fn commit(&mut self, _block: &Block) {
        self.rendered = Some(working_docs::read_snapshot(&self.sandbox_dir).await);
    }

    async fn on_tool_turn_end(&mut self) -> Option<Block> {
        let docs = working_docs::read_snapshot(&self.sandbox_dir).await;
        let threshold = self.stale_after_turns;

        if !self.seen {
            self.seen = true;
            self.last_content = docs.tasks_md;
            return None;
        }
        if self.last_content != docs.tasks_md {
            self.last_content = docs.tasks_md;
            self.stale_turns = 0;
            self.alerted = false;
            return None;
        }

        let missing = match &docs.tasks_md {
            Some(content) if working_docs::has_unchecked_markdown_tasks(content) => false,
            Some(_) => {
                self.stale_turns = 0;
                self.alerted = false;
                return None;
            }
            None => true,
        };

        self.stale_turns += 1;
        if self.stale_turns < threshold || self.alerted {
            return None;
        }
        self.alerted = true;
        let content = if missing {
            format!(
                "{threshold} tool turns have run without a tasks.md. For multi-step work, create tasks.md in the sandbox root with a checklist of the plan — it is the canonical task document and survives context compaction."
            )
        } else {
            format!(
                "tasks.md has unchecked items but has not been updated for {threshold} tool turns. Mark finished items and keep the checklist in sync with actual progress."
            )
        };
        Some(Block::xml("tasks_stale", &TasksStale { content }))
    }

    async fn on_final_turn(&mut self) -> Option<Block> {
        let docs = working_docs::read_snapshot(&self.sandbox_dir).await;
        if !docs.has_unchecked_items() {
            return None;
        }
        Some(Block::xml(
            "system_reminder",
            &SystemReminder {
                content: "tasks.md still has unchecked items. Continue working through the checklist. If user input is required, call ask_user and then proceed.".to_string(),
            },
        ))
    }
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
        Some(Block::xml(
            "system_reminder",
            &SystemReminder {
                content: format!(
                    "Running background terminals:\n{running}Use terminal_read for incremental output, read redirected output files via terminal commands like head/tail/grep/cat when needed, terminal_input for stdin, and terminal_kill to stop tasks."
                ),
            },
        ))
    }
}

#[cfg(feature = "skills")]
#[derive(Serialize)]
struct SkillInstructionXml {
    name: String,
    description: String,
    instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resource_paths: Option<Vec<String>>,
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
            let block = Block::xml(
                "skill_instruction",
                &SkillInstructionXml {
                    name: skill.name.clone(),
                    description: skill.description.clone(),
                    instructions: skill.instructions.clone(),
                    allowed_tools: skill.allowed_tools.clone(),
                    resource_paths: (!resource_paths.is_empty()).then_some(resource_paths),
                },
            );
            if !rendered.is_empty() {
                rendered.push('\n');
            }
            rendered.push_str(block.content());
        }
        Some(Block::new("skill_instructions", rendered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn write_tasks(root: &std::path::Path, content: &str) {
        async_fs::write(root.join("tasks.md"), content)
            .await
            .expect("write tasks.md");
    }

    #[tokio::test]
    async fn an_absent_tasks_document_renders_nothing() {
        let root = tempfile::tempdir().expect("tempdir");
        let source = TasksDocument::new(root.path(), 4);
        assert!(source.render().await.is_none());
    }

    #[tokio::test]
    async fn the_rendered_block_carries_the_document_and_its_changes() {
        let root = tempfile::tempdir().expect("tempdir");
        write_tasks(root.path(), "- [ ] first\n").await;

        let mut source = TasksDocument::new(root.path(), 4);
        let block = source.render().await.expect("renders the document");
        assert!(block.content().contains("- [ ] first"));
        source.commit(&block).await;

        write_tasks(root.path(), "- [x] first\n- [ ] second\n").await;
        let block = source.render().await.expect("renders the document");
        assert!(
            block.content().contains("recent_changes"),
            "a changed document must report its diff: {}",
            block.content()
        );
    }

    #[tokio::test]
    async fn staleness_alerts_once_per_episode_and_resets_on_change() {
        let root = tempfile::tempdir().expect("tempdir");
        write_tasks(root.path(), "- [ ] work\n").await;
        let mut source = TasksDocument::new(root.path(), 2);

        // First turn establishes the baseline.
        assert!(source.on_tool_turn_end().await.is_none());
        // Second turn is the first stale one; the threshold is two.
        assert!(source.on_tool_turn_end().await.is_none());
        let alert = source.on_tool_turn_end().await.expect("alert fires");
        assert!(alert.content().contains("has unchecked items"));
        assert!(
            source.on_tool_turn_end().await.is_none(),
            "the alert must not repeat within one episode"
        );

        write_tasks(root.path(), "- [x] work\n- [ ] more\n").await;
        assert!(source.on_tool_turn_end().await.is_none());
        assert!(source.on_tool_turn_end().await.is_none());
        assert!(
            source.on_tool_turn_end().await.is_some(),
            "a new stale episode must alert again"
        );
    }

    #[tokio::test]
    async fn a_complete_checklist_is_never_stale() {
        let root = tempfile::tempdir().expect("tempdir");
        write_tasks(root.path(), "- [x] done\n").await;
        let mut source = TasksDocument::new(root.path(), 1);

        assert!(source.on_tool_turn_end().await.is_none());
        for _ in 0..5 {
            assert!(source.on_tool_turn_end().await.is_none());
        }
    }

    #[tokio::test]
    async fn a_missing_document_is_reported_differently() {
        let root = tempfile::tempdir().expect("tempdir");
        let mut source = TasksDocument::new(root.path(), 1);

        assert!(source.on_tool_turn_end().await.is_none());
        let alert = source.on_tool_turn_end().await.expect("alert fires");
        assert!(alert.content().contains("without a tasks.md"));
    }

    #[tokio::test]
    async fn unchecked_items_block_the_final_turn() {
        let root = tempfile::tempdir().expect("tempdir");
        write_tasks(root.path(), "- [ ] work\n").await;
        let mut source = TasksDocument::new(root.path(), 4);
        assert!(source.on_final_turn().await.is_some());

        write_tasks(root.path(), "- [x] work\n").await;
        assert!(source.on_final_turn().await.is_none());
    }
}
