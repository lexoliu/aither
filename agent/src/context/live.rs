//! Prefix content that changes while a session runs.
//!
//! Some system blocks describe things that change mid-session: a repository's
//! `AGENTS.md`, memory, the table of commands `terminal` runs. Rewriting such
//! a block at once would invalidate the provider's cached prefix, and leaving
//! it would leave the model believing the old version. Live content takes the
//! third way, the same for every such block:
//!
//! - a change made before any request has carried the prefix is written into
//!   it directly, since nothing cached the old version;
//! - a change made after is staged, the prefix stays byte-identical, and the
//!   change is announced once, as a diff alert appended to the conversation;
//! - the next reassembly (a compaction, or a cache that went cold) folds the
//!   clean current version into the prefix and drops the diff alerts, so the
//!   model reads the content once, with no diff.
//!
//! A host brings its blocks to their current version with
//! [`Context::update_live`](crate::Context::update_live) whenever it reads
//! them; an update that changes nothing announces nothing, so calling it on
//! every turn is cheap and never repeats an alert.

use askama::Template;

use super::system::{
    External, ExternalContent, ExternalSlot, Instruction, InstructionSlot, SystemRegion,
    TextChange, boundary,
};

/// Names a piece of live content in the alert announcing its change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveContent {
    subject: &'static str,
    tag: &'static str,
}

impl LiveContent {
    /// The table of the CLI commands `terminal` runs.
    pub const COMMANDS: Self = Self::new("The table of terminal commands", "commands_diff");

    /// Live content the alert calls `subject`, which opens its first
    /// sentence ("The launch directory's AGENTS.md files"), and whose diff it
    /// wraps in the tag `tag` (`agents_md_diff`).
    #[must_use]
    pub const fn new(subject: &'static str, tag: &'static str) -> Self {
        Self { subject, tag }
    }
}

/// The blocks one [`Context::update_live`](crate::Context::update_live) call
/// brings to their current version; every block that changed goes into the
/// one alert the call appends.
pub struct LiveUpdate<'a> {
    instructions: &'a mut SystemRegion<Instruction>,
    external: &'a mut SystemRegion<External>,
    warm: bool,
    changes: Vec<TextChange>,
}

impl std::fmt::Debug for LiveUpdate<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveUpdate")
            .field("warm", &self.warm)
            .field("changes", &self.changes.len())
            .finish_non_exhaustive()
    }
}

impl<'a> LiveUpdate<'a> {
    pub(crate) const fn new(
        instructions: &'a mut SystemRegion<Instruction>,
        external: &'a mut SystemRegion<External>,
        warm: bool,
    ) -> Self {
        Self {
            instructions,
            external,
            warm,
            changes: Vec::new(),
        }
    }

    /// Brings the instruction block in `slot` to `desired`, stored verbatim;
    /// `None` removes the block.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    pub fn instruction(&mut self, slot: &InstructionSlot, desired: Option<String>) -> &mut Self {
        let change = self.instructions.change(slot, desired, self.warm);
        self.changes.extend(change);
        self
    }

    /// Brings the external-content block in `slot` to `desired`; `None`
    /// removes the block.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    pub fn external(&mut self, slot: &ExternalSlot, desired: Option<ExternalContent>) -> &mut Self {
        let change = self.external.change(slot, desired, self.warm);
        self.changes.extend(change);
        self
    }

    /// The alert announcing every change this update made, `None` when the
    /// model's view did not change.
    pub(crate) fn into_alert(self, content: &LiveContent) -> Option<String> {
        if self.changes.is_empty() {
            return None;
        }
        let diff: String = self
            .changes
            .iter()
            .map(|change| unified_diff(&change.before, &change.after))
            .collect();
        Some(
            LiveContentChangedTemplate {
                subject: content.subject,
                tag: content.tag,
                boundary: boundary(&[&diff]),
                diff: &diff,
            }
            .render()
            .expect("live content alert template must render"),
        )
    }
}

/// The unified diff from `before` to `after`.
pub(crate) fn unified_diff(before: &str, after: &str) -> String {
    similar::TextDiff::from_lines(before, after)
        .unified_diff()
        .context_radius(2)
        .to_string()
}

/// The one alert announcing a change to live content.
#[derive(Template)]
#[template(path = "live_content_changed.txt", escape = "none")]
struct LiveContentChangedTemplate<'a> {
    subject: &'a str,
    tag: &'a str,
    boundary: String,
    diff: &'a str,
}
