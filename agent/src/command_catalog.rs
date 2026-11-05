//! The table of the CLI commands `terminal` runs, as the model reads it.
//!
//! Most commands are fixed when the agent is built, but a
//! [`CommandSet`](aither_sandbox::CommandSet), such as an MCP server's tools,
//! changes while the session runs. The table is live content in
//! [`InstructionSlot::COMMANDS`]: the agent re-reads it at every request
//! boundary and brings the block to it through
//! [`Context::update_live`](crate::Context::update_live), so a command the
//! registry gains is announced once as a diff and listed in the clean table
//! after the next compaction.

use aither_sandbox::AnyCommandSet;
use askama::Template;

use crate::context::InstructionSlot;

/// The commands `terminal` runs: the fixed ones with their descriptions and
/// the sets whose commands are read when the table is.
#[derive(Debug)]
pub struct CommandCatalog {
    fixed: Vec<(String, String)>,
    sets: Vec<AnyCommandSet>,
}

impl CommandCatalog {
    /// A catalog of the `fixed` commands, as `(name, description)` pairs,
    /// followed by the commands `sets` hold.
    pub const fn new(fixed: Vec<(String, String)>, sets: Vec<AnyCommandSet>) -> Self {
        Self { fixed, sets }
    }

    /// The command table block as of now, `None` when there is no command.
    ///
    /// Fixed commands come first in registration order, then each set's in
    /// the order it lists them; a name already listed keeps its first row,
    /// as the registry resolves it.
    pub fn render(&self) -> Option<String> {
        let mut commands: Vec<CommandRow> = self
            .fixed
            .iter()
            .map(|(name, description)| CommandRow {
                name: name.clone(),
                description: description.clone(),
            })
            .collect();
        for set in &self.sets {
            for definition in set.definitions() {
                if commands.iter().any(|row| row.name == definition.name()) {
                    continue;
                }
                commands.push(CommandRow {
                    name: definition.name().to_string(),
                    description: short_description(definition.description()),
                });
            }
        }
        if commands.is_empty() {
            return None;
        }
        Some(
            CommandTableTemplate {
                tag: InstructionSlot::COMMANDS.name(),
                commands,
            }
            .render()
            .expect("command table template must render"),
        )
    }
}

/// A command's description as the table shows it: its first sentence.
pub fn short_description(description: &str) -> String {
    description
        .split('.')
        .next()
        .unwrap_or(description)
        .trim()
        .to_string()
}

struct CommandRow {
    name: String,
    description: String,
}

#[derive(Template)]
#[template(path = "command_table.txt", escape = "none")]
struct CommandTableTemplate<'a> {
    tag: &'a str,
    commands: Vec<CommandRow>,
}
