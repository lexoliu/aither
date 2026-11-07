//! A working-document schema and stop rule unlike any embedder's, for
//! tests.

use super::{DocumentRow, DocumentSchema, ParsedDocument, WorkRef, words};
use crate::{StopDecision, StopHook, TurnBoundaryContext};

/// `* ` lines are open work, `root ` lines name roots, `~ ` lines are rows
/// that are no work, `waits <blocker>` marks what a row waits on, and a
/// `[ ]` line breaks the format.
pub struct ListSchema;

impl DocumentSchema for ListSchema {
    fn parse(&self, content: &str) -> ParsedDocument {
        let mut document = ParsedDocument::new();
        for (index, line) in content.lines().enumerate() {
            let number = index + 1;
            if let Some(text) = line.strip_prefix("root ") {
                for root in WorkRef::parse_all(text) {
                    document.push_root(root);
                }
                document.push_row(DocumentRow::new(number, text).naming(WorkRef::parse_all(text)));
            } else if let Some(text) = line.strip_prefix("* ") {
                let mut row = DocumentRow::new(number, text)
                    .open()
                    .naming(WorkRef::parse_all(text));
                let mut words = words(text);
                if words.any(|word| word == "waits")
                    && let Some(blocker) = words.next()
                {
                    row = row.blocked_on(blocker);
                }
                document.push_row(row);
            } else if let Some(text) = line.strip_prefix("~ ") {
                document.push_row(DocumentRow::new(number, text).naming(WorkRef::parse_all(text)));
            } else if line.starts_with("[ ]") {
                document.push_problem(format!("Line {number} of plan.md is a checkbox."));
            }
        }
        document
    }
}

/// The reminder [`HoldOpenRows`] holds the run with.
pub const CONTINUE: &str = "plan.md still has open rows; continue with them.";

/// Holds the run while the document falls behind, and with [`CONTINUE`]
/// while it has actionable rows.
pub struct HoldOpenRows;

impl StopHook for HoldOpenRows {
    fn on_turn_end(
        &self,
        ctx: &TurnBoundaryContext<'_>,
    ) -> impl Future<Output = StopDecision> + Send {
        let decision = match ctx.document {
            Some(review) if !review.findings().is_empty() => StopDecision::Hold { reminder: None },
            Some(review) if review.actionable_rows().next().is_some() => StopDecision::Hold {
                reminder: Some(CONTINUE.to_string()),
            },
            _ => StopDecision::End,
        };
        std::future::ready(decision)
    }
}
