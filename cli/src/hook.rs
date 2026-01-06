//! Debug hook for CLI agent testing.
//!
//! Provides logging and debugging output for tool calls and agent operations.

use aither_agent::{Hook, HookAction, StopContext, ToolResultContext, ToolUseContext};

/// A debug hook that logs tool calls and agent operations.
///
/// This hook prints colored output for tool calls, results, and completion stats.
#[derive(Debug, Clone, Copy, Default)]
pub struct DebugHook;

impl Hook for DebugHook {
    async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> HookAction {
        println!(
            "\x1b[36m[tool]\x1b[0m {} \x1b[90m(turn {})\x1b[0m",
            ctx.tool_name, ctx.turn
        );

        // Pretty print JSON arguments if possible
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(ctx.arguments) {
            if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                for line in pretty.lines() {
                    println!("  \x1b[90m{line}\x1b[0m");
                }
            }
        }

        HookAction::Continue
    }

    async fn post_tool_use(&self, ctx: &ToolResultContext<'_>) -> HookAction {
        let duration_ms = ctx.duration.as_millis();

        match ctx.result {
            Ok(result) => {
                let truncated = truncate_output(result, 500);
                println!(
                    "\x1b[32m[ok]\x1b[0m {} \x1b[90m({duration_ms}ms)\x1b[0m",
                    ctx.tool_name
                );
                for line in truncated.lines().take(10) {
                    println!("  \x1b[90m{line}\x1b[0m");
                }
                if truncated.lines().count() > 10 {
                    println!("  \x1b[90m... ({} more lines)\x1b[0m", truncated.lines().count() - 10);
                }
            }
            Err(err) => {
                println!(
                    "\x1b[31m[error]\x1b[0m {} \x1b[90m({duration_ms}ms)\x1b[0m",
                    ctx.tool_name
                );
                println!("  \x1b[31m{err}\x1b[0m");
            }
        }

        HookAction::Continue
    }

    async fn on_stop(&self, ctx: &StopContext<'_>) -> HookAction {
        println!(
            "\n\x1b[90m[completed in {} turn(s), reason: {:?}]\x1b[0m",
            ctx.turns, ctx.reason
        );
        HookAction::Continue
    }
}

/// Truncates output to a maximum length, adding ellipsis if truncated.
fn truncate_output(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
