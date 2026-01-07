//! Debug hook for CLI agent testing.
//!
//! Provides logging, debugging output, and permission prompts for tool calls.

use std::io::{self, Write};

use aither_agent::{Hook, PostToolAction, PreToolAction, StopContext, ToolResultContext, ToolUseContext};

/// A debug hook that logs tool calls and asks permission for sensitive operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct DebugHook;

impl Hook for DebugHook {
    async fn pre_tool_use(&self, ctx: &ToolUseContext<'_>) -> PreToolAction {
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

        // Check if this is a sensitive operation
        if is_sensitive(ctx.tool_name, ctx.arguments) {
            if !ask_permission() {
                println!("\x1b[33m[denied]\x1b[0m User denied permission");
                return PreToolAction::Deny("User denied permission".to_string());
            }
        }

        PreToolAction::Allow
    }

    async fn post_tool_use(&self, ctx: &ToolResultContext<'_>) -> PostToolAction {
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
                    println!(
                        "  \x1b[90m... ({} more lines)\x1b[0m",
                        truncated.lines().count() - 10
                    );
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

        PostToolAction::Keep
    }

    async fn on_stop(&self, ctx: &StopContext<'_>) -> Option<String> {
        println!(
            "\n\x1b[90m[completed in {} turn(s), reason: {:?}]\x1b[0m",
            ctx.turns, ctx.reason
        );
        None
    }
}

/// Check if an operation is sensitive and requires user permission.
fn is_sensitive(tool_name: &str, arguments: &str) -> bool {
    match tool_name {
        "command" => true,
        "filesystem" => {
            // Parse to check operation type
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                matches!(
                    parsed.get("operation").and_then(|v| v.as_str()),
                    Some("write" | "append" | "delete")
                )
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Ask user for permission to execute a sensitive operation.
fn ask_permission() -> bool {
    print!("\x1b[33m[permission]\x1b[0m Allow? [y/N] ");
    io::stdout().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
    } else {
        false
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
