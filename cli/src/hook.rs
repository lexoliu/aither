//! Debug hook for CLI agent testing.
//!
//! Provides logging and human-friendly terminal output display.

use aither_agent::{
    Hook, PostToolAction, PreToolAction, StopContext, ToolResultContext, ToolUseContext,
};
use aither_core::llm::{CommandOutcome, ToolResult};

/// A debug hook that logs tool calls with human-friendly terminal output.
#[derive(Debug, Clone, Copy, Default)]
pub struct DebugHook;

impl Hook for DebugHook {
    fn pre_tool_use(
        &self,
        ctx: &ToolUseContext<'_>,
    ) -> impl std::future::Future<Output = PreToolAction> + Send {
        // Terminal is the primary tool; show the command directly.
        if ctx.tool_name == "terminal" {
            if let Some((script, mode)) = parse_terminal_args(ctx.arguments) {
                let mode_indicator = match mode.as_deref() {
                    Some("unsafe") => "\x1b[31m⚠\x1b[0m ",
                    Some("network") => "\x1b[33m⚡\x1b[0m ",
                    _ => "",
                };
                // Show command with $ prefix like a terminal (no truncation)
                let display_script = script.trim().replace('\n', "; ");
                println!("\x1b[32m{mode_indicator}$\x1b[0m {display_script}");
            }
        } else {
            // Non-terminal tools (shouldn't happen in terminal-first architecture)
            println!(
                "\x1b[36m[tool]\x1b[0m {} \x1b[90m(turn {})\x1b[0m",
                ctx.tool_name, ctx.turn
            );
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(ctx.arguments)
                && let Ok(pretty) = serde_json::to_string_pretty(&parsed)
            {
                for line in pretty.lines() {
                    println!("  \x1b[90m{line}\x1b[0m");
                }
            }
        }

        // Permission is handled by TerminalTool's InteractivePermissionHandler
        // No need to check here - would cause duplicate prompts
        std::future::ready(PreToolAction::Allow)
    }

    fn post_tool_use(
        &self,
        ctx: &ToolResultContext<'_>,
    ) -> impl std::future::Future<Output = PostToolAction> + Send {
        let duration_ms = ctx.duration.as_millis();

        if ctx.tool_name == "terminal" {
            if let Some(err) = ctx.result.error_message() {
                println!("\x1b[31m✗ Error:\x1b[0m {err} \x1b[90m({duration_ms}ms)\x1b[0m");
            } else {
                print_terminal_result(ctx.result, duration_ms);
            }
        } else {
            // Non-terminal tools
            if let Some(err) = ctx.result.error_message() {
                println!(
                    "\x1b[31m[error]\x1b[0m {} \x1b[90m({duration_ms}ms)\x1b[0m",
                    ctx.tool_name
                );
                println!("  \x1b[31m{err}\x1b[0m");
            } else {
                let result = ctx.result.render_for_model();
                let truncated = truncate_output(&result, 500);
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
        }

        std::future::ready(PostToolAction::Keep)
    }

    fn on_stop(
        &self,
        ctx: &StopContext<'_>,
    ) -> impl std::future::Future<Output = Option<String>> + Send {
        println!(
            "\n\x1b[90m[completed in {} turn(s), reason: {:?}]\x1b[0m",
            ctx.turns, ctx.reason
        );
        std::future::ready(None)
    }
}

/// Parsed terminal arguments.
fn parse_terminal_args(arguments: &str) -> Option<(String, Option<String>)> {
    let parsed: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let script = parsed.get("script")?.as_str()?.to_string();
    let mode = parsed
        .get("mode")
        .and_then(|v| v.as_str())
        .map(String::from);
    Some((script, mode))
}

/// Print a terminal tool result in human-friendly format, driven by its
/// typed command outcome.
fn print_terminal_result(result: &ToolResult, duration_ms: u128) {
    match result.command().map(|command| &command.outcome) {
        Some(CommandOutcome::Running { task_id, .. }) => {
            println!("\x1b[33m⏳ Background task started:\x1b[0m {task_id}");
            return;
        }
        Some(CommandOutcome::Exited { code }) if *code != 0 => {
            print!("\x1b[31m✗ Exit code {code}:\x1b[0m ");
        }
        Some(CommandOutcome::Killed { task_id, .. }) => {
            print!("\x1b[31m✗ Killed task {task_id}:\x1b[0m ");
        }
        _ => print!("\x1b[90m"),
    }
    println!("({duration_ms}ms)\x1b[0m");

    let text = result.render_for_model();
    let truncated = truncate_output(&text, 2000);
    let lines: Vec<&str> = truncated.lines().collect();
    let max_lines = 20;
    for line in lines.iter().take(max_lines) {
        println!("  {line}");
    }
    if lines.len() > max_lines {
        println!(
            "  \x1b[90m... ({} more lines)\x1b[0m",
            lines.len() - max_lines
        );
    }
}

/// Truncates output to a maximum character count (UTF-8 safe).
fn truncate_output(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => format!("{}[...]", &s[..byte_idx]),
        None => s.to_string(), // String is shorter than max_chars
    }
}
