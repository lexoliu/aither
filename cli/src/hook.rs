//! Debug hook for CLI agent testing.
//!
//! Provides logging and human-friendly terminal output display.

use aither_agent::{
    Hook, PostToolAction, PreToolAction, StopContext, ToolResultContext, ToolUseContext,
};
use aither_core::llm::ToolResult;

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
                let result = render_tool_result(ctx.result);
                if let Some(output) = parse_terminal_result(&result) {
                    print_terminal_output(&output, duration_ms);
                } else {
                    println!("\x1b[90m({duration_ms}ms)\x1b[0m");
                    let truncated = truncate_output(&result, 500);
                    for line in truncated.lines().take(10) {
                        println!("  {line}");
                    }
                }
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
                let result = render_tool_result(ctx.result);
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

fn render_tool_result(result: &ToolResult) -> String {
    match result.render_for_cli() {
        Ok(output) => output,
        Err(error) => {
            panic!("failed to render ToolResult for CLI output: {error}");
        }
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

/// Parsed terminal output.
struct TerminalOutput {
    stdout: Option<String>,
    stderr: Option<String>,
    exit_code: i64,
    task_id: Option<String>,
    status: Option<String>,
}

/// Parse terminal result JSON into human-readable format.
fn parse_terminal_result(result: &str) -> Option<TerminalOutput> {
    let parsed: serde_json::Value = serde_json::from_str(result).ok()?;

    // Handle stdout - can be string or object with "text" field
    let stdout = match parsed.get("stdout") {
        Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(serde_json::Value::Object(obj)) => {
            obj.get("text").and_then(|t| t.as_str()).map(String::from)
        }
        _ => None,
    };

    // Handle stderr similarly
    let stderr = match parsed.get("stderr") {
        Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(serde_json::Value::Object(obj)) => {
            obj.get("text").and_then(|t| t.as_str()).map(String::from)
        }
        _ => None,
    };

    let exit_code = parsed
        .get("exit_code")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    let task_id = parsed
        .get("task_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let status = parsed
        .get("status")
        .and_then(|v| v.as_str())
        .map(String::from);

    Some(TerminalOutput {
        stdout,
        stderr,
        exit_code,
        task_id,
        status,
    })
}

/// Print terminal output in human-friendly format.
fn print_terminal_output(output: &TerminalOutput, duration_ms: u128) {
    // Background task
    if let Some(ref task_id) = output.task_id
        && output.status.as_deref() == Some("running")
    {
        println!("\x1b[33m⏳ Background task started:\x1b[0m {task_id}");
        return;
    }

    // Success indicator
    if output.exit_code == 0 {
        print!("\x1b[90m");
    } else {
        print!("\x1b[31m✗ Exit code {}:\x1b[0m ", output.exit_code);
    }

    // Duration
    println!("({duration_ms}ms)\x1b[0m");

    // Show stdout
    if let Some(ref stdout) = output.stdout {
        let truncated = truncate_output(stdout, 2000);
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

    // Show stderr if present
    if let Some(ref stderr) = output.stderr {
        println!("\x1b[31m  stderr:\x1b[0m");
        let truncated = truncate_output(stderr, 500);
        for line in truncated.lines().take(5) {
            println!("  \x1b[31m{line}\x1b[0m");
        }
    }
}

/// Truncates output to a maximum character count (UTF-8 safe).
fn truncate_output(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => format!("{}[...]", &s[..byte_idx]),
        None => s.to_string(), // String is shorter than max_chars
    }
}
