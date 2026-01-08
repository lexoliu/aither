//! Interactive CLI for testing aither agents.
//!
//! A REPL-style interface for testing agents with OpenAI, Claude, or Gemini.
//!
//! # Bash-First Design
//!
//! The agent has a single `bash` tool. All other capabilities are exposed as
//! terminal commands that can be run within bash scripts:
//!
//! - `websearch "query"` - Search the web
//! - `webfetch "url"` - Fetch and read web pages
//! - `todo ...` - Track tasks
//!
//! Standard bash commands (ls, cat, grep, find, etc.) work normally.
//!
//! # Usage
//!
//! ```bash
//! # Auto-detect provider from available API keys
//! cargo run -p aither-cli
//!
//! # Explicit provider
//! OPENAI_API_KEY=xxx cargo run -p aither-cli -- --provider openai
//! ANTHROPIC_API_KEY=xxx cargo run -p aither-cli -- --provider claude
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --provider gemini
//!
//! # Headless mode (single query, useful for testing/scripting)
//! cargo run -p aither-cli -- --prompt "What is 2+2?"
//! cargo run -p aither-cli -- --prompt "List files" --quiet
//!
//! # With additional MCP servers
//! cargo run -p aither-cli -- --mcp servers.json
//!
//! # Disable Context7 (documentation lookup)
//! cargo run -p aither-cli -- --no-context7
//! ```

mod cloud;
mod hook;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use aither_agent::sandbox::{configure_raw_handler, permission::{BashMode, PermissionHandler, PermissionError, StatefulPermissionHandler}};
use aither_agent::specialized::TaskTool;
use aither_agent::{Agent, BashAgentBuilder, Hook};
use std::sync::Arc;
use async_lock::Mutex as AsyncMutex;
use executor_core::tokio::TokioGlobal;
use aither_core::LanguageModel;
use aither_core::llm::Role;
use aither_mcp::{McpConnection, McpServersConfig};
use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use tracing_subscriber::EnvFilter;

use crate::cloud::{CloudProvider, Provider};
use crate::hook::DebugHook;

/// Interactive permission handler that prompts the user.
///
/// - Sandboxed: always allow (no prompt)
/// - Network: ask once, remember approval
/// - Unsafe: always ask for each script
#[derive(Debug, Default)]
struct InteractivePermissionHandler;

impl PermissionHandler for InteractivePermissionHandler {
    async fn check(&self, mode: BashMode, script: &str) -> Result<bool, PermissionError> {
        // This will be wrapped in StatefulPermissionHandler for network mode
        match mode {
            BashMode::Sandboxed => Ok(true), // Always allow
            BashMode::Network | BashMode::Unsafe => {
                // Display script and ask for permission
                let mode_desc = mode.description();
                eprintln!("\n\x1b[33m⚠ Permission request: {mode_desc}\x1b[0m");
                eprintln!("\x1b[2mScript:\x1b[22m {}", truncate_script(script, 200));
                eprint!("\x1b[33mAllow? [y/N]: \x1b[0m");
                io::stderr().flush().ok();

                // Read single char in raw mode
                let approved = read_yes_no().unwrap_or(false);
                eprintln!();

                if approved {
                    Ok(true)
                } else {
                    Err(PermissionError::Denied("user declined".to_string()))
                }
            }
        }
    }
}

/// Truncate script for display, showing first N chars (UTF-8 safe).
fn truncate_script(s: &str, max_chars: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    // Find byte index at char boundary
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    if end >= s.len() {
        s
    } else {
        format!("{}...", &s[..end])
    }
}

/// Read y/n response from stdin.
fn read_yes_no() -> Result<bool> {
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    loop {
        if event::poll(Duration::from_secs(30))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        eprint!("y");
                        return Ok(true);
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Enter | KeyCode::Esc => {
                        eprint!("n");
                        return Ok(false);
                    }
                    _ => {}
                },
                _ => {}
            }
        } else {
            // Timeout - default to no
            eprint!("n (timeout)");
            return Ok(false);
        }
    }
}

/// Register MCP tools as bash IPC commands.
///
/// This makes MCP tools available as bash commands (e.g., `resolve-library-id "tokio"`)
/// instead of direct LLM tool calls.
fn register_mcp_tools(conn: McpConnection) {
    let conn = Arc::new(AsyncMutex::new(conn));

    // Get tool definitions before moving conn into closures
    let tools: Vec<_> = {
        // We need to block to get definitions since this is called from async context
        // but we can access the definitions synchronously through the stored tools
        let conn_guard = futures_lite::future::block_on(conn.lock());
        conn_guard
            .mcp_definitions()
            .iter()
            .map(|def| {
                let name = def.name.clone();
                let description = def.description.clone().unwrap_or_default();
                let schema = def.input_schema.clone();
                (name, description, schema)
            })
            .collect()
    };

    for (name, description, schema) in tools {
        let conn = conn.clone();
        let tool_name = name.clone();

        // Build help text from schema
        let help = format!("{}\n\nUsage: {} <args>", description, name);

        // Detect primary arg from schema
        let primary_arg = schema
            .get("required")
            .and_then(|r| r.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .map(String::from);

        configure_raw_handler(
            name,
            help,
            primary_arg,
            move |args| {
                let conn = conn.clone();
                let tool_name = tool_name.clone();
                Box::pin(async move {
                    // Parse CLI args into JSON object
                    let arguments = parse_cli_args_to_json(&args);

                    // Call the MCP tool
                    let mut conn = conn.lock().await;
                    match conn.call(&tool_name, arguments).await {
                        Ok(result) => {
                            // Extract text content from result
                            result
                                .content
                                .into_iter()
                                .filter_map(|c| {
                                    if let aither_mcp::Content::Text(text_content) = c {
                                        Some(text_content.text)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                        Err(e) => format!("{{\"error\": \"{}\"}}", e),
                    }
                })
            },
        );
    }
}

/// Parse CLI-style arguments into a JSON object.
///
/// Handles:
/// - `--key value` pairs
/// - `--json '{...}'` for direct JSON input
/// - Positional arguments (mapped to "query", "arg1", etc.)
fn parse_cli_args_to_json(args: &[String]) -> serde_json::Value {
    // Check for --json flag with direct JSON input
    if args.len() >= 2 && args[0] == "--json" {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&args[1]) {
            return parsed;
        }
    }

    let mut map = serde_json::Map::new();
    let mut i = 0;
    let mut positional_idx = 0;

    while i < args.len() {
        if args[i].starts_with("--") {
            let key = args[i].trim_start_matches("--");
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                map.insert(key.to_string(), serde_json::Value::String(args[i + 1].clone()));
                i += 2;
            } else {
                map.insert(key.to_string(), serde_json::Value::Bool(true));
                i += 1;
            }
        } else {
            // Positional argument - use as "query" or "arg0", "arg1", etc.
            let key = if positional_idx == 0 {
                "query".to_string()
            } else {
                format!("arg{}", positional_idx)
            };
            map.insert(key, serde_json::Value::String(args[i].clone()));
            positional_idx += 1;
            i += 1;
        }
    }

    serde_json::Value::Object(map)
}

/// Simple todo list state for CLI.
static TODO_LIST: std::sync::OnceLock<std::sync::RwLock<Vec<(String, String)>>> = std::sync::OnceLock::new();

fn get_todo_list() -> &'static std::sync::RwLock<Vec<(String, String)>> {
    TODO_LIST.get_or_init(|| std::sync::RwLock::new(Vec::new()))
}

/// Register a simple todo command with natural CLI syntax.
///
/// Usage:
/// - `todo add "Task 1" "Task 2"` - add tasks (pending)
/// - `todo start 1` - mark task 1 as in_progress
/// - `todo done 1` - mark task 1 as completed
/// - `todo list` - show current todos
/// - `todo clear` - clear all todos
fn register_simple_todo() {
    let help = r#"Manage task list for tracking progress.

Usage:
  todo add "Task 1" "Task 2" ...  Add new tasks (pending)
  todo start <n>                  Mark task n as in_progress
  todo done <n>                   Mark task n as completed
  todo list                       Show current todos
  todo clear                      Clear all todos

Examples:
  todo add "Implement feature" "Write tests" "Update docs"
  todo start 1
  todo done 1
  todo list"#;

    configure_raw_handler("todo", help, None::<String>, move |args| {
        Box::pin(async move {
            if args.is_empty() {
                return "Usage: todo <add|start|done|list|clear> [args...]".to_string();
            }

            let cmd = args[0].as_str();
            let list = get_todo_list();

            match cmd {
                "add" => {
                    let mut todos = list.write().unwrap();
                    for task in &args[1..] {
                        todos.push((task.clone(), "pending".to_string()));
                    }
                    format!("Added {} task(s). Total: {}", args.len() - 1, todos.len())
                }
                "start" => {
                    if args.len() < 2 {
                        return "Usage: todo start <task_number>".to_string();
                    }
                    let n: usize = match args[1].parse() {
                        Ok(n) => n,
                        Err(_) => return "Invalid task number".to_string(),
                    };
                    let mut todos = list.write().unwrap();
                    if n == 0 || n > todos.len() {
                        return format!("Task {} not found (have {} tasks)", n, todos.len());
                    }
                    // Set all others to pending/completed, this one to in_progress
                    for (i, (_, status)) in todos.iter_mut().enumerate() {
                        if i == n - 1 {
                            *status = "in_progress".to_string();
                        } else if status == "in_progress" {
                            *status = "pending".to_string();
                        }
                    }
                    format!("Started task {}: {}", n, todos[n - 1].0)
                }
                "done" => {
                    if args.len() < 2 {
                        return "Usage: todo done <task_number>".to_string();
                    }
                    let n: usize = match args[1].parse() {
                        Ok(n) => n,
                        Err(_) => return "Invalid task number".to_string(),
                    };
                    let mut todos = list.write().unwrap();
                    if n == 0 || n > todos.len() {
                        return format!("Task {} not found (have {} tasks)", n, todos.len());
                    }
                    todos[n - 1].1 = "completed".to_string();
                    format!("Completed task {}: {}", n, todos[n - 1].0)
                }
                "list" => {
                    let todos = list.read().unwrap();
                    if todos.is_empty() {
                        return "No tasks.".to_string();
                    }
                    let mut out = String::new();
                    for (i, (task, status)) in todos.iter().enumerate() {
                        let marker = match status.as_str() {
                            "completed" => "[x]",
                            "in_progress" => "[>]",
                            _ => "[ ]",
                        };
                        out.push_str(&format!("{} {}. {}\n", marker, i + 1, task));
                    }
                    out
                }
                "clear" => {
                    list.write().unwrap().clear();
                    "Cleared all tasks.".to_string()
                }
                _ => format!("Unknown command: {}. Use add/start/done/list/clear", cmd),
            }
        })
    });
}

/// Interactive CLI for testing aither agents.
#[derive(Parser, Debug)]
#[command(name = "aither", version, about)]
struct Args {
    /// Model to use. Auto-detects provider from model name prefix.
    #[arg(short, long)]
    model: Option<String>,

    /// Provider to use (openai, claude, gemini). Auto-detected if not specified.
    #[arg(short, long)]
    provider: Option<Provider>,

    /// Custom API base URL (for OpenAI-compatible endpoints, proxies, etc.)
    #[arg(short, long)]
    base_url: Option<String>,

    /// Path to MCP servers configuration file (JSON).
    #[arg(long)]
    mcp: Option<PathBuf>,

    /// System prompt for the agent.
    #[arg(short, long)]
    system: Option<String>,

    /// Disable Context7 MCP server (documentation lookup).
    #[arg(long)]
    no_context7: bool,

    /// Single prompt to run (headless mode). Runs query and exits.
    #[arg(long)]
    prompt: Option<String>,

    /// Quiet mode. Only output the response (useful with --prompt for scripting).
    #[arg(short, long)]
    quiet: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Create cloud provider
    let base_url = args.base_url.as_deref();
    let (cloud, model) = if let Some(provider) = args.provider {
        let model = args.model.as_deref().unwrap_or(provider.default_model());
        (
            CloudProvider::with_base_url(provider, model, base_url)?,
            model.to_string(),
        )
    } else {
        CloudProvider::auto(args.model.as_deref(), base_url)?
    };

    if !args.quiet {
        println!("Aither Agent CLI");
        println!("Provider: {}", cloud.provider());
        println!("Model: {model}");
        if args.prompt.is_none() {
            println!("Commands: /quit, /clear, /history, /compact");
        }
        println!();
    }

    // Headless mode: run single prompt and exit
    if let Some(ref prompt) = args.prompt {
        return run_headless(cloud, &args, prompt).await;
    }

    run_repl(cloud, &args).await
}

async fn build_agent(cloud: CloudProvider, args: &Args) -> Result<Agent<CloudProvider, CloudProvider, CloudProvider, aither_agent::HCons<DebugHook, ()>>> {
    // Create bash tool (creates random four-word working dir under system temp)
    // Uses interactive permission handler:
    // - Sandboxed: always allow (no prompt)
    // - Network: ask once, then remember
    // - Unsafe: always ask for each script
    let workdir_parent = std::env::temp_dir().join("aither");
    let permission_handler = StatefulPermissionHandler::new(InteractivePermissionHandler);
    let bash_tool = aither_agent::sandbox::BashTool::new_in(&workdir_parent, permission_handler, TokioGlobal).await?;

    // Create bash-centric agent builder
    // All tools become IPC commands accessible via bash
    let mut builder = BashAgentBuilder::new(cloud.clone(), bash_tool)
        // Built-in IPC tools
        .ipc_tool(aither_agent::websearch::WebSearchTool::default())
        .ipc_tool(aither_agent::webfetch::WebFetchTool::new())
        .with_ask(cloud.clone()); // Fast LLM for processing piped content

    // Register todo with simple CLI syntax (not JSON)
    register_simple_todo();
    builder = builder.tool_description("todo", "Manage task list (add/start/done/list/clear)");

    // Built-in reload command
    builder = builder.tool_description("reload", "Load file content back into context");

    // Add Context7 MCP server by default (documentation lookup)
    if !args.no_context7 {
        match McpConnection::http("https://mcp.context7.com/mcp").await {
            Ok(conn) => {
                if !args.quiet {
                    println!("Connected to Context7 MCP server");
                }
                // Collect MCP tool descriptions and register
                for def in conn.mcp_definitions() {
                    let desc = def.description.clone().unwrap_or_default();
                    let desc = desc.split('.').next().unwrap_or(&desc).trim().to_string();
                    builder = builder.tool_description(def.name.clone(), desc);
                }
                register_mcp_tools(conn);
            }
            Err(e) => {
                if !args.quiet {
                    eprintln!("Warning: Failed to connect to Context7: {e}");
                }
            }
        }
    }

    // Add MCP connections from config file
    if let Some(ref mcp_path) = args.mcp {
        let config_str = std::fs::read_to_string(mcp_path)
            .with_context(|| format!("failed to read MCP config from {}", mcp_path.display()))?;
        let config: McpServersConfig = serde_json::from_str(&config_str)
            .with_context(|| "failed to parse MCP config")?;

        let connections = McpConnection::from_configs(&config).await?;
        for (name, conn) in connections {
            if !args.quiet {
                println!("Connected to MCP server: {name}");
            }
            for def in conn.mcp_definitions() {
                let desc = def.description.clone().unwrap_or_default();
                let desc = desc.split('.').next().unwrap_or(&desc).trim().to_string();
                builder = builder.tool_description(def.name.clone(), desc);
            }
            register_mcp_tools(conn);
        }
    }

    // Create TaskTool and register as bash IPC command
    let task_tool = TaskTool::new(cloud).with_builtins();
    let mut task_desc = String::from("Spawn subagent for complex tasks (types: ");
    let subagent_names: Vec<_> = task_tool.type_descriptions().iter().map(|(n, _)| *n).collect();
    task_desc.push_str(&subagent_names.join(", "));
    task_desc.push(')');
    builder = builder.ipc_tool_with_desc(task_tool, task_desc);

    // Add system prompt
    let builder = if let Some(ref system) = args.system {
        builder.system_prompt_raw(system)
    } else {
        builder.with_default_prompt()
    };

    Ok(builder.hook(DebugHook).build())
}

/// Run a single prompt and exit (headless mode).
async fn run_headless(cloud: CloudProvider, args: &Args, prompt: &str) -> Result<()> {
    let mut agent = build_agent(cloud, args).await?;

    match agent.query(prompt).await {
        Ok(response) => {
            println!("{response}");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

async fn run_repl(cloud: CloudProvider, args: &Args) -> Result<()> {
    let mut agent = build_agent(cloud, args).await?;

    println!("Agent ready. Type your message or a command.\n");

    // REPL loop
    loop {
        let Some(input) = read_line("You> ")? else {
            break;
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" | "/q" => break,
                "/clear" => {
                    agent.clear_history();
                    println!("History cleared.");
                    continue;
                }
                "/history" => {
                    print_history(&agent);
                    continue;
                }
                "/compact" => {
                    match agent.compact().await {
                        Ok(Some(result)) => {
                            println!(
                                "\x1b[2mCompacted {} messages into summary, starting fresh session\x1b[22m",
                                result.messages_compacted
                            );
                        }
                        Ok(None) => {
                            println!("Nothing to compact (no conversation history).");
                        }
                        Err(e) => {
                            println!("\x1b[31mCompaction failed: {e}\x1b[0m");
                        }
                    }
                    continue;
                }
                cmd => {
                    println!("Unknown command: {cmd}");
                    println!("Available: /quit, /clear, /history, /compact");
                    continue;
                }
            }
        }

        // Query the agent
        println!("\nAgent>");
        match agent.query(input).await {
            Ok(response) => {
                println!("{response}");
            }
            Err(e) => {
                println!("\x1b[31mError: {e}\x1b[0m");
            }
        }
        println!();
    }

    println!("Goodbye!");
    Ok(())
}

fn print_history<Advanced, Balanced, Fast, H>(agent: &Agent<Advanced, Balanced, Fast, H>)
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    H: Hook,
{
    let history = agent.history();
    if history.is_empty() {
        println!("No conversation history.");
    } else {
        println!("Conversation history ({} messages):", history.len());
        for msg in &history {
            let role = match msg.role() {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                Role::Tool => "Tool",
            };
            let content = truncate(msg.content(), 100);
            println!("  [{role}] {content}");
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

fn read_line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    let mut buffer = String::new();
    loop {
        if event::poll(Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Esc => {
                        print!("\r\n");
                        io::stdout().flush().ok();
                        return Ok(None);
                    }
                    KeyCode::Enter => {
                        print!("\r\n");
                        io::stdout().flush().ok();
                        return Ok(Some(buffer));
                    }
                    KeyCode::Backspace => {
                        if buffer.pop().is_some() {
                            print!("\u{8} \u{8}");
                            io::stdout().flush().ok();
                        }
                    }
                    KeyCode::Char(c) => {
                        buffer.push(c);
                        print!("{c}");
                        io::stdout().flush().ok();
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
}

struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}
