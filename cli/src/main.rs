//! Interactive CLI for testing aither agents.
//!
//! A REPL-style interface for testing agents with OpenAI, Claude, or Gemini.
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
//! # With specific model (auto-detects provider from model name)
//! OPENAI_API_KEY=xxx cargo run -p aither-cli -- --model gpt-4o
//! ANTHROPIC_API_KEY=xxx cargo run -p aither-cli -- --model claude-sonnet-4-20250514
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --model gemini-2.5-pro
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
//!
//! # Default Tools
//!
//! The CLI includes the following tools by default:
//! - **FileSystem**: Read/write/search files in the current directory
//! - **Command**: Execute shell commands
//! - **Todo**: Track tasks
//! - **Context7**: Look up library documentation (via MCP)

mod cloud;
mod hook;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use aither_agent::specialized::{SubagentType, TaskTool};
use aither_agent::{Agent, Hook};
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

    /// Disable filesystem tool.
    #[arg(long)]
    no_fs: bool,

    /// Disable command execution tool.
    #[arg(long)]
    no_command: bool,

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
            println!("Commands: /quit, /clear, /history");
        }
        println!();
    }

    // Headless mode: run single prompt and exit
    if let Some(ref prompt) = args.prompt {
        return run_headless(cloud, &args, prompt).await;
    }

    run_repl(cloud, &args).await
}

async fn build_agent(cloud: CloudProvider, args: &Args) -> Result<Agent<CloudProvider, aither_agent::HCons<DebugHook, ()>>> {
    // Clone cloud for task tool before moving into builder
    let cloud_for_tasks = cloud.clone();

    // Build the agent
    let mut builder = Agent::builder(cloud).hook(DebugHook);

    // Add system prompt
    if let Some(ref system) = args.system {
        builder = builder.system_prompt(system);
    } else {
        builder = builder.system_prompt(include_str!("prompts/system.md"));
    }

    // Add filesystem tool
    if !args.no_fs {
        builder = builder.tool(aither_agent::filesystem::FileSystemTool::new("."));
    }

    // Add command tool
    if !args.no_command {
        builder = builder.tool(aither_agent::command::CommandTool::new("."));
    }

    // Add todo tool
    builder = builder.tool(aither_agent::TodoTool::new());

    // Add task tool with explore subagent
    // Note: SubagentType builders return AgentBuilder (not built Agent)
    // so that TaskTool can add display hooks for real-time feedback
    let mut task_tool = TaskTool::new(cloud_for_tasks);
    task_tool.register(
        "explore",
        SubagentType::new(
            "Fast codebase explorer for finding files, searching code, and analyzing structure",
            |llm| {
                Agent::builder(llm)
                    .system_prompt(include_str!("prompts/explore.md"))
                    .tool(aither_agent::filesystem::FileSystemTool::new("."))
            },
        ),
    );
    task_tool.register(
        "plan",
        SubagentType::new(
            "Software architect for designing implementation plans",
            |llm| {
                Agent::builder(llm)
                    .system_prompt(include_str!("prompts/plan.md"))
                    .tool(aither_agent::filesystem::FileSystemTool::new("."))
            },
        ),
    );
    builder = builder.tool(task_tool);

    // Add Context7 MCP server by default (documentation lookup)
    if !args.no_context7 {
        match McpConnection::http("https://mcp.context7.com/mcp").await {
            Ok(conn) => {
                if !args.quiet {
                    println!("Connected to Context7 MCP server");
                }
                builder = builder.mcp(conn);
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
            builder = builder.mcp(conn);
        }
    }

    Ok(builder.build())
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
                cmd => {
                    println!("Unknown command: {cmd}");
                    println!("Available: /quit, /clear, /history");
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

fn print_history<LLM: LanguageModel, H: Hook>(agent: &Agent<LLM, H>) {
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
