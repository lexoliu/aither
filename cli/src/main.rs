//! Interactive CLI for testing aither agents.
//!
//! A REPL-style interface for testing agents with Gemini, tools, and MCP support.
//!
//! # Usage
//!
//! ```bash
//! # Basic usage
//! GEMINI_API_KEY=xxx cargo run -p aither-cli
//!
//! # With MCP servers
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --mcp servers.json
//!
//! # With specific model
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --model gemini-2.5-pro
//! ```

mod hook;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use aither_agent::{Agent, Hook};
use aither_core::LanguageModel;
use aither_core::llm::Role;
use aither_gemini::Gemini;
use aither_mcp::{McpConnection, McpServersConfig};
use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use tracing_subscriber::EnvFilter;

use crate::hook::DebugHook;

/// Interactive CLI for testing aither agents.
#[derive(Parser, Debug)]
#[command(name = "aither", version, about)]
struct Args {
    /// Gemini model to use.
    #[arg(short, long, default_value = "gemini-flash-latest")]
    model: String,

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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let api_key =
        std::env::var("GEMINI_API_KEY").context("set GEMINI_API_KEY in your environment")?;

    let gemini = Gemini::new(api_key).with_text_model(&args.model);

    println!("Aither Agent CLI");
    println!("Model: {}", args.model);
    println!("Commands: /quit, /clear, /history");
    println!();

    run_repl(gemini, &args).await
}

async fn run_repl(gemini: Gemini, args: &Args) -> Result<()> {
    // Build the agent
    let mut builder = Agent::builder(gemini).hook(DebugHook);

    // Add system prompt
    if let Some(ref system) = args.system {
        builder = builder.system_prompt(system);
    } else {
        builder = builder.system_prompt(
            "You are a helpful AI assistant. Use the available tools to help the user with their tasks.",
        );
    }

    // Add filesystem tool
    if !args.no_fs {
        builder = builder.tool(aither_agent::filesystem::FileSystemTool::read_only("."));
    }

    // Add command tool
    if !args.no_command {
        builder = builder.tool(aither_agent::command::CommandTool::new("."));
    }

    // Add MCP connections
    if let Some(ref mcp_path) = args.mcp {
        let config_str = std::fs::read_to_string(mcp_path)
            .with_context(|| format!("failed to read MCP config from {}", mcp_path.display()))?;
        let config: McpServersConfig = serde_json::from_str(&config_str)
            .with_context(|| "failed to parse MCP config")?;

        let connections = McpConnection::from_configs(&config).await?;
        for (name, conn) in connections {
            println!("Connected to MCP server: {name}");
            builder = builder.mcp(conn);
        }
    }

    let mut agent = builder.build();

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
