//! Interactive CLI for testing aither agents.
//!
//! This crate provides a command-line interface for testing agents interactively.
//!
//! # Features
//!
//! - Interactive REPL with streaming responses
//! - Multi-provider support (`OpenAI`, Claude, Gemini)
//! - Built-in filesystem and command tools
//! - MCP server support via configuration file
//! - Debug hook for logging tool calls
//!
//! # Usage
//!
//! Run the CLI binary with any supported provider:
//!
//! ```bash
//! # Auto-detect from available API keys
//! cargo run -p aither-cli
//!
//! # Explicit provider selection
//! OPENAI_API_KEY=xxx cargo run -p aither-cli -- --provider openai
//! ANTHROPIC_API_KEY=xxx cargo run -p aither-cli -- --provider claude
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --provider gemini
//! ```

mod hook;
mod provider;

pub use aither_cloud::{CloudError, CloudProvider};
pub use hook::DebugHook;
pub use provider::{Provider, auto_detect};
