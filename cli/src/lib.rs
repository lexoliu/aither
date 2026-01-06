//! Interactive CLI for testing aither agents.
//!
//! This crate provides a command-line interface for testing agents interactively.
//!
//! # Features
//!
//! - Interactive REPL with streaming responses
//! - Built-in filesystem and command tools
//! - MCP server support via configuration file
//! - Debug hook for logging tool calls
//!
//! # Usage
//!
//! Run the CLI binary:
//!
//! ```bash
//! GEMINI_API_KEY=xxx cargo run -p aither-cli
//! ```

mod hook;

pub use hook::DebugHook;
