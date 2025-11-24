//! aither-claude
//!
//! Claude provider integration for aither.
//!
//! This crate provides the necessary components to interact with the Claude API through the aither
//! framework. It includes a client for making requests, error handling, and data structures for
//! Claude-specific API responses.
//!
//! ## Features
//!
//! - **Client**: A `Claude` struct to interact with the Claude API.
//! - **Error Handling**: Custom error types for Claude API-specific issues.
//!
//! ## Getting Started
//!
//! To use this crate, add `aither-claude` to your `Cargo.toml` dependencies.
//!
//! ```toml
//! [dependencies]
//! aither-claude = { version = "0.1", path = "../claude" }
//! ```
//!
//! Then, you can create a `Claude` client and use it to make requests:
//!
//! ```no_run
//! use aither_claude::Claude;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = Claude::new("YOUR_CLAUDE_API_KEY".to_string());
//!     // Use the client to interact with the Claude API
//!     Ok(())
//! }
//! ```
#![allow(unused)] // For early development
#![no_std]

extern crate alloc;

use aither_core::Result;
use alloc::string::String;

/// A client for the Claude API.
#[derive(Clone, Debug)]
pub struct Claude {
    api_key: String,
}

impl Claude {
    /// Creates a new `Claude` client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

/// Represents an error that can occur when interacting with the Claude API.
#[derive(Debug)]
pub enum ClaudeError {
    // Placeholder for Claude-specific error types
    #[allow(missing_docs)]
    /// An unknown error occurred.
    Unknown,
}
