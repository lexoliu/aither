//! aither-claude
//!
//! Claude provider integration for aither.
//!
//! This crate provides the necessary components to interact with the Anthropic Claude API
//! through the aither framework. It includes a client for making requests, error handling,
//! and full support for Claude's capabilities.
//!
//! ## Features
//!
//! - **Streaming**: Full SSE streaming support for real-time responses
//! - **Tool Use**: Function calling with automatic iteration loop
//! - **Vision**: Image understanding via base64 or URL references
//! - **Extended Thinking**: Support for Claude's reasoning/thinking mode
//!
//! ## Getting Started
//!
//! ```ignore
//! use aither_claude::Claude;
//! use aither_core::{LanguageModel, llm::oneshot};
//!
//! let client = Claude::new(std::env::var("ANTHROPIC_API_KEY")?);
//!
//! // Simple chat
//! let response = client.respond(oneshot(
//!     "You are a helpful assistant.",
//!     "Explain Rust ownership in one paragraph."
//! )).await?;
//!
//! println!("{response}");
//! ```
//!
//! ## Custom Configuration
//!
//! ```no_run
//! use aither_claude::{Claude, CLAUDE_OPUS_4_0};
//!
//! let client = Claude::builder("your-api-key")
//!     .model(CLAUDE_OPUS_4_0)
//!     .max_tokens(8192)
//!     .build();
//! ```
//!
//! ## Vision Support
//!
//! Claude can analyze images passed as attachments:
//!
//! ```ignore
//! use aither_claude::Claude;
//! use aither_core::{LanguageModel, llm::{LLMRequest, Message}};
//!
//! let client = Claude::new(std::env::var("ANTHROPIC_API_KEY")?);
//!
//! let message = Message::user("What's in this image?")
//!     .with_attachment("https://example.com/image.jpg");
//!
//! let response = client.respond(LLMRequest::new([message])).await?;
//! ```

mod client;
mod constant;
mod error;
mod request;
mod response;

pub use client::{Builder, Claude};
pub use constant::*;
pub use error::ClaudeError;
