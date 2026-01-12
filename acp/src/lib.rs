//! # ACP (Agent Client Protocol) for Aither
//!
//! This crate provides an ACP server implementation for the aither ecosystem,
//! enabling code editors like Zed, Neovim, and JetBrains IDEs to connect to
//! aither agents via a standardized JSON-RPC interface.
//!
//! ## Overview
//!
//! The Agent Client Protocol (ACP) standardizes communication between code editors
//! and AI coding agents. This crate implements the server side of ACP, allowing
//! aither agents to be used from any ACP-compatible editor.
//!
//! Key concepts:
//! - **MCP**: aither acts as *client* connecting to tool servers
//! - **ACP**: aither acts as *server* exposing the agent to editors
//!
//! Both protocols use JSON-RPC 2.0 over stdio, but the roles are reversed.
//!
//! ## Running as an ACP Server
//!
//! To expose an aither agent as an ACP server:
//!
//! ```ignore
//! use aither_acp::AcpServer;
//!
//! // Create and run the ACP server over stdio
//! let mut server = AcpServer::stdio("my-agent", "1.0.0")?;
//! server.run().await?;
//! ```
//!
//! ## Editor Integration
//!
//! ### Zed
//!
//! Add to your Zed settings (`settings.json`):
//!
//! ```json
//! {
//!   "agents": [{
//!     "name": "aither",
//!     "command": "aither",
//!     "args": ["--acp"]
//!   }]
//! }
//! ```
//!
//! ### Protocol Flow
//!
//! 1. Editor launches agent process with ACP flag
//! 2. Editor sends `initialize` request with capabilities
//! 3. Agent responds with its capabilities
//! 4. Editor sends `session/new` to create a conversation
//! 5. Editor sends `session/prompt` requests for user messages
//! 6. Agent streams `session/update` notifications with responses
//! 7. Agent returns `PromptResult` when turn is complete
//!
//! ## Session Updates
//!
//! During prompt processing, the agent sends various update notifications:
//!
//! - `AgentThoughtChunk`: Internal reasoning (extended thinking)
//! - `AgentMessageChunk`: Response text being generated
//! - `Plan`: Task list updates (from TodoWrite)
//! - `ToolCall`: Tool execution started
//! - `ToolCallUpdate`: Tool execution progress/completion
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐         JSON-RPC/stdio         ┌──────────────────┐
//! │   Code Editor   │  ──────────────────────────▶  │  aither-acp      │
//! │   (Zed, etc.)   │  ◀──────────────────────────  │  (ACP Server)    │
//! └─────────────────┘                               └────────┬─────────┘
//!                                                            │
//!                                                            ▼
//!                                                   ┌──────────────────┐
//!                                                   │  aither-agent    │
//!                                                   │  Agent<A,B,F,H>  │
//!                                                   └──────────────────┘
//! ```

mod adapter;
pub mod protocol;
mod server;
mod session;

pub use adapter::{agent_event_to_session_update, todos_to_plan};
pub use protocol::{AcpError, Result};
pub use server::AcpServer;
pub use session::AcpSession;
