//! MCP client implementation.
//!
//! The client connects to MCP servers and provides methods to
//! list and call tools, read resources, etc.

mod client;
mod toolset;

pub(crate) use client::McpClient;
pub use toolset::{McpConnection, McpServerConfig, McpServersConfig, McpToolService};
