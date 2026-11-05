//! # MCP (Model Context Protocol) for Aither
//!
//! This crate provides MCP client and server implementations for the aither ecosystem.
//!
//! ## Overview
//!
//! The Model Context Protocol (MCP) is a standard protocol for connecting AI applications
//! with external data sources and tools. This crate implements:
//!
//! - **MCP Client**: Connect to MCP servers and use their tools with aither agents
//! - **MCP Server**: Expose aither tools as an MCP server for external clients
//!
//! ## Using MCP Tools with an Agent
//!
//! To use MCP server tools with an aither agent, first connect to the server,
//! then pass the connection to the agent builder. Every connection runs a
//! receive loop on the [`executor_core::Executor`] you pass (for example
//! `executor_core::tokio::TokioGlobal`); it answers the server's pings and
//! delivers its notifications to an [`McpHandler`]:
//!
//! ```ignore
//! use aither_agent::Agent;
//! use aither_mcp::McpConnection;
//!
//! // Connect to an MCP server (spawns a child process)
//! let conn = McpConnection::spawn(&executor, "npx", &["-y", "@modelcontextprotocol/server-filesystem", "/path"]).await?;
//!
//! // Build agent with the MCP connection
//! let agent = Agent::builder(llm)
//!     .system_prompt("You are a helpful assistant.")
//!     .mcp(conn)  // Register MCP tools
//!     .build();
//!
//! // The agent can now use tools from the MCP server
//! let response = agent.prompt("List files in the current directory").await?;
//! ```
//!
//! ### Multiple MCP Servers
//!
//! You can connect to multiple MCP servers:
//!
//! ```ignore
//! let filesystem = McpConnection::spawn(&executor, "npx", &["-y", "@modelcontextprotocol/server-filesystem", "/"]).await?;
//! let github = McpConnection::spawn(&executor, "npx", &["-y", "@modelcontextprotocol/server-github"]).await?;
//!
//! let agent = Agent::builder(llm)
//!     .mcp(filesystem)
//!     .mcp(github)
//!     .build();
//! ```
//!
//! ### Loading from Configuration
//!
//! Load MCP server configurations from a JSON file (compatible with Claude Desktop format):
//!
//! ```ignore
//! use aither_mcp::{McpConnection, McpServersConfig};
//!
//! // JSON configuration format:
//! // {
//! //   "filesystem": {
//! //     "command": "npx",
//! //     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
//! //   },
//! //   "github": {
//! //     "command": "npx",
//! //     "args": ["-y", "@modelcontextprotocol/server-github"],
//! //     "env": { "GITHUB_TOKEN": "..." }
//! //   }
//! // }
//!
//! let config: McpServersConfig = serde_json::from_str(&config_json)?;
//! let connections = McpConnection::from_configs(&executor, &config).await?;
//!
//! let mut builder = Agent::builder(llm);
//! for (_name, conn) in connections {
//!     builder = builder.mcp(conn);
//! }
//! let agent = builder.build();
//! ```
//!
//! ### HTTP-based MCP Servers
//!
//! Connect to MCP servers over HTTP (requires the `http` feature):
//!
//! ```ignore
//! let conn = McpConnection::http(&executor, "http://localhost:3000/mcp").await?;
//!
//! // With authentication
//! let conn = McpConnection::http_with_auth(&executor, "http://localhost:3000/mcp", "Bearer token").await?;
//! ```
//!
//! ## Exposing Tools as an MCP Server
//!
//! To expose aither tools as an MCP server (e.g., for Claude Desktop):
//!
//! ```ignore
//! use aither_core::llm::tool::Tools;
//! use aither_mcp::McpServer;
//!
//! // Register your tools
//! let mut tools = Tools::new();
//! tools.register(MyCustomTool::new());
//! tools.register(AnotherTool::new());
//!
//! // Create and run the MCP server over stdio
//! let mut server = McpServer::stdio(tools, "my-server", "1.0.0");
//! server.run().await?;
//! ```
//!
//! ### Claude Desktop Integration
//!
//! To use your MCP server with Claude Desktop, add it to your Claude Desktop config
//! (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "my-server": {
//!       "command": "/path/to/my-server-binary"
//!     }
//!   }
//! }
//! ```
//!
//! ## Error Handling
//!
//! All MCP operations return [`McpError`], which can be matched for specific error handling:
//!
//! ```ignore
//! use aither_mcp::{McpConnection, McpError};
//!
//! match McpConnection::spawn(&executor, "npx", &["-y", "some-server"]).await {
//!     Ok(conn) => { /* use connection */ }
//!     Err(McpError::Io(e)) => eprintln!("IO error: {e}"),
//!     Err(McpError::InvalidConfig(msg)) => eprintln!("Config error: {msg}"),
//!     Err(McpError::Timeout) => eprintln!("Connection timed out"),
//!     Err(e) => eprintln!("Other error: {e}"),
//! }
//! ```

#[cfg(feature = "client")]
mod client;
pub mod protocol;
#[cfg(feature = "server")]
mod server;
#[cfg(all(test, feature = "http"))]
mod test_server;
pub mod transport;

// Re-export main types
#[cfg(feature = "client")]
pub use client::{
    McpClient, McpConnection, McpHandler, McpServerConfig, McpServersConfig, McpToolService,
    ServerNotification, call_result_to_tool_result, register_terminal_commands,
};
pub use protocol::{CallToolResult, Content, McpError, McpToolDefinition};
#[cfg(feature = "server")]
pub use server::McpServer;
