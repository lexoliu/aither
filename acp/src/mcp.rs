//! Connecting the MCP servers a client offers a session.

use aither_mcp::transport::HttpTransport;
use aither_mcp::{McpConnection, McpError};
use async_process::Command;
use executor_core::AnyExecutor;

use crate::protocol::McpServer;

/// Why a session's MCP servers could not be connected.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    /// The server uses a transport the agent did not advertise.
    #[error("MCP server '{0}' uses SSE, which this agent does not support")]
    Unsupported(String),
    /// The server did not connect.
    #[error("could not connect MCP server '{server}': {source}")]
    Failed {
        server: String,
        #[source]
        source: McpError,
    },
}

/// Connects every server concurrently and returns the connections in the
/// order the servers were given.
pub async fn connect_all(
    executor: &AnyExecutor,
    servers: &[McpServer],
) -> Result<Vec<McpConnection>, ConnectError> {
    futures_util::future::try_join_all(servers.iter().map(|server| connect(executor, server))).await
}

async fn connect(
    executor: &AnyExecutor,
    server: &McpServer,
) -> Result<McpConnection, ConnectError> {
    let connected = match server {
        McpServer::Sse(server) => return Err(ConnectError::Unsupported(server.name.clone())),
        McpServer::Http(server) => {
            let transport = server
                .headers
                .iter()
                .fold(HttpTransport::new(&server.url), |transport, header| {
                    transport.with_header(&header.name, &header.value)
                });
            McpConnection::connect(executor, transport, ()).await
        }
        McpServer::Stdio(server) => {
            let mut command = Command::new(&server.command);
            command
                .args(&server.args)
                .envs(server.env.iter().map(|var| (&var.name, &var.value)));
            McpConnection::spawn_command(executor, &mut command).await
        }
    };
    connected.map_err(|source| ConnectError::Failed {
        server: server.name().to_string(),
        source,
    })
}
