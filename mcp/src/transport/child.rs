//! Child process transport for MCP.
//!
//! This transport spawns a subprocess and communicates with it via stdio pipes.

use std::sync::atomic::{AtomicI64, Ordering};

use async_process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use futures_lite::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::debug;

use super::traits::{Result, Transport};
use crate::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId,
};

/// Transport that spawns and communicates with a child process.
///
/// This is typically used to connect to MCP servers that run as separate processes,
/// such as npm packages or standalone binaries.
pub struct ChildProcessTransport {
    /// Child process handle.
    child: Child,
    /// Child's stdin for writing.
    stdin: ChildStdin,
    /// Child's stdout for reading.
    stdout: BufReader<ChildStdout>,
    /// Next request ID.
    next_id: AtomicI64,
    /// Whether the transport is closed.
    closed: bool,
}

impl std::fmt::Debug for ChildProcessTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildProcessTransport")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl ChildProcessTransport {
    /// Spawn a new child process transport.
    ///
    /// # Arguments
    ///
    /// * `program` - The program to execute.
    /// * `args` - Arguments to pass to the program.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned.
    pub fn spawn(program: &str, args: &[&str]) -> Result<Self> {
        debug!("Spawning MCP server: {} {:?}", program, args);

        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture child stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture child stdout".to_string()))?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: AtomicI64::new(1),
            closed: false,
        })
    }

    /// Create a transport from an existing child process.
    ///
    /// # Arguments
    ///
    /// * `child` - Child process with stdin/stdout captured.
    ///
    /// # Errors
    ///
    /// Returns an error if stdin/stdout are not available.
    pub fn from_child(mut child: Child) -> Result<Self> {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("Child process stdin not available".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("Child process stdout not available".to_string()))?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: AtomicI64::new(1),
            closed: false,
        })
    }

    /// Generate the next request ID.
    fn next_request_id(&self) -> RequestId {
        RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Write a message to the child's stdin.
    async fn write_message(&mut self, msg: &impl serde::Serialize) -> Result<()> {
        let json = serde_json::to_string(msg)?;
        debug!("MCP TX: {}", json);

        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        Ok(())
    }

    /// Read a message from the child's stdout.
    async fn read_message(&mut self) -> Result<Option<JsonRpcMessage>> {
        let mut line = String::new();
        match self.stdout.read_line(&mut line).await {
            Ok(0) => Ok(None), // EOF
            Ok(_) => {
                let line = line.trim();
                if line.is_empty() {
                    return Ok(None);
                }
                debug!("MCP RX: {}", line);
                let msg: JsonRpcMessage = serde_json::from_str(line)?;
                Ok(Some(msg))
            }
            Err(e) => Err(McpError::Io(e)),
        }
    }
}

impl Transport for ChildProcessTransport {
    async fn request(&mut self, mut req: JsonRpcRequest) -> Result<JsonRpcResponse> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        // Assign request ID
        let id = self.next_request_id();
        req.id = id.clone();

        // Send request
        self.write_message(&req).await?;

        // Read response
        loop {
            match self.read_message().await? {
                Some(JsonRpcMessage::Response(response)) if response.id == id => {
                    return Ok(response);
                }
                Some(_) => {
                    // Skip non-matching messages
                    continue;
                }
                None => {
                    return Err(McpError::ConnectionClosed);
                }
            }
        }
    }

    async fn notify(&mut self, notif: JsonRpcNotification) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        self.write_message(&notif).await
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        let _ = self.child.kill();
        let _ = self.child.status().await;
        Ok(())
    }
}
