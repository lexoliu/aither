//! Child process transport for MCP.
//!
//! This transport spawns a subprocess and communicates with it via stdio pipes.

use std::process::ExitStatus;
use std::sync::atomic::{AtomicI64, Ordering};

use async_process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use futures_lite::io::{AsyncWriteExt, BufReader};
use tracing::{debug, warn};

use super::lines::read_message;
use super::traits::{BidirectionalTransport, Result, Transport};
use crate::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId,
};

/// Transport that spawns and communicates with a child process.
///
/// This is typically used to connect to MCP servers that run as separate processes,
/// such as JavaScript packages launched through Bun or standalone binaries.
pub struct ChildProcessTransport {
    /// Child process handle.
    child: Child,
    /// Child's stdin for writing.
    stdin: ChildStdin,
    /// Child's stdout for reading.
    stdout: BufReader<ChildStdout>,
    /// Bytes already consumed from stdout that do not yet form a complete
    /// message.
    ///
    /// Keeping the buffer in the transport rather than in the read future
    /// makes `recv` cancellation-safe: dropping the future mid-line loses
    /// nothing.
    read_buf: Vec<u8>,
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

        Self::from_command(Command::new(program).args(args))
    }

    /// Spawn a child process transport from a prepared [`Command`].
    ///
    /// Stdin and stdout are piped for JSON-RPC traffic; stderr is inherited so
    /// diagnostics from the child reach the parent's stderr. Use this instead
    /// of [`spawn`](Self::spawn) when the child needs a custom environment or
    /// working directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned or its pipes cannot
    /// be captured.
    pub fn from_command(command: &mut Command) -> Result<Self> {
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        Self::from_child(child)
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
            read_buf: Vec::new(),
            next_id: AtomicI64::new(1),
            closed: false,
        })
    }

    /// Generate the next request ID.
    fn next_request_id(&self) -> RequestId {
        RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Write a message to the child's stdin.
    async fn write_message(&mut self, json: String) -> Result<()> {
        debug!("MCP TX: {}", json);

        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        Ok(())
    }

    /// Read a message from the child's stdout.
    async fn read_message(&mut self) -> Result<Option<JsonRpcMessage>> {
        read_message(&mut self.stdout, &mut self.read_buf).await
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
        self.write_message(serde_json::to_string(&req)?).await?;

        // Read response
        loop {
            match self.read_message().await? {
                Some(JsonRpcMessage::Response(response)) if response.id == id => {
                    return Ok(response);
                }
                Some(_) => {}
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
        self.write_message(serde_json::to_string(&notif)?).await
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        let _ = self.child.kill();
        let _ = self.child.status().await;
        Ok(())
    }

    async fn exit_status(&mut self) -> Option<ExitStatus> {
        if let Ok(Some(status)) = self.child.try_status() {
            Some(status)
        } else {
            // The pipe is gone but the child is still running; it can no
            // longer be reached, so collect a real status by reaping it.
            let _ = self.child.kill();
            self.child.status().await.ok()
        }
    }
}

impl BidirectionalTransport for ChildProcessTransport {
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>> {
        if self.closed {
            return Ok(None);
        }
        loop {
            match self.read_message().await {
                Err(McpError::Serialization(error)) => {
                    warn!(%error, "ignoring invalid JSON-RPC input");
                }
                result => return result,
            }
        }
    }

    async fn respond(&mut self, response: JsonRpcResponse) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        self.write_message(serde_json::to_string(&response)?).await
    }

    async fn send_request(&mut self, req: JsonRpcRequest) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        self.write_message(serde_json::to_string(&req)?).await
    }
}
