//! Standard I/O transport for MCP.
//!
//! This transport uses stdin/stdout for communication, which is the standard
//! method for MCP servers that run as subprocesses (e.g., Claude Desktop integration).

use std::future::Future;
use std::sync::atomic::{AtomicI64, Ordering};

use async_io::Async;
use futures_lite::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, warn};

use super::traits::{BidirectionalTransport, Result, Transport};
use crate::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId,
};

/// Transport using standard input/output.
///
/// Messages are sent as newline-delimited JSON. This transport is typically
/// used when running as a subprocess where the parent process communicates
/// via pipes.
pub struct StdioTransport {
    /// Async stdin reader.
    stdin: BufReader<Async<std::io::Stdin>>,
    /// Async stdout writer.
    stdout: Async<std::io::Stdout>,
    /// Next request ID.
    next_id: AtomicI64,
    /// Whether the transport is closed.
    closed: bool,
}

impl std::fmt::Debug for StdioTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdioTransport")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl StdioTransport {
    /// Create a new stdio transport.
    ///
    /// # Errors
    ///
    /// Returns an error if stdin/stdout cannot be made async.
    pub fn new() -> std::io::Result<Self> {
        let stdin = Async::new(std::io::stdin())?;
        let stdout = Async::new(std::io::stdout())?;

        Ok(Self {
            stdin: BufReader::new(stdin),
            stdout,
            next_id: AtomicI64::new(1),
            closed: false,
        })
    }

    /// Generate the next request ID.
    fn next_request_id(&self) -> RequestId {
        RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Write a message to stdout.
    async fn write_message(&mut self, json: String) -> Result<()> {
        debug!("MCP TX: {}", json);

        self.stdout.write_all(json.as_bytes()).await?;
        self.stdout.write_all(b"\n").await?;
        self.stdout.flush().await?;

        Ok(())
    }

    /// Read a message from stdin.
    async fn read_message(&mut self) -> Result<Option<JsonRpcMessage>> {
        read_message(&mut self.stdin).await
    }
}

async fn read_message(reader: &mut (impl AsyncBufRead + Unpin)) -> Result<Option<JsonRpcMessage>> {
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(None);
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        debug!("MCP RX: {}", line);
        return serde_json::from_str(line).map(Some).map_err(Into::into);
    }
}

impl Transport for StdioTransport {
    async fn request(&mut self, mut req: JsonRpcRequest) -> Result<JsonRpcResponse> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        // Assign request ID
        let id = self.next_request_id();
        req.id = id.clone();

        // Send request
        self.write_message(serde_json::to_string(&req)?).await?;

        // Read response (simple: expect next message to be our response)
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

    fn close(&mut self) -> impl Future<Output = Result<()>> + Send {
        self.closed = true;
        std::future::ready(Ok(()))
    }
}

impl BidirectionalTransport for StdioTransport {
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
}

#[cfg(test)]
mod tests {
    use futures_lite::io::{BufReader, Cursor};

    use super::read_message;

    #[tokio::test]
    async fn blank_lines_do_not_close_the_transport() {
        {
            let input = b"\n  \r\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
            let mut reader = BufReader::new(Cursor::new(input));

            assert!(read_message(&mut reader).await.unwrap().is_some());
            assert!(read_message(&mut reader).await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn invalid_json_does_not_consume_the_following_message() {
        {
            let input =
                b"not json\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
            let mut reader = BufReader::new(Cursor::new(input));

            assert!(read_message(&mut reader).await.is_err());
            assert!(read_message(&mut reader).await.unwrap().is_some());
        }
    }
}
