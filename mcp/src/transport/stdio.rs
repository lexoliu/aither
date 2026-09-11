//! Standard I/O transport for MCP.
//!
//! This transport uses stdin/stdout for communication, which is the standard
//! method for MCP servers that run as subprocesses (e.g., Claude Desktop integration).

use std::future::Future;
use std::sync::atomic::{AtomicI64, Ordering};

use blocking::Unblock;
use futures_lite::io::{AsyncWriteExt, BufReader};
use tracing::{debug, warn};

use super::lines::read_message;
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
    /// Stdin, read on a blocking thread and surfaced as an async reader.
    stdin: BufReader<Unblock<std::io::Stdin>>,
    /// Stdout, written on a blocking thread.
    stdout: Unblock<std::io::Stdout>,
    /// Bytes already consumed from stdin that do not yet form a complete
    /// message, keeping `recv` cancellation-safe.
    read_buf: Vec<u8>,
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
    /// Standard input and output are driven on blocking threads rather than by
    /// the reactor. Readiness polling cannot carry them: on Windows the
    /// reactor reaches its readiness model through AFD, which works only on
    /// sockets, so a console handle or a pipe cannot be registered at all.
    /// Completion-based I/O does not help either -- this is why tokio also
    /// backs its own stdin with a blocking pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stdin: BufReader::new(Unblock::new(std::io::stdin())),
            stdout: Unblock::new(std::io::stdout()),
            read_buf: Vec::new(),
            next_id: AtomicI64::new(1),
            closed: false,
        }
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
        read_message(&mut self.stdin, &mut self.read_buf).await
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
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

    async fn send_request(&mut self, req: JsonRpcRequest) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        self.write_message(serde_json::to_string(&req)?).await
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use futures_lite::io::{AsyncRead, BufReader, Cursor};

    use super::read_message;

    /// A reader that behaves like `Unblock<Stdin>` at EOF: the first poll
    /// after the data reports EOF, every later poll goes `Pending` because a
    /// fresh read has been handed to the blocking pool. `futures-lite`'s
    /// `fill_buf` polls twice and panics on that second `Pending`.
    struct PendingAfterEof {
        data: Cursor<&'static [u8]>,
        eof_reported: bool,
    }

    impl AsyncRead for PendingAfterEof {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            if self.eof_reported {
                return Poll::Pending;
            }
            let read = futures_lite::ready!(Pin::new(&mut self.data).poll_read(cx, buf))?;
            if read == 0 {
                self.eof_reported = true;
            }
            Poll::Ready(Ok(read))
        }
    }

    #[tokio::test]
    async fn eof_after_the_last_message_closes_the_transport() {
        let input: &[u8] = b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
        let mut reader = BufReader::new(PendingAfterEof {
            data: Cursor::new(input),
            eof_reported: false,
        });
        let mut read_buf = Vec::new();

        assert!(
            read_message(&mut reader, &mut read_buf)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            read_message(&mut reader, &mut read_buf)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn an_unterminated_final_line_is_still_a_message() {
        let input = b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}";
        let mut reader = BufReader::new(Cursor::new(input));
        let mut read_buf = Vec::new();

        assert!(
            read_message(&mut reader, &mut read_buf)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            read_message(&mut reader, &mut read_buf)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn blank_lines_do_not_close_the_transport() {
        {
            let input = b"\n  \r\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
            let mut reader = BufReader::new(Cursor::new(input));
            let mut read_buf = Vec::new();

            assert!(
                read_message(&mut reader, &mut read_buf)
                    .await
                    .unwrap()
                    .is_some()
            );
            assert!(
                read_message(&mut reader, &mut read_buf)
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[tokio::test]
    async fn invalid_json_does_not_consume_the_following_message() {
        {
            let input =
                b"not json\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
            let mut reader = BufReader::new(Cursor::new(input));
            let mut read_buf = Vec::new();

            assert!(read_message(&mut reader, &mut read_buf).await.is_err());
            assert!(
                read_message(&mut reader, &mut read_buf)
                    .await
                    .unwrap()
                    .is_some()
            );
        }
    }
}
