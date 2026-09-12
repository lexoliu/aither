//! Transport over a generic reader/writer pair.
//!
//! [`StreamTransport`] speaks newline-delimited JSON-RPC over any
//! [`AsyncBufRead`]/[`AsyncWrite`] pair — Unix sockets, pipes, in-memory
//! cursors — wherever the caller can split a duplex stream into its halves.
//! Unlike [`StdioTransport`](super::StdioTransport) it makes no assumptions
//! about the backing file descriptors, so the same code serves a spawned MCP
//! server and a daemon listening on a socket.

use std::future::Future;
use std::sync::atomic::{AtomicI64, Ordering};

use futures_lite::io::{AsyncBufRead, AsyncWrite, AsyncWriteExt};
use tracing::{debug, warn};

use super::lines::read_message;
use super::traits::{BidirectionalTransport, Result, Transport};
use crate::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId,
};

/// Newline-delimited JSON-RPC transport over a split stream.
///
/// `recv` is cancellation-safe in the same way as the other transports: bytes
/// staged in `read_buf` survive a dropped future, so it may race other futures
/// in a select-style loop.
pub struct StreamTransport<R, W> {
    /// The read half of the stream.
    reader: R,
    /// The write half of the stream.
    writer: W,
    /// Bytes consumed from `reader` that do not yet form a complete message.
    read_buf: Vec<u8>,
    /// Next request ID.
    next_id: AtomicI64,
    /// Whether the transport is closed.
    closed: bool,
}

impl<R, W> std::fmt::Debug for StreamTransport<R, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamTransport")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<R, W> StreamTransport<R, W>
where
    R: AsyncBufRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    /// Create a transport over the given read and write halves.
    pub const fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            read_buf: Vec::new(),
            next_id: AtomicI64::new(1),
            closed: false,
        }
    }

    /// Generate the next request ID.
    fn next_request_id(&self) -> RequestId {
        RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Write one JSON-RPC message followed by a newline.
    async fn write_message(&mut self, json: String) -> Result<()> {
        debug!("MCP TX: {json}");

        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        Ok(())
    }

    /// Read the next inbound message, tolerating a dropped `recv` future.
    async fn read_message(&mut self) -> Result<Option<JsonRpcMessage>> {
        read_message(&mut self.reader, &mut self.read_buf).await
    }
}

impl<R, W> Transport for StreamTransport<R, W>
where
    R: AsyncBufRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    async fn request(&mut self, mut req: JsonRpcRequest) -> Result<JsonRpcResponse> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }

        let id = self.next_request_id();
        req.id = id.clone();

        self.write_message(serde_json::to_string(&req)?).await?;

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

impl<R, W> BidirectionalTransport for StreamTransport<R, W>
where
    R: AsyncBufRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
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
    use futures_lite::io::{BufReader, Cursor};

    use super::*;
    use crate::protocol::JsonRpcMessage;

    /// Build a transport whose reader serves `input` and whose writer
    /// collects into an in-memory cursor.
    fn transport(
        input: &'static [u8],
    ) -> StreamTransport<BufReader<Cursor<&'static [u8]>>, Cursor<Vec<u8>>> {
        StreamTransport::new(BufReader::new(Cursor::new(input)), Cursor::new(Vec::new()))
    }

    #[tokio::test]
    async fn recv_reads_newline_delimited_messages() {
        let mut transport = transport(
            b"{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n{\"jsonrpc\":\"2.0\",\"method\":\"pong\"}\n",
        );

        for method in ["ping", "pong"] {
            match transport.recv().await.expect("recv") {
                Some(JsonRpcMessage::Notification(notification)) => {
                    assert_eq!(notification.method, method);
                }
                other => panic!("expected notification {method}, got {other:?}"),
            }
        }
        assert!(transport.recv().await.expect("recv").is_none());
    }

    #[tokio::test]
    async fn notify_writes_a_single_json_line() {
        let mut transport = transport(b"");
        transport
            .notify(JsonRpcNotification::with_params(
                "notifications/initialized",
                serde_json::json!({}),
            ))
            .await
            .expect("notify");

        let written = transport.writer.into_inner();
        let line = std::str::from_utf8(&written).expect("utf8");
        assert!(line.ends_with('\n'));
        assert!(line.contains("notifications/initialized"));
    }

    #[tokio::test]
    async fn closed_transport_reports_eof() {
        let mut transport = transport(b"");
        transport.close().await.expect("close");
        assert!(transport.recv().await.expect("recv").is_none());
        assert!(
            transport
                .notify(JsonRpcNotification::new("notifications/cancelled"))
                .await
                .is_err()
        );
    }
}
