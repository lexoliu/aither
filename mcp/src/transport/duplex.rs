//! In-memory duplex transport.
//!
//! A [`DuplexTransport`] pair carries JSON-RPC messages between two peers in
//! the same process. It is useful for tests and for embedding a JSON-RPC peer
//! without spawning a child process.

use std::future::Future;
use std::sync::atomic::{AtomicI64, Ordering};

use async_channel::{Receiver, Sender};

use super::traits::{BidirectionalTransport, Result, Transport};
use crate::protocol::{
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError, RequestId,
};

/// One end of an in-memory bidirectional JSON-RPC channel.
///
/// Create both ends with [`DuplexTransport::pair`]: every message sent on one
/// end arrives on the other. Dropping an end closes the channel; the peer's
/// [`recv`](BidirectionalTransport::recv) then reports `None`.
pub struct DuplexTransport {
    /// Messages received from the peer.
    incoming: Receiver<JsonRpcMessage>,
    /// Messages sent to the peer.
    outgoing: Sender<JsonRpcMessage>,
    /// Next request ID.
    next_id: AtomicI64,
    /// Whether the transport is closed.
    closed: bool,
}

impl std::fmt::Debug for DuplexTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexTransport")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl DuplexTransport {
    /// Create a connected pair of duplex transports.
    #[must_use]
    pub fn pair() -> (Self, Self) {
        let (a_out, a_in) = async_channel::unbounded();
        let (b_out, b_in) = async_channel::unbounded();
        (
            Self {
                incoming: a_in,
                outgoing: b_out,
                next_id: AtomicI64::new(1),
                closed: false,
            },
            Self {
                incoming: b_in,
                outgoing: a_out,
                next_id: AtomicI64::new(1),
                closed: false,
            },
        )
    }

    /// Generate the next request ID.
    fn next_request_id(&self) -> RequestId {
        RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Queue a message for the peer.
    fn send(&self, msg: JsonRpcMessage) -> Result<()> {
        if self.closed {
            return Err(McpError::ConnectionClosed);
        }
        self.outgoing
            .try_send(msg)
            .map_err(|_| McpError::ConnectionClosed)
    }

    /// Receive the next message from the peer.
    async fn recv_message(&self) -> Result<Option<JsonRpcMessage>> {
        if self.closed {
            return Ok(None);
        }
        self.incoming
            .recv()
            .await
            .map_or(Ok(None), |msg| Ok(Some(msg)))
    }
}

impl Transport for DuplexTransport {
    async fn request(&mut self, mut req: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let id = self.next_request_id();
        req.id = id.clone();
        self.send(JsonRpcMessage::Request(req))?;

        loop {
            match self.recv_message().await? {
                Some(JsonRpcMessage::Response(response)) if response.id == id => {
                    return Ok(response);
                }
                Some(_) => {}
                None => return Err(McpError::ConnectionClosed),
            }
        }
    }

    fn notify(&mut self, notif: JsonRpcNotification) -> impl Future<Output = Result<()>> + Send {
        std::future::ready(self.send(JsonRpcMessage::Notification(notif)))
    }

    fn close(&mut self) -> impl Future<Output = Result<()>> + Send {
        self.closed = true;
        self.outgoing.close();
        std::future::ready(Ok(()))
    }
}

impl BidirectionalTransport for DuplexTransport {
    fn recv(&mut self) -> impl Future<Output = Result<Option<JsonRpcMessage>>> + Send {
        self.recv_message()
    }

    fn respond(&mut self, response: JsonRpcResponse) -> impl Future<Output = Result<()>> + Send {
        std::future::ready(self.send(JsonRpcMessage::Response(response)))
    }

    fn send_request(&mut self, req: JsonRpcRequest) -> impl Future<Output = Result<()>> + Send {
        std::future::ready(self.send(JsonRpcMessage::Request(req)))
    }
}
