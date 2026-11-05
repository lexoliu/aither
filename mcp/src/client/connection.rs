//! The receive loop: the sole owner of a client's transport.

use std::collections::HashMap;
use std::sync::Arc;

use async_channel::{Receiver, RecvError, Sender};
use futures_lite::future::or;
use futures_util::stream::{FuturesUnordered, StreamExt};
use tracing::{debug, warn};

use super::handler::{McpHandler, ServerNotification};
use crate::protocol::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
    RequestId,
};
use crate::transport::BidirectionalTransport;

/// What the client asks the loop to write to the transport.
pub(super) enum Command {
    /// A request; `reply` receives its response, or the error that ended the
    /// connection first.
    Request {
        request: JsonRpcRequest,
        reply: Sender<Result<JsonRpcResponse, McpError>>,
    },
    /// A notification; `reply` receives the outcome of the write.
    Notification {
        notification: JsonRpcNotification,
        reply: Sender<Result<(), McpError>>,
    },
}

impl Command {
    fn fail(self, ending: &Ending) {
        match self {
            Self::Request { reply, .. } => {
                let _ = reply.try_send(Err(ending.error()));
            }
            Self::Notification { reply, .. } => {
                let _ = reply.try_send(Err(ending.error()));
            }
        }
    }
}

/// Why the loop stopped.
enum Ending {
    /// The peer hung up, or the client closed the connection.
    Closed(Option<String>),
    /// Reading or writing the transport failed.
    Failed(String),
}

impl Ending {
    fn error(&self) -> McpError {
        match self {
            Self::Closed(None) => McpError::ConnectionClosed,
            Self::Closed(Some(status)) => {
                McpError::Transport(format!("connection closed: server process {status}"))
            }
            Self::Failed(message) => McpError::Transport(format!("connection failed: {message}")),
        }
    }
}

/// One step of the loop.
enum Event {
    /// A command from the client, or every client handle closing.
    Command(Result<Command, RecvError>),
    /// A message from the server, or the end of the transport.
    Incoming(Result<Option<JsonRpcMessage>, McpError>),
    /// A notification handler finished.
    Handled,
}

/// Run the connection until the transport ends or the client closes it.
///
/// Routes each response to the request awaiting it, answers `ping`, answers
/// every other server request with method-not-found (the client advertises no
/// capability that would let a server send one), and hands notifications to
/// `handler`. On exit the transport is closed and every request still
/// pending, queued or sent later fails with an error naming why. `alive` is
/// dropped last, which is how the client learns the loop has finished.
pub(super) async fn run<T, H>(
    mut transport: T,
    commands: Receiver<Command>,
    handler: Arc<H>,
    alive: Sender<()>,
) where
    T: BidirectionalTransport,
    H: McpHandler,
{
    let mut pending: HashMap<RequestId, Sender<Result<JsonRpcResponse, McpError>>> = HashMap::new();
    let mut handling = FuturesUnordered::new();

    let ending = loop {
        // Handlers are polled first so they keep progressing while the
        // server's traffic is a steady stream.
        let event = or(
            async {
                // An empty set resolves to `None` at once; park instead.
                match handling.next().await {
                    Some(()) => Event::Handled,
                    None => std::future::pending().await,
                }
            },
            or(async { Event::Command(commands.recv().await) }, async {
                Event::Incoming(transport.recv().await)
            }),
        )
        .await;

        match event {
            Event::Command(Ok(Command::Request { request, reply })) => {
                pending.insert(request.id.clone(), reply);
                if let Err(error) = transport.send_request(request).await {
                    break Ending::Failed(error.to_string());
                }
            }
            Event::Command(Ok(Command::Notification {
                notification,
                reply,
            })) => {
                let sent = transport.notify(notification).await;
                let failed = sent.as_ref().err().map(ToString::to_string);
                let _ = reply.try_send(sent);
                if let Some(message) = failed {
                    break Ending::Failed(message);
                }
            }
            Event::Command(Err(_)) | Event::Incoming(Ok(None)) => break Ending::Closed(None),
            Event::Incoming(Err(error)) => break Ending::Failed(error.to_string()),
            Event::Incoming(Ok(Some(JsonRpcMessage::Response(response)))) => {
                if let Some(reply) = pending.remove(&response.id) {
                    let _ = reply.try_send(Ok(response));
                } else {
                    debug!(id = ?response.id, "ignoring response with no pending request");
                }
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Request(request)))) => {
                if let Err(error) = transport.respond(answer(request)).await {
                    break Ending::Failed(error.to_string());
                }
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Notification(notification)))) => {
                handling.push(notify(handler.clone(), notification));
            }
            Event::Handled => {}
        }
    };

    drop(handling);
    let exit = transport.exit_status().await;
    let ending = match (ending, exit) {
        (Ending::Closed(_), Some(exit)) => Ending::Closed(Some(exit.to_string())),
        (ending, _) => ending,
    };
    let _ = transport.close().await;
    debug!(error = %ending.error(), "MCP connection closed");

    commands.close();
    while let Ok(command) = commands.try_recv() {
        command.fail(&ending);
    }
    for (_, reply) in pending {
        let _ = reply.try_send(Err(ending.error()));
    }
    drop(alive);
}

/// The response to a request the server sent.
fn answer(request: JsonRpcRequest) -> JsonRpcResponse {
    if request.method == "ping" {
        JsonRpcResponse::success(request.id, serde_json::json!({}))
    } else {
        warn!(method = %request.method, "rejecting a server request for an unadvertised capability");
        JsonRpcResponse::error(request.id, JsonRpcError::method_not_found(&request.method))
    }
}

async fn notify<H: McpHandler>(handler: Arc<H>, notification: JsonRpcNotification) {
    handler
        .notification(ServerNotification::decode(notification))
        .await;
}
