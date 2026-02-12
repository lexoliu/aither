//! Generic request/response channel helpers for UI-mediated tools.

use async_channel::{Receiver, Sender};

const ERR_TOOL_REQUEST_UNAVAILABLE: &str = include_str!("texts/tool_request_unavailable.txt");
const ERR_TOOL_REQUEST_CANCELLED: &str = include_str!("texts/tool_request_cancelled.txt");

/// A UI-bound request carrying tool arguments and a response sender.
#[derive(Debug)]
pub struct ToolRequest<Args, Response> {
    /// Tool arguments provided by the agent.
    pub args: Args,
    response_tx: Sender<Response>,
}

impl<Args, Response> ToolRequest<Args, Response> {
    /// Create a new request with its response channel.
    #[must_use]
    pub const fn new(args: Args, response_tx: Sender<Response>) -> Self {
        Self { args, response_tx }
    }

    /// Respond to the request without blocking the UI thread.
    pub fn respond(self, response: Response) -> Result<(), async_channel::TrySendError<Response>> {
        self.response_tx.try_send(response)
    }
}

/// Request broker for UI-mediated tools.
#[derive(Debug, Clone)]
pub struct ToolRequestBroker<Args, Response> {
    tx: Sender<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestBroker<Args, Response> {
    /// Send a request and await the response.
    pub async fn request(&self, args: Args) -> anyhow::Result<Response> {
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.tx
            .send(ToolRequest::new(args, response_tx))
            .await
            .map_err(|_| anyhow::anyhow!(ERR_TOOL_REQUEST_UNAVAILABLE.trim()))?;

        response_rx
            .recv()
            .await
            .map_err(|_| anyhow::anyhow!(ERR_TOOL_REQUEST_CANCELLED.trim()))
    }
}

/// Queue for pending tool requests.
#[derive(Debug)]
pub struct ToolRequestQueue<Args, Response> {
    rx: Receiver<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestQueue<Args, Response> {
    /// Await the next request.
    pub async fn next(&self) -> Option<ToolRequest<Args, Response>> {
        self.rx.recv().await.ok()
    }

    /// Returns true if there are no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

/// Create a new request broker/queue pair.
#[must_use]
pub fn channel<Args, Response>() -> (
    ToolRequestBroker<Args, Response>,
    ToolRequestQueue<Args, Response>,
) {
    let (tx, rx) = async_channel::unbounded();
    (ToolRequestBroker { tx }, ToolRequestQueue { rx })
}
