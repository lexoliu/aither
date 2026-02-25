//! Generic request/response channel helpers for UI-mediated tools.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Create a bounded request broker/queue pair with explicit backpressure.
#[must_use]
pub fn bounded_channel<Args, Response>(
    capacity: usize,
) -> (
    ToolRequestBroker<Args, Response>,
    ToolRequestQueue<Args, Response>,
) {
    assert!(capacity > 0, "tool request queue capacity must be > 0");
    let (tx, rx) = async_channel::bounded(capacity);
    (ToolRequestBroker { tx }, ToolRequestQueue { rx })
}

// ── RequestApprover ─────────────────────────────────────────

struct PendingEntry<P, R> {
    payload: P,
    tx: Sender<R>,
}

struct ApproverInner<P, R> {
    order: VecDeque<String>,
    pending: HashMap<String, PendingEntry<P, R>>,
}

impl<P, R> Default for ApproverInner<P, R> {
    fn default() -> Self {
        Self {
            order: VecDeque::new(),
            pending: HashMap::new(),
        }
    }
}

/// An ID-tracked, multi-request approval queue for server contexts.
///
/// Each request is assigned a unique ID, stored in FIFO order, and awaits
/// a response from the UI/caller side. Designed for web-server polling patterns
/// where the UI fetches pending requests and responds asynchronously.
///
/// Use [`event_listener::EventListener`] from [`listen`](Self::listen) to
/// efficiently wait for state changes without polling.
///
/// # Type Parameters
///
/// - `P`: Payload type (cloneable request data visible to the UI).
/// - `R`: Response type sent back to the requester.
pub struct RequestApprover<P, R> {
    inner: Mutex<ApproverInner<P, R>>,
    event: event_listener::Event,
    next_id: AtomicU64,
}

impl<P, R> std::fmt::Debug for RequestApprover<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending_count = self.inner.lock().map(|s| s.pending.len()).unwrap_or(0);
        f.debug_struct("RequestApprover")
            .field("pending_count", &pending_count)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl<P: Clone, R> Default for RequestApprover<P, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Clone, R> RequestApprover<P, R> {
    /// Create a new empty approver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(ApproverInner::default()),
            event: event_listener::Event::new(),
            next_id: AtomicU64::new(0),
        }
    }

    /// Enqueue a request and return its ID and a receiver for the response.
    ///
    /// The returned [`Receiver`] yields the response once [`respond`](Self::respond)
    /// is called with the matching ID.
    pub fn enqueue(&self, payload: P) -> (String, Receiver<R>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let (tx, rx) = async_channel::bounded(1);
        {
            let mut inner = self.inner.lock().expect("RequestApprover lock poisoned");
            inner.order.push_back(id.clone());
            inner
                .pending
                .insert(id.clone(), PendingEntry { payload, tx });
        }
        self.event.notify(usize::MAX);
        (id, rx)
    }

    /// Respond to a pending request by ID.
    ///
    /// Returns `true` if the request was found and the response was sent.
    pub fn respond(&self, id: &str, response: R) -> bool {
        let entry = {
            let mut inner = self.inner.lock().expect("RequestApprover lock poisoned");
            if let Some(pos) = inner.order.iter().position(|i| i == id) {
                inner.order.remove(pos);
            }
            inner.pending.remove(id)
        };
        let found = if let Some(entry) = entry {
            entry.tx.try_send(response).is_ok()
        } else {
            false
        };
        self.event.notify(usize::MAX);
        found
    }

    /// Return the oldest pending request (ID + cloned payload) without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<(String, P)> {
        let inner = self.inner.lock().expect("RequestApprover lock poisoned");
        let id = inner.order.front()?;
        let entry = inner.pending.get(id)?;
        Some((id.clone(), entry.payload.clone()))
    }

    /// Return the oldest pending request matching a predicate.
    #[must_use]
    pub fn peek_filtered(&self, predicate: impl Fn(&P) -> bool) -> Option<(String, P)> {
        let inner = self.inner.lock().expect("RequestApprover lock poisoned");
        inner.order.iter().find_map(|id| {
            let entry = inner.pending.get(id)?;
            if predicate(&entry.payload) {
                Some((id.clone(), entry.payload.clone()))
            } else {
                None
            }
        })
    }

    /// Get a cloned payload by request ID.
    #[must_use]
    pub fn get_payload(&self, id: &str) -> Option<P> {
        let inner = self.inner.lock().expect("RequestApprover lock poisoned");
        inner.pending.get(id).map(|e| e.payload.clone())
    }

    /// Returns `true` if there are no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.lock().expect("RequestApprover lock poisoned");
        inner.pending.is_empty()
    }

    /// Create a listener that resolves when the approver state changes.
    ///
    /// Use this for efficient async polling:
    /// ```rust,ignore
    /// loop {
    ///     if let Some((id, payload)) = approver.peek() {
    ///         // handle request
    ///     }
    ///     approver.listen().await;
    /// }
    /// ```
    pub fn listen(&self) -> event_listener::EventListener {
        self.event.listen()
    }
}
