//! Handler trait for agent-to-client traffic.

use std::future::Future;

use aither_mcp::protocol::{JsonRpcError, JsonRpcNotification};
use serde_json::Value;
use tracing::debug;

use crate::protocol::{
    ClientCapabilities, ElicitationCompleteParams, ElicitationCreateParams,
    ElicitationCreateResult, ExtMethod, ReadTextFileParams, ReadTextFileResult,
    RequestPermissionParams, RequestPermissionResult, SessionNotification, TerminalCreateParams,
    TerminalCreateResult, TerminalExitStatus, TerminalKillParams, TerminalKillResult,
    TerminalOutputParams, TerminalOutputResult, TerminalReleaseParams, TerminalReleaseResult,
    TerminalWaitForExitParams, WriteTextFileParams, WriteTextFileResult,
};

/// Handles traffic the agent initiates toward the client.
///
/// Implementors receive every `session/update` notification and every
/// agent-to-client request the connection accepts. The methods are called on
/// the connection task, so their futures are polled inside the caller's
/// executor context; a long-running method (such as a permission dialog)
/// stalls further inbound processing until it resolves.
///
/// The file-system, terminal, elicitation, and extension methods have
/// default implementations that report the method as not found (or, for
/// notifications, ignore it); a handler that supports any of them must also
/// advertise the matching capability from [`capabilities`](Self::capabilities).
pub trait ClientHandler: Send + Sync + 'static {
    /// Capabilities advertised to the agent in the `initialize` request.
    ///
    /// The default is [`ClientCapabilities::default`]: no file-system access,
    /// no terminal, auth, or elicitation support.
    fn capabilities(&self) -> ClientCapabilities {
        ClientCapabilities::default()
    }

    /// Handle a `session/update` notification.
    ///
    /// Called once per update, in the order the agent sent them. Errors
    /// cannot be reported back; failures should be logged by the handler.
    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send;

    /// Handle a `session/request_permission` request.
    ///
    /// The agent is blocked until the returned outcome is sent back, so the
    /// implementation decides how to prompt the user or auto-approve.
    fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send;

    /// Handle an `elicitation/create` request: the agent asks for structured
    /// input, either a form (`mode: "form"`) or a URL visit (`mode: "url"`).
    ///
    /// The default responds with a JSON-RPC method-not-found error; a handler
    /// that overrides it should advertise
    /// [`ClientCapabilities::elicitation`].
    fn elicitation_create(
        &self,
        _params: ElicitationCreateParams,
    ) -> impl Future<Output = Result<ElicitationCreateResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("elicitation/create")) }
    }

    /// Handle an `elicitation/complete` notification: a URL-mode elicitation
    /// the agent asked for is finished.
    ///
    /// The default ignores the notification.
    fn elicitation_complete(
        &self,
        _params: ElicitationCompleteParams,
    ) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }

    /// Handle an agent-to-client request for a custom `_`-prefixed
    /// extension method.
    ///
    /// `method` is the extension's name and `params` its raw JSON payload;
    /// the returned [`Value`] becomes the JSON-RPC result. The default
    /// responds with a method-not-found error, as ACP requires for
    /// unrecognized extension requests.
    fn ext_request(
        &self,
        method: ExtMethod,
        _params: Option<Value>,
    ) -> impl Future<Output = Result<Value, JsonRpcError>> + Send {
        async move { Err(JsonRpcError::method_not_found(method.as_str())) }
    }

    /// Handle any agent-to-client notification this crate does not model —
    /// typically a `_`-prefixed extension notification.
    ///
    /// The default ignores the notification, as ACP requires for
    /// unrecognized extension notifications.
    fn notification(&self, notification: JsonRpcNotification) -> impl Future<Output = ()> + Send {
        debug!(method = %notification.method, "ignoring agent notification");
        std::future::ready(())
    }

    /// Handle an `fs/read_text_file` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn read_text_file(
        &self,
        _params: ReadTextFileParams,
    ) -> impl Future<Output = Result<ReadTextFileResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("fs/read_text_file")) }
    }

    /// Handle an `fs/write_text_file` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn write_text_file(
        &self,
        _params: WriteTextFileParams,
    ) -> impl Future<Output = Result<WriteTextFileResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("fs/write_text_file")) }
    }

    /// Handle a `terminal/create` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn terminal_create(
        &self,
        _params: TerminalCreateParams,
    ) -> impl Future<Output = Result<TerminalCreateResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("terminal/create")) }
    }

    /// Handle a `terminal/output` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn terminal_output(
        &self,
        _params: TerminalOutputParams,
    ) -> impl Future<Output = Result<TerminalOutputResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("terminal/output")) }
    }

    /// Handle a `terminal/wait_for_exit` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn terminal_wait_for_exit(
        &self,
        _params: TerminalWaitForExitParams,
    ) -> impl Future<Output = Result<TerminalExitStatus, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("terminal/wait_for_exit")) }
    }

    /// Handle a `terminal/kill` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn terminal_kill(
        &self,
        _params: TerminalKillParams,
    ) -> impl Future<Output = Result<TerminalKillResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("terminal/kill")) }
    }

    /// Handle a `terminal/release` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn terminal_release(
        &self,
        _params: TerminalReleaseParams,
    ) -> impl Future<Output = Result<TerminalReleaseResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("terminal/release")) }
    }
}
