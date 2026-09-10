//! Handler trait for agent-to-client traffic.

use std::future::Future;

use aither_mcp::protocol::JsonRpcError;

use crate::protocol::{
    ClientCapabilities, ReadTextFileParams, ReadTextFileResult, RequestPermissionParams,
    RequestPermissionResult, SessionNotification, TerminalCreateParams, TerminalCreateResult,
    TerminalExitStatus, TerminalKillParams, TerminalKillResult, TerminalOutputParams,
    TerminalOutputResult, TerminalReleaseParams, TerminalReleaseResult, TerminalWaitForExitParams,
    WriteTextFileParams, WriteTextFileResult,
};

/// Handles traffic the agent initiates toward the client.
///
/// Implementors receive every `session/update` notification and every
/// agent-to-client request the connection accepts. The methods are called on
/// the connection task, so their futures are polled inside the caller's
/// executor context; a long-running method (such as a permission dialog)
/// stalls further inbound processing until it resolves.
///
/// The file-system and terminal methods have default implementations that
/// report the method as not found; a handler that supports any of them must
/// also advertise the matching flag from [`capabilities`](Self::capabilities).
pub trait ClientHandler: Send + Sync + 'static {
    /// Capabilities advertised to the agent in the `initialize` request.
    ///
    /// The default is [`ClientCapabilities::default`]: no file-system access
    /// and no terminal support.
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
