//! # aither-acp
//!
//! `Agent Client Protocol` (ACP) implementation for the aither agent framework.
//!
//! This crate implements both directions of ACP v1:
//!
//! - [`AcpServer`] runs an aither [`LanguageModel`](aither_core::LanguageModel)
//!   as an ACP agent over stdio, answering `initialize`, `session/new`,
//!   `session/prompt`, and `session/stop` while streaming [`SessionUpdate`]
//!   notifications.
//! - [`AcpClient`] connects to a spawned or piped ACP agent (such as
//!   `devin acp`) over any
//!   [`BidirectionalTransport`](aither_mcp::transport::BidirectionalTransport),
//!   driving sessions,
//!   prompts, modes, and config options while a [`ClientHandler`] receives
//!   streamed updates and answers permission, file-system, and terminal
//!   requests from the agent.
//!
//! # Protocol Overview
//!
//! ACP uses JSON-RPC 2.0 over newline-delimited stdio or pipes. ACP describes
//! the client direction (editor → agent): an aither agent is the ACP **server**
//! (agent) and [`AcpClient`] plays the ACP **client** (editor) role.
//!
//! # Client
//!
//! ```no_run
//! use aither_acp::{AcpClient, ContentBlock, TextContent};
//! # use aither_acp::{ClientCapabilities, ClientHandler, SessionNotification};
//! # use aither_acp::{RequestPermissionParams, RequestPermissionResult, RequestPermissionOutcome};
//! # use aither_mcp::protocol::JsonRpcError;
//! # use std::future::Future;
//!
//! # struct H;
//! # impl ClientHandler for H {
//! #     fn session_update(&self, _: SessionNotification) -> impl Future<Output = ()> + Send {
//! #         async {}
//! #     }
//! #     fn request_permission(&self, _: RequestPermissionParams)
//! #         -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send
//! #     {
//! #         async { Ok(RequestPermissionResult { outcome: RequestPermissionOutcome::Cancelled, meta: None }) }
//! #     }
//! # }
//! # async fn run() -> Result<(), aither_acp::ClientError> {
//! let (client, connection) = AcpClient::spawn(
//!     "devin",
//!     &["acp"],
//!     [],
//!     "/tmp",
//!     H,
//! )?;
//! tokio::spawn(connection);
//!
//! let init = client.initialize().await?;
//! let session = client.new_session("/tmp", vec![]).await?;
//! let result = client
//!     .prompt(&session.session_id, vec![ContentBlock::Text(TextContent {
//!         text: "hi".to_string(),
//!         annotations: None,
//!     })])
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Server
//!
//! ```no_run
//! use aither_acp::{AcpError, AcpServer};
//! use aither_agent::Agent;
//! use aither_core::LanguageModel;
//! use aither_core::llm::{Event, LLMRequest, model::Profile};
//! use futures_lite::{Stream, stream};
//! use std::future::Future;
//!
//! #[derive(Clone)]
//! struct EchoModel;
//!
//! #[derive(Debug)]
//! struct EchoError;
//! impl std::fmt::Display for EchoError {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         f.write_str("echo model error")
//!     }
//! }
//! impl std::error::Error for EchoError {}
//!
//! impl LanguageModel for EchoModel {
//!     type Error = EchoError;
//!
//!     fn respond(&self, _request: LLMRequest) -> impl Stream<Item = Result<Event, EchoError>> + Send {
//!         stream::once(Ok(Event::Text("Hello from ACP".to_string())))
//!     }
//!
//!     fn profile(&self) -> impl Future<Output = Profile> + Send {
//!         std::future::ready(Profile::new("echo", "test", "echo", "Echo model", 128_000))
//!     }
//! }
//!
//! # async fn run() -> Result<(), AcpError> {
//! let mut server = AcpServer::stdio("echo-agent", "0.1.0", |_cwd| async {
//!     Ok::<_, AcpError>(Agent::new(EchoModel))
//! });
//! server.run().await
//! # }
//! ```
//!
//! # Error Handling
//!
//! [`ClientError`] distinguishes a closed transport (with the agent's exit
//! status when it ran as a child process), a JSON-RPC error returned by the
//! agent, and a protocol violation. When the connection closes, every pending
//! request fails rather than hanging.

mod adapter;
mod client;
mod protocol;
mod server;
mod session;

pub use adapter::{agent_event_to_session_update, todos_to_plan};
pub use client::{AcpClient, ClientError, ClientHandler};
pub use protocol::*;
pub use server::AcpServer;
pub use session::AcpSession;
