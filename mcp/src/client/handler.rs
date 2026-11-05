//! Handlers for what an MCP server sends on its own.

use std::future::Future;

use crate::protocol::{JsonRpcNotification, LoggingMessageParams, ProgressNotificationParams};

/// A notification an MCP server sent to the client.
///
/// Notifications with a known method and well-formed parameters are decoded;
/// everything else, including a known method whose parameters do not parse,
/// arrives unchanged as [`Other`](Self::Other).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ServerNotification {
    /// `notifications/tools/list_changed`: the server's tool list changed.
    ToolsListChanged,
    /// `notifications/resources/list_changed`: the server's resource list
    /// changed.
    ResourcesListChanged,
    /// `notifications/prompts/list_changed`: the server's prompt list changed.
    PromptsListChanged,
    /// `notifications/message`: a log entry from the server.
    Logging(LoggingMessageParams),
    /// `notifications/progress`: progress on a request the client sent.
    Progress(ProgressNotificationParams),
    /// Any other notification.
    Other(JsonRpcNotification),
}

impl ServerNotification {
    pub(super) fn decode(notification: JsonRpcNotification) -> Self {
        fn params<T: serde::de::DeserializeOwned>(notification: &JsonRpcNotification) -> Option<T> {
            serde_json::from_value(notification.params.clone()?).ok()
        }

        match notification.method.as_str() {
            "notifications/tools/list_changed" => Self::ToolsListChanged,
            "notifications/resources/list_changed" => Self::ResourcesListChanged,
            "notifications/prompts/list_changed" => Self::PromptsListChanged,
            "notifications/message" => {
                params(&notification).map_or(Self::Other(notification), Self::Logging)
            }
            "notifications/progress" => {
                params(&notification).map_or(Self::Other(notification), Self::Progress)
            }
            _ => Self::Other(notification),
        }
    }
}

/// Receives the notifications a server sends to an [`McpClient`](crate::McpClient).
///
/// Pass a handler to [`McpClient::connect`](crate::McpClient::connect); `()`
/// ignores everything. Each notification is handled on its own future that
/// the connection polls alongside its traffic, so a handler may call back
/// into the client, for example to re-list tools on
/// [`ServerNotification::ToolsListChanged`], without stalling the connection.
pub trait McpHandler: Send + Sync + 'static {
    /// Called once per notification the server sends.
    fn notification(&self, notification: ServerNotification) -> impl Future<Output = ()> + Send {
        let _ = notification;
        std::future::ready(())
    }
}

impl McpHandler for () {}
