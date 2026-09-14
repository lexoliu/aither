//! `devin acp` private surface.
//!
//! What `devin acp` speaks beyond the common protocol:
//!
//! - `agentCapabilities._meta` feature flags under the `cognition.ai/`
//!   prefix ([`capability`]);
//! - `initialize` result `_meta.mcpConfigPath` ([`mcp_config_path`]);
//! - `_cognition.ai/*` agent→client notifications ([`notification`]);
//! - the `devin-browser` [`authenticate`](crate::AcpClient::authenticate)
//!   method ID.

use std::collections::BTreeMap;

use aither_mcp::protocol::JsonRpcNotification;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::{AgentCapabilities, InitializeResult};

/// Devin's browser sign-in `authenticate` method ID.
pub const BROWSER_AUTH_METHOD: &str = "devin-browser";

/// `initialize` result `_meta` key carrying the path of the MCP config
/// file devin loaded.
pub const MCP_CONFIG_PATH: &str = "mcpConfigPath";

/// The `agentCapabilities._meta` feature flags `devin acp` advertises.
pub mod capability {
    /// The client exposes more than one workspace root.
    pub const MULTI_ROOT_WORKSPACE: &str = "cognition.ai/multiRootWorkspace";
    /// The client can rename sessions.
    pub const SESSION_RENAME: &str = "cognition.ai/sessionRename";
    /// The client can produce shareable session links.
    pub const SESSION_SHARE: &str = "cognition.ai/sessionShare";
    /// The client tracks document open/close/save lifecycles.
    pub const DOCUMENT_LIFECYCLE: &str = "cognition.ai/documentLifecycle";
    /// The client reports user edits to documents.
    pub const USER_EDITS: &str = "cognition.ai/userEdits";
    /// The client tracks terminal lifecycles beyond tool calls.
    pub const TERMINAL_LIFECYCLE: &str = "cognition.ai/terminalLifecycle";
    /// The client supplies user-level configuration.
    pub const USER_CONFIG: &str = "cognition.ai/userConfig";
    /// The client supports interactive user shell commands.
    pub const USER_SHELL_COMMAND: &str = "cognition.ai/userShellCommand";
    /// The client lets the agent enumerate editable commands.
    pub const EDITABLE_COMMANDS: &str = "cognition.ai/editableCommands";
    /// The client reports command revisions.
    pub const COMMAND_REVISION: &str = "cognition.ai/commandRevision";
    /// The client supports chained sessions.
    pub const CHAINS: &str = "cognition.ai/chains";
    /// The client supports multi-part plans.
    pub const MEGAPLAN: &str = "cognition.ai/megaplan";
    /// The client resolves rule-file mentions.
    pub const RULE_MENTIONS: &str = "cognition.ai/ruleMentions";
}

/// Whether `agentCapabilities._meta` carries `key` with a `true` value.
#[must_use]
pub fn supports(caps: &AgentCapabilities, key: &str) -> bool {
    caps.meta
        .as_ref()
        .and_then(|meta| meta.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The MCP config file path devin reports on `initialize`, if any.
#[must_use]
pub fn mcp_config_path(result: &InitializeResult) -> Option<&str> {
    result.meta.as_ref()?.get(MCP_CONFIG_PATH)?.as_str()
}

/// `_cognition.ai/output` — devin pushes log lines (MCP startup, internal
/// progress) outside the session transcript.
pub const OUTPUT_NOTIFICATION: &str = "_cognition.ai/output";

/// `_cognition.ai/mcp/serversChanged` — the MCP server set changed at
/// runtime.
pub const MCP_SERVERS_CHANGED_NOTIFICATION: &str = "_cognition.ai/mcp/serversChanged";

/// Params of an [`OUTPUT_NOTIFICATION`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputNotification {
    /// Channel producing the message (e.g. `MCP: <name>`).
    pub channel: String,
    /// The message text.
    pub message: String,
    /// Severity (`info`, `warning`, `error`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    /// Session the message belongs to; empty when connection-scoped.
    #[serde(default)]
    pub session_id: String,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A `_cognition.ai/*` notification from devin.
#[derive(Debug, Clone)]
pub enum DevinNotification {
    /// A log line outside the session transcript.
    Output(OutputNotification),
    /// The MCP server set changed; carries the raw params.
    McpServersChanged(Option<Value>),
    /// Another `_cognition.ai` notification; method and params preserved.
    Other {
        /// The notification's method name.
        method: String,
        /// The notification's params.
        params: Option<Value>,
    },
}

/// Interpret an agent→client notification as a [`DevinNotification`].
///
/// Returns `None` for notifications outside the `_cognition.ai/`
/// namespace.
#[must_use]
pub fn notification(notification: &JsonRpcNotification) -> Option<DevinNotification> {
    match notification.method.as_str() {
        OUTPUT_NOTIFICATION => serde_json::from_value(notification.params.clone()?)
            .ok()
            .map(DevinNotification::Output),
        MCP_SERVERS_CHANGED_NOTIFICATION => Some(DevinNotification::McpServersChanged(
            notification.params.clone(),
        )),
        method if method.starts_with("_cognition.ai/") => Some(DevinNotification::Other {
            method: method.to_string(),
            params: notification.params.clone(),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Capability flags are `true` values under `agentCapabilities._meta`.
    #[test]
    fn capability_flags() {
        let caps: AgentCapabilities = serde_json::from_value(json!({
            "_meta": {
                "cognition.ai/megaplan": true,
                "cognition.ai/chains": false,
            },
        }))
        .expect("deserializes");
        assert!(supports(&caps, capability::MEGAPLAN));
        assert!(!supports(&caps, capability::CHAINS));
        assert!(!supports(&caps, capability::RULE_MENTIONS));
        assert!(!supports(
            &AgentCapabilities::default(),
            capability::MEGAPLAN
        ));
    }

    /// `mcpConfigPath` reads off the initialize result's `_meta`.
    #[test]
    fn mcp_config_path_extraction() {
        let result: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": 1,
            "_meta": {"mcpConfigPath": "/home/me/.devin/mcp.json"},
        }))
        .expect("deserializes");
        assert_eq!(mcp_config_path(&result), Some("/home/me/.devin/mcp.json"));

        let result: InitializeResult =
            serde_json::from_value(json!({"protocolVersion": 1})).expect("deserializes");
        assert_eq!(mcp_config_path(&result), None);
    }

    /// `_cognition.ai/*` notifications decode; others return `None`.
    #[test]
    fn notification_parsing() {
        let output = JsonRpcNotification::with_params(
            OUTPUT_NOTIFICATION,
            json!({
                "channel": "MCP: fs",
                "message": "started",
                "level": "info",
                "sessionId": "s1",
            }),
        );
        let Some(DevinNotification::Output(output)) = notification(&output) else {
            panic!("expected output");
        };
        assert_eq!(output.channel, "MCP: fs");
        assert_eq!(output.session_id, "s1");

        let changed =
            JsonRpcNotification::with_params(MCP_SERVERS_CHANGED_NOTIFICATION, json!({"n": 2}));
        let Some(DevinNotification::McpServersChanged(params)) = notification(&changed) else {
            panic!("expected servers-changed");
        };
        assert_eq!(params, Some(json!({"n": 2})));

        let other = JsonRpcNotification::new("_cognition.ai/futureThing");
        let Some(DevinNotification::Other { method, .. }) = notification(&other) else {
            panic!("expected other");
        };
        assert_eq!(method, "_cognition.ai/futureThing");

        let foreign = JsonRpcNotification::new("_acme/ping");
        assert!(notification(&foreign).is_none());
    }
}
