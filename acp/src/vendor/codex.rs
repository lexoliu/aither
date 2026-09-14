//! Codex-specific ACP surface.
//!
//! What codex-acp-compatible agents speak beyond the common protocol:
//!
//! - the `_meta.codex` namespace (subagent refs, collaboration routing,
//!   phase labels) plus codex-emitted top-level `_meta` keys on tool calls;
//! - the `models` extra on `session/new|load|resume` results;
//! - auth method IDs and the gateway auth flow;
//! - legacy methods that predate the `_` naming rule (`session/set_model`,
//!   `authentication/status`, `authentication/logout`) and the
//!   `_codex/session/goal_control` alias for
//!   [`crate::ext::goal::METHOD`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::client::{AcpClient, ClientError, ClientHandler};
use crate::protocol::ExtMethod;

/// The `_meta` namespace Codex writes under.
pub const NAMESPACE: &str = "codex";

/// Legacy alias for [`crate::ext::goal::METHOD`], kept for agents that
/// predate the provider-neutral method name.
pub const LEGACY_GOAL_CONTROL_METHOD: ExtMethod = ExtMethod::new("_codex/session/goal_control");

/// Legacy model-switch method; superseded by `session/set_config_option`.
pub const SET_MODEL_METHOD: &str = "session/set_model";

/// Legacy pull-style auth status request; superseded by
/// [`crate::ext::auth_status`].
pub const AUTHENTICATION_STATUS_METHOD: &str = "authentication/status";

/// Legacy logout request; superseded by the standard `logout`.
pub const AUTHENTICATION_LOGOUT_METHOD: &str = "authentication/logout";

/// The `codex` object inside a `_meta` value.
fn codex_meta(meta: Option<&Value>) -> Option<&Value> {
    meta?.get(NAMESPACE)
}

/// A subagent reference carried in a tool call's `_meta.codex.subagent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentRef {
    /// Thread ID of the spawned subagent.
    pub thread_id: String,
    /// Agent path within the delegation tree.
    pub path: String,
    /// The activity kind that produced the call.
    pub activity: String,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `_meta.codex.subagent` reference on a tool call, if any.
#[must_use]
pub fn subagent(meta: Option<&Value>) -> Option<SubagentRef> {
    serde_json::from_value(codex_meta(meta)?.get("subagent")?.clone()).ok()
}

/// Collaboration routing carried in a collab tool call's
/// `_meta.codex.collaboration`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Collaboration {
    /// The collab tool being invoked.
    pub tool: String,
    /// Thread initiating the collaboration.
    pub sender_thread_id: String,
    /// Threads receiving the collaboration.
    #[serde(default)]
    pub receiver_thread_ids: Vec<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `_meta.codex.collaboration` object on a tool call, if any.
#[must_use]
pub fn collaboration(meta: Option<&Value>) -> Option<Collaboration> {
    serde_json::from_value(codex_meta(meta)?.get("collaboration")?.clone()).ok()
}

/// The `_meta.codex.phase` label on a tool call or plan entry, if any.
#[must_use]
pub fn phase(meta: Option<&Value>) -> Option<&str> {
    codex_meta(meta)?.get("phase")?.as_str()
}

/// Top-level `_meta` key marking a tool call as backed by an MCP tool.
pub const IS_MCP_TOOL_CALL: &str = "is_mcp_tool_call";

/// Whether a tool call's `_meta.is_mcp_tool_call` marks it as MCP-backed.
#[must_use]
pub fn is_mcp_tool_call(meta: Option<&Value>) -> bool {
    meta.and_then(|meta| meta.get(IS_MCP_TOOL_CALL))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Top-level `_meta` key carrying the terminal a tool call ran in.
pub const TERMINAL_INFO: &str = "terminal_info";

/// Top-level `_meta` key carrying a terminal's exit status.
pub const TERMINAL_EXIT: &str = "terminal_exit";

/// The `_meta.terminal_info` object on a terminal-backed tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalInfo {
    /// Working directory the command ran in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Client-side terminal ID backing the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_id: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `_meta.terminal_info` object on a tool call, if any.
#[must_use]
pub fn terminal_info(meta: Option<&Value>) -> Option<TerminalInfo> {
    serde_json::from_value(meta?.get(TERMINAL_INFO)?.clone()).ok()
}

/// The `_meta.terminal_exit` object on a terminal-backed tool call update.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalExit {
    /// Process exit code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    /// Signal that terminated the process, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    /// Client-side terminal ID backing the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_id: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `_meta.terminal_exit` object on a tool call update, if any.
#[must_use]
pub fn terminal_exit(meta: Option<&Value>) -> Option<TerminalExit> {
    serde_json::from_value(meta?.get(TERMINAL_EXIT)?.clone()).ok()
}

/// The `models` extra Codex returns on `session/new|load|resume` results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModels {
    /// Models the session can switch between.
    #[serde(default)]
    pub available_models: Vec<SessionModel>,
    /// Currently selected model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_model_id: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One entry of [`SessionModels::available_models`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModel {
    /// Model identifier for [`set_model`].
    pub model_id: String,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `models` key on session results' `extra` fields.
pub const MODELS: &str = "models";

/// Extract the [`SessionModels`] extra from a session result.
///
/// Works on the `extra` map of `SessionNewResult`, `SessionLoadResult`,
/// and `SessionResumeResult`.
#[must_use]
pub fn models(extra: &BTreeMap<String, Value>) -> Option<SessionModels> {
    serde_json::from_value(extra.get(MODELS)?.clone()).ok()
}

/// `session/set_model` request parameters (legacy method).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SetModelParams {
    session_id: String,
    model_id: String,
}

/// Legacy `session/set_model` request; superseded by
/// `session/set_config_option` but still the only model switch on older
/// agents.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn set_model<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
    model_id: &str,
) -> Result<(), ClientError> {
    client
        .request::<_, Value>(
            SET_MODEL_METHOD,
            &SetModelParams {
                session_id: session_id.to_string(),
                model_id: model_id.to_string(),
            },
        )
        .await?;
    Ok(())
}

/// The legacy `authentication/status` response (pull-style; superseded by
/// [`crate::ext::auth_status`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthenticationStatus {
    /// Signed in with an API key.
    #[serde(rename = "api-key")]
    ApiKey,
    /// Signed in with a `ChatGPT` account.
    #[serde(rename = "chat-gpt")]
    ChatGpt {
        /// The account's email address.
        email: String,
    },
    /// Signed in through a custom model gateway.
    #[serde(rename = "gateway")]
    Gateway {
        /// The gateway's display name.
        name: String,
    },
    /// Not signed in.
    #[serde(rename = "unauthenticated")]
    Unauthenticated,
}

/// Legacy `authentication/status` request.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn authentication_status<H: ClientHandler>(
    client: &AcpClient<H>,
) -> Result<AuthenticationStatus, ClientError> {
    client
        .request(AUTHENTICATION_STATUS_METHOD, &json!({}))
        .await
}

/// Legacy `authentication/logout` request; superseded by the standard
/// `logout`.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn authentication_logout<H: ClientHandler>(
    client: &AcpClient<H>,
) -> Result<(), ClientError> {
    client
        .request::<_, Value>(AUTHENTICATION_LOGOUT_METHOD, &json!({}))
        .await?;
    Ok(())
}

/// Codex `authenticate` method IDs and the gateway auth flow.
pub mod auth {
    use std::collections::HashMap;

    use serde::Serialize;
    use serde_json::json;

    use crate::client::{AcpClient, ClientError, ClientHandler};
    use crate::protocol::{AuthenticateParams, ClientCapabilities};

    /// Paste an API key.
    pub const API_KEY: &str = "api-key";

    /// Browser sign-in with a `ChatGPT` account.
    pub const CHAT_GPT: &str = "chat-gpt";

    /// Device-code sign-in with a `ChatGPT` account. Requires the client to
    /// advertise URL elicitation support.
    pub const CHAT_GPT_DEVICE_CODE: &str = "chat-gpt-device-code";

    /// Authenticate through a custom model gateway. Only offered when the
    /// client advertised [`advertise_gateway`].
    pub const GATEWAY: &str = "gateway";

    /// The `clientCapabilities.auth._meta` key gating the gateway method.
    pub const GATEWAY_CAPABILITY: &str = "gateway";

    /// The `_meta` key carrying [`GatewayParams`] on `authenticate`.
    pub const GATEWAY_META: &str = "gateway";

    /// Advertise gateway-auth support on client capabilities.
    ///
    /// Sets `clientCapabilities.auth._meta.gateway: true`.
    pub fn advertise_gateway(caps: &mut ClientCapabilities) {
        let auth = caps.auth.get_or_insert_with(Default::default);
        let meta = auth.meta.get_or_insert_with(|| json!({}));
        if let Some(object) = meta.as_object_mut() {
            object.insert(GATEWAY_CAPABILITY.to_string(), json!(true));
        }
    }

    /// The `_meta.gateway` payload for a gateway `authenticate`.
    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GatewayParams {
        /// Gateway base URL.
        pub base_url: String,
        /// Extra headers sent to the gateway.
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        pub headers: HashMap<String, String>,
        /// Human-readable gateway name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub provider_name: Option<String>,
    }

    impl GatewayParams {
        /// Build gateway params for `base_url`.
        #[must_use]
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into(),
                headers: HashMap::new(),
                provider_name: None,
            }
        }

        /// Add a header sent to the gateway.
        #[must_use]
        pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
            self.headers.insert(name.into(), value.into());
            self
        }

        /// Set the gateway's display name.
        #[must_use]
        pub fn provider_name(mut self, name: impl Into<String>) -> Self {
            self.provider_name = Some(name.into());
            self
        }
    }

    /// `authenticate` with `methodId: "gateway"` and the connection details
    /// under `_meta.gateway`.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or the agent replies with
    /// an error or a malformed result.
    pub async fn authenticate_gateway<H: ClientHandler>(
        client: &AcpClient<H>,
        params: GatewayParams,
    ) -> Result<(), ClientError> {
        client
            .authenticate(AuthenticateParams::new(GATEWAY).meta(json!({ GATEWAY_META: params })))
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::protocol::ClientCapabilities;

    /// `_meta.codex.*` accessors decode their payloads.
    #[test]
    fn meta_accessors() {
        let meta = json!({
            "codex": {
                "subagent": {"threadId": "th-1", "path": "root/0", "activity": "tool"},
                "collaboration": {
                    "tool": "collab",
                    "senderThreadId": "th-0",
                    "receiverThreadIds": ["th-1"],
                },
                "phase": "exec",
            },
            "is_mcp_tool_call": true,
            "terminal_info": {"cwd": "/work", "terminalId": "term-1"},
            "terminal_exit": {"exitCode": 0, "terminalId": "term-1"},
        });
        let sub = subagent(Some(&meta)).expect("subagent");
        assert_eq!(sub.thread_id, "th-1");
        let collab = collaboration(Some(&meta)).expect("collaboration");
        assert_eq!(collab.receiver_thread_ids, ["th-1"]);
        assert_eq!(phase(Some(&meta)), Some("exec"));
        assert!(is_mcp_tool_call(Some(&meta)));
        assert_eq!(
            terminal_info(Some(&meta)).and_then(|info| info.terminal_id),
            Some("term-1".to_string())
        );
        assert_eq!(
            terminal_exit(Some(&meta)).and_then(|exit| exit.exit_code),
            Some(0)
        );

        let other = json!({"codex": {}});
        assert!(subagent(Some(&other)).is_none());
        assert!(phase(Some(&other)).is_none());
        assert!(!is_mcp_tool_call(Some(&other)));
        assert!(subagent(None).is_none());
    }

    /// The `models` extra decodes out of a session result's `extra` map.
    #[test]
    fn models_extra() {
        let extra = BTreeMap::from([(
            MODELS.to_string(),
            json!({
                "availableModels": [
                    {"modelId": "m1", "name": "Model One", "custom": 9},
                ],
                "currentModelId": "m1",
            }),
        )]);
        let parsed = models(&extra).expect("models");
        assert_eq!(parsed.current_model_id.as_deref(), Some("m1"));
        assert_eq!(parsed.available_models[0].model_id, "m1");
        assert_eq!(parsed.available_models[0].extra["custom"], 9);

        assert!(models(&BTreeMap::new()).is_none());
    }

    /// Legacy auth statuses deserialize by `type` tag.
    #[test]
    fn authentication_status_wire_format() {
        let status: AuthenticationStatus =
            serde_json::from_value(json!({"type": "chat-gpt", "email": "me"}))
                .expect("deserializes");
        let AuthenticationStatus::ChatGpt { email } = status else {
            panic!("expected chat-gpt");
        };
        assert_eq!(email, "me");

        let status: AuthenticationStatus =
            serde_json::from_value(json!({"type": "unauthenticated"})).expect("deserializes");
        assert!(matches!(status, AuthenticationStatus::Unauthenticated));
    }

    /// `advertise_gateway` sets `auth._meta.gateway: true`.
    #[test]
    fn advertise_gateway_capability() {
        let mut caps = ClientCapabilities::default();
        auth::advertise_gateway(&mut caps);
        let json = serde_json::to_value(&caps).expect("serializes");
        assert_eq!(json["auth"]["_meta"]["gateway"], true);
    }

    /// Gateway params nest under `_meta.gateway` on `authenticate`.
    #[test]
    fn gateway_params_wire_format() {
        let params = auth::GatewayParams::new("http://localhost:8787")
            .header("x-key", "v")
            .provider_name("Internal GW");
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["baseUrl"], "http://localhost:8787");
        assert_eq!(json["headers"]["x-key"], "v");
        assert_eq!(json["providerName"], "Internal GW");
    }
}
