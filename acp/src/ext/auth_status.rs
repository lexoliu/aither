//! `_auth/status_update` — agent-pushed connection auth identity.
//!
//! A provider-neutral convention: an agent that pushes auth status updates
//! advertises an (empty) `authStatus` object under `agentCapabilities._meta`.
//! Whenever the auth identity changes it sends an
//! `_auth/status_update` notification.

use std::collections::BTreeMap;

use aither_mcp::protocol::JsonRpcNotification;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::macros::string_enum;
use crate::protocol::{AgentCapabilities, ExtMethod};

/// The auth-status extension notification.
pub const METHOD: ExtMethod = ExtMethod::new("_auth/status_update");

/// `AgentCapabilities.meta` key — presence of an object means the agent
/// pushes [`METHOD`] notifications.
pub const CAPABILITY: &str = "authStatus";

/// Whether the agent advertised `_auth/status_update` support.
#[must_use]
pub fn advertised(caps: &AgentCapabilities) -> bool {
    caps.meta
        .as_ref()
        .and_then(|meta| meta.get(CAPABILITY))
        .is_some_and(Value::is_object)
}

string_enum! {
    /// Category of the credential the agent uses.
    pub enum AuthStatusKind {
        /// An interactive account session (e.g. browser sign-in).
        Account = "account",
        /// A pasted API key.
        ApiKey = "api_key",
        /// A custom model gateway.
        Gateway = "gateway",
        /// Credentials supplied externally (environment, file).
        External = "external",
        /// No credentials.
        None = "none",
    }
}

/// Account details of the current auth identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthStatusAccount {
    /// The account's email address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// The account's organization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub organization: Option<String>,
    /// The account's plan tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `authStatus` object carried by an `_auth/status_update`
/// notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthStatus {
    /// What kind of credential is in use.
    pub kind: AuthStatusKind,
    /// Human-readable label for the identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Additional human-readable detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Account details, when `kind` is [`AuthStatusKind::Account`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<AuthStatusAccount>,
    /// The vendor's display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Parse the [`AuthStatus`] out of an [`ExtMethod`] notification.
///
/// Returns `None` for notifications that are not [`METHOD`] or carry no
/// `authStatus` object.
#[must_use]
pub fn notification(notification: &JsonRpcNotification) -> Option<AuthStatus> {
    if notification.method != METHOD.as_str() {
        return None;
    }
    serde_json::from_value(notification.params.as_ref()?.get("authStatus")?.clone()).ok()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The capability is any object under `agentCapabilities._meta.authStatus`.
    #[test]
    fn advertised_capability() {
        let caps: AgentCapabilities = serde_json::from_value(json!({
            "_meta": {"authStatus": {}},
        }))
        .expect("deserializes");
        assert!(advertised(&caps));

        let caps: AgentCapabilities =
            serde_json::from_value(json!({"_meta": {"authStatus": true}})).expect("deserializes");
        assert!(!advertised(&caps));

        assert!(!advertised(&AgentCapabilities::default()));
    }

    /// A matching notification decodes the `authStatus` object.
    #[test]
    fn notification_parsing() {
        let notification = JsonRpcNotification::with_params(
            METHOD.as_str(),
            json!({
                "authStatus": {
                    "kind": "account",
                    "label": "me",
                    "account": {"plan": "pro"},
                    "vendorField": 1,
                }
            }),
        );
        let status = super::notification(&notification).expect("decodes");
        assert_eq!(status.kind, AuthStatusKind::Account);
        assert_eq!(status.label.as_deref(), Some("me"));
        assert_eq!(status.account.and_then(|a| a.plan).as_deref(), Some("pro"));
        assert_eq!(status.extra["vendorField"], 1);

        // Other methods and malformed payloads return None.
        let other = JsonRpcNotification::new("_acme/ping");
        assert!(super::notification(&other).is_none());
        let malformed =
            JsonRpcNotification::with_params(METHOD.as_str(), json!({"authStatus": "oops"}));
        assert!(super::notification(&malformed).is_none());
    }

    /// Unknown kinds survive round-tripping.
    #[test]
    fn unknown_kind_preserved() {
        let status: AuthStatus =
            serde_json::from_value(json!({"kind": "ssh"})).expect("deserializes");
        assert_eq!(status.kind, AuthStatusKind::Other("ssh".to_string()));
        assert_eq!(
            serde_json::to_value(&status).expect("serializes")["kind"],
            "ssh"
        );
    }
}
