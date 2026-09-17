//! Authentication methods and capabilities.
//!
//! Covers the `authenticate` and `logout` requests plus the `auth`
//! capability objects on both [`ClientCapabilities`](super::ClientCapabilities)
//! and [`AgentCapabilities`](super::AgentCapabilities).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `authenticate` request parameters: runs one of the [`AuthMethod`]s the
/// agent advertised in [`InitializeResult::auth_methods`](super::InitializeResult::auth_methods).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthenticateParams {
    /// ID of the auth method to run.
    pub method_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl AuthenticateParams {
    /// Build params selecting `method_id`.
    #[must_use]
    pub fn new(method_id: impl Into<String>) -> Self {
        Self {
            method_id: method_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

impl From<&str> for AuthenticateParams {
    fn from(method_id: &str) -> Self {
        Self::new(method_id)
    }
}

impl From<String> for AuthenticateParams {
    fn from(method_id: String) -> Self {
        Self::new(method_id)
    }
}

/// `authenticate` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthenticateResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `logout` request parameters. The method takes no fields besides `_meta`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogoutParams {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl LogoutParams {
    /// Build empty logout params.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// `logout` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogoutResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// An authentication method the agent advertises in `initialize`.
///
/// `agent` methods — the default kind, sent with no `type` field — are run
/// inside the agent (e.g. a browser flow). `terminal` methods describe a
/// command the client runs in a terminal it owns.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthMethod {
    /// A method the client runs in its terminal; `"type": "terminal"`.
    #[serde(rename = "terminal")]
    Terminal(AuthMethodTerminal),
    /// A method the agent runs itself; sent with no `type` field.
    #[serde(untagged)]
    Agent(AuthMethodAgent),
}

impl AuthMethod {
    /// The method's unique ID.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Agent(agent) => &agent.id,
            Self::Terminal(terminal) => &terminal.id,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Agent(agent) => &agent.name,
            Self::Terminal(terminal) => &terminal.name,
        }
    }

    /// Human-readable description, if provided.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        match self {
            Self::Agent(agent) => agent.description.as_deref(),
            Self::Terminal(terminal) => terminal.description.as_deref(),
        }
    }

    /// Extension metadata.
    #[must_use]
    pub const fn meta(&self) -> Option<&Value> {
        match self {
            Self::Agent(agent) => agent.meta.as_ref(),
            Self::Terminal(terminal) => terminal.meta.as_ref(),
        }
    }
}

impl From<AuthMethodAgent> for AuthMethod {
    fn from(method: AuthMethodAgent) -> Self {
        Self::Agent(method)
    }
}

impl From<AuthMethodTerminal> for AuthMethod {
    fn from(method: AuthMethodTerminal) -> Self {
        Self::Terminal(method)
    }
}

/// An auth method the agent runs itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthMethodAgent {
    /// Unique ID passed to `authenticate`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl AuthMethodAgent {
    /// Build an agent-run auth method.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            meta: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// An auth method the client runs in a terminal it owns: the agent names a
/// program to spawn and the client executes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthMethodTerminal {
    /// Unique ID passed to `authenticate`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Program and arguments to run in the terminal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment for the spawned program.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// What the client offers for authentication (`clientCapabilities.auth`).
///
/// `terminal` tells the agent it may advertise
/// [`AuthMethod::Terminal`] methods; `_meta` carries vendor flags such as
/// gateway-auth opt-ins.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientAuthCapabilities {
    /// Whether the client can run terminal auth methods.
    #[serde(default)]
    pub terminal: bool,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// What the agent offers for authentication (`agentCapabilities.auth`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentAuthCapabilities {
    /// Whether the agent supports `logout`. `null` and absent both mean no.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logout: Option<LogoutCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Marker for `logout` support (`agentCapabilities.auth.logout`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogoutCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn authenticate_params_wire_format() {
        let json =
            serde_json::to_value(AuthenticateParams::new("devin-browser").meta(json!({"k": 1})))
                .expect("serializes");
        assert_eq!(json["methodId"], "devin-browser");
        assert_eq!(json["_meta"]["k"], 1);
    }

    #[test]
    fn agent_auth_method_has_no_type_tag() {
        let method = AuthMethod::from(AuthMethodAgent::new("oauth", "Sign in"));
        let json = serde_json::to_value(&method).expect("serializes");
        assert!(json.get("type").is_none());
        assert_eq!(json["id"], "oauth");

        let parsed: AuthMethod = serde_json::from_value(json).expect("deserializes");
        assert_eq!(parsed.id(), "oauth");
    }

    #[test]
    fn terminal_auth_method_carries_type_tag() {
        let method: AuthMethod = serde_json::from_value(json!({
            "type": "terminal",
            "id": "gh",
            "name": "GitHub CLI",
            "args": ["gh", "auth", "login"],
            "env": {"BROWSER": "none"},
        }))
        .expect("deserializes");
        let AuthMethod::Terminal(terminal) = &method else {
            panic!("expected terminal method");
        };
        assert_eq!(terminal.id, "gh");
        assert_eq!(terminal.args, ["gh", "auth", "login"]);
        assert_eq!(terminal.env["BROWSER"], "none");
    }

    #[test]
    fn agent_auth_capabilities_round_trip() {
        let caps: AgentAuthCapabilities =
            serde_json::from_value(json!({"logout": {}})).expect("deserializes");
        assert!(caps.logout.is_some());
        let caps: AgentAuthCapabilities = serde_json::from_value(json!({})).expect("deserializes");
        assert!(caps.logout.is_none());
    }
}
