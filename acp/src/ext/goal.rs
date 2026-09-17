//! `_session/goal` — the session goal extension, version 1.
//!
//! A provider-neutral convention: an agent that supports long-running goals
//! advertises a [`GoalCapability`] under `_meta.goal` on its `initialize`
//! result. The client steers the goal through `_session/goal` requests and
//! reads snapshots back from `session_info_update` `_meta.goal` payloads.
//!
//! ```no_run
//! # use aither_acp::{AcpClient, ext};
//! # async fn run<H: aither_acp::ClientHandler>(client: AcpClient<H>) -> Result<(), aither_acp::ClientError> {
//! let init = client.initialize().await?;
//! if let Some(cap) = ext::goal::capability(&init) {
//!     assert_eq!(cap.control_method, ext::goal::METHOD.as_str());
//!     ext::goal::set(&client, "sess-1", "ship the feature").await?;
//!     ext::goal::pause(&client, "sess-1").await?;
//!     ext::goal::clear(&client, "sess-1").await?;
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{AcpClient, ClientError, ClientHandler};
use crate::macros::string_enum;
use crate::protocol::{ExtMethod, InitializeResult, SessionInfoUpdate};

/// The goal-control extension method.
pub const METHOD: ExtMethod = ExtMethod::new("_session/goal");

/// Extension version this module models.
pub const VERSION: u64 = 1;

/// `InitializeResult.meta` key carrying the [`GoalCapability`].
pub const CAPABILITY: &str = "goal";

/// `SessionInfoUpdate.meta` key carrying the [`GoalSnapshot`].
pub const SNAPSHOT_KEY: &str = "goal";

string_enum! {
    /// An action accepted by `_session/goal`.
    pub enum GoalAction {
        /// Set or replace the session's goal; requires `objective`.
        Set = "set",
        /// Pause the goal without clearing it.
        Pause = "pause",
        /// Resume a paused goal.
        Resume = "resume",
        /// Clear the goal.
        Clear = "clear",
    }
}

string_enum! {
    /// Lifecycle state of a session goal.
    pub enum GoalStatus {
        /// Actively pursued.
        Active = "active",
        /// Paused by the user or the agent.
        Paused = "paused",
        /// Waiting on something the agent cannot resolve itself.
        Blocked = "blocked",
        /// Stopped by a budget or rate limit.
        Limited = "limited",
        /// Achieved.
        Complete = "complete",
    }
}

/// The `_meta.goal` capability object an agent advertises on `initialize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalCapability {
    /// Extension version the agent implements; [`VERSION`] here.
    pub version: u64,
    /// Method name the agent accepts for goal control — [`METHOD`].
    pub control_method: String,
    /// Actions the agent accepts.
    #[serde(default)]
    pub actions: Vec<GoalAction>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The goal capability an agent advertised, if any.
#[must_use]
pub fn capability(result: &InitializeResult) -> Option<GoalCapability> {
    result
        .meta
        .as_ref()?
        .get(CAPABILITY)
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}

/// Whether the agent advertised goal support and accepts `action`.
#[must_use]
pub fn accepts(result: &InitializeResult, action: &GoalAction) -> bool {
    capability(result).is_some_and(|cap| cap.actions.contains(action))
}

/// A snapshot of the session's goal state, carried in `session_info_update`
/// `_meta.goal`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalSnapshot {
    /// What the goal is.
    pub objective: String,
    /// Current lifecycle state.
    pub status: GoalStatus,
    /// Iterations consumed toward the goal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u64>,
    /// Why the goal last paused, blocked, or limited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_reason: Option<String>,
    /// Unix timestamp (ms) the goal was created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<f64>,
    /// Unix timestamp (ms) of the last change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<f64>,
    /// Token budget, if the agent enforces one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_budget: Option<u64>,
    /// Tokens consumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_used: Option<u64>,
    /// Wall-clock seconds consumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_used_seconds: Option<f64>,
    /// Method name the snapshot was produced under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_method: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// What a `session_info_update` reports about the goal.
#[derive(Debug, Clone)]
pub enum GoalUpdate {
    /// The current goal state; boxed to keep the enum small.
    Snapshot(Box<GoalSnapshot>),
    /// The goal was cleared (an explicit `null` `_meta.goal`).
    Cleared,
}

/// Extract the goal update from a `session_info_update` payload.
///
/// Returns `None` when the update carries no `goal` key at all, and
/// [`GoalUpdate::Cleared`] when it carries an explicit `null`.
#[must_use]
pub fn update(update: &SessionInfoUpdate) -> Option<GoalUpdate> {
    match update.meta.as_ref()?.get(SNAPSHOT_KEY)? {
        Value::Null => Some(GoalUpdate::Cleared),
        value => serde_json::from_value(value.clone())
            .ok()
            .map(|snapshot| GoalUpdate::Snapshot(Box::new(snapshot))),
    }
}

/// `_session/goal` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalControlParams {
    /// Session the goal belongs to.
    pub session_id: String,
    /// Action to perform.
    pub action: GoalAction,
    /// Goal text; required by [`GoalAction::Set`], absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
}

impl GoalControlParams {
    /// Build params for `action` on `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, action: GoalAction) -> Self {
        Self {
            session_id: session_id.into(),
            action,
            objective: None,
        }
    }

    /// Set the objective (required for [`GoalAction::Set`]).
    #[must_use]
    pub fn objective(mut self, objective: impl Into<String>) -> Self {
        self.objective = Some(objective.into());
        self
    }
}

/// Send a `_session/goal` control request.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn control<H: ClientHandler>(
    client: &AcpClient<H>,
    params: GoalControlParams,
) -> Result<(), ClientError> {
    client.ext_request::<_, Value>(&METHOD, &params).await?;
    Ok(())
}

/// Set or replace the session's goal.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn set<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
    objective: &str,
) -> Result<(), ClientError> {
    control(
        client,
        GoalControlParams::new(session_id, GoalAction::Set).objective(objective),
    )
    .await
}

/// Pause the session's goal without clearing it.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn pause<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
) -> Result<(), ClientError> {
    control(
        client,
        GoalControlParams::new(session_id, GoalAction::Pause),
    )
    .await
}

/// Resume a paused goal.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn resume<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
) -> Result<(), ClientError> {
    control(
        client,
        GoalControlParams::new(session_id, GoalAction::Resume),
    )
    .await
}

/// Clear the session's goal.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn clear<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
) -> Result<(), ClientError> {
    control(
        client,
        GoalControlParams::new(session_id, GoalAction::Clear),
    )
    .await
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The capability object lives under `_meta.goal` on `initialize`.
    #[test]
    fn capability_extracted_from_meta() {
        let result: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": 1,
            "_meta": {
                "goal": {
                    "version": 1,
                    "controlMethod": "_session/goal",
                    "actions": ["set", "pause", "resume", "clear"],
                    "vendor": "x",
                }
            }
        }))
        .expect("deserializes");
        let cap = capability(&result).expect("capability present");
        assert_eq!(cap.version, 1);
        assert_eq!(cap.control_method, "_session/goal");
        assert_eq!(cap.extra["vendor"], "x");
        assert!(accepts(&result, &GoalAction::Set));
        assert!(!accepts(&result, &GoalAction::Other("skip".to_string())));
    }

    /// No capability is reported when `_meta.goal` is absent.
    #[test]
    fn capability_absent() {
        let result: InitializeResult =
            serde_json::from_value(json!({"protocolVersion": 1})).expect("deserializes");
        assert!(capability(&result).is_none());
        assert!(!accepts(&result, &GoalAction::Set));
    }

    /// `session_info_update._meta.goal` decodes into a snapshot; explicit
    /// `null` means cleared.
    #[test]
    fn update_extraction() {
        let update = SessionInfoUpdate {
            meta: Some(json!({
                "goal": {"objective": "ship", "status": "active", "iterations": 3}
            })),
            ..SessionInfoUpdate::default()
        };
        let Some(GoalUpdate::Snapshot(snapshot)) = super::update(&update) else {
            panic!("expected snapshot");
        };
        assert_eq!(snapshot.objective, "ship");
        assert_eq!(snapshot.status, GoalStatus::Active);
        assert_eq!(snapshot.iterations, Some(3));

        let update = SessionInfoUpdate {
            meta: Some(json!({"goal": null})),
            ..SessionInfoUpdate::default()
        };
        assert!(matches!(super::update(&update), Some(GoalUpdate::Cleared)));

        let update = SessionInfoUpdate::default();
        assert!(super::update(&update).is_none());
    }

    /// Unknown goal statuses survive round-tripping.
    #[test]
    fn unknown_status_preserved() {
        let snapshot: GoalSnapshot = serde_json::from_value(json!({
            "objective": "x",
            "status": "snoozed",
        }))
        .expect("deserializes");
        assert_eq!(snapshot.status, GoalStatus::Other("snoozed".to_string()));
        assert_eq!(
            serde_json::to_value(&snapshot).expect("serializes")["status"],
            "snoozed"
        );
    }

    /// Control params serialize `action` and optional `objective`.
    #[test]
    fn control_params_wire_format() {
        let params = GoalControlParams::new("s1", GoalAction::Set).objective("win");
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["action"], "set");
        assert_eq!(json["objective"], "win");

        let json = serde_json::to_value(GoalControlParams::new("s1", GoalAction::Pause))
            .expect("serializes");
        assert_eq!(json["action"], "pause");
        assert!(json.get("objective").is_none());
    }
}
