//! `async_task_*` session updates and `_session/async_task/stop`.
//!
//! A provider-neutral convention for long-running background work — shells,
//! dev servers, watchers — that an agent owns on the session's behalf. The
//! agent reports spawns, progress, and terminal states through
//! `session/update` notifications carrying `async_task_*` update kinds; the
//! client can stop a task with [`STOP_METHOD`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{AcpClient, ClientError, ClientHandler};
use crate::macros::string_enum;
use crate::protocol::{ExtMethod, SessionUpdate};

/// Extension method to stop a background task.
pub const STOP_METHOD: ExtMethod = ExtMethod::new("_session/async_task/stop");

/// The `async_task_spawned` update kind.
pub const SPAWNED_UPDATE: &str = "async_task_spawned";

/// The `async_task_progress` update kind.
pub const PROGRESS_UPDATE: &str = "async_task_progress";

/// The `async_task_state_update` update kind.
pub const STATE_UPDATE: &str = "async_task_state_update";

string_enum! {
    /// Lifecycle state of a background task.
    pub enum AsyncTaskState {
        /// Producing output.
        Running = "running",
        /// Suspended.
        Paused = "paused",
        /// Finished successfully.
        Completed = "completed",
        /// Terminated with an error.
        Failed = "failed",
        /// Stopped by the client.
        Stopped = "stopped",
    }
}

/// `_session/async_task/stop` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsyncTaskStopParams {
    /// Session that owns the task.
    pub session_id: String,
    /// ID of the task to stop.
    pub async_task_id: String,
}

impl AsyncTaskStopParams {
    /// Build stop params for `async_task_id` on `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, async_task_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            async_task_id: async_task_id.into(),
        }
    }
}

/// `_session/async_task/stop` result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsyncTaskStopResult {
    /// Whether the agent stopped the task.
    pub stopped: bool,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Stop a background task the agent reported.
///
/// Returns `true` when the agent confirmed it stopped the task.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn stop<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
    async_task_id: &str,
) -> Result<bool, ClientError> {
    let result: AsyncTaskStopResult = client
        .ext_request(
            &STOP_METHOD,
            &AsyncTaskStopParams::new(session_id, async_task_id),
        )
        .await?;
    Ok(result.stopped)
}

/// An `async_task_spawned` update payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsyncTaskSpawned {
    /// ID the agent assigned to the task.
    pub async_task_id: String,
    /// Human-readable task name.
    pub name: String,
    /// What kind of task it is (e.g. `shell`, `dev_server`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    /// Whether the task's output should appear in the transcript.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_in_transcript: Option<bool>,
    /// Whether [`STOP_METHOD`] applies to this task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub can_stop: Option<bool>,
    /// Tool call the task is associated with, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// An `async_task_progress` update payload. Task-specific progress fields
/// land in `extra`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsyncTaskProgress {
    /// ID of the task making progress.
    pub async_task_id: String,
    /// Tool call the task is associated with, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// An `async_task_state_update` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsyncTaskStateUpdate {
    /// ID of the task that changed state.
    pub async_task_id: String,
    /// New lifecycle state.
    pub state: AsyncTaskState,
    /// Tool call the task is associated with, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A decoded async-task session update.
#[derive(Debug, Clone)]
pub enum AsyncTaskUpdate {
    /// A background task was spawned.
    Spawned(AsyncTaskSpawned),
    /// A background task reported progress.
    Progress(AsyncTaskProgress),
    /// A background task changed state.
    State(AsyncTaskStateUpdate),
}

/// Interpret a [`SessionUpdate`] as an async-task update.
///
/// Returns `None` for every update that is not an `async_task_*` kind, and
/// for malformed payloads.
#[must_use]
pub fn update(update: &SessionUpdate) -> Option<AsyncTaskUpdate> {
    let SessionUpdate::Other(value) = update else {
        return None;
    };
    // `extra` should not absorb the `sessionUpdate` tag.
    let mut object = value.as_object()?.clone();
    let kind = object.remove("sessionUpdate")?.as_str()?.to_string();
    let value = Value::Object(object);
    match kind.as_str() {
        SPAWNED_UPDATE => serde_json::from_value(value)
            .ok()
            .map(AsyncTaskUpdate::Spawned),
        PROGRESS_UPDATE => serde_json::from_value(value)
            .ok()
            .map(AsyncTaskUpdate::Progress),
        STATE_UPDATE => serde_json::from_value(value)
            .ok()
            .map(AsyncTaskUpdate::State),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// All three `async_task_*` update kinds decode from `Other` payloads.
    #[test]
    fn update_parsing() {
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "async_task_spawned",
            "asyncTaskId": "t1",
            "name": "dev server",
            "taskType": "dev_server",
            "canStop": true,
            "port": 8080,
        }))
        .expect("deserializes");
        let Some(AsyncTaskUpdate::Spawned(spawned)) = super::update(&update) else {
            panic!("expected spawned");
        };
        assert_eq!(spawned.async_task_id, "t1");
        assert_eq!(spawned.task_type.as_deref(), Some("dev_server"));
        assert_eq!(spawned.can_stop, Some(true));
        assert_eq!(spawned.extra["port"], 8080);

        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "async_task_progress",
            "asyncTaskId": "t1",
            "bytes": 128,
        }))
        .expect("deserializes");
        let Some(AsyncTaskUpdate::Progress(progress)) = super::update(&update) else {
            panic!("expected progress");
        };
        assert_eq!(progress.extra["bytes"], 128);

        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "async_task_state_update",
            "asyncTaskId": "t1",
            "state": "stopped",
        }))
        .expect("deserializes");
        let Some(AsyncTaskUpdate::State(state)) = super::update(&update) else {
            panic!("expected state");
        };
        assert_eq!(state.state, AsyncTaskState::Stopped);

        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "vendor_thing",
        }))
        .expect("deserializes");
        assert!(super::update(&update).is_none());
    }

    /// Stop params and result carry `sessionId`/`asyncTaskId`/`stopped`.
    #[test]
    fn stop_wire_format() {
        let params = AsyncTaskStopParams::new("s1", "t1");
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["asyncTaskId"], "t1");

        let result: AsyncTaskStopResult =
            serde_json::from_value(json!({"stopped": true, "extra": 1})).expect("deserializes");
        assert!(result.stopped);
        assert_eq!(result.extra["extra"], 1);
    }
}
