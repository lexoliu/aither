//! Session lifecycle methods: `session/list`, `session/delete`,
//! `session/close`.
//!
//! These are capability-gated: `session/list` and `session/delete` require
//! the matching [`SessionCapabilities`](super::SessionCapabilities) entries,
//! and `session/close` requires `sessionCapabilities.close`. Sessions keep
//! running until `session/close` (or the connection drops), so a client that
//! opens many sessions should close the ones it no longer needs.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `session/list` request parameters. All filters are optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionListParams {
    /// Only list sessions with this working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Cursor from a previous `session/list` response's `nextCursor`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionListParams {
    /// Build an unfiltered `session/list` request.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter to sessions with this working directory.
    #[must_use]
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Continue a previous paginated listing.
    #[must_use]
    pub fn cursor(mut self, cursor: impl Into<String>) -> Self {
        self.cursor = Some(cursor.into());
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// `session/list` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionListResult {
    /// Sessions the agent knows about.
    #[serde(default)]
    pub sessions: Vec<SessionInfo>,
    /// Cursor to pass as [`SessionListParams::cursor`] for the next page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A session returned by `session/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    /// Session ID usable with `session/load` or `session/resume`.
    pub session_id: String,
    /// Working directory the session was created with.
    pub cwd: PathBuf,
    /// Extra workspace roots the session was created or loaded with. These
    /// must be re-sent when loading or resuming — the agent does not
    /// restore them implicitly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Human-readable title, if the agent tracks one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// ISO 8601 timestamp of last activity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

/// `session/delete` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionDeleteParams {
    /// Session to delete from the agent's history.
    pub session_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionDeleteParams {
    /// Build params for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
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

impl From<&str> for SessionDeleteParams {
    fn from(session_id: &str) -> Self {
        Self::new(session_id)
    }
}

impl From<String> for SessionDeleteParams {
    fn from(session_id: String) -> Self {
        Self::new(session_id)
    }
}

/// `session/delete` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionDeleteResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `session/close` request parameters: ends a live session without deleting
/// its history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCloseParams {
    /// Session to close.
    pub session_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionCloseParams {
    /// Build params for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
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

impl From<&str> for SessionCloseParams {
    fn from(session_id: &str) -> Self {
        Self::new(session_id)
    }
}

impl From<String> for SessionCloseParams {
    fn from(session_id: String) -> Self {
        Self::new(session_id)
    }
}

/// `session/close` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCloseResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn list_params_and_result_round_trip() {
        let params = SessionListParams::new().cwd("/repo").cursor("page-2");
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["cwd"], "/repo");
        assert_eq!(json["cursor"], "page-2");

        let result: SessionListResult = serde_json::from_value(json!({
            "sessions": [{
                "sessionId": "s1",
                "cwd": "/repo",
                "additionalDirectories": ["/other"],
                "title": "T",
                "updatedAt": "2025-01-01T00:00:00Z",
                "vendor_field": 1,
            }],
            "nextCursor": "page-3",
        }))
        .expect("deserializes");
        assert_eq!(result.sessions[0].session_id, "s1");
        assert_eq!(result.sessions[0].title.as_deref(), Some("T"));
        assert_eq!(result.next_cursor.as_deref(), Some("page-3"));
        // Unknown fields on SessionInfo are preserved verbatim.
        assert_eq!(result.sessions[0].extra["vendor_field"], 1);
        let back = serde_json::to_value(&result.sessions[0]).expect("serializes");
        assert_eq!(back["additionalDirectories"], json!(["/other"]));
        assert_eq!(back["vendor_field"], 1);
    }

    #[test]
    fn delete_and_close_carry_session_id_and_meta() {
        for json in [
            serde_json::to_value(SessionDeleteParams::new("s1").meta(json!({"k": 1})))
                .expect("serializes"),
            serde_json::to_value(SessionCloseParams::new("s1").meta(json!({"k": 1})))
                .expect("serializes"),
        ] {
            assert_eq!(json["sessionId"], "s1");
            assert_eq!(json["_meta"]["k"], 1);
        }
    }
}
