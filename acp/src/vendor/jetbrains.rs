//! The `_meta.jetbrains.air` namespace — `JetBrains`'s capability
//! advertisement and per-update metadata convention, spoken by e.g.
//! codex-acp when the client is a `JetBrains` IDE.
//!
//! The shape is a single `_meta.jetbrains.air` object carrying a `version`
//! and a `capabilities` array; individual session updates hang extra keys
//! under the same `air` object (e.g. `air.asyncTasks.backgrounded` on a
//! `tool_call_update`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::protocol::ClientCapabilities;

/// The `_meta` key carrying the `JetBrains` namespace.
pub const META_KEY: &str = "jetbrains";

/// The key inside [`META_KEY`] carrying the AIR object.
pub const AIR_KEY: &str = "air";

/// The `version` key inside the AIR object.
pub const VERSION_KEY: &str = "version";

/// The `capabilities` key inside the AIR object.
pub const CAPABILITIES_KEY: &str = "capabilities";

/// AIR extension version this module speaks.
pub const VERSION: u64 = 1;

/// Keys usable inside the AIR `capabilities` array.
pub mod capability {
    /// The agent reports session-ending failures through
    /// `_meta.jetbrains.air.sessionFailure`.
    pub const SESSION_FAILURE: &str = "sessionFailure";
    /// The agent produces `agentFileChangeReport` updates.
    pub const AGENT_FILE_CHANGE_REPORT: &str = "agentFileChangeReport";
    /// The agent exposes native subagent sessions.
    pub const NATIVE_SUBAGENT_SESSIONS: &str = "nativeSubagentSessions";
    /// The agent reports background async tasks.
    pub const ASYNC_TASKS: &str = "asyncTasks";
    /// Config options carry a `recommendedValue`.
    pub const RECOMMENDED_CONFIG_VALUE: &str = "recommendedValue";
}

/// The key under `air` marking a `tool_call_update` whose command moved to
/// background execution: `air.asyncTasks.backgrounded`.
pub const ASYNC_TASKS_KEY: &str = "asyncTasks";

/// The `backgrounded` flag inside `air.asyncTasks`.
pub const BACKGROUNDED_KEY: &str = "backgrounded";

/// The parsed `_meta.jetbrains.air` object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AirMeta {
    /// AIR version; [`VERSION`] for agents this module interoperates with.
    pub version: u64,
    /// Capability keys the remote side supports.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Other keys under `air` (per-update payloads), preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Parse the AIR object out of a `_meta` value.
#[must_use]
pub fn air(meta: Option<&Value>) -> Option<AirMeta> {
    serde_json::from_value(meta?.get(META_KEY)?.get(AIR_KEY)?.clone()).ok()
}

/// Whether `meta` advertises AIR support at version >= [`VERSION`] with
/// `capability` in its capabilities array.
#[must_use]
pub fn supports(meta: Option<&Value>, capability: &str) -> bool {
    air(meta).is_some_and(|air| {
        air.version >= VERSION && air.capabilities.iter().any(|cap| cap == capability)
    })
}

/// Whether a `tool_call_update`'s `_meta` marks it as backgrounded via
/// `_meta.jetbrains.air.asyncTasks.backgrounded`.
#[must_use]
pub fn backgrounded(meta: Option<&Value>) -> bool {
    meta.and_then(|meta| meta.get(META_KEY))
        .and_then(|meta| meta.get(AIR_KEY))
        .and_then(|air| air.get(ASYNC_TASKS_KEY))
        .and_then(|tasks| tasks.get(BACKGROUNDED_KEY))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Merge `key: value` into `_meta.jetbrains.air` of `meta`, preserving all
/// other `_meta` content and setting `version` to [`VERSION`].
///
/// Pass `None` for `meta` to build a fresh `_meta` object.
#[must_use]
pub fn with_meta(meta: Option<Value>, key: &str, value: Value) -> Value {
    let mut root = meta.unwrap_or_else(|| json!({}));
    let Some(root_object) = root.as_object_mut() else {
        return root;
    };
    let jetbrains = root_object
        .entry(META_KEY.to_string())
        .or_insert_with(|| json!({}));
    let Some(jetbrains_object) = jetbrains.as_object_mut() else {
        return root;
    };
    let air = jetbrains_object
        .entry(AIR_KEY.to_string())
        .or_insert_with(|| json!({}));
    if let Some(air_object) = air.as_object_mut() {
        air_object.insert(VERSION_KEY.to_string(), json!(VERSION));
        air_object.insert(key.to_string(), value);
    }
    root
}

/// Advertise AIR support on client capabilities, preserving any `_meta`
/// content already present.
///
/// Sets `_meta.jetbrains.air` to `{version, capabilities: keys}`.
pub fn advertise(caps: &mut ClientCapabilities, keys: &[&str]) {
    caps.meta = Some(with_meta(caps.meta.take(), CAPABILITIES_KEY, json!(keys)));
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// `air` decodes `_meta.jetbrains.air` and keeps unknown keys.
    #[test]
    fn air_parsing() {
        let meta = json!({
            "jetbrains": {
                "air": {
                    "version": 1,
                    "capabilities": ["asyncTasks", "sessionFailure"],
                    "asyncTasks": {"backgrounded": true},
                }
            }
        });
        let parsed = air(Some(&meta)).expect("air");
        assert_eq!(parsed.version, 1);
        assert!(supports(Some(&meta), capability::ASYNC_TASKS));
        assert!(supports(Some(&meta), capability::SESSION_FAILURE));
        assert!(!supports(Some(&meta), "agentMode"));
        assert!(backgrounded(Some(&meta)));

        // Older versions and absent namespaces report no support.
        let old = json!({"jetbrains": {"air": {"version": 0, "capabilities": ["asyncTasks"]}}});
        assert!(!supports(Some(&old), capability::ASYNC_TASKS));
        assert!(air(None).is_none());
        assert!(!backgrounded(None));
    }

    /// `with_meta` merges into `_meta.jetbrains.air` without losing other
    /// `_meta` content.
    #[test]
    fn with_meta_merges() {
        let meta = with_meta(
            Some(json!({"other": 1, "jetbrains": {"air": {"capabilities": ["a"]}}})),
            ASYNC_TASKS_KEY,
            json!({"backgrounded": true}),
        );
        assert_eq!(meta["other"], 1);
        assert_eq!(meta["jetbrains"]["air"]["version"], VERSION);
        assert_eq!(meta["jetbrains"]["air"]["capabilities"], json!(["a"]));
        assert_eq!(meta["jetbrains"]["air"]["asyncTasks"]["backgrounded"], true);

        let fresh = with_meta(None, CAPABILITIES_KEY, json!(["x"]));
        assert_eq!(fresh["jetbrains"]["air"]["capabilities"], json!(["x"]));
    }

    /// `advertise` writes the AIR capability list onto client capabilities.
    #[test]
    fn advertise_capabilities() {
        let mut caps = ClientCapabilities {
            meta: Some(json!({"existing": true})),
            ..ClientCapabilities::default()
        };
        advertise(&mut caps, &[capability::ASYNC_TASKS]);
        let meta = caps.meta.expect("meta");
        assert_eq!(meta["existing"], true);
        let parsed = air(Some(&meta)).expect("air");
        assert_eq!(parsed.capabilities, [capability::ASYNC_TASKS]);
    }
}
