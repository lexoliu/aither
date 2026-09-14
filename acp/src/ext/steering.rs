//! `_session/steering` — inject a prompt into a running turn.
//!
//! A provider-neutral convention: an agent advertises steering support with
//! a `_meta.steering: {supported: true}` object on its `initialize` result.
//! Steering lets the client add content to a turn that is already running,
//! rather than waiting for it to finish.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{AcpClient, ClientError, ClientHandler};
use crate::macros::string_enum;
use crate::protocol::{ContentBlock, ExtMethod, InitializeResult};

/// The steering extension method.
pub const METHOD: ExtMethod = ExtMethod::new("_session/steering");

/// `InitializeResult.meta` key carrying the steering support object.
pub const CAPABILITY: &str = "steering";

/// Whether the agent advertised steering support.
#[must_use]
pub fn advertised(result: &InitializeResult) -> bool {
    result
        .meta
        .as_ref()
        .and_then(|meta| meta.get(CAPABILITY))
        .and_then(|cap| cap.get("supported"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

string_enum! {
    /// How the agent applied a steering prompt.
    pub enum SteeringOutcome {
        /// The prompt was added to the running turn.
        Injected = "injected",
        /// The prompt could not be merged, so the agent started a new turn.
        StartedNewTurn = "startedNewTurn",
        /// The steering request was rejected.
        Failed = "failed",
    }
}

/// `_session/steering` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SteeringParams {
    /// Session whose running turn should receive the prompt.
    pub session_id: String,
    /// Prompt content to inject.
    pub prompt: Vec<ContentBlock>,
}

impl SteeringParams {
    /// Build steering params for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, prompt: Vec<ContentBlock>) -> Self {
        Self {
            session_id: session_id.into(),
            prompt,
        }
    }
}

/// `_session/steering` result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SteeringResult {
    /// How the agent applied the prompt.
    pub outcome: SteeringOutcome,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Steer the session's running turn with additional prompt content.
///
/// # Errors
///
/// Returns an error if the connection is closed or the agent replies with an
/// error or a malformed result.
pub async fn steer<H: ClientHandler>(
    client: &AcpClient<H>,
    session_id: &str,
    prompt: Vec<ContentBlock>,
) -> Result<SteeringOutcome, ClientError> {
    let result: SteeringResult = client
        .ext_request(&METHOD, &SteeringParams::new(session_id, prompt))
        .await?;
    Ok(result.outcome)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::protocol::TextContent;

    /// Steering support is a `supported: true` flag under `_meta.steering`.
    #[test]
    fn advertised_flag() {
        let result: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": 1,
            "_meta": {"steering": {"supported": true}},
        }))
        .expect("deserializes");
        assert!(advertised(&result));

        let result: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": 1,
            "_meta": {"steering": {"supported": false}},
        }))
        .expect("deserializes");
        assert!(!advertised(&result));

        let result: InitializeResult =
            serde_json::from_value(json!({"protocolVersion": 1})).expect("deserializes");
        assert!(!advertised(&result));
    }

    /// Params carry the prompt content blocks verbatim.
    #[test]
    fn params_wire_format() {
        let params = SteeringParams::new(
            "s1",
            vec![ContentBlock::Text(TextContent {
                text: "keep going".to_string(),
                annotations: None,
                meta: None,
            })],
        );
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["prompt"][0]["type"], "text");
        assert_eq!(json["prompt"][0]["text"], "keep going");
    }

    /// Outcomes tag in camelCase; unknown outcomes survive round-tripping.
    #[test]
    fn outcome_wire_format() {
        let result: SteeringResult =
            serde_json::from_value(json!({"outcome": "startedNewTurn"})).expect("deserializes");
        assert_eq!(result.outcome, SteeringOutcome::StartedNewTurn);

        let result: SteeringResult =
            serde_json::from_value(json!({"outcome": "queued"})).expect("deserializes");
        assert_eq!(result.outcome, SteeringOutcome::Other("queued".to_string()));
        assert_eq!(
            serde_json::to_value(&result).expect("serializes")["outcome"],
            "queued"
        );
    }
}
