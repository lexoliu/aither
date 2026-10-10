//! serde types for the JSON transport between this crate and the Swift bridge.
//!
//! The Swift half lives in `swift/Wire.swift`; keep the two in lockstep.

use serde::{Deserialize, Serialize};

/// Full request payload handed to `aither_apple_request_set_json`.
#[derive(Debug, Serialize)]
pub struct WireRequest {
    /// System instructions: one string per `Message::System`, in order.
    pub instructions: Vec<String>,
    /// Transcript entries before the final prompt.
    pub history: Vec<WireEntry>,
    pub prompt: WirePrompt,
    /// `"text"` or `"guided"` (the latter when `schema` is set).
    pub mode: &'static str,
    /// Whether declared tools suspend at the boundary (`respond`) or are
    /// executed natively (`respond_with_tools`).
    pub tools_are_external: bool,
    pub options: WireOptions,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<WireSchemaRoot>,
    pub include_reasoning: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum WireEntry {
    #[serde(rename = "prompt")]
    Prompt {
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        image_indexes: Vec<usize>,
    },
    #[serde(rename = "response")]
    Response { text: String },
    #[serde(rename = "tool_calls")]
    ToolCalls { calls: Vec<WireToolCall> },
    #[serde(rename = "tool_output")]
    ToolOutput {
        id: String,
        name: String,
        text: String,
    },
}

#[derive(Debug, Serialize)]
pub struct WireToolCall {
    pub id: String,
    pub name: String,
    /// Arguments serialized as a JSON document string.
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct WirePrompt {
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub image_indexes: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct WireOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_response_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<WireSampling>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calling_mode: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum WireSampling {
    #[serde(rename = "top_k")]
    TopK {
        k: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
    },
    #[serde(rename = "top_p")]
    TopP {
        p: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
    },
}

#[derive(Debug, Serialize)]
pub struct WireTool {
    pub name: String,
    pub description: String,
    pub schema: WireSchemaRoot,
}

/// A dynamic schema plus its named dependencies (`$defs`), keyed by name so
/// `Reference` nodes resolve against the same names Swift registers.
#[derive(Debug, Serialize)]
pub struct WireSchemaRoot {
    pub schema: WireSchema,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub defs: std::collections::BTreeMap<String, WireSchema>,
}

/// Provider-neutral schema tree mirroring `WireSchema` on the Swift side.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum WireSchema {
    #[serde(rename = "object")]
    Object {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        properties: Vec<WireProperty>,
    },
    #[serde(rename = "any_of")]
    AnyOf {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        choices: Vec<Self>,
    },
    #[serde(rename = "enumeration")]
    Enumeration {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        values: Vec<String>,
    },
    #[serde(rename = "primitive")]
    Primitive {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        kind: &'static str,
        guides: WireGuides,
    },
    #[serde(rename = "array")]
    Array {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        items: Box<Self>,
        #[serde(skip_serializing_if = "Option::is_none")]
        min_items: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_items: Option<usize>,
    },
    /// `{ "type": "reference", "ref": "<def name>" }` — matches the Swift
    /// decoder, which reads `ref`.
    #[serde(rename = "reference")]
    Reference {
        #[serde(rename = "ref")]
        name: String,
    },
    #[serde(rename = "null")]
    Null,
}

#[derive(Debug, Serialize)]
pub struct WireProperty {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: WireSchema,
    #[serde(skip_serializing_if = "core::ops::Not::not")]
    pub optional: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct WireGuides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integer_minimum: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integer_maximum: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
}

// ---- Swift -> Rust payloads ----

/// The terminal `end` event payload.
#[derive(Debug, Deserialize)]
pub struct WireEnd {
    pub status: WireStatus,
    pub code: Option<String>,
    pub message: Option<String>,
    pub tool: Option<String>,
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WireStatus {
    Completed,
    Cancelled,
    Error,
}

#[derive(Debug, Deserialize)]
pub struct WireUsage {
    pub input: i64,
    pub output: i64,
    pub reasoning: i64,
    pub cached: i64,
}

/// Internal call routing. The native framework owns model IDs and pairs outputs;
/// the bridge sequence identifies only the suspended continuation.
#[derive(Debug, Deserialize)]
pub struct WireToolEvent {
    pub seq: u64,
    pub name: String,
    pub arguments: String,
}

/// The `tool_batch` event payload (external mode).
#[derive(Debug, Deserialize)]
pub struct WireToolBatch {
    pub calls: Vec<WireCapturedCall>,
}

#[derive(Debug, Deserialize)]
pub struct WireCapturedCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
