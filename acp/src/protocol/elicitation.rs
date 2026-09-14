//! Elicitation: agent-initiated structured user input.
//!
//! An agent that needs structured input — a form the user fills in, or a URL
//! the user visits — sends `elicitation/create`. The client answers with an
//! [`ElicitationAction`] and, for `url` mode, completes the flow with an
//! `elicitation/complete` notification once the user is done.
//!
//! Support is negotiated through
//! [`ClientCapabilities::elicitation`](super::ClientCapabilities::elicitation).

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

use super::RequestId;

/// `elicitation/create` request parameters.
///
/// `scope` (a `sessionId`/`toolCallId` pair, or a `requestId`) is flattened
/// into [`ElicitationMode`]'s variants, matching the wire layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationCreateParams {
    /// How the client should collect the input: a form or a URL visit.
    #[serde(flatten)]
    pub mode: ElicitationMode,
    /// Human-readable explanation of what is being asked and why.
    pub message: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ElicitationCreateParams {
    /// Build a request with the given mode and message.
    #[must_use]
    pub fn new(mode: impl Into<ElicitationMode>, message: impl Into<String>) -> Self {
        Self {
            mode: mode.into(),
            message: message.into(),
            meta: None,
        }
    }

    /// The scope this elicitation belongs to.
    #[must_use]
    pub const fn scope(&self) -> &ElicitationScope {
        self.mode.scope()
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// The interaction the client should run for an elicitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ElicitationMode {
    /// Render `requestedSchema` as a form and collect the values.
    Form(ElicitationFormMode),
    /// Open `url` and wait for the matching `elicitation/complete`.
    Url(ElicitationUrlMode),
    /// A mode this crate does not model; the raw fields are preserved.
    #[serde(untagged)]
    Other(OtherElicitationMode),
}

impl ElicitationMode {
    /// The scope this elicitation belongs to.
    #[must_use]
    pub const fn scope(&self) -> &ElicitationScope {
        match self {
            Self::Form(form) => &form.scope,
            Self::Url(url) => &url.scope,
            Self::Other(other) => &other.scope,
        }
    }
}

impl From<ElicitationFormMode> for ElicitationMode {
    fn from(mode: ElicitationFormMode) -> Self {
        Self::Form(mode)
    }
}

impl From<ElicitationUrlMode> for ElicitationMode {
    fn from(mode: ElicitationUrlMode) -> Self {
        Self::Url(mode)
    }
}

/// `form` mode: the client renders [`ElicitationSchema`] and returns the
/// collected values in [`ElicitationAcceptAction::content`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationFormMode {
    /// What the elicitation belongs to.
    #[serde(flatten)]
    pub scope: ElicitationScope,
    /// JSON Schema describing the values to collect.
    pub requested_schema: ElicitationSchema,
}

impl ElicitationFormMode {
    /// Build a form-mode request for `scope`.
    #[must_use]
    pub fn new(scope: impl Into<ElicitationScope>, requested_schema: ElicitationSchema) -> Self {
        Self {
            scope: scope.into(),
            requested_schema,
        }
    }
}

/// `url` mode: the client opens `url` and waits for `elicitation/complete`
/// with the matching `elicitation_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationUrlMode {
    /// What the elicitation belongs to.
    #[serde(flatten)]
    pub scope: ElicitationScope,
    /// ID the agent later echoes in `elicitation/complete`.
    pub elicitation_id: String,
    /// URL to open.
    pub url: String,
}

impl ElicitationUrlMode {
    /// Build a url-mode request for `scope`.
    #[must_use]
    pub fn new(
        scope: impl Into<ElicitationScope>,
        elicitation_id: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            scope: scope.into(),
            elicitation_id: elicitation_id.into(),
            url: url.into(),
        }
    }
}

/// An elicitation mode this crate does not model. The raw fields are
/// preserved so a client can still forward them.
#[derive(Debug, Clone, Serialize)]
pub struct OtherElicitationMode {
    /// The unrecognized `mode` tag.
    pub mode: String,
    /// The scope, if it uses one of the known scope shapes.
    #[serde(flatten)]
    pub scope: ElicitationScope,
    /// Every remaining field of the mode object.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_ELICITATION_MODES: &[&str] = &["form", "url"];

impl<'de> Deserialize<'de> for OtherElicitationMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let mode = fields
            .remove("mode")
            .ok_or_else(|| D::Error::missing_field("mode"))?;
        let Value::String(mode) = mode else {
            return Err(D::Error::custom("`mode` must be a string"));
        };
        if KNOWN_ELICITATION_MODES.contains(&mode.as_str()) {
            return Err(D::Error::custom(format!(
                "known elicitation mode `{mode}` did not match its schema"
            )));
        }
        let scope = serde_json::from_value(Value::Object(fields.clone().into_iter().collect()))
            .map_err(D::Error::custom)?;
        // Scope fields belong to the scope, not to the mode's own extras.
        fields.remove("sessionId");
        fields.remove("toolCallId");
        fields.remove("requestId");
        Ok(Self {
            mode,
            scope,
            fields,
        })
    }
}

/// What an elicitation belongs to: a session (optionally narrowed to one
/// tool call) or another in-flight request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ElicitationScope {
    /// Scoped to a session, optionally to one tool call inside it.
    Session(ElicitationSessionScope),
    /// Scoped to a specific request by its JSON-RPC ID.
    Request(ElicitationRequestScope),
}

impl From<ElicitationSessionScope> for ElicitationScope {
    fn from(scope: ElicitationSessionScope) -> Self {
        Self::Session(scope)
    }
}

impl From<ElicitationRequestScope> for ElicitationScope {
    fn from(scope: ElicitationRequestScope) -> Self {
        Self::Request(scope)
    }
}

/// Session scope for an elicitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationSessionScope {
    /// Session the elicitation belongs to.
    pub session_id: String,
    /// Tool call the elicitation belongs to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ElicitationSessionScope {
    /// Build a session scope.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            tool_call_id: None,
        }
    }

    /// Narrow the scope to one tool call.
    #[must_use]
    pub fn tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }
}

/// Request scope for an elicitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationRequestScope {
    /// ID of the request this elicitation refines.
    pub request_id: RequestId,
}

impl ElicitationRequestScope {
    /// Build a request scope.
    #[must_use]
    pub fn new(request_id: impl Into<RequestId>) -> Self {
        Self {
            request_id: request_id.into(),
        }
    }
}

/// `elicitation/create` response: the user's answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElicitationCreateResult {
    /// What the user did.
    #[serde(flatten)]
    pub action: ElicitationAction,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ElicitationCreateResult {
    /// Build a response for `action`.
    #[must_use]
    pub fn new(action: impl Into<ElicitationAction>) -> Self {
        Self {
            action: action.into(),
            meta: None,
        }
    }

    /// `accept` response carrying the collected form values.
    #[must_use]
    pub fn accept(content: BTreeMap<String, ElicitationContentValue>) -> Self {
        Self::new(ElicitationAcceptAction {
            content: Some(content),
        })
    }

    /// `decline` response: the user refused to provide input.
    #[must_use]
    pub fn decline() -> Self {
        Self::new(ElicitationAction::Decline)
    }

    /// `cancel` response: the user dismissed the interaction.
    #[must_use]
    pub fn cancel() -> Self {
        Self::new(ElicitationAction::Cancel)
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// How the user answered an elicitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ElicitationAction {
    /// The user provided input; `content` carries the collected values for
    /// form mode (absent for url mode, where the agent already has them).
    Accept(ElicitationAcceptAction),
    /// The user refused to provide input.
    Decline,
    /// The user dismissed the interaction.
    Cancel,
    /// An action this crate does not model; the raw fields are preserved.
    #[serde(untagged)]
    Other(OtherElicitationAction),
}

impl From<ElicitationAcceptAction> for ElicitationAction {
    fn from(action: ElicitationAcceptAction) -> Self {
        Self::Accept(action)
    }
}

/// The `accept` action payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElicitationAcceptAction {
    /// Values collected for a `form` elicitation, keyed by property name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<BTreeMap<String, ElicitationContentValue>>,
}

/// An elicitation action this crate does not model.
#[derive(Debug, Clone, Serialize)]
pub struct OtherElicitationAction {
    /// The unrecognized `action` tag.
    pub action: String,
    /// Every remaining field of the action object.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_ELICITATION_ACTIONS: &[&str] = &["accept", "decline", "cancel"];

impl<'de> Deserialize<'de> for OtherElicitationAction {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let action = fields
            .remove("action")
            .ok_or_else(|| D::Error::missing_field("action"))?;
        let Value::String(action) = action else {
            return Err(D::Error::custom("`action` must be a string"));
        };
        if KNOWN_ELICITATION_ACTIONS.contains(&action.as_str()) {
            return Err(D::Error::custom(format!(
                "known elicitation action `{action}` did not match its schema"
            )));
        }
        Ok(Self { action, fields })
    }
}

/// One collected form value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ElicitationContentValue {
    /// A text value.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A non-integer numeric value.
    Number(f64),
    /// A flag value.
    Boolean(bool),
    /// Selected entries of a multi-select.
    StringArray(Vec<String>),
}

impl From<String> for ElicitationContentValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for ElicitationContentValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<i64> for ElicitationContentValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for ElicitationContentValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for ElicitationContentValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<Vec<String>> for ElicitationContentValue {
    fn from(value: Vec<String>) -> Self {
        Self::StringArray(value)
    }
}

/// `elicitation/complete` notification parameters: the agent tells the
/// client a URL-mode elicitation is finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationCompleteParams {
    /// ID from the original `url`-mode `elicitation/create`.
    pub elicitation_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// The JSON Schema for a `form` elicitation. Only `"type": "object"` is
/// defined.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationSchema {
    /// Schema kind; always `"object"`.
    #[serde(rename = "type", default)]
    pub kind: ElicitationSchemaType,
    /// Title to render above the form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Properties to collect, keyed by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, ElicitationPropertySchema>,
    /// Names of required properties.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    /// Human-readable description of the form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ElicitationSchema {
    /// Build an empty object schema.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a property; `required` marks it mandatory.
    #[must_use]
    pub fn property(
        mut self,
        name: impl Into<String>,
        schema: impl Into<ElicitationPropertySchema>,
        required: bool,
    ) -> Self {
        let name = name.into();
        if required {
            self.required
                .get_or_insert_with(Vec::new)
                .push(name.clone());
        }
        self.properties.insert(name, schema.into());
        self
    }

    /// Set the title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
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

/// The only schema kind ACP elicitation defines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElicitationSchemaType {
    /// An object with named properties.
    #[default]
    Object,
}

/// Schema for one form property.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ElicitationPropertySchema {
    /// A text field.
    String(StringPropertySchema),
    /// A non-integer numeric field.
    Number(NumberPropertySchema),
    /// An integer numeric field.
    Integer(IntegerPropertySchema),
    /// A checkbox.
    Boolean(BooleanPropertySchema),
    /// A multi-select.
    Array(MultiSelectPropertySchema),
    /// A property kind this crate does not model; the raw fields are
    /// preserved.
    #[serde(untagged)]
    Other(OtherElicitationPropertySchema),
}

impl From<StringPropertySchema> for ElicitationPropertySchema {
    fn from(schema: StringPropertySchema) -> Self {
        Self::String(schema)
    }
}

impl From<NumberPropertySchema> for ElicitationPropertySchema {
    fn from(schema: NumberPropertySchema) -> Self {
        Self::Number(schema)
    }
}

impl From<IntegerPropertySchema> for ElicitationPropertySchema {
    fn from(schema: IntegerPropertySchema) -> Self {
        Self::Integer(schema)
    }
}

impl From<BooleanPropertySchema> for ElicitationPropertySchema {
    fn from(schema: BooleanPropertySchema) -> Self {
        Self::Boolean(schema)
    }
}

impl From<MultiSelectPropertySchema> for ElicitationPropertySchema {
    fn from(schema: MultiSelectPropertySchema) -> Self {
        Self::Array(schema)
    }
}

/// Schema for a text property.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StringPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum text length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<u32>,
    /// Maximum text length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u32>,
    /// Regex the value must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Well-known text format (email, URI, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<ElicitationStringFormat>,
    /// Value used when the user does not fill the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    /// Fixed set of allowed values.
    #[serde(rename = "enum", default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Allowed values with titles and descriptions.
    #[serde(rename = "oneOf", default, skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<ElicitationEnumOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// One labelled choice in a [`StringPropertySchema::one_of`] list or a
/// [`MultiSelectItems::Titled`] item set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElicitationEnumOption {
    /// The value reported when this option is selected.
    #[serde(rename = "const")]
    pub value: String,
    /// Human-readable label.
    pub title: String,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Well-known text formats for a string property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ElicitationStringFormat {
    /// An email address.
    Email,
    /// A URI.
    Uri,
    /// A date (`YYYY-MM-DD`).
    Date,
    /// A date-time (`YYYY-MM-DDThh:mm:ss`).
    DateTime,
}

/// Schema for a non-integer numeric property.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NumberPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Schema for an integer property.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntegerPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<i64>,
    /// Maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<i64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<i64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Schema for a boolean property.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BooleanPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Default state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<bool>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Schema for a multi-select property.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiSelectPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum selections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_items: Option<u64>,
    /// Maximum selections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_items: Option<u64>,
    /// The selectable items.
    pub items: MultiSelectItems,
    /// Pre-selected values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Vec<String>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// The item set of a multi-select property: a bare value list
/// (`"type": "string", "enum": [...]`) or a titled list (`"anyOf": [...]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MultiSelectItems {
    /// A bare list of selectable values.
    String(StringMultiSelectItems),
    /// Selectable values with titles.
    #[serde(untagged)]
    Titled(TitledMultiSelectItems),
    /// An item set this crate does not model; the raw fields are preserved.
    #[serde(untagged)]
    Other(OtherMultiSelectItems),
}

/// A bare `{"type": "string", "enum": [...]}` item set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringMultiSelectItems {
    /// Selectable values.
    #[serde(rename = "enum")]
    pub values: Vec<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A titled `{"anyOf": [...]}` item set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitledMultiSelectItems {
    /// Selectable options.
    #[serde(rename = "anyOf")]
    pub options: Vec<ElicitationEnumOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// An item-set shape this crate does not model.
///
/// The `fields` bag holds raw JSON, so equality is not derivable.
#[derive(Debug, Clone, Serialize)]
pub struct OtherMultiSelectItems {
    /// The unrecognized `type` tag.
    #[serde(rename = "type")]
    pub kind: String,
    /// Every remaining field of the item set.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_MULTI_SELECT_ITEM_TYPES: &[&str] = &["string"];

impl<'de> Deserialize<'de> for OtherMultiSelectItems {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let kind = fields
            .remove("type")
            .ok_or_else(|| D::Error::missing_field("type"))?;
        let Value::String(kind) = kind else {
            return Err(D::Error::custom("`type` must be a string"));
        };
        if KNOWN_MULTI_SELECT_ITEM_TYPES.contains(&kind.as_str()) {
            return Err(D::Error::custom(format!(
                "known multi-select item type `{kind}` did not match its schema"
            )));
        }
        Ok(Self { kind, fields })
    }
}

/// A property kind this crate does not model.
///
/// The `fields` bag holds raw JSON, so equality is not derivable.
#[derive(Debug, Clone, Serialize)]
pub struct OtherElicitationPropertySchema {
    /// The unrecognized `type` tag.
    #[serde(rename = "type")]
    pub kind: String,
    /// Every remaining field of the schema.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_PROPERTY_TYPES: &[&str] = &["string", "number", "integer", "boolean", "array"];

impl<'de> Deserialize<'de> for OtherElicitationPropertySchema {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let kind = fields
            .remove("type")
            .ok_or_else(|| D::Error::missing_field("type"))?;
        let Value::String(kind) = kind else {
            return Err(D::Error::custom("`type` must be a string"));
        };
        if KNOWN_PROPERTY_TYPES.contains(&kind.as_str()) {
            return Err(D::Error::custom(format!(
                "known elicitation property type `{kind}` did not match its schema"
            )));
        }
        Ok(Self { kind, fields })
    }
}

/// What the client offers for elicitation (`clientCapabilities.elicitation`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElicitationCapabilities {
    /// Whether the client can render forms. `{}` advertises support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<ElicitationFormCapabilities>,
    /// Whether the client can open URLs for url-mode elicitation. `{}`
    /// advertises support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<ElicitationUrlCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Marker for form elicitation support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElicitationFormCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Marker for URL elicitation support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElicitationUrlCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A form-mode create request flattens mode and scope into the params.
    #[test]
    fn form_create_params_wire_format() {
        let params = ElicitationCreateParams::new(
            ElicitationFormMode::new(
                ElicitationSessionScope::new("s1").tool_call_id("tc-9"),
                ElicitationSchema::new().property(
                    "name",
                    StringPropertySchema {
                        title: Some("Name".to_string()),
                        ..StringPropertySchema::default()
                    },
                    true,
                ),
            ),
            "Who are you?",
        );
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["mode"], "form");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["toolCallId"], "tc-9");
        assert_eq!(json["requestedSchema"]["type"], "object");
        assert_eq!(json["requestedSchema"]["required"], json!(["name"]));

        let parsed: ElicitationCreateParams = serde_json::from_value(json).expect("deserializes");
        let ElicitationMode::Form(form) = &parsed.mode else {
            panic!("expected form mode");
        };
        let ElicitationScope::Session(scope) = &form.scope else {
            panic!("expected session scope");
        };
        assert_eq!(scope.session_id, "s1");
        assert_eq!(scope.tool_call_id.as_deref(), Some("tc-9"));
    }

    /// A url-mode create request carries the elicitation ID and URL.
    #[test]
    fn url_create_params_wire_format() {
        let params = ElicitationCreateParams::new(
            ElicitationUrlMode::new(
                ElicitationRequestScope::new(RequestId::from(7_i64)),
                "elic-1",
                "http://localhost:8787/auth",
            ),
            "Sign in",
        );
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["mode"], "url");
        assert_eq!(json["requestId"], 7);
        assert_eq!(json["elicitationId"], "elic-1");

        let parsed: ElicitationCreateParams = serde_json::from_value(json).expect("deserializes");
        let ElicitationMode::Url(url) = &parsed.mode else {
            panic!("expected url mode");
        };
        assert_eq!(url.elicitation_id, "elic-1");
        let ElicitationScope::Request(scope) = &url.scope else {
            panic!("expected request scope");
        };
        assert_eq!(scope.request_id, RequestId::from(7_i64));
    }

    /// An unknown mode lands in `Other` with every field preserved.
    #[test]
    fn unknown_mode_preserved() {
        let params: ElicitationCreateParams = serde_json::from_value(json!({
            "mode": "voice",
            "sessionId": "s1",
            "message": "Say it",
            "pitch": "high",
        }))
        .expect("deserializes");
        let ElicitationMode::Other(other) = &params.mode else {
            panic!("expected Other mode");
        };
        assert_eq!(other.mode, "voice");
        let ElicitationScope::Session(scope) = &other.scope else {
            panic!("expected session scope");
        };
        assert_eq!(scope.session_id, "s1");
        assert_eq!(other.fields["pitch"], "high");
        let back = serde_json::to_value(&params).expect("serializes");
        assert_eq!(back["mode"], "voice");
        assert_eq!(back["pitch"], "high");
    }

    /// Results flatten the action; unknown actions land in `Other`.
    #[test]
    fn result_actions_wire_format() {
        let result =
            ElicitationCreateResult::accept(BTreeMap::from([("age".to_string(), 42_i64.into())]));
        let json = serde_json::to_value(&result).expect("serializes");
        assert_eq!(json["action"], "accept");
        assert_eq!(json["content"]["age"], 42);

        let parsed: ElicitationCreateResult =
            serde_json::from_value(json!({"action": "decline"})).expect("deserializes");
        assert!(matches!(parsed.action, ElicitationAction::Decline));

        let parsed: ElicitationCreateResult =
            serde_json::from_value(json!({"action": "defer", "eta": 5})).expect("deserializes");
        let ElicitationAction::Other(other) = &parsed.action else {
            panic!("expected Other action");
        };
        assert_eq!(other.action, "defer");
        assert_eq!(other.fields["eta"], 5);
    }

    /// Property schemas tag by `type`; unknown kinds land in `Other`.
    #[test]
    fn property_schema_variants() {
        let schema: ElicitationPropertySchema = serde_json::from_value(json!({
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b"]},
            "minItems": 1,
        }))
        .expect("deserializes");
        let ElicitationPropertySchema::Array(multi) = &schema else {
            panic!("expected array schema");
        };
        let MultiSelectItems::String(items) = &multi.items else {
            panic!("expected string items");
        };
        assert_eq!(items.values, ["a", "b"]);

        let schema: ElicitationPropertySchema =
            serde_json::from_value(json!({"type": "color", "swatches": 3})).expect("deserializes");
        let ElicitationPropertySchema::Other(other) = &schema else {
            panic!("expected Other schema");
        };
        assert_eq!(other.kind, "color");
        assert_eq!(other.fields["swatches"], 3);
    }

    /// `elicitation/complete` carries just the elicitation ID.
    #[test]
    fn complete_params_wire_format() {
        let params: ElicitationCompleteParams =
            serde_json::from_value(json!({"elicitationId": "e1"})).expect("deserializes");
        assert_eq!(params.elicitation_id, "e1");
        assert_eq!(
            serde_json::to_value(&params).expect("serializes"),
            json!({"elicitationId": "e1"})
        );
    }
}
