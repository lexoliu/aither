//! # LLM Tool Calling Framework
//!
//!
//! Type-safe tool calling system for Large Language Models. Enables LLMs to execute external
//! functions, access APIs, and interact with systems through well-defined interfaces.
//!
//! ## How a model calls a tool
//!
//! A [`Tool`] declares a name, a description, and an `Arguments` type whose
//! JSON schema is sent to the provider. When the model decides to use a tool it
//! emits a [`ToolCall`](crate::llm::ToolCall) event carrying the tool name and
//! arguments as JSON; this crate does not execute it. Executing the call and
//! feeding the result back is the job of a higher layer such as `aither-agent`,
//! which keeps tool execution under the caller's control.
//!
//! ## Core Components
//!
//! - [`Tool`] - Trait for defining executable tools
//! - [`Tools`] - Registry for managing multiple tools  
//! - [`tool::ToolDefinition`] - Metadata and schema for LLM consumption
//!
//! ## Quick Start
//!
//! ```rust
//! use aither_core::llm::{Tool, ToolResult};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//! use std::borrow::Cow;
//!
//! /// Performs basic math operations.
//! #[derive(JsonSchema, Deserialize)]
//! struct MathArgs {
//!     /// Operation: "add", "subtract", "multiply", "divide"
//!     operation: String,
//!     /// First number
//!     a: f64,
//!     /// Second number
//!     b: f64,
//! }
//!
//! struct Calculator;
//!
//! impl Tool for Calculator {
//!     fn name(&self) -> Cow<'static, str> {
//!         Cow::Borrowed("calculator")
//!     }
//!
//!     type Arguments = MathArgs;
//!     type Res = ToolResult;
//!
//!     async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
//!         let result = match args.operation.as_str() {
//!             "add" => args.a + args.b,
//!             "subtract" => args.a - args.b,
//!             "multiply" => args.a * args.b,
//!             "divide" if args.b != 0.0 => args.a / args.b,
//!             "divide" => return Err(anyhow::Error::msg("Division by zero")),
//!             _ => return Err(anyhow::Error::msg("Unknown operation")),
//!         };
//!         Ok(ToolResult::text(result.to_string()))
//!     }
//! }
//! ```
//!
//! ## Schema Design Best Practices
//!
//! ### 1. Use Clear Documentation Comments
//! Doc comments automatically become schema descriptions:
//!
//! ```rust,ignore
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(JsonSchema, Deserialize)]
//! struct WeatherArgs {
//!     /// City name (e.g., "London", "Tokyo", "New York")
//!     city: String,
//!     /// Temperature unit: "celsius" or "fahrenheit"
//!     #[serde(default = "default_celsius")]
//!     unit: String,
//! }
//!
//! fn default_celsius() -> String { "celsius".to_string() }
//! ```
//!
//! ### 2. Prefer Enums Over Strings
//! Enums provide clear constraints for LLMs:
//!
//! ```rust,ignore
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(JsonSchema, Deserialize)]
//! enum Priority { Low, Medium, High, Critical }
//!
//! #[derive(JsonSchema, Deserialize)]  
//! struct TaskArgs {
//!     /// Task description
//!     description: String,
//!     /// Task priority level
//!     priority: Priority,
//! }
//! ```
//!
//! ### 3. Add Validation Constraints
//! Use schemars attributes for validation:
//!
//! ```rust,ignore
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(JsonSchema, Deserialize)]
//! struct UserArgs {
//!     /// Valid email address
//!     #[schemars(regex(pattern = "^[^@]+@[^@]+\\.[^@]+$"))]
//!     email: String,
//!     /// Age between 13 and 120
//!     #[schemars(range(min = 13, max = 120))]
//!     age: u8,
//!     /// Bio text, max 500 characters
//!     #[schemars(length(max = 500))]
//!     bio: Option<String>,
//! }
//! ```
//!
//! ### 4. Structure Complex Data
//! Break down complex parameters into nested types:
//!
//! ```rust,ignore
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(JsonSchema, Deserialize)]
//! struct Address {
//!     street: String,
//!     city: String,
//!     /// Two-letter country code (e.g., "US", "GB", "JP")
//!     country: String,
//! }
//!
//! #[derive(JsonSchema, Deserialize)]
//! struct CreateUserArgs {
//!     name: String,
//!     address: Address,
//!     /// List of user interests
//!     #[schemars(length(max = 10))]
//!     interests: Vec<String>,
//! }
//! ```
//!

// Re-export procedural macros
#[cfg(feature = "derive")]
pub use aither_derive::tool;
use alloc::borrow::Cow;
use serde_json::Value;

use crate::Result;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{boxed::Box, collections::BTreeMap};
use core::any::Any;
use core::fmt::{Debug, Display};
use core::{future::Future, pin::Pin};
pub use mime::Mime;
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

/// Final structured result from a tool execution.
///
/// This keeps successful content and tool-level failures in distinct variants so
/// UIs and runtimes do not need to infer errors from free-form strings.
///
/// # Example
///
/// ```rust,ignore
/// use aither::llm::tool::ToolResult;
///
/// fn search_tool() -> ToolResult {
///     ToolResult::text("Found 3 results...")
/// }
///
/// fn delete_tool() -> ToolResult {
///     ToolResult::Done
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum ToolResult {
    /// Tool completed with no output to return.
    Done,

    /// UTF-8 plain text output.
    Text {
        /// Plain text content.
        text: String,
    },

    /// Tab-separated tabular output.
    Tsv {
        /// TSV content.
        text: String,
    },

    /// Structured JSON output.
    Json {
        /// JSON content.
        value: Value,
    },

    /// Binary or media payload.
    Binary {
        /// MIME type of the payload (for example `image/png`).
        mime: String,
        /// Raw content bytes.
        content: Vec<u8>,
    },

    /// Tool-level error. This is distinct from transport/runtime failures.
    Error {
        /// Tool-level error message.
        message: String,
    },
}

impl ToolResult {
    /// Creates a plain text result.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Creates a TSV result.
    #[must_use]
    pub fn tsv(s: impl Into<String>) -> Self {
        Self::Tsv { text: s.into() }
    }

    /// Creates a JSON result from a serializable value.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn json<T: Serialize>(value: &T) -> Result<Self> {
        Ok(Self::Json {
            value: serde_json::to_value(value)?,
        })
    }

    /// Creates a JSON result from an already-materialized JSON value.
    #[must_use]
    pub const fn json_value(value: Value) -> Self {
        Self::Json { value }
    }

    /// Creates an image result.
    #[must_use]
    pub fn image(data: Vec<u8>, media_type: &str) -> Self {
        Self::Binary {
            mime: parse_media_type_or_octet_stream(media_type),
            content: data,
        }
    }

    /// Creates a binary result.
    #[must_use]
    pub fn binary(data: Vec<u8>) -> Self {
        Self::Binary {
            mime: mime::APPLICATION_OCTET_STREAM.essence_str().to_string(),
            content: data,
        }
    }

    /// Creates a typed tool error result.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Returns `true` if this is a `Done` variant.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Returns `true` if this is a typed tool error.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Returns plain textual content for text-like variants.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } | Self::Tsv { text } => Some(text),
            Self::Error { message } => Some(message),
            Self::Done | Self::Json { .. } | Self::Binary { .. } => None,
        }
    }

    /// Returns the error message if this is a typed tool error.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            Self::Done
            | Self::Text { .. }
            | Self::Tsv { .. }
            | Self::Json { .. }
            | Self::Binary { .. } => None,
        }
    }

    /// Projects the result into a textual representation safe to re-inject into model context.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub fn render_for_model(&self) -> Result<String> {
        match self {
            Self::Done => Ok(String::new()),
            Self::Text { text } | Self::Tsv { text } => Ok(text.clone()),
            Self::Json { value } => Ok(serde_json::to_string(value)?),
            Self::Binary { mime, content } => {
                let mut rendered = String::new();
                rendered.push_str("[binary tool result: ");
                rendered.push_str(mime);
                rendered.push_str(", ");
                rendered.push_str(content.len().to_string().as_str());
                rendered.push_str(" bytes]");
                Ok(rendered)
            }
            Self::Error { message } => Ok(message.clone()),
        }
    }

    /// Renders the result for CLI display.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub fn render_for_cli(&self) -> Result<String> {
        match self {
            Self::Done => Ok(String::new()),
            Self::Text { text } | Self::Tsv { text } => Ok(text.clone()),
            Self::Json { value } => Ok(serde_json::to_string_pretty(value)?),
            Self::Binary { mime, content } => {
                let mut rendered = String::new();
                rendered.push_str("[binary tool result: ");
                rendered.push_str(mime);
                rendered.push_str(", ");
                rendered.push_str(content.len().to_string().as_str());
                rendered.push_str(" bytes]");
                Ok(rendered)
            }
            Self::Error { message } => Ok(message.clone()),
        }
    }

    /// Parses and returns the MIME type when this result carries binary content.
    #[must_use]
    pub fn mime(&self) -> Option<Mime> {
        match self {
            Self::Binary { mime, .. } => mime.parse().ok(),
            Self::Done
            | Self::Text { .. }
            | Self::Tsv { .. }
            | Self::Json { .. }
            | Self::Error { .. } => None,
        }
    }

    /// Returns raw bytes for binary results.
    #[must_use]
    pub fn content(&self) -> Option<&[u8]> {
        match self {
            Self::Binary { content, .. } => Some(content),
            Self::Done
            | Self::Text { .. }
            | Self::Tsv { .. }
            | Self::Json { .. }
            | Self::Error { .. } => None,
        }
    }
}

/// Conversion trait for values returned by [`Tool::call`].
///
/// The conversion itself is fallible so tool authors can return types like
/// `Result<T: Serialize, E: Error>` and still surface serialization failures as
/// framework errors while preserving typed tool errors inside [`ToolResult`].
pub trait IntoToolResult {
    /// Converts the value into a final [`ToolResult`].
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion cannot be completed.
    fn into_tool_result(self) -> Result<ToolResult>;
}

impl IntoToolResult for ToolResult {
    fn into_tool_result(self) -> Result<ToolResult> {
        Ok(self)
    }
}

impl IntoToolResult for () {
    fn into_tool_result(self) -> Result<ToolResult> {
        Ok(ToolResult::Done)
    }
}

impl IntoToolResult for String {
    fn into_tool_result(self) -> Result<ToolResult> {
        Ok(ToolResult::text(self))
    }
}

impl IntoToolResult for &str {
    fn into_tool_result(self) -> Result<ToolResult> {
        Ok(ToolResult::text(self))
    }
}

impl IntoToolResult for Cow<'_, str> {
    fn into_tool_result(self) -> Result<ToolResult> {
        Ok(ToolResult::text(self.into_owned()))
    }
}

impl IntoToolResult for Value {
    fn into_tool_result(self) -> Result<ToolResult> {
        Ok(ToolResult::json_value(self))
    }
}

impl<T> IntoToolResult for Option<T>
where
    T: IntoToolResult,
{
    fn into_tool_result(self) -> Result<ToolResult> {
        self.map_or_else(|| Ok(ToolResult::Done), IntoToolResult::into_tool_result)
    }
}

impl<T, E> IntoToolResult for core::result::Result<T, E>
where
    T: Serialize,
    E: Display,
{
    fn into_tool_result(self) -> Result<ToolResult> {
        match self {
            Ok(value) => serialize_success_value(&value),
            Err(error) => Ok(ToolResult::error(error.to_string())),
        }
    }
}

fn parse_media_type_or_octet_stream(media_type: &str) -> String {
    media_type
        .parse::<Mime>()
        .unwrap_or(mime::APPLICATION_OCTET_STREAM)
        .essence_str()
        .to_string()
}

fn serialize_success_value<T: Serialize>(value: &T) -> Result<ToolResult> {
    let value = serde_json::to_value(value)?;
    if let Some(tsv) = json_value_to_tsv(&value) {
        return Ok(ToolResult::tsv(tsv));
    }

    match value {
        Value::String(text) => Ok(ToolResult::text(text)),
        other => Ok(ToolResult::json_value(other)),
    }
}

/// Converts a JSON value into TSV when it represents an object or non-empty array.
#[must_use]
pub fn json_value_to_tsv(value: &Value) -> Option<String> {
    let rows = match value {
        Value::Array(arr) if !arr.is_empty() => arr
            .iter()
            .map(|value| flatten_json_value(value, ""))
            .collect::<Vec<_>>(),
        Value::Object(_) => alloc::vec![flatten_json_value(value, "")],
        Value::Array(_) | Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
            return None;
        }
    };

    if rows.is_empty() {
        return None;
    }

    let mut columns: Vec<String> = Vec::new();
    let mut seen: alloc::collections::BTreeSet<String> = alloc::collections::BTreeSet::new();
    for row in &rows {
        for (key, _) in row {
            if seen.insert(key.clone()) {
                columns.push(key.clone());
            }
        }
    }

    if columns.is_empty() {
        return None;
    }

    let mut tsv = String::new();
    for (index, column) in columns.iter().enumerate() {
        if index > 0 {
            tsv.push('\t');
        }
        tsv.push_str(&escape_tsv_field(column));
    }
    tsv.push('\n');

    for row in &rows {
        let row_map: alloc::collections::BTreeMap<&str, &str> = row
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect::<alloc::collections::BTreeMap<&str, &str>>();
        for (index, column) in columns.iter().enumerate() {
            if index > 0 {
                tsv.push('\t');
            }
            if let Some(value) = row_map.get(column.as_str()) {
                tsv.push_str(&escape_tsv_field(value));
            }
        }
        tsv.push('\n');
    }

    Some(tsv)
}

fn flatten_json_value(value: &Value, prefix: &str) -> Vec<(String, String)> {
    let mut flattened = Vec::new();
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let full_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flattened.extend(flatten_json_value(child, &full_key));
            }
        }
        Value::Array(_) => {
            let serialized = serde_json::to_string(value).unwrap_or_default();
            flattened.push((prefix.to_string(), serialized));
        }
        Value::String(text) => {
            flattened.push((prefix.to_string(), text.clone()));
        }
        Value::Number(number) => {
            flattened.push((prefix.to_string(), number.to_string()));
        }
        Value::Bool(boolean) => {
            flattened.push((prefix.to_string(), boolean.to_string()));
        }
        Value::Null => {
            flattened.push((prefix.to_string(), String::new()));
        }
    }
    flattened
}

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

/// Tools that can be called by language models.
///
/// # Example
///
/// ```rust,ignore
/// use aither::llm::{Tool, ToolResult};
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(JsonSchema, Deserialize)]
/// struct CalculatorArgs {
///     operation: String,
///     a: f64,
///     b: f64,
/// }
///
/// struct Calculator;
///
/// impl Tool for Calculator {
///     type Arguments = CalculatorArgs;
///     type Res = ToolResult;
///
///     async fn call(&mut self, args: Self::Arguments) -> aither::Result<Self::Res> {
///         match args.operation.as_str() {
///             "add" => Ok(ToolResult::text((args.a + args.b).to_string())),
///             "subtract" => Ok(ToolResult::text((args.a - args.b).to_string())),
///             "multiply" => Ok(ToolResult::text((args.a * args.b).to_string())),
///             "divide" => {
///                 if args.b != 0.0 {
///                     Ok(ToolResult::text((args.a / args.b).to_string()))
///                 } else {
///                     Err(anyhow::Error::msg("Division by zero"))
///                 }
///             }
///             _ => Err(anyhow::Error::msg("Unknown operation")),
///         }
///     }
/// }
/// ```
pub trait Tool: Send + Sync {
    /// Tool name. Must be unique.
    fn name(&self) -> Cow<'static, str>;

    /// What the tool does, as shown to the model.
    ///
    /// This is the single most important thing a model uses to decide whether
    /// to call a tool, so it must not be empty — [`Tools::register`] rejects a
    /// tool whose description is blank.
    ///
    /// The default implementation reads the rustdoc comment on
    /// [`Self::Arguments`], which `schemars` records in the generated schema.
    /// Override it to supply the description directly.
    fn description(&self) -> Cow<'static, str> {
        description_from_schema::<Self::Arguments>().unwrap_or_default()
    }

    /// Tool arguments type. Must implement [`schemars::JsonSchema`] and [`serde::de::DeserializeOwned`].
    /// Its rustdoc becomes the default tool description.
    type Arguments: Send + JsonSchema + DeserializeOwned;

    /// Raw return type from the tool implementation.
    type Res: IntoToolResult + Send;

    /// Executes the tool with the provided arguments.
    ///
    /// Returns a value that can be converted into a final [`ToolResult`].
    ///
    /// Tools that need mutable state should use interior mutability (e.g., `Mutex`).
    fn call(&self, arguments: Self::Arguments) -> impl Future<Output = Result<Self::Res>> + Send;
}

/// Utility to convert a serializable value to a pretty-printed JSON string.
///
/// # Example
/// ```rust,ignore
/// use aither::llm::tool::json;
/// use serde::Serialize;
/// #[derive(Serialize)]
/// struct Data {
///     name: String,
///     value: u32,
/// }
/// let data = Data {
///     name: "example".to_string(),
///     value: 42,
/// };
/// let json_str = json(&data);
/// println!("{}", json_str);
/// ```
///
/// # Errors
///
/// Returns an error if the value cannot be serialized to JSON, which happens
/// for types such as maps with non-string keys.
pub fn json<T: Serialize>(value: &T) -> Result<String> {
    let value = serde_json::to_value(value)?;

    Ok(value
        .as_str()
        .map_or_else(|| format!("{value:#}"), ToString::to_string))
}

trait ToolImpl: Send + Sync + Any {
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>;

    /// The cached definition. Borrowed, so registering a tool does not have to
    /// clone its argument schema.
    fn definition(&self) -> &ToolDefinition;

    /// Upcast to [`Any`] so [`Tools::get`] can recover the concrete tool.
    ///
    /// Casting the `Box<dyn ToolImpl>` itself would downcast the box rather
    /// than the tool inside it, and so never match.
    fn as_any(&self) -> &dyn Any;

    /// Mutable counterpart of [`Self::as_any`].
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Dynamic tool implementation for type-erased tools.
struct DynToolImpl<F>
where
    F: Fn(&str) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>> + Send + Sync,
{
    definition: ToolDefinition,
    handler: F,
}

impl<F> ToolImpl for DynToolImpl<F>
where
    F: Fn(&str) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>> + Send + Sync + 'static,
{
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        (self.handler)(args)
    }

    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Whether a schema describes a JSON object, and so can be sent to a provider
/// as tool arguments without being wrapped in [`ToolArgument`].
fn schema_is_object(value: &Value) -> bool {
    matches!(value.get("type").and_then(Value::as_str), Some("object"))
        || value.get("properties").is_some()
        || value.get("oneOf").is_some()
        || value.get("anyOf").is_some()
        || value.get("$defs").is_some()
}

fn is_object<T: JsonSchema>() -> bool {
    schema_is_object(&schema_for!(T).to_value())
}

/// Builds the argument schema for a tool, wrapping scalars so the root is
/// always an object as providers require.
fn arguments_schema<T: JsonSchema>() -> Schema {
    if is_object::<T>() {
        schema_for!(T)
    } else {
        schema_for!(ToolArgument<T>)
    }
}

/// Reads the `description` a `JsonSchema` derive records from a type's rustdoc.
fn description_from_schema<T: JsonSchema>() -> Option<Cow<'static, str>> {
    schema_for!(T)
        .to_value()
        .get("description")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(|text| Cow::Owned(text.to_string()))
}

/// A registered tool together with everything derived from its type.
///
/// The argument schema and the "are these arguments an object?" decision are
/// properties of `T::Arguments` alone, so they are computed once here rather
/// than rebuilt on every invocation.
struct RegisteredTool<T: Tool> {
    tool: T,
    definition: ToolDefinition,
    args_are_object: bool,
}

impl<T: Tool> RegisteredTool<T> {
    fn new(tool: T) -> Self {
        let definition = ToolDefinition::new(&tool);
        let args_are_object = is_object::<T::Arguments>();
        Self {
            tool,
            definition,
            args_are_object,
        }
    }
}

impl<T: Tool + 'static> ToolImpl for RegisteredTool<T> {
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        let result = if self.args_are_object {
            serde_json::from_str::<T::Arguments>(args)
        } else {
            serde_json::from_str::<ToolArgument<T::Arguments>>(args).map(|wrapper| wrapper.value)
        };

        let Ok(arguments) = result else {
            // Cold path: spelling the schema out for the model is worth the
            // allocation only when it has actually got the call wrong.
            let name = self.definition.name().to_string();
            let schema_str =
                serde_json::to_string_pretty(&self.definition.arguments_openai_schema())
                    .unwrap_or_else(|_| "{}".to_string());
            return Box::pin(async move {
                Err(anyhow::Error::msg(format!(
                    "Invalid arguments for tool '{name}'. Expected schema:\n{schema_str}"
                )))
            });
        };

        Box::pin(async move { Tool::call(&self.tool, arguments).await?.into_tool_result() })
    }

    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn as_any(&self) -> &dyn Any {
        &self.tool
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.tool
    }
}

/// A tool definition carried something that is not a JSON schema.
///
/// JSON Schema allows an object or a bare boolean; anything else — a string, an
/// array, a number — describes nothing a model could fill in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidSchema {
    /// The tool whose schema was rejected.
    name: Cow<'static, str>,
}

impl InvalidSchema {
    /// The name of the tool whose schema was rejected.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Display for InvalidSchema {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "tool '{}' has an argument schema that is neither an object nor a boolean",
            self.name
        )
    }
}

impl core::error::Error for InvalidSchema {}

/// Why a tool could not be added to a [`Tools`] registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterError {
    /// A tool with this name is already registered.
    ///
    /// Names address tools in a model's tool-call, so they must be unique.
    DuplicateName(Cow<'static, str>),

    /// The tool's description is empty.
    ///
    /// A description is what a model uses to decide whether to call a tool, so
    /// an empty one makes the tool unusable rather than merely undocumented.
    EmptyDescription(Cow<'static, str>),
}

impl Display for RegisterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateName(name) => {
                write!(f, "a tool named '{name}' is already registered")
            }
            Self::EmptyDescription(name) => write!(
                f,
                "tool '{name}' has an empty description; add a rustdoc comment to its \
                 Arguments type or implement Tool::description"
            ),
        }
    }
}

impl core::error::Error for RegisterError {}

/// Tool registry for managing and calling tools by name.
///
///
/// # Example
///
/// ```rust,ignore
/// use aither::llm::tool::Tools;
///
/// let mut tools = Tools::new();
/// // tools.register(Calculator);
/// let definitions = tools.definitions();
/// // let result = tools.call("calculator", r#"{"operation": "add", "a": 5, "b": 3}"#).await;
/// ```
pub struct Tools {
    tools: BTreeMap<Cow<'static, str>, Box<dyn ToolImpl>>,
}

impl Debug for Tools {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tools")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Tool definition including schema for language models.
///
/// Used to provide language models with information about available [`Tool`]s.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToolDefinition {
    /// Tool name.
    name: Cow<'static, str>,
    /// Tool description.
    description: Cow<'static, str>,
    /// JSON schema for tool arguments.
    arguments: Schema,
}

impl ToolDefinition {
    /// Creates a tool definition for a given tool type.
    ///
    /// The description comes from [`Tool::description`], which by default reads
    /// the rustdoc on the tool's `Arguments` type.
    #[must_use]
    pub fn new<T: Tool>(tool: &T) -> Self {
        Self {
            name: tool.name(),
            description: tool.description(),
            arguments: arguments_schema::<T::Arguments>(),
        }
    }

    /// Creates a tool definition from raw parts.
    ///
    /// This is useful for creating definitions from external sources like MCP servers.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSchema`] if the value is not a JSON schema — that is,
    /// anything other than an object or a boolean. The schema often comes from
    /// a remote server, so this is a rejection to report, not a bug to panic
    /// on.
    pub fn from_parts(
        name: Cow<'static, str>,
        description: Cow<'static, str>,
        schema: Value,
    ) -> core::result::Result<Self, InvalidSchema> {
        let arguments: Schema = schema
            .try_into()
            .map_err(|_| InvalidSchema { name: name.clone() })?;

        Ok(Self {
            name,
            description,
            arguments,
        })
    }

    /// Returns the tool's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tool's description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Return an OpenAI-compatible JSON schema for the tool's arguments.
    ///
    /// This schema would have an object type at the root, as required by `OpenAI`.
    #[must_use]
    pub fn arguments_openai_schema(&self) -> serde_json::Value {
        let mut inner = self.arguments.clone().to_value();
        clean_schema(&mut inner);

        inner
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ToolArgument<T> {
    value: T,
}

fn clean_schema(value: &mut Value) {
    // First pass: extract $defs for reference resolution
    let defs = extract_defs(value);

    // Second pass: resolve refs and clean
    resolve_and_clean(value, &defs);

    // Clean up root-level schema
    if let Value::Object(map) = value {
        // Remove root-level description - it's already used as tool description
        // Keeping it duplicates the description in the API request
        map.remove("description");

        // Ensure root has type: object (required by OpenAI function calling)
        if map.contains_key("properties") && !map.contains_key("type") {
            map.insert("type".to_string(), Value::String("object".to_string()));
        }
    }
}

/// Extracts `$defs` or `definitions` from the root schema.
fn extract_defs(value: &Value) -> serde_json::Map<String, Value> {
    if let Value::Object(map) = value
        && let Some(Value::Object(defs)) = map.get("$defs").or_else(|| map.get("definitions"))
    {
        return defs.clone();
    }
    serde_json::Map::new()
}

/// Resolves `$ref` and cleans the schema recursively.
#[allow(clippy::too_many_lines)]
fn resolve_and_clean(value: &mut Value, defs: &serde_json::Map<String, Value>) {
    resolve_and_clean_inner(value, defs, false);
}

/// Inner recursive function with flag to track if we're inside a properties object.
#[allow(clippy::too_many_lines)]
fn resolve_and_clean_inner(
    value: &mut Value,
    defs: &serde_json::Map<String, Value>,
    inside_properties: bool,
) {
    match value {
        Value::Object(map) => {
            // Handle $ref - inline the referenced definition, preserving sibling properties
            if let Some(Value::String(ref_path)) = map.remove("$ref")
                && let Some(Value::Object(resolved_map)) = resolve_ref(&ref_path, defs)
            {
                // Merge resolved definition with any existing properties (like description)
                // Resolved definition takes precedence for conflicts except description
                let existing_description = map.remove("description");
                for (k, v) in resolved_map {
                    map.entry(k).or_insert(v);
                }
                // Preserve the field-level description if it exists
                if let Some(desc) = existing_description {
                    map.insert("description".to_string(), desc);
                }
            }

            // Convert "const" to "enum" with single value (before filtering)
            if let Some(const_val) = map.remove("const") {
                map.insert("enum".to_string(), Value::Array(alloc::vec![const_val]));
            }

            // Flatten oneOf/anyOf variants (before filtering, since oneOf is not in allowed list)
            if let Some(Value::Array(variants)) =
                map.remove("oneOf").or_else(|| map.remove("anyOf"))
            {
                // Check if this is a simple string enum (variants have const/type but no properties)
                let is_simple_enum = variants.iter().all(|v| {
                    if let Value::Object(vm) = v {
                        (vm.contains_key("const") || vm.contains_key("enum"))
                            && !vm.contains_key("properties")
                    } else {
                        false
                    }
                });

                if is_simple_enum {
                    // Collect all const/enum values into a single enum array
                    let mut enum_values: alloc::vec::Vec<Value> = alloc::vec::Vec::new();
                    let mut variant_type: Option<String> = None;

                    for variant in &variants {
                        if let Value::Object(vm) = variant {
                            if let Some(const_val) = vm.get("const")
                                && !enum_values.contains(const_val)
                            {
                                enum_values.push(const_val.clone());
                            }
                            if let Some(Value::Array(arr)) = vm.get("enum") {
                                for val in arr {
                                    if !enum_values.contains(val) {
                                        enum_values.push(val.clone());
                                    }
                                }
                            }
                            if variant_type.is_none()
                                && let Some(Value::String(t)) = vm.get("type")
                            {
                                variant_type = Some(t.clone());
                            }
                        }
                    }

                    if !enum_values.is_empty() {
                        map.insert("enum".to_string(), Value::Array(enum_values));
                        if let Some(t) = variant_type {
                            map.insert("type".to_string(), Value::String(t));
                        }
                    }
                } else {
                    // Complex variants with properties - merge them
                    let mut all_properties = serde_json::Map::new();

                    for variant in variants {
                        if let Value::Object(variant_map) = variant
                            && let Some(Value::Object(props)) = variant_map.get("properties")
                        {
                            for (key, val) in props {
                                // Extract enum value - handle both "enum" and "const"
                                let new_values: Option<alloc::vec::Vec<Value>> =
                                    if let Value::Object(val_obj) = val {
                                        if let Some(Value::Array(arr)) = val_obj.get("enum") {
                                            Some(arr.clone())
                                        } else {
                                            val_obj
                                                .get("const")
                                                .map(|const_val| alloc::vec![const_val.clone()])
                                        }
                                    } else {
                                        None
                                    };

                                if all_properties.contains_key(key) {
                                    // Merge enum/const values into existing
                                    if let Some(values) = new_values
                                        && let Some(Value::Object(existing_obj)) =
                                            all_properties.get_mut(key)
                                        && let Some(Value::Array(existing_enum)) =
                                            existing_obj.get_mut("enum")
                                    {
                                        for e in values {
                                            if !existing_enum.contains(&e) {
                                                existing_enum.push(e);
                                            }
                                        }
                                    }
                                } else {
                                    // First time seeing this property - convert const to enum
                                    let mut val_clone = val.clone();
                                    if let Value::Object(obj) = &mut val_clone
                                        && let Some(const_val) = obj.remove("const")
                                    {
                                        obj.insert(
                                            "enum".to_string(),
                                            Value::Array(alloc::vec![const_val]),
                                        );
                                    }
                                    all_properties.insert(key.clone(), val_clone);
                                }
                            }
                        }
                    }

                    // Set type as object if we have properties
                    if !all_properties.is_empty() {
                        map.insert("type".to_string(), Value::String("object".to_string()));
                        map.insert("properties".to_string(), Value::Object(all_properties));
                    }
                }
            }

            // Only filter schema keywords, not property names inside "properties"
            // OpenAPI schema subset supported by most LLM providers
            if !inside_properties {
                let allowed = [
                    "type",
                    "description",
                    "properties",
                    "required",
                    "items",
                    "enum",
                    "nullable",
                ];
                map.retain(|k, _| allowed.contains(&k.as_str()));
            }

            // Simplify "type" arrays like ["string", "null"] to single type
            if let Some(Value::Array(types)) = map.get("type") {
                // Filter out "null" and take the first non-null type
                let non_null: Vec<&Value> = types
                    .iter()
                    .filter(|t| !matches!(t, Value::String(s) if s == "null"))
                    .collect();
                if non_null.len() == 1 {
                    map.insert("type".to_string(), non_null[0].clone());
                }
            }

            // Recursively clean all values
            for (key, v) in map.iter_mut() {
                // When entering "properties", its children are property definitions
                let child_inside_props = key == "properties";
                resolve_and_clean_inner(v, defs, child_inside_props);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                resolve_and_clean_inner(v, defs, false);
            }
        }
        _ => {}
    }
}

/// Resolves a `$ref` path like `#/$defs/FsOperation` to its definition.
fn resolve_ref(ref_path: &str, defs: &serde_json::Map<String, Value>) -> Option<Value> {
    // Handle common patterns: #/$defs/Name or #/definitions/Name
    let name = ref_path
        .strip_prefix("#/$defs/")
        .or_else(|| ref_path.strip_prefix("#/definitions/"))?;

    defs.get(name).cloned()
}

impl Default for Tools {
    fn default() -> Self {
        Self::new()
    }
}

impl Tools {
    /// Creates a new empty tools registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    /// Retrieves a tool by type.
    ///
    /// Returns `None` if the tool is not found.
    #[must_use]
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Tool + 'static,
    {
        self.tools
            .values()
            .find_map(|tool| tool.as_any().downcast_ref::<T>())
    }

    /// Retrieves a mutable reference to a tool by type.
    ///
    /// Returns `None` if the tool is not found.
    #[must_use]
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Tool + 'static,
    {
        self.tools
            .values_mut()
            .find_map(|tool| tool.as_any_mut().downcast_mut::<T>())
    }

    /// Returns definitions of all registered tools.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect()
    }

    /// Registers a new tool.
    ///
    /// The tool must implement [`Tool`] and be `'static`.
    ///
    /// # Errors
    ///
    /// Returns [`RegisterError::DuplicateName`] if a tool of that name is
    /// already registered, or [`RegisterError::EmptyDescription`] if the tool
    /// has no description — a model cannot use a tool it cannot read about, so
    /// this is rejected rather than silently passed on.
    pub fn register<T: Tool + 'static>(
        &mut self,
        tool: T,
    ) -> core::result::Result<(), RegisterError> {
        self.insert(Box::new(RegisteredTool::new(tool)))
    }

    /// Registers a dynamic tool with a pre-made definition and handler.
    ///
    /// This is useful for type-erased tools (e.g., child terminal tools for subagents)
    /// where the concrete type isn't known at compile time.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::register`].
    pub fn register_dyn<F>(
        &mut self,
        definition: ToolDefinition,
        handler: F,
    ) -> core::result::Result<(), RegisterError>
    where
        F: Fn(&str) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        self.insert(Box::new(DynToolImpl {
            definition,
            handler,
        }))
    }

    fn insert(&mut self, tool: Box<dyn ToolImpl>) -> core::result::Result<(), RegisterError> {
        let name = tool.definition().name.clone();
        if self.tools.contains_key(&name) {
            return Err(RegisterError::DuplicateName(name));
        }
        if tool.definition().description().trim().is_empty() {
            return Err(RegisterError::EmptyDescription(name));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Removes a tool from the registry.
    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
    }

    /// Calls a tool by name with JSON arguments.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found, arguments cannot be parsed,
    /// or tool execution fails.
    pub async fn call(&self, name: &str, args: &str) -> Result<ToolResult> {
        if let Some(tool) = self.tools.get(name) {
            tool.call(args).await
        } else {
            Err(anyhow::Error::msg(format!("Tool '{name}' not found")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString, vec};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    /// Performs basic mathematical operations.
    #[derive(JsonSchema, Deserialize, Debug, PartialEq)]
    struct CalculatorArgs {
        operation: String,
        a: f64,
        b: f64,
    }

    struct Calculator;

    impl Tool for Calculator {
        fn name(&self) -> Cow<'static, str> {
            "calculator".into()
        }
        type Arguments = CalculatorArgs;
        type Res = ToolResult;

        fn call(&self, args: Self::Arguments) -> impl Future<Output = Result<Self::Res>> + Send {
            core::future::ready(match args.operation.as_str() {
                "add" => Ok(ToolResult::text((args.a + args.b).to_string())),
                "subtract" => Ok(ToolResult::text((args.a - args.b).to_string())),
                "multiply" => Ok(ToolResult::text((args.a * args.b).to_string())),
                "divide" => {
                    if args.b == 0.0 {
                        Err(anyhow::Error::msg("Division by zero"))
                    } else {
                        Ok(ToolResult::text((args.a / args.b).to_string()))
                    }
                }
                _ => Err(anyhow::Error::msg(format!(
                    "Unknown operation: {}",
                    args.operation
                ))),
            })
        }
    }

    /// Greets a person by name.
    #[derive(JsonSchema, Deserialize)]
    struct GreetArgs {
        name: String,
    }

    struct Greeter;

    impl Tool for Greeter {
        fn name(&self) -> Cow<'static, str> {
            "greeter".into()
        }
        type Arguments = GreetArgs;
        type Res = ToolResult;

        fn call(&self, args: Self::Arguments) -> impl Future<Output = Result<Self::Res>> + Send {
            core::future::ready(Ok(ToolResult::text(format!("Hello, {}!", args.name))))
        }
    }

    #[test]
    fn from_parts_accepts_object_and_boolean_schemas() {
        // JSON Schema allows a bare boolean as well as an object.
        for schema in [
            serde_json::json!({"type": "object"}),
            serde_json::json!(true),
        ] {
            assert!(
                ToolDefinition::from_parts("t".into(), "does a thing".into(), schema.clone())
                    .is_ok(),
                "{schema} should be accepted"
            );
        }
    }

    #[test]
    fn from_parts_rejects_non_schema_values() {
        // A server that sends any of these describes nothing a model could
        // fill in. Rejecting must not panic: the value came off the wire.
        for schema in [
            serde_json::json!("a string"),
            serde_json::json!([1, 2, 3]),
            serde_json::json!(7),
            serde_json::json!(null),
        ] {
            let result = ToolDefinition::from_parts("weird".into(), "d".into(), schema.clone());
            let Err(err) = result else {
                panic!("{schema} should be rejected");
            };
            assert_eq!(err.name(), "weird");
        }
    }

    #[test]
    fn json_utility() {
        let value = serde_json::json!({
            "name": "test",
            "value": 42
        });

        let json_str = json(&value).expect("a JSON value always serializes");
        assert!(json_str.contains("\"name\": \"test\""));
        assert!(json_str.contains("\"value\": 42"));
    }

    #[test]
    fn tool_definition_creation() {
        let calculator = Calculator;
        let definition = ToolDefinition::new(&calculator);

        assert_eq!(definition.name, "calculator");
        assert_eq!(
            definition.description,
            "Performs basic mathematical operations."
        );
        // Schema should be present - just check it exists
        // The exact structure of schemars::Schema is implementation detail
    }

    #[test]
    fn tools_creation() {
        let tools = Tools::new();
        assert_eq!(tools.definitions().len(), 0);
    }

    #[test]
    fn tools_default() {
        let tools = Tools::default();
        assert_eq!(tools.definitions().len(), 0);
    }

    #[tokio::test]
    async fn tools_register_and_call() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");

        let definitions = tools.definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "calculator");

        let result = tools
            .call("calculator", r#"{"operation": "add", "a": 5, "b": 3}"#)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_text(), Some("8"));
    }

    #[tokio::test]
    async fn calculator_operations() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");

        // Test addition
        let result = tools
            .call("calculator", r#"{"operation": "add", "a": 10, "b": 5}"#)
            .await;
        assert_eq!(result.unwrap().as_text(), Some("15"));

        // Test subtraction
        let result = tools
            .call(
                "calculator",
                r#"{"operation": "subtract", "a": 10, "b": 3}"#,
            )
            .await;
        assert_eq!(result.unwrap().as_text(), Some("7"));

        // Test multiplication
        let result = tools
            .call("calculator", r#"{"operation": "multiply", "a": 4, "b": 3}"#)
            .await;
        assert_eq!(result.unwrap().as_text(), Some("12"));

        // Test division
        let result = tools
            .call("calculator", r#"{"operation": "divide", "a": 15, "b": 3}"#)
            .await;
        assert_eq!(result.unwrap().as_text(), Some("5"));
    }

    #[tokio::test]
    async fn calculator_division_by_zero() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");

        let result = tools
            .call("calculator", r#"{"operation": "divide", "a": 10, "b": 0}"#)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Division by zero"));
    }

    #[tokio::test]
    async fn calculator_unknown_operation() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");

        let result = tools
            .call("calculator", r#"{"operation": "modulo", "a": 10, "b": 3}"#)
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown operation")
        );
    }

    #[tokio::test]
    async fn multiple_tools() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");
        tools.register(Greeter).expect("greeter registers");

        let definitions = tools.definitions();
        assert_eq!(definitions.len(), 2);

        // Find calculator and greeter in definitions
        let calc_def = definitions.iter().find(|d| d.name == "calculator").unwrap();
        let greet_def = definitions.iter().find(|d| d.name == "greeter").unwrap();

        assert_eq!(
            calc_def.description,
            "Performs basic mathematical operations."
        );
        assert_eq!(greet_def.description, "Greets a person by name.");

        // Test both tools
        let calc_result = tools
            .call("calculator", r#"{"operation": "add", "a": 2, "b": 3}"#)
            .await;
        assert_eq!(calc_result.unwrap().as_text(), Some("5"));

        let greet_result = tools.call("greeter", r#"{"name": "Alice"}"#).await;
        assert_eq!(greet_result.unwrap().as_text(), Some("Hello, Alice!"));
    }

    #[derive(Debug, Serialize)]
    struct TableRow {
        name: &'static str,
        count: u32,
    }

    #[derive(Debug, Serialize)]
    struct NestedTableRow {
        user: TableRow,
        ok: bool,
    }

    #[derive(Debug)]
    struct ToolFailure(&'static str);

    impl core::fmt::Display for ToolFailure {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl core::error::Error for ToolFailure {}

    #[test]
    fn into_tool_result_string_is_plain_text() {
        assert_eq!(
            String::from("hello").into_tool_result().unwrap(),
            ToolResult::text("hello")
        );
    }

    #[test]
    fn into_tool_result_str_is_plain_text() {
        assert_eq!(
            "hello".into_tool_result().unwrap(),
            ToolResult::text("hello")
        );
    }

    #[test]
    fn into_tool_result_option_none_is_done() {
        let result = Option::<String>::None.into_tool_result().unwrap();
        assert_eq!(result, ToolResult::Done);
    }

    #[test]
    fn into_tool_result_option_some_delegates() {
        let result = Some("hello").into_tool_result().unwrap();
        assert_eq!(result, ToolResult::text("hello"));
    }

    #[test]
    fn into_tool_result_result_ok_string_is_text() {
        let result = core::result::Result::<String, ToolFailure>::Ok(String::from("hello"))
            .into_tool_result()
            .unwrap();
        assert_eq!(result, ToolResult::text("hello"));
    }

    #[test]
    fn into_tool_result_result_ok_object_is_tsv() {
        let result = core::result::Result::<TableRow, ToolFailure>::Ok(TableRow {
            name: "alpha",
            count: 3,
        })
        .into_tool_result()
        .unwrap();

        assert_eq!(result, ToolResult::tsv("count\tname\n3\talpha\n"));
    }

    #[test]
    fn into_tool_result_result_ok_array_of_objects_is_tsv() {
        let result = core::result::Result::<Vec<TableRow>, ToolFailure>::Ok(vec![
            TableRow {
                name: "alpha",
                count: 3,
            },
            TableRow {
                name: "beta",
                count: 5,
            },
        ])
        .into_tool_result()
        .unwrap();

        assert_eq!(result, ToolResult::tsv("count\tname\n3\talpha\n5\tbeta\n"));
    }

    #[test]
    fn into_tool_result_result_ok_scalar_is_json() {
        let result = core::result::Result::<bool, ToolFailure>::Ok(true)
            .into_tool_result()
            .unwrap();
        assert_eq!(result, ToolResult::json_value(Value::Bool(true)));
    }

    #[test]
    fn into_tool_result_result_err_is_typed_error() {
        let result = core::result::Result::<TableRow, ToolFailure>::Err(ToolFailure("boom"))
            .into_tool_result()
            .unwrap();
        assert_eq!(result, ToolResult::error("boom"));
        assert!(result.is_error());
        assert_eq!(result.error_message(), Some("boom"));
    }

    #[test]
    fn json_value_to_tsv_flattens_nested_objects() {
        let value = serde_json::to_value(NestedTableRow {
            user: TableRow {
                name: "alpha",
                count: 3,
            },
            ok: true,
        })
        .unwrap();

        assert_eq!(
            json_value_to_tsv(&value),
            Some("ok\tuser.count\tuser.name\ntrue\t3\talpha\n".to_string())
        );
    }

    #[tokio::test]
    async fn tool_not_found() {
        let tools = Tools::new();

        let result = tools.call("nonexistent", "{}").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Tool 'nonexistent' not found")
        );
    }

    #[tokio::test]
    async fn invalid_json() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");

        let result = tools.call("calculator", "invalid json").await;
        assert!(result.is_err());
    }

    #[test]
    fn tools_unregister() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");
        tools.register(Greeter).expect("greeter registers");

        assert_eq!(tools.definitions().len(), 2);

        tools.unregister("calculator");
        assert_eq!(tools.definitions().len(), 1);

        let remaining = &tools.definitions()[0];
        assert_eq!(remaining.name, "greeter");

        tools.unregister("greeter");
        assert_eq!(tools.definitions().len(), 0);
    }

    #[test]
    fn tools_debug() {
        let mut tools = Tools::new();
        tools.register(Calculator).expect("calculator registers");
        tools.register(Greeter).expect("greeter registers");

        let debug_str = format!("{tools:?}");
        assert!(debug_str.contains("Tools"));
        assert!(debug_str.contains("calculator"));
        assert!(debug_str.contains("greeter"));
    }

    #[test]
    fn tool_definition_debug() {
        let calculator = Calculator;
        let definition = ToolDefinition::new(&calculator);
        let debug_str = format!("{definition:?}");

        assert!(debug_str.contains("ToolDefinition"));
        assert!(debug_str.contains("calculator"));
        assert!(debug_str.contains("Performs basic mathematical operations"));
    }

    #[test]
    fn tool_definition_clone() {
        let calculator = Calculator;
        let original = ToolDefinition::new(&calculator);
        let cloned = original.clone();

        assert_eq!(original.name, cloned.name);
        assert_eq!(original.description, cloned.description);
    }

    #[test]
    fn schema_preserves_enum() {
        #[derive(JsonSchema, Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Status {
            Pending,
            InProgress,
            Completed,
        }

        #[allow(dead_code)]
        #[derive(JsonSchema, Deserialize)]
        struct Item {
            status: Status,
        }

        #[allow(dead_code)]
        #[derive(JsonSchema, Deserialize)]
        struct Args {
            items: Vec<Item>,
        }

        struct TestTool;

        impl Tool for TestTool {
            fn name(&self) -> Cow<'static, str> {
                "test".into()
            }
            type Arguments = Args;
            type Res = ToolResult;

            fn call(
                &self,
                _args: Self::Arguments,
            ) -> impl Future<Output = Result<Self::Res>> + Send {
                core::future::ready(Ok(ToolResult::text("ok")))
            }
        }

        let tool = TestTool;
        let def = ToolDefinition::new(&tool);
        let schema = def.arguments_openai_schema();

        // Check that status has enum values
        let schema_obj = schema.as_object().expect("schema should be object");
        let properties = schema_obj
            .get("properties")
            .expect("should have properties")
            .as_object()
            .unwrap();
        let items = properties
            .get("items")
            .expect("should have items")
            .as_object()
            .unwrap();
        let item_props = items
            .get("items")
            .expect("items should have items schema")
            .as_object()
            .unwrap();
        let item_properties = item_props
            .get("properties")
            .expect("item should have properties")
            .as_object()
            .unwrap();
        let status = item_properties
            .get("status")
            .expect("should have status")
            .as_object()
            .unwrap();

        // Status should have enum
        assert!(
            status.contains_key("enum"),
            "Status should have enum field. Full schema: {}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }

    #[test]
    fn schema_ref_resolution() {
        // Test that $ref schemas get resolved and enum values preserved
        let raw_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "status": {
                    "$ref": "#/$defs/Status"
                }
            },
            "$defs": {
                "Status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"]
                }
            }
        });

        let mut schema = raw_schema;
        clean_schema(&mut schema);

        let props = schema.get("properties").unwrap().as_object().unwrap();
        let status = props.get("status").unwrap().as_object().unwrap();

        assert!(
            status.contains_key("enum"),
            "Status should have enum after ref resolution. Got: {}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }

    #[test]
    fn schema_nested_ref_in_array() {
        // Test nested $ref inside array items (like TodoWriteArgs)
        let raw_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/TodoItem"
                    }
                }
            },
            "$defs": {
                "TodoItem": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "status": { "$ref": "#/$defs/TodoStatus" }
                    },
                    "required": ["content", "status"]
                },
                "TodoStatus": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"]
                }
            }
        });

        let mut schema = raw_schema;
        clean_schema(&mut schema);

        // Navigate to status
        let props = schema.get("properties").unwrap().as_object().unwrap();
        let todos = props.get("todos").unwrap().as_object().unwrap();
        let items = todos.get("items").unwrap().as_object().unwrap();
        let item_props = items.get("properties").unwrap().as_object().unwrap();
        let status = item_props.get("status").unwrap().as_object().unwrap();

        assert!(
            status.contains_key("enum"),
            "Nested status should have enum. Full schema: {}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }
}
