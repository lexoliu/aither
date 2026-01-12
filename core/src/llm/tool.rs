//! # LLM Tool Calling Framework
//!
//!
//! Type-safe tool calling system for Large Language Models. Enables LLMs to execute external
//! functions, access APIs, and interact with systems through well-defined interfaces.
//!
//! //! ## How LLM call external tools?
//!
//! TODO
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
//! use aither::llm::Tool;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
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
//!     const NAME: &str = "calculator";
//!     const DESCRIPTION: &str = "Performs basic math operations";
//!     type Arguments = MathArgs;
//!
//!     async fn call(&self, args: Self::Arguments) -> aither::Result<ToolOutput> {
//!         let result = match args.operation.as_str() {
//!             "add" => args.a + args.b,
//!             "subtract" => args.a - args.b,
//!             "multiply" => args.a * args.b,
//!             "divide" if args.b != 0.0 => args.a / args.b,
//!             "divide" => return Err(anyhow::Error::msg("Division by zero")),
//!             _ => return Err(anyhow::Error::msg("Unknown operation")),
//!         };
//!         Ok(ToolOutput::text(result.to_string()))
//!     }
//! }
//! ```
//!
//! ## Schema Design Best Practices
//!
//! ### 1. Use Clear Documentation Comments
//! Doc comments automatically become schema descriptions:
//!
//! ```rust
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
//! ```rust
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
//! ```rust
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
//! ```rust
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
use core::fmt::Debug;
use core::{future::Future, pin::Pin};
pub use mime::Mime;
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

/// Output from a tool execution.
///
/// Tools return either:
/// - `Done` - operation completed with no output (e.g., "file deleted")
/// - `Output` - operation produced content with a MIME type
///
/// # Example
///
/// ```rust
/// use aither::llm::tool::ToolOutput;
///
/// // Tool that produces text
/// fn search_tool() -> ToolOutput {
///     ToolOutput::text("Found 3 results...")
/// }
///
/// // Tool that completes without output
/// fn delete_tool() -> ToolOutput {
///     ToolOutput::Done
/// }
/// ```
#[derive(Debug, Clone)]
pub enum ToolOutput {
    /// Tool completed with no output to return.
    /// No file will be created, nothing to save.
    Done,

    /// Tool produced output with content and MIME type.
    Output {
        /// MIME type of the content (e.g., `text/plain`, `image/png`)
        mime: Mime,
        /// Raw content bytes
        content: Vec<u8>,
    },
}

impl ToolOutput {
    /// Creates a text output (UTF-8 string).
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        let s = s.into();
        Self::Output {
            mime: mime::TEXT_PLAIN_UTF_8,
            content: s.into_bytes(),
        }
    }

    /// Creates a JSON output from a serializable value.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn json<T: Serialize>(value: &T) -> Result<Self> {
        let bytes = serde_json::to_vec(value)?;
        Ok(Self::Output {
            mime: mime::APPLICATION_JSON,
            content: bytes,
        })
    }

    /// Creates an image output.
    #[must_use]
    pub fn image(data: Vec<u8>, media_type: &str) -> Self {
        Self::Output {
            mime: media_type.parse().unwrap_or(mime::APPLICATION_OCTET_STREAM),
            content: data,
        }
    }

    /// Creates a binary output.
    #[must_use]
    pub fn binary(data: Vec<u8>) -> Self {
        Self::Output {
            mime: mime::APPLICATION_OCTET_STREAM,
            content: data,
        }
    }

    /// Returns `true` if this is a `Done` variant.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Returns the content if this is an `Output` variant.
    #[must_use]
    pub fn content(&self) -> Option<&[u8]> {
        match self {
            Self::Done => None,
            Self::Output { content, .. } => Some(content),
        }
    }

    /// Returns the MIME type if this is an `Output` variant.
    #[must_use]
    pub fn mime(&self) -> Option<&Mime> {
        match self {
            Self::Done => None,
            Self::Output { mime, .. } => Some(mime),
        }
    }

    /// Converts the output to a string if it's text content.
    ///
    /// Returns `None` if:
    /// - This is a `Done` variant
    /// - The content is not valid UTF-8
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Done => None,
            Self::Output { content, .. } => core::str::from_utf8(content).ok(),
        }
    }
}

/// Tools that can be called by language models.
///
/// # Example
///
/// ```rust
/// use aither::llm::{Tool, ToolOutput};
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
///     const NAME: &str = "calculator";
///     const DESCRIPTION: &str = "Performs basic mathematical operations";
///     type Arguments = CalculatorArgs;
///
///     async fn call(&mut self, args: Self::Arguments) -> aither::Result<ToolOutput> {
///         match args.operation.as_str() {
///             "add" => Ok(ToolOutput::text((args.a + args.b).to_string())),
///             "subtract" => Ok(ToolOutput::text((args.a - args.b).to_string())),
///             "multiply" => Ok(ToolOutput::text((args.a * args.b).to_string())),
///             "divide" => {
///                 if args.b != 0.0 {
///                     Ok(ToolOutput::text((args.a / args.b).to_string()))
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

    /// Tool arguments type. Must implement [`schemars::JsonSchema`] and [`serde::de::DeserializeOwned`].
    /// Description is extracted from the rustdoc on this type via JsonSchema.
    type Arguments: Send + JsonSchema + DeserializeOwned;

    /// Executes the tool with the provided arguments.
    ///
    /// Returns a [`ToolOutput`] containing the tool's output or [`ToolOutput::Done`] for no output.
    ///
    /// Tools that need mutable state should use interior mutability (e.g., `Mutex`).
    fn call(&self, arguments: Self::Arguments) -> impl Future<Output = Result<ToolOutput>> + Send;
}

/// Utility to convert a serializable value to a pretty-printed JSON string.
///
/// # Example
/// ```rust
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
#[must_use]
pub fn json<T: Serialize>(value: &T) -> String {
    let value = serde_json::to_value(value).expect("Failed to convert value to JSON");

    value
        .as_str()
        .map_or_else(|| format!("{value:#}"), ToString::to_string)
}

trait ToolImpl: Send + Sync + Any {
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + '_>>;
    fn definition(&self) -> ToolDefinition;
}

/// Dynamic tool implementation for type-erased tools.
struct DynToolImpl<F>
where
    F: Fn(&str) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send>> + Send + Sync,
{
    definition: ToolDefinition,
    handler: F,
}

impl<F> ToolImpl for DynToolImpl<F>
where
    F: Fn(&str) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send>> + Send + Sync + 'static,
{
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + '_>> {
        (self.handler)(args)
    }

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }
}

fn is_object<T: JsonSchema>() -> bool {
    let schema = schema_for!(T);
    let value: Value = schema.to_value();

    matches!(value.get("type").and_then(Value::as_str), Some("object"))
        || value.get("properties").is_some()
        || value.get("oneOf").is_some()
        || value.get("anyOf").is_some()
        || value.get("$defs").is_some()
}

impl<T: Tool + 'static> ToolImpl for T {
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + '_>> {
        let is_object = is_object::<T::Arguments>();

        let result = if is_object {
            serde_json::from_str::<T::Arguments>(args)
        } else {
            serde_json::from_str::<ToolArgument<T::Arguments>>(args).map(|wrapper| wrapper.value)
        };

        let Ok(arguments) = result else {
            let name = self.name();
            let mut schema = schema_for!(T::Arguments).to_value();
            clean_schema(&mut schema);
            let schema_str =
                serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".to_string());
            return Box::pin(async move {
                Err(anyhow::Error::msg(format!(
                    "Invalid arguments for tool '{name}'. Expected schema:\n{schema_str}"
                )))
            });
        };

        Box::pin(async move { Tool::call(self, arguments).await })
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(self)
    }
}

/// Tool registry for managing and calling tools by name.
///
///
/// # Example
///
/// ```rust
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
    /// The description is extracted from the Arguments struct's rustdoc comment
    /// (via schemars::JsonSchema). Falls back to `Tool::description()` if no
    /// rustdoc is present on the Args struct.
    #[must_use]
    pub fn new<T: Tool>(tool: &T) -> Self {
        let mut arguments = schema_for!(T::Arguments);

        if !is_object::<T::Arguments>() {
            arguments = schema_for!(ToolArgument<T::Arguments>);
        }

        // Extract description from schema (rustdoc on Args struct)
        let description = arguments
            .clone()
            .to_value()
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| Cow::Owned(s.to_string()))
            .unwrap_or_default();

        Self {
            name: tool.name(),
            description,
            arguments,
        }
    }

    /// Creates a tool definition from raw parts.
    ///
    /// This is useful for creating definitions from external sources like MCP servers.
    ///
    /// # Panics
    ///
    /// Panics if the schema value is not a valid JSON schema (must be object or bool).
    #[must_use]
    pub fn from_parts(
        name: Cow<'static, str>,
        description: Cow<'static, str>,
        schema: Value,
    ) -> Self {
        let arguments: Schema = schema
            .try_into()
            .expect("Schema must be a JSON object or boolean");

        Self {
            name,
            description,
            arguments,
        }
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
    /// This schema would have an object type at the root, as required by OpenAI.
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
    if let Value::Object(map) = value {
        if let Some(Value::Object(defs)) = map.get("$defs").or_else(|| map.get("definitions")) {
            return defs.clone();
        }
    }
    serde_json::Map::new()
}

/// Resolves `$ref` and cleans the schema recursively.
fn resolve_and_clean(value: &mut Value, defs: &serde_json::Map<String, Value>) {
    resolve_and_clean_inner(value, defs, false);
}

/// Inner recursive function with flag to track if we're inside a properties object.
fn resolve_and_clean_inner(
    value: &mut Value,
    defs: &serde_json::Map<String, Value>,
    inside_properties: bool,
) {
    match value {
        Value::Object(map) => {
            // Handle $ref - inline the referenced definition, preserving sibling properties
            if let Some(Value::String(ref_path)) = map.remove("$ref") {
                if let Some(Value::Object(resolved_map)) = resolve_ref(&ref_path, defs) {
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
                            if let Some(const_val) = vm.get("const") {
                                if !enum_values.contains(const_val) {
                                    enum_values.push(const_val.clone());
                                }
                            }
                            if let Some(Value::Array(arr)) = vm.get("enum") {
                                for val in arr {
                                    if !enum_values.contains(val) {
                                        enum_values.push(val.clone());
                                    }
                                }
                            }
                            if variant_type.is_none() {
                                if let Some(Value::String(t)) = vm.get("type") {
                                    variant_type = Some(t.clone());
                                }
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
                        if let Value::Object(variant_map) = variant {
                            if let Some(Value::Object(props)) = variant_map.get("properties") {
                                for (key, val) in props {
                                    // Extract enum value - handle both "enum" and "const"
                                    let new_values: Option<alloc::vec::Vec<Value>> =
                                        if let Value::Object(val_obj) = val {
                                            if let Some(Value::Array(arr)) = val_obj.get("enum") {
                                                Some(arr.clone())
                                            } else if let Some(const_val) = val_obj.get("const") {
                                                Some(alloc::vec![const_val.clone()])
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        };

                                    if all_properties.contains_key(key) {
                                        // Merge enum/const values into existing
                                        if let Some(values) = new_values {
                                            if let Some(Value::Object(existing_obj)) =
                                                all_properties.get_mut(key)
                                            {
                                                if let Some(Value::Array(existing_enum)) =
                                                    existing_obj.get_mut("enum")
                                                {
                                                    for e in values {
                                                        if !existing_enum.contains(&e) {
                                                            existing_enum.push(e);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        // First time seeing this property - convert const to enum
                                        let mut val_clone = val.clone();
                                        if let Value::Object(obj) = &mut val_clone {
                                            if let Some(const_val) = obj.remove("const") {
                                                obj.insert(
                                                    "enum".to_string(),
                                                    Value::Array(alloc::vec![const_val]),
                                                );
                                            }
                                        }
                                        all_properties.insert(key.clone(), val_clone);
                                    }
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
            .find_map(|tool| (tool as &dyn Any).downcast_ref::<T>())
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
            .find_map(|tool| (tool as &mut dyn Any).downcast_mut::<T>())
    }

    /// Returns definitions of all registered tools.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    /// Registers a new tool. Replaces existing tool with same name.
    ///
    /// The tool must implement [`Tool`] and be `'static`.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name();
        // Check if conflict exists
        assert!(
            !self.tools.contains_key(&name),
            "Tool with name '{}' is already registered",
            name
        );

        self.tools.insert(name, Box::new(tool) as Box<dyn ToolImpl>);
    }

    /// Registers a dynamic tool with a pre-made definition and handler.
    ///
    /// This is useful for type-erased tools (e.g., child bash tools for subagents)
    /// where the concrete type isn't known at compile time.
    pub fn register_dyn<F>(&mut self, definition: ToolDefinition, handler: F)
    where
        F: Fn(&str) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let name = definition.name.clone();
        assert!(
            !self.tools.contains_key(&name),
            "Tool with name '{}' is already registered",
            name
        );

        self.tools.insert(
            name,
            Box::new(DynToolImpl {
                definition,
                handler,
            }),
        );
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
    pub async fn call(&self, name: &str, args: &str) -> Result<ToolOutput> {
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
    use alloc::{format, string::ToString};
    use schemars::JsonSchema;
    use serde::Deserialize;

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

        async fn call(&self, args: Self::Arguments) -> Result<ToolOutput> {
            match args.operation.as_str() {
                "add" => Ok(ToolOutput::text((args.a + args.b).to_string())),
                "subtract" => Ok(ToolOutput::text((args.a - args.b).to_string())),
                "multiply" => Ok(ToolOutput::text((args.a * args.b).to_string())),
                "divide" => {
                    if args.b == 0.0 {
                        Err(anyhow::Error::msg("Division by zero"))
                    } else {
                        Ok(ToolOutput::text((args.a / args.b).to_string()))
                    }
                }
                _ => Err(anyhow::Error::msg(format!(
                    "Unknown operation: {}",
                    args.operation
                ))),
            }
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

        async fn call(&self, args: Self::Arguments) -> Result<ToolOutput> {
            Ok(ToolOutput::text(format!("Hello, {}!", args.name)))
        }
    }

    #[test]
    fn json_utility() {
        let value = serde_json::json!({
            "name": "test",
            "value": 42
        });

        let json_str = json(&value);
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
            "Performs basic mathematical operations"
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
        tools.register(Calculator);

        let definitions = tools.definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "calculator");

        let result = tools
            .call("calculator", r#"{"operation": "add", "a": 5, "b": 3}"#)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), Some("8"));
    }

    #[tokio::test]
    async fn calculator_operations() {
        let mut tools = Tools::new();
        tools.register(Calculator);

        // Test addition
        let result = tools
            .call("calculator", r#"{"operation": "add", "a": 10, "b": 5}"#)
            .await;
        assert_eq!(result.unwrap().as_str(), Some("15"));

        // Test subtraction
        let result = tools
            .call(
                "calculator",
                r#"{"operation": "subtract", "a": 10, "b": 3}"#,
            )
            .await;
        assert_eq!(result.unwrap().as_str(), Some("7"));

        // Test multiplication
        let result = tools
            .call("calculator", r#"{"operation": "multiply", "a": 4, "b": 3}"#)
            .await;
        assert_eq!(result.unwrap().as_str(), Some("12"));

        // Test division
        let result = tools
            .call("calculator", r#"{"operation": "divide", "a": 15, "b": 3}"#)
            .await;
        assert_eq!(result.unwrap().as_str(), Some("5"));
    }

    #[tokio::test]
    async fn calculator_division_by_zero() {
        let mut tools = Tools::new();
        tools.register(Calculator);

        let result = tools
            .call("calculator", r#"{"operation": "divide", "a": 10, "b": 0}"#)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Division by zero"));
    }

    #[tokio::test]
    async fn calculator_unknown_operation() {
        let mut tools = Tools::new();
        tools.register(Calculator);

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
        tools.register(Calculator);
        tools.register(Greeter);

        let definitions = tools.definitions();
        assert_eq!(definitions.len(), 2);

        // Find calculator and greeter in definitions
        let calc_def = definitions.iter().find(|d| d.name == "calculator").unwrap();
        let greet_def = definitions.iter().find(|d| d.name == "greeter").unwrap();

        assert_eq!(
            calc_def.description,
            "Performs basic mathematical operations"
        );
        assert_eq!(greet_def.description, "Greets a person by name");

        // Test both tools
        let calc_result = tools
            .call("calculator", r#"{"operation": "add", "a": 2, "b": 3}"#)
            .await;
        assert_eq!(calc_result.unwrap().as_str(), Some("5"));

        let greet_result = tools.call("greeter", r#"{"name": "Alice"}"#).await;
        assert_eq!(greet_result.unwrap().as_str(), Some("Hello, Alice!"));
    }

    #[tokio::test]
    async fn tool_not_found() {
        let mut tools = Tools::new();

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
        tools.register(Calculator);

        let result = tools.call("calculator", "invalid json").await;
        assert!(result.is_err());
    }

    #[test]
    fn tools_unregister() {
        let mut tools = Tools::new();
        tools.register(Calculator);
        tools.register(Greeter);

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
        tools.register(Calculator);
        tools.register(Greeter);

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

        #[derive(JsonSchema, Deserialize)]
        struct Item {
            status: Status,
        }

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

            async fn call(&self, _args: Self::Arguments) -> Result<ToolOutput> {
                Ok(ToolOutput::text("ok"))
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

        let mut schema = raw_schema.clone();
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

        let mut schema = raw_schema.clone();
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
