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
//! - [`ToolDefinition`] - Metadata and schema for LLM consumption
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
//!     async fn call(&mut self, args: Self::Arguments) -> aither::Result {
//!         let result = match args.operation.as_str() {
//!             "add" => args.a + args.b,
//!             "subtract" => args.a - args.b,
//!             "multiply" => args.a * args.b,
//!             "divide" if args.b != 0.0 => args.a / args.b,
//!             "divide" => return Err(anyhow::Error::msg("Division by zero")),
//!             _ => return Err(anyhow::Error::msg("Unknown operation")),
//!         };
//!         Ok(result.to_string())
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

use crate::Result;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{boxed::Box, collections::BTreeMap};
use core::fmt::Debug;
use core::{future::Future, pin::Pin};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

/// Tools that can be called by language models.
///
/// # Example
///
/// ```rust
/// use aither::llm::Tool;
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
///     async fn call(&mut self, args: Self::Arguments) -> aither::Result {
///         match args.operation.as_str() {
///             "add" => Ok((args.a + args.b).to_string()),
///             "subtract" => Ok((args.a - args.b).to_string()),
///             "multiply" => Ok((args.a * args.b).to_string()),
///             "divide" => {
///                 if args.b != 0.0 {
///                     Ok((args.a / args.b).to_string())
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
    /// Tool description for the language model.
    fn description(&self) -> Cow<'static, str>;

    /// Tool arguments type. Must implement [`schemars::JsonSchema`] and [`serde::de::DeserializeOwned`].
    type Arguments: JsonSchema + DeserializeOwned;

    /// Executes the tool with the provided arguments.
    ///
    /// Returns a [`crate::Result`] containing the tool's output.
    fn call(&mut self, arguments: Self::Arguments) -> impl Future<Output = Result> + Send;
}

/// Serializes a value to JSON string.
///
/// Convenience function for tools that need to return JSON responses.
/// Uses [`serde_json::to_string_pretty`] internally.
///
/// # Panics
///
/// Panics if the value cannot be serialized to JSON.
pub fn json<T: Serialize>(value: &T) -> String {
    let value = serde_json::to_value(value).expect("Failed to convert value to JSON");

    value
        .as_str()
        .map_or_else(|| format!("{value:#}"), ToString::to_string)
}

trait ToolImpl: Send + Sync {
    fn call(&mut self, args: String) -> Pin<Box<dyn Future<Output = Result> + Send + '_>>;
    fn definition(&self) -> ToolDefinition;
}

impl<T: Tool> ToolImpl for T {
    fn call(&mut self, args: String) -> Pin<Box<dyn Future<Output = Result> + Send + '_>> {
        Box::pin(async move {
            let arguments: T::Arguments = serde_json::from_str(&args)?;
            self.call(arguments).await
        })
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name(),
            description: self.description(),
            arguments: schema_for!(T::Arguments),
        }
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
    tools: BTreeMap<String, Box<dyn ToolImpl>>,
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
    #[must_use]
    pub fn new<T: Tool>(tool: &T) -> Self {
        Self {
            name: tool.name(),
            description: tool.description(),
            arguments: schema_for!(T::Arguments),
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

    /// Returns the JSON schema for the tool's arguments.
    #[must_use]
    pub const fn arguments_schema(&self) -> &Schema {
        &self.arguments
    }
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

    /// Returns definitions of all registered tools.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    /// Registers a new tool. Replaces existing tool with same name.
    ///
    /// The tool must implement [`Tool`] and be `'static`.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools
            .insert(tool.name().to_string(), Box::new(tool) as Box<dyn ToolImpl>);
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
    pub async fn call(&mut self, name: &str, args: String) -> Result {
        if let Some(tool) = self.tools.get_mut(name) {
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
        fn description(&self) -> Cow<'static, str> {
            "Performs basic mathematical operations".into()
        }
        type Arguments = CalculatorArgs;

        async fn call(&mut self, args: Self::Arguments) -> Result {
            match args.operation.as_str() {
                "add" => Ok((args.a + args.b).to_string()),
                "subtract" => Ok((args.a - args.b).to_string()),
                "multiply" => Ok((args.a * args.b).to_string()),
                "divide" => {
                    if args.b == 0.0 {
                        Err(anyhow::Error::msg("Division by zero"))
                    } else {
                        Ok((args.a / args.b).to_string())
                    }
                }
                _ => Err(anyhow::Error::msg(format!(
                    "Unknown operation: {}",
                    args.operation
                ))),
            }
        }
    }

    #[derive(JsonSchema, Deserialize)]
    struct GreetArgs {
        name: String,
    }

    struct Greeter;

    impl Tool for Greeter {
        fn name(&self) -> Cow<'static, str> {
            "greeter".into()
        }
        fn description(&self) -> Cow<'static, str> {
            "Greets a person by name".into()
        }
        type Arguments = GreetArgs;

        async fn call(&mut self, args: Self::Arguments) -> Result {
            Ok(format!("Hello, {}!", args.name))
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
            .call(
                "calculator",
                r#"{"operation": "add", "a": 5, "b": 3}"#.to_string(),
            )
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "8");
    }

    #[tokio::test]
    async fn calculator_operations() {
        let mut tools = Tools::new();
        tools.register(Calculator);

        // Test addition
        let result = tools
            .call(
                "calculator",
                r#"{"operation": "add", "a": 10, "b": 5}"#.to_string(),
            )
            .await;
        assert_eq!(result.unwrap(), "15");

        // Test subtraction
        let result = tools
            .call(
                "calculator",
                r#"{"operation": "subtract", "a": 10, "b": 3}"#.to_string(),
            )
            .await;
        assert_eq!(result.unwrap(), "7");

        // Test multiplication
        let result = tools
            .call(
                "calculator",
                r#"{"operation": "multiply", "a": 4, "b": 3}"#.to_string(),
            )
            .await;
        assert_eq!(result.unwrap(), "12");

        // Test division
        let result = tools
            .call(
                "calculator",
                r#"{"operation": "divide", "a": 15, "b": 3}"#.to_string(),
            )
            .await;
        assert_eq!(result.unwrap(), "5");
    }

    #[tokio::test]
    async fn calculator_division_by_zero() {
        let mut tools = Tools::new();
        tools.register(Calculator);

        let result = tools
            .call(
                "calculator",
                r#"{"operation": "divide", "a": 10, "b": 0}"#.to_string(),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Division by zero"));
    }

    #[tokio::test]
    async fn calculator_unknown_operation() {
        let mut tools = Tools::new();
        tools.register(Calculator);

        let result = tools
            .call(
                "calculator",
                r#"{"operation": "modulo", "a": 10, "b": 3}"#.to_string(),
            )
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
            .call(
                "calculator",
                r#"{"operation": "add", "a": 2, "b": 3}"#.to_string(),
            )
            .await;
        assert_eq!(calc_result.unwrap(), "5");

        let greet_result = tools
            .call("greeter", r#"{"name": "Alice"}"#.to_string())
            .await;
        assert_eq!(greet_result.unwrap(), "Hello, Alice!");
    }

    #[tokio::test]
    async fn tool_not_found() {
        let mut tools = Tools::new();

        let result = tools.call("nonexistent", "{}".to_string()).await;
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

        let result = tools.call("calculator", "invalid json".to_string()).await;
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
}
