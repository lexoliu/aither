//! Tool-to-command conversion using JSON schema.
//!
//! Wraps any `Tool` as an IPC command callable from the sandbox.
//! Uses the existing JSON schema to generate CLI parsing - no clap needed.
//!
//! # Architecture
//!
//! Tools are registered with [`ToolRegistryBuilder`] and stored in a
//! [`ToolRegistry`]. [`ToolCallCommand`] implements `IpcCommand` and
//! dispatches to the appropriate handler based on the tool name.
//!
//! # Example
//!
//! ```rust,ignore
//! use aither_sandbox::command::{ToolRegistryBuilder, register_tool_command};
//! use aither_sandbox::builtin::builtin_router;
//!
//! // Configure tools
//! let mut registry = ToolRegistryBuilder::new();
//! registry.configure_tool(websearch_tool);
//! registry.configure_tool(webfetch_tool);
//! let registry = std::sync::Arc::new(registry.build("./outputs"));
//!
//! // Register tool commands with router
//! let router = builtin_router();
//! let router = register_tool_command(router, registry.clone(), "websearch");
//! let router = register_tool_command(router, registry.clone(), "webfetch");
//! ```

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use aither_core::llm::Tool;
use leash::IpcCommand;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Tool Registry
// ============================================================================

/// Type-erased tool handler function.
type ToolHandlerFn =
    Box<dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Tool entry in the registry.
struct ToolEntry {
    handler: ToolHandlerFn,
    help: String,
    positional_args: Vec<String>,
    stdin_arg: Option<String>,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("help", &self.help)
            .field("positional_args", &self.positional_args)
            .field("stdin_arg", &self.stdin_arg)
            .finish_non_exhaustive()
    }
}

use crate::output::{INLINE_OUTPUT_LIMIT, generate_word_filename};

/// Handles large outputs by saving to file in the sandbox output directory.
///
/// Returns the output as-is if small enough, or saves to file and returns
/// a reference message if the output exceeds the limit.
async fn handle_large_output(output: String, output_dir: &PathBuf) -> String {
    if output.len() <= INLINE_OUTPUT_LIMIT {
        return output;
    }

    // Generate four-random-words filename (consistent with rest of codebase)
    let filename = format!("{}.txt", generate_word_filename());
    let filepath = output_dir.join(&filename);

    let line_count = output.lines().count();

    async_fs::write(&filepath, &output)
        .await
        .unwrap_or_else(|e| panic!("failed to write output to {}: {}", filepath.display(), e));

    format!(
        "Output saved to outputs/{} ({} lines, {} bytes)",
        filename,
        line_count,
        output.len()
    )
}

// ============================================================================
// Bash Tool Factory (for creating child bash tools for subagents)
// ============================================================================

use aither_core::llm::tool::ToolDefinition;

/// A type-erased handler function for bash tools.
pub type DynToolHandler =
    Arc<dyn Fn(&str) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// A type-erased bash tool that can be registered with AgentTools.
pub struct DynBashTool {
    /// Tool definition (name, description, schema).
    pub definition: ToolDefinition,
    /// Handler that takes JSON args and returns result.
    pub handler: DynToolHandler,
}

impl std::fmt::Debug for DynBashTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynBashTool")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

/// Builder for tool registries.
#[derive(Default)]
pub struct ToolRegistryBuilder {
    entries: HashMap<String, ToolEntry>,
}

impl ToolRegistryBuilder {
    /// Creates an empty registry builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Configures a tool to be callable from the sandbox.
    ///
    /// The positional arguments are automatically detected from the schema -
    /// all required fields become positional args in order.
    /// For example, if a tool has `query: String` as its first required field,
    /// then `websearch "rust"` → `websearch --query "rust"`.
    pub fn configure_tool<T>(&mut self, tool: T)
    where
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let schema = serde_json::to_value(schemars::schema_for!(T::Arguments))
            .expect("failed to serialize tool schema");

        // Auto-detect positional args from schema: all required fields
        let positional_args = detect_positional_args(&schema);
        // Auto-detect stdin_arg from schema: optional "input" field
        let stdin_arg = detect_stdin_arg(&schema);

        let tool = Arc::new(tool);
        let tool_name = tool.name().to_string();
        let help_text = schema_to_help(&schema);

        let handler: ToolHandlerFn = Box::new(move |args: Vec<String>| {
            let tool = tool.clone();
            let schema = schema.clone();
            Box::pin(async move {
                let json_args = cli_to_json(&schema, &args)
                    .unwrap_or_else(|e| panic!("failed to parse CLI args: {e}"));
                let parsed = serde_json::from_value(json_args)
                    .unwrap_or_else(|e| panic!("failed to parse tool arguments: {e}"));
                let output = tool
                    .call(parsed)
                    .await
                    .unwrap_or_else(|e| panic!("tool execution failed: {e}"));
                output.as_str().unwrap_or("").to_string()
            })
        });

        self.entries.insert(
            tool_name,
            ToolEntry {
                handler,
                help: help_text,
                positional_args,
                stdin_arg,
            },
        );
    }

    /// Registers a raw handler function as an IPC command.
    ///
    /// This is useful for dynamic tools like MCP tools that are discovered at runtime.
    pub fn configure_raw_handler<F>(
        &mut self,
        name: impl Into<String>,
        help: impl Into<String>,
        positional_args: Vec<String>,
        handler: F,
    ) where
        F: Fn(Vec<String>) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync + 'static,
    {
        self.entries.insert(
            name.into(),
            ToolEntry {
                handler: Box::new(handler),
                help: help.into(),
                positional_args,
                stdin_arg: None,
            },
        );
    }

    /// Builds an immutable registry with a concrete output directory.
    #[must_use]
    pub fn build(self, output_dir: impl Into<PathBuf>) -> ToolRegistry {
        ToolRegistry {
            entries: self.entries,
            output_dir: output_dir.into(),
        }
    }
}

/// Immutable registry of tool handlers.
#[derive(Debug)]
pub struct ToolRegistry {
    entries: HashMap<String, ToolEntry>,
    output_dir: PathBuf,
}

/// Detects positional arguments from a JSON schema.
///
/// Returns all required field names in order, which will be used for
/// positional argument conversion in wrapper scripts.
fn detect_positional_args(schema: &Value) -> Vec<String> {
    // Get the required array
    let required = schema.get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    required
}

/// Detects the stdin argument from a JSON schema.
///
/// Returns "input" if the schema has an "input" property that is not required.
/// This allows stdin piping: `cat file | command "prompt"` passes stdin as --input.
fn detect_stdin_arg(schema: &Value) -> Option<String> {
    // Check if "input" property exists
    let properties = schema.get("properties")?.as_object()?;
    if !properties.contains_key("input") {
        return None;
    }

    // Check that "input" is NOT required (has a default)
    let required = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    if required.contains(&"input") {
        return None;
    }

    Some("input".to_string())
}

impl ToolRegistry {
    /// Queries a tool handler by name.
    ///
    /// Large outputs (> 4000 chars) are automatically saved to file.
    pub async fn query_tool_handler(&self, tool_name: &str, args: &[String]) -> String {
        let entry = self.entries.get(tool_name);
        match entry {
            Some(entry) => {
                let output = (entry.handler)(args.to_vec()).await;
                handle_large_output(output, &self.output_dir).await
            }
            None => format!(
                "Error: unknown command '{}'. Run 'help' to see available commands.",
                tool_name
            ),
        }
    }

    /// Returns whether a tool is registered.
    #[must_use]
    pub fn is_tool_configured(&self, tool_name: &str) -> bool {
        self.entries.contains_key(tool_name)
    }

    /// Returns the help text for a tool.
    #[must_use]
    pub fn tool_help(&self, tool_name: &str) -> Option<String> {
        self.entries.get(tool_name).map(|e| e.help.clone())
    }

    /// Returns the positional arguments for a tool.
    #[must_use]
    pub fn tool_positional_args(&self, tool_name: &str) -> Vec<String> {
        self.entries
            .get(tool_name)
            .map(|e| e.positional_args.clone())
            .unwrap_or_default()
    }

    /// Returns the stdin argument for a tool.
    #[must_use]
    pub fn tool_stdin_arg(&self, tool_name: &str) -> Option<String> {
        self.entries
            .get(tool_name)
            .and_then(|e| e.stdin_arg.clone())
    }

    /// Returns list of registered tool names.
    #[must_use]
    pub fn registered_tool_names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }
}

// ============================================================================
// IPC Command for Tools
// ============================================================================

/// IPC command for invoking tools from the sandbox.
///
/// The tool_name comes from the IPC method name, and args are flattened
/// from the key-value pairs sent by leash-ipc.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallCommand {
    /// Shared tool registry.
    #[serde(skip)]
    pub registry: Arc<ToolRegistry>,
    /// Name of the tool to invoke (from IPC method name, not serialized).
    #[serde(skip)]
    pub tool_name: String,
    /// Tool arguments as key-value pairs (flattened from IPC params).
    #[serde(flatten)]
    pub args: std::collections::HashMap<String, Value>,
}

impl ToolCallCommand {
    /// Creates a new tool call command.
    #[must_use]
    pub fn new(tool_name: impl Into<String>, registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            tool_name: tool_name.into(),
            args: std::collections::HashMap::new(),
        }
    }

    /// Convert args HashMap to CLI-style Vec<String> for the handler.
    ///
    /// If there's an "args" key with an array value, use those as positional args.
    /// Otherwise, convert key-value pairs to `--key value` format.
    fn args_to_cli(&self) -> Vec<String> {
        // Check for positional args array (from `todo add "Task 1"` style commands)
        if let Some(Value::Array(arr)) = self.args.get("args") {
            return arr
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Number(n) => Some(n.to_string()),
                    Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                })
                .collect();
        }

        // Otherwise, convert to --key value format
        let mut cli_args = Vec::new();
        for (key, value) in &self.args {
            cli_args.push(format!("--{key}"));
            match value {
                Value::String(s) => cli_args.push(s.clone()),
                Value::Number(n) => cli_args.push(n.to_string()),
                Value::Bool(b) => cli_args.push(b.to_string()),
                _ => cli_args.push(value.to_string()),
            }
        }
        cli_args
    }
}

impl IpcCommand for ToolCallCommand {
    type Response = Value;

    fn name(&self) -> String {
        self.tool_name.clone()
    }

    fn positional_args(&self) -> Cow<'static, [Cow<'static, str>]> {
        let args = self.registry.tool_positional_args(&self.tool_name);
        if args.is_empty() {
            Cow::Borrowed(&[])
        } else {
            Cow::Owned(args.into_iter().map(Cow::Owned).collect())
        }
    }

    fn stdin_arg(&self) -> Option<Cow<'static, str>> {
        self.registry.tool_stdin_arg(&self.tool_name).map(Cow::Owned)
    }

    fn set_method_name(&mut self, name: &str) {
        self.tool_name = name.to_string();
    }

    fn apply_args(&mut self, params: &[u8]) -> Result<(), leash::rmp_serde::decode::Error> {
        // Params are a flattened HashMap that maps directly to args.
        // tool_name is set via set_method_name and preserved here.
        self.args = leash::rmp_serde::from_slice(params)?;
        Ok(())
    }

    async fn handle(&mut self) -> Value {
        let cli_args = self.args_to_cli();

        // Handle help flags
        if has_help_flag(&cli_args) {
            let help = self
                .registry
                .tool_help(&self.tool_name)
                .unwrap_or_else(|| format!("No help available for '{}'", self.tool_name));
            return Value::String(help);
        }

        let result = self
            .registry
            .query_tool_handler(&self.tool_name, &cli_args)
            .await;

        // Parse the result as JSON to avoid double-serialization.
        // Tool outputs are JSON strings, so we need to parse them to Value.
        serde_json::from_str(&result).unwrap_or_else(|_| Value::String(result))
    }
}

/// Registers a tool command with the IPC router.
///
/// The tool must have been previously configured in the registry.
///
/// # Example
///
/// ```rust,ignore
/// use aither_sandbox::command::register_tool_command;
/// use leash::IpcRouter;
///
/// let router = IpcRouter::new();
/// let registry = std::sync::Arc::new(ToolRegistryBuilder::new().build("./outputs"));
/// let router = register_tool_command(router, registry, "websearch");
/// ```
pub fn register_tool_command(
    router: leash::IpcRouter,
    registry: Arc<ToolRegistry>,
    tool_name: &str,
) -> leash::IpcRouter {
    router.register(ToolCallCommand::new(tool_name, registry))
}

// ============================================================================
// ToolCommand wrapper (for direct use without IPC)
// ============================================================================

/// Wraps a Tool as an IPC command using schema-driven CLI parsing.
pub struct ToolCommand<T: Tool> {
    tool: T,
    schema: Value,
}

impl<T: Tool + std::fmt::Debug> std::fmt::Debug for ToolCommand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCommand")
            .field("tool", &self.tool)
            .finish_non_exhaustive()
    }
}

impl<T: Tool> ToolCommand<T>
where
    T::Arguments: DeserializeOwned + JsonSchema,
{
    /// Creates a new command wrapper for the given tool.
    pub fn new(tool: T) -> Self {
        let schema = schemars::schema_for!(T::Arguments);
        let schema = serde_json::to_value(schema).expect("failed to serialize tool schema");
        Self { tool, schema }
    }

    /// Returns the tool name.
    pub fn name(&self) -> Cow<'static, str> {
        self.tool.name()
    }

    /// Generates help text from the JSON schema.
    #[must_use]
    pub fn help(&self) -> String {
        schema_to_help(&self.schema)
    }

    /// Parses CLI arguments and executes the tool.
    ///
    /// # Errors
    ///
    /// Returns an error if argument parsing fails or tool execution fails.
    pub async fn execute(&self, args: &[String]) -> anyhow::Result<String> {
        let json_args = cli_to_json(&self.schema, args)?;
        let parsed: T::Arguments = serde_json::from_value(json_args)?;
        let output = self.tool.call(parsed).await?;
        // Convert ToolOutput to string for CLI output
        Ok(output.as_str().unwrap_or("").to_string())
    }
}

/// IPC command wrapper that holds a Tool directly (no global state).
///
/// Use this with `router.register()` to add tools to the IPC router
/// without relying on the global tool registry.
pub struct IpcToolCommand<T: Tool> {
    tool: Arc<T>,
    schema: Value,
    name: String,
    positional_args: Vec<String>,
    stdin_arg: Option<String>,
    help: String,
    /// Arguments received from IPC call.
    args: HashMap<String, Value>,
}

impl<T: Tool + std::fmt::Debug> std::fmt::Debug for IpcToolCommand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcToolCommand")
            .field("name", &self.name)
            .field("tool", &self.tool)
            .finish_non_exhaustive()
    }
}

impl<T: Tool + Clone> Clone for IpcToolCommand<T> {
    fn clone(&self) -> Self {
        Self {
            tool: self.tool.clone(),
            schema: self.schema.clone(),
            name: self.name.clone(),
            positional_args: self.positional_args.clone(),
            stdin_arg: self.stdin_arg.clone(),
            help: self.help.clone(),
            args: HashMap::new(),
        }
    }
}

impl<T> IpcToolCommand<T>
where
    T: Tool + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    /// Creates a new IPC command wrapper for the given tool.
    pub fn new(tool: T) -> Self {
        let schema = serde_json::to_value(schemars::schema_for!(T::Arguments))
            .expect("failed to serialize tool schema");
        let name = tool.name().to_string();
        let positional_args = detect_positional_args(&schema);
        let stdin_arg = detect_stdin_arg(&schema);
        let help = schema_to_help(&schema);

        Self {
            tool: Arc::new(tool),
            schema,
            name,
            positional_args,
            stdin_arg,
            help,
            args: HashMap::new(),
        }
    }

    fn args_to_cli(&self) -> Vec<String> {
        // Allow raw CLI args passthrough (positional array)
        if let Some(Value::Array(arr)) = self.args.get("args") {
            return arr
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Number(n) => Some(n.to_string()),
                    Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                })
                .collect();
        }

        let mut cli_args = Vec::new();
        for (key, value) in &self.args {
            match value {
                Value::Bool(true) => cli_args.push(format!("--{key}")),
                Value::Bool(false) => {}
                Value::String(s) => {
                    cli_args.push(format!("--{key}"));
                    cli_args.push(s.clone());
                }
                Value::Number(n) => {
                    cli_args.push(format!("--{key}"));
                    cli_args.push(n.to_string());
                }
                Value::Array(arr) => {
                    for item in arr {
                        cli_args.push(format!("--{key}"));
                        if let Some(s) = item.as_str() {
                            cli_args.push(s.to_string());
                        } else {
                            cli_args.push(item.to_string());
                        }
                    }
                }
                _ => {
                    cli_args.push(format!("--{key}"));
                    cli_args.push(value.to_string());
                }
            }
        }
        cli_args
    }
}

impl<T> IpcCommand for IpcToolCommand<T>
where
    T: Tool + Clone + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Response = Value;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn positional_args(&self) -> Cow<'static, [Cow<'static, str>]> {
        if self.positional_args.is_empty() {
            Cow::Borrowed(&[])
        } else {
            Cow::Owned(
                self.positional_args
                    .iter()
                    .map(|s| Cow::Owned(s.clone()))
                    .collect(),
            )
        }
    }

    fn stdin_arg(&self) -> Option<Cow<'static, str>> {
        self.stdin_arg.clone().map(Cow::Owned)
    }

    fn set_method_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn apply_args(&mut self, params: &[u8]) -> Result<(), leash::rmp_serde::decode::Error> {
        // Only update args, preserve the tool instance with its state
        self.args = leash::rmp_serde::from_slice(params)?;
        Ok(())
    }

    async fn handle(&mut self) -> Value {
        let cli_args = self.args_to_cli();

        // Handle help flags
        if has_help_flag(&cli_args) {
            return Value::String(self.help.clone());
        }

        // Parse CLI args to JSON
        let json_args = match cli_to_json(&self.schema, &cli_args) {
            Ok(args) => args,
            Err(e) => return Value::String(format!("Error: {e}")),
        };

        // Deserialize and call tool
        let parsed: T::Arguments = match serde_json::from_value(json_args) {
            Ok(args) => args,
            Err(e) => return Value::String(format!("Error: failed to parse arguments: {e}")),
        };

        match self.tool.call(parsed).await {
            Ok(output) => {
                let result = output.as_str().unwrap_or("").to_string();
                serde_json::from_str(&result).unwrap_or_else(|_| Value::String(result))
            }
            Err(e) => Value::String(format!("Error: {e}")),
        }
    }
}

impl<T> Serialize for IpcToolCommand<T>
where
    T: Tool,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.args.serialize(serializer)
    }
}

/// Registers a tool directly with the IPC router (no global state).
///
/// This is the preferred way to register tools. The tool is wrapped
/// in an IPC command and added to the router.
///
/// # Example
///
/// ```rust,ignore
/// use aither_sandbox::command::register_tool_direct;
/// use leash::IpcRouter;
///
/// let router = IpcRouter::new();
/// let router = register_tool_direct(router, my_tool);
/// ```
pub fn register_tool_direct<T>(router: leash::IpcRouter, tool: T) -> leash::IpcRouter
where
    T: Tool + Send + Sync + Clone + Default + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    router.register(IpcToolCommand::new(tool))
}

/// Converts CLI arguments to JSON using schema information.
///
/// Handles:
/// - Tagged enums (`#[serde(tag = "operation")]`) → subcommands
/// - Long options → `--flag value`, `--flag=value`, `--no-flag`
/// - Short options → `-f`, clusters like `-abc`, and values `-oVALUE` / `-o=VALUE`
/// - `--` to end option parsing
/// - Struct fields → `--flag value` or positional args
/// - Optional fields → optional flags
///
/// # Errors
///
/// Returns an error if the arguments don't match the schema.
pub fn cli_to_json(schema: &Value, args: &[String]) -> anyhow::Result<Value> {
    // Check if this is a tagged enum (oneOf with discriminator)
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        // Look for serde tag annotation in schema
        if let Some(tag) = find_serde_tag(schema) {
            return parse_tagged_enum(schema, one_of, &tag, args);
        }
    }

    // Otherwise treat as a simple object
    parse_object(schema, args)
}

/// Finds the serde tag field name from schema extensions.
fn find_serde_tag(schema: &Value) -> Option<String> {
    // Check for discriminator property (OpenAPI style)
    if let Some(disc) = schema.get("discriminator") {
        if let Some(prop) = disc.get("propertyName").and_then(Value::as_str) {
            return Some(prop.to_string());
        }
    }

    // Check oneOf variants for common const field (serde tag pattern)
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array) {
        if let Some(first) = variants.first() {
            if let Some(props) = first.get("properties").and_then(Value::as_object) {
                for (name, prop) in props {
                    // If this property has a const value, it's likely the tag
                    if prop.get("const").is_some() || prop.get("enum").is_some() {
                        return Some(name.clone());
                    }
                }
            }
        }
    }

    None
}

/// Parses a tagged enum from CLI arguments.
fn parse_tagged_enum(
    _root_schema: &Value,
    variants: &[Value],
    tag: &str,
    args: &[String],
) -> anyhow::Result<Value> {
    let args = if args.first().map(|s| s.as_str()) == Some("--") {
        &args[1..]
    } else {
        args
    };

    if args.is_empty() {
        let variant_names: Vec<_> = variants
            .iter()
            .filter_map(|v| get_variant_name(v, tag))
            .collect();
        anyhow::bail!("expected subcommand: {}", variant_names.join(", "));
    }

    let subcommand = &args[0];
    let remaining = &args[1..];

    // Find matching variant
    for variant in variants {
        if let Some(name) = get_variant_name(variant, tag) {
            if name.eq_ignore_ascii_case(subcommand) {
                // Parse the variant's fields
                let mut result = parse_object(variant, remaining)?;

                // Add the tag field
                if let Value::Object(ref mut map) = result {
                    map.insert(tag.to_string(), Value::String(name));
                }

                return Ok(result);
            }
        }
    }

    let variant_names: Vec<_> = variants
        .iter()
        .filter_map(|v| get_variant_name(v, tag))
        .collect();
    anyhow::bail!(
        "unknown subcommand '{}', expected one of: {}",
        subcommand,
        variant_names.join(", ")
    );
}

/// Gets the variant name from a schema object.
fn get_variant_name(schema: &Value, tag: &str) -> Option<String> {
    // Look for const value in tag property
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        if let Some(prop) = props.get(tag) {
            if let Some(const_val) = prop.get("const").and_then(Value::as_str) {
                return Some(const_val.to_string());
            }
            // Also check enum with single value
            if let Some(enum_vals) = prop.get("enum").and_then(Value::as_array) {
                if enum_vals.len() == 1 {
                    return enum_vals[0].as_str().map(String::from);
                }
            }
        }
    }

    // Fallback: use title
    schema
        .get("title")
        .and_then(Value::as_str)
        .map(String::from)
}

/// Returns true if args contain -h/--help before "--".
fn has_help_flag(args: &[String]) -> bool {
    let mut end_of_options = false;
    for arg in args {
        if end_of_options {
            break;
        }
        if arg == "--" {
            end_of_options = true;
            continue;
        }
        if arg == "--help" || arg == "-h" {
            return true;
        }
        if arg.starts_with('-') && !arg.starts_with("--") {
            if arg.chars().skip(1).any(|c| c == 'h') {
                return true;
            }
        }
    }
    false
}

fn build_short_option_maps(
    properties: &serde_json::Map<String, Value>,
) -> (HashMap<char, String>, HashMap<String, char>) {
    let mut taken = HashSet::new();
    taken.insert('h');
    let mut short_to_field = HashMap::new();
    let mut field_to_short = HashMap::new();

    for field in properties.keys() {
        let long = field.replace('_', "-");
        if let Some(ch) = pick_short_option(&long, &taken) {
            taken.insert(ch);
            short_to_field.insert(ch, field.clone());
            field_to_short.insert(field.clone(), ch);
        }
    }

    (short_to_field, field_to_short)
}

fn pick_short_option(long: &str, taken: &HashSet<char>) -> Option<char> {
    for ch in long.chars() {
        if !ch.is_ascii_alphabetic() {
            continue;
        }
        let lower = ch.to_ascii_lowercase();
        if !taken.contains(&lower) {
            return Some(lower);
        }
    }
    None
}

/// Finds the most similar option name for typo suggestions.
fn find_similar_option<'a>(input: &str, options: impl Iterator<Item = &'a str>) -> Option<String> {
    let input_lower = input.to_lowercase().replace('_', "-");
    options
        .map(|opt| (opt, strsim::levenshtein(&input_lower, &opt.to_lowercase())))
        .filter(|(_, dist)| *dist <= 2) // Only suggest if edit distance <= 2
        .min_by_key(|(_, dist)| *dist)
        .map(|(opt, _)| opt.to_string())
}

/// Parses an object schema from CLI arguments.
fn parse_object(schema: &Value, args: &[String]) -> anyhow::Result<Value> {
    let mut result: HashMap<String, Value> = HashMap::new();
    let mut positional_idx = 0;

    // Get object properties
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let required: Vec<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    // Positional fields are the required fields in schema order
    let positional_fields = required.clone();
    let (short_to_field, _) = build_short_option_maps(&properties);
    let long_names: Vec<String> = properties.keys().map(|k| k.replace('_', "-")).collect();

    let mut i = 0;
    let mut end_of_options = false;
    while i < args.len() {
        let arg = &args[i];

        if !end_of_options && arg == "--" {
            end_of_options = true;
            i += 1;
            continue;
        }

        if !end_of_options && arg == "-" {
            // Single "-" is treated as positional
            if positional_idx < positional_fields.len() {
                let field_name = &positional_fields[positional_idx];
                if let Some(prop_schema) = properties.get(field_name) {
                    let prop_type = get_instance_type(prop_schema);
                    result.insert(field_name.clone(), parse_value(arg, prop_type));
                }
                positional_idx += 1;
            } else {
                anyhow::bail!("unexpected positional argument: {}", arg);
            }
            i += 1;
            continue;
        }

        if !end_of_options && arg.starts_with("--") {
            // Named argument
            let flag = &arg[2..];
            let (mut name, value) = if let Some(eq_pos) = flag.find('=') {
                (
                    flag[..eq_pos].to_string(),
                    Some(flag[eq_pos + 1..].to_string()),
                )
            } else {
                (flag.to_string(), None)
            };

            // Normalize long name (allow underscores)
            name = name.replace('_', "-");

            let mut negated = false;
            if let Some(stripped) = name.strip_prefix("no-") {
                name = stripped.to_string();
                negated = true;
            }

            // Convert kebab-case to snake_case for matching
            let field_name = name.replace('-', "_");

            if let Some(prop_schema) = properties.get(&field_name) {
                let prop_type = get_instance_type(prop_schema);

                if negated && prop_type != Some("boolean") {
                    anyhow::bail!("--no-{} is only valid for boolean options", name);
                }

                let parsed_value = if prop_type == Some("boolean") {
                    if negated && value.is_some() {
                        anyhow::bail!("unexpected value for --no-{}", name);
                    }
                    if let Some(v) = value {
                        parse_value(&v, prop_type)
                    } else {
                        Value::Bool(!negated)
                    }
                } else {
                    let val = value.or_else(|| {
                        i += 1;
                        args.get(i).cloned()
                    });
                    match val {
                        Some(v) => parse_value(&v, prop_type),
                        None => anyhow::bail!("missing value for --{}", name),
                    }
                };

                insert_value(&mut result, &field_name, parsed_value, prop_schema);
            } else {
                let suggestion = find_similar_option(&name, long_names.iter().map(String::as_str));
                if let Some(similar) = suggestion {
                    anyhow::bail!("unknown option: --{}. Did you mean --{}?", name, similar);
                } else {
                    anyhow::bail!("unknown option: --{}", name);
                }
            }
        } else if !end_of_options && arg.starts_with('-') && arg.len() > 1 {
            parse_short_options(arg, args, &mut i, &properties, &short_to_field, &mut result)?;
        } else {
            // Positional argument
            if positional_idx < positional_fields.len() {
                let field_name = &positional_fields[positional_idx];
                if let Some(prop_schema) = properties.get(field_name) {
                    let prop_type = get_instance_type(prop_schema);
                    result.insert(field_name.clone(), parse_value(arg, prop_type));
                }
                positional_idx += 1;
            } else {
                anyhow::bail!("unexpected positional argument: {}", arg);
            }
        }

        i += 1;
    }

    // Check required fields
    for req in &required {
        if !result.contains_key(req) {
            anyhow::bail!("missing required argument: {}", req);
        }
    }

    Ok(Value::Object(result.into_iter().collect()))
}

fn parse_short_options(
    arg: &str,
    args: &[String],
    i: &mut usize,
    properties: &serde_json::Map<String, Value>,
    short_to_field: &HashMap<char, String>,
    result: &mut HashMap<String, Value>,
) -> anyhow::Result<()> {
    let mut cluster = &arg[1..];
    let mut value_from_eq = None;
    if let Some(eq_pos) = cluster.find('=') {
        value_from_eq = Some(cluster[eq_pos + 1..].to_string());
        cluster = &cluster[..eq_pos];
    }

    let mut chars = cluster.chars().peekable();
    while let Some(ch) = chars.next() {
        let field_name = short_to_field
            .get(&ch)
            .ok_or_else(|| anyhow::anyhow!("unknown option: -{ch}"))?
            .clone();
        let prop_schema = properties
            .get(&field_name)
            .ok_or_else(|| anyhow::anyhow!("unknown option: -{ch}"))?;
        let prop_type = get_instance_type(prop_schema);

        if prop_type == Some("boolean") {
            if chars.peek().is_none() {
                if let Some(v) = value_from_eq.take() {
                    let parsed = parse_value(&v, prop_type);
                    insert_value(result, &field_name, parsed, prop_schema);
                } else {
                    insert_value(result, &field_name, Value::Bool(true), prop_schema);
                }
            } else {
                if value_from_eq.is_some() {
                    anyhow::bail!("unexpected value for -{}", ch);
                }
                insert_value(result, &field_name, Value::Bool(true), prop_schema);
            }
            continue;
        }

        let value = if let Some(v) = value_from_eq.take() {
            if chars.peek().is_some() {
                anyhow::bail!("unexpected value for -{}", ch);
            }
            v
        } else if chars.peek().is_some() {
            let rest: String = chars.collect();
            rest
        } else {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for -{}", ch))?
        };

        let parsed = parse_value(&value, prop_type);
        insert_value(result, &field_name, parsed, prop_schema);
        break;
    }

    Ok(())
}

fn insert_value(
    result: &mut HashMap<String, Value>,
    field_name: &str,
    value: Value,
    prop_schema: &Value,
) {
    if get_instance_type(prop_schema) == Some("array") {
        match result.get_mut(field_name) {
            Some(Value::Array(existing)) => match value {
                Value::Array(mut items) => existing.append(&mut items),
                other => existing.push(other),
            },
            _ => {
                let mut items = Vec::new();
                match value {
                    Value::Array(mut arr) => items.append(&mut arr),
                    other => items.push(other),
                }
                result.insert(field_name.to_string(), Value::Array(items));
            }
        }
    } else {
        result.insert(field_name.to_string(), value);
    }
}

/// Gets the instance type from a schema.
/// Handles both simple types and Option<T>.
fn get_instance_type(schema: &Value) -> Option<&str> {
    match schema.get("type") {
        // Direct string type: "type": "integer"
        Some(Value::String(t)) => Some(t.as_str()),

        // Array of types (schemars 1.0 for Option<T>): "type": ["integer", "null"]
        Some(Value::Array(types)) => {
            // Find the non-null type
            for t in types {
                if let Some(s) = t.as_str() {
                    if s != "null" {
                        return Some(s);
                    }
                }
            }
            None
        }

        // No direct type - try anyOf/oneOf (older schemars or complex types)
        None => {
            let variants = schema
                .get("anyOf")
                .or_else(|| schema.get("oneOf"))
                .and_then(Value::as_array)?;

            for variant in variants {
                if let Some(t) = variant.get("type").and_then(Value::as_str) {
                    if t != "null" {
                        return Some(t);
                    }
                }
            }
            None
        }

        _ => None,
    }
}

/// Parses a string value according to expected type.
fn parse_value(s: &str, expected_type: Option<&str>) -> Value {
    match expected_type {
        Some("integer") => s
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        Some("number") => s
            .parse::<f64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        Some("boolean") => match s.to_lowercase().as_str() {
            "true" | "1" | "yes" => Value::Bool(true),
            "false" | "0" | "no" => Value::Bool(false),
            _ => Value::String(s.to_string()),
        },
        Some("array") => {
            // Try parsing as JSON array, fallback to single-element array
            serde_json::from_str(s)
                .unwrap_or_else(|_| Value::Array(vec![Value::String(s.to_string())]))
        }
        Some("object") => {
            // Try parsing as JSON object
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()))
        }
        _ => Value::String(s.to_string()),
    }
}

/// Generates help text from a JSON schema.
#[must_use]
pub fn schema_to_help(schema: &Value) -> String {
    let mut help = String::new();

    // Title and description
    if let Some(title) = schema.get("title").and_then(Value::as_str) {
        help.push_str(title);
        help.push('\n');
    }
    if let Some(desc) = schema.get("description").and_then(Value::as_str) {
        help.push_str(desc);
        help.push('\n');
    }

    help.push_str("\nUsage:\n");

    // Check for tagged enum (subcommands)
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array) {
        if let Some(tag) = find_serde_tag(schema) {
            help.push_str("  <subcommand> [options]\n\n");
            help.push_str("Subcommands:\n");

            for variant in variants {
                if let Some(name) = get_variant_name(variant, &tag) {
                    help.push_str(&format!("  {name}"));
                    if let Some(desc) = variant.get("description").and_then(Value::as_str) {
                        help.push_str(&format!(" - {desc}"));
                    }
                    help.push('\n');
                }
            }
            help.push_str("\nOptions:\n  -h, --help  Show help\n");
            return help;
        }
    }

    // Simple object
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        // Show positional usage for required args
        if !required.is_empty() {
            let positional: Vec<_> = required
                .iter()
                .map(|n| format!("<{}>", n.replace('_', "-")))
                .collect();
            help.push_str(&format!("  {} [options]\n", positional.join(" ")));
        } else {
            help.push_str("  [options]\n");
        }

        help.push_str("\nOptions:\n  -h, --help  Show help\n");
        help.push_str("\nArguments:\n");

        let (_, field_to_short) = build_short_option_maps(props);

        for (name, prop) in props {
            let is_required = required.contains(&name.as_str());
            let flag = name.replace('_', "-");

            if let Some(short) = field_to_short.get(name) {
                help.push_str(&format!("  -{short}, --{flag}"));
            } else {
                help.push_str(&format!("  --{flag}"));
            }
            if is_required {
                help.push_str(" (required)");
            }

            if let Some(desc) = prop.get("description").and_then(Value::as_str) {
                help.push_str(&format!("\n      {desc}"));
            }
            help.push('\n');
        }
    }

    help
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct SimpleArgs {
        #[schemars(description = "The input file path")]
        path: String,
        #[schemars(description = "Number of lines to read")]
        count: Option<i32>,
    }

    #[test]
    fn test_parse_simple_args() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Positional
        let result = cli_to_json(&schema, &["foo.txt".to_string()]).unwrap();
        assert_eq!(result["path"], "foo.txt");

        // Named
        let result = cli_to_json(&schema, &["--path".to_string(), "bar.txt".to_string()]).unwrap();
        assert_eq!(result["path"], "bar.txt");

        // With optional
        let result = cli_to_json(
            &schema,
            &[
                "--path".to_string(),
                "baz.txt".to_string(),
                "--count".to_string(),
                "10".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(result["path"], "baz.txt");
        assert_eq!(result["count"], 10);
    }

    #[test]
    fn test_short_flags() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(
            &schema,
            &[
                "-p".to_string(),
                "short.txt".to_string(),
                "-c".to_string(),
                "7".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(result["path"], "short.txt");
        assert_eq!(result["count"], 7);
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ClusterArgs {
        /// Verbose output
        verbose: bool,
        /// Output path
        output: String,
    }

    #[test]
    fn test_short_cluster_with_value() {
        let schema = schemars::schema_for!(ClusterArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(&schema, &["-vofile.txt".to_string()]).unwrap();
        assert_eq!(result["verbose"], true);
        assert_eq!(result["output"], "file.txt");
    }

    #[test]
    fn test_end_of_options() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(&schema, &["--".to_string(), "-dash.txt".to_string()]).unwrap();
        assert_eq!(result["path"], "-dash.txt");
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TwoPositionalArgs {
        /// First required arg
        first: String,
        /// Second required arg
        second: String,
    }

    #[test]
    fn test_multiple_positional_args() {
        let schema = schemars::schema_for!(TwoPositionalArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Both as positional
        let result = cli_to_json(&schema, &["hello".to_string(), "world".to_string()]).unwrap();
        assert_eq!(result["first"], "hello");
        assert_eq!(result["second"], "world");

        // Mix positional and named
        let result = cli_to_json(
            &schema,
            &[
                "hello".to_string(),
                "--second".to_string(),
                "world".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(result["first"], "hello");
        assert_eq!(result["second"], "world");
    }

    #[test]
    fn test_typo_suggestion() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Typo in flag name
        let err = cli_to_json(&schema, &["--paht".to_string(), "foo.txt".to_string()]).unwrap_err();
        assert!(err.to_string().contains("Did you mean --path?"));
    }

    #[test]
    fn test_schema_to_help() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();
        let help = schema_to_help(&schema);

        assert!(help.contains("<path>")); // Shows positional usage
        assert!(help.contains("-p, --path"));
        assert!(help.contains("--path"));
        assert!(help.contains("-c, --count"));
        assert!(help.contains("--count"));
    }

    #[test]
    fn test_tool_call_command_serialization() {
        // Simulate IPC params (key-value pairs from leash-ipc)
        let json = r#"{"query": "rust async", "max": 5}"#;
        let tmp = tempfile::tempdir().unwrap();
        let registry = ToolRegistryBuilder::new().build(tmp.path());
        let mut cmd = ToolCallCommand::new("websearch", Arc::new(registry));
        cmd.args = serde_json::from_str(json).unwrap();

        // tool_name is not in JSON (skipped), args are flattened
        assert_eq!(cmd.args.len(), 2);
        assert_eq!(cmd.args.get("query").unwrap(), "rust async");
        assert_eq!(cmd.args.get("max").unwrap(), 5);

        // Test args_to_cli conversion
        let cli_args = cmd.args_to_cli();
        assert!(cli_args.contains(&"--query".to_string()));
        assert!(cli_args.contains(&"rust async".to_string()));
    }

    #[test]
    fn test_tool_call_command_name() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = ToolRegistryBuilder::new().build(tmp.path());
        let cmd = ToolCallCommand::new("my_tool", Arc::new(registry));
        assert_eq!(cmd.name(), "my_tool");
    }
}
