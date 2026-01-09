//! Tool-to-command conversion using JSON schema.
//!
//! Wraps any `Tool` as an IPC command callable from the sandbox.
//! Uses the existing JSON schema to generate CLI parsing - no clap needed.
//!
//! # Architecture
//!
//! Tools are registered with [`configure_tool`], which stores a type-erased
//! handler in a global registry. [`ToolCallCommand`] implements `IpcCommand`
//! and dispatches to the appropriate handler based on the tool name.
//!
//! # Example
//!
//! ```rust,ignore
//! use aither_sandbox::command::{configure_tool, register_tool_command};
//! use aither_sandbox::builtin::builtin_router;
//!
//! // Configure tools (stores handlers in global registry)
//! configure_tool(websearch_tool);
//! configure_tool(webfetch_tool);
//!
//! // Register tool commands with router
//! let router = builtin_router();
//! let router = register_tool_command(router, "websearch");
//! let router = register_tool_command(router, "webfetch");
//! ```

use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};

use aither_core::llm::Tool;
use leash::IpcCommand;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Tool Registry (global handler storage)
// ============================================================================

/// Type-erased tool handler function.
type ToolHandlerFn =
    Box<dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Tool entry in the registry.
struct ToolEntry {
    handler: ToolHandlerFn,
    help: String,
    primary_arg: Option<String>,
    stdin_arg: Option<String>,
}

/// Global registry of tools.
static TOOL_REGISTRY: OnceLock<RwLock<HashMap<String, ToolEntry>>> = OnceLock::new();

fn get_registry() -> &'static RwLock<HashMap<String, ToolEntry>> {
    TOOL_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

// ============================================================================
// Output Size Handling (system-level offload for large outputs)
// ============================================================================

use crate::output::INLINE_OUTPUT_LIMIT;

/// Counter for generating unique output filenames.
static OUTPUT_COUNTER: AtomicU32 = AtomicU32::new(1);

/// Handles large outputs by saving to file if needed.
/// Returns the output as-is if small enough, or a file reference message if saved.
fn handle_large_output(output: String) -> String {
    if output.len() <= INLINE_OUTPUT_LIMIT {
        return output;
    }

    // Large output - save to file
    let count = OUTPUT_COUNTER.fetch_add(1, Ordering::SeqCst);
    let filename = format!("outputs/{:03}.txt", count);

    // Count lines for the message
    let line_count = output.lines().count();

    // Try to save - if it fails, fall back to truncated inline output
    match std::fs::write(&filename, &output) {
        Ok(()) => {
            format!("Output saved to {} ({} lines, {} bytes)", filename, line_count, output.len())
        }
        Err(e) => {
            // Fallback: show truncated output with error note
            let truncated: String = output.chars().take(INLINE_OUTPUT_LIMIT).collect();
            format!("{}\n\n[truncated - failed to save full output: {}]", truncated, e)
        }
    }
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

/// Type-erased bash tool factory.
type BashToolFactory = Arc<dyn Fn() -> DynBashTool + Send + Sync>;

/// Global factory for creating child bash tools.
static BASH_TOOL_FACTORY: OnceLock<RwLock<Option<BashToolFactory>>> = OnceLock::new();

fn get_bash_factory() -> &'static RwLock<Option<BashToolFactory>> {
    BASH_TOOL_FACTORY.get_or_init(|| RwLock::new(None))
}

/// Sets the global bash tool factory.
///
/// This should be called once during initialization with a closure that creates
/// child bash tools. Subagents created by TaskTool will use this factory.
///
/// # Example
///
/// ```rust,ignore
/// let bash_tool = BashTool::new_in(...).await?;
/// set_bash_tool_factory({
///     let bash_tool = bash_tool.clone();
///     move || {
///         let child = bash_tool.child();
///         child.to_dyn()
///     }
/// });
/// ```
pub fn set_bash_tool_factory<F>(factory: F)
where
    F: Fn() -> DynBashTool + Send + Sync + 'static,
{
    *get_bash_factory().write().expect("bash factory poisoned") = Some(Arc::new(factory));
}

/// Creates a child bash tool using the global factory.
///
/// Returns `None` if no factory has been set.
pub fn create_child_bash_tool() -> Option<DynBashTool> {
    let guard = get_bash_factory().read().expect("bash factory poisoned");
    guard.as_ref().map(|factory| factory())
}

/// Configures a tool to be callable from the sandbox.
///
/// This stores a type-erased handler in the global registry. Call this
/// before creating the sandbox to make tools available as IPC commands.
///
/// The primary argument for positional argument conversion is automatically
/// detected from the schema - the first required field becomes the primary arg.
/// For example, if a tool has `query: String` as its first required field,
/// then `websearch "rust"` → `websearch --query "rust"`.
///
/// # Example
///
/// ```rust,ignore
/// use aither_sandbox::command::configure_tool;
///
/// let websearch = WebSearchTool::new(api_key);
/// configure_tool(websearch);
/// // Primary arg auto-detected from schema (e.g., "query")
/// ```
pub fn configure_tool<T>(tool: T)
where
    T: Tool + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    let schema = serde_json::to_value(schemars::schema_for!(T::Arguments)).unwrap_or(Value::Null);

    // Auto-detect primary_arg from schema: first required field
    let primary_arg = detect_primary_arg(&schema);
    // Auto-detect stdin_arg from schema: optional "input" field
    let stdin_arg = detect_stdin_arg(&schema);

    let tool = Arc::new(tool);
    let tool_name = tool.name().to_string();
    let help_text = schema_to_help(&schema);

    let handler: ToolHandlerFn = Box::new(move |args: Vec<String>| {
        let tool = tool.clone();
        let schema = schema.clone();
        Box::pin(async move {
            match cli_to_json(&schema, &args) {
                Ok(json_args) => match serde_json::from_value(json_args) {
                    Ok(parsed) => match tool.call(parsed).await {
                        Ok(output) => output.as_str().unwrap_or("").to_string(),
                        Err(e) => format!("Error: {}", e),
                    },
                    Err(e) => format!("Error: failed to parse arguments: {}", e),
                },
                Err(e) => format!("Error: {}", e),
            }
        })
    });

    get_registry()
        .write()
        .expect("tool registry poisoned")
        .insert(
            tool_name,
            ToolEntry {
                handler,
                help: help_text,
                primary_arg,
                stdin_arg,
            },
        );
}

/// Detects the primary argument from a JSON schema.
///
/// Returns the first required field name, which will be used for
/// positional argument conversion in wrapper scripts.
fn detect_primary_arg(schema: &Value) -> Option<String> {
    // Get the required array
    let required = schema.get("required")?.as_array()?;

    // Get the first required field
    let first_required = required.first()?.as_str()?;

    Some(first_required.to_string())
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

/// Queries a tool handler by name.
///
/// Large outputs (> 4000 chars) are automatically saved to file.
async fn query_tool_handler(tool_name: &str, args: &[String]) -> String {
    // Get the handler future while holding the lock, then release before awaiting
    let future = {
        let registry = get_registry().read().expect("tool registry poisoned");
        match registry.get(tool_name) {
            Some(entry) => Some((entry.handler)(args.to_vec())),
            None => None,
        }
    };

    match future {
        Some(fut) => {
            let output = fut.await;
            // Handle large outputs by saving to file
            handle_large_output(output)
        }
        None => format!("Error: unknown command '{}'. Run 'help' to see available commands.", tool_name),
    }
}

/// Returns whether a tool is registered.
#[must_use]
pub fn is_tool_configured(tool_name: &str) -> bool {
    get_registry()
        .read()
        .expect("tool registry poisoned")
        .contains_key(tool_name)
}

/// Returns the help text for a tool.
#[must_use]
pub fn get_tool_help(tool_name: &str) -> Option<String> {
    get_registry()
        .read()
        .expect("tool registry poisoned")
        .get(tool_name)
        .map(|e| e.help.clone())
}

/// Returns the primary argument for a tool.
#[must_use]
pub fn get_tool_primary_arg(tool_name: &str) -> Option<String> {
    get_registry()
        .read()
        .expect("tool registry poisoned")
        .get(tool_name)
        .and_then(|e| e.primary_arg.clone())
}

/// Returns the stdin argument for a tool.
#[must_use]
pub fn get_tool_stdin_arg(tool_name: &str) -> Option<String> {
    get_registry()
        .read()
        .expect("tool registry poisoned")
        .get(tool_name)
        .and_then(|e| e.stdin_arg.clone())
}

/// Returns list of registered tool names.
#[must_use]
pub fn registered_tool_names() -> Vec<String> {
    get_registry()
        .read()
        .expect("tool registry poisoned")
        .keys()
        .cloned()
        .collect()
}

/// Registers a raw handler function as an IPC command.
///
/// This is useful for dynamic tools like MCP tools that are discovered at runtime.
/// The handler receives CLI-style arguments and returns a string result.
///
/// # Arguments
///
/// * `name` - The command name (e.g., "resolve-library-id")
/// * `help` - Help text shown when user runs `<name> --help`
/// * `primary_arg` - Optional name of the primary argument for positional conversion
/// * `handler` - Async function that processes args and returns result
///
/// # Example
///
/// ```rust,ignore
/// configure_raw_handler(
///     "my-tool",
///     "Usage: my-tool --query <text>",
///     Some("query"),
///     move |args| {
///         let query = args.get(0).cloned().unwrap_or_default();
///         Box::pin(async move { format!("Result: {query}") })
///     },
/// );
/// ```
pub fn configure_raw_handler<F>(name: impl Into<String>, help: impl Into<String>, primary_arg: Option<String>, handler: F)
where
    F: Fn(Vec<String>) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync + 'static,
{
    get_registry()
        .write()
        .expect("tool registry poisoned")
        .insert(
            name.into(),
            ToolEntry {
                handler: Box::new(handler),
                help: help.into(),
                primary_arg,
                stdin_arg: None,
            },
        );
}

// ============================================================================
// IPC Command for Tools
// ============================================================================

/// IPC command for invoking tools from the sandbox.
///
/// The tool_name comes from the IPC method name, and args are flattened
/// from the key-value pairs sent by leash-ipc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCommand {
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
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
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

    fn primary_arg(&self) -> Option<Cow<'static, str>> {
        get_tool_primary_arg(&self.tool_name).map(Cow::Owned)
    }

    fn stdin_arg(&self) -> Option<Cow<'static, str>> {
        get_tool_stdin_arg(&self.tool_name).map(Cow::Owned)
    }

    fn set_method_name(&mut self, name: &str) {
        self.tool_name = name.to_string();
    }

    async fn handle(&mut self) -> Value {
        let cli_args = self.args_to_cli();

        // Handle --help flag
        if cli_args.first().map(|s| s.as_str()) == Some("--help") {
            let help = get_tool_help(&self.tool_name)
                .unwrap_or_else(|| format!("No help available for '{}'", self.tool_name));
            return Value::String(help);
        }

        let result = query_tool_handler(&self.tool_name, &cli_args).await;

        // Parse the result as JSON to avoid double-serialization.
        // Tool outputs are JSON strings, so we need to parse them to Value.
        serde_json::from_str(&result).unwrap_or_else(|_| Value::String(result))
    }
}

/// Registers a tool command with the IPC router.
///
/// The tool must have been previously configured with [`configure_tool`].
///
/// # Example
///
/// ```rust,ignore
/// use aither_sandbox::command::register_tool_command;
/// use leash::IpcRouter;
///
/// let router = IpcRouter::new();
/// let router = register_tool_command(router, "websearch");
/// ```
pub fn register_tool_command(router: leash::IpcRouter, tool_name: &str) -> leash::IpcRouter {
    router.register(ToolCallCommand::new(tool_name))
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
        let schema = serde_json::to_value(schema).unwrap_or(Value::Null);
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

/// Converts CLI arguments to JSON using schema information.
///
/// Handles:
/// - Tagged enums (`#[serde(tag = "operation")]`) → subcommands
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
    schema.get("title").and_then(Value::as_str).map(String::from)
}

/// Finds the most similar option name for typo suggestions.
fn find_similar_option<'a>(input: &str, options: impl Iterator<Item = &'a str>) -> Option<String> {
    let input_lower = input.to_lowercase().replace('-', "_");
    options
        .map(|opt| (opt, strsim::levenshtein(&input_lower, &opt.to_lowercase())))
        .filter(|(_, dist)| *dist <= 2) // Only suggest if edit distance <= 2
        .min_by_key(|(_, dist)| *dist)
        .map(|(opt, _)| opt.replace('_', "-"))
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
        .map(|arr| arr.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default();

    // Positional fields are the required fields in schema order
    let positional_fields = required.clone();

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];

        if arg.starts_with("--") {
            // Named argument
            let flag = &arg[2..];
            let (name, value) = if let Some(eq_pos) = flag.find('=') {
                (flag[..eq_pos].to_string(), Some(flag[eq_pos + 1..].to_string()))
            } else {
                (flag.to_string(), None)
            };

            // Convert kebab-case to snake_case for matching
            let field_name = name.replace('-', "_");

            if let Some(prop_schema) = properties.get(&field_name) {
                let prop_type = get_instance_type(prop_schema);

                let parsed_value = if prop_type == Some("boolean") {
                    // Boolean flags don't need a value
                    Value::Bool(true)
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

                result.insert(field_name, parsed_value);
            } else {
                let suggestion = find_similar_option(&name, properties.keys().map(String::as_str));
                if let Some(similar) = suggestion {
                    anyhow::bail!("unknown option: --{}. Did you mean --{}?", name, similar);
                } else {
                    anyhow::bail!("unknown option: --{}", name);
                }
            }
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
            serde_json::from_str(s).unwrap_or_else(|_| Value::Array(vec![Value::String(s.to_string())]))
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
            let positional: Vec<_> = required.iter().map(|n| format!("<{}>", n.replace('_', "-"))).collect();
            help.push_str(&format!("  {} [options]\n", positional.join(" ")));
        } else {
            help.push_str("  [options]\n");
        }

        help.push_str("\nArguments:\n");

        for (name, prop) in props {
            let is_required = required.contains(&name.as_str());
            let flag = name.replace('_', "-");

            help.push_str(&format!("  --{flag}"));
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
        let result = cli_to_json(&schema, &["hello".to_string(), "--second".to_string(), "world".to_string()]).unwrap();
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
        assert!(help.contains("--path"));
        assert!(help.contains("--count"));
    }

    #[test]
    fn test_tool_call_command_serialization() {
        // Simulate IPC params (key-value pairs from leash-ipc)
        let json = r#"{"query": "rust async", "max": 5}"#;
        let mut deserialized: ToolCallCommand = serde_json::from_str(json).unwrap();
        deserialized.tool_name = "websearch".to_string(); // Set by router

        // tool_name is not in JSON (skipped), args are flattened
        assert_eq!(deserialized.args.len(), 2);
        assert_eq!(deserialized.args.get("query").unwrap(), "rust async");
        assert_eq!(deserialized.args.get("max").unwrap(), 5);

        // Test args_to_cli conversion
        let cli_args = deserialized.args_to_cli();
        assert!(cli_args.contains(&"--query".to_string()));
        assert!(cli_args.contains(&"rust async".to_string()));
    }

    #[test]
    fn test_tool_call_command_name() {
        let cmd = ToolCallCommand::new("my_tool");
        assert_eq!(cmd.name(), "my_tool");
    }
}
