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
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use aither_core::llm::{IntoToolResult, Tool, ToolResult};
use askama::Template;
use base64::Engine as _;
use heel::IpcCommand;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Tool Registry
// ============================================================================

/// Payload exchanged between IPC command handlers and terminal command wrappers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommandPayload {
    /// UTF-8 text output.
    Text {
        /// Text content.
        content: String,
    },
    /// Structured JSON output.
    Json {
        /// JSON value.
        value: Value,
    },
    /// Binary output encoded as base64 with a MIME type.
    Binary {
        /// MIME type of the binary payload.
        mime: String,
        /// Base64-encoded bytes.
        data_base64: String,
    },
}

#[derive(Template)]
#[template(path = "command_output_saved.txt", escape = "none")]
struct CommandOutputSavedTemplate<'a> {
    preview: &'a str,
    filename: &'a str,
    line_count: usize,
    byte_count: usize,
}

#[derive(Template)]
#[template(path = "tool_usage_error.txt", escape = "none")]
struct ToolUsageErrorTemplate<'a> {
    message: &'a str,
    tool_name: &'a str,
}

#[derive(Template)]
#[template(path = "ipc_gateway_help.txt", escape = "none")]
struct IpcGatewayHelpTemplate<'a> {
    tool_names: &'a str,
}

fn prefix_error(prefix: &str, error: &impl std::fmt::Display) -> String {
    let error_text = error.to_string();
    let mut message = String::with_capacity(prefix.len() + error_text.len());
    message.push_str(prefix);
    message.push_str(error_text.as_str());
    message
}

fn tool_usage_error(message: &str, tool_name: &str) -> String {
    ToolUsageErrorTemplate { message, tool_name }
        .render()
        .expect("tool usage error template must render")
}

fn output_saved_text(
    rendered: &str,
    filename: &str,
    line_count: usize,
    byte_count: usize,
) -> String {
    let preview = crate::output::head_preview(rendered, usize::MAX, INLINE_OUTPUT_LIMIT);
    CommandOutputSavedTemplate {
        preview: preview.trim_end_matches('\n'),
        filename,
        line_count,
        byte_count,
    }
    .render()
    .expect("command output saved template must render")
}

fn unknown_command_text(tool_name: &str) -> String {
    let mut message = String::with_capacity(tool_name.len() + 47);
    message.push_str("unknown command '");
    message.push_str(tool_name);
    message.push_str("'. Run 'help' to see available commands.");
    message
}

fn no_help_available_text(tool_name: &str) -> String {
    let mut message = String::with_capacity(tool_name.len() + 23);
    message.push_str("No help available for '");
    message.push_str(tool_name);
    message.push('\'');
    message
}

fn push_help_line(help: &mut String, line: &str) {
    help.push_str(line);
    help.push('\n');
}

fn render_positional_usage(name: &str) -> String {
    let kebab = name.replace('_', "-");
    let mut output = String::with_capacity(kebab.len() + 2);
    output.push('<');
    output.push_str(kebab.as_str());
    output.push('>');
    output
}

fn render_repeatable_usage(name: &str) -> String {
    let kebab = name.replace('_', "-");
    let mut output = String::with_capacity(kebab.len() + 12);
    output.push_str("--");
    output.push_str(kebab.as_str());
    output.push_str(" <value>...");
    output
}

fn render_flag_prefix(flag: &str, short: Option<char>) -> String {
    let mut output = String::with_capacity(flag.len() + 8);
    output.push_str("  ");
    if let Some(short) = short {
        output.push('-');
        output.push(short);
        output.push_str(", ");
    }
    output.push_str("--");
    output.push_str(flag);
    output
}

impl CommandPayload {
    fn render_for_cli(&self) -> Result<String, String> {
        match self {
            Self::Text { content } => Ok(content.clone()),
            Self::Json { value } => serde_json::to_string_pretty(value)
                .map_err(|error| prefix_error("failed to encode JSON output: ", &error)),
            Self::Binary { mime, data_base64 } => {
                let mut rendered = String::new();
                rendered.push_str("[binary payload: ");
                rendered.push_str(mime);
                rendered.push_str(", ");
                rendered.push_str(data_base64.len().to_string().as_str());
                rendered.push_str(" base64 chars]");
                Ok(rendered)
            }
        }
    }
}

/// Command result envelope returned to IPC callers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandEnvelope {
    /// Whether command execution succeeded.
    pub ok: bool,
    /// Optional successful command payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<CommandPayload>,
    /// Optional command error text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CommandEnvelope {
    const fn success(payload: CommandPayload) -> Self {
        Self {
            ok: true,
            payload: Some(payload),
            error: None,
        }
    }

    const fn success_empty() -> Self {
        Self {
            ok: true,
            payload: None,
            error: None,
        }
    }

    fn failure(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            payload: None,
            error: Some(message.into()),
        }
    }
}

/// Type-erased tool handler function.
type ToolHandlerFn = Box<
    dyn Fn(
            Vec<String>,
        ) -> Pin<Box<dyn Future<Output = Result<Option<CommandPayload>, String>> + Send>>
        + Send
        + Sync,
>;

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
async fn handle_large_output(output: Option<CommandPayload>, output_dir: &Path) -> CommandEnvelope {
    let Some(output) = output else {
        return CommandEnvelope::success_empty();
    };

    let rendered = match output.render_for_cli() {
        Ok(rendered) => rendered,
        Err(error) => return CommandEnvelope::failure(error),
    };

    if rendered.is_empty() {
        return CommandEnvelope::success_empty();
    }

    if rendered.len() <= INLINE_OUTPUT_LIMIT {
        return CommandEnvelope::success(output);
    }

    // Generate four-random-words filename (consistent with rest of codebase)
    let mut filename = generate_word_filename();
    filename.push_str(".txt");
    let filepath = output_dir.join(&filename);

    let line_count = rendered.lines().count();

    if let Err(error) = async_fs::write(&filepath, &rendered).await {
        let filepath_text = filepath.display().to_string();
        let error_text = error.to_string();
        let mut message = String::with_capacity(filepath_text.len() + error_text.len() + 25);
        message.push_str("failed to write output to ");
        message.push_str(filepath_text.as_str());
        message.push_str(": ");
        message.push_str(error_text.as_str());
        return CommandEnvelope::failure(message);
    }

    CommandEnvelope::success(CommandPayload::Text {
        content: output_saved_text(&rendered, filename.as_str(), line_count, rendered.len()),
    })
}

fn tool_output_to_payload(output: impl IntoToolResult) -> Result<Option<CommandPayload>, String> {
    match output
        .into_tool_result()
        .map_err(|error| prefix_error("failed to convert tool result: ", &error))?
    {
        ToolResult::Done => Ok(None),
        ToolResult::Text { text } | ToolResult::Tsv { text } => {
            Ok(Some(CommandPayload::Text { content: text }))
        }
        ToolResult::Json { value } => Ok(Some(CommandPayload::Json { value })),
        ToolResult::Binary { mime, content } => Ok(Some(CommandPayload::Binary {
            mime,
            data_base64: base64::engine::general_purpose::STANDARD.encode(content),
        })),
        ToolResult::Error { message } => Err(message),
    }
}

// ============================================================================
// Terminal Tool Factory (for creating child terminal tools for subagents)
// ============================================================================

use aither_core::llm::tool::ToolDefinition;

/// A type-erased handler function for terminal tools.
pub type DynToolHandler =
    Arc<dyn Fn(&str) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> + Send + Sync>;

/// One type-erased tool in a terminal capability bundle.
pub struct DynTerminalToolEntry {
    /// Tool definition exposed to the model.
    pub definition: ToolDefinition,
    /// Handler that takes JSON arguments and returns a result.
    pub handler: DynToolHandler,
}

impl std::fmt::Debug for DynTerminalToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTerminalToolEntry")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

/// A type-erased terminal capability bundle for child agents.
pub struct DynTerminalTool {
    pub(crate) entries: Vec<DynTerminalToolEntry>,
    pub(crate) background_receiver: crate::terminal::BackgroundTaskReceiver,
    pub(crate) permission_receiver: crate::terminal::PermissionEventReceiver,
    pub(crate) job_registry: crate::job_registry::JobRegistry,
    pub(crate) working_dir: PathBuf,
}

impl DynTerminalTool {
    /// Returns the receiver for background terminal completions.
    #[must_use]
    pub fn background_receiver(&self) -> crate::terminal::BackgroundTaskReceiver {
        self.background_receiver.clone()
    }

    /// Returns the receiver for terminal permission lifecycle events.
    #[must_use]
    pub fn permission_receiver(&self) -> crate::terminal::PermissionEventReceiver {
        self.permission_receiver.clone()
    }

    /// Returns the child terminal job registry.
    #[must_use]
    pub fn job_registry(&self) -> crate::job_registry::JobRegistry {
        self.job_registry.clone()
    }

    /// Returns the shared terminal working directory.
    #[must_use]
    pub const fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// Consumes the bundle and returns every native terminal tool entry.
    #[must_use]
    pub fn into_entries(self) -> Vec<DynTerminalToolEntry> {
        self.entries
    }
}

impl std::fmt::Debug for DynTerminalTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTerminalTool")
            .field("entries", &self.entries)
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

/// Builder for tool registries.
#[derive(Debug, Default)]
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
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized.
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

        let name_for_errors = tool_name.clone();
        let handler: ToolHandlerFn = Box::new(move |args: Vec<String>| {
            let tool = tool.clone();
            let schema = schema.clone();
            let name = name_for_errors.clone();
            Box::pin(async move {
                tracing::debug!(tool = %name, ?args, "configure_tool handler: raw CLI args");
                let json_args = match cli_to_json(&schema, &args) {
                    Ok(v) => v,
                    Err(e) => {
                        let message = e.to_string();
                        return Err(tool_usage_error(message.as_str(), name.as_str()));
                    }
                };
                tracing::debug!(tool = %name, json = %json_args, "configure_tool handler: parsed JSON");
                let parsed = match serde_json::from_value(json_args) {
                    Ok(v) => v,
                    Err(e) => {
                        let message = prefix_error("invalid arguments: ", &e);
                        return Err(tool_usage_error(message.as_str(), name.as_str()));
                    }
                };
                let output = tool.call(parsed).await.map_err(|error| error.to_string())?;
                tool_output_to_payload(output)
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

    /// Configures a schema-defined handler to be callable from the sandbox.
    ///
    /// This is the dynamic counterpart to [`Self::configure_tool`]. The CLI
    /// wrapper is derived entirely from the provided [`ToolDefinition`]'s JSON
    /// schema, so external tools such as MCP definitions get the same
    /// positional arguments, `--help`, and usage validation behavior as native
    /// `Tool` implementations.
    pub fn configure_definition_handler<F>(&mut self, definition: &ToolDefinition, handler: F)
    where
        F: Fn(
                Value,
            )
                -> Pin<Box<dyn Future<Output = Result<Option<CommandPayload>, String>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let name = definition.name().to_string();
        let description = definition.description().to_string();
        let schema = definition.arguments_openai_schema();
        let positional_args = detect_positional_args(&schema);
        let stdin_arg = detect_stdin_arg(&schema);
        let help_text = if description.is_empty() {
            schema_to_help(&schema)
        } else {
            let mut help = String::with_capacity(description.len() + 2);
            help.push_str(description.as_str());
            help.push_str("\n\n");
            help.push_str(schema_to_help(&schema).as_str());
            help
        };

        let schema_for_handler = schema;
        let name_for_handler = name.clone();
        let handler = Arc::new(handler);
        let raw_handler: ToolHandlerFn = Box::new(move |args: Vec<String>| {
            let schema = schema_for_handler.clone();
            let name = name_for_handler.clone();
            let handler = handler.clone();
            Box::pin(async move {
                let json_args = match cli_to_json(&schema, &args) {
                    Ok(value) => value,
                    Err(error) => {
                        let message = error.to_string();
                        return Err(tool_usage_error(message.as_str(), name.as_str()));
                    }
                };
                handler(json_args).await
            })
        });

        self.entries.insert(
            name,
            ToolEntry {
                handler: raw_handler,
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
        F: Fn(
                Vec<String>,
            )
                -> Pin<Box<dyn Future<Output = Result<Option<CommandPayload>, String>> + Send>>
            + Send
            + Sync
            + 'static,
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
/// Returns required field names that are suitable for positional argument
/// conversion in wrapper scripts. Array-typed fields are excluded because
/// they need repeated `--flag value` syntax (e.g. `--options A --options B`).
fn detect_positional_args(schema: &Value) -> Vec<String> {
    let properties = schema.get("properties").and_then(Value::as_object);

    let required: Vec<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    required
        .into_iter()
        .filter(|name| {
            // Exclude array-typed fields -- they require repeated --flag syntax
            if let Some(props) = properties
                && let Some(prop_schema) = props.get(name)
            {
                return get_instance_type(prop_schema, prop_schema) != Some("array");
            }
            true
        })
        .collect()
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
    pub async fn query_tool_handler(&self, tool_name: &str, args: &[String]) -> CommandEnvelope {
        let entry = self.entries.get(tool_name);
        match entry {
            Some(entry) => match (entry.handler)(args.to_vec()).await {
                Ok(output) => handle_large_output(output, &self.output_dir).await,
                Err(error) => CommandEnvelope::failure(error),
            },
            None => CommandEnvelope::failure(unknown_command_text(tool_name)),
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
/// The `tool_name` comes from the IPC method name, and args are flattened
/// from the key-value pairs sent by `heel ipc`.
#[derive(Debug, Clone)]
pub struct ToolCallCommand {
    /// Shared tool registry.
    pub registry: Arc<ToolRegistry>,
    /// Name of the tool this instance invokes.
    ///
    /// One instance is registered per tool, so the name is fixed at
    /// construction rather than set per request.
    pub tool_name: String,
}

impl ToolCallCommand {
    /// Creates a new tool call command.
    #[must_use]
    pub fn new(tool_name: impl Into<String>, registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            tool_name: tool_name.into(),
        }
    }
}

pub fn flatten_args_to_cli(args: &std::collections::HashMap<String, Value>) -> Vec<String> {
    if let Some(Value::Array(arr)) = args.get("args") {
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
    let mut keys = args.keys().map(String::as_str).collect::<Vec<_>>();
    keys.sort_unstable();
    for key in keys {
        let value = args
            .get(key)
            .expect("flatten_args_to_cli keys must resolve back into the map");
        flatten_named_arg_to_cli(&mut cli_args, key, value);
    }
    cli_args
}

fn push_long_option(cli_args: &mut Vec<String>, key: &str) {
    let mut option = String::with_capacity(key.len() + 2);
    option.push_str("--");
    option.push_str(key);
    cli_args.push(option);
}

fn flatten_named_arg_to_cli(cli_args: &mut Vec<String>, key: &str, value: &Value) {
    match value {
        Value::Bool(true) => push_long_option(cli_args, key),
        Value::Bool(false) => {}
        Value::String(text) => {
            push_long_option(cli_args, key);
            cli_args.push(text.clone());
        }
        Value::Number(number) => {
            push_long_option(cli_args, key);
            cli_args.push(number.to_string());
        }
        Value::Array(items) => {
            for item in items {
                push_long_option(cli_args, key);
                if let Some(text) = item.as_str() {
                    cli_args.push(text.to_string());
                } else {
                    cli_args.push(item.to_string());
                }
            }
        }
        _ => {
            push_long_option(cli_args, key);
            cli_args.push(value.to_string());
        }
    }
}

impl IpcCommand for ToolCallCommand {
    type Args = std::collections::HashMap<String, Value>;
    type Response = CommandEnvelope;

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.tool_name.clone())
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
        self.registry
            .tool_stdin_arg(&self.tool_name)
            .map(Cow::Owned)
    }

    async fn handle(&self, args: Self::Args) -> CommandEnvelope {
        let cli_args = flatten_args_to_cli(&args);
        tracing::info!(tool = %self.tool_name, args = ?cli_args, "ToolCallCommand::handle invoked");

        // Handle help flags
        if has_help_flag(&cli_args) {
            let help = self
                .registry
                .tool_help(&self.tool_name)
                .unwrap_or_else(|| no_help_available_text(self.tool_name.as_str()));
            tracing::info!(tool = %self.tool_name, "Returning help text");
            return CommandEnvelope::success(CommandPayload::Text { content: help });
        }

        tracing::info!(tool = %self.tool_name, "Calling registry.query_tool_handler");
        self.registry
            .query_tool_handler(&self.tool_name, &cli_args)
            .await
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
/// use heel::IpcRouter;
///
/// let router = IpcRouter::new();
/// let registry = std::sync::Arc::new(ToolRegistryBuilder::new().build("./outputs"));
/// let router = register_tool_command(router, registry, "websearch");
/// ```
#[must_use]
pub fn register_tool_command(
    router: heel::IpcRouter,
    registry: Arc<ToolRegistry>,
    tool_name: &str,
) -> heel::IpcRouter {
    router.register(ToolCallCommand::new(tool_name, registry))
}

#[derive(Debug, Clone)]
pub struct IpcGatewayCommand {
    pub registry: Arc<ToolRegistry>,
}

impl IpcGatewayCommand {
    #[must_use]
    pub const fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }
}

impl IpcCommand for IpcGatewayCommand {
    type Args = std::collections::HashMap<String, Value>;
    type Response = CommandEnvelope;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("ipc")
    }

    async fn handle(&self, args: Self::Args) -> CommandEnvelope {
        let cli_args = flatten_args_to_cli(&args);
        if cli_args.is_empty() {
            return CommandEnvelope::failure("usage: ipc <tool> [args ...]");
        }

        if has_help_flag(&cli_args) && cli_args.len() == 1 {
            let mut names = self.registry.registered_tool_names();
            names.sort();
            let joined_names = names.join(", ");
            return CommandEnvelope::success(CommandPayload::Text {
                content: IpcGatewayHelpTemplate {
                    tool_names: joined_names.as_str(),
                }
                .render()
                .expect("ipc gateway help template must render"),
            });
        }

        let tool_name = &cli_args[0];
        let tool_args = cli_args[1..].to_vec();
        self.registry
            .query_tool_handler(tool_name, &tool_args)
            .await
    }
}

/// Registers the generic `ipc` gateway command that dispatches to any tool by name.
#[must_use]
pub fn register_ipc_gateway_command(
    router: heel::IpcRouter,
    registry: Arc<ToolRegistry>,
) -> heel::IpcRouter {
    router.register(IpcGatewayCommand::new(registry))
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
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized.
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
        let payload = tool_output_to_payload(output).map_err(anyhow::Error::msg)?;
        payload.map_or_else(
            || Ok(String::new()),
            |payload| payload.render_for_cli().map_err(anyhow::Error::msg),
        )
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
        }
    }
}

impl<T> IpcToolCommand<T>
where
    T: Tool + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    /// Creates a new IPC command wrapper for the given tool.
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized.
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
        }
    }
}

impl<T> IpcCommand for IpcToolCommand<T>
where
    T: Tool + Clone + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Args = HashMap<String, Value>;
    type Response = CommandEnvelope;

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
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

    async fn handle(&self, args: Self::Args) -> CommandEnvelope {
        let cli_args = flatten_args_to_cli(&args);

        // Handle help flags
        if has_help_flag(&cli_args) {
            return CommandEnvelope::success(CommandPayload::Text {
                content: self.help.clone(),
            });
        }

        // Parse CLI args to JSON
        let json_args = match cli_to_json(&self.schema, &cli_args) {
            Ok(args) => args,
            Err(e) => {
                let message = e.to_string();
                return CommandEnvelope::failure(tool_usage_error(
                    message.as_str(),
                    self.name.as_str(),
                ));
            }
        };

        // Deserialize and call tool
        let parsed: T::Arguments = match serde_json::from_value(json_args) {
            Ok(args) => args,
            Err(e) => {
                let message = prefix_error("invalid arguments: ", &e);
                return CommandEnvelope::failure(tool_usage_error(
                    message.as_str(),
                    self.name.as_str(),
                ));
            }
        };

        match self.tool.call(parsed).await {
            Ok(output) => match tool_output_to_payload(output) {
                Ok(Some(payload)) => CommandEnvelope::success(payload),
                Ok(None) => CommandEnvelope::success_empty(),
                Err(error) => CommandEnvelope::failure(error),
            },
            Err(e) => CommandEnvelope::failure(e.to_string()),
        }
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
/// use heel::IpcRouter;
///
/// let router = IpcRouter::new();
/// let router = register_tool_direct(router, my_tool);
/// ```
pub fn register_tool_direct<T>(router: heel::IpcRouter, tool: T) -> heel::IpcRouter
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
    parse_object(schema, schema, args)
}

/// Finds the serde tag field name from schema extensions.
fn find_serde_tag(schema: &Value) -> Option<String> {
    // Check for discriminator property (OpenAPI style)
    if let Some(disc) = schema.get("discriminator")
        && let Some(prop) = disc.get("propertyName").and_then(Value::as_str)
    {
        return Some(prop.to_string());
    }

    // Check oneOf variants for common const field (serde tag pattern)
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array)
        && let Some(first) = variants.first()
        && let Some(props) = first.get("properties").and_then(Value::as_object)
    {
        for (name, prop) in props {
            // If this property has a const value, it's likely the tag
            if prop.get("const").is_some() || prop.get("enum").is_some() {
                return Some(name.clone());
            }
        }
    }

    None
}

/// Parses a tagged enum from CLI arguments.
fn parse_tagged_enum(
    root_schema: &Value,
    variants: &[Value],
    tag: &str,
    args: &[String],
) -> anyhow::Result<Value> {
    let args = if args.first().map(std::string::String::as_str) == Some("--") {
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
        if let Some(name) = get_variant_name(variant, tag)
            && name.eq_ignore_ascii_case(subcommand)
        {
            let mut variant_schema = variant.clone();
            remove_required_field(&mut variant_schema, tag);
            // Parse the variant's fields
            let mut result = parse_object(root_schema, &variant_schema, remaining)?;

            // Add the tag field
            if let Value::Object(ref mut map) = result {
                map.insert(tag.to_string(), Value::String(name));
            }

            return Ok(result);
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

fn remove_required_field(schema: &mut Value, field: &str) {
    if let Some(required) = schema.get_mut("required").and_then(Value::as_array_mut) {
        required.retain(|entry| entry.as_str() != Some(field));
    }
}

/// Gets the variant name from a schema object.
fn get_variant_name(schema: &Value, tag: &str) -> Option<String> {
    // Look for const value in tag property
    if let Some(props) = schema.get("properties").and_then(Value::as_object)
        && let Some(prop) = props.get(tag)
    {
        if let Some(const_val) = prop.get("const").and_then(Value::as_str) {
            return Some(const_val.to_string());
        }
        // Also check enum with single value
        if let Some(enum_vals) = prop.get("enum").and_then(Value::as_array)
            && enum_vals.len() == 1
        {
            return enum_vals[0].as_str().map(String::from);
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
    let args = if args.first().map(std::string::String::as_str) == Some("--") {
        &args[1..]
    } else {
        args
    };
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
        if arg.starts_with('-') && !arg.starts_with("--") && arg.chars().skip(1).any(|c| c == 'h') {
            return true;
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
fn parse_object(root_schema: &Value, schema: &Value, args: &[String]) -> anyhow::Result<Value> {
    let args = if args.first().map(std::string::String::as_str) == Some("--") {
        &args[1..]
    } else {
        args
    };
    let schema = ObjectParseSchema::new(root_schema, schema);
    let mut state = ObjectParseState::default();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];

        if !state.end_of_options && arg == "--" {
            state.end_of_options = true;
            i += 1;
            continue;
        }

        if !state.end_of_options && arg.starts_with("--") {
            parse_long_option(root_schema, args, &mut i, &schema, &mut state)?;
        } else if !state.end_of_options && arg.starts_with('-') && arg.len() > 1 && arg != "-" {
            parse_short_options(
                root_schema,
                arg,
                args,
                &mut i,
                &schema.properties,
                &schema.short_to_field,
                &mut state.result,
            )?;
        } else {
            parse_positional_arg(root_schema, &schema, &mut state, arg)?;
        }

        i += 1;
    }

    if schema.command_capture_enabled
        && !state.seen_positionals.is_empty()
        && !state.result.contains_key("command")
    {
        state.result.insert(
            "command".to_string(),
            Value::String(state.seen_positionals.join(" ")),
        );
    }

    for req in &schema.required {
        if !state.result.contains_key(req) {
            anyhow::bail!("missing required argument: {req}");
        }
    }

    Ok(Value::Object(state.result.into_iter().collect()))
}

struct ObjectParseSchema {
    properties: serde_json::Map<String, Value>,
    required: Vec<String>,
    positional_fields: Vec<String>,
    command_capture_enabled: bool,
    short_to_field: HashMap<char, String>,
    long_names: Vec<String>,
}

impl ObjectParseSchema {
    fn new(root_schema: &Value, schema: &Value) -> Self {
        let resolved = resolved_schema(root_schema, schema);
        let properties = resolved
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let required: Vec<String> = resolved
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        let positional_fields = positional_fields(root_schema, &properties, &required);
        let command_capture_enabled = properties
            .get("command")
            .is_some_and(|schema| get_instance_type(root_schema, schema) == Some("string"));
        let (short_to_field, _) = build_short_option_maps(&properties);
        let long_names = properties.keys().map(|k| k.replace('_', "-")).collect();
        Self {
            properties,
            required,
            positional_fields,
            command_capture_enabled,
            short_to_field,
            long_names,
        }
    }
}

#[derive(Default)]
struct ObjectParseState {
    result: HashMap<String, Value>,
    positional_idx: usize,
    seen_positionals: Vec<String>,
    end_of_options: bool,
}

fn positional_fields(
    root_schema: &Value,
    properties: &serde_json::Map<String, Value>,
    required: &[String],
) -> Vec<String> {
    required
        .iter()
        .filter(|name| {
            properties.get(name.as_str()).is_none_or(|prop_schema| {
                get_instance_type(root_schema, prop_schema) != Some("array")
            })
        })
        .cloned()
        .collect()
}

fn parse_positional_arg(
    root_schema: &Value,
    schema: &ObjectParseSchema,
    state: &mut ObjectParseState,
    arg: &str,
) -> anyhow::Result<()> {
    if state.positional_idx < schema.positional_fields.len() {
        let field_name = &schema.positional_fields[state.positional_idx];
        if let Some(prop_schema) = schema.properties.get(field_name) {
            let prop_type = get_instance_type(root_schema, prop_schema);
            state
                .result
                .insert(field_name.clone(), parse_value(arg, prop_type));
        }
        state.positional_idx += 1;
        state.seen_positionals.push(arg.to_string());
    } else if schema.command_capture_enabled {
        state.seen_positionals.push(arg.to_string());
    } else {
        anyhow::bail!("unexpected positional argument: {arg}");
    }
    Ok(())
}

fn parse_long_option(
    root_schema: &Value,
    args: &[String],
    i: &mut usize,
    schema: &ObjectParseSchema,
    state: &mut ObjectParseState,
) -> anyhow::Result<()> {
    let arg = &args[*i];
    let flag = &arg[2..];
    let (mut name, value) = flag.find('=').map_or_else(
        || (flag.to_string(), None),
        |eq_pos| {
            (
                flag[..eq_pos].to_string(),
                Some(flag[eq_pos + 1..].to_string()),
            )
        },
    );
    name = name.replace('_', "-");
    let (normalized_name, negated) = name.strip_prefix("no-").map_or_else(
        || (name.clone(), false),
        |stripped| (stripped.to_string(), true),
    );
    name = normalized_name;
    let field_name = name.replace('-', "_");
    let Some(prop_schema) = schema.properties.get(&field_name) else {
        return bail_unknown_long_option(&name, &schema.long_names);
    };
    let parsed_value =
        parse_long_option_value(root_schema, args, i, &name, value, negated, prop_schema)?;
    insert_value(
        root_schema,
        &mut state.result,
        &field_name,
        parsed_value,
        prop_schema,
    );
    Ok(())
}

fn parse_long_option_value(
    root_schema: &Value,
    args: &[String],
    i: &mut usize,
    name: &str,
    value: Option<String>,
    negated: bool,
    prop_schema: &Value,
) -> anyhow::Result<Value> {
    let prop_type = get_instance_type(root_schema, prop_schema);
    if negated && prop_type != Some("boolean") {
        anyhow::bail!("--no-{name} is only valid for boolean options");
    }
    if prop_type == Some("boolean") {
        if negated && value.is_some() {
            anyhow::bail!("unexpected value for --no-{name}");
        }
        return Ok(value.map_or(Value::Bool(!negated), |v| parse_value(&v, prop_type)));
    }
    let val = value.or_else(|| {
        *i += 1;
        args.get(*i).cloned()
    });
    val.map_or_else(
        || anyhow::bail!("missing value for --{name}"),
        |v| Ok(parse_value(&v, prop_type)),
    )
}

fn bail_unknown_long_option(name: &str, long_names: &[String]) -> anyhow::Result<()> {
    let suggestion = find_similar_option(name, long_names.iter().map(String::as_str));
    if let Some(similar) = suggestion {
        anyhow::bail!("unknown option: --{name}. Did you mean --{similar}?");
    }
    anyhow::bail!("unknown option: --{name}");
}

fn parse_short_options(
    root_schema: &Value,
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
        let prop_type = get_instance_type(root_schema, prop_schema);

        if prop_type == Some("boolean") {
            if chars.peek().is_none() {
                if let Some(v) = value_from_eq.take() {
                    let parsed = parse_value(&v, prop_type);
                    insert_value(root_schema, result, &field_name, parsed, prop_schema);
                } else {
                    insert_value(
                        root_schema,
                        result,
                        &field_name,
                        Value::Bool(true),
                        prop_schema,
                    );
                }
            } else {
                if value_from_eq.is_some() {
                    anyhow::bail!("unexpected value for -{ch}");
                }
                insert_value(
                    root_schema,
                    result,
                    &field_name,
                    Value::Bool(true),
                    prop_schema,
                );
            }
            continue;
        }

        let value = if let Some(v) = value_from_eq.take() {
            if chars.peek().is_some() {
                anyhow::bail!("unexpected value for -{ch}");
            }
            v
        } else if chars.peek().is_some() {
            let rest: String = chars.collect();
            rest
        } else {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for -{ch}"))?
        };

        let parsed = parse_value(&value, prop_type);
        insert_value(root_schema, result, &field_name, parsed, prop_schema);
        break;
    }

    Ok(())
}

fn insert_value(
    root_schema: &Value,
    result: &mut HashMap<String, Value>,
    field_name: &str,
    value: Value,
    prop_schema: &Value,
) {
    if get_instance_type(root_schema, prop_schema) == Some("array") {
        if let Some(Value::Array(existing)) = result.get_mut(field_name) {
            match value {
                Value::Array(mut items) => existing.append(&mut items),
                other => existing.push(other),
            }
        } else {
            let mut items = Vec::new();
            match value {
                Value::Array(mut arr) => items.append(&mut arr),
                other => items.push(other),
            }
            result.insert(field_name.to_string(), Value::Array(items));
        }
    } else {
        result.insert(field_name.to_string(), value);
    }
}

/// Gets the instance type from a schema.
/// Handles both simple types and Option<T>.
fn get_instance_type<'a>(root_schema: &'a Value, schema: &'a Value) -> Option<&'a str> {
    let schema = resolved_schema(root_schema, schema);
    match schema.get("type") {
        // Direct string type: "type": "integer"
        Some(Value::String(t)) => Some(t.as_str()),

        // Array of types (schemars 1.0 for Option<T>): "type": ["integer", "null"]
        Some(Value::Array(types)) => {
            // Find the non-null type
            for t in types {
                if let Some(s) = t.as_str()
                    && s != "null"
                {
                    return Some(s);
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
                if let Some(t) = variant.get("type").and_then(Value::as_str)
                    && t != "null"
                {
                    return Some(t);
                }
            }
            None
        }

        _ => None,
    }
}

fn resolved_schema<'a>(root_schema: &'a Value, schema: &'a Value) -> &'a Value {
    let mut current = schema;
    while let Some(reference) = current.get("$ref").and_then(Value::as_str) {
        let Some(next) = resolve_local_ref(root_schema, reference) else {
            break;
        };
        if std::ptr::eq(current, next) {
            break;
        }
        current = next;
    }
    current
}

fn resolve_local_ref<'a>(root_schema: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix("#/")?;
    let mut current = root_schema;
    for segment in pointer.split('/') {
        let segment = segment.replace("~1", "/").replace("~0", "~");
        current = current.get(segment.as_str())?;
    }
    Some(current)
}

/// Parses a string value according to expected type.
fn parse_value(s: &str, expected_type: Option<&str>) -> Value {
    match expected_type {
        Some("integer") => s
            .parse::<i64>()
            .map_or_else(|_| Value::String(s.to_string()), Value::from),
        Some("number") => s
            .parse::<f64>()
            .map_or_else(|_| Value::String(s.to_string()), Value::from),
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
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array)
        && let Some(tag) = find_serde_tag(schema)
    {
        help.push_str("  <subcommand> [options]\n\n");
        help.push_str("Subcommands:\n");

        for variant in variants {
            if let Some(name) = get_variant_name(variant, &tag) {
                help.push_str("  ");
                help.push_str(name.as_str());
                if let Some(desc) = variant.get("description").and_then(Value::as_str) {
                    help.push_str(" - ");
                    help.push_str(desc);
                }
                help.push('\n');
            }
        }
        help.push_str("\nOptions:\n  -h, --help  Show help\n");
        return help;
    }

    // Simple object
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        // Show positional usage: only non-array required fields are positional
        let positional: Vec<_> = required
            .iter()
            .filter(|n| {
                props
                    .get(**n)
                    .is_none_or(|s| get_instance_type(s, s) != Some("array"))
            })
            .map(|n| render_positional_usage(n))
            .collect();
        // Show required array fields as repeatable flags in usage
        let repeatable: Vec<_> = required
            .iter()
            .filter(|n| {
                props
                    .get(**n)
                    .is_some_and(|s| get_instance_type(s, s) == Some("array"))
            })
            .map(|n| render_repeatable_usage(n))
            .collect();

        let mut usage_parts = Vec::new();
        if !positional.is_empty() {
            usage_parts.push(positional.join(" "));
        }
        for r in &repeatable {
            usage_parts.push(r.clone());
        }
        usage_parts.push("[options]".to_string());
        let usage = usage_parts.join(" ");
        help.push_str("  ");
        push_help_line(&mut help, usage.as_str());

        help.push_str("\nOptions:\n  -h, --help  Show help\n");
        help.push_str("\nArguments:\n");

        let (_, field_to_short) = build_short_option_maps(props);

        for (name, prop) in props {
            let is_required = required.contains(&name.as_str());
            let is_array = get_instance_type(prop, prop) == Some("array");
            let flag = name.replace('_', "-");

            let short = field_to_short.get(name).copied();
            help.push_str(render_flag_prefix(flag.as_str(), short).as_str());
            if is_array {
                help.push_str(" <value>  (repeatable");
                if is_required {
                    help.push_str(", required");
                }
                help.push(')');
            } else if is_required {
                help.push_str(" (required)");
            }

            if let Some(desc) = prop.get("description").and_then(Value::as_str) {
                help.push_str("\n      ");
                help.push_str(desc);
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
    fn test_leading_separator_stripped() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Leading "--" from `heel ipc` is stripped; flags still work after it
        let result = cli_to_json(
            &schema,
            &[
                "--".to_string(),
                "--path".to_string(),
                "foo.txt".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(result["path"], "foo.txt");
    }

    #[test]
    fn test_end_of_options_after_separator() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Double "--": first is the `heel ipc` separator, second is end-of-options
        let result = cli_to_json(
            &schema,
            &["--".to_string(), "--".to_string(), "-dash.txt".to_string()],
        )
        .unwrap();
        assert_eq!(result["path"], "-dash.txt");
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TwoPositionalArgs {
        /// First required arg
        first: String,
        /// Second required arg
        second: String,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(tag = "operation", rename_all = "snake_case")]
    enum TaggedOperationArgs {
        Read { path: String },
        Write { path: String, content: String },
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum NestedTriggerArgs {
        Webhook,
        Event { event_name: String },
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct NestedObjectArgs {
        name: String,
        trigger: NestedTriggerArgs,
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
    fn test_tagged_enum_subcommand_parsing() {
        let schema = schemars::schema_for!(TaggedOperationArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(
            &schema,
            &[
                "read".to_string(),
                "--path".to_string(),
                "foo.txt".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(result["operation"], "read");
        assert_eq!(result["path"], "foo.txt");

        let err = cli_to_json(
            &schema,
            &[
                "write".to_string(),
                "--path".to_string(),
                "foo.txt".to_string(),
            ],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required argument: content"),
            "got: {err}"
        );
    }

    #[test]
    fn test_nested_ref_object_flag_parsing() {
        let schema = schemars::schema_for!(NestedObjectArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(
            &schema,
            &[
                "--name".to_string(),
                "demo".to_string(),
                "--trigger".to_string(),
                "{\"kind\":\"webhook\"}".to_string(),
            ],
        )
        .unwrap();

        assert_eq!(result["name"], "demo");
        assert_eq!(result["trigger"]["kind"], "webhook");
    }

    #[test]
    fn test_typo_suggestion() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Typo in flag name -- deliberate, it is what the suggestion is for.
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
        // Simulate IPC params (key-value pairs from `heel ipc`)
        let json = r#"{"query": "rust async", "max": 5}"#;
        let tmp = tempfile::tempdir().unwrap();
        let registry = ToolRegistryBuilder::new().build(tmp.path());
        let cmd = ToolCallCommand::new("websearch", Arc::new(registry));
        let args: std::collections::HashMap<String, Value> = serde_json::from_str(json).unwrap();

        // the tool name rides on the command, the wire carries only arguments
        assert_eq!(cmd.name(), "websearch");
        assert_eq!(args.len(), 2);
        assert_eq!(args.get("query").unwrap(), "rust async");
        assert_eq!(args.get("max").unwrap(), 5);

        let cli_args = flatten_args_to_cli(&args);
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

    // ========================================================================
    // Array / repeated flag tests
    // ========================================================================

    #[derive(Debug, Deserialize, JsonSchema)]
    struct AskUserLikeArgs {
        /// The question to ask
        question: String,
        /// Options to choose from
        options: Vec<String>,
        /// Allow multiple selections
        #[serde(default)]
        multi_select: bool,
    }

    fn touch_test_arg_types() {
        let simple = SimpleArgs {
            path: "file.txt".to_string(),
            count: Some(1),
        };
        assert_eq!(simple.path, "file.txt");
        assert_eq!(simple.count, Some(1));

        let cluster = ClusterArgs {
            verbose: true,
            output: "out.txt".to_string(),
        };
        assert!(cluster.verbose);
        assert_eq!(cluster.output, "out.txt");

        let two = TwoPositionalArgs {
            first: "a".to_string(),
            second: "b".to_string(),
        };
        assert_eq!((two.first.as_str(), two.second.as_str()), ("a", "b"));

        let read = TaggedOperationArgs::Read {
            path: "input.txt".to_string(),
        };
        let TaggedOperationArgs::Read { path } = read else {
            panic!("expected read operation");
        };
        assert_eq!(path, "input.txt");

        let write = TaggedOperationArgs::Write {
            path: "output.txt".to_string(),
            content: "body".to_string(),
        };
        let TaggedOperationArgs::Write { path, content } = write else {
            panic!("expected write operation");
        };
        assert_eq!((path.as_str(), content.as_str()), ("output.txt", "body"));

        let trigger = NestedTriggerArgs::Event {
            event_name: "ready".to_string(),
        };
        let NestedTriggerArgs::Event { event_name } = trigger else {
            panic!("expected event trigger");
        };
        assert_eq!(event_name, "ready");

        let nested = NestedObjectArgs {
            name: "job".to_string(),
            trigger: NestedTriggerArgs::Webhook,
        };
        assert_eq!(nested.name, "job");
        assert!(matches!(nested.trigger, NestedTriggerArgs::Webhook));

        let ask = AskUserLikeArgs {
            question: "Continue?".to_string(),
            options: vec!["yes".to_string()],
            multi_select: false,
        };
        assert_eq!(ask.question, "Continue?");
        assert_eq!(ask.options, ["yes"]);
        assert!(!ask.multi_select);
    }

    #[test]
    fn test_repeated_flag_for_array() {
        touch_test_arg_types();
        let schema = schemars::schema_for!(AskUserLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(
            &schema,
            &[
                "--question".into(),
                "Pick one".into(),
                "--options".into(),
                "A".into(),
                "--options".into(),
                "B".into(),
                "--options".into(),
                "C".into(),
            ],
        )
        .unwrap();

        assert_eq!(result["question"], "Pick one");
        assert_eq!(result["options"], serde_json::json!(["A", "B", "C"]));
    }

    #[test]
    fn test_positional_then_repeated_flag() {
        let schema = schemars::schema_for!(AskUserLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // question as positional, options as repeated flags
        let result = cli_to_json(
            &schema,
            &[
                "Pick one".into(),
                "--options".into(),
                "A".into(),
                "--options".into(),
                "B".into(),
            ],
        )
        .unwrap();

        assert_eq!(result["question"], "Pick one");
        assert_eq!(result["options"], serde_json::json!(["A", "B"]));
    }

    #[test]
    fn test_array_not_positional() {
        let schema = schemars::schema_for!(AskUserLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Only question should be positional, not options
        let positional = detect_positional_args(&schema);
        assert_eq!(positional, vec!["question"]);
    }

    #[test]
    fn test_missing_required_array() {
        let schema = schemars::schema_for!(AskUserLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Provide question but no options -> should fail with clear message
        let err = cli_to_json(&schema, &["Pick one".into()]).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required argument: options"),
            "got: {err}"
        );
    }

    #[test]
    fn test_boolean_flag_with_array() {
        let schema = schemars::schema_for!(AskUserLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let result = cli_to_json(
            &schema,
            &[
                "--question".into(),
                "Pick many".into(),
                "--options".into(),
                "X".into(),
                "--options".into(),
                "Y".into(),
                "--multi-select".into(),
            ],
        )
        .unwrap();

        assert_eq!(result["question"], "Pick many");
        assert_eq!(result["options"], serde_json::json!(["X", "Y"]));
        assert_eq!(result["multi_select"], true);
    }

    // ========================================================================
    // Help text tests
    // ========================================================================

    #[test]
    fn test_help_shows_array_as_repeatable() {
        let schema = schemars::schema_for!(AskUserLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();
        let help = schema_to_help(&schema);

        // Usage line should show options as repeatable flag, not positional
        assert!(
            help.contains("--options <value>..."),
            "usage should show --options as repeatable, got:\n{help}"
        );
        // question should be positional
        assert!(
            help.contains("<question>"),
            "usage should show <question> as positional, got:\n{help}"
        );
        // Arguments section should mark options as repeatable
        assert!(
            help.contains("repeatable"),
            "options should be marked repeatable, got:\n{help}"
        );
    }

    // ========================================================================
    // Error message tests
    // ========================================================================

    #[test]
    fn test_unknown_option_error() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let err = cli_to_json(&schema, &["--unknown".into(), "val".into()]).unwrap_err();
        assert!(err.to_string().contains("unknown option: --unknown"));
    }

    #[test]
    fn test_unexpected_positional_error() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        // Two positional args when only one is expected
        let err = cli_to_json(&schema, &["a.txt".into(), "extra".into()]).unwrap_err();
        assert!(err.to_string().contains("unexpected positional argument"));
    }

    #[test]
    fn test_missing_value_error() {
        let schema = schemars::schema_for!(SimpleArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let err = cli_to_json(&schema, &["--path".into()]).unwrap_err();
        // path is consumed as positional, so we get missing required
        // Actually --path starts with --, so it enters the named path and needs a value
        assert!(err.to_string().contains("missing"), "got: {err}");
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SessionLikeArgs {
        query: String,
        #[serde(default)]
        command: Option<String>,
    }

    #[test]
    fn test_command_field_captures_single_positional_for_session_like_tools() {
        let schema = schemars::schema_for!(SessionLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let value = cli_to_json(&schema, &["gold".into()]).unwrap();
        assert_eq!(value["query"], "gold");
        assert_eq!(value["command"], "gold");
    }

    #[test]
    fn test_command_field_captures_multi_positional_for_session_like_tools() {
        let schema = schemars::schema_for!(SessionLikeArgs);
        let schema = serde_json::to_value(schema).unwrap();

        let value = cli_to_json(&schema, &["abc123".into(), "search".into(), "gold".into()])
            .expect("session-like positional command should parse");
        assert_eq!(value["query"], "abc123");
        assert_eq!(value["command"], "abc123 search gold");
    }

    // ========================================================================
    // ToolCallCommand args_to_cli with arrays
    // ========================================================================

    #[test]
    fn test_tool_call_command_args_to_cli_with_array() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = ToolRegistryBuilder::new().build(tmp.path());
        let _cmd = ToolCallCommand::new("ask_user", Arc::new(registry));
        let args: std::collections::HashMap<String, Value> =
            serde_json::from_value(serde_json::json!({
                "question": "Pick one",
                "options": ["A", "B", "C"]
            }))
            .unwrap();

        let cli = flatten_args_to_cli(&args);
        // Should produce repeated --options flags
        let options_count = cli.iter().filter(|a| *a == "--options").count();
        assert_eq!(options_count, 3, "expected 3 --options flags, got: {cli:?}");
        assert!(cli.contains(&"A".to_string()));
        assert!(cli.contains(&"B".to_string()));
        assert!(cli.contains(&"C".to_string()));
    }

    #[test]
    fn test_tool_call_command_raw_args_passthrough() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = ToolRegistryBuilder::new().build(tmp.path());
        let _cmd = ToolCallCommand::new("ask_user", Arc::new(registry));
        // Simulates what `heel ipc` sends
        let args: std::collections::HashMap<String, Value> =
            serde_json::from_value(serde_json::json!({
                "args": ["--question", "Pick one", "--options", "A", "--options", "B"]
            }))
            .unwrap();

        let cli = flatten_args_to_cli(&args);
        assert_eq!(
            cli,
            vec!["--question", "Pick one", "--options", "A", "--options", "B"]
        );
    }

    #[test]
    fn test_flatten_args_to_cli_sorts_structured_keys_deterministically() {
        let args = serde_json::from_value::<HashMap<String, Value>>(serde_json::json!({
            "zebra": "last",
            "alpha": "first",
            "multi": ["x", "y"],
            "enabled": true
        }))
        .expect("structured args should deserialize");

        let cli = flatten_args_to_cli(&args);
        assert_eq!(
            cli,
            vec![
                "--alpha",
                "first",
                "--enabled",
                "--multi",
                "x",
                "--multi",
                "y",
                "--zebra",
                "last"
            ]
        );
    }

    #[tokio::test]
    async fn test_handle_large_output_write_failure_returns_error_message() {
        let tmp = tempfile::tempdir().unwrap();
        let output_dir = tmp.path().join("not-a-directory");
        std::fs::write(&output_dir, "occupied").unwrap();
        let output = Some(CommandPayload::Text {
            content: "x".repeat(INLINE_OUTPUT_LIMIT + 1),
        });

        let result = handle_large_output(output, &output_dir).await;
        assert!(!result.ok);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("failed to write output")
        );
    }

    #[tokio::test]
    async fn test_ask_user_e2e_repeated_option() {
        use aither_core::llm::{Tool, ToolResult};
        use std::borrow::Cow;

        #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
        struct Question {
            section: String,
            #[serde(rename = "question")]
            prompt: String,
            option: Vec<String>,
            #[serde(default)]
            multi_select: bool,
        }

        #[derive(Debug, Clone, Deserialize, JsonSchema)]
        struct AskUserArgs {
            #[serde(default)]
            question: Option<String>,
            #[serde(default, alias = "options", alias = "choices")]
            option: Vec<String>,
            #[serde(default)]
            multi_select: bool,
            #[serde(default)]
            questions: Vec<Question>,
        }

        #[derive(Debug, Clone)]
        struct FakeAskUser;
        impl Tool for FakeAskUser {
            fn name(&self) -> Cow<'static, str> {
                "ask_user".into()
            }
            type Arguments = AskUserArgs;
            type Res = ToolResult;

            fn call(
                &self,
                args: Self::Arguments,
            ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send
            {
                // Return the parsed args as JSON so we can inspect
                std::future::ready(ToolResult::json(&serde_json::json!({
                    "question": args.question,
                    "option": args.option,
                    "multi_select": args.multi_select,
                    "questions": args.questions,
                })))
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(FakeAskUser);
        let registry = std::sync::Arc::new(builder.build(tmp.path()));

        // Simulate what `heel ipc` sends: all args in an "args" array
        let cmd = ToolCallCommand::new("ask_user", registry);
        let args = serde_json::from_value(serde_json::json!({
            "args": [
                "--", "--question", "你喜欢哪种薯条？",
                "--option", "原味", "--option", "番茄味", "--option", "芝士味"
            ]
        }))
        .unwrap();

        let result = cmd.handle(args).await;
        tracing::debug!("result: {}", serde_json::to_string_pretty(&result).unwrap());
        assert!(result.ok, "command should succeed: {result:?}");

        let Some(CommandPayload::Json { value: parsed }) = result.payload else {
            panic!("expected JSON payload, got {result:?}");
        };
        let options = parsed["option"].as_array().expect("option should be array");
        assert_eq!(options.len(), 3, "expected 3 options, got: {options:?}");
        assert_eq!(options[0], "原味");
        assert_eq!(options[1], "番茄味");
        assert_eq!(options[2], "芝士味");
    }
}
