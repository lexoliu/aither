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

use aither_core::llm::tool::ToolResultPart;
use aither_core::llm::{IntoToolResult, Tool, ToolContext, ToolResult};
use askama::Template;
use heel::IpcCommand;
use heel::ipc::IpcError;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Tool Registry
// ============================================================================

/// Payload an IPC command returns over the wire.
///
/// Binary results never reach it: [`tool_output_to_payload`] writes them to
/// the output directory and the payload names the stored file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommandPayload {
    /// UTF-8 text output.
    Text {
        /// Text content.
        content: String,
    },
}

#[derive(Template)]
#[template(path = "stored_binary_part.txt", escape = "none")]
struct StoredBinaryPartTemplate<'a> {
    mime: &'a str,
    byte_count: usize,
    url: &'a str,
    path: &'a str,
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
    let preview = crate::output_compress::stored_preview(rendered, usize::MAX, INLINE_OUTPUT_LIMIT);
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
    /// The text a terminal caller reads for this payload.
    #[must_use]
    pub fn render_for_cli(&self) -> String {
        let Self::Text { content } = self;
        content.clone()
    }
}

/// Writes a binary payload into `output_dir` and renders the line that
/// references the stored file, the way terminal output references stored
/// images.
async fn stored_binary_text(
    mime: &str,
    content: &[u8],
    output_dir: &Path,
) -> Result<String, String> {
    let (url, path) = crate::output::save_binary_to_file(output_dir, content, mime)
        .await
        .map_err(|error| prefix_error("failed to store binary tool result: ", &error))?;
    Ok(StoredBinaryPartTemplate {
        mime,
        byte_count: content.len(),
        url: &url,
        path: &path.display().to_string(),
    }
    .render()
    .expect("stored binary part template must render"))
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
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send>>
        + Send
        + Sync,
>;

/// Tool entry in the registry.
struct ToolEntry {
    handler: ToolHandlerFn,
    help: String,
    stdin_arg: Option<String>,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("help", &self.help)
            .field("stdin_arg", &self.stdin_arg)
            .finish_non_exhaustive()
    }
}

/// Which side of the authority boundary a command registry serves.
///
/// The main agent's terminal serves [`CommandRole::Main`]; every terminal built
/// for a subagent, at any nesting depth, serves [`CommandRole::Subagent`], so a
/// subagent can never reach a handler reserved for the main agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandRole {
    /// The main agent.
    Main,
    /// A subagent, or a subagent of one.
    Subagent,
}

/// What a command resolves to in each role; a subagent role without an entry
/// gets the refusal naming the command as reserved for the main agent.
#[derive(Debug)]
struct RoleEntries<T> {
    main: T,
    subagent: Option<T>,
}

impl<T: Clone> Clone for RoleEntries<T> {
    fn clone(&self) -> Self {
        Self {
            main: self.main.clone(),
            subagent: self.subagent.clone(),
        }
    }
}

impl<T> RoleEntries<T> {
    /// One entry serving both roles.
    fn shared(entry: T) -> Self
    where
        T: Clone,
    {
        Self {
            main: entry.clone(),
            subagent: Some(entry),
        }
    }

    /// The entry `role` reaches; `None` when the command is reserved for the
    /// main agent.
    const fn for_role(&self, role: CommandRole) -> Option<&T> {
        match role {
            CommandRole::Main => Some(&self.main),
            CommandRole::Subagent => self.subagent.as_ref(),
        }
    }
}

/// One command call a registry served: what it resolved to, what it was
/// given and what it answered.
#[derive(Debug, Clone, Copy)]
pub struct CommandCall<'a> {
    /// The role of the registry that served the call.
    pub role: CommandRole,
    /// The resolved command name.
    pub name: &'a str,
    /// The arguments the command was called with.
    pub args: &'a [String],
    /// What the call answered.
    pub outcome: &'a CommandEnvelope,
}

/// Watches the commands a registry serves.
///
/// The registry awaits the observer after the command has answered and before
/// the answer returns to the caller, so an observer delays the call by as
/// long as its future runs; it cannot change the answer. An observer whose
/// work is slow starts it on an executor and returns at once.
pub trait CommandObserver: Send + Sync + 'static {
    /// Called once for every served call of a resolved command.
    fn command_called(&self, call: &CommandCall<'_>) -> impl Future<Output = ()> + Send;
}

/// What a [`CommandGate`] decided about one shell command.
///
/// There is no grant to give: an allowed command runs under the permissions
/// it would have had without a gate, and the permission path still decides
/// anything beyond the sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// The command may go on to the permission path and run.
    Allow,
    /// The command does not run; `reason` reaches the model as its result.
    Block {
        /// Why the command was refused.
        reason: String,
    },
}

/// A command the gate blocked, as the refusal the model reads in place of
/// the command's result.
#[derive(Debug)]
pub struct CommandBlocked {
    pub reason: String,
}

#[derive(Template)]
#[template(path = "command_blocked.txt")]
struct CommandBlockedTemplate<'a> {
    reason: &'a str,
}

impl std::fmt::Display for CommandBlocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rendered = CommandBlockedTemplate {
            reason: &self.reason,
        }
        .render()
        .map_err(|_| std::fmt::Error)?;
        f.write_str(&rendered)
    }
}

impl std::error::Error for CommandBlocked {}

/// Decides whether a shell command may run, before it does.
///
/// Every command of the `terminal` and `terminal_background` tools, the
/// main agent's and every subagent's, is put to the gate of its terminal's
/// registry in full, ahead of the permission path.
pub trait CommandGate: Send + Sync + 'static {
    /// Decides on `command`, the full script text.
    fn check(&self, command: &str) -> impl Future<Output = GateDecision> + Send;
}

/// Object-safe twin of [`CommandGate`].
trait CommandGateImpl: Send + Sync {
    fn check<'a>(
        &'a self,
        command: &'a str,
    ) -> Pin<Box<dyn Future<Output = GateDecision> + Send + 'a>>;

    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: CommandGate> CommandGateImpl for T {
    fn check<'a>(
        &'a self,
        command: &'a str,
    ) -> Pin<Box<dyn Future<Output = GateDecision> + Send + 'a>> {
        Box::pin(CommandGate::check(self, command))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A type-erased [`CommandGate`].
#[derive(Clone)]
pub struct AnyCommandGate(Arc<dyn CommandGateImpl>);

impl AnyCommandGate {
    /// Erases `gate`.
    #[must_use]
    pub fn new(gate: impl CommandGate) -> Self {
        Self(Arc::new(gate))
    }

    /// The wrapped gate, when it is a `T`.
    #[must_use]
    pub fn downcast_ref<T: CommandGate>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref()
    }

    /// Puts `command` to the gate.
    pub async fn check(&self, command: &str) -> GateDecision {
        self.0.check(command).await
    }
}

impl std::fmt::Debug for AnyCommandGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyCommandGate")
    }
}

/// Object-safe twin of [`CommandObserver`].
trait CommandObserverImpl: Send + Sync {
    fn command_called<'a>(
        &'a self,
        call: &'a CommandCall<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: CommandObserver> CommandObserverImpl for T {
    fn command_called<'a>(
        &'a self,
        call: &'a CommandCall<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(CommandObserver::command_called(self, call))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A type-erased [`CommandObserver`].
#[derive(Clone)]
pub struct AnyCommandObserver(Arc<dyn CommandObserverImpl>);

impl AnyCommandObserver {
    /// Erases `observer`.
    #[must_use]
    pub fn new(observer: impl CommandObserver) -> Self {
        Self(Arc::new(observer))
    }

    /// The wrapped observer, when it is a `T`; how a host that installed an
    /// observer finds it again through a terminal's binding.
    #[must_use]
    pub fn downcast_ref<T: CommandObserver>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref()
    }

    async fn command_called(&self, call: &CommandCall<'_>) {
        self.0.command_called(call).await;
    }
}

impl std::fmt::Debug for AnyCommandObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyCommandObserver")
    }
}

/// Builds a command's handler for the terminal that serves it.
type BindHandlerFn = Box<dyn Fn(&TerminalBinding) -> ToolHandlerFn + Send + Sync>;

/// A command whose handler is built per terminal, from the terminal's
/// [`TerminalBinding`].
struct BoundEntry {
    bind: BindHandlerFn,
    help: String,
    stdin_arg: Option<String>,
}

impl std::fmt::Debug for BoundEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundEntry")
            .field("help", &self.help)
            .field("stdin_arg", &self.stdin_arg)
            .finish_non_exhaustive()
    }
}

impl BoundEntry {
    fn instantiate(&self, binding: &TerminalBinding) -> ToolEntry {
        ToolEntry {
            handler: (self.bind)(binding),
            help: self.help.clone(),
            stdin_arg: self.stdin_arg.clone(),
        }
    }
}

/// What a command name resolves to in one role.
struct Resolved<'a> {
    /// The handler; `None` when the command is reserved for the main agent.
    handler: Option<&'a ToolHandlerFn>,
    help: &'a str,
    stdin_arg: Option<&'a str>,
}

#[derive(Template)]
#[template(path = "reserved_for_main.txt", escape = "none")]
struct ReservedForMainTemplate<'a> {
    name: &'a str,
}

fn reserved_for_main_text(name: &str) -> String {
    ReservedForMainTemplate { name }
        .render()
        .expect("reserved-for-main template must render")
}

use crate::output::{INLINE_OUTPUT_LIMIT, generate_word_filename};
use crate::terminal::TerminalBinding;

/// Handles large outputs by saving to file in the sandbox output directory.
///
/// Returns the output as-is if small enough, or saves to file and returns
/// a reference message if the output exceeds the limit.
async fn handle_large_output(output: ToolResult, output_dir: &Path) -> CommandEnvelope {
    let output = match tool_output_to_payload(output, output_dir).await {
        Ok(Some(output)) => output,
        Ok(None) => return CommandEnvelope::success_empty(),
        Err(error) => return CommandEnvelope::failure(error),
    };

    let rendered = output.render_for_cli();

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

/// Converts a finished tool result into a command payload for IPC dispatch.
///
/// Everything becomes text for the model: text-like results pass through, a
/// command result renders its prose, and every binary payload, a lone one
/// included, is written to `output_dir` and replaced by a line naming the
/// stored file. A typed tool error becomes the command's error.
///
/// # Errors
///
/// Returns an error when the conversion fails, a binary payload cannot be
/// stored, or the result is a typed tool error (surfaced as the command
/// error).
pub async fn tool_output_to_payload(
    output: impl IntoToolResult,
    output_dir: &Path,
) -> Result<Option<CommandPayload>, String> {
    let text = match output
        .into_tool_result()
        .map_err(|error| prefix_error("failed to convert tool result: ", &error))?
    {
        ToolResult::Done => return Ok(None),
        ToolResult::Error { message } => return Err(message),
        ToolResult::Binary { mime, content } => {
            stored_binary_text(&mime, &content, output_dir).await?
        }
        ToolResult::Parts { parts } => {
            let mut rendered = String::new();
            for part in &parts {
                let text = match part {
                    ToolResultPart::Binary { mime, content } => {
                        stored_binary_text(mime, content, output_dir).await?
                    }
                    ToolResultPart::Text { .. } | ToolResultPart::Tsv { .. } => {
                        part.render_for_model()
                    }
                };
                if text.is_empty() {
                    continue;
                }
                if !rendered.is_empty() {
                    rendered.push('\n');
                }
                rendered.push_str(&text);
            }
            rendered
        }
        result @ (ToolResult::Text { .. } | ToolResult::Tsv { .. } | ToolResult::Command(_)) => {
            result.render_for_model()
        }
    };
    Ok((!text.is_empty()).then_some(CommandPayload::Text { content: text }))
}

// ============================================================================
// Terminal Tool Factory (for creating child terminal tools for subagents)
// ============================================================================

use aither_core::llm::tool::ToolDefinition;

/// A type-erased handler function for terminal tools: JSON arguments and the
/// call's [`ToolContext`] in, result out.
pub type DynToolHandler = Arc<
    dyn Fn(&str, ToolContext) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> + Send + Sync,
>;

/// One type-erased tool in a terminal capability bundle.
pub struct DynTerminalToolEntry {
    /// Tool definition exposed to the model.
    pub definition: ToolDefinition,
    /// Handler that takes JSON arguments and the call's context and returns a
    /// result.
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
    pub(crate) command_names: Vec<String>,
    pub(crate) prepared_tools: crate::ToolCatalog,
    pub(crate) commit_ledger: Option<crate::subagent::CommitLedger>,
    pub(crate) role: CommandRole,
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

    /// Operational catalog inherited from the initialized parent environment.
    #[must_use]
    pub const fn prepared_tools(&self) -> &crate::ToolCatalog {
        &self.prepared_tools
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

    /// Lists the commits of the subagent this bundle serves, when it was
    /// launched with a commit grant.
    #[must_use]
    pub const fn commit_ledger(&self) -> Option<&crate::subagent::CommitLedger> {
        self.commit_ledger.as_ref()
    }

    /// The role the bundle's command registry serves: [`CommandRole::Main`]
    /// for the main agent's terminal, [`CommandRole::Subagent`] for a
    /// subagent's.
    #[must_use]
    pub const fn role(&self) -> CommandRole {
        self.role
    }

    /// Whether a CLI command with `name` is registered in the shared registry.
    #[must_use]
    pub fn has_command(&self, name: &str) -> bool {
        self.command_names.iter().any(|n| n == name)
    }

    /// The terminal controls (`terminal_kill`, `terminal_input`,
    /// `terminal_read`) bound to this bundle's own background jobs.
    ///
    /// They are not part of [`into_entries`](Self::into_entries): an agent
    /// that runs its own terminal controls on these gets control over this
    /// terminal's jobs and no other's.
    #[must_use]
    pub fn controls(&self) -> Vec<DynTerminalToolEntry> {
        use crate::builtin::{InputTerminalTool, KillTerminalTool, ReadTerminalDeltaTool};
        use crate::terminal::erase_terminal_tool;

        vec![
            erase_terminal_tool(KillTerminalTool::new(self.job_registry.clone())),
            erase_terminal_tool(InputTerminalTool::new(self.job_registry.clone())),
            erase_terminal_tool(ReadTerminalDeltaTool::new(self.job_registry.clone())),
        ]
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
///
/// Every command is registered for both [`CommandRole`]s;
/// [`Self::main_only`] reserves what it registers for the main agent (the
/// subagent role then gets a refusal naming the command as reserved for the
/// main agent), and [`Self::configure_role_tools`] registers one handler per
/// role. A command whose handler depends on the terminal serving it is
/// registered with [`Self::configure_terminal_bound_tool`] and built anew for
/// every terminal, a subagent's included.
#[derive(Debug, Default)]
pub struct ToolRegistryBuilder {
    entries: HashMap<String, RoleEntries<Arc<ToolEntry>>>,
    bound: HashMap<String, RoleEntries<Arc<BoundEntry>>>,
    observer: Option<AnyCommandObserver>,
    gate: Option<AnyCommandGate>,
}

impl ToolRegistryBuilder {
    /// Creates an empty registry builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn insert_entry(&mut self, name: String, entries: RoleEntries<Arc<ToolEntry>>) {
        self.bound.remove(&name);
        self.entries.insert(name, entries);
    }

    fn insert_bound(&mut self, name: String, entries: RoleEntries<Arc<BoundEntry>>) {
        self.entries.remove(&name);
        self.bound.insert(name, entries);
    }

    /// Configures a tool to be callable from the sandbox by both roles.
    ///
    /// The tool's argument schema is what the sandbox-side command line parses
    /// against — positional words bind to required fields, so a tool with a
    /// single required field accepts a single positional argument.
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized.
    pub fn configure_tool<T>(&mut self, tool: T)
    where
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let (name, entry) = tool_entry(tool);
        self.insert_entry(name, RoleEntries::shared(Arc::new(entry)));
    }

    /// Configures a tool built for each terminal that serves the registry,
    /// from that terminal's [`TerminalBinding`]: its working directory and
    /// the launcher for the subagents it starts. A subagent's terminal builds
    /// its own instance, so the command acts with that subagent's authority.
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized, and, when a
    /// terminal binds the registry, if the built tool is not named `name`.
    pub fn configure_terminal_bound_tool<T, F>(&mut self, name: impl Into<String>, bind: F)
    where
        F: Fn(&TerminalBinding) -> T + Send + Sync + 'static,
        T: Tool + Send + Sync + 'static,
        T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let name = name.into();
        let schema = tool_schema::<T::Arguments>();
        let help = schema_to_help(&schema);
        let stdin_arg = detect_stdin_arg(&schema);
        let bound_name = name.clone();
        let entry = BoundEntry {
            bind: Box::new(move |binding| {
                let tool = bind(binding);
                assert_eq!(
                    tool.name(),
                    bound_name,
                    "a terminal-bound command keeps the name it was registered under"
                );
                tool_handler(Arc::new(tool), schema.clone())
            }),
            help,
            stdin_arg,
        };
        self.insert_bound(name, RoleEntries::shared(Arc::new(entry)));
    }

    /// Reserves for the main agent every command `configure` registers: a
    /// subagent's call is refused with a message naming the command as
    /// reserved for the main agent, while help and stdin handling stay those
    /// of the real command.
    pub fn main_only(&mut self, configure: impl FnOnce(&mut Self)) {
        let mut reserved = Self::new();
        configure(&mut reserved);
        for (name, entries) in reserved.entries {
            self.insert_entry(
                name,
                RoleEntries {
                    main: entries.main,
                    subagent: None,
                },
            );
        }
        for (name, entries) in reserved.bound {
            self.insert_bound(
                name,
                RoleEntries {
                    main: entries.main,
                    subagent: None,
                },
            );
        }
    }

    /// Configures one command with a handler per role.
    ///
    /// # Panics
    /// Panics if the tools do not share a name, or a tool argument schema
    /// cannot be serialized.
    pub fn configure_role_tools<M, S>(&mut self, main: M, subagent: S)
    where
        M: Tool + Send + Sync + 'static,
        M::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
        S: Tool + Send + Sync + 'static,
        S::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
    {
        let (name, main) = tool_entry(main);
        let (subagent_name, subagent) = tool_entry(subagent);
        assert_eq!(
            name, subagent_name,
            "a command's role handlers must share its name"
        );
        self.insert_entry(
            name,
            RoleEntries {
                main: Arc::new(main),
                subagent: Some(Arc::new(subagent)),
            },
        );
    }

    /// Configures a schema-defined handler to be callable from the sandbox by
    /// both roles.
    ///
    /// This is the dynamic counterpart to [`Self::configure_tool`]. The CLI
    /// wrapper is derived entirely from the provided [`ToolDefinition`]'s JSON
    /// schema, so external tools such as MCP definitions get the same
    /// positional arguments, `--help`, and usage validation behavior as native
    /// `Tool` implementations.
    pub fn configure_definition_handler<F>(&mut self, definition: &ToolDefinition, handler: F)
    where
        F: Fn(Value) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let (name, entry) = definition_entry(definition, handler);
        self.insert_entry(name, RoleEntries::shared(Arc::new(entry)));
    }

    /// Registers a raw handler function as an IPC command for both roles.
    ///
    /// This is useful for dynamic tools like MCP tools that are discovered at runtime.
    pub fn configure_raw_handler<F>(
        &mut self,
        name: impl Into<String>,
        help: impl Into<String>,
        handler: F,
    ) where
        F: Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        self.insert_entry(
            name.into(),
            RoleEntries::shared(Arc::new(ToolEntry {
                handler: Box::new(handler),
                help: help.into(),
                stdin_arg: None,
            })),
        );
    }

    /// Has `observer` told about every call of a resolved command the main
    /// agent's registry serves. A subagent's registry is observed by the
    /// observer given to [`ToolRegistry::for_subagent`].
    pub fn observe_commands(&mut self, observer: AnyCommandObserver) {
        self.observer = Some(observer);
    }

    /// Has `gate` decide on every shell command the main agent's terminal
    /// runs. A subagent's terminal is gated by the gate given to
    /// [`ToolRegistry::for_subagent`].
    pub fn gate_commands(&mut self, gate: AnyCommandGate) {
        self.gate = Some(gate);
    }

    /// Builds an immutable main-agent registry with a concrete output
    /// directory; [`ToolRegistry::for_subagent`] derives the subagent view.
    /// Terminal-bound commands are built when a terminal serves it.
    #[must_use]
    pub fn build(self, output_dir: impl Into<PathBuf>) -> ToolRegistry {
        ToolRegistry {
            entries: self.entries,
            bound: self.bound,
            instances: HashMap::new(),
            output_dir: output_dir.into(),
            role: CommandRole::Main,
            observer: self.observer,
            gate: self.gate,
        }
    }
}

/// The JSON schema of a tool's arguments.
fn tool_schema<A: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(A)).expect("failed to serialize tool schema")
}

/// The handler running `tool` with arguments parsed against `schema`.
fn tool_handler<T>(tool: Arc<T>, schema: Value) -> ToolHandlerFn
where
    T: Tool + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    let name_for_errors = tool.name().to_string();
    Box::new(move |args: Vec<String>| {
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
            // A shell invocation has nowhere to show progress.
            let output = tool
                .call(parsed, ToolContext::new())
                .await
                .map_err(|error| error.to_string())?;
            output
                .into_tool_result()
                .map_err(|error| prefix_error("failed to convert tool result: ", &error))
        })
    })
}

/// Builds the registry entry for a [`Tool`].
fn tool_entry<T>(tool: T) -> (String, ToolEntry)
where
    T: Tool + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    let schema = tool_schema::<T::Arguments>();
    // Auto-detect stdin_arg from schema: optional "input" field
    let stdin_arg = detect_stdin_arg(&schema);
    let help = schema_to_help(&schema);
    let name = tool.name().to_string();
    (
        name,
        ToolEntry {
            handler: tool_handler(Arc::new(tool), schema),
            help,
            stdin_arg,
        },
    )
}

/// Builds the registry entry for a schema-defined handler.
fn definition_entry<F>(definition: &ToolDefinition, handler: F) -> (String, ToolEntry)
where
    F: Fn(Value) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send>>
        + Send
        + Sync
        + 'static,
{
    let name = definition.name().to_string();
    let description = definition.description().to_string();
    let schema = definition.arguments_openai_schema();
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

    (
        name,
        ToolEntry {
            handler: raw_handler,
            help: help_text,
            stdin_arg,
        },
    )
}

/// Immutable registry of tool handlers, as one [`CommandRole`] sees them.
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    entries: HashMap<String, RoleEntries<Arc<ToolEntry>>>,
    bound: HashMap<String, RoleEntries<Arc<BoundEntry>>>,
    /// The terminal-bound commands built for the terminal serving this
    /// registry, in its role.
    instances: HashMap<String, Arc<ToolEntry>>,
    output_dir: PathBuf,
    role: CommandRole,
    observer: Option<AnyCommandObserver>,
    gate: Option<AnyCommandGate>,
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
    pub(crate) fn with_terminal_controls(
        mut self,
        jobs: crate::JobRegistry,
        output_dir: PathBuf,
    ) -> Self {
        let mut controls = ToolRegistryBuilder::new();
        controls.configure_tool(crate::builtin::KillTerminalTool::new(jobs.clone()));
        controls.configure_tool(crate::builtin::InputTerminalTool::new(jobs.clone()));
        controls.configure_tool(crate::builtin::ReadTerminalDeltaTool::new(jobs));
        self.entries.extend(controls.entries);
        self.output_dir = output_dir;
        self
    }

    /// Builds the terminal-bound commands for the terminal `binding`
    /// describes, in this registry's role.
    pub(crate) fn bind(mut self, binding: &TerminalBinding) -> Self {
        self.instances = self
            .bound
            .iter()
            .filter_map(|(name, entries)| {
                entries
                    .for_role(self.role)
                    .map(|entry| (name.clone(), Arc::new(entry.instantiate(binding))))
            })
            .collect();
        self
    }

    /// The role this registry serves.
    #[must_use]
    pub const fn role(&self) -> CommandRole {
        self.role
    }

    /// The registry a subagent's terminal serves: the same commands, each
    /// resolved to its subagent-role handler, with terminal-bound commands
    /// left for the subagent's terminal to build. Asking a subagent registry
    /// for its subagent registry returns the same view, so nesting never
    /// escalates.
    ///
    /// The subagent's calls are told to `observer`, not to this registry's
    /// observer, which belongs to the agent that launches the subagent; the
    /// same holds for `gate`.
    #[must_use]
    pub fn for_subagent(
        &self,
        observer: Option<AnyCommandObserver>,
        gate: Option<AnyCommandGate>,
    ) -> Self {
        Self {
            entries: self.entries.clone(),
            bound: self.bound.clone(),
            instances: HashMap::new(),
            output_dir: self.output_dir.clone(),
            role: CommandRole::Subagent,
            observer,
            gate,
        }
    }

    /// The gate every shell command of the terminal serving this registry
    /// is put to.
    #[must_use]
    pub const fn gate(&self) -> Option<&AnyCommandGate> {
        self.gate.as_ref()
    }

    /// The observer told about the calls this registry serves.
    #[must_use]
    pub const fn observer(&self) -> Option<&AnyCommandObserver> {
        self.observer.as_ref()
    }

    /// What `tool_name` resolves to in this registry's role.
    ///
    /// # Panics
    /// Panics when `tool_name` is a terminal-bound command and no terminal
    /// has bound this registry; a terminal binds the registry it serves.
    fn resolve(&self, tool_name: &str) -> Option<Resolved<'_>> {
        if let Some(bound) = self.bound.get(tool_name) {
            let entry = bound.for_role(self.role);
            let shown = entry.unwrap_or(&bound.main);
            return Some(Resolved {
                handler: entry.map(|_| {
                    &self
                        .instances
                        .get(tool_name)
                        .expect("a terminal binds the registry it serves")
                        .handler
                }),
                help: &shown.help,
                stdin_arg: shown.stdin_arg.as_deref(),
            });
        }
        let entries = self.entries.get(tool_name)?;
        let shown = entries.for_role(self.role).unwrap_or(&entries.main);
        Some(Resolved {
            handler: entries.for_role(self.role).map(|entry| &entry.handler),
            help: &shown.help,
            stdin_arg: shown.stdin_arg.as_deref(),
        })
    }

    /// Queries a tool handler by name.
    ///
    /// Large outputs (> 4000 chars) are automatically saved to file.
    pub async fn query_tool_handler(&self, tool_name: &str, args: &[String]) -> CommandEnvelope {
        match self.resolve(tool_name) {
            Some(Resolved {
                handler: Some(handler),
                ..
            }) => {
                let outcome = match handler(args.to_vec()).await {
                    Ok(output) => handle_large_output(output, &self.output_dir).await,
                    Err(error) => CommandEnvelope::failure(error),
                };
                if let Some(observer) = &self.observer {
                    observer
                        .command_called(&CommandCall {
                            role: self.role,
                            name: tool_name,
                            args,
                            outcome: &outcome,
                        })
                        .await;
                }
                outcome
            }
            Some(Resolved { handler: None, .. }) => {
                CommandEnvelope::failure(reserved_for_main_text(tool_name))
            }
            None => CommandEnvelope::failure(unknown_command_text(tool_name)),
        }
    }

    /// Returns whether a tool is registered.
    #[must_use]
    pub fn is_tool_configured(&self, tool_name: &str) -> bool {
        self.entries.contains_key(tool_name) || self.bound.contains_key(tool_name)
    }

    /// Returns whether a tool is reserved for the main agent in this
    /// registry's role, so a call to it is refused.
    #[must_use]
    pub fn is_reserved(&self, tool_name: &str) -> bool {
        self.bound.get(tool_name).map_or_else(
            || {
                self.entries
                    .get(tool_name)
                    .is_some_and(|entries| entries.for_role(self.role).is_none())
            },
            |bound| bound.for_role(self.role).is_none(),
        )
    }

    /// Returns the help text for a tool.
    #[must_use]
    pub fn tool_help(&self, tool_name: &str) -> Option<String> {
        self.resolve(tool_name)
            .map(|resolved| resolved.help.to_string())
    }

    /// Returns the stdin argument for a tool.
    #[must_use]
    pub fn tool_stdin_arg(&self, tool_name: &str) -> Option<String> {
        self.resolve(tool_name)
            .and_then(|resolved| resolved.stdin_arg.map(str::to_string))
    }

    /// Returns list of registered tool names.
    #[must_use]
    pub fn registered_tool_names(&self) -> Vec<String> {
        self.entries
            .keys()
            .chain(self.bound.keys())
            .cloned()
            .collect()
    }

    /// Returns the registered tool names this registry's role can actually
    /// run, leaving out commands reserved for the main agent.
    #[must_use]
    pub fn available_tool_names(&self) -> Vec<String> {
        self.registered_tool_names()
            .into_iter()
            .filter(|name| !self.is_reserved(name))
            .collect()
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

/// Renders a tool call's outcome the way the terminal caller expects it:
/// content on stdout for success, a transport error for failure so `heel ipc`
/// exits non-zero.
pub fn envelope_to_cli(envelope: CommandEnvelope) -> Result<String, IpcError> {
    if !envelope.ok {
        return Err(IpcError::remote(
            envelope
                .error
                .unwrap_or_else(|| "command failed".to_string()),
        ));
    }
    Ok(envelope
        .payload
        .map_or_else(String::new, |payload| payload.render_for_cli()))
}

/// Rebuilds the command line a `heel ipc` request carries.
///
/// Raw-args commands deliver their argv untouched in `args`; named entries —
/// the declared stdin field, or named parameters from a bound-args client —
/// are appended as `--name value` pairs, unless the argv already declared
/// them (an explicit flag on the command line wins over piped input).
pub fn flatten_args_to_cli(args: &std::collections::HashMap<String, Value>) -> Vec<String> {
    let mut cli_args = Vec::new();
    if let Some(Value::Array(arr)) = args.get("args") {
        cli_args.extend(arr.iter().filter_map(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }));
    }

    // Options already spelled out in the verbatim argv keep their value; a
    // same-named bound field would only come from piped stdin.
    let declared: HashSet<String> = cli_args
        .iter()
        .filter_map(|arg| arg.strip_prefix("--"))
        .filter_map(|name| name.split('=').next())
        .map(str::to_string)
        .collect();

    let mut keys = args
        .keys()
        .map(String::as_str)
        .filter(|key| *key != "args")
        .collect::<Vec<_>>();
    keys.sort_unstable();
    for key in keys {
        if declared.contains(key.replace('_', "-").as_str()) {
            continue;
        }
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
    type Response = String;

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.tool_name.clone())
    }

    /// The schema parser on this side binds positionals itself; the client
    /// cannot know them for tagged-enum tools, so every word goes through.
    fn raw_args(&self) -> bool {
        true
    }

    fn stdin_arg(&self) -> Option<Cow<'static, str>> {
        self.registry
            .tool_stdin_arg(&self.tool_name)
            .map(Cow::Owned)
    }

    async fn handle(&self, args: Self::Args) -> Result<String, IpcError> {
        let cli_args = flatten_args_to_cli(&args);
        tracing::info!(tool = %self.tool_name, args = ?cli_args, "ToolCallCommand::handle invoked");

        // Handle help flags
        if has_help_flag(&cli_args) {
            let help = self
                .registry
                .tool_help(&self.tool_name)
                .unwrap_or_else(|| no_help_available_text(self.tool_name.as_str()));
            tracing::info!(tool = %self.tool_name, "Returning help text");
            return Ok(help);
        }

        tracing::info!(tool = %self.tool_name, "Calling registry.query_tool_handler");
        envelope_to_cli(
            self.registry
                .query_tool_handler(&self.tool_name, &cli_args)
                .await,
        )
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
    type Response = String;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("ipc")
    }

    fn raw_args(&self) -> bool {
        true
    }

    async fn handle(&self, args: Self::Args) -> Result<String, IpcError> {
        let cli_args = flatten_args_to_cli(&args);
        if cli_args.is_empty() {
            return Err(IpcError::remote("usage: ipc <tool> [args ...]"));
        }

        if has_help_flag(&cli_args) && cli_args.len() == 1 {
            let mut names = self.registry.registered_tool_names();
            names.sort();
            let joined_names = names.join(", ");
            return Ok(IpcGatewayHelpTemplate {
                tool_names: joined_names.as_str(),
            }
            .render()
            .expect("ipc gateway help template must render"));
        }

        let tool_name = &cli_args[0];
        let tool_args = cli_args[1..].to_vec();
        envelope_to_cli(
            self.registry
                .query_tool_handler(tool_name, &tool_args)
                .await,
        )
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
    output_dir: PathBuf,
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
    /// Creates a new command wrapper for the given tool. Binary results are
    /// stored in `output_dir`.
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized.
    pub fn new(tool: T, output_dir: impl Into<PathBuf>) -> Self {
        let schema = schemars::schema_for!(T::Arguments);
        let schema = serde_json::to_value(schema).expect("failed to serialize tool schema");
        Self {
            tool,
            schema,
            output_dir: output_dir.into(),
        }
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
        // A CLI invocation has nowhere to show progress.
        let output = self.tool.call(parsed, ToolContext::new()).await?;
        let payload = tool_output_to_payload(output, &self.output_dir)
            .await
            .map_err(anyhow::Error::msg)?;
        Ok(payload.map_or_else(String::new, |payload| payload.render_for_cli()))
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
    stdin_arg: Option<String>,
    help: String,
    output_dir: PathBuf,
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
            stdin_arg: self.stdin_arg.clone(),
            help: self.help.clone(),
            output_dir: self.output_dir.clone(),
        }
    }
}

impl<T> IpcToolCommand<T>
where
    T: Tool + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    /// Creates a new IPC command wrapper for the given tool. Binary results
    /// are stored in `output_dir`.
    ///
    /// # Panics
    /// Panics if the tool argument schema cannot be serialized.
    pub fn new(tool: T, output_dir: impl Into<PathBuf>) -> Self {
        let schema = serde_json::to_value(schemars::schema_for!(T::Arguments))
            .expect("failed to serialize tool schema");
        let name = tool.name().to_string();
        let stdin_arg = detect_stdin_arg(&schema);
        let help = schema_to_help(&schema);

        Self {
            tool: Arc::new(tool),
            schema,
            name,
            stdin_arg,
            help,
            output_dir: output_dir.into(),
        }
    }
}

impl<T> IpcCommand for IpcToolCommand<T>
where
    T: Tool + Clone + Send + Sync + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Args = HashMap<String, Value>;
    type Response = String;

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    fn raw_args(&self) -> bool {
        true
    }

    fn stdin_arg(&self) -> Option<Cow<'static, str>> {
        self.stdin_arg.clone().map(Cow::Owned)
    }

    async fn handle(&self, args: Self::Args) -> Result<String, IpcError> {
        let cli_args = flatten_args_to_cli(&args);

        // Handle help flags
        if has_help_flag(&cli_args) {
            return Ok(self.help.clone());
        }

        // Parse CLI args to JSON
        let json_args = match cli_to_json(&self.schema, &cli_args) {
            Ok(args) => args,
            Err(e) => {
                let message = e.to_string();
                return Err(IpcError::remote(tool_usage_error(
                    message.as_str(),
                    self.name.as_str(),
                )));
            }
        };

        // Deserialize and call tool
        let parsed: T::Arguments = match serde_json::from_value(json_args) {
            Ok(args) => args,
            Err(e) => {
                let message = prefix_error("invalid arguments: ", &e);
                return Err(IpcError::remote(tool_usage_error(
                    message.as_str(),
                    self.name.as_str(),
                )));
            }
        };

        // A shell invocation has nowhere to show progress.
        match self.tool.call(parsed, ToolContext::new()).await {
            Ok(output) => tool_output_to_payload(output, &self.output_dir)
                .await
                .map(|payload| payload.map_or_else(String::new, |payload| payload.render_for_cli()))
                .map_err(IpcError::remote),
            Err(e) => Err(IpcError::remote(e.to_string())),
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
/// let router = register_tool_direct(router, my_tool, outputs_dir);
/// ```
pub fn register_tool_direct<T>(
    router: heel::IpcRouter,
    tool: T,
    output_dir: impl Into<PathBuf>,
) -> heel::IpcRouter
where
    T: Tool + Send + Sync + Clone + Default + 'static,
    T::Arguments: DeserializeOwned + JsonSchema + Send + 'static,
{
    router.register(IpcToolCommand::new(tool, output_dir))
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

    fn single_stored_file(dir: &Path) -> PathBuf {
        let mut files = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 1);
        files.remove(0)
    }

    #[tokio::test]
    async fn binary_tool_result_parts_are_stored_and_referenced() {
        let dir = tempfile::tempdir().unwrap();
        let png = vec![0x89, 0x50, 0x4E, 0x47];
        let payload = tool_output_to_payload(
            ToolResult::parts(vec![
                ToolResultPart::text("see below"),
                ToolResultPart::image(png.clone(), "image/png"),
            ]),
            dir.path(),
        )
        .await
        .unwrap()
        .expect("mixed parts produce a payload");

        let CommandPayload::Text { content } = payload;
        let file = single_stored_file(dir.path());
        assert_eq!(file.extension().unwrap(), "png");
        assert_eq!(std::fs::read(&file).unwrap(), png);
        let name = file.file_name().unwrap().to_str().unwrap();
        assert_eq!(
            content,
            format!(
                "see below\n[image/png, 4 bytes, stored at outputs/{name} ({})]",
                file.display()
            )
        );
    }

    #[tokio::test]
    async fn lone_binary_tool_result_is_stored_and_referenced() {
        let dir = tempfile::tempdir().unwrap();
        let png = vec![0x89, 0x50, 0x4E, 0x47];
        let payload =
            tool_output_to_payload(ToolResult::image(png.clone(), "image/png"), dir.path())
                .await
                .unwrap()
                .expect("a lone image produces a payload");

        let CommandPayload::Text { content } = payload;
        let file = single_stored_file(dir.path());
        assert_eq!(std::fs::read(&file).unwrap(), png);
        let name = file.file_name().unwrap().to_str().unwrap();
        assert_eq!(
            content,
            format!(
                "[image/png, 4 bytes, stored at outputs/{name} ({})]",
                file.display()
            )
        );
    }

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

        // Bare words bind to required fields in order, but `options` is an
        // array — it only parses through repeated `--options` flags.
        let result = cli_to_json(
            &schema,
            &["Pick one".into(), "--options".into(), "A".into()],
        )
        .expect("parses");
        assert_eq!(result["question"], "Pick one");
        assert_eq!(result["options"], serde_json::json!(["A"]));
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

    #[test]
    fn test_flatten_args_to_cli_merges_verbatim_argv_and_named_fields() {
        // What `heel ipc --raw-args` sends for a piped invocation: argv in
        // "args" plus the stdin field bound by name.
        let args = serde_json::from_value::<HashMap<String, Value>>(serde_json::json!({
            "args": ["render", "--detail", "full"],
            "input": "piped contents"
        }))
        .expect("structured args should deserialize");

        let cli = flatten_args_to_cli(&args);
        assert_eq!(
            cli,
            vec!["render", "--detail", "full", "--input", "piped contents"]
        );
    }

    #[test]
    fn test_flatten_args_to_cli_explicit_flag_wins_over_bound_stdin() {
        // `cmd --input explicit < file` — the flag on the command line decides,
        // not the piped contents.
        let args = serde_json::from_value::<HashMap<String, Value>>(serde_json::json!({
            "args": ["--input", "explicit"],
            "input": "piped contents"
        }))
        .expect("structured args should deserialize");

        let cli = flatten_args_to_cli(&args);
        assert_eq!(cli, vec!["--input", "explicit"]);
    }

    #[tokio::test]
    async fn test_handle_large_output_write_failure_returns_error_message() {
        let tmp = tempfile::tempdir().unwrap();
        let output_dir = tmp.path().join("not-a-directory");
        std::fs::write(&output_dir, "occupied").unwrap();
        let output = ToolResult::text("x".repeat(INLINE_OUTPUT_LIMIT + 1));

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
                _cx: aither_core::llm::ToolContext,
            ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send
            {
                // Return the parsed args as text so we can inspect
                std::future::ready(Ok(ToolResult::text(format!(
                    "question={:?} option={:?} multi_select={:?} questions={:?}",
                    args.question, args.option, args.multi_select, args.questions
                ))))
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

        let rendered = cmd.handle(args).await.expect("command should succeed");
        tracing::debug!("result: {rendered}");
        assert!(
            rendered.contains("option=[\"原味\", \"番茄味\", \"芝士味\"]"),
            "{rendered}"
        );
        assert!(
            rendered.contains("question=Some(\"你喜欢哪种薯条？\")"),
            "{rendered}"
        );
    }

    #[tokio::test]
    async fn test_tagged_enum_tool_receives_subcommands_through_raw_args() {
        use aither_core::llm::{Tool, ToolResult};
        use std::borrow::Cow;

        /// Echoes which variant the schema parser picked, like `browser` does.
        #[derive(Debug, Clone)]
        struct FakeFileOps;
        impl Tool for FakeFileOps {
            fn name(&self) -> Cow<'static, str> {
                "file_ops".into()
            }
            type Arguments = TaggedOperationArgs;
            type Res = ToolResult;

            fn call(
                &self,
                args: Self::Arguments,
                _cx: aither_core::llm::ToolContext,
            ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send
            {
                let text = match args {
                    TaggedOperationArgs::Read { path } => format!("read {path}"),
                    TaggedOperationArgs::Write { path, content } => {
                        format!("write {path} = {content}")
                    }
                };
                std::future::ready(Ok(ToolResult::Text { text }))
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(FakeFileOps);
        let registry = std::sync::Arc::new(builder.build(tmp.path()));
        let cmd = ToolCallCommand::new("file_ops", registry);

        // `file_ops read input.txt` — the subcommand word and the variant's
        // positional field both reach the schema parser untouched.
        let read = cmd
            .handle(
                serde_json::from_value(serde_json::json!({
                    "args": ["read", "input.txt"]
                }))
                .unwrap(),
            )
            .await
            .expect("read parses and runs");
        assert_eq!(read, "read input.txt");

        let write = cmd
            .handle(
                serde_json::from_value(serde_json::json!({
                    "args": ["write", "out.txt", "hello"]
                }))
                .unwrap(),
            )
            .await
            .expect("write parses and runs");
        assert_eq!(write, "write out.txt = hello");

        // An unknown subcommand fails the call rather than printing an
        // `ok:false` payload to a shell that cannot tell it from success.
        let error = cmd
            .handle(
                serde_json::from_value(serde_json::json!({
                    "args": ["explode"]
                }))
                .unwrap(),
            )
            .await
            .expect_err("unknown subcommand must fail");
        assert!(
            error.to_string().contains("unknown subcommand 'explode'"),
            "unexpected error: {error}"
        );
    }

    /// Echoes which role's handler ran.
    #[derive(Debug, Deserialize, JsonSchema)]
    struct RoleArgs {}

    struct RoleTool {
        name: &'static str,
        answer: &'static str,
    }

    impl aither_core::llm::Tool for RoleTool {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            self.name.into()
        }
        type Arguments = RoleArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::Text {
                text: self.answer.to_string(),
            }))
        }
    }

    async fn run_command(registry: &ToolRegistry, name: &str) -> CommandEnvelope {
        registry.query_tool_handler(name, &[]).await
    }

    #[tokio::test]
    async fn main_only_command_is_refused_to_a_subagent_and_works_for_the_main_agent() {
        let tmp = tempfile::tempdir().unwrap();
        let mut builder = ToolRegistryBuilder::new();
        builder.main_only(|reserved| {
            reserved.configure_tool(RoleTool {
                name: "reserved",
                answer: "main ran it",
            });
        });
        builder.configure_tool(RoleTool {
            name: "shared",
            answer: "anyone ran it",
        });
        let main = builder.build(tmp.path());
        let subagent = main.for_subagent(None, None);

        let ok = run_command(&main, "reserved").await;
        assert!(ok.ok, "the main agent keeps the command: {ok:?}");
        assert_eq!(ok.payload.unwrap().render_for_cli(), "main ran it");

        let refused = run_command(&subagent, "reserved").await;
        assert!(!refused.ok);
        let error = refused.error.unwrap();
        assert!(
            error.contains("`reserved` is reserved for the main agent"),
            "{error}"
        );
        assert!(subagent.is_reserved("reserved"));
        assert!(!subagent.available_tool_names().contains(&"reserved".into()));
        assert!(
            subagent
                .registered_tool_names()
                .contains(&"reserved".into())
        );

        let shared = run_command(&subagent, "shared").await;
        assert!(shared.ok, "a shared command stays available: {shared:?}");
    }

    #[tokio::test]
    async fn role_split_command_runs_each_roles_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_role_tools(
            RoleTool {
                name: "split",
                answer: "main handler",
            },
            RoleTool {
                name: "split",
                answer: "subagent handler",
            },
        );
        let main = builder.build(tmp.path());
        let subagent = main.for_subagent(None, None);

        let answer =
            |envelope: CommandEnvelope| envelope.payload.expect("the handler ran").render_for_cli();
        assert_eq!(main.role(), CommandRole::Main);
        assert_eq!(answer(run_command(&main, "split").await), "main handler");
        assert_eq!(subagent.role(), CommandRole::Subagent);
        assert_eq!(
            answer(run_command(&subagent, "split").await),
            "subagent handler"
        );
    }

    #[test]
    fn a_subagent_registry_delegates_the_subagent_registry() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = ToolRegistryBuilder::new()
            .build(tmp.path())
            .for_subagent(None, None)
            .for_subagent(None, None);
        assert_eq!(nested.role(), CommandRole::Subagent);
    }

    /// Records the role and name of every call it is told about.
    struct CallLog(Arc<std::sync::Mutex<Vec<(CommandRole, String)>>>);

    impl CommandObserver for CallLog {
        async fn command_called(&self, call: &CommandCall<'_>) {
            self.0
                .lock()
                .unwrap()
                .push((call.role, call.name.to_string()));
        }
    }

    #[tokio::test]
    async fn each_registry_reports_its_calls_to_its_own_observer() {
        let tmp = tempfile::tempdir().unwrap();
        let (main_calls, subagent_calls) = (Arc::default(), Arc::default());
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(RoleTool {
            name: "shared",
            answer: "anyone ran it",
        });
        builder.observe_commands(AnyCommandObserver::new(CallLog(Arc::clone(&main_calls))));
        let main = builder.build(tmp.path());
        let subagent = main.for_subagent(
            Some(AnyCommandObserver::new(CallLog(Arc::clone(
                &subagent_calls,
            )))),
            None,
        );
        let unobserved = main.for_subagent(None, None);

        run_command(&main, "shared").await;
        run_command(&subagent, "shared").await;
        run_command(&unobserved, "shared").await;
        run_command(&subagent, "unknown").await;

        assert_eq!(
            *main_calls.lock().unwrap(),
            [(CommandRole::Main, "shared".to_string())]
        );
        assert_eq!(
            *subagent_calls.lock().unwrap(),
            [(CommandRole::Subagent, "shared".to_string())]
        );
        assert!(main.observer().is_some() && unobserved.observer().is_none());
    }
}
