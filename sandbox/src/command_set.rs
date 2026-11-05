//! Terminal commands whose set changes while a session runs.
//!
//! A [`ToolRegistry`](crate::ToolRegistry) is built once, but some commands
//! are not known once: an MCP server announces a new tool list with
//! `notifications/tools/list_changed` at any time. A [`CommandSet`] is asked
//! for its commands every time the registry lists or resolves one, so a
//! command the set gains is runnable from the next terminal execution and one
//! it loses stops being runnable, with no rebuild of the registry.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use aither_core::llm::tool::ToolDefinition;
use aither_core::llm::{CallIdentity, ToolResult};
use serde_json::Value;

/// Terminal commands whose set can change while the session runs, such as
/// the tools of an MCP server.
///
/// Register one with
/// [`ToolRegistryBuilder::command_set`](crate::ToolRegistryBuilder::command_set).
/// Each command's command line, `--help` and validation derive from its
/// [`ToolDefinition`]'s schema, exactly as for
/// [`configure_definition_handler`](crate::ToolRegistryBuilder::configure_definition_handler).
pub trait CommandSet: Send + Sync + 'static {
    /// The set's commands as of now.
    fn definitions(&self) -> Vec<ToolDefinition>;

    /// Runs the command `name` with the JSON arguments its command line
    /// parsed to, on behalf of the `terminal` call `call` (`None` outside
    /// one).
    ///
    /// The error is the text the caller sees.
    fn call(
        &self,
        name: &str,
        call: Option<CallIdentity>,
        arguments: Value,
    ) -> impl Future<Output = Result<ToolResult, String>> + Send;
}

/// Object-safe twin of [`CommandSet`].
trait CommandSetImpl: Send + Sync {
    fn definitions(&self) -> Vec<ToolDefinition>;

    fn call<'a>(
        &'a self,
        name: &'a str,
        call: Option<CallIdentity>,
        arguments: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send + 'a>>;
}

impl<T: CommandSet> CommandSetImpl for T {
    fn definitions(&self) -> Vec<ToolDefinition> {
        CommandSet::definitions(self)
    }

    fn call<'a>(
        &'a self,
        name: &'a str,
        call: Option<CallIdentity>,
        arguments: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult, String>> + Send + 'a>> {
        Box::pin(CommandSet::call(self, name, call, arguments))
    }
}

/// A type-erased [`CommandSet`], shared by every registry that serves it.
#[derive(Clone)]
pub struct AnyCommandSet(Arc<dyn CommandSetImpl>);

impl AnyCommandSet {
    /// Erases `set`.
    #[must_use]
    pub fn new(set: impl CommandSet) -> Self {
        Self(Arc::new(set))
    }

    /// See [`CommandSet::definitions`].
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.0.definitions()
    }

    /// See [`CommandSet::call`].
    pub fn call<'a>(
        &'a self,
        name: &'a str,
        call: Option<CallIdentity>,
        arguments: Value,
    ) -> impl Future<Output = Result<ToolResult, String>> + Send + 'a {
        self.0.call(name, call, arguments)
    }
}

impl std::fmt::Debug for AnyCommandSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyCommandSet")
    }
}
