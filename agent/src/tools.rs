//! Tools registry for the agent.
//!
//! All registered tools are always loaded into the LLM context.

use aither_core::llm::tool::{
    IntoToolResult, RegisterError, Tool, ToolDefinition, ToolResult, Tools as CoreTools,
};
#[cfg(feature = "mcp")]
use aither_mcp::{McpConnection, McpToolService};

/// Tools registry used by the agent.
///
/// All tools are eager-loaded and included in every LLM request.
pub struct AgentTools {
    /// Always-loaded tools.
    eager: CoreTools,

    /// MCP connections (when mcp feature is enabled).
    #[cfg(feature = "mcp")]
    mcp: Vec<McpToolService>,
}

impl Default for AgentTools {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AgentTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("AgentTools");
        s.field("eager", &self.eager);
        #[cfg(feature = "mcp")]
        s.field("mcp", &self.mcp);
        s.finish()
    }
}

impl AgentTools {
    /// Creates a new empty tools registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            eager: CoreTools::new(),
            #[cfg(feature = "mcp")]
            mcp: Vec::new(),
        }
    }

    /// Registers an eager (always-loaded) tool.
    ///
    /// # Errors
    ///
    /// Returns [`RegisterError`] if the name collides with a tool that is
    /// already registered, or the tool has no description for the model to
    /// read.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<(), RegisterError> {
        self.eager.register(tool)
    }

    /// Registers a dynamic terminal tool (type-erased).
    ///
    /// This is used for child terminal tools in subagents where the concrete type
    /// is not known at compile time.
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn register_dyn_terminal(&mut self, dyn_tool: aither_sandbox::DynTerminalTool) {
        use futures_core::Future;
        use std::pin::Pin;

        for entry in dyn_tool.into_entries() {
            let handler = entry.handler;
            self.eager.register_dyn(
                entry.definition,
                move |args: &str| -> Pin<
                    Box<dyn Future<Output = aither_core::Result<ToolResult>> + Send>,
                > {
                let handler = handler.clone();
                let args = args.to_string();
                Box::pin(async move {
                    handler(&args).await.into_tool_result()
                })
                },
            )
            .expect("dynamic terminal tool registration must succeed: duplicate name or missing description");
        }
    }

    /// Returns definitions of all registered tools.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.eager.definitions()
    }

    /// Returns all tool definitions (eager + MCP).
    #[must_use]
    pub fn active_definitions(&self) -> Vec<ToolDefinition> {
        #[cfg(feature = "mcp")]
        {
            let mut defs = self.definitions();
            for conn in &self.mcp {
                defs.extend(conn.definitions());
            }
            defs
        }

        #[cfg(not(feature = "mcp"))]
        {
            self.definitions()
        }
    }

    /// Calls a tool by name with JSON arguments.
    ///
    /// Searches eager tools first, then MCP tools.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn call(&self, name: &str, args: &str) -> aither_core::Result<ToolResult> {
        if self.eager.definitions().iter().any(|d| d.name() == name) {
            return self.eager.call(name, args).await;
        }

        #[cfg(feature = "mcp")]
        for conn in &self.mcp {
            if conn.has_tool(name) {
                let args_value: serde_json::Value =
                    serde_json::from_str(args).map_err(|e| anyhow::anyhow!("Invalid JSON: {e}"))?;

                let result = conn
                    .call(name, args_value)
                    .await
                    .map_err(|e| anyhow::anyhow!("MCP tool error: {e}"))?;

                let output = result
                    .content
                    .into_iter()
                    .filter_map(|c| match c {
                        aither_mcp::Content::Text(t) => Some(t.text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                return if result.is_error {
                    Err(anyhow::anyhow!("{output}"))
                } else {
                    output.into_tool_result()
                };
            }
        }

        Err(anyhow::anyhow!("Tool '{name}' not found"))
    }

    /// Returns a reference to the underlying eager tools.
    #[must_use]
    pub const fn eager(&self) -> &CoreTools {
        &self.eager
    }

    /// Returns a mutable reference to the underlying eager tools.
    pub const fn eager_mut(&mut self) -> &mut CoreTools {
        &mut self.eager
    }

    /// Registers an MCP connection.
    ///
    /// All tools from the MCP server will be available for the agent to use.
    #[cfg(feature = "mcp")]
    pub fn register_mcp(&mut self, conn: McpConnection) {
        self.mcp.push(McpToolService::new(conn));
    }

    /// Returns the number of registered MCP connections.
    #[cfg(feature = "mcp")]
    #[must_use]
    pub const fn mcp_count(&self) -> usize {
        self.mcp.len()
    }

    /// Returns definitions from all MCP connections.
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn mcp_definitions(&self) -> Vec<ToolDefinition> {
        self.mcp
            .iter()
            .flat_map(aither_mcp::McpToolService::definitions)
            .collect()
    }

    /// Merge another `AgentTools` into this one.
    ///
    /// Note: MCP connections are not cloned (they would require ownership transfer).
    pub fn merge(&mut self, _other: Self) {
        // Core tools cannot be merged without re-registering concrete tool instances.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::borrow::Cow;

    struct DummyTool {
        name: String,
    }

    /// Arguments for the dummy test tool.
    #[derive(Debug, JsonSchema, Deserialize)]
    struct DummyArgs {}

    impl Tool for DummyTool {
        fn name(&self) -> Cow<'static, str> {
            Cow::Owned(self.name.clone())
        }

        type Arguments = DummyArgs;

        type Res = ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(ToolResult::text("ok")))
        }
    }

    #[test]
    fn test_register_eager() {
        let mut tools = AgentTools::new();
        tools
            .register(DummyTool {
                name: "test".to_string(),
            })
            .expect("dummy tool should register");

        assert_eq!(tools.definitions().len(), 1);
    }
}
