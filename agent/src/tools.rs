//! Tools registry for the agent.
//!
//! All registered tools are always loaded into the LLM context.

use aither_core::llm::tool::{
    IntoToolResult, RegisterError, Tool, ToolContext, ToolDefinition, ToolResult,
    Tools as CoreTools,
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
                move |args: &str, cx: ToolContext| -> Pin<
                    Box<dyn Future<Output = aither_core::Result<ToolResult>> + Send>,
                > {
                let handler = handler.clone();
                let args = args.to_string();
                Box::pin(async move {
                    handler(&args, cx).await.into_tool_result()
                })
                },
            )
            .expect("dynamic terminal tool registration must succeed: duplicate name or missing description");
        }
    }

    /// The tools of a fork: `definitions` exactly as the main agent's request
    /// carried them, run by the entries and terminal controls of `terminal`
    /// of the same name, so a control acts on the fork's own jobs. A
    /// definition `terminal` has no entry for belongs to the main agent and
    /// refuses when called. The native `compact` definition is the exception:
    /// it runs the real tool over `compact`, the fork's own signal, so a call
    /// compacts the fork.
    pub(crate) fn forked(
        definitions: &[ToolDefinition],
        terminal: aither_sandbox::DynTerminalTool,
        compact: Option<&std::sync::Arc<crate::compact_tool::CompactSignal>>,
        notices: &crate::AnyNotices,
    ) -> Self {
        use futures_core::Future;
        use std::pin::Pin;

        let controls = terminal.controls();
        let mut handlers: std::collections::HashMap<_, _> = terminal
            .into_entries()
            .into_iter()
            .chain(controls)
            .map(|entry| (entry.definition.name().to_string(), entry.handler))
            .collect();
        let mut tools = Self::new();
        for definition in definitions {
            let name = definition.name().to_string();
            let handler = handlers.remove(&name);
            let compact = compact
                .filter(|_| name == crate::compact_tool::COMPACT_TOOL_NAME)
                .map(|signal| {
                    crate::compact_tool::CompactTool::new(
                        std::sync::Arc::clone(signal),
                        notices.clone(),
                    )
                });
            let notices = notices.clone();
            tools
                .eager
                .register_dyn(
                    definition.clone(),
                    move |args: &str,
                          cx: ToolContext|
                          -> Pin<
                        Box<dyn Future<Output = aither_core::Result<ToolResult>> + Send>,
                    > {
                        if let Some(tool) = compact.clone() {
                            let args = args.to_string();
                            return Box::pin(async move {
                                let args = serde_json::from_str(&args).map_err(|error| {
                                    anyhow::anyhow!("Invalid arguments: {error}")
                                })?;
                                Tool::call(&tool, args, cx).await?.into_tool_result()
                            });
                        }
                        let Some(handler) = handler.clone() else {
                            let refusal =
                                notices.fork_tool_refusal(&crate::ForkToolRefusal { tool: &name });
                            return Box::pin(async move { Err(anyhow::anyhow!(refusal)) });
                        };
                        let args = args.to_string();
                        Box::pin(async move { handler(&args, cx).await.into_tool_result() })
                    },
                )
                .expect("a definition from a request registers: names are unique and described");
        }
        tools
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
    /// Searches eager tools first, then MCP tools. An eager tool runs with
    /// `cx` as its [`ToolContext`]; an MCP tool runs in its server, which
    /// reports to nobody here because no progress token is sent with the
    /// request.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn call(
        &self,
        name: &str,
        args: &str,
        cx: ToolContext,
    ) -> aither_core::Result<ToolResult> {
        if self.eager.definitions().iter().any(|d| d.name() == name) {
            return self.eager.call(name, args, cx).await;
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

                return aither_mcp::call_result_to_tool_result(result)
                    .map_err(|e| anyhow::anyhow!("MCP tool error: {e}"));
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
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(ToolResult::text("ok")))
        }
    }

    /// A tool that answers with a PNG.
    #[cfg(feature = "mcp")]
    struct Screenshot;

    #[cfg(feature = "mcp")]
    impl Tool for Screenshot {
        fn name(&self) -> Cow<'static, str> {
            Cow::Borrowed("screenshot")
        }

        type Arguments = DummyArgs;

        type Res = ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(ToolResult::image(
                vec![0x89, b'P', b'N', b'G'],
                "image/png",
            )))
        }
    }

    #[cfg(feature = "mcp")]
    #[tokio::test]
    async fn an_mcp_image_result_reaches_the_model_as_an_image() {
        use aither_mcp::transport::DuplexTransport;

        let (server_end, client_end) = DuplexTransport::pair();
        let mut exposed = aither_core::llm::tool::Tools::new();
        exposed.register(Screenshot).expect("screenshot registers");
        let mut mcp_server = aither_mcp::McpServer::new(server_end, exposed, "screens", "1");
        let running = tokio::spawn(async move { mcp_server.run().await });

        let conn = McpConnection::connect(&executor_core::tokio::TokioGlobal, client_end, ())
            .await
            .expect("connects");
        let mut tools = AgentTools::new();
        tools.register_mcp(conn);

        let result = tools
            .call("screenshot", "{}", aither_core::llm::ToolContext::new())
            .await
            .expect("the call succeeds");

        assert_eq!(
            result,
            ToolResult::Binary {
                mime: "image/png".to_string(),
                content: vec![0x89, b'P', b'N', b'G'],
            }
        );
        running.abort();
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
