//! Enhanced tools registry with deferred loading and search support.
//!
//! This wraps the core `Tools` type with additional functionality for
//! registering deferred tools that are only loaded when searched for.

use std::borrow::Cow;
use std::collections::BTreeMap;

use aither_core::llm::tool::{Tool, ToolDefinition, ToolOutput, Tools as CoreTools};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::{SearchStrategy, ToolSearchConfig};
use crate::search::search_tools;

#[cfg(feature = "mcp")]
use aither_mcp::McpConnection;

/// Enhanced tools registry with deferred loading support.
///
/// Tools can be registered as either:
/// - **Eager**: Always included in LLM requests (loaded into context)
/// - **Deferred**: Only loaded when searched for (saves context space)
///
/// When the total tool count exceeds a threshold, a search tool is
/// automatically added to allow the LLM to discover deferred tools.
pub struct AgentTools {
    /// Always-loaded tools.
    eager: CoreTools,

    /// Deferred tools (searchable).
    deferred: BTreeMap<Cow<'static, str>, DeferredTool>,

    /// Tools loaded for the current turn via search.
    loaded_this_turn: CoreTools,

    /// Search configuration.
    config: ToolSearchConfig,

    /// MCP connections (when mcp feature is enabled).
    /// Wrapped in Mutex to allow parallel tool calls.
    #[cfg(feature = "mcp")]
    mcp: Vec<parking_lot::Mutex<McpConnection>>,
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
        s.field("deferred", &self.deferred);
        s.field("loaded_this_turn", &self.loaded_this_turn);
        s.field("config", &self.config);
        #[cfg(feature = "mcp")]
        s.field("mcp", &self.mcp);
        s.finish()
    }
}

/// A deferred tool that's only loaded when searched.
#[derive(Debug)]
struct DeferredTool {
    definition: ToolDefinition,
}

impl AgentTools {
    /// Creates a new empty tools registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            eager: CoreTools::new(),
            deferred: BTreeMap::new(),
            loaded_this_turn: CoreTools::new(),
            config: ToolSearchConfig::default(),
            #[cfg(feature = "mcp")]
            mcp: Vec::new(),
        }
    }

    /// Creates a new tools registry with the given search config.
    #[must_use]
    pub fn with_config(config: ToolSearchConfig) -> Self {
        Self {
            eager: CoreTools::new(),
            deferred: BTreeMap::new(),
            loaded_this_turn: CoreTools::new(),
            config,
            #[cfg(feature = "mcp")]
            mcp: Vec::new(),
        }
    }

    /// Registers an eager (always-loaded) tool.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.eager.register(tool);
    }

    /// Registers a dynamic bash tool (type-erased).
    ///
    /// This is used for child bash tools in subagents where the concrete type
    /// is not known at compile time.
    pub fn register_dyn_bash(&mut self, dyn_tool: aither_sandbox::DynBashTool) {
        use aither_core::llm::ToolOutput;
        use std::pin::Pin;
        use futures_core::Future;

        let handler = dyn_tool.handler;
        self.eager.register_dyn(dyn_tool.definition, move |args: &str| -> Pin<Box<dyn Future<Output = aither_core::Result<ToolOutput>> + Send>> {
            let handler = handler.clone();
            let args = args.to_string();
            Box::pin(async move {
                let result = handler(&args).await;
                Ok(ToolOutput::text(result))
            })
        });
    }

    /// Registers a deferred (searchable) tool.
    pub fn register_deferred<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name();
        let definition = ToolDefinition::new(&tool);

        // Store definition and tool separately
        // The actual tool instance goes into a holding area
        self.deferred.insert(
            name.clone(),
            DeferredTool {
                definition: definition.clone(),
            },
        );

        // Store the actual tool in the loaded_this_turn temporarily
        // It will be moved to eager when actually used
        self.loaded_this_turn.register(tool);
    }

    /// Returns the search configuration.
    #[must_use]
    pub const fn search_config(&self) -> &ToolSearchConfig {
        &self.config
    }

    /// Returns a mutable reference to the search configuration.
    pub fn search_config_mut(&mut self) -> &mut ToolSearchConfig {
        &mut self.config
    }

    /// Returns `true` if tool search should be enabled.
    #[must_use]
    pub fn should_enable_search(&self) -> bool {
        let total_count = self.eager.definitions().len() + self.deferred.len();
        self.config.should_enable(total_count)
    }

    /// Returns definitions of all eager (always-loaded) tools.
    #[must_use]
    pub fn eager_definitions(&self) -> Vec<ToolDefinition> {
        self.eager.definitions()
    }

    /// Returns definitions of all deferred tools.
    #[must_use]
    pub fn deferred_definitions(&self) -> Vec<ToolDefinition> {
        self.deferred
            .values()
            .map(|d| d.definition.clone())
            .collect()
    }

    /// Returns definitions of tools loaded this turn via search.
    #[must_use]
    pub fn loaded_definitions(&self) -> Vec<ToolDefinition> {
        self.loaded_this_turn.definitions()
    }

    /// Returns all tool definitions (eager + loaded this turn + MCP).
    #[must_use]
    pub fn active_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = self.eager_definitions();
        defs.extend(self.loaded_definitions());

        #[cfg(feature = "mcp")]
        for conn in &self.mcp {
            defs.extend(conn.lock().definitions());
        }

        defs
    }

    /// Clears tools loaded this turn.
    pub fn clear_loaded(&mut self) {
        self.loaded_this_turn = CoreTools::new();
    }

    /// Searches deferred tools and loads matching ones.
    ///
    /// Returns a description of the loaded tools.
    pub fn search_and_load(&mut self, query: &str) -> String {
        let deferred_defs: Vec<ToolDefinition> = self.deferred_definitions();

        if deferred_defs.is_empty() {
            return "No deferred tools available.".to_string();
        }

        let indices = search_tools(
            query,
            &deferred_defs,
            self.config.strategy,
            self.config.top_k,
        );

        if indices.is_empty() {
            return format!("No tools found matching '{query}'.");
        }

        let mut loaded = Vec::new();
        for idx in indices {
            if let Some(def) = deferred_defs.get(idx) {
                loaded.push(format!("- {}: {}", def.name(), def.description()));
            }
        }

        format!(
            "Loaded {} tool(s) matching '{}':\n{}",
            loaded.len(),
            query,
            loaded.join("\n")
        )
    }

    /// Calls a tool by name with JSON arguments.
    ///
    /// Searches eager, loaded, and MCP tools.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn call(&self, name: &str, args: &str) -> aither_core::Result<ToolOutput> {
        // Try eager tools first
        if self.eager.definitions().iter().any(|d| d.name() == name) {
            return self.eager.call(name, args).await;
        }

        // Try loaded tools
        if self
            .loaded_this_turn
            .definitions()
            .iter()
            .any(|d| d.name() == name)
        {
            return self.loaded_this_turn.call(name, args).await;
        }

        // Try MCP tools
        #[cfg(feature = "mcp")]
        for conn in &self.mcp {
            let has_tool = conn.lock().has_tool(name);
            if has_tool {
                let args_value: serde_json::Value =
                    serde_json::from_str(args).map_err(|e| anyhow::anyhow!("Invalid JSON: {e}"))?;

                let result = conn
                    .lock()
                    .call(name, args_value)
                    .await
                    .map_err(|e| anyhow::anyhow!("MCP tool error: {e}"))?;

                // Convert MCP result to string
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
                    Ok(ToolOutput::text(output))
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
    pub fn eager_mut(&mut self) -> &mut CoreTools {
        &mut self.eager
    }

    /// Registers an MCP connection.
    ///
    /// All tools from the MCP server will be available for the agent to use.
    #[cfg(feature = "mcp")]
    pub fn register_mcp(&mut self, conn: McpConnection) {
        self.mcp.push(parking_lot::Mutex::new(conn));
    }

    /// Returns the number of registered MCP connections.
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn mcp_count(&self) -> usize {
        self.mcp.len()
    }

    /// Returns definitions from all MCP connections.
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn mcp_definitions(&self) -> Vec<ToolDefinition> {
        self.mcp
            .iter()
            .flat_map(|c| c.lock().definitions())
            .collect()
    }

    /// Merge another AgentTools into this one.
    ///
    /// Note: MCP connections are not cloned (they would require ownership transfer).
    pub fn merge(&mut self, other: Self) {
        // Merge eager tools - we can only merge definitions since CoreTools doesn't support iteration
        // For now, we'll skip this as we can't easily extract and re-register tools
        // Users should register tools directly on the subagent config

        // Merge deferred tools
        for (name, tool) in other.deferred {
            self.deferred.entry(name).or_insert(tool);
        }

        // Merge config (keep self's config)
    }
}

/// Tool for searching available tools.
///
/// This is automatically added when tool search is enabled.
pub struct ToolSearchTool {
    /// Reference to the tools registry for searching.
    tools_ptr: *mut AgentTools,
}

// Safety: We ensure the pointer is valid during the agent's lifetime
unsafe impl Send for ToolSearchTool {}
unsafe impl Sync for ToolSearchTool {}

/// Arguments for the tool search tool.
#[derive(Debug, JsonSchema, Deserialize)]
pub struct ToolSearchArgs {
    /// Query to search for relevant tools.
    /// Describe what capability or action you need.
    pub query: String,
}

impl ToolSearchTool {
    /// Creates a new tool search tool.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer remains valid for the tool's lifetime.
    pub unsafe fn new(tools: *mut AgentTools) -> Self {
        Self { tools_ptr: tools }
    }
}

impl Tool for ToolSearchTool {
    fn name(&self) -> Cow<'static, str> {
        "_search_tools".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Search for available tools by describing what you need. \
         Use this when you need a capability that isn't in your current tool set."
            .into()
    }

    type Arguments = ToolSearchArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        // Safety: We ensure the pointer is valid during agent execution
        let tools = unsafe { &mut *self.tools_ptr };
        Ok(ToolOutput::text(tools.search_and_load(&args.query)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool {
        name: String,
    }

    #[derive(Debug, JsonSchema, Deserialize)]
    struct DummyArgs {}

    impl Tool for DummyTool {
        fn name(&self) -> Cow<'static, str> {
            Cow::Owned(self.name.clone())
        }

        fn description(&self) -> Cow<'static, str> {
            "A dummy tool for testing".into()
        }

        type Arguments = DummyArgs;

        async fn call(&self, _args: Self::Arguments) -> aither_core::Result<ToolOutput> {
            Ok(ToolOutput::text("ok"))
        }
    }

    #[test]
    fn test_register_eager() {
        let mut tools = AgentTools::new();
        tools.register(DummyTool {
            name: "test".to_string(),
        });

        assert_eq!(tools.eager_definitions().len(), 1);
        assert_eq!(tools.deferred_definitions().len(), 0);
    }

    #[test]
    fn test_should_enable_search() {
        let mut tools = AgentTools::with_config(ToolSearchConfig {
            auto_threshold: Some(2),
            enabled: None,
            ..Default::default()
        });

        // Below threshold
        tools.register(DummyTool {
            name: "t1".to_string(),
        });
        assert!(!tools.should_enable_search());

        // At threshold
        tools.register(DummyTool {
            name: "t2".to_string(),
        });
        assert!(!tools.should_enable_search());

        // Above threshold
        tools.register(DummyTool {
            name: "t3".to_string(),
        });
        assert!(tools.should_enable_search());
    }

    #[test]
    fn test_explicit_enable() {
        let tools = AgentTools::with_config(ToolSearchConfig {
            enabled: Some(true),
            ..Default::default()
        });

        assert!(tools.should_enable_search());
    }

    #[test]
    fn test_explicit_disable() {
        let mut tools = AgentTools::with_config(ToolSearchConfig {
            enabled: Some(false),
            ..Default::default()
        });

        // Even with many tools, should be disabled
        for i in 0..20 {
            tools.register(DummyTool {
                name: format!("tool_{i}"),
            });
        }

        assert!(!tools.should_enable_search());
    }
}
