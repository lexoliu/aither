//! Subagent tool for spawning specialized subagents.
//!
//! Allows the main agent to delegate complex tasks to specialized subagents
//! that run autonomously and return results.
//!
//! Subagents can be defined in files or registered programmatically.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{Tool, ToolOutput},
};
use aither_sandbox::BashToolFactory;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::AgentBuilder;
use crate::fs_util::path_exists;
use crate::subagent_file::SubagentDefinition;

async fn checked_path_exists(path: &Path) -> anyhow::Result<bool> {
    path_exists(path)
        .await
        .map_err(|error| anyhow::anyhow!("failed to inspect path '{}': {error}", path.display()))
}

/// Builder function type for configuring a subagent.
/// Returns an `AgentBuilder` so we can add hooks before building.
/// The builder returns a single-model agent (all tiers use the same LLM).
pub type SubagentBuilder<LLM> = Arc<dyn Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync>;

/// Configuration for a subagent type.
pub struct SubagentType<LLM> {
    /// Description shown to the main agent.
    pub description: String,
    /// Builder function that creates the configured agent builder.
    builder: SubagentBuilder<LLM>,
}

#[derive(Clone, Debug)]
pub struct SubagentFileMount {
    pub virtual_prefix: PathBuf,
    pub host_root: PathBuf,
}

impl SubagentFileMount {
    #[must_use]
    pub fn new(virtual_prefix: impl Into<PathBuf>, host_root: impl Into<PathBuf>) -> Self {
        Self {
            virtual_prefix: virtual_prefix.into(),
            host_root: host_root.into(),
        }
    }
}

impl<LLM> std::fmt::Debug for SubagentType<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentType")
            .field("description", &self.description)
            .finish()
    }
}

impl<LLM: Clone> SubagentType<LLM> {
    /// Create a new subagent type with a builder function.
    ///
    /// The builder function should return an `AgentBuilder` (not a built `Agent`)
    /// so that hooks can be added before building.
    pub fn new<F>(description: impl Into<String>, builder: F) -> Self
    where
        F: Fn(LLM) -> AgentBuilder<LLM, LLM, LLM, ()> + Send + Sync + 'static,
    {
        Self {
            description: description.into(),
            builder: Arc::new(builder),
        }
    }

    /// Get the agent builder for this subagent type.
    pub fn builder(&self, llm: LLM) -> AgentBuilder<LLM, LLM, LLM, ()> {
        (self.builder)(llm)
    }
}

/// Launch a specialized subagent for complex tasks.
///
/// Subagents are autonomous agents that run in their own context window.
/// Their work does not consume your context, and they return only their
/// final results. This enables:
///
/// - Context isolation for research-heavy tasks
/// - Parallel execution of independent investigations
/// - Specialized capabilities per subagent type
///
/// Each subagent type has focused tools and prompts optimized for its purpose.
/// Choose the subagent type that best matches the task requirements.
///
/// You can also load custom subagents from `.md` files by specifying a path
/// (e.g., `./custom-agent.md` or `.subagents/researcher.md`).
#[derive(Debug, Clone, JsonSchema, Deserialize)]
pub struct SubagentArgs {
    /// Subagent type name (e.g., "explore", "plan") or path to a subagent file.
    /// If the value contains '/' or ends with '.md', it's treated as a file path.
    pub subagent: String,
    /// The detailed task prompt for the subagent.
    pub prompt: String,
}

/// A tool that spawns specialized subagents to handle complex tasks.
///
/// Allows the main agent to delegate work to specialized subagents
/// that have different capabilities and run in isolated context windows.
///
/// # Example
///
/// ```rust,ignore
/// use aither_agent::specialized::task::{SubagentTool, SubagentType};
///
/// // Create subagent tool
/// let mut subagent_tool = SubagentTool::new(llm.clone());
///
/// // Register an "explore" subagent type
/// subagent_tool.register("explore", SubagentType::new(
///     "Codebase explorer for finding files and searching code",
///     |llm| Agent::builder(llm)
///         .system_prompt("You are a codebase explorer...")
///         .tool(FsTool::new())
///         .max_iterations(15)
///         .build()
/// ));
///
/// let agent = Agent::builder(llm)
///     .tool(subagent_tool)
///     .build();
/// ```
pub struct SubagentTool<LLM> {
    llm: LLM,
    types: HashMap<String, SubagentType<LLM>>,
    /// Base directory for resolving relative paths (e.g., sandbox directory).
    base_dir: Option<PathBuf>,
    /// Ordered mount map used to resolve virtual sandbox paths into host paths.
    mounts: Vec<SubagentFileMount>,
    /// Factory for creating child bash tools for subagents.
    bash_tool_factory: Option<BashToolFactory>,
}

impl<LLM> std::fmt::Debug for SubagentTool<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_names: Vec<_> = self.types.keys().cloned().collect();
        f.debug_struct("SubagentTool")
            .field("type_names", &type_names)
            .field("base_dir", &self.base_dir)
            .finish()
    }
}

impl<LLM: Clone> SubagentTool<LLM> {
    /// Creates a new `TaskTool` with the given LLM.
    pub fn new(llm: LLM) -> Self {
        Self {
            llm,
            types: HashMap::new(),
            base_dir: None,
            mounts: Vec::new(),
            bash_tool_factory: None,
        }
    }

    /// Sets the base directory for resolving relative paths.
    ///
    /// Paths like `.subagents/...` and `.skills/...` will be resolved
    /// relative to this directory.
    #[must_use]
    pub fn with_base_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Sets explicit mount mappings for resolving subagent file paths.
    ///
    /// Mappings are checked in order and matched by virtual path prefix.
    #[must_use]
    pub fn with_file_mounts<I>(mut self, mounts: I) -> Self
    where
        I: IntoIterator<Item = SubagentFileMount>,
    {
        self.mounts = mounts.into_iter().collect();
        self
    }

    /// Sets the bash tool factory used for subagents.
    #[must_use]
    pub fn with_bash_tool_factory(mut self, factory: BashToolFactory) -> Self {
        self.bash_tool_factory = Some(factory);
        self
    }

    /// Register a subagent type.
    pub fn register(&mut self, name: impl Into<String>, subagent: SubagentType<LLM>) {
        let name = name.into();
        self.types.insert(name, subagent);
    }

    /// Register a subagent type (builder pattern).
    #[must_use]
    pub fn with_type(mut self, name: impl Into<String>, subagent: SubagentType<LLM>) -> Self {
        self.register(name, subagent);
        self
    }

    /// Returns a list of registered subagent types with their descriptions.
    ///
    /// Useful for injecting into system prompts so the main agent knows
    /// what subagents are available.
    pub fn type_descriptions(&self) -> Vec<(&str, &str)> {
        self.types
            .iter()
            .map(|(name, t)| (name.as_str(), t.description.as_str()))
            .collect()
    }

    fn effective_mounts(&self) -> Vec<SubagentFileMount> {
        if !self.mounts.is_empty() {
            return self.mounts.clone();
        }

        let base = self
            .base_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(Path::new(".")));
        vec![
            SubagentFileMount::new(PathBuf::from("."), base.clone()),
            SubagentFileMount::new(PathBuf::from(".subagents"), base.join(".subagents")),
            SubagentFileMount::new(PathBuf::from(".skills"), base.join(".skills")),
        ]
    }

    fn strip_leading_curdir(path: &Path) -> PathBuf {
        let mut stripped = PathBuf::new();
        for component in path.components() {
            if matches!(component, Component::CurDir) {
                continue;
            }
            stripped.push(component.as_os_str());
        }
        stripped
    }

    fn to_mount_relative(path: &Path, virtual_prefix: &Path) -> Option<PathBuf> {
        let normalized_path = Self::strip_leading_curdir(path);
        if normalized_path.as_os_str().is_empty() {
            return None;
        }
        if normalized_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return None;
        }

        let normalized_prefix = Self::strip_leading_curdir(virtual_prefix);
        if normalized_prefix.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return None;
        }
        if normalized_prefix.as_os_str().is_empty() {
            return Some(normalized_path);
        }

        if normalized_path.starts_with(&normalized_prefix) {
            let Ok(stripped) = normalized_path.strip_prefix(&normalized_prefix) else {
                return None;
            };
            let relative = Self::strip_leading_curdir(stripped);
            if relative.as_os_str().is_empty() {
                None
            } else {
                Some(relative)
            }
        } else {
            None
        }
    }

    fn mount_specificity(prefix: &Path) -> usize {
        Self::strip_leading_curdir(prefix).components().count()
    }

    fn matching_mounts<'a>(
        mounts: &'a [SubagentFileMount],
        file_path: &Path,
    ) -> Vec<(usize, &'a SubagentFileMount, PathBuf)> {
        let mut matched = mounts
            .iter()
            .enumerate()
            .filter_map(|(index, mount)| {
                Self::to_mount_relative(file_path, &mount.virtual_prefix)
                    .map(|relative| (index, mount, relative))
            })
            .collect::<Vec<_>>();
        matched.sort_by(|(index_a, mount_a, _), (index_b, mount_b, _)| {
            Self::mount_specificity(&mount_b.virtual_prefix)
                .cmp(&Self::mount_specificity(&mount_a.virtual_prefix))
                .then(index_a.cmp(index_b))
        });
        matched
    }

    async fn resolve_relative_path_through_mounts(
        &self,
        requested_display: &Path,
        file_path: &Path,
        mounts: &[SubagentFileMount],
    ) -> anyhow::Result<PathBuf> {
        let mut attempted = Vec::new();
        let matched = Self::matching_mounts(mounts, file_path);

        for (_, mount, relative) in matched {
            let candidate = mount.host_root.join(relative);
            let display_prefix = Self::strip_leading_curdir(&mount.virtual_prefix);
            let display_path = if display_prefix.as_os_str().is_empty() {
                candidate
                    .strip_prefix(&mount.host_root)
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|_| requested_display.to_path_buf())
            } else {
                display_prefix.join(
                    candidate
                        .strip_prefix(&mount.host_root)
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|_| requested_display.to_path_buf()),
                )
            };
            attempted.push(display_path.display().to_string());
            if checked_path_exists(&candidate).await? {
                return Ok(candidate);
            }
        }

        if attempted.is_empty() {
            return Err(anyhow::anyhow!(
                "Subagent file '{}' is outside mounted paths or uses forbidden path components",
                requested_display.display()
            ));
        }

        Err(anyhow::anyhow!(
            "Subagent file '{}' not found. Attempted: {}",
            requested_display.display(),
            attempted.join(", ")
        ))
    }

    async fn resolve_subagent_path(&self, file_path: &Path) -> anyhow::Result<PathBuf> {
        let mounts = self.effective_mounts();
        if file_path.is_absolute() {
            if let Some(base_dir) = &self.base_dir
                && let Ok(stripped) = file_path.strip_prefix(base_dir)
            {
                let relative = Self::strip_leading_curdir(stripped);
                if !relative.as_os_str().is_empty() {
                    return self
                        .resolve_relative_path_through_mounts(&relative, &relative, &mounts)
                        .await;
                }
            }

            if checked_path_exists(file_path).await? {
                return Ok(file_path.to_path_buf());
            }
            return Err(anyhow::anyhow!(
                "Subagent file '{}' not found",
                file_path.display()
            ));
        }

        self.resolve_relative_path_through_mounts(file_path, file_path, &mounts)
            .await
    }
}

impl<LLM: LanguageModel + Clone> SubagentTool<LLM> {
    /// Register a subagent from a definition (builder pattern).
    ///
    /// Creates a subagent type from a `SubagentDefinition` loaded from a file.
    /// The subagent will use the definition's system prompt and max iterations.
    #[must_use]
    pub fn with_definition(self, def: SubagentDefinition) -> Self {
        let subagent = SubagentType::new(def.description.clone(), move |llm| {
            crate::AgentBuilder::new(llm)
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations)
        });
        self.with_type(def.id, subagent)
    }

    /// Register all builtin subagents.
    ///
    /// Loads subagents from embedded definition files (explore, plan, etc.).
    #[must_use]
    pub fn with_builtins(mut self) -> Self {
        for def in crate::builtin_subagents() {
            self = self.with_definition(def);
        }
        self
    }
}

impl<LLM> Tool for SubagentTool<LLM>
where
    LLM: LanguageModel + Clone + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("subagent")
    }

    type Arguments = SubagentArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        // Determine if subagent is a file path or a registered type name
        // File paths contain '/' or end with '.md'
        let is_file_path = args.subagent.contains('/') || args.subagent.ends_with(".md");

        let (subagent_id, agent_builder) = if is_file_path {
            // Load subagent from file
            // Resolve paths using explicit search roots.
            let file_path = PathBuf::from(&args.subagent);
            let resolved_path = self.resolve_subagent_path(&file_path).await?;

            let def = SubagentDefinition::from_file_async(&resolved_path)
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to read subagent file '{}': {e}",
                        resolved_path.display()
                    )
                })?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Invalid subagent definition in '{}'",
                        resolved_path.display()
                    )
                })?;

            let builder = crate::AgentBuilder::new(self.llm.clone())
                .system_prompt(&def.system_prompt)
                .max_iterations(def.max_iterations);

            (def.id, builder)
        } else {
            // Use registered subagent type
            let type_name = &args.subagent;
            let subagent_type = self.types.get(type_name).ok_or_else(|| {
                let available: Vec<&str> =
                    self.types.keys().map(std::string::String::as_str).collect();
                anyhow::anyhow!(
                    "Unknown subagent type '{}'. Available: {}",
                    type_name,
                    available.join(", ")
                )
            })?;
            (type_name.clone(), subagent_type.builder(self.llm.clone()))
        };

        tracing::info!(subagent = %subagent_id, "Starting subagent");

        // Add child bash tool if factory is configured
        let agent_builder = if let Some(factory) = &self.bash_tool_factory {
            let dyn_bash = factory
                .create()
                .await
                .map_err(|e| anyhow::anyhow!("failed to create subagent bash tool: {e}"))?;
            agent_builder.dyn_bash(dyn_bash)
        } else {
            agent_builder
        };

        let mut agent = agent_builder.build();

        // Run the subagent with the prompt
        let result = agent
            .query(&args.prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Subagent '{subagent_id}' error: {e}"))?;

        tracing::info!(subagent = %subagent_id, "Subagent completed");

        Ok(ToolOutput::text(format!(
            "[Subagent '{subagent_id}' completed]\n\n{result}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn subagent_args_schema() {
        let schema = schemars::schema_for!(SubagentArgs);
        let value = schema.to_value();
        let props = value.get("properties").expect("should have properties");
        assert!(props.get("subagent").is_some());
        assert!(props.get("prompt").is_some());

        // Check required order
        let required = value.get("required").expect("should have required");
        assert!(required.is_array());
    }

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "aither-subagent-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create test temp dir");
        dir
    }

    #[tokio::test]
    async fn resolves_file_through_mount_prefix() {
        let root = unique_temp_dir("mount-resolve");
        let skills_root = root.join("skills");
        let skill_file = skills_root.join("slide").join("SKILL.md");
        std::fs::create_dir_all(skill_file.parent().expect("skill parent"))
            .expect("create skill parent");
        std::fs::write(&skill_file, "# slide").expect("write skill file");

        let tool = SubagentTool::new(()).with_file_mounts([
            SubagentFileMount::new(".", root.join("workspace")),
            SubagentFileMount::new(".skills", skills_root.clone()),
        ]);
        let resolved = tool
            .resolve_subagent_path(Path::new(".skills/slide/SKILL.md"))
            .await
            .expect("resolve mounted skill path");
        assert_eq!(resolved, skill_file);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn rejects_parent_traversal_paths() {
        let root = unique_temp_dir("mount-traversal");
        let tool = SubagentTool::new(())
            .with_file_mounts([SubagentFileMount::new(".skills", root.join("skills"))]);

        let error = tool
            .resolve_subagent_path(Path::new("../outside.md"))
            .await
            .expect_err("parent traversal should fail");
        assert!(
            error
                .to_string()
                .contains("outside mounted paths or uses forbidden path components"),
            "unexpected error: {error}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn missing_file_error_does_not_leak_host_mount_path() {
        let root = unique_temp_dir("mount-errors");
        let skills_root = root.join("skills");
        std::fs::create_dir_all(&skills_root).expect("create skills root");
        let tool = SubagentTool::new(())
            .with_file_mounts([SubagentFileMount::new(".skills", skills_root.clone())]);

        let error = tool
            .resolve_subagent_path(Path::new(".skills/missing.md"))
            .await
            .expect_err("missing file should fail");
        let message = error.to_string();
        assert!(
            message.contains(".skills/missing.md"),
            "missing virtual path in error: {message}"
        );
        assert!(
            !message.contains(skills_root.to_string_lossy().as_ref()),
            "host path leaked in error: {message}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn absolute_workspace_skills_path_resolves_to_skills_mount() {
        let root = unique_temp_dir("absolute-workspace-skills");
        let workspace_root = root.join("workspace");
        let skills_root = root.join("skills");
        let session_skills_file = workspace_root.join(".skills/slide/subagents/art_direction.md");
        let mounted_skill_file = skills_root.join("slide/subagents/art_direction.md");
        std::fs::create_dir_all(session_skills_file.parent().expect("session skill parent"))
            .expect("create session skill parent");
        std::fs::create_dir_all(mounted_skill_file.parent().expect("mounted skill parent"))
            .expect("create mounted skill parent");
        std::fs::write(&mounted_skill_file, "# art direction").expect("write mounted skill file");

        let tool = SubagentTool::new(())
            .with_base_dir(workspace_root.clone())
            .with_file_mounts([
                SubagentFileMount::new(".", workspace_root.clone()),
                SubagentFileMount::new(".skills", skills_root.clone()),
            ]);

        let resolved = tool
            .resolve_subagent_path(&session_skills_file)
            .await
            .expect("resolve absolute workspace .skills path");
        assert_eq!(resolved, mounted_skill_file);

        let _ = std::fs::remove_dir_all(root);
    }
}
