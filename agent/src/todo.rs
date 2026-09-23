//! Built-in todo list system for tracking long tasks.
//!
//! Provides a tool for agents to manage tasks during complex operations.
//! Designed to work like Claude Code's `TodoWrite` for tracking multi-step work.

use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::tool::ToolDefinition;

    #[test]
    fn todo_schema_has_status_enum() {
        let tool = TodoTool::new();
        let def = ToolDefinition::new(&tool);
        let schema = def.arguments_openai_schema();

        // Navigate to status field
        let schema_obj = schema.as_object().expect("schema should be object");
        let properties = schema_obj
            .get("properties")
            .expect("should have properties")
            .as_object()
            .unwrap();
        let todos = properties
            .get("todos")
            .expect("should have todos")
            .as_object()
            .unwrap();
        let items = todos
            .get("items")
            .expect("todos should have items")
            .as_object()
            .unwrap();
        let item_props = items
            .get("properties")
            .expect("item should have properties")
            .as_object()
            .unwrap();
        let status = item_props
            .get("status")
            .expect("should have status")
            .as_object()
            .unwrap();

        // Status should have enum
        assert!(
            status.contains_key("enum") || status.contains_key("type"),
            "Status should have enum or type field. Full schema: {}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }
}

use aither_core::llm::{Tool, ToolResult};
use arc_swap::ArcSwap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Status of a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Task not yet started.
    Pending,
    /// Task currently being worked on.
    InProgress,
    /// Task finished.
    Completed,
}

/// A single todo item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoItem {
    /// Description of what needs to be done (imperative form).
    /// Example: "Run tests", "Fix authentication bug"
    pub content: String,
    /// Current status of the task.
    pub status: TodoStatus,
    /// Present continuous form shown during execution.
    /// Example: "Running tests", "Fixing authentication bug"
    #[serde(rename = "activeForm")]
    pub active_form: String,
}

/// Shared todo list state.
#[derive(Debug, Clone, Default)]
pub struct TodoList {
    items: Arc<ArcSwap<Vec<TodoItem>>>,
}

impl TodoList {
    /// Creates a new empty todo list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all items in the list.
    ///
    /// # Panics
    ///
    /// Panics if the list lock was poisoned by a panic in another thread.
    #[must_use]
    pub fn items(&self) -> Vec<TodoItem> {
        self.items.load_full().as_ref().clone()
    }

    /// Replaces the entire todo list with new items.
    ///
    /// # Panics
    ///
    /// Panics if the list lock was poisoned by a panic in another thread.
    pub fn write(&self, items: Vec<TodoItem>) {
        self.items.store(Arc::new(items));
    }

    /// Clears all tasks.
    ///
    /// # Panics
    ///
    /// Panics if the list lock was poisoned by a panic in another thread.
    pub fn clear(&self) {
        self.items.store(Arc::new(Vec::new()));
    }

    /// Returns the currently in-progress task, if any.
    ///
    /// # Panics
    ///
    /// Panics if the list lock was poisoned by a panic in another thread.
    #[must_use]
    pub fn current_task(&self) -> Option<TodoItem> {
        self.items
            .load()
            .iter()
            .find(|i| i.status == TodoStatus::InProgress)
            .cloned()
    }

    /// Returns a formatted summary of progress.
    ///
    /// # Panics
    ///
    /// Panics if the list lock was poisoned by a panic in another thread.
    #[must_use]
    pub fn progress_summary(&self) -> String {
        let items = self.items.load();
        if items.is_empty() {
            return String::new();
        }

        let completed = items
            .iter()
            .filter(|i| i.status == TodoStatus::Completed)
            .count();
        let in_progress = items
            .iter()
            .filter(|i| i.status == TodoStatus::InProgress)
            .count();
        let pending = items
            .iter()
            .filter(|i| i.status == TodoStatus::Pending)
            .count();
        let total = items.len();

        let mut summary = format!("Progress: {completed}/{total} completed");
        if in_progress > 0
            && let Some(current) = items.iter().find(|i| i.status == TodoStatus::InProgress)
        {
            let _ = write!(summary, " | Current: {}", current.active_form);
        }
        if pending > 0 {
            let _ = write!(summary, " | {pending} pending");
        }
        summary
    }
}

/// Manage an in-memory structured task list for tracking progress on complex work.
///
/// This tool only updates runtime task state shown in UI/context.
/// It does NOT create or edit tasks.md on disk.
///
/// Use proactively when tasks require 3+ steps, involve multiple files,
/// or need careful organization. Updates replace the entire list.
///
/// Task states: pending, `in_progress`, completed.
/// Keep exactly one task `in_progress` at a time.
/// Mark tasks complete immediately when done.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoWriteArgs {
    /// The complete updated todo list. This replaces any existing todos.
    pub todos: Vec<TodoItem>,
}

/// Tool for managing a todo list.
///
/// This tool helps track progress on complex, multi-step tasks.
/// Use it proactively when working on tasks that require multiple steps.
#[derive(Debug, Clone)]
pub struct TodoTool {
    list: TodoList,
}

impl TodoTool {
    /// Creates a new todo tool with its own list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            list: TodoList::new(),
        }
    }

    /// Creates a todo tool sharing the given list.
    #[must_use]
    pub const fn with_list(list: TodoList) -> Self {
        Self { list }
    }

    /// Returns a reference to the underlying todo list.
    #[must_use]
    pub const fn list(&self) -> &TodoList {
        &self.list
    }
}

impl Default for TodoTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for TodoTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("todo")
    }

    type Arguments = TodoWriteArgs;
    type Res = ToolResult;

    fn call(
        &self,
        arguments: Self::Arguments,
    ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
        std::future::ready((|| {
            // Validate: at most one task should be in_progress
            let in_progress_count = arguments
                .todos
                .iter()
                .filter(|t| t.status == TodoStatus::InProgress)
                .count();

            if in_progress_count > 1 {
                return Err(anyhow::anyhow!(
                    "Only one task should be in_progress at a time, found {in_progress_count}"
                ));
            }

            self.list.write(arguments.todos);

            // TodoWrite succeeds with no output - the UI shows the todo list separately
            Ok(ToolResult::Done)
        })())
    }
}
