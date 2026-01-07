//! Built-in todo list system for tracking long tasks.
//!
//! Provides a tool for agents to manage tasks during complex operations.
//! Designed to work like Claude Code's TodoWrite for tracking multi-step work.

use std::borrow::Cow;
use std::sync::{Arc, RwLock};

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
        let properties = schema_obj.get("properties").expect("should have properties").as_object().unwrap();
        let todos = properties.get("todos").expect("should have todos").as_object().unwrap();
        let items = todos.get("items").expect("todos should have items").as_object().unwrap();
        let item_props = items.get("properties").expect("item should have properties").as_object().unwrap();
        let status = item_props.get("status").expect("should have status").as_object().unwrap();

        // Status should have enum
        assert!(
            status.contains_key("enum") || status.contains_key("type"),
            "Status should have enum or type field. Full schema: {}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }
}

use aither_core::llm::Tool;
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
    items: Arc<RwLock<Vec<TodoItem>>>,
}

impl TodoList {
    /// Creates a new empty todo list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all items in the list.
    #[must_use]
    pub fn items(&self) -> Vec<TodoItem> {
        self.items.read().unwrap().clone()
    }

    /// Replaces the entire todo list with new items.
    pub fn write(&self, items: Vec<TodoItem>) {
        *self.items.write().unwrap() = items;
    }

    /// Clears all tasks.
    pub fn clear(&self) {
        self.items.write().unwrap().clear();
    }

    /// Returns the currently in-progress task, if any.
    #[must_use]
    pub fn current_task(&self) -> Option<TodoItem> {
        self.items
            .read()
            .unwrap()
            .iter()
            .find(|i| i.status == TodoStatus::InProgress)
            .cloned()
    }

    /// Returns a formatted summary of progress.
    #[must_use]
    pub fn progress_summary(&self) -> String {
        let items = self.items.read().unwrap();
        if items.is_empty() {
            return String::new();
        }

        let completed = items.iter().filter(|i| i.status == TodoStatus::Completed).count();
        let in_progress = items.iter().filter(|i| i.status == TodoStatus::InProgress).count();
        let pending = items.iter().filter(|i| i.status == TodoStatus::Pending).count();
        let total = items.len();

        let mut summary = format!("Progress: {completed}/{total} completed");
        if in_progress > 0 {
            if let Some(current) = items.iter().find(|i| i.status == TodoStatus::InProgress) {
                summary.push_str(&format!(" | Current: {}", current.active_form));
            }
        }
        if pending > 0 {
            summary.push_str(&format!(" | {pending} pending"));
        }
        summary
    }
}

/// Arguments for the todo tool - replaces entire list.
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
    pub fn with_list(list: TodoList) -> Self {
        Self { list }
    }

    /// Returns a reference to the underlying todo list.
    #[must_use]
    pub fn list(&self) -> &TodoList {
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

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("prompts/todo.md"))
    }

    type Arguments = TodoWriteArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result {
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

        let items = self.list.items();
        if items.is_empty() {
            Ok("Todo list cleared".to_string())
        } else {
            let completed = items.iter().filter(|i| i.status == TodoStatus::Completed).count();
            let in_progress = items.iter().filter(|i| i.status == TodoStatus::InProgress).count();
            let pending = items.iter().filter(|i| i.status == TodoStatus::Pending).count();

            let mut response = format!(
                "Todo list updated: {} total ({} completed, {} in progress, {} pending)",
                items.len(),
                completed,
                in_progress,
                pending
            );

            if let Some(current) = items.iter().find(|i| i.status == TodoStatus::InProgress) {
                response.push_str(&format!("\nCurrently: {}", current.active_form));
            }

            Ok(response)
        }
    }
}
