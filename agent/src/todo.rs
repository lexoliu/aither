//! An ordered todo list that can be updated by the LLM as a tool.

use std::sync::{Arc, Mutex, MutexGuard};

use aither_core::llm::Tool;
use anyhow::bail;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Ordered todo list that doubles as a tool so the LLM can update progress.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoList {
    tasks: Vec<TodoListItem>,
}

impl TodoList {
    const TOOL_NAME: &'static str = "todo_list_updater";

    /// Creates a new `TodoList` from a list of task descriptions.
    #[must_use]
    pub fn new(tasks: Vec<String>) -> Self {
        Self {
            tasks: tasks
                .into_iter()
                .map(|description| TodoListItem {
                    description,
                    completed: false,
                })
                .collect(),
        }
    }

    /// Creates a new `TodoList` from a vector of items.
    #[must_use]
    pub const fn from_items(tasks: Vec<TodoListItem>) -> Self {
        Self { tasks }
    }

    /// Updates the todo list with a new set of tasks.
    pub fn update(&mut self, tasks: Vec<TodoListItem>) {
        self.tasks = tasks;
    }

    /// Marks the task at the given index as completed.
    pub fn tick(&mut self, index: usize) {
        if let Some(item) = self.tasks.get_mut(index) {
            item.completed = true;
        }
    }

    /// Marks the task with the given description as completed.
    pub fn mark_completed(&mut self, description: &str) -> bool {
        if let Some(item) = self
            .tasks
            .iter_mut()
            .find(|item| item.description == description)
        {
            item.completed = true;
            true
        } else {
            false
        }
    }

    /// Checks if all tasks in the todo list are completed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.tasks.iter().all(|item| item.completed)
    }

    /// Returns the number of tasks in the todo list.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Checks if the todo list is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Returns an iterator over the pending (not completed) tasks.
    pub fn pending(&self) -> impl Iterator<Item = &TodoListItem> {
        self.tasks.iter().filter(|task| !task.completed)
    }

    /// Returns a slice of all tasks in the todo list.
    #[must_use]
    pub fn items(&self) -> &[TodoListItem] {
        &self.tasks
    }

    /// Consumes the todo list and returns the underlying vector of tasks.
    #[must_use]
    pub fn into_items(self) -> Vec<TodoListItem> {
        self.tasks
    }
}

/// A single item in the todo list.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoListItem {
    /// The description of the task to be completed.
    description: String,
    /// Indicates whether the task has been completed.
    completed: bool,
}

impl TodoListItem {
    /// Returns the description of the task.
    #[must_use]
    pub const fn description(&self) -> &str {
        self.description.as_str()
    }

    /// Returns whether the task has been completed.
    #[must_use]
    pub const fn completed(&self) -> bool {
        self.completed
    }
}

/// Updates the agent's todo list.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub enum TodoListUpdate {
    /// Mark a task as completed by its index
    Tick {
        /// Task description that was completed
        content: String,
    },
    /// Full update with new tasks
    Replace {
        /// New list of tasks to replace the current todo list
        tasks: Vec<TodoListItem>,
    },
}

impl Tool for TodoList {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        Self::TOOL_NAME.into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Updates the agent's todo list based on completed tasks.".into()
    }
    type Arguments = TodoListUpdate;
    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        match arguments {
            TodoListUpdate::Tick { content } => {
                if !self.mark_completed(&content) {
                    bail!(
                        "Task with description '{content}' not found in todo list. Please check the task description and try again.",
                    );
                }
            }
            TodoListUpdate::Replace { tasks } => {
                self.update(tasks);
            }
        }
        Ok("Todo list updated.".into())
    }
}

/// Shared todo list that can be registered as a tool while remaining accessible.
#[derive(Debug, Clone)]
pub struct SharedTodoList {
    inner: Arc<Mutex<TodoList>>,
}

impl SharedTodoList {
    /// Creates a new shared todo list.
    #[must_use]
    pub fn new(list: TodoList) -> Self {
        Self {
            inner: Arc::new(Mutex::new(list)),
        }
    }

    /// Returns a mutex guard for accessing the todo list.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    pub fn guard(&self) -> MutexGuard<'_, TodoList> {
        self.inner.lock().expect("Shared todo list mutex poisoned")
    }

    /// Executes a function with a reference to the todo list.
    pub fn with<R>(&self, f: impl FnOnce(&TodoList) -> R) -> R {
        let guard = self.guard();
        f(&guard)
    }

    /// Executes a function with a mutable reference to the todo list.
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut TodoList) -> R) -> R {
        let mut guard = self.guard();
        f(&mut guard)
    }
}

impl Default for SharedTodoList {
    fn default() -> Self {
        Self::new(TodoList::new(Vec::new()))
    }
}

impl Tool for SharedTodoList {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        TodoList::TOOL_NAME.into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Updates the agent's todo list based on completed tasks.".into()
    }

    type Arguments = TodoListUpdate;

    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        let mut guard = self.guard();
        match arguments {
            TodoListUpdate::Tick { content } => {
                if !guard.mark_completed(&content) {
                    bail!(
                        "Task with description '{content}' not found in todo list. Please check the task description and try again.",
                    );
                }
            }
            TodoListUpdate::Replace { tasks } => {
                guard.update(tasks);
            }
        }
        drop(guard);
        Ok("Todo list updated.".into())
    }
}
