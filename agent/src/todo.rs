use aither_core::llm::Tool;
use anyhow::bail;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoList {
    tasks: Vec<TodoListItem>,
}

impl TodoList {
    pub fn new(tasks: Vec<String>) -> Self {
        Self {
            tasks: tasks
                .into_iter()
                .map(|desc| TodoListItem {
                    description: desc,
                    completed: false,
                })
                .collect(),
        }
    }

    pub fn update(&mut self, tasks: Vec<TodoListItem>) {
        self.tasks = tasks;
    }

    pub fn tick(&mut self, index: usize) {
        if let Some(item) = self.tasks.get_mut(index) {
            item.completed = true;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoListItem {
    /// The description of the task to be completed.
    description: String,
    /// Indicates whether the task has been completed.
    completed: bool,
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
        "todo_list_updater".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Updates the agent's todo list based on completed tasks.".into()
    }
    type Arguments = TodoListUpdate;
    async fn call(&mut self, arguments: Self::Arguments) -> aither_core::Result {
        match arguments {
            TodoListUpdate::Tick { content } => {
                // Find the task by content and mark it as completed
                // If task not found, LLM may generate a wrong content, return an error instead(so it would retry)

                if let Some((index, _)) = self
                    .tasks
                    .iter()
                    .enumerate()
                    .find(|(_, item)| item.description == content)
                {
                    self.tick(index);
                } else {
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
