use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedFacts {
    pub facts: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Action {
    Add,
    Update,
    Delete,
    Noop,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryDecision {
    /// The operation to perform.
    pub action: Action,
    /// The ID of the existing memory to update or delete. Required for UPDATE and DELETE.
    pub memory_id: Option<String>,
    /// The new content for the memory. Required for UPDATE.
    pub new_content: Option<String>,
    /// Brief reasoning for the decision.
    pub reasoning: String,
}
