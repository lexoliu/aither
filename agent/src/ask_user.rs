//! Ask-user tool for UI-mediated choice prompts.

use std::borrow::Cow;

use aither_core::llm::tool::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tool_request::{
    ToolRequest, ToolRequestBroker, ToolRequestQueue, channel as tool_request_channel,
};

/// Ask the user a question with predefined options.
///
/// Displays a dialog to the user with the question and options.
/// The user selects one or more options, and the selection is returned as JSON array.
///
/// Use this when you need user input to make a decision:
/// - Choosing between implementation approaches
/// - Selecting which files to modify
/// - Confirming destructive operations
/// - Getting preferences (code style, naming, etc.)
///
/// # Examples
///
/// Single choice (default):
/// ```bash
/// ask_user "Which database?" --options PostgreSQL --options MySQL --options SQLite
/// ```
///
/// Multiple choice:
/// ```bash
/// ask_user "Select features to enable:" --options Auth --options API --options UI --multi-select
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AskUserArgs {
    /// The question to ask the user.
    pub question: String,
    /// Options to choose from (use --options flag for each option).
    #[serde(alias = "choices")]
    pub options: Vec<String>,
    /// Allow selecting multiple options (use --multi-select flag).
    #[serde(default)]
    pub multi_select: bool,
}

/// A request from the agent to ask the user a question.
pub type AskUserRequest = ToolRequest<AskUserArgs, Vec<String>>;

/// Broker for ask_user requests.
pub type AskUserBroker = ToolRequestBroker<AskUserArgs, Vec<String>>;

/// Queue for pending ask_user requests.
pub type AskUserQueue = ToolRequestQueue<AskUserArgs, Vec<String>>;

/// Create a new ask_user channel pair.
#[must_use]
pub fn channel() -> (AskUserBroker, AskUserQueue) {
    tool_request_channel()
}

/// Tool implementation for asking the user questions.
#[derive(Clone)]
pub struct AskUserTool {
    broker: AskUserBroker,
}

impl AskUserTool {
    /// Create a new AskUserTool with the given sender.
    pub fn new(broker: AskUserBroker) -> Self {
        Self { broker }
    }
}

impl Tool for AskUserTool {
    fn name(&self) -> Cow<'static, str> {
        "ask_user".into()
    }

    type Arguments = AskUserArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let response = self.broker.request(args).await?;
        Ok(ToolOutput::json(&response)?)
    }
}
