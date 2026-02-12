//! Ask-user tool for UI-mediated choice prompts.

use std::borrow::Cow;

use aither_core::llm::tool::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tool_request::{
    ToolRequest, ToolRequestBroker, ToolRequestQueue, channel as tool_request_channel,
};

/// A single question within an `ask_user` request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Question {
    /// Short section label for navigation (e.g. "database", "theme").
    pub section: String,
    /// The question text to display.
    pub question: String,
    /// Options to choose from.
    #[serde(alias = "options")]
    pub option: Vec<String>,
    /// Allow selecting multiple options.
    #[serde(default)]
    pub multi_select: bool,
}

/// Ask the user one or more questions with predefined options.
///
/// Displays an interactive prompt where the user picks from options,
/// types a custom answer, or skips the question.
///
/// # Single question (simple form)
///
/// ```bash
/// ask_user "Which database?" --option PostgreSQL --option MySQL --option SQLite
/// ```
///
/// # Multiple questions (use --questions with JSON)
///
/// ```bash
/// ask_user --questions '[
///   {"section":"database","question":"Which DB?","option":["PostgreSQL","MySQL","SQLite"]},
///   {"section":"theme","question":"Color theme?","option":["Light","Dark","System"]}
/// ]'
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AskUserArgs {
    /// The question to ask (for single-question mode).
    #[serde(default)]
    pub question: Option<String>,
    /// Option to choose from (use --option flag for each, repeatable).
    #[serde(default, alias = "options", alias = "choices")]
    pub option: Vec<String>,
    /// Allow selecting multiple options (for single-question mode).
    #[serde(default)]
    pub multi_select: bool,
    /// Multiple questions (JSON array). Overrides `question/options/multi_select` if set.
    #[serde(default)]
    pub questions: Vec<Question>,
}

impl AskUserArgs {
    /// Normalize into a list of questions.
    /// If `questions` is non-empty, use that directly.
    /// Otherwise, build a single question from the flat fields.
    #[must_use] 
    pub fn into_questions(self) -> Vec<Question> {
        if !self.questions.is_empty() {
            return self.questions;
        }
        if let Some(q) = self.question {
            vec![Question {
                section: String::new(),
                question: q,
                option: self.option,
                multi_select: self.multi_select,
            }]
        } else {
            Vec::new()
        }
    }
}

/// The user's answer to a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnswer {
    /// Section label (matches `Question::section`).
    pub section: String,
    /// Selected option values, or custom text.
    pub answer: AnswerValue,
}

/// Possible answer values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnswerValue {
    /// User selected from predefined options.
    Selected(Vec<String>),
    /// User typed a custom answer.
    Custom(String),
    /// User skipped this question.
    Skipped,
}

/// A request from the agent to ask the user questions.
pub type AskUserRequest = ToolRequest<AskUserArgs, Vec<QuestionAnswer>>;

/// Broker for `ask_user` requests.
pub type AskUserBroker = ToolRequestBroker<AskUserArgs, Vec<QuestionAnswer>>;

/// Queue for pending `ask_user` requests.
pub type AskUserQueue = ToolRequestQueue<AskUserArgs, Vec<QuestionAnswer>>;

/// Create a new `ask_user` channel pair.
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
    /// Create a new `AskUserTool` with the given sender.
    #[must_use] 
    pub const fn new(broker: AskUserBroker) -> Self {
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
        ToolOutput::json(&response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deser_option_array() {
        let json = serde_json::json!({
            "question": "请选择你喜欢的水果？",
            "option": ["苹果", "香蕉", "橘子", "梨"]
        });
        let args: AskUserArgs = serde_json::from_value(json).unwrap();
        assert_eq!(args.option.len(), 4);
        assert_eq!(args.option, vec!["苹果", "香蕉", "橘子", "梨"]);
        let questions = args.into_questions();
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].option.len(), 4);
    }

    #[test]
    fn test_schema_option_is_array() {
        let schema = schemars::schema_for!(AskUserArgs);
        let val = serde_json::to_value(schema).unwrap();
        tracing::debug!(
            "AskUserArgs schema:\n{}",
            serde_json::to_string_pretty(&val).unwrap()
        );

        let props = val.get("properties").unwrap().as_object().unwrap();
        let opt = props.get("option").unwrap();
        tracing::debug!(
            "option field schema: {}",
            serde_json::to_string_pretty(opt).unwrap()
        );

        let opt_type = opt.get("type").and_then(|t| t.as_str());
        assert_eq!(
            opt_type,
            Some("array"),
            "option must be array type in schema"
        );
    }

    #[test]
    fn test_cli_to_json_with_real_schema() {
        let schema = schemars::schema_for!(AskUserArgs);
        let schema_val = serde_json::to_value(schema).unwrap();

        let result = aither_sandbox::cli_to_json(
            &schema_val,
            &[
                "--question".into(),
                "请选择你喜欢的水果？".into(),
                "--option".into(),
                "苹果".into(),
                "--option".into(),
                "香蕉".into(),
                "--option".into(),
                "橘子".into(),
                "--option".into(),
                "梨".into(),
            ],
        )
        .unwrap();

        tracing::debug!(
            "cli_to_json result: {}",
            serde_json::to_string_pretty(&result).unwrap()
        );

        let args: AskUserArgs = serde_json::from_value(result).unwrap();
        assert_eq!(
            args.option.len(),
            4,
            "expected 4 options, got: {:?}",
            args.option
        );
        assert_eq!(args.option, vec!["苹果", "香蕉", "橘子", "梨"]);
    }
}
