//! Ask-user tool for UI-mediated choice prompts.

use std::borrow::Cow;

use askama::Template;

use aither_core::llm::ToolContext;
use aither_core::llm::tool::{Tool, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::external_hook::{self, ExternalHooks};
#[cfg(feature = "languages")]
use crate::language::LanguageGuard;
use crate::tool_request::{
    ToolRequest, ToolRequestBroker, ToolRequestQueue, channel as tool_request_channel,
};

/// The name the `ask_user` tool is called by.
pub(crate) const TOOL_NAME: &str = "ask_user";

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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QuestionAnswer {
    /// Section label (matches `Question::section`).
    pub section: String,
    /// Selected option values, or custom text.
    pub answer: AnswerValue,
}

/// Possible answer values.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum AnswerValue {
    /// User selected from predefined options.
    Selected(Vec<String>),
    /// User typed a custom answer.
    Custom(String),
    /// User skipped this question.
    Skipped,
}

impl std::fmt::Display for AnswerValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Selected(options) => write!(f, "{}", options.join(", ")),
            Self::Custom(text) => f.write_str(text),
            Self::Skipped => f.write_str("(skipped)"),
        }
    }
}

#[derive(Template)]
#[template(path = "ask_user_result.txt", escape = "none")]
struct AskUserResultTemplate<'a> {
    answers: &'a [QuestionAnswer],
}

/// A request from the agent to ask the user questions.
pub type AskUserRequest = ToolRequest<AskUserRequestPayload, Vec<QuestionAnswer>>;

/// Internal request payload sent from a session-bound tool to the host runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskUserRequestPayload {
    /// Session issuing the request.
    pub session_id: String,
    /// User-visible request body.
    pub request: AskUserArgs,
}

/// Broker for `ask_user` requests.
pub type AskUserBroker = ToolRequestBroker<AskUserRequestPayload, Vec<QuestionAnswer>>;

/// Queue for pending `ask_user` requests.
pub type AskUserQueue = ToolRequestQueue<AskUserRequestPayload, Vec<QuestionAnswer>>;

/// Create a new `ask_user` channel pair.
#[must_use]
pub fn channel() -> (AskUserBroker, AskUserQueue) {
    tool_request_channel()
}

/// Tool implementation for asking the user questions.
#[derive(Debug, Clone)]
pub struct AskUserTool {
    broker: AskUserBroker,
    session_id: String,
    hooks: Option<ExternalHooks>,
    #[cfg(feature = "languages")]
    languages: Option<LanguageGuard>,
    /// Renders its refusals.
    #[cfg_attr(not(feature = "languages"), allow(dead_code))]
    notices: crate::AnyNotices,
}

impl AskUserTool {
    /// Create a new `AskUserTool` with the given sender.
    #[must_use]
    pub fn new(session_id: impl Into<String>, broker: AskUserBroker) -> Self {
        Self {
            broker,
            session_id: session_id.into(),
            hooks: None,
            #[cfg(feature = "languages")]
            languages: None,
            notices: crate::AnyNotices::default(),
        }
    }

    /// Renders its refusals through `notices`, the agent's;
    /// [`DefaultNotices`](crate::DefaultNotices) by default.
    #[must_use]
    pub fn with_notices(mut self, notices: crate::AnyNotices) -> Self {
        self.notices = notices;
        self
    }

    /// Refuses a question, or options, not in the communication language
    /// `guard` enforces, and raises the alert that tells the agent to ask
    /// again. The user's custom answers are recorded as theirs, so the agent
    /// may quote them verbatim in its records.
    #[cfg(feature = "languages")]
    #[must_use]
    pub fn with_languages(mut self, guard: LanguageGuard) -> Self {
        self.languages = Some(guard);
        self
    }

    /// Tells `hooks` which questions were asked and how the user answered.
    #[must_use]
    pub fn with_hooks(mut self, hooks: ExternalHooks) -> Self {
        self.hooks = Some(hooks);
        self
    }
}

impl Tool for AskUserTool {
    fn name(&self) -> Cow<'static, str> {
        TOOL_NAME.into()
    }

    type Arguments = AskUserArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        #[cfg(feature = "languages")]
        if let Some(guard) = &self.languages {
            for question in args.clone().into_questions() {
                let parts = std::iter::once(question.question.as_str())
                    .chain(question.option.iter().map(String::as_str));
                if let Err(refusal) = guard.check_question(parts) {
                    return Ok(ToolResult::error(self.notices.language_refusal(&refusal)));
                }
            }
        }
        let questions = if self.hooks.is_some() {
            args.clone().into_questions()
        } else {
            Vec::new()
        };
        let response = self
            .broker
            .request(AskUserRequestPayload {
                session_id: self.session_id.clone(),
                request: args,
            })
            .await?;
        #[cfg(feature = "languages")]
        if let Some(guard) = &self.languages {
            for answer in &response {
                if let AnswerValue::Custom(text) = &answer.answer {
                    guard.record_user_message(text);
                }
            }
        }
        if let Some(hooks) = &self.hooks {
            hooks
                .fire(&external_hook::AskUser {
                    questions,
                    answers: response.clone(),
                })
                .await;
        }
        Ok(ToolResult::text(
            AskUserResultTemplate { answers: &response }
                .render()
                .expect("ask_user result template must render"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "languages")]
    fn chinese_tool() -> (AskUserTool, AskUserQueue, LanguageGuard) {
        let guard = LanguageGuard::new(crate::Languages::new("Chinese".parse().unwrap()));
        let (broker, queue) = channel();
        (
            AskUserTool::new("session", broker).with_languages(guard.clone()),
            queue,
            guard,
        )
    }

    #[cfg(feature = "languages")]
    fn ask(question: &str, options: &[&str]) -> AskUserArgs {
        AskUserArgs {
            question: Some(question.to_string()),
            option: options.iter().map(ToString::to_string).collect(),
            multi_select: false,
            questions: Vec::new(),
        }
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn an_english_question_is_refused_and_raises_an_alert() {
        let (tool, queue, guard) = chinese_tool();
        let result = tool
            .call(
                ask("Which database should we use?", &["PostgreSQL", "MySQL"]),
                ToolContext::new(),
            )
            .await
            .expect("a refusal is a tool result");
        assert!(result.is_error(), "{result:?}");
        assert!(
            queue.is_empty(),
            "a refused question never reaches the user"
        );
        assert_eq!(guard.take_alerts().len(), 1);
    }

    #[cfg(feature = "languages")]
    #[tokio::test]
    async fn a_chinese_question_with_english_code_reaches_the_user() {
        let (tool, queue, guard) = chinese_tool();
        let asked = tool.call(
            ask(
                "应该用 `tokio::spawn` 还是 spawn_blocking 运行 parse_config？",
                &["`tokio::spawn`", "spawn_blocking", "PostgreSQL"],
            ),
            ToolContext::new(),
        );
        let answered = async {
            let request = queue.next().await.expect("the question arrives");
            request
                .respond(vec![QuestionAnswer {
                    section: String::new(),
                    answer: AnswerValue::Custom("两个都不用".to_string()),
                }])
                .expect("the tool is waiting");
        };
        let (result, ()) = futures_lite::future::zip(asked, answered).await;
        assert!(!result.expect("the call succeeds").is_error());
        assert_eq!(guard.take_alerts().len(), 0);
    }

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
