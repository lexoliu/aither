//! Ask command - query a fast LLM about piped content.
//!
//! Supports various input formats: text, markdown, images, PDFs.
//!
//! # Usage
//!
//! ```bash
//! cat large_file.txt | ask "summarize this"
//! websearch "rust async" | ask "what are the key patterns?"
//! ```

use std::borrow::Cow;

use askama::Template;

use aither_core::LanguageModel;
use aither_core::llm::ToolContext;
use aither_core::llm::model::Parameters;
use aither_core::llm::{Event, LLMRequest, Message, Tool, ToolResult};
use futures_lite::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Query an LLM about piped content.
///
/// Reads content from stdin and processes it with a language model.
/// Only the model's response is returned to your context - the input content
/// is not included in the response, saving context space for large inputs.
///
/// This is useful when you need to analyze, summarize, or extract information
/// from data without loading the full content into your context window.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AskArgs {
    /// The question or instruction about the input content.
    pub prompt: String,

    /// Input content (from stdin/pipe).
    #[serde(default)]
    pub input: String,

    /// Which model tier answers: `fast` (default; cheapest, lowest latency),
    /// `balanced`, or `flagship` (highest quality, for hard reasoning).
    #[serde(default)]
    pub model: AskModelTier,
}

/// Model tier selector for the ask command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AskModelTier {
    /// Highest-quality model for hard reasoning.
    Flagship,
    /// Balanced quality/cost model.
    Balanced,
    /// Cheapest, lowest-latency model (default).
    #[default]
    Fast,
}

#[derive(Template)]
#[template(path = "ask_message.txt", escape = "none")]
struct AskMessageTemplate<'a> {
    input: &'a str,
    prompt: &'a str,
}

/// The ask command tool.
///
/// Holds one LLM per selectable tier; [`AskCommand::new`] starts with the
/// same model on every tier and hosts override individual tiers via the
/// builder methods. Each request carries the parameters of the model that
/// answers it.
#[derive(Debug)]
pub struct AskCommand<LLM> {
    flagship: LLM,
    balanced: LLM,
    fast: LLM,
    /// The parameters of the model the command was created with.
    parameters: Parameters,
    /// The flagship tier's own model's parameters, once it has one.
    flagship_parameters: Option<Parameters>,
    /// The balanced tier's own model's parameters, once it has one.
    balanced_parameters: Option<Parameters>,
}

impl<LLM: Clone> AskCommand<LLM> {
    /// Creates a new ask command answering every tier with the given LLM.
    pub fn new(llm: LLM) -> Self {
        Self {
            flagship: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            parameters: Parameters::default(),
            flagship_parameters: None,
            balanced_parameters: None,
        }
    }

    /// Sets the parameters of the requests sent on the model the command
    /// was created with, on every tier that still answers with it.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Overrides the model used for `model=flagship`, whose requests carry
    /// `parameters`.
    #[must_use]
    pub fn with_flagship(mut self, llm: LLM, parameters: Parameters) -> Self {
        self.flagship = llm;
        self.flagship_parameters = Some(parameters);
        self
    }

    /// Overrides the model used for `model=balanced`, whose requests carry
    /// `parameters`.
    #[must_use]
    pub fn with_balanced(mut self, llm: LLM, parameters: Parameters) -> Self {
        self.balanced = llm;
        self.balanced_parameters = Some(parameters);
        self
    }
}

impl<LLM> AskCommand<LLM> {
    /// The model that answers `tier`, with the parameters of its requests.
    fn model_for(&self, tier: AskModelTier) -> (&LLM, &Parameters) {
        match tier {
            AskModelTier::Flagship => (
                &self.flagship,
                self.flagship_parameters
                    .as_ref()
                    .unwrap_or(&self.parameters),
            ),
            AskModelTier::Balanced => (
                &self.balanced,
                self.balanced_parameters
                    .as_ref()
                    .unwrap_or(&self.parameters),
            ),
            AskModelTier::Fast => (&self.fast, &self.parameters),
        }
    }
}

impl<LLM: LanguageModel> Tool for AskCommand<LLM> {
    fn name(&self) -> Cow<'static, str> {
        "ask".into()
    }

    type Arguments = AskArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        if args.input.is_empty() {
            return Ok(ToolResult::text("No input provided. Pipe content to ask."));
        }

        let user_content = AskMessageTemplate {
            input: &args.input,
            prompt: &args.prompt,
        }
        .render()
        .expect("ask message template must render");

        let (llm, parameters) = self.model_for(args.model);
        let request =
            LLMRequest::new(vec![Message::user(user_content)]).with_parameters(parameters.clone());
        let response = llm.respond(request);

        // Collect the response text
        futures_lite::pin!(response);
        let mut result = String::new();
        while let Some(event) = response.next().await {
            let event = event.map_err(|e| anyhow::anyhow!("{e}"))?;
            if let Event::Text(text) = event {
                result.push_str(&text);
            }
        }

        Ok(ToolResult::text(result))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use aither_core::llm::model::{Profile, ReasoningEffort};
    use futures_lite::Stream;

    use super::*;

    #[derive(Debug)]
    struct NeverError;

    impl std::fmt::Display for NeverError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("never fails")
        }
    }

    impl std::error::Error for NeverError {}

    impl aither_core::llm::Classify for NeverError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    /// A model that answers with its name and records the parameters of
    /// each request under it.
    #[derive(Clone)]
    struct RecordingLlm {
        name: &'static str,
        seen: Arc<Mutex<Vec<(&'static str, Parameters)>>>,
    }

    impl LanguageModel for RecordingLlm {
        type Error = NeverError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
            self.seen
                .lock()
                .expect("seen lock")
                .push((self.name, request.parameters().clone()));
            futures_lite::stream::iter([Ok(Event::Text(self.name.to_string()))])
        }

        fn profile(&self) -> impl Future<Output = Result<Profile, Self::Error>> + Send {
            std::future::ready(Ok(Profile::new(
                self.name,
                "test",
                self.name,
                "recording model",
                100_000,
            )))
        }
    }

    #[tokio::test]
    async fn each_tier_asks_with_the_parameters_of_its_model() {
        let seen = Arc::default();
        let model = |name| RecordingLlm {
            name,
            seen: Arc::clone(&seen),
        };
        let shared = Parameters::default().reasoning_effort(ReasoningEffort::Low);
        let flagship = Parameters::default()
            .reasoning_effort(ReasoningEffort::High)
            .websearch(true);
        let ask = AskCommand::new(model("fast"))
            .with_parameters(shared.clone())
            .with_flagship(model("flagship"), flagship.clone());

        for tier in [
            AskModelTier::Fast,
            AskModelTier::Balanced,
            AskModelTier::Flagship,
        ] {
            let args = AskArgs {
                prompt: "summarize".to_string(),
                input: "some text".to_string(),
                model: tier,
            };
            Tool::call(&ask, args, ToolContext::new())
                .await
                .expect("ask answers");
        }

        assert_eq!(
            *seen.lock().expect("seen lock"),
            [
                ("fast", shared.clone()),
                ("fast", shared),
                ("flagship", flagship),
            ],
            "a tier on the created-with model takes its parameters, one with its own model takes that model's"
        );
    }
}
