use std::sync::{Arc, Mutex};

use aither_core::LanguageModel;
use aither_core::llm::{Event, LLMRequest, model::Parameters, model::Profile};
use futures::Stream;
use schemars::JsonSchema;
use serde::Deserialize;

use super::{JudgmentError, judge};

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

/// Answers every structured request with `answer` and records the requests.
#[derive(Clone)]
struct Answering {
    answer: &'static str,
    requests: Arc<Mutex<Vec<LLMRequest>>>,
}

impl Answering {
    fn new(answer: &'static str) -> Self {
        Self {
            answer,
            requests: Arc::default(),
        }
    }
}

impl LanguageModel for Answering {
    type Error = NeverError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.requests.lock().unwrap().push(request);
        futures_lite::stream::iter([Ok(Event::Text(self.answer.to_string()))])
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "judge", "test", "judge", "answers", 128_000,
        )))
    }
}

/// The caller's own system prompt.
#[derive(askama::Template)]
#[template(source = "Judge whether the text is about {{ topic }}.", ext = "txt")]
struct About<'a> {
    topic: &'a str,
}

/// The caller's own answer type.
#[derive(Debug, Deserialize, JsonSchema, PartialEq, Eq)]
struct Verdict {
    relevant: bool,
    reason: String,
}

#[tokio::test]
async fn a_judgment_decodes_into_the_callers_type_from_its_rendered_prompt() {
    let model = Answering::new(r#"{"relevant":false,"reason":"it is about cooking"}"#);
    let verdict: Verdict = judge(
        &model,
        Parameters::default(),
        &About { topic: "compilers" },
        "a recipe for bread",
    )
    .await
    .expect("the judgment decodes");
    assert_eq!(
        verdict,
        Verdict {
            relevant: false,
            reason: "it is about cooking".to_string(),
        }
    );
    let request = {
        let mut requests = model.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "one request");
        requests.remove(0)
    };
    let contents: Vec<&str> = request
        .messages()
        .iter()
        .map(aither_core::llm::Message::content)
        .collect();
    assert_eq!(
        contents[..2],
        [
            "Judge whether the text is about compilers.",
            "a recipe for bread"
        ]
    );
    assert!(
        request.parameters().structured_outputs,
        "the answer is structured"
    );
}

#[tokio::test]
async fn an_answer_the_type_does_not_decode_is_an_error() {
    let model = Answering::new("not json at all");
    let result: Result<Verdict, _> = judge(
        &model,
        Parameters::default(),
        &About { topic: "compilers" },
        "anything",
    )
    .await;
    assert!(matches!(result, Err(JudgmentError::Model(_))), "{result:?}");
}
