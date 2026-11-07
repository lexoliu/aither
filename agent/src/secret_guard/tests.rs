use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use aither_core::{
    LanguageModel,
    llm::{Event, LLMRequest, Message, ReasoningState, ToolCall, model::Profile},
};
use futures_lite::{StreamExt, stream};
use serde_json::json;

use super::{SecretGuard, SecretKind, SecretShape};

// Credentials are assembled from parts so this file never holds one whole
// for a repository scanner to flag. None of them is live.
const OPENAI_KEY: &str = concat!("sk-", "proj-", "T3BlbkFJ7xQm2Vd9Lp4Rz8Kc1Wn6Yh5Gs0Aa");
const ANTHROPIC_KEY: &str = concat!("sk-", "ant-api03-", "Zq8vR2mK9xL4pN7wT1yB6cD3fG5hJ0sAeUi");
const GITHUB_TOKEN: &str = concat!("gh", "p_", "R7mK2xQ9vL4pN8wT1yB6cD3fG5hJ0sAeUiZo");
/// AWS's own documentation example.
const AWS_KEY_ID: &str = concat!("AKIA", "IOSFODNN7EXAMPLE");
const GOOGLE_KEY: &str = concat!("AI", "za", "SyD4x9Qm2Vd7Lp4Rz8Kc1Wn6Yh5Gs0AaBb3");
/// jwt.io's example token.
const JWT: &str = concat!(
    "eyJhbGciOiJIUzI1NiJ9",
    ".",
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
    ".",
    "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
);
const PRIVATE_KEY: &str = concat!(
    "-----BEGIN OPENSSH ",
    "PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ\n",
    "-----END OPENSSH PRIVATE KEY-----"
);

#[derive(Debug, Clone, Default)]
struct RecordingModel {
    requests: Arc<Mutex<Vec<LLMRequest>>>,
}

impl RecordingModel {
    fn sent(&self) -> Vec<LLMRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[derive(Debug)]
struct NeverFails;

impl std::fmt::Display for NeverFails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never fails")
    }
}

impl std::error::Error for NeverFails {}

impl aither_core::llm::Classify for NeverFails {
    fn failure(&self) -> aither_core::llm::Failure {
        aither_core::llm::Failure::permanent()
    }
}

impl LanguageModel for RecordingModel {
    type Error = NeverFails;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl futures_core::Stream<Item = Result<Event, Self::Error>> + Send {
        self.requests.lock().unwrap().push(request);
        stream::iter([Ok(Event::Text("ok".to_string()))])
    }

    fn profile(&self) -> impl Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new("mock", "mock", "test", "mock", 0)))
    }
}

/// Records each event's fields, so a test can read what was logged.
#[derive(Clone, Default)]
struct LogCapture {
    lines: Arc<Mutex<Vec<String>>>,
}

struct FieldWriter<'a>(&'a mut String);

impl tracing::field::Visit for FieldWriter<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        write!(self.0, "{}={value:?} ", field.name()).unwrap();
    }
}

impl tracing::Subscriber for LogCapture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut line = String::new();
        event.record(&mut FieldWriter(&mut line));
        self.lines.lock().unwrap().push(line);
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

/// Sends `request` through `model`, returning what was logged meanwhile.
///
/// The guard replaces secrets as the request is handed to the model, so
/// capturing that call captures its log line.
async fn send(model: &impl LanguageModel, request: LLMRequest) -> Vec<String> {
    let capture = LogCapture::default();
    let events = tracing::subscriber::with_default(capture.clone(), || model.respond(request));
    let events: Vec<_> = events.collect().await;
    assert!(events.iter().all(Result::is_ok));
    capture.lines.lock().unwrap().clone()
}

fn tool_output(content: &str) -> LLMRequest {
    LLMRequest::new([
        Message::system("You are a coding agent."),
        Message::user("Read the env file."),
        Message::assistant_with_tool_calls(
            "",
            vec![ToolCall::new(
                "call_1",
                "terminal",
                json!({ "command": "cat .env" }),
            )],
        ),
        Message::tool("call_1", content),
    ])
}

#[tokio::test]
async fn a_registered_secret_reaches_the_provider_as_its_named_placeholder() {
    let guard = SecretGuard::new();
    guard.register("SERVICE_PASSWORD", "hunter2-correct-horse");
    let provider = RecordingModel::default();
    let model = guard.guard(provider.clone());

    let logged = send(
        &model,
        tool_output("SERVICE_USER=lexo\nSERVICE_PASSWORD=hunter2-correct-horse\n"),
    )
    .await;

    let [request] = provider.sent().try_into().unwrap();
    assert_eq!(
        request.messages()[3].content(),
        "SERVICE_USER=lexo\nSERVICE_PASSWORD=<secret: SERVICE_PASSWORD>\n"
    );
    assert_eq!(logged.len(), 1, "one line per request: {logged:?}");
    assert!(logged[0].contains("SERVICE_PASSWORD"), "{logged:?}");
    assert!(!logged[0].contains("hunter2"), "the value is never logged");
}

#[tokio::test]
async fn a_secret_registered_mid_session_is_replaced_in_the_next_request() {
    // The embedder keeps this handle; the model holds a clone of it.
    let embedder = SecretGuard::new();
    let provider = RecordingModel::default();
    let model = embedder.guard(provider.clone());
    let request = tool_output("ticket pin: plaintext-session-word");

    send(&model, request.clone()).await;
    embedder.register("TICKET_PIN", "plaintext-session-word");
    send(&model, request).await;

    let [first, second] = provider.sent().try_into().unwrap();
    assert_eq!(
        first.messages()[3].content(),
        "ticket pin: plaintext-session-word"
    );
    assert_eq!(
        second.messages()[3].content(),
        "ticket pin: <secret: TICKET_PIN>"
    );
}

#[test]
fn registering_a_name_again_replaces_its_value() {
    let guard = SecretGuard::new();
    guard.register("TOKEN", "first-value-0001");
    guard.register("TOKEN", "second-value-0002");
    assert_eq!(
        guard.redact("first-value-0001 second-value-0002"),
        "first-value-0001 <secret: TOKEN>"
    );
}

#[test]
fn a_registered_value_is_named_even_where_it_has_a_key_shape() {
    let guard = SecretGuard::new();
    guard.register("OPENAI_API_KEY", OPENAI_KEY);
    let text = format!("OPENAI_API_KEY={OPENAI_KEY}\nGITHUB_TOKEN={GITHUB_TOKEN}");
    assert_eq!(
        guard.redact(&text),
        "OPENAI_API_KEY=<secret: OPENAI_API_KEY>\nGITHUB_TOKEN=<secret>"
    );
}

#[test]
fn key_shaped_strings_are_replaced() {
    let guard = SecretGuard::new();
    let cases = [
        (OPENAI_KEY.to_string(), SecretShape::ProviderKey),
        (ANTHROPIC_KEY.to_string(), SecretShape::ProviderKey),
        (GITHUB_TOKEN.to_string(), SecretShape::ProviderKey),
        (GOOGLE_KEY.to_string(), SecretShape::ProviderKey),
        (AWS_KEY_ID.to_string(), SecretShape::AwsAccessKeyId),
        (JWT.to_string(), SecretShape::Jwt),
        (PRIVATE_KEY.to_string(), SecretShape::PrivateKey),
    ];
    for (secret, shape) in cases {
        let text = format!("found {secret} in the output");
        assert_eq!(
            guard.redact(&text),
            "found <secret> in the output",
            "{shape}"
        );
        let [finding] = guard.find(&text).try_into().unwrap();
        assert_eq!(finding.kind, SecretKind::Shaped(shape));
        assert_eq!(&text[finding.range], secret);
    }
}

#[test]
fn credentials_known_by_their_context_are_replaced() {
    let guard = SecretGuard::new();
    let cases = [
        (
            "Authorization: Bearer 9f86d081884c7d659a2feaa0c55ad015",
            "Authorization: Bearer <secret>",
        ),
        (
            "export DEPLOY_TOKEN=5b2c9e7f1a3d4e6b8c0d",
            "export DEPLOY_TOKEN=<secret>",
        ),
        (
            r#"{"api_key": "q8Zr4Lm2Xv7Np1Wt9Ks3"}"#,
            r#"{"api_key": "<secret>"}"#,
        ),
        (
            "db.password: 'Pa55w0rdPa55w0rd99'",
            "db.password: '<secret>'",
        ),
    ];
    for (text, redacted) in cases {
        assert_eq!(guard.redact(text), redacted);
    }
}

#[test]
fn text_that_only_resembles_a_key_passes_unchanged() {
    let guard = SecretGuard::new();
    let base64_png = concat!(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4",
        "2mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg",
        "AKIAIOSFODNN7EXAMPLEeyJhbGciOiJIUzI1NiJ9QmFzZTY0",
        "AIzaSyD4x9Qm2Vd7Lp4Rz8Kc1Wn6Yh5Gs0AaBb3xyz=="
    );
    let texts = [
        "commit e31f3d64a1b2c3d4e5f60718293a4b5c6d7e8f90 merged into dev",
        "sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "session 550e8400-e29b-41d4-a716-446655440000 resumed",
        base64_png,
        "let ghp_count = sk_learn_model.task_token_count();",
        "max_tokens: 8192\nTOKEN_LIMIT=4096",
        r#"tokenizer = "cl100k_base_tokenizer_v2""#,
        "password: self.password.clone(),",
        "pub fn api_key(&self) -> &str { &self.api_key }",
        "Bearer authentication is described in RFC 6750.",
        r#"<div class="sk-circle-fade-delay-animation-2">"#,
        "AKIAIOSFODNN7EXAMPLEXYZ is longer than an access key ID",
        "密钥已经轮换。キーはローテーションされました。키가 교체되었습니다.",
    ];
    for text in texts {
        assert_eq!(guard.redact(text), text);
        assert!(guard.check(text).is_ok(), "{text}");
    }
}

#[test]
fn replacement_keeps_cjk_text_around_a_secret() {
    let guard = SecretGuard::new();
    guard.register("会话令牌", "plaintext-session-word");
    let text = format!(
        "密钥{OPENAI_KEY}已轮换。キーはplaintext-session-wordです。키{GITHUB_TOKEN}가 교체되었습니다."
    );
    assert_eq!(
        guard.redact(&text),
        "密钥<secret>已轮换。キーは<secret: 会话令牌>です。키<secret>가 교체되었습니다."
    );
}

#[tokio::test]
async fn signed_reasoning_and_signature_bound_calls_pass_unchanged() {
    let guard = SecretGuard::new();
    guard.register("DEPLOY_KEY", "plaintext-session-word");
    let provider = RecordingModel::default();
    let model = guard.guard(provider.clone());
    let reasoning = ReasoningState::new(
        "claude",
        r#"{"type":"thinking","thinking":"the key is plaintext-session-word","signature":"c2ln"}"#,
    );
    let signed_call = ToolCall::new(
        "call_signed",
        "terminal",
        json!({ "command": "echo plaintext-session-word" }),
    )
    .with_reasoning_state(ReasoningState::new("gemini", "c2lnbmF0dXJl"));
    let plain_call = ToolCall::new(
        "call_plain",
        "terminal",
        json!({ "command": ["echo", "plaintext-session-word"] }),
    );
    let assistant = Message::assistant_with_reasoning(
        "Using plaintext-session-word now.",
        vec![signed_call.clone(), plain_call],
        vec![reasoning.clone()],
    );

    send(&model, LLMRequest::new([assistant])).await;

    let [request] = provider.sent().try_into().unwrap();
    let message = &request.messages()[0];
    assert_eq!(message.content(), "Using <secret: DEPLOY_KEY> now.");
    assert_eq!(message.reasoning(), [reasoning]);
    assert_eq!(message.tool_calls()[0], signed_call);
    assert_eq!(
        message.tool_calls()[1].arguments,
        json!({ "command": ["echo", "<secret: DEPLOY_KEY>"] })
    );
}

#[tokio::test]
async fn structured_generation_is_guarded() {
    let guard = SecretGuard::new();
    guard.register("DEPLOY_KEY", "plaintext-session-word");
    let provider = RecordingModel::default();
    let model = guard.guard(provider.clone());

    let _: Result<String, _> = model
        .generate(tool_output("key: plaintext-session-word"))
        .await;

    let [request] = provider.sent().try_into().unwrap();
    assert_eq!(request.messages()[3].content(), "key: <secret: DEPLOY_KEY>");
}

#[test]
fn check_refuses_text_carrying_a_secret_without_quoting_it() {
    let guard = SecretGuard::new();
    guard.register("DEPLOY_KEY", "plaintext-session-word");
    let text = format!("deploy with plaintext-session-word and {GITHUB_TOKEN}");

    let refusal = guard.check(&text).unwrap_err();

    let kinds: Vec<_> = refusal.findings.iter().map(|f| f.kind.clone()).collect();
    assert_eq!(
        kinds,
        [
            SecretKind::Registered {
                name: "DEPLOY_KEY".into()
            },
            SecretKind::Shaped(SecretShape::ProviderKey),
        ]
    );
    let message = crate::Notices::secret_refusal(&crate::DefaultNotices, &refusal);
    assert_eq!(
        message,
        "Refused: the text carries secrets (secret DEPLOY_KEY, provider API key). \
         Remove them before publishing."
    );
    assert!(!format!("{guard:?}").contains("plaintext"));
}

#[test]
#[should_panic(expected = "secret EMPTY is registered empty")]
fn an_empty_secret_is_a_mistake() {
    SecretGuard::new().register("EMPTY", "");
}
