//! GitHub Copilot client implementation.

use crate::{
    CopilotError,
    constant::{COPILOT_BASE_URL, COPILOT_INTEGRATION_ID, DEFAULT_MODEL, EDITOR_VERSION},
};
use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest, Message, Role, ToolCall, Usage,
        model::{
            Ability, OpenAIPromptCacheRetention, Parameters, Profile as ModelProfile, ToolChoice,
        },
        tool::ToolDefinition,
    },
};
use async_io::Timer;
use futures_core::Stream;
use futures_lite::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use zenwave::{Client, client, header};

/// GitHub Copilot language model client.
///
/// Uses the OpenAI-compatible chat completions API at `api.githubcopilot.com`.
#[derive(Clone, Debug)]
pub struct Copilot {
    inner: Arc<Config>,
}

impl Copilot {
    /// Create a new Copilot client with the given OAuth token.
    pub fn new(token: impl Into<String>) -> Self {
        Self::builder(token).build()
    }

    /// Create a builder for configuring the Copilot client.
    #[must_use]
    pub fn builder(token: impl Into<String>) -> Builder {
        Builder::new(token)
    }

    /// Override the default chat model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).model = model.into().trim().to_string();
        self
    }

    /// Override the REST base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Provide an OAuth token so the client can refresh session tokens on 401s.
    #[must_use]
    pub fn with_oauth_token(mut self, token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).oauth_token = Some(token.into());
        self
    }
}

impl LanguageModel for Copilot {
    type Error = CopilotError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let (messages, parameters, tool_defs) = request.into_parts();
        let tool_defs = filter_tool_definitions(tool_defs, &parameters.tool_choice);

        async_stream::stream! {
            if parameters.cache.claude.is_some() || parameters.cache.gemini.is_some() {
                yield Err(CopilotError::Api(
                    "Copilot provider only accepts cache.openai settings".to_string(),
                ));
                return;
            }

            let payload_messages = match to_chat_messages(&messages).await {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(CopilotError::Api(error));
                    return;
                }
            };
            let openai_tools = if tool_defs.is_empty() {
                None
            } else {
                Some(convert_tools(&tool_defs))
            };

            let mut events = chat_completions_stream(cfg, payload_messages, &parameters, openai_tools);

            while let Some(event) = events.next().await {
                yield event;
            }
        }
    }

    fn profile(&self) -> impl std::future::Future<Output = ModelProfile> + Send {
        let cfg = self.inner.clone();
        async move {
            // Try to get context window from models database
            let context_length = aither_models::lookup(&cfg.model)
                .and_then(aither_models::ModelEntry::max_input_tokens)
                .unwrap_or_else(|| {
                    const DEFAULT_CONTEXT_LENGTH: u32 = 8192;
                    tracing::warn!(
                        model = %cfg.model,
                        fallback_context_length = DEFAULT_CONTEXT_LENGTH,
                        "Copilot model missing context metadata in aither-models, using fallback",
                    );
                    DEFAULT_CONTEXT_LENGTH
                });

            ModelProfile::new(
                cfg.model.clone(),
                "GitHub Copilot",
                cfg.model.clone(),
                "GitHub Copilot model",
                context_length,
            )
            .with_ability(Ability::ToolUse)
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    token: String,
    base_url: String,
    model: String,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
}

/// Builder for [`Copilot`] clients.
#[derive(Debug)]
pub struct Builder {
    token: String,
    base_url: String,
    model: String,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
}

impl Builder {
    fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base_url: COPILOT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            editor_version: EDITOR_VERSION.to_string(),
            integration_id: COPILOT_INTEGRATION_ID.to_string(),
            oauth_token: None,
        }
    }

    /// Set a custom base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into().trim().to_string();
        self
    }

    /// Set the editor version header.
    #[must_use]
    pub fn editor_version(mut self, version: impl Into<String>) -> Self {
        self.editor_version = version.into();
        self
    }

    /// Set the integration ID header.
    #[must_use]
    pub fn integration_id(mut self, id: impl Into<String>) -> Self {
        self.integration_id = id.into();
        self
    }

    /// Provide an OAuth token so the client can refresh session tokens on 401s.
    #[must_use]
    pub fn oauth_token(mut self, token: impl Into<String>) -> Self {
        self.oauth_token = Some(token.into());
        self
    }

    /// Build the Copilot client.
    #[must_use]
    pub fn build(self) -> Copilot {
        Copilot {
            inner: Arc::new(Config {
                token: self.token,
                base_url: self.base_url,
                model: self.model,
                editor_version: self.editor_version,
                integration_id: self.integration_id,
                oauth_token: self.oauth_token,
            }),
        }
    }
}

// === Request/Response Types ===

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessagePayload>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoicePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Debug, Serialize, Clone)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize, Clone)]
struct ChatMessagePayload {
    role: &'static str,
    content: ContentPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum ContentPayload {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPayload },
}

#[derive(Debug, Clone, Serialize)]
struct ImageUrlPayload {
    url: String,
}

#[derive(Debug, Serialize, Clone)]
struct ChatToolCallPayload {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ChatToolCallFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ChatToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Clone)]
struct ToolPayload {
    r#type: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoicePayload {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize, Clone)]
struct ToolChoiceFunction {
    name: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    usage: Option<ChatCompletionUsage>,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct ChatCompletionUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokenDetails>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct PromptTokenDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct CompletionTokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)]
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<ChunkToolFunction>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// === Conversion Functions ===

async fn to_chat_messages(messages: &[Message]) -> Result<Vec<ChatMessagePayload>, String> {
    let mut payloads = Vec::with_capacity(messages.len());
    for msg in messages {
        let (role, tool_calls, tool_call_id) = match msg.role() {
            Role::System => ("system", None, None),
            Role::User => ("user", None, None),
            Role::Assistant => {
                let calls = if msg.tool_calls().is_empty() {
                    None
                } else {
                    Some(
                        msg.tool_calls()
                            .iter()
                            .map(|tc| ChatToolCallPayload {
                                id: tc.id.clone(),
                                kind: "function",
                                function: ChatToolCallFunction {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.to_string(),
                                },
                            })
                            .collect(),
                    )
                };
                ("assistant", calls, None)
            }
            Role::Tool => (
                "tool",
                None,
                Some(msg.tool_call_id().unwrap_or_default().to_string()),
            ),
        };
        payloads.push(ChatMessagePayload {
            role,
            content: build_content(msg).await?,
            tool_calls,
            tool_call_id,
        });
    }
    Ok(payloads)
}

async fn build_content(message: &Message) -> Result<ContentPayload, String> {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return Ok(ContentPayload::Text(message.content().to_owned()));
    }

    let mut parts = Vec::with_capacity(attachments.len() + 1);
    for attachment in attachments {
        let media_type = attachment.media_type().as_ref();
        if !media_type.starts_with("image/") {
            return Err(format!(
                "Copilot Chat Completions does not support attachment MIME type '{media_type}'"
            ));
        }
        let data_url = url_to_data_url(attachment.url(), media_type).await?;
        parts.push(ContentPart::ImageUrl {
            image_url: ImageUrlPayload { url: data_url },
        });
    }

    if !message.content().is_empty() {
        parts.push(ContentPart::Text {
            text: message.content().to_owned(),
        });
    }

    Ok(ContentPayload::Parts(parts))
}

async fn url_to_data_url(url: &url::Url, media_type: &str) -> Result<String, String> {
    match url.scheme() {
        "data" | "http" | "https" => Ok(url.as_str().to_string()),
        "file" => read_file_to_data_url(url, media_type).await,
        scheme => Err(format!(
            "Copilot does not support attachment URL scheme '{scheme}'"
        )),
    }
}

async fn read_file_to_data_url(url: &url::Url, media_type: &str) -> Result<String, String> {
    use base64::Engine;

    let path = url
        .to_file_path()
        .map_err(|()| "attachment file URL could not be converted to a path".to_string())?;
    let data = async_fs::read(&path)
        .await
        .map_err(|error| format!("failed to read attachment '{}': {error}", path.display()))?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
    Ok(format!("data:{media_type};base64,{base64_data}"))
}

fn convert_tools(tool_defs: &[ToolDefinition]) -> Vec<ToolPayload> {
    tool_defs
        .iter()
        .map(|def| ToolPayload {
            r#type: "function",
            function: ToolFunction {
                name: def.name().to_string(),
                description: def.description().to_string(),
                parameters: def.arguments_openai_schema(),
            },
        })
        .collect()
}

fn tool_choice(params: &Parameters, has_tools: bool) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => Some(ToolChoicePayload::Mode("auto")),
        ToolChoice::Required => Some(ToolChoicePayload::Mode("required")),
        ToolChoice::None => Some(ToolChoicePayload::Mode("none")),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Function {
            kind: "function",
            function: ToolChoiceFunction { name: name.clone() },
        }),
    }
}

fn filter_tool_definitions(defs: Vec<ToolDefinition>, choice: &ToolChoice) -> Vec<ToolDefinition> {
    match choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => defs
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => defs,
    }
}

fn prompt_cache_retention(params: &Parameters) -> Option<&'static str> {
    params
        .cache
        .openai
        .as_ref()
        .and_then(|cache| cache.retention)
        .map(OpenAIPromptCacheRetention::as_str)
}

// === Streaming ===

const SSE_FIRST_EVENT_TIMEOUT: Duration = Duration::from_secs(90);
const SSE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

async fn open_sse_stream(
    cfg: &Config,
    request: &ChatCompletionRequest,
) -> Result<zenwave::sse::SseStream, CopilotError> {
    let endpoint = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut backend = client();

    let build_result = backend
        .post(&endpoint)
        .and_then(|b| {
            b.header(
                header::AUTHORIZATION.as_str(),
                format!("Bearer {}", cfg.token),
            )
        })
        .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-copilot/0.1"))
        .and_then(|b| b.header(header::ACCEPT.as_str(), "text/event-stream"))
        .and_then(|b| b.header("editor-version", cfg.editor_version.clone()))
        .and_then(|b| b.header("Copilot-Integration-Id", cfg.integration_id.clone()));

    let builder = build_result.map_err(CopilotError::Http)?;

    match builder
        .json_body(request)
        .map_err(CopilotError::Http)?
        .sse()
        .await
    {
        Ok(stream) => Ok(stream),
        Err(zenwave::Error::Timeout) => Err(CopilotError::Timeout),
        Err(e) => Err(CopilotError::Http(e)),
    }
}

const fn is_unauthorized(err: &CopilotError) -> bool {
    matches!(
        err,
        CopilotError::Http(zenwave::Error::Http { status, .. }) if status.as_u16() == 401
    )
}

async fn refresh_session_config(cfg: &Config) -> Result<Option<Config>, CopilotError> {
    let Some(oauth_token) = cfg.oauth_token.as_deref() else {
        return Ok(None);
    };

    let session = crate::auth::get_session_token(oauth_token).await?;

    let mut refreshed = cfg.clone();
    refreshed.token = session.token;
    refreshed.base_url = session.api_endpoint;

    Ok(Some(refreshed))
}

async fn open_sse_stream_with_refresh(
    cfg: &mut Config,
    request: &ChatCompletionRequest,
) -> Result<zenwave::sse::SseStream, CopilotError> {
    match open_sse_stream(cfg, request).await {
        Ok(stream) => Ok(stream),
        Err(err) if is_unauthorized(&err) => match refresh_session_config(cfg).await? {
            Some(refreshed) => {
                *cfg = refreshed;
                open_sse_stream(cfg, request).await
            }
            None => Err(err),
        },
        Err(err) => Err(err),
    }
}

struct ProcessedSse {
    events: Vec<Event>,
    saw_payload: bool,
    done: bool,
}

enum SseStep {
    Event(Result<zenwave::sse::Event, zenwave::sse::ParseError>),
    Closed,
    IdleTimeout,
    FirstEventTimeout,
}

async fn next_sse_event(
    sse_stream: &mut zenwave::sse::SseStream,
    saw_payload: bool,
    last_progress: Instant,
) -> SseStep {
    let timeout = if saw_payload {
        SSE_IDLE_TIMEOUT
    } else {
        SSE_FIRST_EVENT_TIMEOUT
    };
    let elapsed = Instant::now().saturating_duration_since(last_progress);
    if elapsed >= timeout {
        return if saw_payload {
            SseStep::IdleTimeout
        } else {
            SseStep::FirstEventTimeout
        };
    }
    futures_lite::future::race(
        async {
            sse_stream
                .next()
                .await
                .map_or(SseStep::Closed, SseStep::Event)
        },
        async {
            Timer::after(timeout.saturating_sub(elapsed)).await;
            if saw_payload {
                SseStep::IdleTimeout
            } else {
                SseStep::FirstEventTimeout
            }
        },
    )
    .await
}

fn process_sse_data(
    data: &str,
    usage: &mut Option<Usage>,
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
    saw_finish: &mut bool,
) -> Result<ProcessedSse, CopilotError> {
    if data.is_empty() {
        return Ok(ProcessedSse {
            events: Vec::new(),
            saw_payload: false,
            done: false,
        });
    }
    if data == "[DONE]" {
        return Ok(ProcessedSse {
            events: Vec::new(),
            saw_payload: false,
            done: true,
        });
    }
    if let Ok(error_obj) = serde_json::from_str::<Value>(data)
        && let Some(error) = error_obj.get("error")
    {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown API error");
        return Err(CopilotError::Api(msg.to_string()));
    }

    let chunk: ChatCompletionChunk = serde_json::from_str(data).map_err(CopilotError::Json)?;
    let mut events = Vec::new();
    let mut saw_payload = chunk.usage.is_some();
    if let Some(chunk_usage) = &chunk.usage {
        *usage = Some(usage_from_chat_completion(chunk_usage));
    }
    if !chunk.choices.is_empty()
        && chunk
            .choices
            .iter()
            .all(|choice| choice.finish_reason.is_some())
    {
        *saw_finish = true;
    }
    for choice in &chunk.choices {
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            saw_payload = true;
            events.push(Event::Text(content.clone()));
        }
        if let Some(calls) = &choice.delta.tool_calls
            && !calls.is_empty()
        {
            saw_payload = true;
            accumulate_tool_calls(calls, tool_calls);
        }
    }
    Ok(ProcessedSse {
        events,
        saw_payload,
        done: false,
    })
}

fn accumulate_tool_calls(
    calls: &[ChunkToolCall],
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
) {
    for call in calls {
        let index = call.index.unwrap_or(0);
        let acc = tool_calls.entry(index).or_default();
        if let Some(id) = &call.id {
            acc.id = Some(id.clone());
        }
        if let Some(function) = &call.function {
            if let Some(name) = &function.name {
                acc.name = Some(name.clone());
            }
            if let Some(args) = &function.arguments {
                acc.arguments.push_str(args);
            }
        }
    }
}

fn accumulated_tool_call_events(tool_calls: HashMap<usize, ToolCallAccumulator>) -> Vec<Event> {
    let mut sorted_calls: Vec<_> = tool_calls.into_iter().collect();
    sorted_calls.sort_by_key(|(index, _)| *index);
    sorted_calls
        .into_iter()
        .filter_map(|(_, acc)| {
            let (Some(id), Some(name)) = (acc.id, acc.name) else {
                return None;
            };
            let arguments = if acc.arguments.is_empty() {
                Value::Object(serde_json::Map::default())
            } else {
                serde_json::from_str(&acc.arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::default()))
            };
            Some(Event::ToolCall(ToolCall {
                id,
                name,
                arguments,
                // Chat Completions carries no reasoning signature to round-trip.
                reasoning_state: None,
            }))
        })
        .collect()
}

fn chat_completions_stream(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    params: &Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send + Unpin {
    let params = params.clone();
    Box::pin(chat_completions_stream_inner(
        cfg,
        payload_messages,
        params,
        tools,
    ))
}

fn chat_completions_stream_inner(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    params: Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send {
    async_stream::stream! {
        let has_tools = tools.as_ref().is_some_and(|t| !t.is_empty());
        let mut cfg = cfg.as_ref().clone();
        let request = ChatCompletionRequest {
            model: cfg.model.clone(),
            messages: payload_messages,
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: params.temperature,
            top_p: params.top_p,
            max_tokens: params.max_tokens,
            tools,
            tool_choice: tool_choice(&params, has_tools),
            prompt_cache_key: params
                .cache
                .openai
                .as_ref()
                .and_then(|cache| cache.key.clone()),
            prompt_cache_retention: prompt_cache_retention(&params),
        };

        tracing::debug!(
            request = %serde_json::to_string_pretty(&request).unwrap_or_default(),
            "Sending Copilot chat completion request"
        );

        let sse_stream = match open_sse_stream_with_refresh(&mut cfg, &request).await {
            Ok(stream) => stream,
            Err(err) => {
                yield Err(err);
                return;
            }
        };

        let events = stream_chat_completion_events(sse_stream);
        futures_lite::pin!(events);
        while let Some(event) = events.next().await {
            yield event;
        }
    }
}

fn stream_chat_completion_events(
    sse_stream: zenwave::sse::SseStream,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send {
    async_stream::stream! {
        let mut event_count = 0usize;
        let mut saw_finish = false;
        let mut saw_payload = false;
        let mut last_progress = Instant::now();
        let mut tool_calls: HashMap<usize, ToolCallAccumulator> = HashMap::new();
        let mut usage: Option<Usage> = None;

        futures_lite::pin!(sse_stream);
        loop {
            let next = next_sse_event(&mut sse_stream, saw_payload, last_progress).await;
            match next {
                SseStep::Event(Ok(event)) => {
                    let data = event.text_data();
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    tracing::trace!(sse_event = %data, "Received Copilot SSE event");
                    event_count += 1;
                    let processed = match process_sse_data(
                        data,
                        &mut usage,
                        &mut tool_calls,
                        &mut saw_finish,
                    ) {
                        Ok(processed) => processed,
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    };
                    if processed.done {
                        break;
                    }
                    if processed.saw_payload {
                        saw_payload = true;
                        last_progress = Instant::now();
                    }
                    for event in processed.events {
                        yield Ok(event);
                    }
                }
                SseStep::Event(Err(err)) => {
                    yield Err(CopilotError::Stream(err));
                    return;
                }
                SseStep::Closed => break,
                SseStep::IdleTimeout => {
                    tracing::warn!("Copilot SSE idle timeout; ending stream");
                    break;
                }
                SseStep::FirstEventTimeout => {
                    yield Err(CopilotError::Timeout);
                    return;
                }
            }
            if saw_finish {
                break;
            }
        }

        tracing::debug!(event_count, "Processed Copilot SSE events");
        for event in accumulated_tool_call_events(tool_calls) {
            yield Ok(event);
        }
        if let Some(final_usage) = usage {
            yield Ok(Event::Usage(final_usage));
        }
    }
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

fn usage_from_chat_completion(raw: &ChatCompletionUsage) -> Usage {
    Usage {
        prompt_tokens: raw.prompt_tokens,
        completion_tokens: raw.completion_tokens,
        total_tokens: raw.total_tokens,
        reasoning_tokens: raw
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        cache_read_tokens: raw
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        cache_write_tokens: None,
        cost_usd: None,
        stop_reason: None,
    }
}
