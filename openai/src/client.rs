use crate::{
    DEEPSEEK_BASE_URL, DEFAULT_AUDIO_FORMAT, DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_VOICE,
    DEFAULT_BASE_URL, DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL, DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL, DEFAULT_MODERATION_MODEL, DEFAULT_TRANSCRIPTION_MODEL, OPENROUTER_BASE_URL,
    error::OpenAIError,
    request::{
        ChatCompletionRequest, ChatMessagePayload, ParameterSnapshot, ResponsesInputItem,
        ResponsesRequest, ResponsesTool, ToolPayload, convert_responses_tools, convert_tools,
        responses_tool_choice, to_chat_messages, to_responses_input,
    },
    response::{ChatCompletionChunk, ResponsesOutputItem, ResponsesStreamEvent, should_skip_event},
};
use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest, ToolCall,
        model::{Ability, Profile as ModelProfile, ToolChoice},
        oneshot,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use std::{future::Future, sync::Arc, time::Duration};
use zenwave::{Client, client, header};

/// Configuration for request retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Initial delay before first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier for exponential backoff.
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a config with no retries.
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Calculate delay for a given attempt number (0-indexed).
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_ms =
            self.initial_delay.as_millis() as f64 * self.backoff_multiplier.powi(attempt as i32);
        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }
}

/// Check if an error is retryable.
const fn is_retryable_error(err: &OpenAIError) -> bool {
    match err {
        // Network/transport errors are retryable
        OpenAIError::Http(_) => true,
        // Body errors (connection issues) may be retryable
        OpenAIError::Body(_) => true,
        // SSE stream errors may be retryable
        OpenAIError::Stream(_) => true,
        // Rate limit and server errors are retryable
        OpenAIError::RateLimit { .. } => true,
        OpenAIError::ServerError { .. } => true,
        // Timeout is retryable
        OpenAIError::Timeout => true,
        // API errors are generally not retryable (bad request, auth, etc.)
        OpenAIError::Api(_) => false,
        // Parse errors are not retryable
        OpenAIError::Json(_) => false,
        // Decode errors are not retryable
        OpenAIError::Decode(_) => false,
    }
}

/// Get retry delay for an error, respecting Retry-After header for rate limits.
fn get_retry_delay(err: &OpenAIError, attempt: u32, config: &RetryConfig) -> Duration {
    if let OpenAIError::RateLimit {
        retry_after: Some(delay),
        ..
    } = err
    {
        // Respect Retry-After header, but cap at max_delay
        return (*delay).min(config.max_delay);
    }
    config.delay_for_attempt(attempt)
}

/// Sleep for the given duration (runtime-agnostic).
async fn sleep(duration: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        async_io::Timer::after(duration).await;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = duration;
    }
}

/// Result of attempting to establish an SSE stream.
type SseStreamResult =
    Result<Vec<Result<zenwave::sse::Event, zenwave::sse::ParseError>>, OpenAIError>;

/// Attempt to make an SSE request with retry logic.
///
/// Returns the collected SSE events on success, or the last error on failure.
async fn sse_request_with_retry<F, Fut>(cfg: &Config, make_request: F) -> SseStreamResult
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = SseStreamResult>,
{
    let retry_config = &cfg.retry;
    let mut attempt = 0;

    loop {
        match make_request().await {
            Ok(events) => return Ok(events),
            Err(err) => {
                // Check if we should retry
                if attempt < retry_config.max_retries && is_retryable_error(&err) {
                    let delay = get_retry_delay(&err, attempt, retry_config);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = retry_config.max_retries,
                        delay_ms = delay.as_millis(),
                        error = %err,
                        "Request failed, retrying"
                    );
                    sleep(delay).await;
                    attempt += 1;
                } else {
                    return Err(err);
                }
            }
        }
    }
}

/// `OpenAI` model backed by the Responses API by default, with legacy
/// `chat.completions` support for compatibility.
#[derive(Clone, Debug)]
pub struct OpenAI {
    inner: Arc<Config>,
}

/// Selects which `OpenAI` API surface to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiKind {
    /// The recommended Responses API.
    Responses,
    /// Legacy Chat Completions API (deprecated by `OpenAI`).
    ChatCompletions,
}

impl OpenAI {
    /// Create a new client using the provided API key and default model.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder(api_key).build()
    }

    /// Create a client configured for [`Deepseek`](https://api-docs.deepseek.com)'s OpenAI-compatible endpoint.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::builder(api_key).base_url(DEEPSEEK_BASE_URL).build()
    }

    /// Create a client configured for [`OpenRouter`](https://openrouter.ai)'s OpenAI-compatible endpoint.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::builder(api_key).base_url(OPENROUTER_BASE_URL).build()
    }

    /// Start building an [`OpenAI`] client with custom configuration.
    #[must_use]
    pub fn builder(api_key: impl Into<String>) -> Builder {
        Builder::new(api_key)
    }

    /// Override the default chat model in-place.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).chat_model = sanitize_model(model);
        self
    }

    /// Override the REST base URL (useful for OpenAI-compatible endpoints).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Select which `OpenAI` API to call.
    #[must_use]
    pub fn with_api(mut self, api: ApiKind) -> Self {
        Arc::make_mut(&mut self.inner).api_kind = api;
        self
    }

    /// Use the recommended Responses API.
    #[must_use]
    pub fn with_responses_api(self) -> Self {
        self.with_api(ApiKind::Responses)
    }

    /// Use the legacy Chat Completions API (deprecated by `OpenAI`).
    #[must_use]
    pub fn with_chat_completions_api(self) -> Self {
        self.with_api(ApiKind::ChatCompletions)
    }

    /// Send deprecated `max_tokens` alongside `max_completion_tokens` for compatibility.
    ///
    /// `OpenAI` deprecates `max_tokens` and it is incompatible with reasoning models.
    #[must_use]
    pub fn with_legacy_max_tokens(mut self, enabled: bool) -> Self {
        Arc::make_mut(&mut self.inner).legacy_max_tokens = enabled;
        self
    }

    /// Override the embeddings model identifier.
    #[must_use]
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        let model = sanitize_model(model);
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.embedding_model = model;
        if let Some(dim) = infer_embedding_dim(&cfg.embedding_model) {
            cfg.embedding_dimensions = dim;
        }
        self
    }

    /// Override the embedding dimension (defaults depend on model).
    #[must_use]
    pub fn with_embedding_dimensions(mut self, dimensions: usize) -> Self {
        Arc::make_mut(&mut self.inner).embedding_dimensions = dimensions;
        self
    }

    /// Override the image generation model identifier.
    #[must_use]
    pub fn with_image_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).image_model = sanitize_model(model);
        self
    }

    /// Override the default text-to-speech model identifier.
    #[must_use]
    pub fn with_audio_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).audio_model = sanitize_model(model);
        self
    }

    /// Set the default text-to-speech voice (e.g., `alloy`).
    #[must_use]
    pub fn with_audio_voice(mut self, voice: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).audio_voice = voice.into();
        self
    }

    /// Set the preferred audio format (`mp3`, `wav`, `flac`).
    #[must_use]
    pub fn with_audio_format(mut self, format: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).audio_format = format.into();
        self
    }

    /// Override the transcription model identifier.
    #[must_use]
    pub fn with_transcription_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).transcription_model = sanitize_model(model);
        self
    }

    /// Override the moderation model identifier.
    #[must_use]
    pub fn with_moderation_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).moderation_model = sanitize_model(model);
        self
    }

    pub(crate) fn config(&self) -> Arc<Config> {
        self.inner.clone()
    }
}

impl LanguageModel for OpenAI {
    type Error = OpenAIError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let (messages, parameters, tool_defs) = request.into_parts();
        let tool_defs = filter_tool_definitions(tool_defs, &parameters.tool_choice);
        let mut snapshot = ParameterSnapshot::from(&parameters);
        snapshot.legacy_max_tokens = cfg.legacy_max_tokens;
        let has_attachments = messages.iter().any(|msg| !msg.attachments().is_empty());

        async_stream::stream! {
            if cfg.api_kind == ApiKind::ChatCompletions && has_attachments {
                yield Err(OpenAIError::Api(
                    "Chat Completions does not support file attachments; use Responses API".to_string(),
                ));
                return;
            }

            let messages = if has_attachments {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    match crate::attachments::resolve_messages(&cfg, messages).await {
                        Ok(resolved) => resolved,
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    yield Err(OpenAIError::Api(
                        "file:// attachments are not supported on wasm32 targets".to_string(),
                    ));
                    return;
                }
            } else {
                messages
            };

            match cfg.api_kind {
                ApiKind::ChatCompletions => {
                    let payload_messages = to_chat_messages(&messages);
                    let openai_tools = if tool_defs.is_empty() {
                        None
                    } else {
                        Some(convert_tools(tool_defs))
                    };

                    let mut events = chat_completions_stream(
                        cfg,
                        payload_messages,
                        snapshot,
                        openai_tools,
                    );

                    while let Some(event) = events.next().await {
                        yield event;
                    }
                }
                ApiKind::Responses => {
                    let responses_input = match to_responses_input(&messages) {
                        Ok(input) => input,
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    };

                    let mut events = responses_stream(
                        cfg,
                        responses_input,
                        snapshot,
                        tool_defs,
                    );

                    while let Some(event) = events.next().await {
                        yield event;
                    }
                }
            }
        }
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.respond(oneshot(
            "Continue the user provided text without additional commentary.",
            prefix,
        ))
    }

    fn profile(&self) -> impl Future<Output = ModelProfile> + Send {
        let cfg = self.inner.clone();
        async move {
            // Try to fetch context window from API, fallback to models database
            let context_length = match fetch_model_context_length(&cfg).await {
                Ok(len) => len,
                Err(e) => {
                    tracing::debug!("API did not return context_length: {e}");
                    // Fallback to models database
                    aither_models::lookup(&cfg.chat_model)
                        .map(|info| info.context_window)
                        .unwrap_or_else(|| {
                            tracing::warn!(
                                "Model '{}' not found in database, using default 128k",
                                cfg.chat_model
                            );
                            128_000
                        })
                }
            };

            let mut profile = ModelProfile::new(
                cfg.chat_model.clone(),
                "OpenAI",
                cfg.chat_model.clone(),
                "OpenAI GPT family model",
                context_length,
            )
            .with_ability(Ability::ToolUse);
            for ability in &cfg.native_abilities {
                if !profile.abilities.contains(ability) {
                    profile.abilities.push(*ability);
                }
            }
            profile
        }
    }
}

/// Fetch context window size from the models API.
async fn fetch_model_context_length(cfg: &Config) -> Result<u32, OpenAIError> {
    use crate::response::ModelsListResponse;

    let url = format!("{}/models", cfg.base_url.trim_end_matches('/'));
    let mut backend = client();
    let response: ModelsListResponse = backend
        .get(&url)
        .map_err(OpenAIError::Http)?
        .header(
            header::AUTHORIZATION.as_str(),
            format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::Http)?
        .json()
        .await
        .map_err(OpenAIError::Http)?;

    // Find matching model
    let mut model_found = false;
    for model in response.data {
        if model.id == cfg.chat_model {
            model_found = true;
            if let Some(ctx) = model.context_length.or(model.max_tokens) {
                tracing::debug!(
                    "Fetched context_length={} for model '{}'",
                    ctx,
                    cfg.chat_model
                );
                return Ok(ctx);
            }
        }
    }

    // More specific error message
    if model_found {
        Err(OpenAIError::Api(format!(
            "Model '{}' found but missing context_length field (proxy may not support it)",
            cfg.chat_model
        )))
    } else {
        Err(OpenAIError::Api(format!(
            "Model '{}' not found in /models response",
            cfg.chat_model
        )))
    }
}

fn chat_completions_stream(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    snapshot: ParameterSnapshot,
    openai_tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, OpenAIError>> + Send + Unpin {
    Box::pin(chat_completions_stream_inner(
        cfg,
        payload_messages,
        snapshot,
        openai_tools,
    ))
}

/// Accumulated tool call state for streaming.
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Make a chat completions SSE request (single attempt).
async fn chat_completions_request(
    cfg: &Config,
    request: &ChatCompletionRequest,
) -> SseStreamResult {
    let endpoint = cfg.request_url("/chat/completions");
    let mut backend = client();

    let build_result = backend
        .post(endpoint)
        .and_then(|b| b.header(header::AUTHORIZATION.as_str(), cfg.request_auth()))
        .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-openai/0.1"))
        .and_then(|b| b.header(header::ACCEPT.as_str(), "text/event-stream"));

    let mut builder = build_result.map_err(OpenAIError::Http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org.clone())
            .map_err(OpenAIError::Http)?;
    }

    let sse_stream = match builder
        .json_body(request)
        .map_err(OpenAIError::Http)?
        .sse()
        .await
    {
        Ok(stream) => stream,
        Err(zenwave::Error::Timeout) => return Err(OpenAIError::Timeout),
        Err(e) => return Err(OpenAIError::Http(e)),
    };

    let events: Vec<_> = sse_stream
        .filter_map(|event| match event {
            Ok(e) if !should_skip_event(&e) && e.text_data() != "[DONE]" => Some(Ok(e)),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
        .collect()
        .await;

    Ok(events)
}

fn chat_completions_stream_inner(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    snapshot: ParameterSnapshot,
    openai_tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, OpenAIError>> + Send {
    let include_reasoning = snapshot.include_reasoning;

    async_stream::stream! {
        let request = ChatCompletionRequest::new(
            cfg.chat_model.clone(),
            payload_messages,
            &snapshot,
            openai_tools,
            true,
        );

        tracing::debug!(request = %serde_json::to_string_pretty(&request).unwrap_or_default(), "Sending chat completion request");

        // Make request with retry
        let events = match sse_request_with_retry(&cfg, || chat_completions_request(&cfg, &request)).await {
            Ok(events) => events,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        tracing::debug!(event_count = events.len(), "Collected SSE events");

        if events.is_empty() {
            tracing::warn!("No SSE events received from API");
        }

        // Accumulate tool calls by index - streaming sends id/name first, then arguments incrementally
        let mut tool_calls: std::collections::HashMap<usize, ToolCallAccumulator> =
            std::collections::HashMap::new();
        let mut _text_yielded = false;

        for event in events {
            match event {
                Ok(e) => {
                    let data = e.text_data();
                    tracing::debug!(sse_event = %data, "Received SSE event");

                    // Check for API error response
                    if let Ok(error_obj) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(error) = error_obj.get("error") {
                            let msg = error.get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown API error");
                            yield Err(OpenAIError::Api(msg.to_string()));
                            return;
                        }
                    }

                    match serde_json::from_str::<ChatCompletionChunk>(data) {
                        Ok(chunk) => {
                            // Emit text events
                            for choice in &chunk.choices {
                                // Check for malformed function call
                                if let Some(ref reason) = choice.finish_reason {
                                    if reason.contains("malformed_function_call") {
                                        yield Err(OpenAIError::Api("malformed function call".to_string()));
                                        return;
                                    }
                                }

                                if let Some(content) = &choice.delta.content {
                                    if !content.is_empty() {
                                        yield Ok(Event::Text(content.clone()));
                                    }
                                }
                                // Emit reasoning if enabled
                                if include_reasoning {
                                    if let Some(reasoning) = &choice.delta.reasoning_content {
                                        if !reasoning.is_empty() {
                                            yield Ok(Event::Reasoning(reasoning.clone()));
                                        }
                                    }
                                }
                                // Accumulate tool calls
                                if let Some(calls) = &choice.delta.tool_calls {
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
                            }
                        }
                        Err(e) => {
                            yield Err(OpenAIError::from(e));
                            return;
                        }
                    }
                }
                Err(e) => {
                    yield Err(OpenAIError::from(e));
                    return;
                }
            }
        }

        // Emit accumulated tool calls at the end
        let mut sorted_calls: Vec<_> = tool_calls.into_iter().collect();
        sorted_calls.sort_by_key(|(index, _)| *index);
        for (_, acc) in sorted_calls {
            if let (Some(id), Some(name)) = (acc.id, acc.name) {
                let arguments = if acc.arguments.is_empty() {
                    serde_json::Value::Object(Default::default())
                } else {
                    serde_json::from_str(&acc.arguments)
                        .unwrap_or(serde_json::Value::Object(Default::default()))
                };
                yield Ok(Event::ToolCall(ToolCall {
                    id,
                    name,
                    arguments,
                }));
            }
        }
    }
}

fn responses_stream(
    cfg: Arc<Config>,
    input: Vec<ResponsesInputItem>,
    snapshot: ParameterSnapshot,
    tool_defs: Vec<aither_core::llm::tool::ToolDefinition>,
) -> impl Stream<Item = Result<Event, OpenAIError>> + Send + Unpin {
    Box::pin(responses_stream_inner(cfg, input, snapshot, tool_defs))
}

/// Accumulated function call state for Responses API streaming.
#[derive(Debug, Default)]
struct FunctionCallAccumulator {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Make a responses API SSE request (single attempt).
async fn responses_request(cfg: &Config, request: &ResponsesRequest) -> SseStreamResult {
    let endpoint = cfg.request_url("/responses");
    let mut backend = client();

    let build_result = backend
        .post(endpoint)
        .and_then(|b| b.header(header::AUTHORIZATION.as_str(), cfg.request_auth()))
        .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-openai/0.1"))
        .and_then(|b| b.header(header::ACCEPT.as_str(), "text/event-stream"));

    let mut builder = build_result.map_err(OpenAIError::Http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org.clone())
            .map_err(OpenAIError::Http)?;
    }

    let sse_stream = match builder
        .json_body(request)
        .map_err(OpenAIError::Http)?
        .sse()
        .await
    {
        Ok(stream) => stream,
        Err(zenwave::Error::Timeout) => return Err(OpenAIError::Timeout),
        Err(e) => return Err(OpenAIError::Http(e)),
    };

    let events: Vec<_> = sse_stream
        .filter_map(|event| match event {
            Ok(e) if !should_skip_event(&e) && e.text_data() != "[DONE]" => Some(Ok(e)),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
        .collect()
        .await;

    Ok(events)
}

fn responses_stream_inner(
    cfg: Arc<Config>,
    input: Vec<ResponsesInputItem>,
    snapshot: ParameterSnapshot,
    tool_defs: Vec<aither_core::llm::tool::ToolDefinition>,
) -> impl Stream<Item = Result<Event, OpenAIError>> + Send {
    let include_reasoning = snapshot.include_reasoning;

    async_stream::stream! {
        let mut response_tools = convert_responses_tools(tool_defs);
        let allow_builtins = matches!(snapshot.tool_choice, ToolChoice::Auto | ToolChoice::Required);
        if allow_builtins {
            if snapshot.websearch {
                response_tools.push(ResponsesTool::WebSearch);
            }
            if snapshot.code_execution {
                response_tools.push(ResponsesTool::CodeInterpreter);
            }
        }
        let response_tools = if response_tools.is_empty() {
            None
        } else {
            Some(response_tools)
        };
        let has_tools = response_tools.is_some();
        let tool_choice = responses_tool_choice(&snapshot, has_tools);

        let request = ResponsesRequest::new(
            cfg.chat_model.clone(),
            input,
            &snapshot,
            response_tools,
            tool_choice,
            true, // stream: true
        );

        tracing::debug!(request = %serde_json::to_string_pretty(&request).unwrap_or_default(), "Sending responses request");

        // Make request with retry
        let events = match sse_request_with_retry(&cfg, || responses_request(&cfg, &request)).await {
            Ok(events) => events,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        tracing::debug!(event_count = events.len(), "Collected Responses API SSE events");

        // Accumulate function calls by item_id
        let mut function_calls: std::collections::HashMap<String, FunctionCallAccumulator> =
            std::collections::HashMap::new();

        for event in events {
            match event {
                Ok(e) => {
                    let data = e.text_data();
                    tracing::trace!(sse_event = %data, "Received Responses API SSE event");

                    // Check for API error response
                    if let Ok(error_obj) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(error) = error_obj.get("error") {
                            let msg = error.get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown API error");
                            yield Err(OpenAIError::Api(msg.to_string()));
                            return;
                        }
                    }

                    match serde_json::from_str::<ResponsesStreamEvent>(data) {
                        Ok(stream_event) => {
                            match stream_event {
                                ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
                                    if !delta.is_empty() {
                                        yield Ok(Event::Text(delta));
                                    }
                                }
                                ResponsesStreamEvent::ReasoningTextDelta { delta, .. } |
                                ResponsesStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
                                    if include_reasoning && !delta.is_empty() {
                                        yield Ok(Event::Reasoning(delta));
                                    }
                                }
                                ResponsesStreamEvent::OutputItemAdded { item, .. } => {
                                    // When a function_call item is added, capture id and name
                                    if let ResponsesOutputItem::FunctionCall { id, call_id, name, .. } = item {
                                        let acc = function_calls.entry(id.clone()).or_default();
                                        acc.call_id = call_id.or(Some(id));
                                        acc.name = Some(name);
                                    }
                                }
                                ResponsesStreamEvent::FunctionCallArgumentsDelta { delta, item_id, .. } => {
                                    let acc = function_calls.entry(item_id).or_default();
                                    acc.arguments.push_str(&delta);
                                }
                                ResponsesStreamEvent::FunctionCallArgumentsDone { arguments, item_id, .. } => {
                                    let acc = function_calls.entry(item_id).or_default();
                                    acc.arguments = arguments;
                                }
                                ResponsesStreamEvent::OutputItemDone { item, .. } => {
                                    // Emit function call when item is done
                                    if let ResponsesOutputItem::FunctionCall { id, call_id, name, arguments } = item {
                                        let call_id = call_id.unwrap_or(id);
                                        let args = serde_json::from_str(&arguments)
                                            .unwrap_or(serde_json::Value::Object(Default::default()));
                                        yield Ok(Event::ToolCall(ToolCall {
                                            id: call_id,
                                            name,
                                            arguments: args,
                                        }));
                                    }
                                }
                                ResponsesStreamEvent::ResponseFailed { error } => {
                                    let msg = error
                                        .and_then(|e| e.message)
                                        .unwrap_or_else(|| "Response failed".to_string());
                                    yield Err(OpenAIError::Api(msg));
                                    return;
                                }
                                ResponsesStreamEvent::Error { message, .. } => {
                                    let msg = message.unwrap_or_else(|| "Unknown error".to_string());
                                    yield Err(OpenAIError::Api(msg));
                                    return;
                                }
                                // Ignore other events
                                _ => {}
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, data = %data, "Failed to parse Responses API event");
                        }
                    }
                }
                Err(e) => {
                    yield Err(OpenAIError::from(e));
                    return;
                }
            }
        }

        // Emit any remaining accumulated function calls (fallback if OutputItemDone wasn't received)
        for (_item_id, acc) in function_calls {
            if let (Some(call_id), Some(name)) = (acc.call_id, acc.name) {
                // Skip if already emitted via OutputItemDone
                if !acc.arguments.is_empty() {
                    let args = serde_json::from_str(&acc.arguments)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                    yield Ok(Event::ToolCall(ToolCall {
                        id: call_id,
                        name,
                        arguments: args,
                    }));
                }
            }
        }
    }
}

fn filter_tool_definitions(
    defs: Vec<aither_core::llm::tool::ToolDefinition>,
    choice: &ToolChoice,
) -> Vec<aither_core::llm::tool::ToolDefinition> {
    match choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => defs
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => defs,
    }
}

/// Builder for [`OpenAI`] clients.
#[derive(Debug)]
pub struct Builder {
    api_key: String,
    base_url: String,
    api_kind: ApiKind,
    chat_model: String,
    embedding_model: String,
    embedding_dimensions: usize,
    image_model: String,
    audio_model: String,
    audio_voice: String,
    audio_format: String,
    transcription_model: String,
    moderation_model: String,
    legacy_max_tokens: bool,
    organization: Option<String>,
    native_abilities: Vec<Ability>,
    retry: RetryConfig,
    request_timeout: Duration,
}

/// Default request timeout (5 minutes - generous for long completions).
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

impl Builder {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_kind: ApiKind::Responses,
            chat_model: DEFAULT_MODEL.to_string(),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            embedding_dimensions: DEFAULT_EMBEDDING_DIM,
            image_model: DEFAULT_IMAGE_MODEL.to_string(),
            audio_model: DEFAULT_AUDIO_MODEL.to_string(),
            audio_voice: DEFAULT_AUDIO_VOICE.to_string(),
            audio_format: DEFAULT_AUDIO_FORMAT.to_string(),
            transcription_model: DEFAULT_TRANSCRIPTION_MODEL.to_string(),
            moderation_model: DEFAULT_MODERATION_MODEL.to_string(),
            legacy_max_tokens: false,
            organization: None,
            native_abilities: Vec::new(),
            retry: RetryConfig::default(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    /// Set a custom API base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Use the recommended Responses API.
    #[must_use]
    pub const fn use_responses_api(mut self) -> Self {
        self.api_kind = ApiKind::Responses;
        self
    }

    /// Use the legacy Chat Completions API (deprecated by `OpenAI`).
    #[must_use]
    pub const fn use_chat_completions_api(mut self) -> Self {
        self.api_kind = ApiKind::ChatCompletions;
        self
    }

    /// Send deprecated `max_tokens` alongside `max_completion_tokens` for compatibility.
    ///
    /// `OpenAI` deprecates `max_tokens` and it is incompatible with reasoning models.
    #[must_use]
    pub const fn legacy_max_tokens(mut self, enabled: bool) -> Self {
        self.legacy_max_tokens = enabled;
        self
    }

    /// Select a model identifier (e.g., `gpt-4o-mini`, `o1-mini`).
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.chat_model = sanitize_model(model);
        self
    }

    /// Select the embeddings model identifier.
    #[must_use]
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        let model = sanitize_model(model);
        if let Some(dim) = infer_embedding_dim(&model) {
            self.embedding_dimensions = dim;
        }
        self.embedding_model = model;
        self
    }

    /// Override the embedding vector dimension.
    #[must_use]
    pub const fn embedding_dimensions(mut self, dimensions: usize) -> Self {
        self.embedding_dimensions = dimensions;
        self
    }

    /// Select the image model identifier.
    #[must_use]
    pub fn image_model(mut self, model: impl Into<String>) -> Self {
        self.image_model = sanitize_model(model);
        self
    }

    /// Select the default audio generation model.
    #[must_use]
    pub fn audio_model(mut self, model: impl Into<String>) -> Self {
        self.audio_model = sanitize_model(model);
        self
    }

    /// Override the default TTS voice.
    #[must_use]
    pub fn audio_voice(mut self, voice: impl Into<String>) -> Self {
        self.audio_voice = voice.into();
        self
    }

    /// Override the synthesized audio format.
    #[must_use]
    pub fn audio_format(mut self, format: impl Into<String>) -> Self {
        self.audio_format = format.into();
        self
    }

    /// Select the transcription model identifier.
    #[must_use]
    pub fn transcription_model(mut self, model: impl Into<String>) -> Self {
        self.transcription_model = sanitize_model(model);
        self
    }

    /// Select the moderation model identifier.
    #[must_use]
    pub fn moderation_model(mut self, model: impl Into<String>) -> Self {
        self.moderation_model = sanitize_model(model);
        self
    }

    /// Attach an `OpenAI` organization header.
    #[must_use]
    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Declare extra native capabilities (e.g., web search, PDF understanding) supported by the upstream model.
    #[must_use]
    pub fn native_capabilities(mut self, abilities: impl IntoIterator<Item = Ability>) -> Self {
        for ability in abilities {
            if !self.native_abilities.contains(&ability) {
                self.native_abilities.push(ability);
            }
        }
        self
    }

    /// Mark this model as having built-in web search support.
    #[must_use]
    pub fn enable_native_web_search(self) -> Self {
        self.native_capabilities([Ability::WebSearch])
    }

    /// Mark this model as having native PDF/document understanding.
    #[must_use]
    pub fn enable_native_pdf(self) -> Self {
        self.native_capabilities([Ability::Pdf])
    }

    /// Configure retry behavior for failed requests.
    ///
    /// By default, requests are retried up to 3 times with exponential backoff.
    /// Retries happen on network errors and certain HTTP status codes (429, 500, 502, 503, 504).
    #[must_use]
    pub const fn retry(mut self, config: RetryConfig) -> Self {
        self.retry = config;
        self
    }

    /// Set maximum number of retry attempts.
    #[must_use]
    pub const fn max_retries(mut self, max_retries: u32) -> Self {
        self.retry.max_retries = max_retries;
        self
    }

    /// Disable retries entirely.
    #[must_use]
    pub fn no_retry(mut self) -> Self {
        self.retry = RetryConfig::none();
        self
    }

    /// Set the request timeout.
    ///
    /// Default is 5 minutes, which is generous for long completions.
    /// The timeout applies to the entire request, including connection and response streaming.
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Consume the builder and create an [`OpenAI`] client.
    #[must_use]
    pub fn build(self) -> OpenAI {
        OpenAI {
            inner: Arc::new(Config {
                api_key: self.api_key,
                base_url: self.base_url,
                api_kind: self.api_kind,
                chat_model: self.chat_model,
                embedding_model: self.embedding_model,
                embedding_dimensions: self.embedding_dimensions,
                image_model: self.image_model,
                audio_model: self.audio_model,
                audio_voice: self.audio_voice,
                audio_format: self.audio_format,
                transcription_model: self.transcription_model,
                moderation_model: self.moderation_model,
                legacy_max_tokens: self.legacy_max_tokens,
                organization: self.organization,
                native_abilities: self.native_abilities,
                retry: self.retry,
                request_timeout: self.request_timeout,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) api_kind: ApiKind,
    pub(crate) chat_model: String,
    pub(crate) embedding_model: String,
    pub(crate) embedding_dimensions: usize,
    pub(crate) image_model: String,
    pub(crate) audio_model: String,
    pub(crate) audio_voice: String,
    pub(crate) audio_format: String,
    pub(crate) transcription_model: String,
    pub(crate) moderation_model: String,
    pub(crate) legacy_max_tokens: bool,
    pub(crate) organization: Option<String>,
    pub(crate) native_abilities: Vec<Ability>,
    pub(crate) retry: RetryConfig,
    pub(crate) request_timeout: Duration,
}

impl Config {
    pub(crate) fn request_url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    pub(crate) fn request_auth(&self) -> String {
        format!("Bearer {}", self.api_key)
    }
}

fn sanitize_model(model: impl Into<String>) -> String {
    model.into().trim().to_string()
}

fn infer_embedding_dim(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-large" => Some(3072),
        "text-embedding-3-small" | "text-embedding-ada-002" => Some(1536),
        _ => None,
    }
}
