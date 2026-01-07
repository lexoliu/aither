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
    response::{ChatCompletionChunk, ResponsesOutput, should_skip_event},
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
use std::{future::Future, pin::Pin, sync::Arc};
use zenwave::{Client, client, header};

/// `OpenAI` model backed by the Responses API by default, with legacy
/// `chat.completions` support for compatibility.
#[derive(Clone, Debug)]
pub struct OpenAI {
    inner: Arc<Config>,
}

/// Selects which OpenAI API surface to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiKind {
    /// The recommended Responses API.
    Responses,
    /// Legacy Chat Completions API (deprecated by OpenAI).
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

    /// Select which OpenAI API to call.
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

    /// Use the legacy Chat Completions API (deprecated by OpenAI).
    #[must_use]
    pub fn with_chat_completions_api(self) -> Self {
        self.with_api(ApiKind::ChatCompletions)
    }

    /// Send deprecated `max_tokens` alongside `max_completion_tokens` for compatibility.
    ///
    /// OpenAI deprecates `max_tokens` and it is incompatible with reasoning models.
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

        let payload_messages = to_chat_messages(&messages);
        let responses_input = to_responses_input(&messages);

        async_stream::stream! {
            match cfg.api_kind {
                ApiKind::ChatCompletions => {
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
            let mut profile = ModelProfile::new(
                cfg.chat_model.clone(),
                "OpenAI",
                cfg.chat_model.clone(),
                "OpenAI GPT family model",
                128_000,
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

fn chat_completions_stream_inner(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    snapshot: ParameterSnapshot,
    openai_tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, OpenAIError>> + Send {
    let include_reasoning = snapshot.include_reasoning;

    async_stream::stream! {
        let endpoint = cfg.request_url("/chat/completions");
        let mut backend = client();
        let mut builder = backend.post(endpoint);
        builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
        builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
        if let Some(org) = &cfg.organization {
            builder = builder.header("OpenAI-Organization", org.clone());
        }
        builder = builder.header(header::ACCEPT.as_str(), "text/event-stream");

        let request = ChatCompletionRequest::new(
            cfg.chat_model.clone(),
            payload_messages,
            &snapshot,
            openai_tools,
            true,
        );

        let sse_stream = match builder.json_body(&request).sse().await {
            Ok(stream) => stream,
            Err(e) => {
                yield Err(OpenAIError::Api(format!("HTTP request failed: {e}")));
                return;
            }
        };

        let events: Vec<_> = sse_stream
            .filter_map(|event| match event {
                Ok(e) if !should_skip_event(&e) && e.text_data() != "[DONE]" => Some(Ok(e)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .collect()
            .await;

        // Accumulate tool calls by index - streaming sends id/name first, then arguments incrementally
        let mut tool_calls: std::collections::HashMap<usize, ToolCallAccumulator> =
            std::collections::HashMap::new();

        for event in events {
            match event {
                Ok(e) => {
                    tracing::debug!(sse_event = %e.text_data(), "Received SSE event");
                    match serde_json::from_str::<ChatCompletionChunk>(e.text_data()) {
                        Ok(chunk) => {
                            // Emit text events
                            for choice in &chunk.choices {
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

        let endpoint = cfg.request_url("/responses");
        let mut backend = client();
        let mut builder = backend.post(endpoint);
        builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
        builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
        if let Some(org) = &cfg.organization {
            builder = builder.header("OpenAI-Organization", org.clone());
        }

        let request = ResponsesRequest::new(
            cfg.chat_model.clone(),
            input,
            &snapshot,
            response_tools,
            tool_choice,
        );

        let response: ResponsesOutput = match builder.json_body(&request).json().await {
            Ok(response) => response,
            Err(error) => {
                yield Err(OpenAIError::Api(format!("HTTP request failed: {error}")));
                return;
            }
        };

        let (texts, reasoning, tool_calls, _) = response.into_parts();

        // Emit text events
        for text in texts {
            if !text.is_empty() {
                yield Ok(Event::Text(text));
            }
        }

        // Emit reasoning events
        if include_reasoning {
            for step in reasoning {
                if !step.is_empty() {
                    yield Ok(Event::Reasoning(step));
                }
            }
        }

        // Emit tool call events (NOT executed - consumer handles execution)
        for call in tool_calls {
            let arguments = serde_json::from_str(&call.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            yield Ok(Event::ToolCall(ToolCall {
                id: call.call_id,
                name: call.name,
                arguments,
            }));
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
}

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
    pub fn use_responses_api(mut self) -> Self {
        self.api_kind = ApiKind::Responses;
        self
    }

    /// Use the legacy Chat Completions API (deprecated by OpenAI).
    #[must_use]
    pub fn use_chat_completions_api(mut self) -> Self {
        self.api_kind = ApiKind::ChatCompletions;
        self
    }

    /// Send deprecated `max_tokens` alongside `max_completion_tokens` for compatibility.
    ///
    /// OpenAI deprecates `max_tokens` and it is incompatible with reasoning models.
    #[must_use]
    pub fn legacy_max_tokens(mut self, enabled: bool) -> Self {
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
