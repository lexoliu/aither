use crate::{
    DEEPSEEK_BASE_URL, DEFAULT_AUDIO_FORMAT, DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_VOICE,
    DEFAULT_BASE_URL, DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL, DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL, DEFAULT_MODERATION_MODEL, DEFAULT_TRANSCRIPTION_MODEL, OPENROUTER_BASE_URL,
    error::OpenAIError,
    request::{
        ChatCompletionRequest, ChatMessagePayload, ChatToolCallPayload, ChatToolFunctionPayload,
        ParameterSnapshot, ResponsesInputItem, ResponsesRequest, ResponsesTool,
        convert_responses_tools, convert_tools, responses_tool_choice, to_chat_messages,
        to_responses_input,
    },
    response::{ChatCompletionChunk, ChatCompletionResponse, ResponsesOutput, should_skip_event},
};
use aither_core::{
    LanguageModel,
    llm::{
        LLMRequest, LLMResponse, ReasoningStream, ResponseChunk,
        model::{Ability, Profile as ModelProfile},
        oneshot,
        tool::{ToolDefinition, Tools},
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use std::{future::Future, pin::Pin, sync::Arc};
use zenwave::{Client, client, error::BoxHttpError, header};

const MAX_TOOL_ITERATIONS: usize = 8;

type BoxedResponseStream<'a> =
    Pin<Box<dyn Stream<Item = Result<ResponseChunk, OpenAIError>> + Send + 'a>>;

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

    fn respond(&self, request: LLMRequest) -> impl LLMResponse<Error = Self::Error> {
        let cfg = self.inner.clone();
        let (messages, parameters, tools) = request.into_parts();
        let mut snapshot = ParameterSnapshot::from(&parameters);
        snapshot.legacy_max_tokens = cfg.legacy_max_tokens;

        let payload_messages = to_chat_messages(&messages);
        let responses_input = to_responses_input(&messages);

        let tool_defs = tools
            .as_deref()
            .map(|t| t.definitions())
            .unwrap_or_default();
        let tool_defs_responses = tool_defs.clone();
        let has_function_tools = !tool_defs.is_empty();
        if !has_function_tools {
            snapshot.tool_choice = None;
        }

        let stream: BoxedResponseStream<'_> = match cfg.api_kind {
            ApiKind::ChatCompletions => {
                if has_function_tools {
                    Box::pin(chat_completions_tool_loop(
                        cfg,
                        payload_messages,
                        snapshot,
                        tool_defs,
                        tools,
                    ))
                } else {
                    Box::pin(chat_completions_stream(cfg, payload_messages, snapshot))
                }
            }
            ApiKind::Responses => Box::pin(responses_tool_loop(
                cfg,
                responses_input,
                snapshot,
                tool_defs_responses,
                tools,
            )),
        };

        ReasoningStream::new(stream)
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
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
) -> impl Stream<Item = Result<ResponseChunk, OpenAIError>> + Send {
    let include_reasoning = snapshot.include_reasoning;
    let init_future = async move {
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
            None,
            true,
        );

        builder.json_body(&request).sse().await
    };

    let events = futures_lite::stream::iter(vec![init_future])
        .then(|fut| fut)
        .filter_map(Result::ok)
        .flatten();

    events
        .filter_map(|event| match &event {
            Ok(e) if should_skip_event(e) => None,
            Ok(e) if e.text_data() == "[DONE]" => None,
            _ => Some(event),
        })
        .map(move |event| match event {
            Ok(e) => serde_json::from_str::<ChatCompletionChunk>(e.text_data())
                .map(|chunk| chunk.into_chunk_filtered(include_reasoning))
                .map_err(OpenAIError::from),
            Err(err) => Err(OpenAIError::from(err)),
        })
}

fn chat_completions_tool_loop<'tools>(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    snapshot: ParameterSnapshot,
    tool_defs: Vec<ToolDefinition>,
    tools: Option<&'tools mut Tools>,
) -> impl Stream<Item = Result<ResponseChunk, OpenAIError>> + Send + 'tools {
    enum State<'tools> {
        Processing {
            iterations: usize,
            messages: Vec<ChatMessagePayload>,
            tools: Option<&'tools mut Tools>,
        },
        Done,
    }

    let openai_tools = if tool_defs.is_empty() {
        None
    } else {
        Some(convert_tools(tool_defs))
    };

    futures_lite::stream::unfold(
        State::Processing {
            iterations: 0,
            messages: payload_messages,
            tools,
        },
        move |state| {
            let cfg = cfg.clone();
            let snapshot = snapshot.clone();
            let openai_tools = openai_tools.clone();

            async move {
                let (iterations, mut messages, mut tools) = match state {
                    State::Processing {
                        iterations,
                        messages,
                        tools,
                    } => (iterations, messages, tools),
                    State::Done => return None,
                };

                let next_iteration = iterations + 1;
                if next_iteration > MAX_TOOL_ITERATIONS {
                    return Some((
                        Err(OpenAIError::Api(
                            "exceeded OpenAI tool calling iteration limit".into(),
                        )),
                        State::Done,
                    ));
                }

                let endpoint = cfg.request_url("/chat/completions");
                let mut backend = client();
                let mut builder = backend.post(endpoint);
                builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
                builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
                if let Some(org) = &cfg.organization {
                    builder = builder.header("OpenAI-Organization", org.clone());
                }

                let request = ChatCompletionRequest::new(
                    cfg.chat_model.clone(),
                    messages.clone(),
                    &snapshot,
                    openai_tools.clone(),
                    false,
                );

                let response: ChatCompletionResponse =
                    match builder.json_body(&request).json().await {
                        Ok(response) => response,
                        Err(error) => {
                            return Some((
                                Err(OpenAIError::Http(BoxHttpError::from(Box::new(error)))),
                                State::Done,
                            ));
                        }
                    };

                let Some(message) = response.into_primary() else {
                    return Some((
                        Err(OpenAIError::Api(
                            "chat completion response missing message".into(),
                        )),
                        State::Done,
                    ));
                };

                let (texts, reasoning, tool_calls) = message.into_parts();

                if !tool_calls.is_empty() {
                    let Some(tool_registry) = &mut tools else {
                        return Some((
                            Err(OpenAIError::Api(
                                "tool call requested but no tools available".into(),
                            )),
                            State::Done,
                        ));
                    };

                    let tool_payloads: Vec<ChatToolCallPayload> = tool_calls
                        .iter()
                        .map(|call| ChatToolCallPayload {
                            id: call.id.clone(),
                            kind: "function",
                            function: ChatToolFunctionPayload {
                                name: call.function.name.clone(),
                                arguments: call.function.arguments.clone(),
                            },
                        })
                        .collect();

                    let content = texts.join("");
                    messages.push(ChatMessagePayload::assistant_tool_calls(
                        content,
                        tool_payloads,
                    ));

                    for call in tool_calls {
                        let output = match tool_registry
                            .call(&call.function.name, &call.function.arguments)
                            .await
                        {
                            Ok(output) => output,
                            Err(err) => {
                                return Some((Err(OpenAIError::Api(err.to_string())), State::Done));
                            }
                        };
                        messages.push(ChatMessagePayload::tool_output(call.id, output));
                    }

                    return Some((
                        Ok(ResponseChunk::default()),
                        State::Processing {
                            iterations: next_iteration,
                            messages,
                            tools,
                        },
                    ));
                }

                let mut chunk = ResponseChunk::default();
                for text in texts {
                    chunk.push_text(text);
                }
                if snapshot.include_reasoning {
                    for step in reasoning {
                        chunk.push_reasoning(step);
                    }
                }

                Some((Ok(chunk), State::Done))
            }
        },
    )
}

fn responses_tool_loop<'tools>(
    cfg: Arc<Config>,
    input: Vec<ResponsesInputItem>,
    snapshot: ParameterSnapshot,
    tool_defs: Vec<ToolDefinition>,
    tools: Option<&'tools mut Tools>,
) -> impl Stream<Item = Result<ResponseChunk, OpenAIError>> + Send + 'tools {
    enum State<'tools> {
        Processing {
            iterations: usize,
            input: Vec<ResponsesInputItem>,
            tools: Option<&'tools mut Tools>,
        },
        Done,
    }

    let has_function_tools = !tool_defs.is_empty();
    let mut response_tools = convert_responses_tools(tool_defs);
    if snapshot.websearch {
        response_tools.push(ResponsesTool::WebSearch);
    }
    if snapshot.code_execution {
        response_tools.push(ResponsesTool::CodeInterpreter);
    }
    let response_tools = if response_tools.is_empty() {
        None
    } else {
        Some(response_tools)
    };
    let tool_choice = if has_function_tools {
        responses_tool_choice(&snapshot)
    } else {
        None
    };

    futures_lite::stream::unfold(
        State::Processing {
            iterations: 0,
            input,
            tools,
        },
        move |state| {
            let cfg = cfg.clone();
            let snapshot = snapshot.clone();
            let response_tools = response_tools.clone();
            let tool_choice = tool_choice.clone();

            async move {
                let (iterations, mut input, mut tools) = match state {
                    State::Processing {
                        iterations,
                        input,
                        tools,
                    } => (iterations, input, tools),
                    State::Done => return None,
                };

                let next_iteration = iterations + 1;
                if next_iteration > MAX_TOOL_ITERATIONS {
                    return Some((
                        Err(OpenAIError::Api(
                            "exceeded OpenAI tool calling iteration limit".into(),
                        )),
                        State::Done,
                    ));
                }

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
                    input.clone(),
                    &snapshot,
                    response_tools.clone(),
                    tool_choice.clone(),
                );

                let response: ResponsesOutput = match builder.json_body(&request).json().await {
                    Ok(response) => response,
                    Err(error) => {
                        return Some((
                            Err(OpenAIError::Http(BoxHttpError::from(Box::new(error)))),
                            State::Done,
                        ));
                    }
                };

                let (texts, reasoning, tool_calls, _) = response.into_parts();

                if !tool_calls.is_empty() {
                    let Some(tool_registry) = &mut tools else {
                        return Some((
                            Err(OpenAIError::Api(
                                "tool call requested but no tools available".into(),
                            )),
                            State::Done,
                        ));
                    };

                    for call in tool_calls {
                        let output = match tool_registry.call(&call.name, &call.arguments).await {
                            Ok(output) => output,
                            Err(err) => {
                                return Some((Err(OpenAIError::Api(err.to_string())), State::Done));
                            }
                        };
                        input.push(ResponsesInputItem::function_call_output(
                            call.call_id,
                            output,
                        ));
                    }

                    return Some((
                        Ok(ResponseChunk::default()),
                        State::Processing {
                            iterations: next_iteration,
                            input,
                            tools,
                        },
                    ));
                }

                let mut chunk = ResponseChunk::default();
                for text in texts {
                    chunk.push_text(text);
                }
                if snapshot.include_reasoning {
                    for step in reasoning {
                        chunk.push_reasoning(step);
                    }
                }

                Some((Ok(chunk), State::Done))
            }
        },
    )
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
