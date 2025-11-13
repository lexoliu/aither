use crate::{
    DEEPSEEK_BASE_URL, DEFAULT_AUDIO_FORMAT, DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_VOICE,
    DEFAULT_BASE_URL, DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL, DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL, DEFAULT_MODERATION_MODEL, DEFAULT_TRANSCRIPTION_MODEL, OPENROUTER_BASE_URL,
    error::OpenAIError,
    request::{ChatCompletionRequest, ParameterSnapshot, convert_tools, to_chat_messages},
    response::{ChatCompletionChunk, should_skip_event},
};
use aither_core::{
    LanguageModel,
    llm::{
        Message,
        model::{Ability, Parameters, Profile as ModelProfile},
        tool::Tools,
    },
};
use async_stream::try_stream;
use futures_core::Stream;
use futures_lite::{StreamExt, pin};
use std::{future::Future, sync::Arc};
use zenwave::{Client, client, header};

/// `OpenAI` chat model backed by the `chat.completions` API.
#[derive(Clone, Debug)]
pub struct OpenAI {
    inner: Arc<Config>,
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
        messages: &[Message],
        tools: &mut Tools,
        parameters: &Parameters,
    ) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let payload_messages = to_chat_messages(messages);
        let snapshot = ParameterSnapshot::from(parameters);
        let openai_tools = convert_tools(tools.definitions());

        try_stream! {
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

            builder = builder.json_body(&request).map_err(OpenAIError::from)?;
            let mut stream = builder.sse().await.map_err(OpenAIError::from)?;

            while let Some(event) = stream.next().await {
                let event = event.map_err(OpenAIError::from)?;
                if should_skip_event(&event) {
                    continue;
                }
                if event.text_data() == "[DONE]" {
                    break;
                }
                let chunk: ChatCompletionChunk =
                    serde_json::from_str(event.text_data()).map_err(OpenAIError::from)?;
                if let Some(text) = chunk.into_text() {
                    if !text.is_empty() {
                        yield text;
                    }
                }
            }
        }
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        let messages = vec![
            Message::system("Continue the user provided text without additional commentary."),
            Message::user(prefix),
        ];
        let model = self.clone();
        try_stream! {
            let mut tools = Tools::new();
            let parameters = Parameters::default();
            let stream = model.respond(&messages, &mut tools, &parameters);
            pin!(stream);
            while let Some(chunk) = stream.next().await {
                yield chunk?;
            }
        }
    }

    fn profile(&self) -> impl Future<Output = ModelProfile> + Send {
        let cfg = self.inner.clone();
        async move {
            ModelProfile::new(
                cfg.chat_model.clone(),
                "OpenAI",
                cfg.chat_model.clone(),
                "OpenAI GPT family model",
                128_000,
            )
            .with_ability(Ability::ToolUse)
        }
    }
}

/// Builder for [`OpenAI`] clients.
#[derive(Debug)]
pub struct Builder {
    api_key: String,
    base_url: String,
    chat_model: String,
    embedding_model: String,
    embedding_dimensions: usize,
    image_model: String,
    audio_model: String,
    audio_voice: String,
    audio_format: String,
    transcription_model: String,
    moderation_model: String,
    organization: Option<String>,
}

impl Builder {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            chat_model: DEFAULT_MODEL.to_string(),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            embedding_dimensions: DEFAULT_EMBEDDING_DIM,
            image_model: DEFAULT_IMAGE_MODEL.to_string(),
            audio_model: DEFAULT_AUDIO_MODEL.to_string(),
            audio_voice: DEFAULT_AUDIO_VOICE.to_string(),
            audio_format: DEFAULT_AUDIO_FORMAT.to_string(),
            transcription_model: DEFAULT_TRANSCRIPTION_MODEL.to_string(),
            moderation_model: DEFAULT_MODERATION_MODEL.to_string(),
            organization: None,
        }
    }

    /// Set a custom API base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
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

    /// Consume the builder and create an [`OpenAI`] client.
    #[must_use]
    pub fn build(self) -> OpenAI {
        OpenAI {
            inner: Arc::new(Config {
                api_key: self.api_key,
                base_url: self.base_url,
                chat_model: self.chat_model,
                embedding_model: self.embedding_model,
                embedding_dimensions: self.embedding_dimensions,
                image_model: self.image_model,
                audio_model: self.audio_model,
                audio_voice: self.audio_voice,
                audio_format: self.audio_format,
                transcription_model: self.transcription_model,
                moderation_model: self.moderation_model,
                organization: self.organization,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) chat_model: String,
    pub(crate) embedding_model: String,
    pub(crate) embedding_dimensions: usize,
    pub(crate) image_model: String,
    pub(crate) audio_model: String,
    pub(crate) audio_voice: String,
    pub(crate) audio_format: String,
    pub(crate) transcription_model: String,
    pub(crate) moderation_model: String,
    pub(crate) organization: Option<String>,
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
