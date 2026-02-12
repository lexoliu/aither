//! Local `mistral.rs` integration for `aither` traits.
//!
//! This crate provides a single backend type, [`Mistral`], that can be configured with
//! separate local model IDs for LLM, embedding, and diffusion image generation workloads.

mod error;
#[cfg(feature = "llm")]
mod provider;

pub use error::MistralError;
#[cfg(feature = "llm")]
pub use provider::MistralProvider;

use std::sync::{Arc, Mutex};

use aither_core::{
    EmbeddingModel, LanguageModel,
    image::{Data as ImageData, ImageGenerator, Prompt, Size},
    llm::{
        Event, LLMRequest, Message, ToolCall, Usage,
        model::{Ability, Profile, ToolChoice},
        tool::ToolDefinition,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use mistralrs::core::{EmbeddingLoaderType, ImageChoice, NormalLoaderType};
use mistralrs::{DiffusionLoaderType, Model};

#[cfg(feature = "image")]
use base64::{Engine as _, engine::general_purpose};
#[cfg(feature = "embedding")]
use mistralrs::EmbeddingModelBuilder;
#[cfg(feature = "image")]
use mistralrs::{DiffusionGenerationParams, ImageGenerationResponseFormat};
#[cfg(feature = "llm")]
use mistralrs::{
    Function, RequestBuilder, TextMessageRole, TextModelBuilder, Tool,
    ToolChoice as MistralToolChoice, ToolType,
};
#[cfg(feature = "llm")]
use std::collections::HashMap;

/// Local mistral.rs-backed model implementing `aither` traits.
#[derive(Clone)]
pub struct Mistral {
    inner: Arc<Mutex<Inner>>,
    embedding_dimensions: usize,
}

impl core::fmt::Debug for Mistral {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mistral")
            .field("embedding_dimensions", &self.embedding_dimensions)
            .finish_non_exhaustive()
    }
}

struct Inner {
    llm_model_id: Option<String>,
    embedding_model_id: Option<String>,
    image_model_id: Option<String>,
    llm_loader: NormalLoaderType,
    embedding_loader: EmbeddingLoaderType,
    image_loader: DiffusionLoaderType,
    llm: Option<Arc<Model>>,
    embedding: Option<Arc<Model>>,
    image: Option<Arc<Model>>,
}

impl Default for Mistral {
    fn default() -> Self {
        Self::new()
    }
}

impl Mistral {
    /// Create a new mistral backend with no preconfigured model IDs.
    #[must_use] 
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                llm_model_id: None,
                embedding_model_id: None,
                image_model_id: None,
                llm_loader: NormalLoaderType::Mistral,
                embedding_loader: EmbeddingLoaderType::Qwen3Embedding,
                image_loader: DiffusionLoaderType::Flux,
                llm: None,
                embedding: None,
                image: None,
            })),
            embedding_dimensions: 1024,
        }
    }

    /// Set model ID used for LLM inference.
    #[must_use]
    pub fn with_llm_model(self, model_id: impl Into<String>) -> Self {
        let this = self;
        {
            let mut inner = this.inner.lock().expect("mistral state poisoned");
            inner.llm_model_id = Some(model_id.into());
            inner.llm = None;
        }
        this
    }

    /// Set model ID used for embeddings.
    #[must_use]
    pub fn with_embedding_model(self, model_id: impl Into<String>) -> Self {
        let this = self;
        {
            let mut inner = this.inner.lock().expect("mistral state poisoned");
            inner.embedding_model_id = Some(model_id.into());
            inner.embedding = None;
        }
        this
    }

    /// Set model ID used for image generation.
    #[must_use]
    pub fn with_image_model(self, model_id: impl Into<String>) -> Self {
        let this = self;
        {
            let mut inner = this.inner.lock().expect("mistral state poisoned");
            inner.image_model_id = Some(model_id.into());
            inner.image = None;
        }
        this
    }

    /// Set LLM loader type used by mistral.rs.
    #[must_use]
    pub fn with_llm_loader(self, loader: NormalLoaderType) -> Self {
        let this = self;
        {
            let mut inner = this.inner.lock().expect("mistral state poisoned");
            inner.llm_loader = loader;
            inner.llm = None;
        }
        this
    }

    /// Set embedding loader type used by mistral.rs.
    #[must_use]
    pub fn with_embedding_loader(self, loader: EmbeddingLoaderType) -> Self {
        let this = self;
        {
            let mut inner = this.inner.lock().expect("mistral state poisoned");
            inner.embedding_loader = loader;
            inner.embedding = None;
        }
        this
    }

    /// Set image loader type used by mistral.rs.
    #[must_use]
    pub fn with_image_loader(self, loader: DiffusionLoaderType) -> Self {
        let this = self;
        {
            let mut inner = this.inner.lock().expect("mistral state poisoned");
            inner.image_loader = loader;
            inner.image = None;
        }
        this
    }

    /// Override the advertised embedding dimensions.
    #[must_use]
    pub const fn with_embedding_dimensions(mut self, dim: usize) -> Self {
        self.embedding_dimensions = dim;
        self
    }

    #[cfg(feature = "llm")]
    async fn ensure_llm(inner: &Arc<Mutex<Inner>>) -> Result<Arc<Model>, MistralError> {
        if let Some(model) = inner.lock().expect("mistral state poisoned").llm.clone() {
            return Ok(model);
        }

        let (model_id, loader) = {
            let guard = inner.lock().expect("mistral state poisoned");
            (
                guard
                    .llm_model_id
                    .clone()
                    .ok_or(MistralError::MissingModel("llm_model_id"))?,
                guard.llm_loader.clone(),
            )
        };

        let built = Arc::new(
            TextModelBuilder::new(model_id)
                .with_loader_type(loader)
                .build()
                .await?,
        );

        let mut guard = inner.lock().expect("mistral state poisoned");
        if guard.llm.is_none() {
            guard.llm = Some(built);
        }
        Ok(guard.llm.clone().expect("llm model must be initialized"))
    }

    #[cfg(feature = "embedding")]
    async fn ensure_embedding(inner: &Arc<Mutex<Inner>>) -> Result<Arc<Model>, MistralError> {
        if let Some(model) = inner
            .lock()
            .expect("mistral state poisoned")
            .embedding
            .clone()
        {
            return Ok(model);
        }

        let (model_id, loader) = {
            let guard = inner.lock().expect("mistral state poisoned");
            (
                guard
                    .embedding_model_id
                    .clone()
                    .ok_or(MistralError::MissingModel("embedding_model_id"))?,
                guard.embedding_loader.clone(),
            )
        };

        let built = Arc::new(
            EmbeddingModelBuilder::new(model_id)
                .with_loader_type(loader)
                .build()
                .await?,
        );

        let mut guard = inner.lock().expect("mistral state poisoned");
        if guard.embedding.is_none() {
            guard.embedding = Some(built);
        }
        Ok(guard
            .embedding
            .clone()
            .expect("embedding model must be initialized"))
    }

    #[cfg(feature = "image")]
    async fn ensure_image(inner: &Arc<Mutex<Inner>>) -> Result<Arc<Model>, MistralError> {
        if let Some(model) = inner.lock().expect("mistral state poisoned").image.clone() {
            return Ok(model);
        }

        let (model_id, loader) = {
            let guard = inner.lock().expect("mistral state poisoned");
            (
                guard
                    .image_model_id
                    .clone()
                    .ok_or(MistralError::MissingModel("image_model_id"))?,
                guard.image_loader.clone(),
            )
        };

        let built = Arc::new(
            mistralrs::DiffusionModelBuilder::new(model_id, loader)
                .build()
                .await?,
        );

        let mut guard = inner.lock().expect("mistral state poisoned");
        if guard.image.is_none() {
            guard.image = Some(built);
        }
        Ok(guard
            .image
            .clone()
            .expect("image model must be initialized"))
    }
}

#[cfg(feature = "llm")]
impl LanguageModel for Mistral {
    type Error = MistralError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let inner = self.inner.clone();
        async_stream::stream! {
            let (messages, parameters, tool_defs) = request.into_parts();
            let model = match Self::ensure_llm(&inner).await {
                Ok(model) => model,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            let req = to_mistral_request(messages, &parameters, tool_defs);
            let response = match model.send_chat_request(req).await {
                Ok(response) => response,
                Err(err) => {
                    yield Err(MistralError::from(err));
                    return;
                }
            };

            if let Some(choice) = response.choices.first() {
                if let Some(reasoning) = &choice.message.reasoning_content {
                    if !reasoning.is_empty() {
                        yield Ok(Event::Reasoning(reasoning.clone()));
                    }
                }

                if let Some(content) = &choice.message.content {
                    if !content.is_empty() {
                        yield Ok(Event::Text(content.clone()));
                    }
                }

                if let Some(tool_calls) = &choice.message.tool_calls {
                    for tool_call in tool_calls {
                        let arguments = match serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments) {
                            Ok(value) => value,
                            Err(_) => serde_json::Value::String(tool_call.function.arguments.clone()),
                        };
                        yield Ok(Event::ToolCall(ToolCall::new(
                            tool_call.id.clone(),
                            tool_call.function.name.clone(),
                            arguments,
                        )));
                    }
                }
            }

            let prompt_tokens = u32::try_from(response.usage.prompt_tokens).ok();
            let completion_tokens = u32::try_from(response.usage.completion_tokens).ok();
            let total_tokens = u32::try_from(response.usage.total_tokens).ok();
            yield Ok(Event::Usage(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                reasoning_tokens: None,
                cache_read_tokens: None,
                cache_write_tokens: None,
                cost_usd: None,
            }));
        }
    }

    fn profile(&self) -> impl core::future::Future<Output = Profile> + Send {
        let inner = self.inner.clone();
        async move {
            let guard = inner.lock().expect("mistral state poisoned");
            let name = guard
                .llm_model_id
                .clone()
                .unwrap_or_else(|| "mistral-local".to_string());
            let context_length = aither_models::lookup(&name)
                .map_or(32_768, |model| model.context_window);

            Profile::new(
                name.clone(),
                "mistral",
                name,
                "mistral.rs local language model",
                context_length,
            )
            .with_abilities([Ability::ToolUse])
        }
    }
}

#[cfg(feature = "embedding")]
impl EmbeddingModel for Mistral {
    fn dim(&self) -> usize {
        self.embedding_dimensions
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl core::future::Future<Output = aither_core::Result<Vec<f32>>> + Send {
        let inner = self.inner.clone();
        let text = text.to_string();
        async move {
            let model = Self::ensure_embedding(&inner).await?;
            model.generate_embedding(text).await}
    }
}

#[cfg(feature = "image")]
impl ImageGenerator for Mistral {
    type Error = MistralError;

    fn create(
        &self,
        prompt: Prompt,
        size: Size,
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let inner = self.inner.clone();
        let prompt_text = prompt.text().to_owned();
        let params = DiffusionGenerationParams {
            width: size.width() as usize,
            height: size.height() as usize,
        };

        futures_lite::stream::iter(vec![async move {
            let model = Self::ensure_image(&inner).await?;
            let response = model
                .generate_image(prompt_text, ImageGenerationResponseFormat::B64Json, params)
                .await?;
            decode_images(response.data)
        }])
        .then(|fut| fut)
        .map(|result| result.map(futures_lite::stream::iter).ok())
        .filter_map(core::convert::identity)
        .flatten()
    }

    fn edit(
        &self,
        _prompt: Prompt,
        _mask: &[u8],
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        futures_lite::stream::iter(vec![Err(MistralError::Api(
            "mistral.rs image edit is not supported".to_string(),
        ))])
    }
}

#[cfg(feature = "llm")]
fn to_mistral_request(
    messages: Vec<Message>,
    parameters: &aither_core::llm::model::Parameters,
    tool_defs: Vec<ToolDefinition>,
) -> RequestBuilder {
    let mut request = RequestBuilder::new();

    for message in messages {
        request = match message {
            Message::User { content, .. } => request.add_message(TextMessageRole::User, content),
            Message::System { content } => request.add_message(TextMessageRole::System, content),
            Message::Assistant {
                content,
                tool_calls,
            } => {
                if tool_calls.is_empty() {
                    request.add_message(TextMessageRole::Assistant, content)
                } else {
                    request.add_message_with_tool_call(
                        TextMessageRole::Assistant,
                        content,
                        tool_calls
                            .into_iter()
                            .enumerate()
                            .map(|(index, call)| mistralrs::ToolCallResponse {
                                index,
                                id: call.id,
                                tp: mistralrs::ToolCallType::Function,
                                function: mistralrs::CalledFunction {
                                    name: call.name,
                                    arguments: call.arguments.to_string(),
                                },
                            })
                            .collect(),
                    )
                }
            }
            Message::Tool {
                content,
                tool_call_id,
            } => request.add_tool_message(content, tool_call_id),
        };
    }

    if let Some(temp) = parameters.temperature {
        request = request.set_sampler_temperature(f64::from(temp));
    }
    if let Some(top_p) = parameters.top_p {
        request = request.set_sampler_topp(f64::from(top_p));
    }
    if let Some(max_tokens) = parameters.max_tokens {
        request = request.set_sampler_max_len(max_tokens as usize);
    }
    if let Some(freq) = parameters.frequency_penalty {
        request = request.set_sampler_frequency_penalty(freq);
    }
    if let Some(presence) = parameters.presence_penalty {
        request = request.set_sampler_presence_penalty(presence);
    }
    if let Some(stops) = &parameters.stop {
        request = request.set_sampler_stop_toks(mistralrs::StopTokens::Seqs(stops.clone()));
    }

    let tools = tool_defs
        .iter()
        .map(convert_tool_definition)
        .collect::<Vec<_>>();
    if !tools.is_empty() {
        request = request.set_tools(tools.clone());
    }

    let tool_choice = match &parameters.tool_choice {
        ToolChoice::None => MistralToolChoice::None,
        ToolChoice::Auto | ToolChoice::Required => MistralToolChoice::Auto,
        ToolChoice::Exact(name) => tools
            .into_iter()
            .find(|tool| tool.function.name == *name)
            .map_or(MistralToolChoice::Auto, MistralToolChoice::Tool),
    };
    request.set_tool_choice(tool_choice)
}

#[cfg(feature = "llm")]
fn convert_tool_definition(def: &ToolDefinition) -> Tool {
    let parameters = match def.arguments_openai_schema() {
        serde_json::Value::Object(map) => Some(map.into_iter().collect::<HashMap<_, _>>()),
        _ => None,
    };

    Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some(def.description().to_owned()),
            name: def.name().to_owned(),
            parameters,
        },
    }
}

#[cfg(feature = "image")]
fn decode_images(
    data: Vec<ImageChoice>,
) -> Result<Vec<Result<ImageData, MistralError>>, MistralError> {
    let mut out = Vec::new();
    for item in data {
        if let Some(raw) = item.b64_json {
            out.push(
                general_purpose::STANDARD
                    .decode(raw)
                    .map_err(MistralError::from),
            );
        } else if let Some(url) = item.url {
            out.push(Err(MistralError::Api(format!(
                "mistral.rs returned URL output ({url}); use B64Json format"
            ))));
        }
    }
    if out.is_empty() {
        return Err(MistralError::Api(
            "image generation response missing image payload".to_string(),
        ));
    }
    Ok(out)
}
