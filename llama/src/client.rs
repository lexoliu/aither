use crate::error::LlamaError;
use aither_core::{
    EmbeddingModel, LanguageModel,
    llm::{
        Event, LLMRequest, Message, Role, ToolCall,
        model::{Ability, Parameters, Profile, ToolChoice},
        tool::ToolDefinition,
    },
};
use futures_core::Stream;
use llama_cpp_2::{
    LlamaCppError,
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    openai::OpenAIChatTemplateParams,
    sampling::LlamaSampler,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

fn init_backend() -> Result<LlamaBackend, LlamaError> {
    match LlamaBackend::init() {
        Ok(backend) => Ok(backend),
        Err(LlamaCppError::BackendAlreadyInitialized) => Ok(LlamaBackend {}),
        Err(err) => Err(LlamaError::Model(err.to_string())),
    }
}

/// Local llama.cpp model wrapper implementing aither traits.
#[derive(Debug, Clone)]
pub struct Llama {
    inner: Arc<LlamaConfig>,
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
}

impl Llama {
    /// Load a GGUF model from disk.
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, LlamaError> {
        Self::builder(model_path).build()
    }

    /// Start building a llama backend with custom options.
    #[must_use]
    pub fn builder(model_path: impl AsRef<Path>) -> Builder {
        Builder::new(model_path)
    }

    /// Override chat template name or full template string.
    #[must_use]
    pub fn with_chat_template(mut self, template: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).chat_template = Some(template.into());
        self
    }

    /// Override generation context size.
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        Arc::make_mut(&mut self.inner).n_ctx = Some(n_ctx);
        self
    }
}

impl LanguageModel for Llama {
    type Error = LlamaError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let outcome = run_response(&self.model, &self.backend, &self.inner, request).map_or_else(
            |err| vec![Err(err)],
            |events| events.into_iter().map(Ok).collect::<Vec<_>>(),
        );
        futures_lite::stream::iter(outcome)
    }

    fn profile(&self) -> impl std::future::Future<Output = Profile> + Send {
        let cfg = self.inner.clone();
        let context_length = cfg
            .n_ctx
            .unwrap_or_else(|| self.model.n_ctx_train().max(512));
        async move {
            let model_name = cfg
                .model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("llama-local")
                .to_string();
            Profile::new(
                model_name.clone(),
                "llama.cpp",
                model_name,
                "Local llama.cpp GGUF model",
                context_length,
            )
            .with_abilities([Ability::ToolUse, Ability::Reasoning])
        }
    }
}

impl EmbeddingModel for Llama {
    fn dim(&self) -> usize {
        self.model.n_embd() as usize
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = aither_core::Result<Vec<f32>>> + Send {
        let model = self.model.clone();
        let cfg = self.inner.clone();
        let input = text.to_owned();

        async move {
            let mut context = create_context(&model, &self.backend, &cfg, true)?;

            let tokens = model
                .str_to_token(&input, AddBos::Never)
                .map_err(|err| LlamaError::Token(err.to_string()))?;
            if tokens.is_empty() {
                return Err(LlamaError::Unsupported(
                    "cannot embed empty token sequence".to_string(),
                )
                .into());
            }

            let mut batch = LlamaBatch::new(tokens.len(), 1);
            batch
                .add_sequence(&tokens, 0, true)
                .map_err(|err| LlamaError::Decode(err.to_string()))?;
            context
                .decode(&mut batch)
                .map_err(|err| LlamaError::Decode(err.to_string()))?;

            if let Ok(embedding) = context.embeddings_seq_ith(0) {
                return Ok(embedding.to_vec());
            }

            let last_index = (tokens.len() - 1) as i32;
            let embedding = context
                .embeddings_ith(last_index)
                .map_err(|err| LlamaError::Decode(err.to_string()))?;
            Ok(embedding.to_vec())
        }
    }
}

#[derive(Debug, Clone)]
struct LlamaConfig {
    model_path: PathBuf,
    chat_template: Option<String>,
    n_ctx: Option<u32>,
    n_threads: i32,
    n_threads_batch: i32,
}

/// Builder for local llama.cpp model configuration.
#[derive(Debug, Clone)]
pub struct Builder {
    model_path: PathBuf,
    n_gpu_layers: u32,
    use_mlock: bool,
    n_ctx: Option<u32>,
    chat_template: Option<String>,
    n_threads: i32,
    n_threads_batch: i32,
    backend: Option<Arc<LlamaBackend>>,
}

impl Builder {
    fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            n_gpu_layers: 0,
            use_mlock: false,
            n_ctx: None,
            chat_template: None,
            n_threads: 4,
            n_threads_batch: 4,
            backend: None,
        }
    }

    /// Number of layers offloaded to GPU.
    #[must_use]
    pub const fn n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Keep model pages in RAM when possible.
    #[must_use]
    pub const fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    /// Generation/embedding context size.
    #[must_use]
    pub const fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = Some(n_ctx);
        self
    }

    /// Override model chat template name/content.
    #[must_use]
    pub fn chat_template(mut self, template: impl Into<String>) -> Self {
        self.chat_template = Some(template.into());
        self
    }

    /// Number of decode threads.
    #[must_use]
    pub const fn n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Number of batch decode threads.
    #[must_use]
    pub const fn n_threads_batch(mut self, n_threads_batch: i32) -> Self {
        self.n_threads_batch = n_threads_batch;
        self
    }

    /// Build the local llama provider.
    pub fn build(self) -> Result<Llama, LlamaError> {
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(self.n_gpu_layers)
            .with_use_mlock(self.use_mlock);
        let backend = if let Some(backend) = self.backend {
            backend
        } else {
            Arc::new(init_backend()?)
        };
        let model = LlamaModel::load_from_file(backend.as_ref(), &self.model_path, &model_params)
            .map_err(|err| LlamaError::Model(err.to_string()))?;

        Ok(Llama {
            inner: Arc::new(LlamaConfig {
                model_path: self.model_path,
                chat_template: self.chat_template,
                n_ctx: self.n_ctx,
                n_threads: self.n_threads,
                n_threads_batch: self.n_threads_batch,
            }),
            model: Arc::new(model),
            backend,
        })
    }
}

fn run_response(
    model: &LlamaModel,
    backend: &LlamaBackend,
    cfg: &LlamaConfig,
    request: LLMRequest,
) -> Result<Vec<Event>, LlamaError> {
    let (messages, parameters, tool_defs) = request.into_parts();

    if messages.iter().any(|msg| !msg.attachments().is_empty()) {
        return Err(LlamaError::Unsupported(
            "attachments are not supported by aither-llama".to_string(),
        ));
    }

    let tool_defs = filter_tool_definitions(tool_defs, &parameters.tool_choice);
    let template = resolve_chat_template(model, cfg)?;
    let prompt = build_prompt(model, &template, &messages, &parameters, &tool_defs)?;

    let mut context = create_context(model, backend, cfg, false)?;
    let prompt_tokens = model
        .str_to_token(&prompt.template_result.prompt, AddBos::Never)
        .map_err(|err| LlamaError::Token(err.to_string()))?;

    if prompt_tokens.is_empty() {
        return Err(LlamaError::Unsupported(
            "empty prompt after template rendering".to_string(),
        ));
    }

    let mut init_batch = LlamaBatch::new(prompt_tokens.len(), 1);
    init_batch
        .add_sequence(&prompt_tokens, 0, false)
        .map_err(|err| LlamaError::Decode(err.to_string()))?;
    context
        .decode(&mut init_batch)
        .map_err(|err| LlamaError::Decode(err.to_string()))?;

    let mut sampler = build_sampler(&parameters);
    sampler.accept_many(prompt_tokens.iter());

    let mut events = Vec::new();
    let mut generated = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut pos = prompt_tokens.len() as i32;
    let max_tokens = parameters.max_tokens.unwrap_or(512);

    for _ in 0..max_tokens {
        let token = sampler.sample(&context, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|err| LlamaError::Token(err.to_string()))?;

        if !piece.is_empty() {
            generated.push_str(&piece);
            events.push(Event::Text(piece));
        }

        let mut step_batch = LlamaBatch::new(1, 1);
        step_batch
            .add(token, pos, &[0], true)
            .map_err(|err| LlamaError::Decode(err.to_string()))?;
        context
            .decode(&mut step_batch)
            .map_err(|err| LlamaError::Decode(err.to_string()))?;
        pos += 1;
    }

    if let Ok(parsed) = prompt
        .template_result
        .parse_response_oaicompat(&generated, false)
    {
        for call in parse_openai_message(&parsed)? {
            events.push(Event::ToolCall(call));
        }
    }

    Ok(events)
}

fn create_context<'a>(
    model: &'a LlamaModel,
    backend: &LlamaBackend,
    cfg: &LlamaConfig,
    embeddings: bool,
) -> Result<llama_cpp_2::context::LlamaContext<'a>, LlamaError> {
    let mut params = LlamaContextParams::default();
    if let Some(n_ctx) = cfg.n_ctx {
        params = params.with_n_ctx(NonZeroU32::new(n_ctx));
    }
    params = params.with_n_threads(cfg.n_threads);
    params = params.with_n_threads_batch(cfg.n_threads_batch);

    if embeddings {
        params = params.with_embeddings(true);
        params = params.with_pooling_type(LlamaPoolingType::Last);
    }

    model
        .new_context(backend, params)
        .map_err(|err| LlamaError::Context(err.to_string()))
}

fn resolve_chat_template(
    model: &LlamaModel,
    cfg: &LlamaConfig,
) -> Result<LlamaChatTemplate, LlamaError> {
    if let Some(template) = &cfg.chat_template {
        return LlamaChatTemplate::new(template).map_err(|err| LlamaError::Model(err.to_string()));
    }
    match model.chat_template(None) {
        Ok(template) => Ok(template),
        Err(_) => {
            LlamaChatTemplate::new("chatml").map_err(|err| LlamaError::Model(err.to_string()))
        }
    }
}

struct Prompt {
    template_result: llama_cpp_2::model::ChatTemplateResult,
}

fn build_prompt(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    messages: &[Message],
    parameters: &Parameters,
    tool_defs: &[ToolDefinition],
) -> Result<Prompt, LlamaError> {
    let messages_json = serde_json::to_string(&messages_to_openai(messages))?;
    let tools_json = tools_to_openai_json(tool_defs);

    let template_result = model
        .apply_chat_template_oaicompat(
            template,
            &OpenAIChatTemplateParams {
                messages_json: &messages_json,
                tools_json: tools_json.as_deref(),
                tool_choice: None,
                json_schema: None,
                grammar: None,
                reasoning_format: None,
                chat_template_kwargs: None,
                add_generation_prompt: true,
                use_jinja: true,
                parallel_tool_calls: true,
                enable_thinking: parameters.include_reasoning,
                add_bos: false,
                add_eos: false,
                parse_tool_calls: !tool_defs.is_empty(),
            },
        )
        .map_err(|err| LlamaError::Model(err.to_string()))?;

    Ok(Prompt { template_result })
}

fn filter_tool_definitions(
    tool_defs: Vec<ToolDefinition>,
    tool_choice: &ToolChoice,
) -> Vec<ToolDefinition> {
    match tool_choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => tool_defs
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => tool_defs,
    }
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAIToolFunction,
}

#[derive(Serialize)]
struct OpenAIToolFunction {
    name: String,
    arguments: String,
}

fn messages_to_openai(messages: &[Message]) -> Vec<OpenAIMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role() {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };

            let tool_calls = if msg.tool_calls().is_empty() {
                None
            } else {
                Some(
                    msg.tool_calls()
                        .iter()
                        .map(|call| OpenAIToolCall {
                            id: call.id.clone(),
                            kind: "function",
                            function: OpenAIToolFunction {
                                name: call.name.clone(),
                                arguments: call.arguments.to_string(),
                            },
                        })
                        .collect(),
                )
            };

            OpenAIMessage {
                role: role.to_string(),
                content: msg.content().to_string(),
                tool_calls,
                tool_call_id: msg.tool_call_id().map(ToString::to_string),
            }
        })
        .collect()
}

fn tools_to_openai_json(tool_defs: &[ToolDefinition]) -> Option<String> {
    if tool_defs.is_empty() {
        return None;
    }

    let tools: Vec<Value> = tool_defs
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.arguments_openai_schema(),
                }
            })
        })
        .collect();

    serde_json::to_string(&tools).ok()
}

fn build_sampler(parameters: &Parameters) -> LlamaSampler {
    let mut samplers = Vec::new();

    if parameters.repetition_penalty.is_some()
        || parameters.frequency_penalty.is_some()
        || parameters.presence_penalty.is_some()
    {
        samplers.push(LlamaSampler::penalties(
            64,
            parameters.repetition_penalty.unwrap_or(1.0),
            parameters.frequency_penalty.unwrap_or(0.0),
            parameters.presence_penalty.unwrap_or(0.0),
        ));
    }

    if let Some(top_k) = parameters.top_k {
        samplers.push(LlamaSampler::top_k(top_k as i32));
    }
    if let Some(top_p) = parameters.top_p {
        samplers.push(LlamaSampler::top_p(top_p, 1));
    }
    if let Some(min_p) = parameters.min_p {
        samplers.push(LlamaSampler::min_p(min_p, 1));
    }

    if parameters.temperature.unwrap_or(0.8) <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.push(LlamaSampler::temp(parameters.temperature.unwrap_or(0.8)));
        samplers.push(LlamaSampler::dist(sampling_seed(parameters.seed)));
    }

    if samplers.is_empty() {
        samplers.push(LlamaSampler::dist(sampling_seed(parameters.seed)));
    }

    LlamaSampler::chain_simple(samplers)
}

fn sampling_seed(seed: Option<u32>) -> u32 {
    if let Some(seed) = seed {
        return seed;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos() as u64)
        .unwrap_or(1);
    ((now ^ (now >> 32)) & 0xFFFF_FFFF) as u32
}

fn parse_openai_message(json_text: &str) -> Result<Vec<ToolCall>, LlamaError> {
    let value: Value = serde_json::from_str(json_text)?;

    let tool_calls = value
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| {
            value
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("tool_calls"))
                .and_then(Value::as_array)
                .cloned()
        })
        .unwrap_or_default();

    let mut calls = Vec::new();
    for call in &tool_calls {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("llama_tool_call")
            .to_string();
        let function = call.get("function").unwrap_or(&Value::Null);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let arguments = function
            .get("arguments")
            .and_then(Value::as_str)
            .map(|raw| {
                serde_json::from_str::<Value>(raw)
                    .unwrap_or_else(|_| Value::String(raw.to_string()))
            })
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

        calls.push(ToolCall::new(id, name, arguments));
    }

    Ok(calls)
}
