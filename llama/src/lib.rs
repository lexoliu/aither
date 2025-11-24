//! llama.cpp-backed `LanguageModel` built on top of the `llama_cpp` crate.
//!
//! This crate wraps the safe high-level bindings from [`llama_cpp`] and bridges
//! them into the `aither-core` [`LanguageModel`] trait. Each [`Llama`]
//! instance loads a GGUF checkpoint, and every [`LanguageModel::respond`]
//! call spins up a dedicated inference session so calls can run concurrently.
//! Standard llama.cpp sampling controls (temperature, top-p, repetition
//! penalties, stop sequences, etc.) are wired into [`aither_core::llm::model::Parameters`].
//!
//! ```no_run
//! use aither::llm::{LanguageModel, LLMRequest, Message};
//! use aither_llama::Llama;
//!
//! # fn main() -> anyhow::Result<()> {
//! let llama = Llama::builder("./models/llama-3.1-8b-instruct.Q4_K_M.gguf")
//!     .context_length(8_192)
//!     .threads(8)
//!     .build()?;
//!
//! let request = LLMRequest::new([Message::user("List three famous observatories.")]);
//! let mut response = llama.respond(request);
//!
//! futures_lite::future::block_on(async {
//!     use futures_lite::StreamExt;
//!     while let Some(chunk) = response.next().await {
//!         print!("{}", chunk?);
//!     }
//!     Ok::<_, anyhow::Error>(())
//! })?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

use std::{
    ffi::OsStr,
    fmt,
    future::Future,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};

use aither_core::{
    LanguageModel,
    llm::{
        LLMRequest, LLMResponse, Message, ReasoningStream, Role,
        model::{Parameters, Profile},
    },
};
use async_channel::Sender;
use llama_cpp::{
    LlamaContextError, LlamaLoadError, LlamaModel, LlamaParams, SessionParams,
    standard_sampler::{SamplerStage, StandardSampler},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use thiserror::Error;

/// Errors produced by the llama.cpp provider.
#[derive(Debug, Error)]
pub enum LlamaError {
    /// Loading the GGUF failed.
    #[error("failed to load model: {0}")]
    Load(#[from] LlamaLoadError),
    /// Creating or driving a session failed.
    #[error("llama.cpp session error: {0}")]
    Context(#[from] LlamaContextError),
    /// Sending a streaming chunk failed because the receiver went away.
    #[error("response stream receiver dropped")]
    ResponseChannelClosed,
}

/// Builder for [`Llama`] models.
pub struct LlamaBuilder {
    path: PathBuf,
    llama_params: LlamaParams,
    session_params: SessionParams,
    sampling_defaults: SamplingDefaults,
    description: Option<String>,
    display_name: Option<String>,
}

impl fmt::Debug for LlamaBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlamaBuilder")
            .field("path", &self.path)
            .field("context_length", &self.session_params.n_ctx)
            .field("threads", &self.session_params.n_threads)
            .finish()
    }
}

impl LlamaBuilder {
    /// Creates a new builder pointing at the provided GGUF checkpoint.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            llama_params: LlamaParams::default(),
            session_params: SessionParams::default(),
            sampling_defaults: SamplingDefaults::default(),
            description: None,
            display_name: None,
        }
    }

    /// Sets the number of GPU layers to offload (requires enabling GPU features on `llama_cpp`).
    #[must_use]
    pub fn gpu_layers(mut self, layers: u32) -> Self {
        self.llama_params.n_gpu_layers = layers;
        self
    }

    /// Overrides the context window advertised to llama.cpp.
    #[must_use]
    pub fn context_length(mut self, n_ctx: u32) -> Self {
        self.session_params.n_ctx = n_ctx;
        self
    }

    /// Sets the CPU thread count for generation.
    #[must_use]
    pub const fn threads(mut self, threads: u32) -> Self {
        self.session_params.n_threads = threads;
        self.session_params.n_threads_batch = threads;
        self
    }

    /// Sets the prompt-processing batch size.
    #[must_use]
    pub const fn batch(mut self, batch: u32) -> Self {
        self.session_params.n_batch = batch;
        self.session_params.n_ubatch = batch;
        self
    }

    /// Sets a friendly model name exposed through [`Profile`].
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Sets a profile description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adjusts the default sampling temperature.
    #[must_use]
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.sampling_defaults.temperature = temp;
        self
    }

    /// Adjusts the default nucleus sampling threshold.
    #[must_use]
    pub const fn top_p(mut self, value: f32) -> Self {
        self.sampling_defaults.top_p = value;
        self
    }

    /// Adjusts the default top-k value.
    #[must_use]
    pub const fn top_k(mut self, value: i32) -> Self {
        self.sampling_defaults.top_k = value;
        self
    }

    /// Adjusts the default repetition penalty.
    #[must_use]
    pub const fn repetition_penalty(mut self, value: f32) -> Self {
        self.sampling_defaults.repetition_penalty = value;
        self
    }

    /// Overrides the default max tokens per completion.
    #[must_use]
    pub const fn max_tokens(mut self, max: usize) -> Self {
        self.sampling_defaults.max_tokens = max;
        self
    }

    /// Loads the GGUF checkpoint and builds a [`Llama`] language model.
    ///
    /// # Errors
    /// Returns [`LlamaError::Load`] if llama.cpp fails to load the checkpoint.
    pub fn build(self) -> Result<Llama, LlamaError> {
        let model = LlamaModel::load_from_file(&self.path, self.llama_params)?;
        let profile = Profile::new(
            self.display_name
                .clone()
                .or_else(|| infer_name(&self.path))
                .unwrap_or_else(|| "local-llama".to_string()),
            "local",
            self.path
                .file_name()
                .and_then(OsStr::to_str)
                .unwrap_or("llama.cpp")
                .to_string(),
            self.description
                .clone()
                .unwrap_or_else(|| format!("llama.cpp model loaded from {}", self.path.display())),
            self.session_params.n_ctx,
        );

        Ok(Llama {
            inner: Arc::new(LlamaInner {
                model,
                session_params: self.session_params,
                sampling: self.sampling_defaults,
                profile,
                seed_rng: Mutex::new(StdRng::from_entropy()),
            }),
        })
    }
}

/// llama.cpp-backed language model.
#[derive(Clone)]
pub struct Llama {
    inner: Arc<LlamaInner>,
}

impl fmt::Debug for Llama {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Llama")
            .field("context_length", &self.inner.session_params.n_ctx)
            .finish()
    }
}

impl Llama {
    /// Starts building a llama.cpp-backed language model.
    #[must_use]
    pub fn builder(path: impl Into<PathBuf>) -> LlamaBuilder {
        LlamaBuilder::new(path)
    }

    fn resolve_sampling(&self, params: &Parameters) -> SamplingPlan {
        let defaults = &self.inner.sampling;
        SamplingPlan {
            temperature: params.temperature.unwrap_or(defaults.temperature).max(0.05),
            top_p: params.top_p.unwrap_or(defaults.top_p).clamp(0.0, 1.0),
            top_k: params
                .top_k
                .map_or(defaults.top_k, |value| value as i32)
                .max(1),
            min_p: params.min_p,
            repetition_penalty: params
                .repetition_penalty
                .unwrap_or(defaults.repetition_penalty)
                .max(0.1),
            frequency_penalty: params
                .frequency_penalty
                .unwrap_or(defaults.frequency_penalty),
            presence_penalty: params.presence_penalty.unwrap_or(defaults.presence_penalty),
            max_tokens: params
                .max_tokens
                .map(|value| value as usize)
                .unwrap_or(defaults.max_tokens)
                .max(1),
            seed: params.seed,
            stop_sequences: params.stop.clone().unwrap_or_default(),
        }
    }
}

impl LanguageModel for Llama {
    type Error = LlamaError;

    fn respond(&self, request: LLMRequest) -> impl LLMResponse<Error = Self::Error> {
        let prompt = format_prompt(request.messages());
        let sampling = self.resolve_sampling(request.parameters());
        let mut session_params = self.inner.session_params.clone();
        session_params.seed = sampling.seed.unwrap_or_else(|| self.inner.next_seed());

        let inner = self.inner.clone();
        let (sender, receiver) = async_channel::bounded(32);

        thread::spawn(move || {
            let result = run_session(
                inner.model.clone(),
                session_params,
                prompt,
                sampling,
                sender.clone(),
            );
            if let Err(err) = result {
                let _ = sender.send_blocking(Err(err));
            }
        });

        ReasoningStream::new(receiver)
    }

    fn profile(&self) -> impl Future<Output = Profile> + Send {
        let profile = self.inner.profile.clone();
        async move { profile }
    }
}

struct LlamaInner {
    model: LlamaModel,
    session_params: SessionParams,
    sampling: SamplingDefaults,
    profile: Profile,
    seed_rng: Mutex<StdRng>,
}

impl LlamaInner {
    fn next_seed(&self) -> u32 {
        self.seed_rng
            .lock()
            .map(|mut rng| rng.r#gen::<u32>())
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
struct SamplingDefaults {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    max_tokens: usize,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 256,
        }
    }
}

#[derive(Clone)]
struct SamplingPlan {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: Option<f32>,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    max_tokens: usize,
    seed: Option<u32>,
    stop_sequences: Vec<String>,
}

fn run_session(
    model: LlamaModel,
    session_params: SessionParams,
    prompt: String,
    sampling: SamplingPlan,
    sender: Sender<Result<aither_core::llm::ResponseChunk, LlamaError>>,
) -> Result<(), LlamaError> {
    let mut session = model.create_session(session_params.clone())?;
    session.advance_context(prompt).map_err(LlamaError::from)?;

    let sampler = build_sampler(&sampling);
    let handle = session
        .start_completing_with(sampler, sampling.max_tokens)
        .map_err(LlamaError::from)?;

    let mut pending = String::new();
    let stops = sampling.stop_sequences;
    let max_stop = stops.iter().map(String::len).max().unwrap_or(0);

    for piece in handle.into_strings() {
        if piece.is_empty() {
            continue;
        }
        pending.push_str(&piece);

        if stops.is_empty() {
            let chunk = std::mem::take(&mut pending);
            send_text(&chunk, &sender)?;
            continue;
        }

        if let Some(idx) = stop_position(&pending, &stops) {
            if idx > 0 {
                send_text(&pending[..idx], &sender)?;
            }
            return Ok(());
        }

        if pending.len() > max_stop {
            let emit_len = pending.len() - max_stop;
            if emit_len > 0 {
                let chunk = pending[..emit_len].to_string();
                pending.replace_range(..emit_len, "");
                send_text(&chunk, &sender)?;
            }
        }
    }

    if !pending.is_empty() {
        send_text(&pending, &sender)?;
    }

    Ok(())
}

fn stop_position(buffer: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter_map(|stop| buffer.find(stop).map(|idx| (idx, stop.len())))
        .min_by_key(|(idx, _)| *idx)
        .map(|(idx, _)| idx)
}

fn build_sampler(plan: &SamplingPlan) -> StandardSampler {
    let mut stages = Vec::new();
    stages.push(SamplerStage::RepetitionPenalty {
        repetition_penalty: plan.repetition_penalty,
        frequency_penalty: plan.frequency_penalty,
        presence_penalty: plan.presence_penalty,
        last_n: 64,
    });
    stages.push(SamplerStage::TopK(plan.top_k));
    if plan.top_p < 1.0 {
        stages.push(SamplerStage::TopP(plan.top_p));
    }
    if let Some(min_p) = plan.min_p {
        stages.push(SamplerStage::MinP(min_p));
    }
    stages.push(SamplerStage::Temperature(plan.temperature));
    StandardSampler::new_softmax(stages, 1)
}

fn send_text(
    text: &str,
    sender: &Sender<Result<aither_core::llm::ResponseChunk, LlamaError>>,
) -> Result<(), LlamaError> {
    if text.is_empty() {
        return Ok(());
    }
    let mut chunk = aither_core::llm::ResponseChunk::default();
    chunk.push_text(text);
    sender
        .send_blocking(Ok(chunk))
        .map_err(|_| LlamaError::ResponseChannelClosed)
}

fn infer_name(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(OsStr::to_str)
        .map(|stem| stem.trim().to_string())
        .filter(|stem| !stem.is_empty())
}

fn format_prompt(messages: &[Message]) -> String {
    let mut system = String::new();
    let mut transcript = String::new();

    for message in messages {
        match message.role() {
            Role::System => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(message.content());
            }
            Role::User => append_block(&mut transcript, "User", message.content()),
            Role::Assistant => append_block(&mut transcript, "Assistant", message.content()),
            Role::Tool => append_block(&mut transcript, "Tool Output", message.content()),
        }
    }

    if system.is_empty() {
        system = "You are a helpful assistant.".to_string();
    }

    format!(
        "### System Prompt\n{}\n\n### Conversation\n{}\n\nAssistant:\n",
        system.trim(),
        transcript.trim()
    )
}

fn append_block(buffer: &mut String, label: &str, content: &str) {
    if !buffer.is_empty() {
        buffer.push('\n');
    }
    buffer.push_str(label);
    buffer.push_str(":\n");
    buffer.push_str(content);
    buffer.push('\n');
}
