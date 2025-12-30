//! Claude API client implementation.

use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{
        LLMRequest, ReasoningStream, ResponseChunk,
        model::{Ability, Profile as ModelProfile},
        oneshot,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use tracing::debug;
use zenwave::{Client, client, header};

use crate::{
    constant::{
        ANTHROPIC_VERSION, CLAUDE_BASE_URL, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, MAX_TOOL_ITERATIONS,
    },
    error::ClaudeError,
    request::{
        ContentBlock, ContentPayload, MessagePayload, MessagesRequest, ParameterSnapshot,
        convert_tools, to_claude_messages,
    },
    response::{StreamState, parse_event, should_skip_event},
};

/// Claude chat model client for the Anthropic Messages API.
///
/// # Example
///
/// ```ignore
/// use aither_claude::Claude;
/// use aither_core::{LanguageModel, llm::oneshot};
///
/// let client = Claude::new(std::env::var("ANTHROPIC_API_KEY")?);
///
/// let response = client.respond(oneshot(
///     "You are a helpful assistant.",
///     "What is the capital of France?"
/// )).await?;
///
/// println!("{response}");
/// ```
#[derive(Clone, Debug)]
pub struct Claude {
    inner: Arc<Config>,
}

impl Claude {
    /// Create a new client using the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder(api_key).build()
    }

    /// Start building a Claude client with custom configuration.
    #[must_use]
    pub fn builder(api_key: impl Into<String>) -> Builder {
        Builder::new(api_key)
    }

    /// Override the default model in-place.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).model = sanitize_model(model);
        self
    }

    /// Override the base URL (useful for proxies or local deployments).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Override the default max_tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        Arc::make_mut(&mut self.inner).default_max_tokens = max_tokens;
        self
    }

    pub(crate) fn config(&self) -> Arc<Config> {
        self.inner.clone()
    }
}

impl LanguageModel for Claude {
    type Error = ClaudeError;

    #[allow(clippy::too_many_lines)]
    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl aither_core::llm::LLMResponse<Error = Self::Error> {
        enum State<'tools> {
            Processing {
                iterations: usize,
                messages: Vec<MessagePayload>,
                tools: Option<&'tools mut aither_core::llm::tool::Tools>,
            },
            Done,
        }

        let cfg = self.config();
        let (core_messages, parameters, tools) = request.into_parts();
        let (system_prompt, claude_messages) = to_claude_messages(&core_messages);
        let tool_defs = tools.as_ref().map(|t| t.definitions()).unwrap_or_default();
        let claude_tools = if tool_defs.is_empty() {
            None
        } else {
            Some(convert_tools(tool_defs))
        };
        let snapshot = ParameterSnapshot::from(&parameters);
        let max_tokens = snapshot.max_tokens.unwrap_or(cfg.default_max_tokens);

        let stream = futures_lite::stream::unfold(
            State::Processing {
                iterations: 0,
                messages: claude_messages,
                tools,
            },
            move |state| {
                let cfg = cfg.clone();
                let system_prompt = system_prompt.clone();
                let claude_tools = claude_tools.clone();
                let snapshot = snapshot.clone();

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
                            Err(ClaudeError::Api(
                                "exceeded Claude tool calling iteration limit".into(),
                            )),
                            State::Done,
                        ));
                    }

                    // Build and send request
                    let request_body = MessagesRequest {
                        model: cfg.model.clone(),
                        max_tokens,
                        messages: messages.clone(),
                        system: system_prompt.clone(),
                        stream: true,
                        temperature: snapshot.temperature,
                        top_p: snapshot.top_p,
                        top_k: snapshot.top_k,
                        stop_sequences: snapshot.stop_sequences.clone(),
                        tools: claude_tools.clone(),
                    };

                    debug!("Claude request: {:?}", request_body);

                    let endpoint = cfg.request_url("/v1/messages");
                    let mut backend = client();
                    let mut builder = backend.post(endpoint);

                    // Claude-specific headers
                    builder = builder.header("x-api-key", cfg.api_key.clone());
                    builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
                    builder = builder.header(header::CONTENT_TYPE.as_str(), "application/json");
                    builder = builder.header(header::ACCEPT.as_str(), "text/event-stream");
                    builder = builder.header(header::USER_AGENT.as_str(), "aither-claude/0.1");

                    let sse_stream = match builder.json_body(&request_body).sse().await {
                        Ok(stream) => stream,
                        Err(e) => {
                            return Some((
                                Err(ClaudeError::Api(format!("HTTP request failed: {e}"))),
                                State::Done,
                            ));
                        }
                    };

                    // Collect all events and process
                    let mut state = StreamState::new();
                    let mut collected_text = String::new();
                    let mut collected_reasoning = String::new();

                    let events: Vec<_> = sse_stream
                        .filter_map(|event| match event {
                            Ok(e) if !should_skip_event(&e) => Some(Ok(e)),
                            Ok(_) => None,
                            Err(e) => Some(Err(e)),
                        })
                        .collect()
                        .await;

                    for event in events {
                        match event {
                            Ok(e) => {
                                if let Err(e) = parse_event(&e, &mut state) {
                                    return Some((Err(e), State::Done));
                                }
                            }
                            Err(e) => return Some((Err(ClaudeError::from(e)), State::Done)),
                        }
                    }

                    // Extract collected text/reasoning from stream state
                    for block in &state.blocks {
                        match block {
                            crate::response::BlockState::Text(text) => {
                                collected_text.push_str(text);
                            }
                            crate::response::BlockState::Thinking(thinking) => {
                                collected_reasoning.push_str(thinking);
                            }
                            crate::response::BlockState::ToolUse { .. } => {
                                // Tool use blocks don't contribute to text
                            }
                        }
                    }

                    debug!("Claude response state: {:?}", state);

                    // Check if we need to handle tool calls
                    if state.has_tool_calls() {
                        // Build assistant message with tool_use blocks
                        let mut assistant_blocks: Vec<ContentBlock> = Vec::new();

                        // Add any text content first
                        if !collected_text.is_empty() {
                            assistant_blocks.push(ContentBlock::Text {
                                text: collected_text.clone(),
                            });
                        }

                        // Add tool_use blocks
                        for call in &state.tool_calls {
                            assistant_blocks.push(ContentBlock::ToolUse {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                input: call.input.clone(),
                            });
                        }

                        messages.push(MessagePayload {
                            role: "assistant",
                            content: ContentPayload::Blocks(assistant_blocks),
                        });

                        // Execute tools and build user message with results
                        let mut result_blocks: Vec<ContentBlock> = Vec::new();

                        for call in &state.tool_calls {
                            let args_json = serde_json::to_string(&call.input)
                                .unwrap_or_else(|_| "{}".to_string());

                            let tool_output = if let Some(t) = &mut tools {
                                match t.call(&call.name, &args_json).await {
                                    Ok(output) => output,
                                    Err(err) => {
                                        return Some((
                                            Err(ClaudeError::Api(err.to_string())),
                                            State::Done,
                                        ));
                                    }
                                }
                            } else {
                                return Some((
                                    Err(ClaudeError::Api(
                                        "Tool call requested but no tools available".into(),
                                    )),
                                    State::Done,
                                ));
                            };

                            result_blocks.push(ContentBlock::ToolResult {
                                tool_use_id: call.id.clone(),
                                content: tool_output,
                            });
                        }

                        messages.push(MessagePayload {
                            role: "user",
                            content: ContentPayload::Blocks(result_blocks),
                        });

                        // Continue iteration
                        return Some((
                            Ok(ResponseChunk::default()),
                            State::Processing {
                                iterations: next_iteration,
                                messages,
                                tools,
                            },
                        ));
                    }

                    // No tool calls - return the collected response
                    let mut final_chunk = ResponseChunk::default();
                    final_chunk.push_text(collected_text);
                    final_chunk.push_reasoning(collected_reasoning);
                    Some((Ok(final_chunk), State::Done))
                }
            },
        );

        ReasoningStream::new(stream)
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        self.respond(oneshot(
            "Continue the user provided text without additional commentary.",
            prefix,
        ))
    }

    fn profile(&self) -> impl core::future::Future<Output = ModelProfile> + Send {
        let cfg = self.inner.clone();
        async move {
            let mut profile = ModelProfile::new(
                cfg.model.clone(),
                "Anthropic",
                cfg.model.clone(),
                "Claude model by Anthropic",
                200_000, // Claude supports 200K context
            )
            .with_abilities([Ability::ToolUse, Ability::Vision]);

            for ability in &cfg.native_abilities {
                if !profile.abilities.contains(ability) {
                    profile.abilities.push(*ability);
                }
            }
            profile
        }
    }
}

/// Builder for Claude clients.
#[derive(Debug)]
pub struct Builder {
    api_key: String,
    base_url: String,
    model: String,
    default_max_tokens: u32,
    native_abilities: Vec<Ability>,
}

impl Builder {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: CLAUDE_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            native_abilities: Vec::new(),
        }
    }

    /// Set a custom API base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Select a model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = sanitize_model(model);
        self
    }

    /// Set the default max_tokens for requests.
    #[must_use]
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Declare extra native capabilities supported by the model.
    #[must_use]
    pub fn native_capabilities(mut self, abilities: impl IntoIterator<Item = Ability>) -> Self {
        for ability in abilities {
            if !self.native_abilities.contains(&ability) {
                self.native_abilities.push(ability);
            }
        }
        self
    }

    /// Mark this model as having built-in PDF understanding.
    #[must_use]
    pub fn enable_native_pdf(self) -> Self {
        self.native_capabilities([Ability::Pdf])
    }

    /// Consume the builder and create a Claude client.
    #[must_use]
    pub fn build(self) -> Claude {
        Claude {
            inner: Arc::new(Config {
                api_key: self.api_key,
                base_url: self.base_url,
                model: self.model,
                default_max_tokens: self.default_max_tokens,
                native_abilities: self.native_abilities,
            }),
        }
    }
}

/// Internal configuration for the Claude client.
#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) default_max_tokens: u32,
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
}

fn sanitize_model(model: impl Into<String>) -> String {
    model.into().trim().to_string()
}
