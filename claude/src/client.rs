//! Claude API client implementation.

use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest,
        model::{Ability, Profile as ModelProfile},
        oneshot,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use tracing::debug;
use zenwave::{Client, client, header};

use crate::{
    constant::{ANTHROPIC_VERSION, CLAUDE_BASE_URL, DEFAULT_MAX_TOKENS, DEFAULT_MODEL},
    error::ClaudeError,
    request::{MessagesRequest, ParameterSnapshot, convert_tools, to_claude_messages},
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

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.config();
        let (core_messages, parameters, tool_definitions) = request.into_parts();
        let (system_prompt, claude_messages) = to_claude_messages(&core_messages);

        let claude_tools = if tool_definitions.is_empty() {
            None
        } else {
            Some(convert_tools(&tool_definitions))
        };

        let snapshot = ParameterSnapshot::from(&parameters);
        let max_tokens = snapshot.max_tokens.unwrap_or(cfg.default_max_tokens);

        async_stream::stream! {
            // Build and send request
            let request_body = MessagesRequest {
                model: cfg.model.clone(),
                max_tokens,
                messages: claude_messages,
                system: system_prompt,
                stream: true,
                temperature: snapshot.temperature,
                top_p: snapshot.top_p,
                top_k: snapshot.top_k,
                stop_sequences: snapshot.stop_sequences.clone(),
                tools: claude_tools,
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
                    yield Err(ClaudeError::Api(format!("HTTP request failed: {e}")));
                    return;
                }
            };

            // Process SSE events
            let mut state = StreamState::new();

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
                        match parse_event(&e, &mut state) {
                            Ok(llm_events) => {
                                for llm_event in llm_events {
                                    yield Ok(llm_event);
                                }
                            }
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(ClaudeError::from(e));
                        return;
                    }
                }
            }

            // Yield tool call events (NOT executed - consumer handles execution)
            for call in state.tool_calls {
                yield Ok(Event::ToolCall(aither_core::llm::ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.input,
                }));
            }

            debug!("Claude response complete, stop_reason: {:?}", state.stop_reason);
        }
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
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
