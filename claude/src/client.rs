//! Claude API client implementation.

use std::{sync::Arc, time::Duration};

use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest,
        model::{Ability, Profile as ModelProfile, ToolChoice},
        oneshot,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use tracing::debug;
use zenwave::{Client, client, header};

use crate::{
    constant::{
        ANTHROPIC_VERSION, CLAUDE_BASE_URL, DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL, MID_CONVERSATION_OUTPUT_CONFIG_BETA,
    },
    error::ClaudeError,
    request::{
        MessagePayload, MessagesRequest, ParameterSnapshot, apply_cache_strategy,
        build_output_config, build_thinking, convert_native_tools, convert_tools,
        filter_tool_definitions, to_claude_messages, tool_choice_payload,
    },
    response::{MessagesResponse, StreamState, parse_event, response_events, should_skip_event},
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

    /// Override the default `max_tokens`.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        Arc::make_mut(&mut self.inner).default_max_tokens = max_tokens;
        self
    }

    /// Override how long a response stream may stay silent before it ends with
    /// a [`FailureKind::Network`](aither_core::llm::FailureKind::Network)
    /// error. Defaults to 90 seconds.
    #[must_use]
    pub fn with_idle_timeout(mut self, idle_timeout: Duration) -> Self {
        Arc::make_mut(&mut self.inner).idle_timeout = idle_timeout;
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
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.config();
        let (core_messages, parameters, tool_definitions) = request.into_parts();
        let snapshot = ParameterSnapshot::from(&parameters);
        let filtered_tool_definitions =
            filter_tool_definitions(tool_definitions, &snapshot.tool_choice);
        let missing_exact_tool = match &snapshot.tool_choice {
            ToolChoice::Exact(name)
                if filtered_tool_definitions.is_empty()
                    && !native_tool_name_enabled(&snapshot, name) =>
            {
                Some(name.clone())
            }
            _ => None,
        };
        let mut claude_tool_payloads = convert_tools(&filtered_tool_definitions);
        claude_tool_payloads.extend(convert_native_tools(&snapshot.native_tools));
        let has_tools = !claude_tool_payloads.is_empty();
        let mut claude_tools = has_tools.then_some(claude_tool_payloads);
        let claude_tool_choice = tool_choice_payload(
            &snapshot.tool_choice,
            has_tools,
            snapshot.parallel_tool_calls,
        );

        let max_tokens = snapshot.max_tokens.unwrap_or(cfg.default_max_tokens);

        async_stream::stream! {
            let (mut system_prompt, mut claude_messages) =
                match to_claude_messages(&core_messages).await {
                    Ok(payload) => payload,
                    Err(error) => {
                        yield Err(ClaudeError::Api(error));
                        return;
                    }
                };

            if let Some(tool_name) = &missing_exact_tool {
                yield Err(ClaudeError::Api(format!(
                    "Exact tool choice '{tool_name}' is not present in tool definitions"
                )));
                return;
            }

            let top_level_cache_control = if let Some(cache) = snapshot.cache {
                match apply_cache_strategy(
                    &mut system_prompt,
                    &mut claude_messages,
                    &mut claude_tools,
                    cache,
                ) {
                    Ok(control) => control,
                    Err(error) => {
                        yield Err(ClaudeError::Api(error));
                        return;
                    }
                }
            } else {
                None
            };

            let thinking = build_thinking(&snapshot);
            let output_config = match build_output_config(&snapshot) {
                Ok(config) => config,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            // Build and send request
            let stream_requested = snapshot.stream.unwrap_or(true);
            let request_body = MessagesRequest {
                model: cfg.model.clone(),
                max_tokens,
                messages: claude_messages,
                system: system_prompt,
                stream: stream_requested,
                temperature: snapshot.temperature,
                top_p: snapshot.top_p,
                top_k: snapshot.top_k,
                stop_sequences: snapshot.stop_sequences.clone(),
                tools: claude_tools,
                tool_choice: claude_tool_choice,
                thinking,
                output_config,
                cache_control: top_level_cache_control,
            };

            debug!("Claude request: {:?}", request_body);
            let effort_updates = request_body
                .messages
                .iter()
                .any(MessagePayload::is_effort_update);

            let endpoint = cfg.request_url("/v1/messages");
            let mut backend = client();

            // Claude-specific headers
            let builder = match backend
                .post(endpoint)
                .and_then(|b| b.header("x-api-key", cfg.api_key.clone()))
                .and_then(|b| b.header("anthropic-version", ANTHROPIC_VERSION))
                .and_then(|b| b.header(header::CONTENT_TYPE.as_str(), "application/json"))
                .and_then(|b| {
                    if stream_requested {
                        b.header(header::ACCEPT.as_str(), "text/event-stream")
                    } else {
                        Ok(b)
                    }
                })
                .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-claude/0.1"))
                .and_then(|b| {
                    if effort_updates {
                        b.header("anthropic-beta", MID_CONVERSATION_OUTPUT_CONFIG_BETA)
                    } else {
                        Ok(b)
                    }
                })
                .and_then(|b| b.json_body(&request_body))
            {
                Ok(b) => b,
                Err(e) => {
                    yield Err(ClaudeError::Http(e));
                    return;
                }
            };

            if !stream_requested {
                match builder.json::<MessagesResponse>().await {
                    Ok(response) => {
                        for event in response_events(response) {
                            yield Ok(event);
                        }
                    }
                    Err(e) => yield Err(ClaudeError::Http(e)),
                }
                return;
            }

            let sse_stream = match builder.sse().await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(ClaudeError::Http(e));
                    return;
                }
            };
            futures_lite::pin!(sse_stream);

            let mut state = StreamState::new();

            loop {
                // Bytes that never come are the signature of a socket that
                // died while the machine slept.
                let next = futures_lite::future::or(
                    async { Some(sse_stream.next().await) },
                    async {
                        aither_core::time::sleep(cfg.idle_timeout).await;
                        None
                    },
                )
                .await;
                let Some(event) = next else {
                    yield Err(ClaudeError::IdleTimeout { idle: cfg.idle_timeout });
                    return;
                };
                let Some(event) = event else { break };
                let event = match event {
                    Ok(event) => event,
                    Err(e) => {
                        yield Err(ClaudeError::from(e));
                        return;
                    }
                };
                if should_skip_event(&event) {
                    continue;
                }
                match parse_event(&event, &mut state) {
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

            // Tool calls are reported, never executed: the consumer decides.
            for call in state.tool_calls {
                // Claude scopes thinking to the turn, not to the individual
                // call, so the state rides on the assistant message instead.
                yield Ok(Event::tool_call(call.id, call.name, call.input));
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

    fn profile(
        &self,
    ) -> impl core::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            // Try to fetch context window from proxy API (e.g., OpenRouter)
            // Native Anthropic API doesn't expose this
            let context_length = match fetch_model_context_length(&cfg).await {
                Ok(len) => len,
                Err(e) => {
                    tracing::debug!("API did not return context_length: {e}");
                    // Fallback to models database
                    aither_models::lookup(&cfg.model)
                        .and_then(aither_models::ModelEntry::max_input_tokens)
                        .ok_or_else(|| {
                            ClaudeError::Api(format!(
                                "cannot determine context length for model '{model}': \
                                 {models_url} request failed ({e}) and aither-models has no entry",
                                model = cfg.model,
                                models_url = cfg.request_url("/models"),
                            ))
                        })?
                }
            };

            let mut profile = ModelProfile::new(
                cfg.model.clone(),
                "Anthropic",
                cfg.model.clone(),
                "Claude model by Anthropic",
                context_length,
            )
            .with_abilities([Ability::ToolUse, Ability::Vision]);
            if takes_per_message_effort(&cfg) {
                profile.abilities.push(Ability::PerMessageEffort);
            }
            profile.default_reasoning_effort = aither_models::default_reasoning_effort(&cfg.model);

            for ability in &cfg.native_abilities {
                if !profile.abilities.contains(ability) {
                    profile.abilities.push(*ability);
                }
            }
            Ok(profile)
        }
    }
}

/// Whether this client's model takes a per-message effort change: the
/// registry lists the model for it, and the client talks to the Claude API
/// itself, one of the platforms Anthropic documents the beta on (a proxy
/// behind another base URL is not).
fn takes_per_message_effort(cfg: &Config) -> bool {
    cfg.base_url.trim_end_matches('/') == CLAUDE_BASE_URL
        && aither_models::lookup(&cfg.model)
            .is_some_and(|entry| entry.has_ability(Ability::PerMessageEffort))
}

fn native_tool_name_enabled(snapshot: &ParameterSnapshot, name: &str) -> bool {
    let tools = &snapshot.native_tools;
    matches!(
        name,
        "web_search" if tools.web_search
    ) || matches!(
        name,
        "web_fetch" if tools.web_fetch
    ) || matches!(
        name,
        "code_execution" if tools.code_execution
    ) || matches!(
        name,
        "bash" if tools.bash
    ) || matches!(
        name,
        "str_replace_based_edit_tool" if tools.text_editor.is_some()
    )
}

/// Try to fetch context length from the models endpoint.
async fn fetch_model_context_length(cfg: &Config) -> Result<u32, ClaudeError> {
    #[derive(serde::Deserialize)]
    struct ModelsResponse {
        data: Vec<ModelInfo>,
    }
    #[derive(serde::Deserialize)]
    struct ModelInfo {
        id: String,
        /// Anthropic uses `max_tokens`, `OpenRouter` uses `context_length`
        #[serde(default, alias = "max_tokens")]
        context_length: Option<u32>,
    }

    let url = format!("{}/models", cfg.base_url.trim_end_matches('/'));
    let mut backend = client();
    let mut req = backend.get(&url).map_err(ClaudeError::Http)?;

    // Anthropic uses x-api-key header, proxies use Bearer token
    if cfg.base_url.contains("anthropic.com") {
        req = req
            .header("x-api-key", cfg.api_key.clone())
            .map_err(ClaudeError::Http)?
            .header("anthropic-version", ANTHROPIC_VERSION)
            .map_err(ClaudeError::Http)?;
    } else {
        req = req
            .header(
                header::AUTHORIZATION.as_str(),
                format!("Bearer {}", cfg.api_key),
            )
            .map_err(ClaudeError::Http)?;
    }

    let response: ModelsResponse = req.json().await.map_err(ClaudeError::Http)?;

    for model in response.data {
        if model.id == cfg.model
            && let Some(ctx) = model.context_length
        {
            return Ok(ctx);
        }
    }

    Err(ClaudeError::Api(format!(
        "Model '{}' not found or missing context_length",
        cfg.model
    )))
}

/// Builder for Claude clients.
#[derive(Debug)]
pub struct Builder {
    api_key: String,
    base_url: String,
    model: String,
    default_max_tokens: u32,
    native_abilities: Vec<Ability>,
    idle_timeout: Duration,
}

impl Builder {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: CLAUDE_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            native_abilities: Vec::new(),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
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

    /// Set the default `max_tokens` for requests.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Set how long a response stream may stay silent before it ends with a
    /// [`FailureKind::Network`](aither_core::llm::FailureKind::Network) error;
    /// 90 seconds by default.
    #[must_use]
    pub const fn idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.idle_timeout = idle_timeout;
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
                idle_timeout: self.idle_timeout,
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
    pub(crate) idle_timeout: Duration,
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

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::{Message, model::Parameters, model::ReasoningEffort};

    /// A loopback server that records the one request it receives, headers
    /// and body, and answers it with an error.
    fn capturing_server() -> (String, std::sync::mpsc::Receiver<String>) {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind stub server");
        let addr = listener.local_addr().expect("stub server address");
        let (sender, received) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept the request");
            let mut request = Vec::new();
            let mut chunk = [0_u8; 8192];
            let body_start = loop {
                let read = stream.read(&mut chunk).expect("read the request");
                request.extend_from_slice(chunk.get(..read).expect("read within the chunk"));
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let head =
                String::from_utf8_lossy(request.get(..body_start).expect("head")).into_owned();
            let length = head.lines().find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|length| length.trim().parse::<usize>().expect("a numeric length"))
            });
            let complete = |request: &[u8]| {
                length.map_or_else(
                    || request.ends_with(b"0\r\n\r\n"),
                    |length| request.len() >= body_start + length,
                )
            };
            while !complete(&request) {
                let read = stream.read(&mut chunk).expect("read the body");
                request.extend_from_slice(chunk.get(..read).expect("read within the chunk"));
            }
            let raw = String::from_utf8_lossy(request.get(body_start..).expect("body"));
            let body = if length.is_some() {
                raw.into_owned()
            } else {
                dechunk(&raw)
            };
            sender
                .send(format!("{head}{body}"))
                .expect("the test is listening");
            let _ = stream.write_all(
                b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
            );
        });
        (format!("http://{addr}"), received)
    }

    /// The payload of a chunked transfer-encoded body.
    fn dechunk(mut raw: &str) -> String {
        let mut body = String::new();
        loop {
            let (size, rest) = raw.split_once("\r\n").expect("a chunk size line");
            let size = usize::from_str_radix(size.trim(), 16).expect("a hex chunk size");
            if size == 0 {
                return body;
            }
            body.push_str(rest.get(..size).expect("the chunk is complete"));
            raw = rest
                .get(size..)
                .and_then(|rest| rest.strip_prefix("\r\n"))
                .expect("the chunk ends with CRLF");
        }
    }

    /// A request carrying an effort update sends the beta header and the
    /// effort-only system message, with the request-level effort as given.
    #[tokio::test]
    async fn an_effort_update_sends_the_beta_and_the_per_message_form() {
        let (base_url, received) = capturing_server();
        let client = Claude::builder("key")
            .base_url(base_url)
            .model("claude-opus-5-5")
            .build();
        let request = LLMRequest::new(vec![
            Message::user("Plan the migration."),
            Message::assistant("Here is the plan."),
            Message::configuration(ReasoningEffort::Low),
            Message::user("Now rename the config file."),
        ])
        .with_parameters(
            Parameters::default()
                .reasoning_effort(ReasoningEffort::High)
                .stream(false),
        );
        let stream = client.respond(request);
        futures_lite::pin!(stream);
        let _ = stream.next().await;

        let sent = received.recv().expect("the request arrives");
        let (head, body) = sent.split_once("\r\n\r\n").expect("head and body");
        assert!(
            head.to_ascii_lowercase()
                .contains("anthropic-beta: mid-conversation-output-config-2026-07-01"),
            "{head}"
        );
        let body: serde_json::Value = serde_json::from_str(body).expect("a JSON body");
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(
            body["messages"][2],
            serde_json::json!({"role": "system", "content": [], "output_config": {"effort": "low"}})
        );
    }

    #[test]
    fn per_message_effort_is_the_claude_apis_on_the_listed_models() {
        let config = |model: &str, base_url: &str| {
            Claude::builder("key")
                .model(model)
                .base_url(base_url)
                .build()
                .config()
        };
        assert!(takes_per_message_effort(&config(
            "claude-opus-5-5",
            CLAUDE_BASE_URL
        )));
        assert!(!takes_per_message_effort(&config(
            "claude-sonnet-5",
            CLAUDE_BASE_URL
        )));
        assert!(!takes_per_message_effort(&config(
            "claude-opus-5-5",
            "https://openrouter.ai/api/v1"
        )));
    }
}
