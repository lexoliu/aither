use aither_core::{
    LanguageModel,
    llm::{
        Attachment, Event, GenerateError, LLMRequest, Message, ReasoningState, Role, Usage,
        model::{Ability, Parameters, Profile, ToolChoice},
        tool::ToolDefinition,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use tracing::debug;

use crate::PROVIDER_NAME;
use crate::{
    client::stream_generate,
    config::Gemini,
    error::GeminiError,
    types::{
        FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, GeminiContent, GeminiTool,
        GenerateContentRequest, GenerationConfig, GoogleSearch, Part, PromptFeedback, SafetyRating,
        ThinkingConfig, ThinkingLevel, ToolConfig, UsageMetadata,
    },
};
use schemars::schema_for;

impl LanguageModel for Gemini {
    type Error = GeminiError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.config();
        respond_stream(cfg.clone(), request)
    }

    fn generate<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        mut request: LLMRequest,
    ) -> impl core::future::Future<Output = Result<T, GenerateError<Self::Error>>> + Send {
        let schema = schema_for!(T);
        let mut params = request.parameters().clone();
        params.structured_outputs = true;
        params.response_format = Some(schema);
        request = request.with_parameters(params);

        let stream = self.respond(request);
        async move {
            let text = aither_core::llm::collect_text(stream)
                .await
                .map_err(GenerateError::Provider)?;
            serde_json::from_str::<T>(&text).map_err(|source| GenerateError::Parse {
                source,
                response: text.chars().take(500).collect(),
            })
        }
    }

    fn profile(&self) -> impl core::future::Future<Output = Profile> + Send {
        let cfg = self.config().clone();
        async move {
            let model_name = cfg.text_model.trim_start_matches("models/").to_string();

            // Fetch actual context window from API, fallback to models database
            let context_length = match crate::client::get_model_info(&cfg, &cfg.text_model).await {
                Ok(info) => info.input_token_limit,
                Err(e) => {
                    tracing::debug!("API did not return context_length: {e}");
                    // Fallback to models database
                    aither_models::lookup(&model_name)
                        .and_then(aither_models::ModelEntry::max_input_tokens)
                        .unwrap_or_else(|| {
                            panic!(
                                "Gemini model '{model_name}' missing context metadata from provider and aither-models"
                            )
                        })
                }
            };

            let mut profile = Profile::new(
                model_name.clone(),
                "google",
                model_name,
                "Gemini Developer API model",
                context_length,
            )
            .with_abilities([Ability::ToolUse, Ability::Vision, Ability::Audio]);
            for ability in &cfg.native_abilities {
                if !profile.abilities.contains(ability) {
                    profile.abilities.push(*ability);
                }
            }
            profile
        }
    }
}

fn respond_stream(
    cfg: crate::config::GeminiConfig,
    request: LLMRequest,
) -> impl Stream<Item = Result<Event, GeminiError>> + Send {
    Box::pin(respond_stream_inner(cfg, request))
}

/// Assembles the Gemini request for a provider-agnostic one.
///
/// Async because file attachments are uploaded before the request is sent.
///
/// # Errors
///
/// Returns [`GeminiError`] if the request carries another provider's cache
/// settings, names an exact tool that was not supplied, or an attachment cannot
/// be resolved.
async fn build_generate_request(
    cfg: &crate::config::GeminiConfig,
    request: LLMRequest,
) -> Result<GenerateContentRequest, GeminiError> {
    let (messages, parameters, tool_defs) = request.into_parts();

    if parameters.cache.openai.is_some() || parameters.cache.claude.is_some() {
        return Err(GeminiError::Api(
            "Gemini provider only accepts cache.gemini settings".to_string(),
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let messages = crate::attachments::resolve_messages(cfg, messages).await?;
    // wasm32 has no filesystem, so there is nothing to upload and no use for
    // the config here.
    #[cfg(target_arch = "wasm32")]
    let _ = cfg;

    let (system_instruction, contents) = messages_to_gemini(&messages).await?;

    let tool_defs: Vec<ToolDefinition> = match &parameters.tool_choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => tool_defs
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => tool_defs,
    };
    let has_function_tools = !tool_defs.is_empty();

    if let ToolChoice::Exact(name) = &parameters.tool_choice
        && !has_function_tools
    {
        return Err(GeminiError::Api(format!(
            "Exact tool choice '{name}' is not present in tool definitions"
        )));
    }

    let mut tools: Vec<GeminiTool> = Vec::new();
    if has_function_tools {
        tools.push(GeminiTool::FunctionTool {
            function_declarations: convert_tool_definitions(tool_defs),
        });
    }

    // Google's own tools cannot be combined with a narrowed tool choice: the
    // model would be free to answer with a tool the caller did not ask about.
    let builtins_allowed = !matches!(
        parameters.tool_choice,
        ToolChoice::None | ToolChoice::Exact(_)
    );
    let native = &parameters.native_tools.gemini;
    if builtins_allowed && (parameters.websearch || native.google_search) {
        tools.push(GeminiTool::GoogleSearchTool {
            google_search: GoogleSearch {},
        });
    }
    if builtins_allowed && (parameters.code_execution || native.code_execution) {
        tools.push(GeminiTool::CodeExecutionTool {
            code_execution: crate::types::CodeExecution {},
        });
    }
    if builtins_allowed && native.url_context {
        tools.push(GeminiTool::UrlContextTool {
            url_context: crate::types::UrlContext {},
        });
    }

    Ok(GenerateContentRequest {
        system_instruction,
        contents,
        generation_config: build_generation_config(&parameters, None)?,
        tools,
        tool_config: build_tool_config(&parameters, has_function_tools),
        safety_settings: Vec::new(),
        cached_content: parameters
            .cache
            .gemini
            .as_ref()
            .map(|cache| cache.cached_content.clone()),
    })
}

/// Turns one candidate's content into the events it represents, in wire order:
/// reasoning, then text, then built-in tool output, then function calls.
fn events_from_content(content: &GeminiContent) -> Vec<Event> {
    let mut events = Vec::new();

    events.extend(content.reasoning_chunks().into_iter().map(Event::Reasoning));
    events.extend(
        content
            .text_chunks()
            .into_iter()
            .filter(|text| !text.is_empty())
            .map(Event::Text),
    );

    for part in &content.parts {
        if let Some(code) = &part.executable_code {
            events.push(Event::BuiltInToolResult {
                tool: "code_execution".to_string(),
                result: format!("```{}\n{}\n```", code.language.to_lowercase(), code.code),
            });
        }
        if let Some(result) = &part.code_execution_result {
            events.push(Event::BuiltInToolResult {
                tool: "code_execution".to_string(),
                result: format!("```output\n{}\n```", result.output),
            });
        }
    }

    // Tool calls are reported, never executed: the consumer decides.
    events.extend(
        content
            .function_call_parts()
            .into_iter()
            .map(|(call, signature)| {
                // Gemini signs each function call rather than the turn, so the
                // signature rides on the call it belongs to.
                Event::ToolCall(aither_core::llm::ToolCall {
                    id: tool_call_id(),
                    name: call.name,
                    arguments: call.args,
                    reasoning_state: signature
                        .map(|value| ReasoningState::new(PROVIDER_NAME, value)),
                })
            }),
    );

    events
}

fn respond_stream_inner(
    cfg: crate::config::GeminiConfig,
    request: LLMRequest,
) -> impl Stream<Item = Result<Event, GeminiError>> + Send {
    async_stream::stream! {
        let gemini_request = match build_generate_request(&cfg, request).await {
            Ok(request) => request,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        debug!("Gemini request: {:?}", gemini_request);

        // Use streaming endpoint for true streaming output
        let stream = match stream_generate(&cfg, &cfg.text_model, gemini_request).await {
            Ok(s) => s,
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        futures_lite::pin!(stream);
        let mut usage: Option<Usage> = None;
        let mut finish_reason: Option<String> = None;

        while let Some(result) = stream.next().await {
            let response = match result {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e);
                    continue;
                }
            };

            debug!("Gemini stream chunk: {:?}", response);

            if let Some(meta) = &response.usage_metadata {
                usage = Some(usage_from_metadata(meta));
            }

            let Some(candidate) = response.primary_candidate() else {
                // Chunks without candidates carry metadata, not content — but a
                // blocked prompt is reported this way and must not pass silently.
                if let Some(feedback) = &response.prompt_feedback {
                    yield Err(GeminiError::Api(format_prompt_feedback(feedback)));
                }
                continue;
            };

            let Some(content) = &candidate.content else {
                if let Some(reason) = candidate.finish_reason.clone() {
                    finish_reason = Some(reason);
                    break;
                }
                continue;
            };

            for event in events_from_content(content) {
                yield Ok(event);
            }

            if let Some(reason) = candidate.finish_reason.clone() {
                finish_reason = Some(reason);
                break;
            }
        }

        if let Some(mut final_usage) = usage {
            final_usage.stop_reason = finish_reason;
            yield Ok(Event::Usage(final_usage));
        } else if let Some(reason) = finish_reason {
            yield Ok(Event::Usage(Usage {
                stop_reason: Some(reason),
                ..Usage::default()
            }));
        }
    }
}

/// Simple UUID v4 generator for tool call IDs.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{timestamp:032x}")
}

fn tool_call_id() -> String {
    format!("gemini_{}", uuid_v4())
}

fn parse_tool_response(content: &str) -> Option<(String, serde_json::Value, Option<String>)> {
    let content = content.trim_start();
    if !content.starts_with('[') {
        return None;
    }
    let end = content.find(']')?;
    let header = &content[1..end];
    let mut header_split = header.splitn(2, ':');
    let id = header_split.next()?.trim();
    let name = header_split.next()?.trim().to_string();
    let rest = &content[end + 1..];
    let output = rest.strip_prefix(' ').unwrap_or(rest);
    let _ = id;
    let signature = None;
    let response_value = match serde_json::from_str::<serde_json::Value>(output) {
        Ok(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
        Ok(other) => {
            let mut map = serde_json::Map::new();
            map.insert("result".to_string(), other);
            serde_json::Value::Object(map)
        }
        Err(_) => {
            let mut map = serde_json::Map::new();
            map.insert(
                "result".to_string(),
                serde_json::Value::String(output.to_string()),
            );
            serde_json::Value::Object(map)
        }
    };
    Some((name, response_value, signature))
}

const fn usage_from_metadata(meta: &UsageMetadata) -> Usage {
    Usage {
        prompt_tokens: meta.prompt_token_count,
        completion_tokens: meta.candidates_token_count,
        total_tokens: meta.total_token_count,
        reasoning_tokens: meta.thoughts_token_count,
        cache_read_tokens: meta.cached_content_token_count,
        cache_write_tokens: None,
        cost_usd: None,
        stop_reason: None,
    }
}

/// Indexes every tool call by id, yielding its function name and, where the
/// model signed it, its thought signature.
///
/// A tool *result* has to look both up: Gemini scopes the signature to the call
/// rather than the result, so the result cannot carry it.
fn index_tool_calls(messages: &[Message]) -> (HashMap<&str, &str>, HashMap<&str, &str>) {
    let mut names = HashMap::new();
    let mut signatures = HashMap::new();
    for message in messages {
        for tc in message.tool_calls() {
            names.insert(tc.id.as_str(), tc.name.as_str());
            if let Some(signature) = tc
                .reasoning_state
                .as_ref()
                .and_then(|state| state.payload_for(PROVIDER_NAME))
            {
                signatures.insert(tc.id.as_str(), signature);
            }
        }
    }
    (names, signatures)
}

async fn messages_to_gemini(
    messages: &[Message],
) -> Result<(Option<GeminiContent>, Vec<GeminiContent>), GeminiError> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    let (tool_call_names, tool_call_signatures) = index_tool_calls(messages);

    for message in messages {
        match message.role() {
            Role::System => system_parts.push(Part::text(message.content())),
            Role::User => {
                let attachments = message.attachments();
                if attachments.is_empty() {
                    contents.push(GeminiContent::text("user", message.content()));
                } else {
                    // Build parts with attachments
                    let mut parts = Vec::new();

                    // Add attachment parts first
                    for attachment in attachments {
                        parts.push(url_to_part(attachment).await?);
                    }

                    // Add text content
                    if !message.content().is_empty() {
                        parts.push(Part::text(message.content()));
                    }

                    contents.push(GeminiContent::with_parts("user", parts));
                }
            }
            Role::Tool => {
                // Get the function name from the tool_call_id
                let tool_call_id = message.tool_call_id().unwrap_or("");
                let base_id = tool_call_id.to_string();
                let signature = tool_call_signatures
                    .get(tool_call_id)
                    .map(|value| (*value).to_string());

                if let Some(&function_name) = tool_call_names.get(tool_call_id) {
                    // Parse the content as JSON, or wrap it as a string result
                    let response_value =
                        match serde_json::from_str::<serde_json::Value>(message.content()) {
                            Ok(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
                            Ok(other) => {
                                let mut map = serde_json::Map::new();
                                map.insert("result".to_string(), other);
                                serde_json::Value::Object(map)
                            }
                            Err(_) => {
                                let mut map = serde_json::Map::new();
                                map.insert(
                                    "result".to_string(),
                                    serde_json::Value::String(message.content().to_string()),
                                );
                                serde_json::Value::Object(map)
                            }
                        };
                    contents.push(GeminiContent::function_response_with_signature(
                        function_name.to_string(),
                        response_value,
                        signature,
                    ));
                } else {
                    // Fallback: try legacy format or send as user message
                    if let Some((name, response, sig)) = parse_tool_response(message.content()) {
                        contents.push(GeminiContent::function_response_with_signature(
                            name, response, sig,
                        ));
                    } else {
                        // Last resort - wrap as text
                        debug!("Tool message without matching tool_call_id: {}", base_id);
                        contents.push(GeminiContent::text("user", message.content()));
                    }
                }
            }
            Role::Assistant => {
                let tool_calls = message.tool_calls();
                if tool_calls.is_empty() {
                    // Regular text response
                    contents.push(GeminiContent::text("model", message.content()));
                } else {
                    // Assistant message with function calls
                    let mut parts = Vec::new();

                    // Add text content if present
                    if !message.content().is_empty() {
                        parts.push(Part::text(message.content()));
                    }

                    // Add function call parts with thought signatures extracted from IDs
                    for tc in tool_calls {
                        let signature = tc
                            .reasoning_state
                            .as_ref()
                            .and_then(|state| state.payload_for(PROVIDER_NAME))
                            .map(String::from);
                        parts.push(Part::function_call_with_signature(
                            tc.name.clone(),
                            tc.arguments.clone(),
                            signature,
                        ));
                    }

                    contents.push(GeminiContent::with_parts("model", parts));
                }
            }
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiContent::system(system_parts))
    };

    Ok((system_instruction, contents))
}

fn format_prompt_feedback(feedback: &PromptFeedback) -> String {
    let reason = feedback
        .block_reason
        .as_deref()
        .unwrap_or("unknown reason")
        .to_string();
    if feedback.safety_ratings.is_empty() {
        return format!("Gemini response blocked: {reason}");
    }

    let ratings = feedback
        .safety_ratings
        .iter()
        .map(format_safety_rating)
        .collect::<Vec<_>>()
        .join(", ");
    format!("Gemini response blocked: {reason}; safety: {ratings}")
}

fn format_safety_rating(rating: &SafetyRating) -> String {
    let status = rating
        .blocked
        .map_or("unspecified", |b| if b { "blocked" } else { "allowed" });
    let probability = rating.probability.as_deref().unwrap_or("unknown");
    format!("{} ({status}, probability: {probability})", rating.category)
}

/// # Errors
///
/// Returns [`GeminiError`] when the requested reasoning effort has no Gemini
/// equivalent.
fn build_generation_config(
    parameters: &Parameters,
    modalities: Option<Vec<String>>,
) -> Result<Option<GenerationConfig>, GeminiError> {
    let thinking_config = build_thinking_config(parameters)?;
    let mut config = GenerationConfig {
        temperature: parameters.temperature,
        top_p: parameters.top_p,
        top_k: parameters.top_k,
        max_output_tokens: parameters.max_tokens.map(
            #[allow(clippy::cast_possible_wrap)]
            |value| value as i32,
        ),
        stop_sequences: parameters.stop.clone(),
        response_modalities: modalities,
        thinking_config,
        seed: parameters.seed.map(
            #[allow(clippy::cast_possible_wrap)]
            |value| value as i32,
        ),
        presence_penalty: parameters.presence_penalty,
        frequency_penalty: parameters.frequency_penalty,
        response_logprobs: parameters.logprobs,
        logprobs: parameters.top_logprobs.map(i32::from),
        ..Default::default()
    };
    if let Some(schema) = &parameters.response_format {
        config.response_mime_type = Some("application/json".into());
        config.response_json_schema = Some(schema.clone().to_value());
    } else if parameters.structured_outputs {
        config.response_mime_type = Some("application/json".into());
    }
    if config.is_meaningful() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn build_tool_config(parameters: &Parameters, has_tools: bool) -> Option<ToolConfig> {
    if !has_tools {
        return None;
    }
    let (mode, allowed) = match &parameters.tool_choice {
        ToolChoice::None => return None,
        ToolChoice::Auto => (FunctionCallingMode::Auto, None),
        ToolChoice::Required => (FunctionCallingMode::Any, None),
        ToolChoice::Exact(name) => (FunctionCallingMode::Any, Some(vec![name.clone()])),
    };
    Some(ToolConfig {
        function_calling_config: Some(FunctionCallingConfig {
            mode,
            allowed_function_names: allowed,
        }),
    })
}

/// Builds `thinkingConfig` from the portable reasoning parameters.
///
/// Depth (`reasoning_effort`) and visibility (`include_reasoning`) are separate
/// controls: asking for a deeper think does not require the thoughts to be
/// returned, and vice versa. Either one alone is enough to emit the config.
/// # Errors
///
/// Returns [`GeminiError`] when the requested effort has no Gemini equivalent.
fn build_thinking_config(parameters: &Parameters) -> Result<Option<ThinkingConfig>, GeminiError> {
    let thinking_level = parameters
        .reasoning_effort
        .map(ThinkingLevel::try_from)
        .transpose()?;
    if !parameters.include_reasoning && thinking_level.is_none() {
        return Ok(None);
    }
    Ok(Some(ThinkingConfig {
        include_thoughts: parameters.include_reasoning.then_some(true),
        thinking_level,
    }))
}

fn convert_tool_definitions(defs: Vec<ToolDefinition>) -> Vec<FunctionDeclaration> {
    defs.into_iter()
        .map(|tool| FunctionDeclaration {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: Some(tool.arguments_openai_schema()),
        })
        .collect()
}

/// Convert a typed attachment to a Gemini content part.
async fn url_to_part(attachment: &Attachment) -> Result<Part, GeminiError> {
    let url = attachment.url();
    let media_type = attachment.media_type().as_ref();
    match url.scheme() {
        "data" => parse_data_url(url.as_str(), media_type),
        "file" => read_file_to_part(url, media_type).await,
        "http" | "https" => Ok(Part::from_file(media_type, url.as_str())),
        scheme => Err(GeminiError::Api(format!(
            "Gemini does not support attachment URL scheme '{scheme}'"
        ))),
    }
}

fn parse_data_url(url: &str, media_type: &str) -> Result<Part, GeminiError> {
    use base64::Engine;

    let after_data = url
        .strip_prefix("data:")
        .ok_or_else(|| GeminiError::Api("Malformed attachment data URL".to_string()))?;
    let (header, data) = after_data
        .split_once(',')
        .ok_or_else(|| GeminiError::Api("Attachment data URL is missing payload".to_string()))?;
    let encoded_media_type = header.strip_suffix(";base64").ok_or_else(|| {
        GeminiError::Api("Attachment data URL must use base64 encoding".to_string())
    })?;
    if encoded_media_type != media_type {
        return Err(GeminiError::Api(format!(
            "Attachment MIME type '{media_type}' does not match data URL MIME type '{encoded_media_type}'"
        )));
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
    Ok(Part::inline_media(media_type, bytes))
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_file_to_part(url: &url::Url, media_type: &str) -> Result<Part, GeminiError> {
    let path = url.to_file_path().map_err(|()| {
        GeminiError::Api("Attachment file URL could not be converted to a path".to_string())
    })?;
    let data = async_fs::read(&path).await.map_err(|error| {
        GeminiError::Api(format!(
            "Failed to read attachment '{}': {error}",
            path.display()
        ))
    })?;
    Ok(Part::inline_media(media_type, data))
}

#[cfg(target_arch = "wasm32")]
async fn read_file_to_part(_url: &url::Url, _media_type: &str) -> Result<Part, GeminiError> {
    Err(GeminiError::Api(
        "file:// attachments are not supported on wasm32".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::model::ReasoningEffort;

    #[tokio::test]
    async fn data_attachments_preserve_declared_media_types() {
        for media_type in ["image/png", "audio/wav", "video/mp4", "application/pdf"] {
            let attachment = Attachment::new(
                format!("data:{media_type};base64,AA==")
                    .parse()
                    .expect("data URL"),
                media_type.parse().expect("MIME type"),
            );
            let part = url_to_part(&attachment).await.expect("encode attachment");
            let value = serde_json::to_value(part).expect("serialize part");
            assert_eq!(value["inlineData"]["mimeType"], media_type);
        }
    }

    #[tokio::test]
    async fn remote_attachment_uses_file_data_with_media_type() {
        let attachment = Attachment::new(
            "https://ai.google.dev/static/gemini-api/docs/files/sample.pdf"
                .parse()
                .expect("remote URL"),
            "application/pdf".parse().expect("PDF MIME"),
        );
        let part = url_to_part(&attachment).await.expect("encode attachment");
        let value = serde_json::to_value(part).expect("serialize part");
        assert_eq!(value["fileData"]["mimeType"], "application/pdf");
        assert_eq!(
            value["fileData"]["fileUri"],
            "https://ai.google.dev/static/gemini-api/docs/files/sample.pdf"
        );
    }

    #[test]
    fn data_url_mime_mismatch_fails() {
        let error = parse_data_url("data:image/png;base64,AA==", "audio/wav")
            .expect_err("MIME mismatch must fail");
        assert!(error.to_string().contains("does not match"));
    }

    fn thinking_json(params: &Parameters) -> serde_json::Value {
        let config = build_thinking_config(params)
            .expect("effort supported by Gemini")
            .expect("thinking config");
        serde_json::to_value(config).expect("serialize thinking config")
    }

    /// Gemini names the field `thinkingLevel`; there is no `tokenBudget`, so a
    /// misnamed field is accepted by serde and then ignored on the wire.
    #[test]
    fn thinking_config_uses_gemini_field_names() {
        let value = thinking_json(&Parameters::default().reasoning_effort(ReasoningEffort::Medium));
        assert_eq!(value["thinkingLevel"], "medium");
        assert_eq!(value.get("tokenBudget"), None);
        assert_eq!(value.get("thinkingBudget"), None);
    }

    /// `thinkingLevel` and the legacy `thinkingBudget` are mutually exclusive:
    /// sending both is a 400, so only the level is ever modelled.
    #[test]
    fn thinking_config_never_pairs_a_budget_with_a_level() {
        for effort in [
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ] {
            let value = thinking_json(&Parameters::default().reasoning_effort(effort));
            let object = value.as_object().expect("thinking config object");
            assert!(
                object.keys().all(|key| key != "thinkingBudget"),
                "effort {effort:?} paired a budget with a level"
            );
            assert!(object.contains_key("thinkingLevel"));
        }
    }

    #[test]
    fn minimal_effort_maps_to_the_minimal_level() {
        let value =
            thinking_json(&Parameters::default().reasoning_effort(ReasoningEffort::Minimal));
        assert_eq!(value["thinkingLevel"], "minimal");
    }

    /// Gemini's ladder stops at High. Clamping `XHigh` or `Max` down to High would
    /// hand back less reasoning than the caller asked for, silently.
    #[test]
    fn efforts_above_gemini_ceiling_are_rejected() {
        for effort in [ReasoningEffort::XHigh, ReasoningEffort::Max] {
            let error = build_thinking_config(&Parameters::default().reasoning_effort(effort))
                .expect_err("effort above the Gemini ceiling must fail");
            assert!(
                error.to_string().contains("tops out at High"),
                "unexpected error for {effort:?}: {error}"
            );
        }
    }

    /// The signature used to be base64'd into the tool-call id behind a `|ts|`
    /// separator, because core had nowhere to put it. It now rides on the call,
    /// and the id is a plain identifier again.
    #[tokio::test]
    async fn thought_signature_round_trips_without_mangling_the_id() {
        let call = aither_core::llm::ToolCall::new("gemini_abc", "lookup", serde_json::json!({}))
            .with_reasoning_state(ReasoningState::new(PROVIDER_NAME, "sig-xyz"));
        let message = Message::assistant_with_tool_calls("", vec![call]);

        let (_, contents) = messages_to_gemini(&[message]).await.expect("convert");
        let value = serde_json::to_value(&contents).expect("serialize contents");
        let part = &value[0]["parts"][0];

        assert_eq!(part["thoughtSignature"], "sig-xyz");
        assert_eq!(part["functionCall"]["name"], "lookup");
    }

    #[tokio::test]
    async fn generated_tool_call_ids_carry_no_smuggled_signature() {
        assert!(!tool_call_id().contains("|ts|"));
    }

    /// Signatures are Gemini's; another provider's state must not be replayed.
    #[tokio::test]
    async fn foreign_reasoning_state_is_not_sent_as_a_signature() {
        let call = aither_core::llm::ToolCall::new("gemini_abc", "lookup", serde_json::json!({}))
            .with_reasoning_state(ReasoningState::new("anthropic", "sig-from-claude"));
        let message = Message::assistant_with_tool_calls("", vec![call]);

        let (_, contents) = messages_to_gemini(&[message]).await.expect("convert");
        let value = serde_json::to_value(&contents).expect("serialize contents");
        assert_eq!(value[0]["parts"][0].get("thoughtSignature"), None);
    }

    /// Gemini has no "no thinking" level; the caller should omit effort instead.
    #[test]
    fn none_effort_is_rejected() {
        let error =
            build_thinking_config(&Parameters::default().reasoning_effort(ReasoningEffort::None))
                .expect_err("None effort must fail");
        assert!(error.to_string().contains("no 'none' thinking level"));
    }

    /// Depth and visibility are independent: asking for effort alone must still
    /// produce a config, and asking for thoughts alone must still produce one.
    #[test]
    fn effort_and_visibility_are_independent() {
        let effort_only =
            thinking_json(&Parameters::default().reasoning_effort(ReasoningEffort::High));
        assert_eq!(effort_only["thinkingLevel"], "high");
        assert_eq!(effort_only.get("includeThoughts"), None);

        let thoughts_only = thinking_json(&Parameters::default().include_reasoning(true));
        assert_eq!(thoughts_only["includeThoughts"], true);
        assert_eq!(thoughts_only.get("thinkingLevel"), None);

        assert!(
            build_thinking_config(&Parameters::default())
                .expect("no effort requested")
                .is_none()
        );
    }

    /// These reached the wire only if `build_generation_config` populates them;
    /// they were declared on the struct but never filled.
    #[test]
    fn generation_config_forwards_every_sampling_parameter() {
        let params = Parameters::default()
            .seed(42)
            .presence_penalty(0.5)
            .frequency_penalty(0.25)
            .logprobs(true)
            .top_logprobs(3);
        let config = build_generation_config(&params, None)
            .expect("effort supported by Gemini")
            .expect("generation config");
        let value = serde_json::to_value(config).expect("serialize generation config");
        assert_eq!(value["seed"], 42);
        assert_eq!(value["presencePenalty"], 0.5);
        assert_eq!(value["frequencyPenalty"], 0.25);
        assert_eq!(value["responseLogprobs"], true);
        assert_eq!(value["logprobs"], 3);
    }
}
