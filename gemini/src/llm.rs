use aither_core::{
    Error, LanguageModel,
    llm::{
        Event, LLMRequest, Message, Role,
        model::{Ability, Parameters, Profile, ReasoningEffort},
        tool::ToolDefinition,
    },
};
use futures_core::Stream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tracing::debug;

use crate::{
    client::call_generate,
    config::Gemini,
    error::GeminiError,
    types::{
        FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, GeminiContent, GeminiTool,
        GenerateContentRequest, GenerationConfig, GoogleSearch, PromptFeedback, SafetyRating,
        ThinkingConfig, ToolConfig,
    },
};
use schemars::schema_for;

impl LanguageModel for Gemini {
    type Error = GeminiError;

    fn respond(&self, request: LLMRequest) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.config();
        respond_stream(cfg.clone(), request)
    }

    fn generate<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        mut request: LLMRequest,
    ) -> impl core::future::Future<Output = aither_core::Result<T>> + Send {
        let schema = schema_for!(T);
        let mut params = request.parameters().clone();
        params.structured_outputs = true;
        params.response_format = Some(schema);
        request = request.with_parameters(params);

        let stream = self.respond(request);
        async move {
            let text = aither_core::llm::collect_text(stream).await?;
            serde_json::from_str::<T>(&text)
                .map_err(|err| Error::new(err).context("failed to parse structured output"))
        }
    }

    fn profile(&self) -> impl core::future::Future<Output = Profile> + Send {
        let cfg = self.config();
        async move {
            let display = cfg.text_model.trim_start_matches("models/").to_string();
            let mut profile = Profile::new(
                display.clone(),
                "google",
                display,
                "Gemini Developer API model",
                1_000_000,
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

fn respond_stream_inner(
    cfg: crate::config::GeminiConfig,
    request: LLMRequest,
) -> impl Stream<Item = Result<Event, GeminiError>> + Send {
    async_stream::stream! {
        let (messages, parameters, tool_defs) = request.into_parts();
        let (system_instruction, contents) = messages_to_gemini(&messages);
        let tool_config = build_tool_config(&parameters, !tool_defs.is_empty());
        let mut gemini_tools_payload: Vec<GeminiTool> = Vec::new();

        // Add function declarations from aither-core Tools
        if !tool_defs.is_empty() {
            gemini_tools_payload.push(GeminiTool::FunctionTool {
                function_declarations: convert_tool_definitions(tool_defs),
            });
        }

        // Add native Google Search tool if enabled in parameters
        if parameters.websearch {
            gemini_tools_payload.push(GeminiTool::GoogleSearchTool {
                google_search: GoogleSearch {},
            });
        }

        // Add native Code Execution tool if enabled in parameters
        if parameters.code_execution {
            gemini_tools_payload.push(GeminiTool::CodeExecutionTool {
                code_execution: crate::types::CodeExecution {},
            });
        }

        let generation_config = build_generation_config(&parameters, None);

        let gemini_request = GenerateContentRequest {
            system_instruction,
            contents,
            generation_config,
            tools: gemini_tools_payload,
            tool_config,
            safety_settings: Vec::new(),
        };

        debug!("Gemini request: {:?}", gemini_request);

        let response = match call_generate(&cfg, &cfg.text_model, gemini_request).await {
            Ok(r) => r,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        debug!("Gemini response: {:?}", response);

        let Some(candidate) = response.primary_candidate() else {
            let message = if let Some(feedback) = &response.prompt_feedback {
                format_prompt_feedback(feedback)
            } else {
                "Gemini response missing candidate".to_string()
            };
            yield Err(GeminiError::Api(message));
            return;
        };

        let Some(content) = &candidate.content else {
            yield Err(GeminiError::Api("Gemini response missing content".into()));
            return;
        };

        // Emit reasoning events first
        for reasoning in content.reasoning_chunks() {
            yield Ok(Event::Reasoning(reasoning));
        }

        // Emit text events
        for text in content.text_chunks() {
            if !text.is_empty() {
                yield Ok(Event::Text(text));
            }
        }

        // Emit built-in tool results (code execution)
        for part in &content.parts {
            if let Some(code) = &part.executable_code {
                let code_block = format!(
                    "```{}\n{}\n```",
                    code.language.to_lowercase(),
                    code.code
                );
                yield Ok(Event::BuiltInToolResult {
                    tool: "code_execution".to_string(),
                    result: code_block,
                });
            }
            if let Some(result) = &part.code_execution_result {
                let output_block = format!("```output\n{}\n```", result.output);
                yield Ok(Event::BuiltInToolResult {
                    tool: "code_execution".to_string(),
                    result: output_block,
                });
            }
        }

        // Emit tool call events (NOT executed - consumer handles execution)
        if let Some(call) = content.first_function_call() {
            // Generate a unique ID for this tool call
            let call_id = format!("gemini_{}", uuid_v4());
            yield Ok(Event::ToolCall(aither_core::llm::ToolCall {
                id: call_id,
                name: call.name.clone(),
                arguments: call.args.clone(),
            }));
        }
    }
}

/// Simple UUID v4 generator for tool call IDs.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:032x}", timestamp)
}

fn messages_to_gemini(messages: &[Message]) -> (Option<GeminiContent>, Vec<GeminiContent>) {
    let mut contents = Vec::new();
    for message in messages {
        match message.role() {
            Role::System => contents.push(GeminiContent::text("user", message.content())),
            Role::User | Role::Tool => {
                contents.push(GeminiContent::text("user", message.content()));
            }
            Role::Assistant => contents.push(GeminiContent::text("model", message.content())),
        }
    }

    (None, contents)
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
        .map(|b| if b { "blocked" } else { "allowed" })
        .unwrap_or("unspecified");
    let probability = rating.probability.as_deref().unwrap_or("unknown");
    format!("{} ({status}, probability: {probability})", rating.category)
}

fn build_generation_config(
    parameters: &Parameters,
    modalities: Option<Vec<String>>,
) -> Option<GenerationConfig> {
    let thinking_config = build_thinking_config(parameters);
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
        ..Default::default()
    };
    if let Some(schema) = &parameters.response_format {
        config.response_mime_type = Some("application/json".into());
        config.response_json_schema = Some(schema.clone().to_value());
    } else if parameters.structured_outputs {
        config.response_mime_type = Some("application/json".into());
    }
    if config.is_meaningful() {
        Some(config)
    } else {
        None
    }
}

fn build_tool_config(parameters: &Parameters, has_tools: bool) -> Option<ToolConfig> {
    if !has_tools {
        return None;
    }
    let allowed = parameters
        .tool_choice
        .clone()
        .filter(|choices| !choices.is_empty());
    Some(ToolConfig {
        function_calling_config: Some(FunctionCallingConfig {
            mode: if allowed.is_some() {
                FunctionCallingMode::Any
            } else {
                FunctionCallingMode::Auto
            },
            allowed_function_names: allowed,
        }),
    })
}

fn build_thinking_config(parameters: &Parameters) -> Option<ThinkingConfig> {
    if !parameters.include_reasoning {
        return None;
    }
    Some(ThinkingConfig {
        include_thoughts: Some(parameters.include_reasoning),
        token_budget: parameters.reasoning_effort.map(|effort| match effort {
            ReasoningEffort::Minimum => 0,
            ReasoningEffort::Low => 1024,
            ReasoningEffort::Medium => 4096,
            ReasoningEffort::High => 10240,
        }),
        thinking_level: parameters.reasoning_effort.map(|effort| {
            // Gemini does not have a direct mapping for Minimum, so we map it to Low.
            match effort {
                ReasoningEffort::Minimum => "low",
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
            }
            .to_string()
        }),
    })
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
