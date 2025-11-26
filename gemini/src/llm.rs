use aither_core::{
    Error, LanguageModel,
    llm::{
        LLMRequest, Message, ReasoningStream, ResponseChunk, Role,
        model::{Ability, Parameters, Profile, ReasoningEffort},
        tool::ToolDefinition,
    },
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use tracing::debug;

use crate::{
    client::call_generate,
    config::GeminiBackend,
    error::GeminiError,
    types::{
        FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, GeminiContent, GeminiTool,
        GenerateContentRequest, GenerationConfig, GoogleSearch, PromptFeedback, SafetyRating,
        ThinkingConfig, ToolConfig,
    },
};
use schemars::schema_for;

const MAX_TOOL_ITERATIONS: usize = 8;

impl LanguageModel for GeminiBackend {
    type Error = GeminiError;

    #[allow(clippy::too_many_lines)]
    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl aither_core::llm::LLMResponse<Error = Self::Error> {
        enum State<'tools> {
            Processing {
                iterations: usize,
                contents: Vec<GeminiContent>,
                tools: Option<&'tools mut aither_core::llm::tool::Tools>,
            },
            Done,
        }

        let cfg = self.config();
        let (messages, parameters, tools) = request.into_parts();
        let (system_instruction, contents) = messages_to_gemini(&messages);
        let tool_defs = tools.as_ref().map(|t| t.definitions()).unwrap_or_default();
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

        let stream = futures_lite::stream::unfold(
            State::Processing {
                iterations: 0,
                contents,
                tools,
            },
            move |state| {
                let cfg = cfg.clone();
                let system_instruction = system_instruction.clone();
                let generation_config = generation_config.clone();
                let gemini_tools_payload = gemini_tools_payload.clone();
                let tool_config = tool_config.clone();

                async move {
                    let (iterations, mut contents, mut tools) = match state {
                        State::Processing {
                            iterations,
                            contents,
                            tools,
                        } => (iterations, contents, tools),
                        State::Done => return None,
                    };

                    let next_iteration = iterations + 1;
                    if next_iteration > MAX_TOOL_ITERATIONS {
                        return Some((
                            Err(GeminiError::Api(
                                "exceeded Gemini tool calling iteration limit".into(),
                            )),
                            State::Done,
                        ));
                    }

                    let request = GenerateContentRequest {
                        system_instruction: system_instruction.clone(),
                        contents: contents.clone(),
                        generation_config: generation_config.clone(),
                        tools: gemini_tools_payload.clone(),
                        tool_config: tool_config.clone(),
                        safety_settings: Vec::new(),
                    };

                    debug!("Gemini request: {:?}", request);

                    let response = match call_generate(&cfg, &cfg.text_model, request).await {
                        Ok(r) => r,
                        Err(e) => return Some((Err(e), State::Done)),
                    };

                    debug!("Gemini response: {:?}", response);

                    let Some(candidate) = response.primary_candidate() else {
                        let message = if let Some(feedback) = &response.prompt_feedback {
                            format_prompt_feedback(feedback)
                        } else {
                            "Gemini response missing candidate".to_string()
                        };
                        return Some((Err(GeminiError::Api(message)), State::Done));
                    };

                    let Some(content) = &candidate.content else {
                        return Some((
                            Err(GeminiError::Api("Gemini response missing content".into())),
                            State::Done,
                        ));
                    };

                    if let Some(call) = content.first_function_call() {
                        contents.push(content.clone());
                        let args_json =
                            serde_json::to_string(&call.args).unwrap_or_else(|_| "{}".to_string());
                        let tool_output = if let Some(t) = &mut tools {
                            match t.call(&call.name, &args_json).await {
                                Ok(output) => output,
                                Err(err) => {
                                    return Some((
                                        Err(GeminiError::Api(err.to_string())),
                                        State::Done,
                                    ));
                                }
                            }
                        } else {
                            return Some((
                                Err(GeminiError::Api(
                                    "Tool call requested but no tools available".into(),
                                )),
                                State::Done,
                            ));
                        };

                        let response_value = match serde_json::from_str::<Value>(&tool_output) {
                            Ok(Value::Object(map)) => Value::Object(map),
                            Ok(other) => {
                                let mut map = Map::new();
                                map.insert("result".to_string(), other);
                                Value::Object(map)
                            }
                            Err(_) => {
                                let mut map = Map::new();
                                map.insert("result".to_string(), Value::String(tool_output));
                                Value::Object(map)
                            }
                        };
                        contents.push(GeminiContent::function_response(call.name, response_value));

                        return Some((
                            Ok(ResponseChunk::default()),
                            State::Processing {
                                iterations: next_iteration,
                                contents,
                                tools,
                            },
                        ));
                    }

                    let mut chunk = ResponseChunk::default();
                    for text in content.text_chunks() {
                        chunk.push_text(text);
                    }

                    for part in &content.parts {
                        if let Some(code) = &part.executable_code {
                            let code_block = format!(
                                "\n```{}\n{}\n```\n",
                                code.language.to_lowercase(),
                                code.code
                            );
                            chunk.push_text(code_block);
                        }
                        if let Some(result) = &part.code_execution_result {
                            let output_block = format!("\n```output\n{}\n```\n", result.output);
                            chunk.push_text(output_block);
                        }
                    }

                    for reasoning in content.reasoning_chunks() {
                        chunk.push_reasoning(reasoning);
                    }
                    Some((Ok(chunk), State::Done))
                }
            },
        );

        ReasoningStream::new(stream)
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
            let text = stream.await?;
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
