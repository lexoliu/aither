use aither_core::{
    LanguageModel,
    llm::{
        LLMRequest, Message, ReasoningStream, ResponseChunk, Role,
        model::{Ability, Parameters, Profile},
        tool::ToolDefinition,
    },
};
use schemars::Schema;
use serde_json::Value;

use crate::{
    client::call_generate,
    config::GeminiBackend,
    error::GeminiError,
    types::{
        FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, GeminiContent, GeminiTool,
        GenerateContentRequest, GenerationConfig, ThinkingConfig, ToolConfig,
    },
};

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
        let tools_payload = convert_tool_definitions(tool_defs);
        let generation_config = build_generation_config(&parameters, None);
        let thinking_config = build_thinking_config(&parameters);

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
                let tools_payload = tools_payload.clone();
                let tool_config = tool_config.clone();
                let thinking_config = thinking_config.clone();

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
                        tools: tools_payload.clone(),
                        tool_config: tool_config.clone(),
                        thinking_config: thinking_config.clone(),
                        safety_settings: Vec::new(),
                    };

                    let response = match call_generate(cfg.clone(), &cfg.text_model, request).await
                    {
                        Ok(r) => r,
                        Err(e) => return Some((Err(e), State::Done)),
                    };

                    let Some(candidate) = response.primary_candidate() else {
                        return Some((
                            Err(GeminiError::Api("Gemini response missing candidate".into())),
                            State::Done,
                        ));
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
                        let response_value = serde_json::from_str::<Value>(&tool_output)
                            .unwrap_or(Value::String(tool_output));
                        contents.push(GeminiContent::function_response(
                            call.name.clone(),
                            response_value,
                        ));
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
                    for reasoning in content.reasoning_chunks() {
                        chunk.push_reasoning(reasoning);
                    }
                    Some((Ok(chunk), State::Done))
                }
            },
        );

        ReasoningStream::new(stream)
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
    let mut system = String::new();
    let mut contents = Vec::new();
    for message in messages {
        match message.role() {
            Role::System => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(message.content());
            }
            Role::User | Role::Tool => {
                contents.push(GeminiContent::text("user", message.content()));
            }
            Role::Assistant => contents.push(GeminiContent::text("model", message.content())),
        }
    }

    let system_instruction = if system.is_empty() {
        None
    } else {
        Some(GeminiContent::text("user", system))
    };
    (system_instruction, contents)
}

fn build_generation_config(
    parameters: &Parameters,
    modalities: Option<Vec<String>>,
) -> Option<GenerationConfig> {
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
        ..Default::default()
    };
    if let Some(schema) = &parameters.response_format {
        config.response_mime_type = Some("application/json".into());
        config.response_schema = Some(schema_to_value(schema));
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
    if !parameters.include_reasoning && parameters.reasoning_budget_tokens.is_none() {
        return None;
    }
    Some(ThinkingConfig {
        include_thinking: parameters.include_reasoning
            || parameters.reasoning_budget_tokens.is_some(),
        token_budget: parameters.reasoning_budget_tokens.map(|tokens| {
            #[allow(clippy::cast_possible_wrap)]
            {
                tokens as i32
            }
        }),
    })
}

fn convert_tool_definitions(defs: Vec<ToolDefinition>) -> Vec<GeminiTool> {
    if defs.is_empty() {
        return Vec::new();
    }
    let declarations = defs
        .into_iter()
        .map(|tool| FunctionDeclaration {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: Some(schema_to_value(tool.arguments_schema())),
        })
        .collect();
    vec![GeminiTool {
        function_declarations: declarations,
    }]
}

fn schema_to_value(schema: &Schema) -> Value {
    serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(serde_json::Map::default()))
}
