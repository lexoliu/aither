use aither_core::{
    LanguageModel,
    llm::{
        Message, Role,
        model::{Ability, Parameters, Profile},
        tool::{ToolDefinition, Tools},
    },
};
use async_stream::try_stream;
use futures_core::Stream;
use futures_lite::{StreamExt, pin};
use schemars::Schema;
use serde_json::Value;

use crate::{
    client::call_generate,
    config::GeminiBackend,
    error::GeminiError,
    types::{
        FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, GeminiContent, GeminiTool,
        GenerateContentRequest, GenerationConfig, ToolConfig,
    },
};

const MAX_TOOL_ITERATIONS: usize = 8;

impl LanguageModel for GeminiBackend {
    type Error = GeminiError;

    fn respond(
        &self,
        messages: &[Message],
        tools: &mut Tools,
        parameters: &Parameters,
    ) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        let cfg = self.config();
        let (system_instruction, mut contents) = messages_to_gemini(messages);
        let tool_defs = tools.definitions();
        let tool_config = build_tool_config(parameters, !tool_defs.is_empty());
        let tools_payload = convert_tool_definitions(tool_defs);
        let generation_config = build_generation_config(parameters, None);

        try_stream! {
            let mut iterations = 0usize;
            loop {
                iterations += 1;
                if iterations > MAX_TOOL_ITERATIONS {
                    Err(GeminiError::Api("exceeded Gemini tool calling iteration limit".into()))?;
                }
                let request = GenerateContentRequest {
                    system_instruction: system_instruction.clone(),
                    contents: contents.clone(),
                    generation_config: generation_config.clone(),
                    tools: tools_payload.clone(),
                    tool_config: tool_config.clone(),
                    safety_settings: Vec::new(),
                };
                let response = call_generate(cfg.clone(), &cfg.text_model, request).await?;
                if let Some(candidate) = response.primary_candidate() {
                    if let Some(content) = &candidate.content {
                        if let Some(call) = content.first_function_call() {
                            contents.push(content.clone());
                            let args_json =
                                serde_json::to_string(&call.args).unwrap_or_else(|_| "{}".to_string());
                            let tool_output = tools
                                .call(&call.name, args_json)
                                .await
                                .map_err(|err| GeminiError::Api(err.to_string()))?;
                            let response_value = serde_json::from_str::<Value>(&tool_output)
                                .unwrap_or_else(|_| Value::String(tool_output));
                            contents.push(GeminiContent::function_response(
                                call.name.clone(),
                                response_value,
                            ));
                            continue;
                        }

                        for text in content.text_chunks() {
                            if !text.is_empty() {
                                yield text;
                            }
                        }
                        contents.push(content.clone());
                        break;
                    }
                }

                // No candidate -> treat as empty response.
                break;
            }
        }
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<String, Self::Error>> + Send {
        let messages = [
            Message::system("Continue the supplied text."),
            Message::user(prefix),
        ];
        let mut tools = Tools::new();
        let params = Parameters::default();
        let model = self.clone();
        try_stream! {
            let stream = model.respond(&messages, &mut tools, &params);
            pin!(stream);
            while let Some(chunk) = stream.next().await {
                yield chunk?;
            }
        }
    }

    fn profile(&self) -> impl core::future::Future<Output = Profile> + Send {
        let cfg = self.config();
        async move {
            let display = cfg.text_model.trim_start_matches("models/").to_string();
            Profile::new(
                display.clone(),
                "google",
                display,
                "Gemini Developer API model",
                1_000_000,
            )
            .with_abilities([Ability::ToolUse, Ability::Vision, Ability::Audio])
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
            Role::User => contents.push(GeminiContent::text("user", message.content())),
            Role::Assistant => contents.push(GeminiContent::text("model", message.content())),
            Role::Tool => contents.push(GeminiContent::text("user", message.content())),
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
    let mut config = GenerationConfig::default();
    config.temperature = parameters.temperature;
    config.top_p = parameters.top_p;
    config.top_k = parameters.top_k;
    config.max_output_tokens = parameters.max_tokens.map(|value| value as i32);
    config.stop_sequences = parameters.stop.clone();
    config.response_modalities = modalities;
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
    serde_json::to_value(schema).unwrap_or_else(|_| Value::Object(Default::default()))
}
