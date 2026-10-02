//! `LanguageModel` implementation driving the native bridge.
//!
//! Only compiled when the bridge was built (`aither_apple_native`), i.e. on
//! macOS/iOS targets with a Foundation Models SDK.

use core::future::Future;

use aither_core::llm::model::{Ability, Parameters, Profile};
use aither_core::llm::tool::ToolDefinition;
use aither_core::llm::{
    Event, GenerateError, LLMRequest, LLMRequestWithTools, LanguageModel, collect_text,
};
use futures_lite::Stream;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;

use crate::Availability;
use crate::driver::{Driver, PlannedRequest};
use crate::error::AppleError;
use crate::model::{AppleIntelligence, DEFAULT_MODEL_ID};
use crate::params::{self, Capabilities, ToolPlan};
use crate::schema;
use crate::transcript::{self, PreparedEntry};
use crate::wire::{WireEntry, WirePrompt, WireRequest, WireSchemaRoot, WireTool, WireToolCall};

/// Which session mode the request runs in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Plain text generation (or guided via `response_format`).
    Plain,
    /// `respond`: tools are declared but suspended at the native boundary.
    External,
    /// `respond_with_tools`: native tool calls execute against `Tools`.
    Internal,
}

fn capabilities() -> Capabilities {
    Capabilities::from_bits(crate::ffi::native_capabilities())
}

/// Builds the wire request: validate → convert transcript → resolve
/// attachments → serialize.
async fn plan_request(
    request: LLMRequest,
    mode: Mode,
    guided: Option<WireSchemaRoot>,
) -> Result<PlannedRequest, AppleError> {
    plan_with_capabilities(request, mode, guided, capabilities()).await
}

async fn plan_with_capabilities(
    request: LLMRequest,
    mode: Mode,
    guided: Option<WireSchemaRoot>,
    caps: Capabilities,
) -> Result<PlannedRequest, AppleError> {
    let (messages, parameters, tool_definitions) = request.into_parts();
    let mut options = params::to_options(&parameters, caps)?;

    let prepared = transcript::prepare(&messages)?;

    // Guided generation is orthogonal to tool dispatch: `mode` picks the
    // response stream shape, `tools_are_external` picks suspension behavior.
    let schema = match guided {
        Some(root) => Some(root),
        None => response_format_schema(&parameters)?,
    };
    if parameters.structured_outputs && schema.is_none() {
        return Err(AppleError::UnsupportedParameter {
            name: "structured_outputs",
            reason: "requires a response schema".into(),
        });
    }

    let tool_plan = params::tool_plan(&parameters, tool_definitions, caps)?;
    let (tools_are_external, tools) = match tool_plan {
        ToolPlan::NoTools => (mode == Mode::External, Vec::new()),
        ToolPlan::Allowed(defs) => (mode == Mode::External, defs),
        ToolPlan::Required(defs) => {
            options.tool_calling_mode = Some("required");
            (mode == Mode::External, defs)
        }
    };

    if mode == Mode::Plain && !tools.is_empty() {
        return Err(AppleError::UnsupportedRequest(
            "generate cannot suspend for caller-managed tools; use respond with response_format"
                .into(),
        ));
    }
    if schema.is_some() && !caps.supports(Capabilities::GUIDED) {
        return Err(AppleError::UnsupportedCapability(
            "model does not support guided generation".into(),
        ));
    }
    if !tools.is_empty() && !caps.supports(Capabilities::TOOLS) {
        return Err(AppleError::UnsupportedCapability(
            "model does not support tool calling".into(),
        ));
    }

    let wire_tools: Vec<WireTool> = tools.iter().map(tool_to_wire).collect::<Result<_, _>>()?;

    let (history, attachments_to_resolve, prompt_indexes) =
        prepare_attachments(prepared.history, prepared.prompt_attachments);
    if !attachments_to_resolve.is_empty() && !caps.supports(Capabilities::VISION) {
        return Err(AppleError::UnsupportedCapability(
            "model does not support image prompting".into(),
        ));
    }
    // Four concurrent fetches bound memory/network pressure and preserve indexes.
    let images = futures_util::TryStreamExt::try_collect(futures_util::StreamExt::buffered(
        futures_util::stream::iter(
            attachments_to_resolve
                .iter()
                .map(crate::attachments::resolve_image),
        ),
        4,
    ))
    .await?;

    let wire = WireRequest {
        instructions: prepared.instructions,
        history,
        prompt: WirePrompt {
            text: prepared.prompt_text,
            image_indexes: prompt_indexes,
        },
        mode: if schema.is_some() { "guided" } else { "text" },
        tools_are_external,
        options,
        tools: wire_tools,
        schema,
        include_reasoning: parameters.include_reasoning,
    };

    let json = serde_json::to_vec(&wire).map_err(|e| AppleError::Native {
        code: "serialize".into(),
        message: e.to_string(),
    })?;

    Ok(PlannedRequest {
        json,
        images,
        #[cfg(aither_scripted)]
        script: None,
    })
}

fn prepare_attachments(
    entries: Vec<PreparedEntry>,
    prompt_attachments: Vec<aither_core::llm::Attachment>,
) -> (
    Vec<WireEntry>,
    Vec<aither_core::llm::Attachment>,
    Vec<usize>,
) {
    // Resolve attachments: history entries first, then the prompt.
    let mut attachments_to_resolve = Vec::new();
    let mut history = Vec::with_capacity(entries.len());
    for entry in entries {
        history.push(match entry {
            PreparedEntry::Prompt { text, attachments } => {
                let mut indexes = Vec::with_capacity(attachments.len());
                for attachment in attachments {
                    indexes.push(attachments_to_resolve.len());
                    attachments_to_resolve.push(attachment);
                }
                WireEntry::Prompt {
                    text,
                    image_indexes: indexes,
                }
            }
            PreparedEntry::Response { text } => WireEntry::Response { text },
            PreparedEntry::ToolCalls(calls) => WireEntry::ToolCalls {
                calls: calls
                    .into_iter()
                    .map(|c| WireToolCall {
                        id: c.id,
                        name: c.name,
                        arguments: c.arguments,
                    })
                    .collect(),
            },
            PreparedEntry::ToolOutput { id, name, text } => {
                WireEntry::ToolOutput { id, name, text }
            }
        });
    }

    let mut prompt_indexes = Vec::with_capacity(prompt_attachments.len());
    for attachment in prompt_attachments {
        prompt_indexes.push(attachments_to_resolve.len());
        attachments_to_resolve.push(attachment);
    }

    (history, attachments_to_resolve, prompt_indexes)
}

fn response_format_schema(parameters: &Parameters) -> Result<Option<WireSchemaRoot>, AppleError> {
    parameters
        .response_format
        .as_ref()
        .map(|schema| schema::to_wire_schema(schema, "response"))
        .transpose()
}

fn tool_to_wire(definition: &ToolDefinition) -> Result<WireTool, AppleError> {
    Ok(WireTool {
        name: definition.name().to_string(),
        description: definition.description().to_string(),
        schema: schema::value_to_wire_schema(&definition.arguments_openai_schema(), "arguments")?,
    })
}

impl LanguageModel for AppleIntelligence {
    type Error = AppleError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        if let Availability::Unavailable(reason) = Self::availability() {
            return Driver::error(reason.into());
        }
        Driver::external(async move { plan_request(request, Mode::External, None).await })
    }

    fn respond_with_tools(
        &self,
        request: LLMRequestWithTools<'_>,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        if let Availability::Unavailable(reason) = Self::availability() {
            return Driver::error(reason.into());
        }
        let (inner, tools) = request.into_parts();
        Driver::internal(
            async move { plan_request(inner, Mode::Internal, None).await },
            tools,
        )
    }

    async fn generate<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        request: LLMRequest,
    ) -> Result<T, GenerateError<Self::Error>> {
        if let Availability::Unavailable(reason) = Self::availability() {
            return Err(GenerateError::Provider(reason.into()));
        }
        let schema =
            schema::to_wire_schema(&schema_for!(T), "response").map_err(GenerateError::Provider)?;
        let stream =
            Driver::external(async move { plan_request(request, Mode::Plain, Some(schema)).await });
        let text = collect_text(stream)
            .await
            .map_err(GenerateError::Provider)?;
        serde_json::from_str::<T>(&text).map_err(|source| GenerateError::Parse {
            source,
            response: text.chars().take(500).collect(),
        })
    }

    fn profile(&self) -> impl Future<Output = Profile> + Send {
        let caps = capabilities();
        let mut abilities = Vec::new();
        if caps.supports(Capabilities::TOOLS) {
            abilities.push(Ability::ToolUse);
        }
        if caps.supports(Capabilities::VISION) {
            abilities.push(Ability::Vision);
        }
        if caps.supports(Capabilities::REASONING) {
            abilities.push(Ability::Reasoning);
        }
        let context_length = crate::ffi::native_context_size()
            .expect("FoundationModels must expose a valid context size on supported OS versions");
        core::future::ready(
            Profile::new(
                "apple-on-device",
                "apple",
                DEFAULT_MODEL_ID,
                "Apple Intelligence on-device language model",
                context_length,
            )
            .with_abilities(abilities),
        )
    }
}

use crate::model::AppleIntelligenceProvider;
use aither_core::llm::provider::{LanguageModelProvider, Profile as ProviderProfile};

impl LanguageModelProvider for AppleIntelligenceProvider {
    type Model = AppleIntelligence;
    type Error = AppleError;

    async fn list_models(&self) -> Result<Vec<Profile>, Self::Error> {
        match AppleIntelligence::availability() {
            Availability::Available => Ok(vec![AppleIntelligence.profile().await]),
            Availability::Unavailable(reason) => Err(reason.into()),
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send {
        if name != DEFAULT_MODEL_ID {
            return core::future::ready(Err(AppleError::UnknownModel(name.to_string())));
        }
        core::future::ready(AppleIntelligence::new())
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new(
            "apple",
            "On-device Apple Intelligence via the Foundation Models framework",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::{Attachment, Message};

    #[tokio::test]
    async fn capabilities_reject_before_io_or_native_start() {
        let missing = Attachment::new(
            url::Url::parse("file:///definitely-missing-apple-test.png").unwrap(),
            mime::IMAGE_PNG,
        );
        let request = LLMRequest::new([Message::user("image").with_attachment(missing)]);
        assert!(matches!(
            plan_with_capabilities(request, Mode::External, None, Capabilities::default()).await,
            Err(AppleError::UnsupportedCapability(_))
        ));
        let schema =
            schema::value_to_wire_schema(&serde_json::json!({"type":"string"}), "Root").unwrap();
        assert!(matches!(
            plan_with_capabilities(
                LLMRequest::new([Message::user("guided")]),
                Mode::Plain,
                Some(schema),
                Capabilities::default()
            )
            .await,
            Err(AppleError::UnsupportedCapability(_))
        ));
    }
}
