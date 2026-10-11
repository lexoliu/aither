//! Asynchronous compaction into a typed handoff document.
//!
//! Compaction used to be a stop-the-world event: the run halted, the model
//! summarized its own transcript, and only then did work resume. That is the
//! worst possible moment to spend a full request, because the window is
//! already near its limit and the user is already waiting.
//!
//! Here it is forked instead. At a request boundary the window is snapshotted
//! together with the sequence number of the next item — the *fork point* —
//! and a handoff request goes out alongside the normal turns. The fork
//! reproduces the main request's prefix byte for byte, so the provider serves
//! it from the same cache the run just paid for. When the handoff lands, the
//! next request boundary applies it: the conversation is replaced by the
//! document, and every item numbered at or above the fork point is kept
//! verbatim because the document cannot possibly describe it.
//!
//! The result is a *typed* document rather than prose. Structured output is
//! tried first; a model that ignores the schema gets exactly one corrective
//! retry quoting its own failure, and then the compaction is reported as
//! failed instead of silently storing something unusable.

use std::future::Future;
use std::pin::Pin;

use aither_core::llm::model::Parameters;
use aither_core::llm::{Event, LLMRequest, Message};
use aither_core::{LanguageModel, llm::GenerateError};
use askama::Template;
use futures_lite::StreamExt as _;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sync_wrapper::SyncWrapper;

use super::window::Context;

/// Structured handoff document used to recover work after compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct HandoffDocument {
    /// What the user asked for, and how far it has got.
    pub current_task: String,
    /// Decisions, constraints and patterns that must survive.
    pub key_decisions: Vec<String>,
    /// Files read, written, or planned for modification.
    pub files_paths: Vec<String>,
    /// Errors encountered and how they were resolved.
    pub errors_fixes: Vec<String>,
    /// Remaining work, in order, starting with the next action.
    pub pending_work: Vec<String>,
    /// Where the rest of the state can be recovered from.
    pub recovery_hints: Vec<String>,
    /// Mechanical index of on-disk materials produced during the compacted
    /// conversation (`outputs/`, `artifacts/`), each annotated with the
    /// originating command's description.
    ///
    /// Built by the framework, not the model: asking the model for it is
    /// unreliable, and a missing entry directly causes work to be redone.
    #[serde(default)]
    pub file_index: Vec<String>,
}

#[derive(Template)]
#[template(path = "handoff_document.md", escape = "none")]
struct HandoffDocumentTemplate<'a> {
    handoff: &'a HandoffDocument,
}

#[derive(Template)]
#[template(path = "compaction_instruction.txt", escape = "none")]
struct CompactionInstruction<'a> {
    focus: Option<&'a str>,
}

#[derive(Template)]
#[template(path = "compaction_retry.txt", escape = "none")]
struct CompactionRetry<'a> {
    reason: &'a str,
}

impl HandoffDocument {
    /// Validates required handoff fields.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError::Invalid`] when `current_task` is empty: a
    /// handoff without it tells the next instance nothing about what it is
    /// supposed to be doing, and is worse than no handoff at all.
    pub fn validate(self) -> Result<Self, CompactionError> {
        if self.current_task.trim().is_empty() {
            return Err(CompactionError::Invalid {
                reason: "current_task must not be empty".to_string(),
            });
        }
        Ok(self)
    }

    /// Renders this handoff as Markdown.
    ///
    /// # Panics
    ///
    /// Panics if the bundled handoff template cannot be rendered, which is a
    /// build-time error in the template rather than a runtime condition.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        HandoffDocumentTemplate { handoff: self }
            .render()
            .expect("handoff document template must render")
    }
}

/// Why a compaction could not produce a usable handoff document.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompactionError {
    /// The provider failed before producing a response.
    #[error("compaction request failed: {message}")]
    Provider {
        /// The provider's error message.
        message: String,
    },

    /// The model answered, but not with a usable handoff document.
    #[error("compaction produced an invalid handoff document: {reason}")]
    Invalid {
        /// What was wrong with the document.
        reason: String,
    },

    /// There was nothing to compact.
    #[error("compaction found no conversation to summarize")]
    Empty,
}

/// Builds the mechanical file index for a handoff from conversation messages.
///
/// Scans tool results for `outputs/` URLs and `artifacts/` paths and pairs
/// each with the description of the tool call that produced it. Later
/// mentions of the same path win.
#[must_use]
pub fn build_file_index(messages: &[Message]) -> Vec<String> {
    use std::collections::HashMap;

    let mut call_description: HashMap<&str, &str> = HashMap::new();
    for message in messages {
        for call in message.tool_calls() {
            if let Some(description) = call.arguments.get("description").and_then(|v| v.as_str()) {
                call_description.insert(call.id.as_str(), description);
            }
        }
    }

    let mut index: indexmap::IndexMap<String, Option<&str>> = indexmap::IndexMap::new();
    for message in messages {
        let Message::Tool {
            content,
            tool_call_id,
        } = message
        else {
            continue;
        };
        let description = call_description.get(tool_call_id.as_str()).copied();
        for url in crate::output_urls::extract_referenced_urls(content) {
            index.insert(url, description);
        }
        for word in content.split_whitespace() {
            let word = word.trim_matches(|c: char| {
                c == '"' || c == '\'' || c == '`' || c == ',' || c == ')' || c == ']' || c == ';'
            });
            if word.starts_with("artifacts/") && word.len() > "artifacts/".len() {
                index.insert(word.to_string(), description);
            }
        }
    }

    index
        .into_iter()
        .map(|(path, description)| match description {
            Some(description) => format!("{path} — {description}"),
            None => path,
        })
        .collect()
}

/// A compaction snapshot taken at a request boundary.
#[derive(Debug, Clone)]
pub struct CompactionFork {
    /// Sequence number of the first item the handoff will *not* describe.
    fork_point: u64,
    /// The main request's messages, with the handoff instruction appended.
    messages: Vec<Message>,
    /// The conversation the handoff describes, for the mechanical file index.
    conversation: Vec<Message>,
}

impl CompactionFork {
    /// Snapshots `context` for compaction, optionally focused by `focus`.
    ///
    /// The instruction is *appended*: everything before it is exactly what
    /// the main request sent, so the provider's prefix cache still applies.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError::Empty`] when there is no conversation yet.
    ///
    /// # Panics
    ///
    /// Panics if the compiled-in instruction template fails to render, which
    /// would mean the binary was built against a broken template.
    pub fn new(context: &Context, focus: Option<&str>) -> Result<Self, CompactionError> {
        let conversation = context.conversation_messages();
        if conversation.is_empty() {
            return Err(CompactionError::Empty);
        }
        let focus = focus.map(str::trim).filter(|focus| !focus.is_empty());
        let instruction = CompactionInstruction { focus }
            .render()
            .expect("compaction instruction template must render");
        let mut messages = context.build_messages();
        messages.push(Message::user(instruction));
        Ok(Self {
            fork_point: context.fork_point(),
            messages,
            conversation,
        })
    }

    /// The sequence number items must reach to survive this compaction.
    #[must_use]
    pub const fn fork_point(&self) -> u64 {
        self.fork_point
    }

    /// Runs the compaction against `model`.
    ///
    /// Structured output is attempted first. A schema mismatch — the common
    /// failure for models without native structured output — earns one
    /// corrective retry that quotes the parse failure back to the model.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError`] when the provider fails, or when the
    /// retry also fails to produce a valid document.
    ///
    /// # Panics
    ///
    /// Panics if the compiled-in retry template fails to render, which would
    /// mean the binary was built against a broken template.
    pub async fn run<M: LanguageModel>(
        self,
        model: M,
        parameters: Parameters,
    ) -> Result<HandoffDocument, CompactionError> {
        let Self {
            messages,
            conversation,
            ..
        } = self;
        let request = LLMRequest::new(messages.clone()).with_parameters(parameters.clone());
        let reason = match model.generate::<HandoffDocument>(request).await {
            Ok(document) => match document.validate() {
                Ok(mut document) => {
                    document.file_index = build_file_index(&conversation);
                    return Ok(document);
                }
                Err(error) => error.to_string(),
            },
            Err(GenerateError::Provider(error)) => {
                return Err(CompactionError::Provider {
                    message: error.to_string(),
                });
            }
            Err(error @ GenerateError::Parse { .. }) => error.to_string(),
        };

        tracing::warn!(reason = %reason, "retrying compaction with a corrective prompt");
        let retry = CompactionRetry {
            reason: reason.as_str(),
        }
        .render()
        .expect("compaction retry template must render");
        let mut retry_messages = messages;
        retry_messages.push(Message::user(retry));
        let request = LLMRequest::new(retry_messages).with_parameters(parameters);
        let response = collect_text(&model, request).await?;
        let mut document = parse_handoff(&response)?.validate()?;
        document.file_index = build_file_index(&conversation);
        Ok(document)
    }
}

/// A compaction that has landed but not yet been folded into the window.
///
/// The handoff is applied at the next request boundary rather than the moment
/// it arrives: mid-turn the conversation holds tool calls whose results have
/// not been appended yet, and replacing it there would strand them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedCompaction {
    /// Sequence number items must reach to survive the handoff.
    pub fork_point: u64,
    /// The document that replaces everything older than the fork point.
    pub handoff: HandoffDocument,
}

/// The owned, `'static` future a forked compaction runs as.
type CompactionFuture =
    Pin<Box<dyn Future<Output = Result<HandoffDocument, CompactionError>> + Send>>;

/// A compaction that is in flight while the run continues.
///
/// The future is owned and `'static`, which is what lets it be carried across
/// turn boundaries and polled next to the model stream without an executor:
/// the agent races the two with [`futures_lite::future::or`] and keeps
/// whichever finishes.
pub struct PendingCompaction {
    fork_point: u64,
    /// The compaction in flight.
    ///
    /// A future built from provider futures is `Send` but says nothing about
    /// `Sync`, and the agent that owns this has to stay `Sync` to live inside
    /// a tool. [`SyncWrapper`] grants exactly that: `Sync` in exchange for
    /// `&mut`-only access, which is all polling ever needs.
    future: SyncWrapper<CompactionFuture>,
}

impl std::fmt::Debug for PendingCompaction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PendingCompaction")
            .field("fork_point", &self.fork_point)
            .finish_non_exhaustive()
    }
}

impl PendingCompaction {
    /// Starts `fork` against an owned `model`.
    #[must_use]
    pub fn start<M: LanguageModel + 'static>(
        fork: CompactionFork,
        model: M,
        parameters: Parameters,
    ) -> Self {
        let fork_point = fork.fork_point();
        Self {
            fork_point,
            future: SyncWrapper::new(Box::pin(fork.run(model, parameters))),
        }
    }

    /// The sequence number items must reach to survive this compaction.
    #[must_use]
    pub const fn fork_point(&self) -> u64 {
        self.fork_point
    }

    /// Awaits the handoff document.
    pub fn wait(&mut self) -> impl Future<Output = Result<HandoffDocument, CompactionError>> + '_ {
        self.future.get_mut()
    }
}

/// Parses a handoff document from a raw model response.
///
/// Models wrap JSON in prose and code fences no matter what the prompt says,
/// so the outermost braces are located rather than trusting the whole body.
fn parse_handoff(response: &str) -> Result<HandoffDocument, CompactionError> {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Err(CompactionError::Invalid {
            reason: "the model returned an empty response".to_string(),
        });
    }
    let json = match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if end > start => &trimmed[start..=end],
        _ => trimmed,
    };
    serde_json::from_str(json).map_err(|error| CompactionError::Invalid {
        reason: error.to_string(),
    })
}

/// Collects the text of a non-streaming completion.
async fn collect_text<M: LanguageModel>(
    model: &M,
    request: LLMRequest,
) -> Result<String, CompactionError> {
    let stream = model.respond(request);
    futures_lite::pin!(stream);
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(Event::Text(chunk)) => text.push_str(&chunk),
            Ok(_) => {}
            Err(error) => {
                return Err(CompactionError::Provider {
                    message: error.to_string(),
                });
            }
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::ToolCall;

    #[test]
    fn build_file_index_pairs_outputs_with_call_descriptions() {
        let messages = vec![
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_1",
                    "terminal",
                    serde_json::json!({
                        "description": "Fetch WebGPU status pages",
                        "script": "webfetch https://caniuse.com"
                    }),
                )],
            ),
            Message::tool(
                "call_1",
                "Output saved to outputs/amber-oak-swift-river.txt (100 lines, 5000 bytes)",
            ),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_2",
                    "terminal",
                    serde_json::json!({"description": "Write deck", "script": "tee artifacts/slides.md"}),
                )],
            ),
            Message::tool("call_2", "wrote artifacts/slides.md"),
        ];

        let index = build_file_index(&messages);
        assert_eq!(index.len(), 2);
        assert!(
            index[0].contains("outputs/amber-oak-swift-river.txt")
                && index[0].contains("Fetch WebGPU status pages"),
            "index entry should pair path with description: {index:?}"
        );
        assert!(index[1].contains("artifacts/slides.md") && index[1].contains("Write deck"));
    }

    #[test]
    fn build_file_index_keeps_latest_description_per_path() {
        let messages = vec![
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_1",
                    "terminal",
                    serde_json::json!({"description": "first pass"}),
                )],
            ),
            Message::tool("call_1", "see artifacts/report.md"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_2",
                    "terminal",
                    serde_json::json!({"description": "final revision"}),
                )],
            ),
            Message::tool("call_2", "updated artifacts/report.md"),
        ];

        let index = build_file_index(&messages);
        assert_eq!(index.len(), 1);
        assert!(index[0].contains("final revision"));
    }

    #[test]
    fn handoff_markdown_renders_file_index_section() {
        let handoff = HandoffDocument {
            current_task: "task".to_string(),
            file_index: vec!["outputs/a-b-c-d.txt — research notes".to_string()],
            ..HandoffDocument::default()
        };
        let markdown = handoff.render_markdown();
        assert!(markdown.contains("Materials Already On Disk"));
        assert!(markdown.contains("outputs/a-b-c-d.txt — research notes"));
        assert!(markdown.contains("Read the file instead of redoing the work"));
    }

    #[test]
    fn a_handoff_without_a_current_task_is_rejected() {
        let error = HandoffDocument::default()
            .validate()
            .expect_err("empty current_task must not validate");
        assert!(matches!(error, CompactionError::Invalid { .. }));
    }

    #[test]
    fn handoff_parsing_tolerates_fences_and_commentary() {
        let response = "Sure! Here you go:\n```json\n{\"current_task\": \"ship it\", \
            \"key_decisions\": [], \"files_paths\": [], \"errors_fixes\": [], \
            \"pending_work\": [], \"recovery_hints\": []}\n```\nHope that helps.";
        let document = parse_handoff(response).expect("handoff parses");
        assert_eq!(document.current_task, "ship it");
    }

    #[test]
    fn empty_responses_are_reported_as_invalid() {
        let error = parse_handoff("   ").expect_err("empty response must not parse");
        assert!(matches!(error, CompactionError::Invalid { .. }));
    }

    #[test]
    fn forking_an_empty_context_is_an_error() {
        let context = Context::new();
        assert!(matches!(
            CompactionFork::new(&context, None),
            Err(CompactionError::Empty)
        ));
    }

    #[test]
    fn the_fork_reproduces_the_main_prefix_and_appends_the_instruction() {
        let mut context = Context::new();
        context.insert_system_named("base_system", "be useful");
        context.push(Message::user("do the thing"));

        let main = context.build_messages();
        let fork = CompactionFork::new(&context, Some("the failing test")).expect("fork");

        assert_eq!(fork.fork_point(), context.fork_point());
        assert_eq!(fork.messages.len(), main.len() + 1);
        for (index, message) in main.iter().enumerate() {
            assert_eq!(
                message.content(),
                fork.messages[index].content(),
                "fork perturbed prefix message {index}"
            );
        }
        let instruction = fork.messages.last().expect("instruction").content();
        assert!(instruction.contains("Focus the handoff on: the failing test"));
    }
}
