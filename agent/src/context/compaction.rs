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
//! The result is a *typed* document rather than prose, and the type is the
//! embedder's: a [`HandoffSchema`] gives the schema the model fills, the
//! instruction that asks for it, and the text it renders back into the
//! conversation. aither reads from it only what restoration needs, the
//! [`Restoration`]. [`HandoffDocument`] is the schema aither ships.
//!
//! Structured output is tried first; a model that ignores the schema gets
//! exactly one corrective retry quoting its own failure, and then the
//! compaction is reported as failed instead of silently storing something
//! unusable.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use aither_core::llm::{Classify, Event, Failure, LLMRequest, Message, Role};
use aither_core::{LanguageModel, llm::GenerateError};
use askama::Template;
use futures_lite::StreamExt as _;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sync_wrapper::SyncWrapper;

/// A handoff document type a compaction asks the model to fill.
///
/// The type's JSON schema is the structured output the compaction requests,
/// so its fields are what the model writes. The rest is what the embedder
/// decides about its handoff: the instruction that asks for it, the
/// corrective prompt after an unusable answer, what makes a document
/// unusable, and the text the conversation reads in place of everything the
/// handoff replaced.
///
/// aither needs only the [`Restoration`] from it — the file ranges and
/// skills to put back in view — and builds the index of on-disk materials
/// itself, passing it to [`render`](Self::render).
pub trait HandoffSchema: JsonSchema + DeserializeOwned + Send + 'static {
    /// The instruction appended to the conversation to ask for this
    /// document.
    fn instruction(prompt: &HandoffPrompt<'_>) -> String;

    /// The corrective prompt sent once after a response that did not parse
    /// or validate; `reason` says what was wrong with it.
    fn retry(reason: &str) -> String;

    /// Checks a parsed document before it is accepted.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidHandoff`] when the document is unusable; its reason
    /// is quoted to the model in the retry.
    fn validate(&self) -> Result<(), InvalidHandoff> {
        Ok(())
    }

    /// Renders the document as the text the conversation reads from now on.
    ///
    /// `file_index` is the mechanical index of on-disk materials the
    /// compacted conversation produced (`path — description` entries); the
    /// model does not write it, because a missing entry means redone work.
    fn render(&self, file_index: &[String]) -> String;

    /// Takes out what restoration puts back in view.
    fn into_restoration(self) -> Restoration;
}

/// What the instruction of a compaction is rendered from.
#[derive(Debug, Clone, Copy)]
pub struct HandoffPrompt<'a> {
    /// What the handoff should focus on, when a caller said; trimmed and
    /// never empty.
    pub focus: Option<&'a str>,
    /// The session's skills the handoff may name for restoration.
    pub skills: &'a [SkillCatalogEntry],
}

/// What a handoff asks to be put back in view after compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Restoration {
    /// File ranges restored verbatim, in order.
    pub context_files: Vec<ContextFile>,
    /// Names of catalog skills whose instructions are restored; a name not
    /// in the catalog is reported instead.
    pub skills: Vec<String>,
}

/// Why a parsed handoff document is unusable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct InvalidHandoff {
    /// What is wrong with the document, as the retry quotes it.
    pub reason: String,
}

impl InvalidHandoff {
    /// An invalid handoff for `reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl From<InvalidHandoff> for CompactionError {
    fn from(error: InvalidHandoff) -> Self {
        Self::Invalid {
            reason: error.reason,
        }
    }
}

/// The handoff schema aither ships: the task, decisions, errors, pending
/// work and recovery hints, plus the files and skills to restore.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct HandoffDocument {
    /// What the user asked for, and how far it has got.
    pub current_task: String,
    /// Decisions, constraints and patterns that must survive.
    pub key_decisions: Vec<String>,
    /// File ranges the next turns need in view, restored verbatim on apply.
    #[serde(default)]
    pub context_files: Vec<ContextFile>,
    /// Errors encountered and how they were resolved.
    pub errors_fixes: Vec<String>,
    /// Remaining work, in order, starting with the next action.
    pub pending_work: Vec<String>,
    /// Where the rest of the state can be recovered from.
    pub recovery_hints: Vec<String>,
    /// Session skills whose instructions the next turns need in view,
    /// chosen from the catalog the instruction lists.
    #[serde(default)]
    pub skills: Vec<String>,
}

/// A file range the next turns need in view, named by the handoff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ContextFile {
    /// Path to the file, as it appears on disk.
    pub path: String,
    /// The 1-based inclusive line range to restore; absent restores the
    /// whole file.
    pub lines: Option<LineRange>,
    /// Why the next turns need this file in view.
    pub why: String,
}

/// A 1-based inclusive line range inside a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct LineRange {
    /// First line to restore.
    pub start: u32,
    /// Last line to restore.
    pub end: u32,
}

/// A skill the handoff may name for restoration.
///
/// The session's skill catalog is offered to the summarising model so it can
/// name skills by their registry name; on apply, only names found in the
/// catalog are restored, and the rest are reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCatalogEntry {
    /// The name the handoff names.
    pub name: String,
    /// What the skill is for, shown in the compaction prompt.
    pub description: String,
    /// The text restored into `<restored_context>`.
    pub instructions: String,
}

#[derive(Template)]
#[template(path = "handoff_document.md", escape = "none")]
struct HandoffDocumentTemplate<'a> {
    handoff: &'a HandoffDocument,
    file_index: &'a [String],
}

#[derive(Template)]
#[template(path = "compaction_instruction.txt", escape = "none")]
struct CompactionInstruction<'a> {
    focus: Option<&'a str>,
    skills: &'a [SkillCatalogEntry],
}

#[derive(Template)]
#[template(path = "compaction_retry.txt", escape = "none")]
struct CompactionRetry<'a> {
    reason: &'a str,
}

impl HandoffSchema for HandoffDocument {
    /// Asks for every field, offering the skill catalog by name.
    ///
    /// # Panics
    ///
    /// Panics if the bundled instruction template cannot be rendered, which
    /// is a build-time error in the template rather than a runtime condition.
    fn instruction(prompt: &HandoffPrompt<'_>) -> String {
        CompactionInstruction {
            focus: prompt.focus,
            skills: prompt.skills,
        }
        .render()
        .expect("compaction instruction template must render")
    }

    /// Quotes `reason` and restates the keys.
    ///
    /// # Panics
    ///
    /// Panics if the bundled retry template cannot be rendered.
    fn retry(reason: &str) -> String {
        CompactionRetry { reason }
            .render()
            .expect("compaction retry template must render")
    }

    /// Rejects an empty `current_task`: a handoff without it tells the next
    /// instance nothing about what it is supposed to be doing, and is worse
    /// than no handoff at all.
    fn validate(&self) -> Result<(), InvalidHandoff> {
        if self.current_task.trim().is_empty() {
            return Err(InvalidHandoff::new("current_task must not be empty"));
        }
        Ok(())
    }

    /// Renders the document as Markdown, one section per field.
    ///
    /// # Panics
    ///
    /// Panics if the bundled handoff template cannot be rendered.
    fn render(&self, file_index: &[String]) -> String {
        HandoffDocumentTemplate {
            handoff: self,
            file_index,
        }
        .render()
        .expect("handoff document template must render")
    }

    fn into_restoration(self) -> Restoration {
        Restoration {
            context_files: self.context_files,
            skills: self.skills,
        }
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
        /// How the provider error classifies.
        failure: Failure,
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

/// Joins an index entry's path to its description.
const INDEX_SEPARATOR: &str = " — ";

/// The handoff a [`Context`](crate::context::Context) carries between
/// compactions: the text the conversation reads in place of what it
/// replaced, and the file index the next compaction extends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Handoff {
    /// The rendered handoff, as the conversation reads it.
    text: String,
    /// The mechanical index of on-disk materials, carried into the next
    /// compaction so a file indexed once stays indexed.
    file_index: Vec<String>,
}

impl Handoff {
    /// A handoff of plain `text`, with no file index.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            file_index: Vec::new(),
        }
    }

    /// A handoff of `text` that carries `file_index`.
    #[must_use]
    pub const fn with_file_index(text: String, file_index: Vec<String>) -> Self {
        Self { text, file_index }
    }

    /// The handoff text, as the conversation reads it.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The file index entries this handoff carries.
    #[must_use]
    pub fn file_index(&self) -> &[String] {
        &self.file_index
    }
}

/// Builds the mechanical file index for a handoff from conversation messages.
///
/// Scans tool results for `outputs/` URLs and `artifacts/` paths and pairs
/// each with the description of the tool call that produced it. `previous` is
/// the file index of the handoff being replaced: its entries are kept, so a
/// file indexed by an earlier compaction stays indexed. Later mentions of the
/// same path win. `outputs_dir` is the session's outputs directory, whose
/// absolute paths saved-output notices print.
#[must_use]
pub fn build_file_index(
    previous: &[String],
    messages: &[Message],
    outputs_dir: Option<&std::path::Path>,
) -> Vec<String> {
    use std::collections::HashMap;

    let mut call_description: HashMap<&str, &str> = HashMap::new();
    for message in messages {
        for call in message.tool_calls() {
            if let Some(description) = call.arguments.get("description").and_then(|v| v.as_str()) {
                call_description.insert(call.id.as_str(), description);
            }
        }
    }

    let mut index: indexmap::IndexMap<String, Option<String>> = indexmap::IndexMap::new();
    for entry in previous {
        match entry.split_once(INDEX_SEPARATOR) {
            Some((path, description)) => {
                index.insert(path.to_string(), Some(description.to_string()));
            }
            None => {
                index.insert(entry.clone(), None);
            }
        }
    }
    for message in messages {
        let Message::Tool {
            content,
            tool_call_id,
        } = message
        else {
            continue;
        };
        let description = call_description
            .get(tool_call_id.as_str())
            .map(|description| (*description).to_string());
        for url in crate::output_urls::extract_referenced_urls(content, outputs_dir) {
            index.insert(url, description.clone());
        }
        for word in content.split_whitespace() {
            let word = word.trim_matches(|c: char| {
                c == '"' || c == '\'' || c == '`' || c == ',' || c == ')' || c == ']' || c == ';'
            });
            if word.starts_with("artifacts/") && word.len() > "artifacts/".len() {
                index.insert(word.to_string(), description.clone());
            }
        }
    }

    index
        .into_iter()
        .map(|(path, description)| match description {
            Some(description) => format!("{path}{INDEX_SEPARATOR}{description}"),
            None => path,
        })
        .collect()
}

/// A compaction snapshot taken at a request boundary, asking for a `D`.
pub struct CompactionFork<D> {
    /// Sequence number of the first item the handoff will *not* describe.
    fork_point: u64,
    /// The file index of the handoff this compaction replaces.
    previous_file_index: Vec<String>,
    /// The session's outputs directory, for the file index.
    outputs_dir: Option<std::path::PathBuf>,
    /// The main request, source blocks, tools and cache parameters included,
    /// with the handoff instruction appended to its messages. Replaying it is
    /// what lets the provider serve the compaction from the same prefix
    /// cache the run just paid for.
    request: LLMRequest,
    /// The schema the model fills; no `D` is held.
    schema: PhantomData<fn() -> D>,
}

impl<D> std::fmt::Debug for CompactionFork<D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompactionFork")
            .field("fork_point", &self.fork_point)
            .field("schema", &std::any::type_name::<D>())
            .finish_non_exhaustive()
    }
}

impl<D: HandoffSchema> CompactionFork<D> {
    /// Forks `request` — the exact request the run is about to send — into a
    /// compaction request for a `D`, optionally focused by `focus`.
    /// `previous_file_index` is the file index of the handoff the result
    /// will replace.
    ///
    /// The instruction [`D::instruction`](HandoffSchema::instruction)
    /// renders is *appended* and the tool definitions and parameters are
    /// reused untouched: everything up to the instruction is byte-for-byte
    /// what the main request sent, so the provider's prefix cache still
    /// applies.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError::Empty`] when the request carries no
    /// conversation to summarize.
    pub fn new(
        request: &LLMRequest,
        fork_point: u64,
        previous_file_index: &[String],
        focus: Option<&str>,
        skills: &[SkillCatalogEntry],
    ) -> Result<Self, CompactionError> {
        if request
            .messages()
            .iter()
            .all(|message| message.role() == Role::System)
        {
            return Err(CompactionError::Empty);
        }
        let focus = focus.map(str::trim).filter(|focus| !focus.is_empty());
        let instruction = D::instruction(&HandoffPrompt { focus, skills });
        let mut messages = request.messages().to_vec();
        messages.push(Message::user(instruction));
        Ok(Self {
            fork_point,
            previous_file_index: previous_file_index.to_vec(),
            outputs_dir: None,
            request: LLMRequest::new(messages)
                .with_parameters(request.parameters().clone())
                .with_tool_definitions(request.tool_definitions().to_vec()),
            schema: PhantomData,
        })
    }

    /// Sets the session's outputs directory, so the file index also finds
    /// saved outputs the conversation names by absolute path.
    #[must_use]
    pub fn with_outputs_dir(mut self, outputs_dir: Option<std::path::PathBuf>) -> Self {
        self.outputs_dir = outputs_dir;
        self
    }

    /// The sequence number items must reach to survive this compaction.
    #[must_use]
    pub const fn fork_point(&self) -> u64 {
        self.fork_point
    }

    /// The request the fork replays, instruction included.
    #[must_use]
    pub const fn request(&self) -> &LLMRequest {
        &self.request
    }

    /// Runs the compaction against `model`.
    ///
    /// Structured output is attempted first. A schema mismatch — the common
    /// failure for models without native structured output — or a document
    /// [`D::validate`](HandoffSchema::validate) rejects earns one corrective
    /// retry, [`D::retry`](HandoffSchema::retry), that quotes the failure
    /// back to the model. The accepted document is rendered with the file
    /// index and split into the handoff and its restoration.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError`] when the provider fails, or when the
    /// retry also fails to produce a valid document.
    pub async fn run<M: LanguageModel>(
        self,
        model: M,
    ) -> Result<CompletedCompaction, CompactionError> {
        let Self {
            fork_point,
            request,
            previous_file_index,
            outputs_dir,
            ..
        } = self;
        let file_index = build_file_index(
            &previous_file_index,
            request.messages(),
            outputs_dir.as_deref(),
        );
        let reason = match model.generate::<D>(request.clone()).await {
            Ok(document) => match document.validate() {
                Ok(()) => return Ok(CompletedCompaction::new(fork_point, document, file_index)),
                Err(error) => error.reason,
            },
            Err(GenerateError::Provider(error)) => {
                return Err(CompactionError::Provider {
                    message: error.to_string(),
                    failure: error.failure(),
                });
            }
            Err(error @ GenerateError::Parse { .. }) => error.to_string(),
        };

        tracing::warn!(reason = %reason, "retrying compaction with a corrective prompt");
        let (mut messages, parameters, definitions) = request.into_parts();
        messages.push(Message::user(D::retry(&reason)));
        let retry_request = LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tool_definitions(definitions);
        let response = collect_text(&model, retry_request).await?;
        let document = parse_handoff::<D>(&response)?;
        document.validate()?;
        Ok(CompletedCompaction::new(fork_point, document, file_index))
    }
}

impl CompletedCompaction {
    /// Splits an accepted `document` into the handoff it renders to, with
    /// `file_index`, and its restoration.
    pub(crate) fn new<D: HandoffSchema>(
        fork_point: u64,
        document: D,
        file_index: Vec<String>,
    ) -> Self {
        let text = document.render(&file_index);
        Self {
            fork_point,
            handoff: Handoff::with_file_index(text, file_index),
            restoration: document.into_restoration(),
        }
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
    /// The handoff that replaces everything older than the fork point.
    pub handoff: Handoff,
    /// What the handoff asked to put back in view.
    pub restoration: Restoration,
}

/// The owned, `'static` future a forked compaction runs as.
type CompactionFuture =
    Pin<Box<dyn Future<Output = Result<CompletedCompaction, CompactionError>> + Send>>;

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
    pub fn start<D: HandoffSchema, M: LanguageModel + 'static>(
        fork: CompactionFork<D>,
        model: M,
    ) -> Self {
        let fork_point = fork.fork_point();
        Self {
            fork_point,
            future: SyncWrapper::new(Box::pin(fork.run(model))),
        }
    }

    /// The sequence number items must reach to survive this compaction.
    #[must_use]
    pub const fn fork_point(&self) -> u64 {
        self.fork_point
    }

    /// Awaits the completed compaction.
    pub fn wait(
        &mut self,
    ) -> impl Future<Output = Result<CompletedCompaction, CompactionError>> + '_ {
        self.future.get_mut()
    }
}

/// Parses a handoff document from a raw model response.
///
/// Models wrap JSON in prose and code fences no matter what the prompt says,
/// so the outermost braces are located rather than trusting the whole body.
fn parse_handoff<D: DeserializeOwned>(response: &str) -> Result<D, CompactionError> {
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
                    failure: error.failure(),
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

        let index = build_file_index(&[], &messages, None);
        assert_eq!(index.len(), 2);
        assert!(
            index[0].contains("outputs/amber-oak-swift-river.txt")
                && index[0].contains("Fetch WebGPU status pages"),
            "index entry should pair path with description: {index:?}"
        );
        assert!(index[1].contains("artifacts/slides.md") && index[1].contains("Write deck"));
    }

    #[test]
    fn build_file_index_finds_the_absolute_path_a_saved_output_notice_prints() {
        let dir = std::path::Path::new("/work/session/outputs");
        // Joined the way the sandbox names the file, so the separator is the
        // platform's.
        let path = dir.join("amber-oak-swift-river.txt").display().to_string();
        let path = path.as_str();
        let messages = vec![
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    "call_1",
                    "terminal",
                    serde_json::json!({"description": "Run the suite"}),
                )],
            ),
            Message::tool(
                "call_1",
                aither_sandbox::output_saved_text("head", path, 100, 5000),
            ),
        ];
        let index = build_file_index(&[], &messages, Some(dir));
        assert_eq!(index.len(), 1, "{index:?}");
        assert!(index[0].contains(path) && index[0].contains("Run the suite"));
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

        let index = build_file_index(&[], &messages, None);
        assert_eq!(index.len(), 1);
        assert!(index[0].contains("final revision"));
    }

    fn read_file_messages(id: &str, description: &str, path: &str) -> Vec<Message> {
        vec![
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    id,
                    "terminal",
                    serde_json::json!({"description": description}),
                )],
            ),
            Message::tool(id, format!("saved {path}")),
        ]
    }

    /// Compacts `context` the way a run does: indexes the conversation seeded
    /// with the current handoff's index, then folds the result into the window.
    fn compact(context: &mut crate::context::Context) -> Vec<String> {
        let file_index = build_file_index(
            context.handoff_file_index(),
            &context.build_request(&[]),
            None,
        );
        context.apply_handoff(
            Handoff::with_file_index("task".to_string(), file_index.clone()),
            u64::MAX,
            0,
        );
        file_index
    }

    #[test]
    fn the_second_handoff_keeps_files_indexed_by_the_first() {
        let mut context = crate::context::Context::new();
        for message in read_file_messages("call_1", "Fetch notes", "artifacts/notes.md") {
            context.push(message);
        }
        let first = compact(&mut context);
        assert_eq!(first, ["artifacts/notes.md — Fetch notes"]);

        for message in read_file_messages("call_2", "Write deck", "artifacts/slides.md") {
            context.push(message);
        }
        let second = compact(&mut context);
        assert_eq!(
            second,
            [
                "artifacts/notes.md — Fetch notes",
                "artifacts/slides.md — Write deck"
            ]
        );

        for message in read_file_messages("call_3", "Revised notes", "artifacts/notes.md") {
            context.push(message);
        }
        let third = compact(&mut context);
        assert_eq!(
            third,
            [
                "artifacts/notes.md — Revised notes",
                "artifacts/slides.md — Write deck"
            ]
        );
    }

    #[test]
    fn handoff_markdown_renders_file_index_section() {
        let handoff = HandoffDocument {
            current_task: "task".to_string(),
            ..HandoffDocument::default()
        };
        let markdown = handoff.render(&["outputs/a-b-c-d.txt — research notes".to_string()]);
        assert!(markdown.contains("Materials Already On Disk"));
        assert!(markdown.contains("outputs/a-b-c-d.txt — research notes"));
        assert!(markdown.contains("Read the file instead of redoing the work"));
    }

    #[test]
    fn a_handoff_without_a_current_task_is_rejected() {
        let error = HandoffDocument::default()
            .validate()
            .expect_err("empty current_task must not validate");
        assert!(error.reason.contains("current_task"));
    }

    #[test]
    fn handoff_parsing_tolerates_fences_and_commentary() {
        let response = "Sure! Here you go:\n```json\n{\"current_task\": \"ship it\", \
            \"key_decisions\": [], \"files_paths\": [], \"errors_fixes\": [], \
            \"pending_work\": [], \"recovery_hints\": []}\n```\nHope that helps.";
        let document = parse_handoff::<HandoffDocument>(response).expect("handoff parses");
        assert_eq!(document.current_task, "ship it");
    }

    #[test]
    fn empty_responses_are_reported_as_invalid() {
        let error =
            parse_handoff::<HandoffDocument>("   ").expect_err("empty response must not parse");
        assert!(matches!(error, CompactionError::Invalid { .. }));
    }

    #[test]
    fn forking_an_empty_context_is_an_error() {
        let request = LLMRequest::new(vec![Message::system("only the prefix")]);
        assert!(matches!(
            CompactionFork::<HandoffDocument>::new(&request, 0, &[], None, &[]),
            Err(CompactionError::Empty)
        ));
    }

    #[test]
    fn the_fork_reproduces_the_main_request_and_appends_the_instruction() {
        use aither_core::llm::model::Parameters;

        let parameters = Parameters {
            temperature: Some(0.3),
            ..Parameters::default()
        };
        let main = LLMRequest::new(vec![
            Message::system("be useful"),
            Message::system("<tool-outputs>tail block</tool-outputs>"),
            Message::user("do the thing"),
        ])
        .with_parameters(parameters);

        let skills = [SkillCatalogEntry {
            name: "pdf".to_string(),
            description: "read PDFs".to_string(),
            instructions: "how to read PDFs".to_string(),
        }];
        let fork = CompactionFork::<HandoffDocument>::new(
            &main,
            7,
            &[],
            Some("the failing test"),
            &skills,
        )
        .expect("fork");

        assert_eq!(fork.fork_point(), 7);
        assert_eq!(
            fork.request().parameters().temperature,
            main.parameters().temperature,
            "the compaction request reuses the main parameters"
        );
        assert_eq!(
            fork.request().tool_definitions().len(),
            main.tool_definitions().len(),
            "the compaction request carries the same tools"
        );
        assert_eq!(fork.request().messages().len(), main.messages().len() + 1);
        for (index, message) in main.messages().iter().enumerate() {
            assert_eq!(
                message.content(),
                fork.request().messages()[index].content(),
                "fork perturbed prefix message {index}"
            );
        }
        let instruction = fork
            .request()
            .messages()
            .last()
            .expect("instruction")
            .content();
        assert!(instruction.contains("Focus the handoff on: the failing test"));
        assert!(instruction.contains("pdf"), "the skill catalog is offered");
    }

    #[derive(Debug)]
    struct NoError;

    impl std::fmt::Display for NoError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("never fails")
        }
    }

    impl std::error::Error for NoError {}

    impl aither_core::llm::Classify for NoError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    /// Answers every request with a handoff and records it.
    struct CapturingModel {
        seen: std::sync::Mutex<Vec<LLMRequest>>,
    }

    impl LanguageModel for CapturingModel {
        type Error = NoError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl futures_lite::Stream<Item = Result<Event, Self::Error>> + Send {
            self.seen.lock().unwrap().push(request);
            futures_lite::stream::iter([Ok(Event::Text(
                r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#
                    .to_string(),
            ))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<
            Output = Result<aither_core::llm::model::Profile, Self::Error>,
        > + Send {
            std::future::ready(Ok(aither_core::llm::model::Profile::new(
                "capture",
                "test",
                "capture",
                "capturing model",
                1_000_000,
            )))
        }
    }

    /// The compaction request must reach the provider with the session's
    /// cache controls and the structured-output flag `generate` sets.
    #[tokio::test]
    async fn the_compaction_request_keeps_the_cache_controls_beside_structured_outputs() {
        use aither_core::llm::model::{
            ClaudeCacheBreakpointTarget, ClaudePromptCacheStrategy, OpenAIPromptCacheRetention,
            Parameters,
        };

        let messages = vec![
            Message::system("prefix"),
            Message::user("hello"),
            Message::assistant("hi"),
            Message::system("live tail"),
        ];
        let parameters = crate::context::PromptCache::new("session-1").apply(
            Parameters::default(),
            &messages,
            false,
            false,
        );
        let main = LLMRequest::new(messages).with_parameters(parameters);
        let fork = CompactionFork::<HandoffDocument>::new(&main, 0, &[], None, &[]).expect("fork");
        let model = CapturingModel {
            seen: std::sync::Mutex::default(),
        };
        fork.run(&model).await.expect("compaction");

        let request = model.seen.lock().unwrap().remove(0);
        let parameters = request.parameters();
        assert!(parameters.structured_outputs);
        let openai = parameters.cache.openai.as_ref().expect("openai cache");
        assert_eq!(openai.key.as_deref(), Some("session-1:0"));
        assert_eq!(openai.retention, Some(OpenAIPromptCacheRetention::Hours24));
        let ClaudePromptCacheStrategy::Explicit(breakpoints) =
            parameters.cache.claude.expect("claude cache").strategy
        else {
            panic!("expected explicit claude breakpoints");
        };
        assert_eq!(
            breakpoints.first.target,
            ClaudeCacheBreakpointTarget::LastSystem
        );
    }
}
