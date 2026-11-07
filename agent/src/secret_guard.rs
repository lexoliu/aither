//! Keeping secrets out of model requests.
//!
//! A secret that enters a model request has left the machine. Tool output
//! carries them routinely — a `.env` read, a printed config, a token a
//! failing command echoed — and the next request would forward it.
//! [`SecretGuard`] stands between every request and the provider: wrap the
//! embedder's model once with [`SecretGuard::guard`], and every agent,
//! subagent, router, compaction and side request sent on it passes the guard.
//!
//! The guard knows two kinds of secret: values the embedder registers
//! ([`SecretGuard::register`]), replaced with `<secret: NAME>`, and strings
//! in the shape of a key or token ([`SecretShape`]), replaced with `<secret>`.
//! Each request's replacements are logged through `tracing`, by name and
//! shape, never by value.
//!
//! Text the embedder is about to publish is stricter: [`SecretGuard::check`]
//! refuses it instead of replacing what it finds.
//!
//! # What is left as it is
//!
//! Content whose integrity the provider verifies passes unchanged:
//! [`ReasoningState`](aither_core::llm::ReasoningState) payloads (signed or
//! encrypted thinking), and the arguments of a tool call bound to a thought
//! signature. The provider produced that content itself, so it carries
//! nothing the provider has not already seen, and rewriting it would get the
//! turn rejected. Attachments pass unchanged too: their URLs address data,
//! and rewriting one breaks the reference.

mod shape;
#[cfg(test)]
mod tests;

use std::{borrow::Cow, ops::Range, sync::Arc};

use aho_corasick::{AhoCorasick, MatchKind};
use aither_core::{
    LanguageModel,
    llm::{
        Event, GenerateError, LLMRequest, LLMRequestWithTools, Message, RetryPolicy, model::Profile,
    },
};
use arc_swap::ArcSwap;
use askama::Template;
use futures_core::Stream;
use futures_lite::StreamExt;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub use shape::SecretShape;
use shape::SecretShapes;

/// The registry of secrets and the check every model request passes.
///
/// A cheap handle: clones share one registry, so a secret registered
/// through any clone is guarded from the next request on, by every model
/// the guard wraps.
#[derive(Clone)]
pub struct SecretGuard {
    shared: Arc<Shared>,
}

struct Shared {
    shapes: SecretShapes,
    registry: ArcSwap<Registry>,
}

/// The registered secrets, swapped whole on each change.
#[derive(Default)]
struct Registry {
    names: Vec<Arc<str>>,
    values: Vec<Arc<str>>,
    /// Finds every value at once; `None` while nothing is registered.
    matcher: Option<AhoCorasick>,
}

impl Registry {
    fn with(&self, name: &str, value: &str) -> Self {
        let mut names = self.names.clone();
        let mut values = self.values.clone();
        if let Some(index) = names.iter().position(|known| &**known == name) {
            values[index] = value.into();
        } else {
            names.push(name.into());
            values.push(value.into());
        }
        // Leftmost-longest: where one value contains another, the longer
        // one is replaced whole.
        let matcher = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(values.iter().map(|value| value.as_bytes()))
            .expect("a handful of registered secrets fits the automaton");
        Self {
            names,
            values,
            matcher: Some(matcher),
        }
    }
}

impl std::fmt::Debug for SecretGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The names only: a value never reaches a log.
        f.debug_struct("SecretGuard")
            .field("registered", &self.shared.registry.load().names)
            .finish_non_exhaustive()
    }
}

impl Default for SecretGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// What a secret found in text is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretKind {
    /// A value registered under `name`.
    Registered {
        /// The name it was registered under.
        name: Arc<str>,
    },
    /// A string in the shape of a key or token.
    Shaped(SecretShape),
}

impl std::fmt::Display for SecretKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Registered { name } => write!(f, "secret {name}"),
            Self::Shaped(shape) => shape.fmt(f),
        }
    }
}

/// A secret found in text: what it is and where, never its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretFinding {
    /// What the secret is.
    pub kind: SecretKind,
    /// Its byte range in the text checked.
    pub range: Range<usize>,
}

/// Why [`SecretGuard::check`] refused a text; the caller renders what the
/// model reads with [`Notices::secret_refusal`](crate::Notices::secret_refusal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretRefusal {
    /// Every secret found, in text order.
    pub findings: Vec<SecretFinding>,
}

#[derive(Template)]
#[template(path = "secret_placeholder.txt", escape = "none")]
struct PlaceholderTemplate<'a> {
    name: &'a str,
}

/// What stands in for a secret found by its shape.
const SHAPE_PLACEHOLDER: &str = include_str!("texts/secret_shape_placeholder.txt");

/// The replacements made in one request or text, for its log line.
#[derive(Default)]
struct Tally {
    registered: Vec<Arc<str>>,
    shaped: Vec<SecretShape>,
}

impl Tally {
    fn record(&mut self, kind: &SecretKind) {
        match kind {
            SecretKind::Registered { name } => self.registered.push(Arc::clone(name)),
            SecretKind::Shaped(shape) => self.shaped.push(*shape),
        }
    }

    fn log(self, subject: &str) {
        if self.registered.is_empty() && self.shaped.is_empty() {
            return;
        }
        let registered: Vec<&str> = self.registered.iter().map(|name| &**name).collect();
        tracing::info!(
            registered = ?registered,
            shaped = ?self.shaped,
            "replaced secrets in {subject}",
        );
    }
}

impl SecretGuard {
    /// Creates a guard with no registered secrets; it already replaces
    /// strings in the shape of a key or token.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                shapes: SecretShapes::new(),
                registry: ArcSwap::from_pointee(Registry::default()),
            }),
        }
    }

    /// Registers `value` as the secret `name`, replacing the value `name`
    /// had. From the next request on, every occurrence of it is replaced
    /// with `<secret: NAME>`.
    ///
    /// # Panics
    ///
    /// Panics when `value` is empty: an empty secret occurs everywhere, so
    /// registering one is a mistake in the calling program.
    pub fn register(&self, name: impl AsRef<str>, value: impl AsRef<str>) {
        let (name, value) = (name.as_ref(), value.as_ref());
        assert!(!value.is_empty(), "secret {name} is registered empty");
        self.shared
            .registry
            .rcu(|registry| Arc::new(registry.with(name, value)));
    }

    /// Wraps `model` so every request sent on it passes this guard.
    pub fn guard<M>(&self, model: M) -> Guarded<M> {
        Guarded {
            model,
            guard: self.clone(),
        }
    }

    /// Every secret in `text`, in text order, registered values ahead of
    /// shapes where they overlap.
    #[must_use]
    pub fn find(&self, text: &str) -> Vec<SecretFinding> {
        let registry = self.shared.registry.load();
        let mut findings: Vec<SecretFinding> = registry
            .matcher
            .as_ref()
            .map(|matcher| {
                matcher
                    .find_iter(text)
                    .map(|hit| SecretFinding {
                        kind: SecretKind::Registered {
                            name: Arc::clone(&registry.names[hit.pattern().as_usize()]),
                        },
                        range: hit.range(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Registered findings never overlap one another; a shape is kept
        // only where nothing found before it lies.
        let mut taken: Vec<Range<usize>> = findings
            .iter()
            .map(|finding| finding.range.clone())
            .collect();
        let mut shaped = Vec::new();
        self.shared.shapes.find(text, |shape, range| {
            if !taken.iter().any(|kept| overlaps(kept, &range)) {
                taken.push(range.clone());
                shaped.push(SecretFinding {
                    kind: SecretKind::Shaped(shape),
                    range,
                });
            }
        });
        findings.extend(shaped);
        findings.sort_by_key(|finding| finding.range.start);
        findings
    }

    /// Refuses `text` when it carries a secret, for text about to be
    /// published.
    ///
    /// # Errors
    ///
    /// Returns every secret found when there is one.
    pub fn check(&self, text: &str) -> Result<(), SecretRefusal> {
        let findings = self.find(text);
        if findings.is_empty() {
            Ok(())
        } else {
            Err(SecretRefusal { findings })
        }
    }

    /// `text` with each secret replaced by its placeholder, logging the
    /// replacements.
    #[must_use]
    pub fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut tally = Tally::default();
        let redacted = self.redact_text(text, &mut tally);
        tally.log("a text");
        redacted
    }

    /// Replaces each secret in `request` in place, logging the replacements.
    fn redact_request(&self, request: &mut LLMRequest) {
        let mut tally = Tally::default();
        for message in request.messages_mut() {
            match message {
                Message::User { content, .. }
                | Message::System { content }
                | Message::Tool { content, .. } => self.redact_in_place(content, &mut tally),
                // Reasoning state is the provider's own, signed or
                // encrypted, and passes as it is; so do the arguments of a
                // call bound to a thought signature.
                Message::Assistant {
                    content,
                    tool_calls,
                    reasoning: _,
                } => {
                    self.redact_in_place(content, &mut tally);
                    for call in tool_calls
                        .iter_mut()
                        .filter(|call| call.reasoning_state.is_none())
                    {
                        self.redact_value(&mut call.arguments, &mut tally);
                    }
                }
                // An effort level carries no text.
                Message::Configuration { .. } => {}
            }
        }
        tally.log("a model request");
    }

    fn redact_value(&self, value: &mut Value, tally: &mut Tally) {
        match value {
            Value::String(text) => self.redact_in_place(text, tally),
            Value::Array(items) => {
                for item in items {
                    self.redact_value(item, tally);
                }
            }
            Value::Object(fields) => {
                for field in fields.values_mut() {
                    self.redact_value(field, tally);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    fn redact_in_place(&self, text: &mut String, tally: &mut Tally) {
        if let Cow::Owned(redacted) = self.redact_text(text, tally) {
            *text = redacted;
        }
    }

    fn redact_text<'a>(&self, text: &'a str, tally: &mut Tally) -> Cow<'a, str> {
        let findings = self.find(text);
        if findings.is_empty() {
            return Cow::Borrowed(text);
        }
        let mut redacted = String::with_capacity(text.len());
        let mut copied = 0;
        for finding in findings {
            redacted.push_str(&text[copied..finding.range.start]);
            match &finding.kind {
                SecretKind::Registered { name } => PlaceholderTemplate { name }
                    .render_into(&mut redacted)
                    .expect("the secret placeholder renders"),
                SecretKind::Shaped(_) => redacted.push_str(SHAPE_PLACEHOLDER),
            }
            copied = finding.range.end;
            tally.record(&finding.kind);
        }
        redacted.push_str(&text[copied..]);
        Cow::Owned(redacted)
    }
}

const fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// A model whose every request passes a [`SecretGuard`] before it reaches
/// the provider.
///
/// Made with [`SecretGuard::guard`].
#[derive(Debug, Clone)]
pub struct Guarded<M> {
    model: M,
    guard: SecretGuard,
}

impl<M> Guarded<M> {
    /// The wrapped model.
    pub const fn inner(&self) -> &M {
        &self.model
    }

    /// The guard every request passes.
    pub const fn guard(&self) -> &SecretGuard {
        &self.guard
    }
}

impl<M: LanguageModel> LanguageModel for Guarded<M> {
    type Error = M::Error;

    fn respond(
        &self,
        mut request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.guard.redact_request(&mut request);
        self.model.respond(request)
    }

    fn respond_with_tools(
        &self,
        mut request: LLMRequestWithTools<'_>,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.guard.redact_request(request.request_mut());
        self.model.respond_with_tools(request)
    }

    fn generate<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        mut request: LLMRequest,
    ) -> impl Future<Output = Result<T, GenerateError<Self::Error>>> + Send {
        self.guard.redact_request(&mut request);
        self.model.generate(request)
    }

    fn complete(&self, prefix: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let prefix = self.guard.redact(prefix);
        async_stream::stream! {
            let mut events = std::pin::pin!(self.model.complete(&prefix));
            while let Some(event) = events.next().await {
                yield event;
            }
        }
    }

    fn summarize(&self, text: &str) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let text = self.guard.redact(text);
        async_stream::stream! {
            let mut events = std::pin::pin!(self.model.summarize(&text));
            while let Some(event) = events.next().await {
                yield event;
            }
        }
    }

    fn categorize<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<T, GenerateError<Self::Error>>> + Send {
        let text = self.guard.redact(text);
        async move { self.model.categorize(&text).await }
    }

    fn profile(&self) -> impl Future<Output = Result<Profile, Self::Error>> + Send {
        self.model.profile()
    }

    fn retry_policy(&self) -> RetryPolicy {
        self.model.retry_policy()
    }
}
