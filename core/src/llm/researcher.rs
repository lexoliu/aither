//! Deep research workflows and agent-based investigation capabilities.
//!
//! This module provides abstractions for AI-powered research agents that can conduct
//! in-depth investigations by planning, searching, reading sources, and synthesizing findings.

use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use core::future::Future;
use futures_core::Stream;
use futures_lite::{StreamExt, pin};

/// Request describing a deep-research task.
#[derive(Clone, Debug)]
pub struct ResearchRequest {
    /// Main research question or thesis.
    pub query: String,
    /// Optional rubric or format instructions.
    pub instructions: Option<String>,
    /// User-provided materials that should be ingested before browsing.
    pub sources: Vec<ResearchSource>,
    /// Provider options controlling depth and tooling.
    pub options: ResearchOptions,
}

impl ResearchRequest {
    /// Creates a request for a given query/topic.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            instructions: None,
            sources: Vec::new(),
            options: ResearchOptions::default(),
        }
    }

    /// Adds instructions for the researcher.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Adds a source while keeping ownership of `self`.
    #[must_use]
    pub fn with_source(mut self, source: ResearchSource) -> Self {
        self.sources.push(source);
        self
    }

    /// Pushes a source in-place.
    pub fn push_source(&mut self, source: ResearchSource) {
        self.sources.push(source);
    }

    /// Overrides provider options.
    #[must_use]
    pub const fn options(mut self, options: ResearchOptions) -> Self {
        self.options = options;
        self
    }
}

/// Cross-provider toggles for deep research.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResearchOptions {
    /// Max number of planner/executor interactions.
    pub max_interactions: Option<u16>,
    /// Whether the agent may browse the web.
    pub allow_web_browsing: bool,
    /// Whether code execution sandboxes may be used.
    pub allow_code_execution: bool,
    /// Optional sampling temperature.
    pub temperature: Option<f32>,
}

impl ResearchOptions {
    /// Sets the interaction limit.
    #[must_use]
    pub fn max_interactions(mut self, value: impl Into<Option<u16>>) -> Self {
        self.max_interactions = value.into();
        self
    }

    /// Enables or disables browsing.
    #[must_use]
    pub const fn web_browsing(mut self, allow: bool) -> Self {
        self.allow_web_browsing = allow;
        self
    }

    /// Enables or disables code execution.
    #[must_use]
    pub const fn code_execution(mut self, allow: bool) -> Self {
        self.allow_code_execution = allow;
        self
    }

    /// Sets a temperature (or resets it with `None`).
    #[must_use]
    pub fn temperature(mut self, temperature: impl Into<Option<f32>>) -> Self {
        self.temperature = temperature.into();
        self
    }
}

/// Seed information made available to the researcher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResearchSource {
    /// URL that should be fetched/summarized.
    Url {
        /// URL to fetch
        url: String,
        /// Optional label
        label: Option<String>,
    },
    /// Path or identifier for a file attachment.
    File {
        /// File path
        path: String,
    },
    /// Arbitrary note or doc excerpt.
    Note {
        /// Note title
        title: String,
        /// Note content
        content: String,
    },
}

impl ResearchSource {
    /// Creates an unlabeled URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            label: None,
        }
    }

    /// Creates a labeled URL.
    #[must_use]
    pub fn labeled_url(url: impl Into<String>, label: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            label: Some(label.into()),
        }
    }

    /// Creates a file source.
    #[must_use]
    pub fn file(path: impl Into<String>) -> Self {
        Self::File { path: path.into() }
    }

    /// Creates an inline note.
    #[must_use]
    pub fn note(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Note {
            title: title.into(),
            content: content.into(),
        }
    }
}

/// Streaming events produced by deep research providers.
#[derive(Clone, Debug)]
pub enum ResearchEvent {
    /// Planner / executor status or log line.
    Stage {
        /// Logical stage of the workflow.
        stage: ResearchStage,
        /// Human-readable update.
        message: String,
    },
    /// Structured finding discovered mid-run.
    Finding(ResearchFinding),
    /// Citation shared outside of a finding.
    Citation(ResearchCitation),
    /// Finalized report.
    Finalized(ResearchReport),
}

/// Shared stages used to monitor research progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResearchStage {
    /// Planning next steps.
    Planning,
    /// Querying search engines and browsing.
    Searching,
    /// Reading primary sources.
    Reading,
    /// Writing the report.
    Writing,
    /// Report finalized.
    Completed,
}

/// Structured findings gathered by the researcher.
#[derive(Clone, Debug)]
pub struct ResearchFinding {
    /// Headline or claim.
    pub title: String,
    /// Supporting evidence / synthesis.
    pub summary: String,
    /// Optional provider confidence (0-1).
    pub confidence: Option<f32>,
    /// Citations backing the claim.
    pub citations: Vec<ResearchCitation>,
}

impl ResearchFinding {
    /// Creates a new finding.
    #[must_use]
    pub fn new(title: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            summary: summary.into(),
            confidence: None,
            citations: Vec::new(),
        }
    }

    /// Sets a confidence score.
    #[must_use]
    pub const fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Appends a citation to the finding.
    #[must_use]
    pub fn citation(mut self, citation: ResearchCitation) -> Self {
        self.citations.push(citation);
        self
    }
}

/// Normalized citation metadata.
#[derive(Clone, Debug)]
pub struct ResearchCitation {
    /// Source URL or provider reference.
    pub url: String,
    /// Optional title/headline.
    pub title: Option<String>,
    /// Optional snippet for quick reference.
    pub snippet: Option<String>,
}

impl ResearchCitation {
    /// Creates a citation pointing to a URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            title: None,
            snippet: None,
        }
    }

    /// Adds a title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Adds a snippet.
    #[must_use]
    pub fn snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into());
        self
    }
}

/// Final report returned by the researcher.
#[derive(Clone, Debug, Default)]
pub struct ResearchReport {
    /// Executive summary.
    pub summary: String,
    /// Structured findings.
    pub findings: Vec<ResearchFinding>,
    /// Deduplicated citation list.
    pub citations: Vec<ResearchCitation>,
}

impl ResearchReport {
    /// Overrides the summary for builder-style usage.
    #[must_use]
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Adds a finding.
    pub fn push_finding(&mut self, finding: ResearchFinding) {
        self.findings.push(finding);
    }

    /// Adds a citation.
    pub fn push_citation(&mut self, citation: ResearchCitation) {
        self.citations.push(citation);
    }
}

/// Metadata describing capabilities of a research provider.
#[derive(Clone, Debug)]
pub struct ResearcherProfile {
    /// Human-readable provider/model name.
    pub name: String,
    /// Whether streaming updates are supported.
    pub supports_streaming: bool,
    /// Whether the provider can browse the web.
    pub supports_web_browsing: bool,
    /// Whether running code/tools is supported.
    pub supports_code_execution: bool,
}

/// Trait abstracting providers offering "deep research" workflows (`OpenAI`, `Claude`, etc.).
pub trait Researcher: Sized + Send + Sync {
    /// Error produced by streaming events.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Streams the research workflow (planner updates, findings, report).
    fn research(
        &self,
        request: &ResearchRequest,
    ) -> impl Stream<Item = Result<ResearchEvent, Self::Error>> + Send;

    /// Collects the final report by consuming the stream.
    fn report(
        &self,
        request: &ResearchRequest,
    ) -> impl Future<Output = crate::Result<ResearchReport>> + Send {
        research_report(self, request)
    }

    /// Returns provider metadata.
    fn profile(&self) -> impl Future<Output = ResearcherProfile> + Send;
}

macro_rules! impl_researcher {
    ($($name:ident),*) => {
        $(
            impl<T: Researcher> Researcher for $name<T> {
                type Error = T::Error;

                fn research(
                    &self,
                    request: &ResearchRequest,
                ) -> impl Stream<Item = Result<ResearchEvent, Self::Error>> + Send {
                    T::research(self, request)
                }

                fn profile(&self) -> impl Future<Output = ResearcherProfile> + Send {
                    T::profile(self)
                }
            }
        )*
    };
}

impl_researcher!(Arc, Box);

impl<T: Researcher> Researcher for &T {
    type Error = T::Error;

    fn research(
        &self,
        request: &ResearchRequest,
    ) -> impl Stream<Item = Result<ResearchEvent, Self::Error>> + Send {
        T::research(self, request)
    }

    fn profile(&self) -> impl Future<Output = ResearcherProfile> + Send {
        T::profile(self)
    }
}

/// Collects the final report from a research stream.
async fn research_report<R: Researcher>(
    researcher: &R,
    request: &ResearchRequest,
) -> crate::Result<ResearchReport> {
    let stream = researcher.research(request);
    pin!(stream);

    let mut report = ResearchReport::default();

    while let Some(event) = stream.try_next().await.map_err(anyhow::Error::new)? {
        match event {
            ResearchEvent::Finding(finding) => report.push_finding(finding),
            ResearchEvent::Citation(citation) => report.push_citation(citation),
            ResearchEvent::Finalized(final_report) => return Ok(final_report),
            ResearchEvent::Stage { .. } => {}
        }
    }

    Ok(report)
}
