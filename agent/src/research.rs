//! Deep-research orchestration built on top of the generic [`Agent`] runtime.
//!
//! The `DeepResearchAgent` wraps a language model with planning, memory, sub-agents,
//! and a web search provider so that it can satisfy the [`Researcher`] trait from
//! `aither-core`. It always mounts a scout-style sub-agent to quickly summarize
//! freshly fetched SERP results and optionally exposes command execution when the
//! `command` feature is enabled.

use aither_core::llm::Tool;
use aither_core::{
    Error as CoreError, LanguageModel, Result as CoreResult,
    llm::{
        oneshot,
        researcher::{
            ResearchCitation, ResearchEvent, ResearchFinding, ResearchOptions, ResearchReport,
            ResearchRequest, ResearchSource, ResearchStage, Researcher, ResearcherProfile,
        },
    },
};
use async_stream::try_stream;
use core::fmt::{self, Write as _};

use crate::{
    Agent, AgentConfig, FileSystemAccess, ToolingConfig, execute::Executor,
    memory::ContextStrategy, plan, sub_agent::SubAgent,
};

use crate::websearch::{SearchProvider, SearchResult};

/// Research-focused agent that satisfies [`Researcher`] by composing planners,
/// executors, sub-agents, and a [`SearchProvider`].
///
/// It accepts any [`LanguageModel`] implementation so long as it is `Clone`.
/// Web browsing is handled through the `SearchProvider`, and command/tooling
/// support is turned on when the `command` feature is available. A scout
/// sub-agent is spawned per request to triage SERP results before the main
/// agent writes the final report.
#[derive(Debug)]
pub struct DeepResearchAgent<LLM, P>
where
    LLM: LanguageModel + Clone,
    P: SearchProvider + Clone,
{
    llm: LLM,
    search_provider: P,
    executor: ResearchExecutor<LLM, P>,
    researcher_config: AgentConfig,
    scout_config: AgentConfig,
    default_code_execution: bool,
}

/// Error type used by [`DeepResearchAgent`].
#[derive(Debug)]
pub struct DeepResearchError(anyhow::Error);

impl fmt::Display for DeepResearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for DeepResearchError {}

impl From<anyhow::Error> for DeepResearchError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

impl<LLM, P> DeepResearchAgent<LLM, P>
where
    LLM: LanguageModel + Clone + 'static,
    P: SearchProvider + Clone + 'static,
{
    /// Creates a new deep research agent with sensible defaults for context
    /// compression and interaction limits.
    #[must_use]
    pub fn new(llm: LLM, provider: P) -> Self {
        let researcher_tooling =
            ToolingConfig::new(Some(FileSystemAccess::working_directory(true)), false);
        let researcher_config = AgentConfig::new(
            ContextStrategy::Summarize {
                max_messages: 80,
                retain_recent: 16,
                instructions: "Preserve citations, commands, and measurements verbatim.".into(),
            },
            96,
            researcher_tooling,
        );

        let scout_tooling =
            ToolingConfig::new(Some(FileSystemAccess::working_directory(true)), false);

        let scout_config = AgentConfig::new(
            ContextStrategy::SlidingWindow { max_messages: 24 },
            32,
            scout_tooling,
        );

        Self {
            executor: ResearchExecutor::new(llm.clone(), provider.clone()),
            llm,
            search_provider: provider,
            researcher_config,
            scout_config,
            default_code_execution: false,
        }
    }

    /// Enables or disables code execution for every request (in addition to per-request options).
    #[must_use]
    pub const fn with_code_execution(mut self, enabled: bool) -> Self {
        self.default_code_execution = enabled;
        self
    }

    /// Overrides the agent configs used for the primary researcher and scout sub-agent.
    #[must_use]
    pub fn with_configs(mut self, researcher: AgentConfig, scout: AgentConfig) -> Self {
        self.researcher_config = researcher;
        self.scout_config = scout;
        self
    }

    fn build_agent_with_config(
        &self,
        base: &AgentConfig,
        options: &ResearchOptions,
    ) -> Agent<LLM, plan::DefaultPlanner, ResearchExecutor<LLM, P>> {
        let mut config = base.clone();
        if let Some(limit) = options.max_interactions {
            config.set_max_iterations(limit as usize);
        }

        let mut agent = Agent::custom(
            self.llm.clone(),
            plan::DefaultPlanner,
            self.executor.clone(),
            config,
        );

        if options.allow_web_browsing {
            agent.enable_websearch(self.search_provider.clone());
        }

        #[cfg(feature = "command")]
        if self.default_code_execution || options.allow_code_execution {
            agent.enable_shell(None);
        }

        agent
    }

    fn build_primary_agent(
        &self,
        options: &ResearchOptions,
    ) -> Agent<LLM, plan::DefaultPlanner, ResearchExecutor<LLM, P>> {
        self.build_agent_with_config(&self.researcher_config, options)
    }

    fn build_scout_tool(
        &self,
        options: &ResearchOptions,
    ) -> SubAgent<LLM, plan::DefaultPlanner, ResearchExecutor<LLM, P>> {
        let agent = self.build_agent_with_config(&self.scout_config, options);
        agent.into_subagent(
            "scout_researcher",
            "Fast sub-agent that skims search results and produces briefs.",
        )
    }

    fn compose_goal(
        request: &ResearchRequest,
        scout_brief: &str,
        results: &[SearchResult],
    ) -> String {
        let mut goal = format!("Primary research question: {}\n", request.query);
        if let Some(instructions) = &request.instructions {
            goal.push_str("Additional instructions:\n");
            goal.push_str(instructions);
            goal.push('\n');
        }

        if !request.sources.is_empty() {
            goal.push_str("User-provided materials:\n");
            for source in &request.sources {
                goal.push_str("- ");
                match source {
                    ResearchSource::Url { url, label } => {
                        if let Some(label) = label {
                            goal.push_str(label);
                            goal.push_str(" (");
                            goal.push_str(url);
                            goal.push_str(")\n");
                        } else {
                            goal.push_str(url);
                            goal.push('\n');
                        }
                    }
                    ResearchSource::File { path } => {
                        goal.push_str("File: ");
                        goal.push_str(path);
                        goal.push('\n');
                    }
                    ResearchSource::Note { title, content } => {
                        goal.push_str(title);
                        goal.push_str(": ");
                        goal.push_str(content);
                        goal.push('\n');
                    }
                }
            }
        }

        if !results.is_empty() {
            goal.push_str("Recent search results to reference:\n");
            for (idx, result) in results.iter().enumerate() {
                let _ = writeln!(
                    goal,
                    "{}. {} ({})\nSnippet: {}\n",
                    idx + 1,
                    result.title,
                    result.url,
                    result.snippet
                );
            }
        }

        if !scout_brief.is_empty() {
            goal.push_str("\nScout briefing:\n");
            goal.push_str(scout_brief);
            goal.push('\n');
        }

        goal.push_str("\nProduce a concise report with citations for every major claim.");
        goal
    }

    fn build_report(
        summary: String,
        scout_brief: &str,
        results: &[SearchResult],
    ) -> ResearchReport {
        let mut report = ResearchReport::default().summary(summary);

        if !scout_brief.is_empty() {
            let finding = ResearchFinding::new("Scouting synopsis", scout_brief.to_string());
            report.push_finding(finding);
        }

        for result in results {
            let citation = ResearchCitation {
                url: result.url.clone(),
                title: Some(result.title.clone()),
                snippet: Some(result.snippet.clone()),
            };
            report.push_citation(citation.clone());
            let finding =
                ResearchFinding::new(format!("Source: {}", result.title), result.snippet.clone())
                    .citation(citation);
            report.push_finding(finding);
        }

        if report.summary.is_empty() {
            report.summary = "No synthesis produced.".into();
        }

        report
    }

    async fn scout_briefing(
        &self,
        scout: &mut SubAgent<LLM, plan::DefaultPlanner, ResearchExecutor<LLM, P>>,
        request: &ResearchRequest,
        results: &[SearchResult],
    ) -> CoreResult<String> {
        if results.is_empty() {
            return Ok(String::new());
        }
        let mut prompt = format!(
            "Act as a fast triage researcher. Summarize the most relevant trends for: {}\n",
            request.query
        );
        if let Some(instructions) = &request.instructions {
            prompt.push_str("Follow these special instructions:\n");
            prompt.push_str(instructions);
            prompt.push('\n');
        }

        for (idx, result) in results.iter().enumerate() {
            let _ = writeln!(
                prompt,
                "{}. {} ({}) -> {}",
                idx + 1,
                result.title,
                result.url,
                result.snippet
            );
        }

        scout.call(prompt).await
    }
}

fn wrap_core_error(err: &CoreError) -> DeepResearchError {
    DeepResearchError(anyhow::Error::msg(err.to_string()))
}

impl<LLM, P> Researcher for DeepResearchAgent<LLM, P>
where
    LLM: LanguageModel + Clone + Send + Sync + 'static,
    P: SearchProvider + Clone + Send + Sync + 'static,
{
    type Error = DeepResearchError;

    fn research(
        &self,
        request: &ResearchRequest,
    ) -> impl futures_core::Stream<Item = Result<ResearchEvent, Self::Error>> + Send {
        let request = request.clone();
        let mut agent = self.build_primary_agent(&request.options);
        let mut scout = self.build_scout_tool(&request.options);
        let provider = self.search_provider.clone();
        let executor_limit = self.executor.max_results();
        let default_code = self.default_code_execution;

        try_stream! {
            yield ResearchEvent::Stage {
                stage: ResearchStage::Planning,
                message: format!("Planning deep research workflow for '{}'", request.query),
            };

            let mut search_results = Vec::new();
            if request.options.allow_web_browsing {
                yield ResearchEvent::Stage {
                    stage: ResearchStage::Searching,
                    message: "Querying web search provider for fresh sources.".into(),
                };
                search_results = provider
                    .search(&request.query, executor_limit)
                    .await
                    .map_err(|err| wrap_core_error(&err))?;

                for result in &search_results {
                    let finding = ResearchFinding::new(
                        format!("Found: {}", result.title),
                        result.snippet.clone(),
                    )
                    .citation(ResearchCitation {
                        url: result.url.clone(),
                        title: Some(result.title.clone()),
                        snippet: Some(result.snippet.clone()),
                    });
                    yield ResearchEvent::Finding(finding);
                }
            }

            let briefing = self
                .scout_briefing(&mut scout, &request, &search_results)
                .await
                .map_err(|err| wrap_core_error(&err))?;

            yield ResearchEvent::Stage {
                stage: ResearchStage::Reading,
                message: "Scout agent summarized promising leads.".into(),
            };

            agent.register_tool(scout);

            let goal = Self::compose_goal(&request, &briefing, &search_results);

            yield ResearchEvent::Stage {
                stage: ResearchStage::Writing,
                message: if default_code || request.options.allow_code_execution {
                    "Synthesizing report with coding/tooling enabled.".into()
                } else {
                    "Synthesizing report.".into()
                },
            };

            let summary = agent
                .run(&goal)
                .await
                .map_err(|err| wrap_core_error(&err))?;

            let report = Self::build_report(summary, &briefing, &search_results);

            yield ResearchEvent::Finalized(report.clone());
            yield ResearchEvent::Stage {
                stage: ResearchStage::Completed,
                message: "Deep research workflow finished.".into(),
            };
        }
    }

    async fn profile(&self) -> ResearcherProfile {
        ResearcherProfile {
            name: "aither-agent/deep-research".into(),
            supports_streaming: true,
            supports_web_browsing: true,
            supports_code_execution: cfg!(feature = "command"),
        }
    }
}

/// Executor that turns every plan step into a quick search + synthesis.
#[derive(Clone, Debug)]
pub struct ResearchExecutor<LLM, P>
where
    LLM: LanguageModel + Clone,
    P: SearchProvider + Clone,
{
    llm: LLM,
    provider: P,
    max_results: usize,
}

impl<LLM, P> ResearchExecutor<LLM, P>
where
    LLM: LanguageModel + Clone,
    P: SearchProvider + Clone,
{
    /// Creates a new executor that keeps at most five search hits per step.
    #[must_use]
    pub const fn new(llm: LLM, provider: P) -> Self {
        Self {
            llm,
            provider,
            max_results: 5,
        }
    }

    /// Sets the maximum number of web results captured for each execution step.
    #[must_use]
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results.clamp(1, 10);
        self
    }

    /// Returns the number of search results the executor consumes.
    #[must_use]
    pub const fn max_results(&self) -> usize {
        self.max_results
    }
}

impl<LLM, P> Executor for ResearchExecutor<LLM, P>
where
    LLM: LanguageModel + Clone,
    P: SearchProvider + Clone,
{
    async fn execute(&self, step: &str, _state: &crate::AgentState) -> CoreResult<String> {
        let results = self.provider.search(step, self.max_results).await?;
        let mut evidence_block = String::new();
        for (idx, result) in results.iter().enumerate() {
            let _ = writeln!(
                evidence_block,
                "{}. {} ({}) -> {}",
                idx + 1,
                result.title,
                result.url,
                result.snippet
            );
        }

        if evidence_block.is_empty() {
            evidence_block =
                "No supporting sources were found; rely on reasoning and prior context.".into();
        }

        let request = oneshot(
            "You are executing a research step. Synthesize insights grounded in the provided sources.",
            format!("Step: {step}\n\nSources:\n{evidence_block}"),
        );
        self.llm.generate(request).await
    }
}
