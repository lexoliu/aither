//! Builder for constructing agents with custom configuration.
//!
//! The builder pattern allows fluent configuration of agents with
//! tools, hooks, and various settings.
//!
//! The agent-level options — prompts, context policy, prompt cache, request
//! parameters and the rest — are the methods of [`AgentOptions`], which
//! [`AgentBuilder`] and [`TerminalAgentBuilder`](crate::TerminalAgentBuilder)
//! both implement over the same [`AgentSettings`]. An option added there is
//! set the same way on either builder.

use std::marker::PhantomData;
#[cfg(feature = "skills")]
use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{Tool, tool::ToolDefinition},
};
use aither_sandbox::{BackgroundTaskReceiver, JobRegistry, PermissionEventReceiver};
#[cfg(feature = "skills")]
use aither_skills::SkillRegistry;
use askama::Template;

use crate::{
    agent::Agent,
    config::{AgentConfig, AgentKind},
    context::{
        AnyContextSource, Context, ContextPolicy, ContextSource, ExternalContent, ExternalSlot,
        HandoffDocument, HandoffSchema, InstructionSlot, SystemLayout,
    },
    hook::{HCons, Hook},
    model_group::ModelTier,
    request_parameters::{TierParameters, WithParameters},
    tools::AgentTools,
};
use aither_core::llm::model::Parameters;

#[cfg(feature = "mcp")]
use aither_mcp::McpConnection;

/// Builder for constructing agents with custom configuration.
///
/// Supports tiered LLM configuration:
/// - Advanced: Primary model for main reasoning (most capable)
/// - Balanced: Model for moderate tasks like subagents (defaults to advanced)
/// - Fast: Model for quick tasks like translating a reply (defaults to balanced)
///
/// Each model comes with the parameters of the requests aither sends it; see
/// [`AgentOptions::request_parameters`].
///
/// The agent-level options are the methods of [`AgentOptions`].
///
/// # Example
///
/// ```rust,ignore
/// use aither_agent::AgentOptions;
///
/// // Simple: all tiers use the same model
/// let agent = Agent::builder(claude).build();
///
/// // Tiered: different models for different tasks
/// let agent = Agent::builder(opus)                          // Advanced
///     .request_parameters(opus_parameters)
///     .balanced_model(sonnet, sonnet_parameters)            // Balanced
///     .fast_model(haiku, haiku_parameters)                  // Fast
///     .system_prompt("You are a helpful assistant.")
///     .tool(FileSystemTool::read_only("."))
///     .build();
/// ```
#[must_use]
pub struct AgentBuilder<
    Advanced,
    Balanced = Advanced,
    Fast = Balanced,
    H = (),
    Compactor = Advanced,
    Schema = HandoffDocument,
> {
    advanced: Advanced,
    balanced: Balanced,
    fast: Fast,
    /// The model compaction runs on instead of the session's own, with the
    /// parameters of its requests.
    compaction_model: Option<WithParameters<Compactor>>,
    hooks: H,
    settings: AgentSettings,
    wiring: Wiring,
    /// The handoff schema compactions ask for; no value is held.
    handoff_schema: PhantomData<fn() -> Schema>,
}

/// What the [`AgentOptions`] of a builder set.
///
/// Opaque: it is read and written only through the options, and carried
/// whole through the builder steps that change a builder's type.
pub struct AgentSettings {
    /// The parameters of the requests sent on each tier's model.
    pub(crate) parameters: TierParameters,
    pub(crate) tier: ModelTier,
    pub(crate) config: AgentConfig,
    pub(crate) context: Context,
    pub(crate) prompt_cache: crate::context::PromptCache,
    pub(crate) clock: crate::AnyClock,
    pub(crate) context_sources: Vec<AnyContextSource>,
    /// The agent's working document and what the host gives it.
    pub(crate) document: crate::working_document::DocumentWiring,
    /// The embedder's stop rule; without one a run ends whenever the model
    /// ends its turn and no context source objects.
    pub(crate) stop_hook: Option<crate::AnyStopHook>,
    /// Where the built agent publishes each request for a fork to start from.
    pub(crate) fork_source: Option<crate::fork::ForkSource>,
    /// Who the built agent's tool calls are made by.
    pub(crate) identity: aither_core::llm::AgentIdentity,
    /// What renders every alert, notice, reminder and refusal.
    pub(crate) notices: crate::AnyNotices,
    #[cfg(feature = "skills")]
    pub(crate) skill_registry: Option<Arc<SkillRegistry>>,
}

impl std::fmt::Debug for AgentSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSettings")
            .field("tier", &self.tier)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            parameters: TierParameters::default(),
            tier: ModelTier::default(),
            config: AgentConfig::default(),
            context: Context::default(),
            prompt_cache: crate::context::PromptCache::new(uuid::Uuid::new_v4().to_string()),
            clock: crate::AnyClock::default(),
            context_sources: Vec::new(),
            document: crate::working_document::DocumentWiring::default(),
            stop_hook: None,
            fork_source: None,
            identity: aither_core::llm::AgentIdentity::Main,
            notices: crate::AnyNotices::default(),
            #[cfg(feature = "skills")]
            skill_registry: None,
        }
    }
}

/// What the builder wires up from the tools and channels it is given, as
/// opposed to the options a host sets.
#[derive(Default)]
struct Wiring {
    tools: AgentTools,
    connectivity: crate::connectivity::Watch,
    background_receiver: Option<BackgroundTaskReceiver>,
    permission_receiver: Option<PermissionEventReceiver>,
    /// The terminal calls in which a kept command succeeded.
    kept_receiver: Option<aither_sandbox::KeptExchangeReceiver>,
    /// The commands the terminal runs, kept current in the command table.
    commands: Option<crate::command_catalog::CommandCatalog>,
    host_load_events: Option<async_channel::Receiver<crate::HostLoadEvent>>,
    job_registry: Option<JobRegistry>,
    /// Directory the agent's working document is kept in: the directory its
    /// terminal works in.
    sandbox_dir: Option<std::path::PathBuf>,
    /// Directory files are restored from after a forked compaction: the
    /// directory the agent's terminal works in.
    restore_dir: Option<std::path::PathBuf>,
    /// What the `compact` tool shares with the built agent; the tool is
    /// registered at build, over the notices the options settled on.
    compact_signal: Option<std::sync::Arc<crate::compact_tool::CompactSignal>>,
    /// The window state a fork inherits, which replaces initialization.
    inherited: Option<Inherited>,
    /// The tools a fork runs, built at build over the notices the options
    /// settled on; they replace every other tool.
    fork_tools: Option<ForkTools>,
}

/// The tools a fork inherits: the main agent's definitions, run by the
/// fork's own terminal.
struct ForkTools {
    definitions: Vec<ToolDefinition>,
    terminal: aither_sandbox::DynTerminalTool,
}

/// The built agent's tools: a fork's inherited set in place of `tools`, or
/// `tools` with the `compact` tool over `compact`, each rendering through
/// `notices`.
///
/// # Panics
///
/// Panics if `compact` is set and `tools` already holds a tool named
/// `compact`.
fn assemble_tools(
    mut tools: AgentTools,
    fork: Option<ForkTools>,
    compact: Option<&std::sync::Arc<crate::compact_tool::CompactSignal>>,
    notices: &crate::AnyNotices,
) -> AgentTools {
    if let Some(fork) = fork {
        return AgentTools::forked(&fork.definitions, fork.terminal, compact, notices);
    }
    if let Some(signal) = compact {
        tools
            .register(crate::compact_tool::CompactTool::new(
                std::sync::Arc::clone(signal),
                notices.clone(),
            ))
            .expect("a tool named `compact` is already registered");
    }
    tools
}

/// What a fork carries over from the main agent besides its context; the
/// default is what an agent that is no fork starts from.
#[derive(Clone, Default)]
struct Inherited {
    budget: crate::context::ContextBudget,
    profile: Option<aither_core::llm::model::Profile>,
    effort: crate::effort::EffortAnchor,
}

impl<Advanced, Balanced, Fast, H, Compactor, Schema> std::fmt::Debug
    for AgentBuilder<Advanced, Balanced, Fast, H, Compactor, Schema>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("tier", &self.settings.tier)
            .field("config", &self.settings.config)
            .finish_non_exhaustive()
    }
}

pub mod sealed {
    /// Gives [`AgentOptions`](super::AgentOptions) the settings it sets.
    /// Private, so only the crate's own builders take the options.
    pub trait Sealed {
        /// The settings the options write.
        fn settings(&mut self) -> &mut super::AgentSettings;
    }
}

/// The agent-level options of [`AgentBuilder`] and
/// [`TerminalAgentBuilder`](crate::TerminalAgentBuilder).
///
/// Both builders implement it over the same [`AgentSettings`], so every
/// option here — and every one added later — is set the same way on either.
/// The builder steps that change a builder's type (its hooks and its models)
/// are methods of each builder.
pub trait AgentOptions: sealed::Sealed + Sized {
    /// Sets which model tier to use for the agent's main reasoning loop.
    ///
    /// This allows creating subagents that use different capability levels:
    /// - `ModelTier::Advanced`: Use the most capable model (default)
    /// - `ModelTier::Balanced`: Use the balanced model (good for subagents)
    /// - `ModelTier::Fast`: Use the fast model (for quick tasks)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an explore subagent using the balanced model
    /// let explore_agent = Agent::builder(opus)
    ///     .balanced_model(sonnet, Parameters::default())
    ///     .fast_model(haiku, Parameters::default())
    ///     .tier(ModelTier::Balanced)  // Use sonnet for reasoning
    ///     .build();
    /// ```
    #[must_use]
    fn tier(mut self, tier: ModelTier) -> Self {
        self.settings().tier = tier;
        self
    }

    /// Sets the identity every tool call of the built agent carries as the
    /// agent making it. An agent is the main agent unless a subagent tool
    /// builds it as a subagent.
    #[must_use]
    fn identity(mut self, identity: aither_core::llm::AgentIdentity) -> Self {
        self.settings().identity = identity;
        self
    }

    /// Renders every alert, notice, reminder and refusal the model reads
    /// through `notices`: the agent's own, those of the tools the builder
    /// registers (`compact`, the terminal controls, `monitor`) and, on a
    /// [`TerminalAgentBuilder`](crate::TerminalAgentBuilder), the terminal's
    /// and its subagents'. [`DefaultNotices`](crate::DefaultNotices) by
    /// default.
    ///
    /// Objects built outside the builder take the same value themselves: a
    /// hand-built [`ToolRegistryBuilder`](aither_sandbox::ToolRegistryBuilder)
    /// through `notices(notices.sandbox())`, and
    /// [`ExternalHooks::with_notices`](crate::ExternalHooks::with_notices).
    /// A refusal a check returns to the embedder, such as a
    /// [`SecretRefusal`](crate::SecretRefusal), is data the embedder renders
    /// with the same value.
    #[must_use]
    fn notices(mut self, notices: crate::AnyNotices) -> Self {
        self.settings().notices = notices;
        self
    }

    /// Publishes every request the built agent sends to `source`, which is
    /// where the `fork` subagent kind starts from.
    ///
    /// Take the source from
    /// [`SubagentTool::fork_source`](crate::specialized::SubagentTool::fork_source).
    #[must_use]
    fn fork_source(mut self, source: crate::fork::ForkSource) -> Self {
        self.settings().fork_source = Some(source);
        self
    }

    /// Sets the system prompt.
    ///
    /// The system prompt is prepended to every conversation and
    /// remains stable for prompt caching.
    #[must_use]
    fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.settings().config.system_prompt = Some(prompt.into());
        self
    }

    /// Enforces the communication and working languages `guard` holds on the
    /// agent's replies and on its working document, and states them in the system
    /// prompt. Give the same guard (a clone) to the tools and harness
    /// commands that take text from the agent, so their refusals reach it as
    /// alerts. Nothing happens while the communication language is unset or
    /// English.
    #[cfg(feature = "languages")]
    #[must_use]
    fn languages(mut self, guard: crate::LanguageGuard) -> Self {
        self.settings().config.languages = Some(guard);
        self
    }

    /// Sets the maximum number of iterations (turns).
    ///
    /// The agent will stop and return an error if this limit is exceeded.
    #[must_use]
    fn max_iterations(mut self, limit: usize) -> Self {
        self.settings().config.max_iterations = limit;
        self
    }

    /// Sets the context management policy: when compaction starts, where the
    /// window pauses, and the budgets compaction and recovery work within.
    #[must_use]
    fn context_policy(mut self, policy: ContextPolicy) -> Self {
        self.settings().config.context_policy = policy;
        self
    }

    /// Registers a source rendered into the tail of every request.
    ///
    /// Sources are rendered in registration order, after the built-in ones,
    /// and each is told when the request it contributed to was sent.
    #[must_use]
    fn context_source<S: ContextSource>(mut self, source: S) -> Self {
        self.settings()
            .context_sources
            .push(AnyContextSource::new(source));
        self
    }

    /// Sets an optional persona overlay prompt.
    #[must_use]
    fn persona_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.settings().config.persona_prompt = Some(prompt.into());
        self
    }

    /// Sets agent kind (coding or chatbot).
    #[must_use]
    fn agent_kind(mut self, kind: AgentKind) -> Self {
        self.settings().config.agent_kind = kind;
        self
    }

    /// Keeps the working document `spec` describes in the agent's sandbox:
    /// it enters the conversation, its rows are held to the events and
    /// items they track, and each agent with a sandbox, a subagent in its
    /// own space, keeps its own. Without it the agent keeps none.
    #[must_use]
    fn working_document(mut self, spec: crate::working_document::DocumentSpec) -> Self {
        self.settings().document.spec = Some(spec);
        self
    }

    /// Resolves the blockers the working document's rows wait on: a blocker
    /// it reports dead no longer holds its row, which is then stale. Without
    /// a resolver every blocker holds.
    #[must_use]
    fn blocker_resolver(mut self, resolver: impl crate::working_document::BlockerResolver) -> Self {
        self.settings().document.blocker_resolver =
            Some(crate::working_document::AnyBlockerResolver::new(resolver));
        self
    }

    /// Reads the issues and pull requests the working document names
    /// through `tracker`: a row naming a finished one is stale, a row whose
    /// issue is blocked by an open one waits, and each root brings its issue
    /// tree into the context.
    #[must_use]
    fn work_tracker(mut self, tracker: impl crate::working_document::WorkTracker) -> Self {
        self.settings().document.tracker =
            Some(crate::working_document::AnyWorkTracker::new(tracker));
        self
    }

    /// Takes the state changes the working document's rows track from
    /// `events`, the inbox the host records its own on: issues, pull
    /// requests and questions. The agent records its background jobs there
    /// itself. Without it the agent keeps an inbox of its own.
    #[must_use]
    fn work_events(mut self, events: crate::working_document::WorkEvents) -> Self {
        self.settings().document.events = events;
        self
    }

    /// Decides through `hook` whether the run ends each time the model ends
    /// its turn. Without one, the run ends unless a context source objects,
    /// whatever the working document holds. On a
    /// [`TerminalAgentBuilder`](crate::TerminalAgentBuilder) its subagents
    /// follow the same hook.
    #[must_use]
    fn stop_hook(mut self, hook: crate::AnyStopHook) -> Self {
        self.settings().stop_hook = Some(hook);
        self
    }

    /// Sets the skill registry used for runtime skill matching and activation.
    #[cfg(feature = "skills")]
    #[must_use]
    fn skill_registry(mut self, registry: Arc<SkillRegistry>) -> Self {
        self.settings().skill_registry = Some(registry);
        self
    }

    /// Lays the system prompt out in `layout`'s order.
    ///
    /// Every system block, the built-in ones included, renders at its slot's
    /// position. Set the layout before adding blocks the default layout does
    /// not place.
    ///
    /// # Panics
    ///
    /// Panics if a block already added has no slot in `layout`.
    #[must_use]
    fn system_layout(mut self, layout: SystemLayout) -> Self {
        self.settings().context.set_layout(layout);
        self
    }

    /// Inserts or replaces the instruction block in `slot`, wrapped in a tag
    /// named after the slot.
    ///
    /// A built-in slot set here keeps the host's text: this is how a host
    /// replaces built-in prose such as [`InstructionSlot::PERMISSIONS`].
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    #[must_use]
    fn system_named(mut self, slot: &InstructionSlot, content: &str) -> Self {
        self.settings()
            .context
            .instructions_mut()
            .insert_tagged(slot, content);
        self
    }

    /// Inserts or replaces the instruction block in `slot` with raw text.
    ///
    /// Unlike [`system_named`](Self::system_named), the `content` is stored
    /// verbatim without any wrapping, for prose blocks (runtime metadata,
    /// environment hints) where a tag adds tokens without meaning.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    #[must_use]
    fn system_text(mut self, slot: &InstructionSlot, content: impl Into<String>) -> Self {
        self.settings()
            .context
            .instructions_mut()
            .insert(slot, content);
        self
    }

    /// Inserts or replaces the external-content block in `slot`: memory, a
    /// repository's instructions, a fetched page or a file, rendered after
    /// every instruction block inside a boundary it cannot close.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no such slot.
    #[must_use]
    fn external_content(mut self, slot: &ExternalSlot, content: ExternalContent) -> Self {
        self.settings().context.external_mut().insert(slot, content);
        self
    }

    /// Sets the full agent configuration.
    #[must_use]
    fn config(mut self, config: AgentConfig) -> Self {
        self.settings().config = config;
        self
    }

    /// Sets the parameters of every request aither sends the model the
    /// builder was created with: the agent's turns, its compactions on that
    /// model, its cache keep-alives and the first request of a fork, plus
    /// the balanced and fast tiers' requests while they run on that model.
    ///
    /// aither sets the fields it owns per request over them — the prompt-
    /// cache controls, structured output on a compaction, and the token cap
    /// of a keep-alive — and sends every other field as given here. A tier
    /// given a model of its own ([`AgentBuilder::balanced_model`],
    /// [`AgentBuilder::fast_model`]) and the
    /// [`compaction_model`](AgentBuilder::compaction_model) take that
    /// model's parameters instead.
    #[must_use]
    fn request_parameters(mut self, parameters: Parameters) -> Self {
        self.settings().parameters.set_shared(parameters);
        self
    }

    /// Overrides the session's prompt-cache controller.
    ///
    /// The default derives a fresh session key per agent, so every rebuild
    /// asks for a new cache shard; a host that rebuilds its agent supplies
    /// the session's identity (and the replaced agent's epoch, through
    /// [`PromptCache::with_epoch`](crate::context::PromptCache::with_epoch))
    /// so the key stays the same across rebuilds. The same value sets the
    /// Claude breakpoint retention, which is one hour by default.
    #[must_use]
    fn prompt_cache(mut self, cache: crate::context::PromptCache) -> Self {
        self.settings().prompt_cache = cache;
        self
    }

    /// Sets the Claude cache retention of the session's prompt cache,
    /// keeping its session key; the cache lifetime follows the retention.
    #[must_use]
    fn claude_cache_ttl(mut self, ttl: aither_core::llm::model::ClaudePromptCacheTtl) -> Self {
        let settings = self.settings();
        settings.prompt_cache = settings.prompt_cache.clone().with_ttl(ttl);
        self
    }

    /// Sets the conversation's cache lifetime, keeping the rest of its
    /// prompt cache. Idle compaction, idle reassembly and keep-alive follow it.
    #[must_use]
    fn cache_lifetime(mut self, lifetime: crate::context::CacheLifetime) -> Self {
        let settings = self.settings();
        settings.prompt_cache = settings.prompt_cache.clone().with_lifetime(lifetime);
        self
    }

    /// Sets the time source idle compaction, idle reassembly and cache
    /// keep-alive read; the system clock by default.
    #[must_use]
    fn clock(mut self, clock: impl crate::Clock) -> Self {
        self.settings().clock = crate::AnyClock::new(clock);
        self
    }

    /// Enables prompt-cache keep-alive while the run waits on its own
    /// background commands.
    ///
    /// `SubagentTool` sets this on child agents; ordinary agents leave it
    /// off. It is a no-op unless the run's model is an Anthropic one.
    #[must_use]
    fn cache_keep_alive(mut self, keep_alive: crate::config::CacheKeepAlive) -> Self {
        self.settings().config.cache_keep_alive = Some(keep_alive);
        self
    }

    /// Sets what a run does with background commands still running when the
    /// model ends its turn; [`BackgroundJobPolicy::Await`] by default.
    ///
    /// [`BackgroundJobPolicy::Await`]: crate::BackgroundJobPolicy::Await
    #[must_use]
    fn background_jobs(mut self, policy: crate::BackgroundJobPolicy) -> Self {
        self.settings().config.background_jobs = policy;
        self
    }
}

impl<Advanced, Balanced, Fast, H, Compactor, Schema> sealed::Sealed
    for AgentBuilder<Advanced, Balanced, Fast, H, Compactor, Schema>
{
    fn settings(&mut self) -> &mut AgentSettings {
        &mut self.settings
    }
}

impl<Advanced, Balanced, Fast, H, Compactor, Schema> AgentOptions
    for AgentBuilder<Advanced, Balanced, Fast, H, Compactor, Schema>
{
}

impl<LLM: LanguageModel + Clone> AgentBuilder<LLM, LLM, LLM, (), LLM> {
    /// Creates a new agent builder with default configuration.
    ///
    /// All model tiers (advanced/balanced/fast) use the same model.
    pub fn new(llm: LLM) -> Self {
        Self {
            advanced: llm.clone(),
            balanced: llm.clone(),
            fast: llm,
            compaction_model: None,
            hooks: (),
            settings: AgentSettings::default(),
            wiring: Wiring::default(),
            handoff_schema: PhantomData,
        }
    }
}

impl<Advanced, Balanced, Fast, H, Compactor, Schema>
    AgentBuilder<Advanced, Balanced, Fast, H, Compactor, Schema>
where
    Advanced: LanguageModel,
    Balanced: LanguageModel,
    Fast: LanguageModel,
    Compactor: LanguageModel,
    H: Hook,
    Schema: HandoffSchema,
{
    /// Sets the handoff schema every compaction asks the model for, in place
    /// of [`HandoffDocument`].
    ///
    /// The schema's type is the structured output the model fills; its
    /// [`HandoffSchema`] implementation gives the instruction, the retry,
    /// the validation and the text the conversation reads after the handoff.
    pub fn handoff<S2: HandoffSchema>(
        self,
    ) -> AgentBuilder<Advanced, Balanced, Fast, H, Compactor, S2> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            compaction_model: self.compaction_model,
            hooks: self.hooks,
            settings: self.settings,
            wiring: self.wiring,
            handoff_schema: PhantomData,
        }
    }

    /// Sets the balanced model for moderate tasks (e.g., subagents), and the
    /// parameters of the requests aither sends it.
    ///
    /// This model is used when spawning subagents that don't need
    /// the full capabilities of the advanced model.
    pub fn balanced_model<B2: LanguageModel>(
        mut self,
        model: B2,
        parameters: Parameters,
    ) -> AgentBuilder<Advanced, B2, Fast, H, Compactor, Schema> {
        self.settings.parameters.set_balanced(parameters);
        AgentBuilder {
            advanced: self.advanced,
            balanced: model,
            fast: self.fast,
            compaction_model: self.compaction_model,
            hooks: self.hooks,
            settings: self.settings,
            wiring: self.wiring,
            handoff_schema: PhantomData,
        }
    }

    /// Sets the fast model for quick tasks (e.g., translating a reply), and
    /// the parameters of the requests aither sends it.
    ///
    /// This model is used for tasks where speed and cost matter more
    /// than capability.
    pub fn fast_model<F2: LanguageModel>(
        mut self,
        model: F2,
        parameters: Parameters,
    ) -> AgentBuilder<Advanced, Balanced, F2, H, Compactor, Schema> {
        self.settings.parameters.set_fast(parameters);
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: model,
            compaction_model: self.compaction_model,
            hooks: self.hooks,
            settings: self.settings,
            wiring: self.wiring,
            handoff_schema: PhantomData,
        }
    }

    /// Sets the model compaction runs on, instead of the session's own.
    ///
    /// Without one, compaction reads the conversation's prompt cache on the
    /// session's model, and an idle session is compacted shortly before the
    /// cache expires ([`Agent::compact_when_idle`]). A different model cannot
    /// read that cache, so with one set compaction requests go to it, idle
    /// compaction never runs, and an idle gap past the cache lifetime
    /// reassembles the window instead.
    ///
    /// Compaction requests to it carry `parameters`, the model's own, in
    /// place of the session model's, under the cache controls and structured
    /// output aither sets on every compaction.
    pub fn compaction_model<C2: LanguageModel>(
        self,
        model: C2,
        parameters: Parameters,
    ) -> AgentBuilder<Advanced, Balanced, Fast, H, C2, Schema> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            compaction_model: Some(WithParameters {
                model,
                parameters: Box::new(parameters),
            }),
            hooks: self.hooks,
            settings: self.settings,
            wiring: self.wiring,
            handoff_schema: PhantomData,
        }
    }

    /// Adds a hook to intercept agent operations.
    ///
    /// Hooks are composed using the `HCons` pattern, allowing multiple
    /// hooks to be chained at compile time.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder(llm)
    ///     .hook(LoggingHook)
    ///     .hook(ConfirmationHook)
    ///     .build();
    /// // Type: Agent<LLM, HCons<ConfirmationHook, HCons<LoggingHook, ()>>>
    /// ```
    pub fn hook<NH: Hook>(
        self,
        hook: NH,
    ) -> AgentBuilder<Advanced, Balanced, Fast, HCons<NH, H>, Compactor, Schema> {
        AgentBuilder {
            advanced: self.advanced,
            balanced: self.balanced,
            fast: self.fast,
            compaction_model: self.compaction_model,
            hooks: HCons::new(hook, self.hooks),
            settings: self.settings,
            wiring: self.wiring,
            handoff_schema: PhantomData,
        }
    }

    /// Registers an eager (always-loaded) tool.
    ///
    /// Eager tools are included in every LLM request.
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.wiring
            .tools
            .register(tool)
            .expect("tool registration must succeed: duplicate name or missing description");
        self
    }

    /// Registers a dynamic terminal tool (type-erased).
    ///
    /// This is used for child terminal capability bundles in subagents where the
    /// concrete terminal type is not known at compile time.
    ///
    /// # Panics
    ///
    /// Panics if the `knowledge_and_time` prompt template fails to render.
    pub fn dyn_terminal(mut self, dyn_tool: aither_sandbox::DynTerminalTool) -> Self {
        let instructions = self.settings.context.instructions_mut();
        instructions.insert(
            &InstructionSlot::PREPARED_TOOLS,
            dyn_tool
                .prepared_tools()
                .render()
                .expect("prepared tool catalog must render"),
        );
        // Seed knowledge/time guidance against the shared command registry so
        // subagents are not told to run commands that were never registered
        // (e.g. `websearch` when a provider-hosted search tool is active).
        if !instructions.contains(&InstructionSlot::KNOWLEDGE_AND_TIME) {
            let websearch_command = dyn_tool.has_command("websearch");
            instructions.insert_tagged(
                &InstructionSlot::KNOWLEDGE_AND_TIME,
                &crate::agent::KnowledgeAndTimeTemplate { websearch_command }
                    .render()
                    .expect("failed to render knowledge_and_time template"),
            );
        }
        let wiring = &mut self.wiring;
        wiring.background_receiver = Some(dyn_tool.background_receiver());
        wiring.permission_receiver = Some(dyn_tool.permission_receiver());
        wiring.kept_receiver = Some(dyn_tool.kept_receiver());
        wiring.job_registry = Some(dyn_tool.job_registry());
        // Files are restored after compaction, and the working document is kept, in
        // wherever the terminal works: a subagent's are in its own space,
        // never its parent's.
        wiring.restore_dir = Some(dyn_tool.working_dir().clone());
        wiring.sandbox_dir = Some(dyn_tool.working_dir().clone());
        wiring.tools.register_dyn_terminal(dyn_tool);
        self
    }

    /// Registers the native `compact` tool and arms the pre-compaction alert.
    ///
    /// The model calls `compact` to start an asynchronous compaction at the
    /// next request boundary, through the same fork and apply path as
    /// automatic compaction; the call returns at once. When usage comes
    /// within [`ContextPolicy::warn_distance`](crate::ContextPolicy::warn_distance)
    /// of the auto-compact point, one alert tells the model to save its work
    /// and call `compact` itself.
    ///
    /// # Panics
    ///
    /// Panics if a tool named `compact` is already registered.
    pub fn compact_tool(mut self) -> Self {
        assert!(
            self.wiring.compact_signal.is_none()
                && !self
                    .wiring
                    .tools
                    .definitions()
                    .iter()
                    .any(|definition| definition.name() == crate::compact_tool::COMPACT_TOOL_NAME),
            "a tool named `compact` is already registered"
        );
        self.wiring.compact_signal = Some(std::sync::Arc::default());
        self
    }

    /// Makes the built agent a fork of the main agent that sent `fork`.
    ///
    /// The agent starts from the main agent's request: its window, with the
    /// live tail as the last message, its tool definitions as the model saw
    /// them, its request parameters and its prompt-cache identity, so its
    /// first request is the main agent's followed by the prompt it is given
    /// and reads the main agent's cache. Nothing is initialized on top of
    /// that, since initialization would rewrite the window.
    ///
    /// The tools run in `terminal`, the fork's own terminal. A tool of the
    /// main agent that `terminal` does not provide stays in the definitions,
    /// to keep the request identical, and refuses when called. The exception
    /// is `compact`, which compacts the fork through a signal of its own.
    pub fn forked_from(
        mut self,
        fork: &crate::fork::ForkSnapshot,
        terminal: aither_sandbox::DynTerminalTool,
    ) -> Self {
        let settings = &mut self.settings;
        settings.context = fork.context.clone();
        settings.prompt_cache = fork.prompt_cache.clone();
        settings
            .parameters
            .set_shared(fork.request_parameters.clone());
        settings.config.context_policy = fork.context_policy;
        let wiring = &mut self.wiring;
        wiring.inherited = Some(Inherited {
            budget: fork.budget,
            profile: Some(fork.profile.clone()),
            effort: fork.effort,
        });
        wiring.background_receiver = Some(terminal.background_receiver());
        wiring.permission_receiver = Some(terminal.permission_receiver());
        wiring.kept_receiver = Some(terminal.kept_receiver());
        wiring.job_registry = Some(terminal.job_registry());
        // A fork keeps its own working document in its own space, like any
        // subagent, not the main agent's it inherited the context of.
        wiring.restore_dir = Some(terminal.working_dir().clone());
        wiring.sandbox_dir = Some(terminal.working_dir().clone());
        // The inherited `compact` definition runs over the fork's own
        // signal, so a call compacts the fork, never the main agent.
        wiring.compact_signal = fork
            .definitions
            .iter()
            .any(|definition| definition.name() == crate::compact_tool::COMPACT_TOOL_NAME)
            .then(std::sync::Arc::default);
        wiring.fork_tools = Some(ForkTools {
            definitions: fork.definitions.clone(),
            terminal,
        });
        self
    }

    /// Sets the main agent's sandbox directory: its working documents are
    /// supervised and its files restored after compaction.
    pub fn sandbox_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        let path = path.into();
        self.wiring.restore_dir = Some(path.clone());
        self.wiring.sandbox_dir = Some(path);
        self
    }

    /// What renders the built agent's notices, as the options set it so far.
    pub(crate) const fn chosen_notices(&self) -> &crate::AnyNotices {
        &self.settings.notices
    }

    /// The working document the options settled on, which the subagents
    /// keep too.
    pub(crate) const fn chosen_document(&self) -> Option<&crate::working_document::DocumentSpec> {
        self.settings.document.spec.as_ref()
    }

    /// The stop rule the options settled on, which the subagents follow too.
    pub(crate) const fn chosen_stop_hook(&self) -> Option<&crate::AnyStopHook> {
        self.settings.stop_hook.as_ref()
    }

    /// Whether the instruction block in `slot` is already set.
    pub(crate) fn has_instruction(&self, slot: &InstructionSlot) -> bool {
        self.settings.context.instructions().contains(slot)
    }

    /// Lists `commands` in [`InstructionSlot::COMMANDS`] and has the built
    /// agent keep that table current as live content.
    ///
    /// # Panics
    ///
    /// Panics if the layout has no [`InstructionSlot::COMMANDS`] slot.
    pub(crate) fn command_catalog(
        mut self,
        commands: crate::command_catalog::CommandCatalog,
    ) -> Self {
        self.settings
            .context
            .update_live(&crate::context::LiveContent::COMMANDS, |update| {
                update.instruction(&InstructionSlot::COMMANDS, commands.render());
            });
        self.wiring.commands = Some(commands);
        self
    }

    /// Registers an MCP connection.
    ///
    /// All tools from the MCP server will be available for the agent to use.
    /// You can call this method multiple times to register multiple MCP servers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use aither_mcp::McpConnection;
    ///
    /// // Connect to an MCP server
    /// let conn = McpConnection::spawn(&executor, "npx", &["-y", "@modelcontextprotocol/server-filesystem", "/"]).await?;
    ///
    /// let agent = Agent::builder(llm)
    ///     .mcp(conn)
    ///     .build();
    /// ```
    ///
    /// # Multiple Servers
    ///
    /// ```rust,ignore
    /// let filesystem = McpConnection::spawn(&executor, "npx", &["-y", "@modelcontextprotocol/server-filesystem", "/"]).await?;
    /// let github = McpConnection::spawn(&executor, "npx", &["-y", "@modelcontextprotocol/server-github"]).await?;
    ///
    /// let agent = Agent::builder(llm)
    ///     .mcp(filesystem)
    ///     .mcp(github)
    ///     .build();
    /// ```
    ///
    /// # Loading from Configuration
    ///
    /// ```rust,ignore
    /// use aither_mcp::{McpConnection, McpServersConfig};
    ///
    /// let config: McpServersConfig = serde_json::from_str(&config_json)?;
    /// let connections = McpConnection::from_configs(&executor, &config).await?;
    ///
    /// let mut builder = Agent::builder(llm);
    /// for (_name, conn) in connections {
    ///     builder = builder.mcp(conn);
    /// }
    /// let agent = builder.build();
    /// ```
    #[cfg(feature = "mcp")]
    pub fn mcp(mut self, conn: McpConnection) -> Self {
        self.wiring.tools.register_mcp(conn);
        self
    }

    /// Registers the terminal tool for command execution in a sandbox.
    ///
    /// The terminal tool enables command execution with configurable permission modes
    /// (sandboxed, network, unsafe). It creates its own working directory with
    /// four random words and manages output storage internally.
    ///
    /// This also captures the background task receiver for polling completed
    /// background tasks during the agent loop.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use aither_sandbox::{TerminalTool, ToolRegistryBuilder, permission::DenyUnsafe};
    /// use std::sync::Arc;
    ///
    /// // Create terminal tool (creates random working dir like amber-forest-thunder-pearl/)
    /// let terminal_tool = TerminalTool::new_in(parent, DenyUnsafe, executor).await?;
    /// let registry = Arc::new(ToolRegistryBuilder::new().build(terminal_tool.outputs_dir()));
    /// let terminal_tool = terminal_tool.with_registry(registry);
    ///
    /// let agent = Agent::builder(llm)
    ///     .terminal(terminal_tool)
    ///     .build();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn terminal<P, E>(
        mut self,
        terminal_tool: aither_sandbox::TerminalTool<P, E, aither_sandbox::Configured>,
    ) -> Self
    where
        P: aither_sandbox::PermissionHandler + 'static,
        E: executor_core::Executor + Clone + 'static,
    {
        self.settings.context.instructions_mut().insert(
            &InstructionSlot::PREPARED_TOOLS,
            terminal_tool
                .prepared_tools_context()
                .expect("terminal tools must be initialized before agent construction"),
        );
        let wiring = &mut self.wiring;
        wiring.background_receiver = Some(terminal_tool.background_receiver());
        wiring.permission_receiver = Some(terminal_tool.permission_receiver());
        wiring.kept_receiver = Some(terminal_tool.kept_receiver());
        wiring.job_registry = Some(terminal_tool.job_registry());
        wiring.sandbox_dir = Some(terminal_tool.working_dir().clone());
        wiring.restore_dir = Some(terminal_tool.working_dir().clone());
        wiring
            .tools
            .register(terminal_tool)
            .expect("tool registration must succeed: duplicate name or missing description");
        self
    }

    /// Ends a provider response stream in flight when the machine wakes from
    /// sleep or loses its network.
    ///
    /// The attempt fails with an [`AgentError::Llm`](crate::AgentError::Llm)
    /// whose failure is
    /// [`FailureKind::Network`](aither_core::llm::FailureKind::Network), so the
    /// caller retries instead of sitting on a dead socket. Nothing about the
    /// change reaches the agent's context.
    #[cfg(feature = "connectivity")]
    pub fn connectivity(mut self, connectivity: aither_connectivity::Connectivity) -> Self {
        self.wiring.connectivity = crate::connectivity::Watch::new(connectivity);
        self
    }

    /// Starts a [`HostLoadMonitor`](crate::HostLoadMonitor) on `executor` and
    /// tells the agent, once per event, when machine load becomes high and when
    /// it ends.
    ///
    /// With `memory`, the same readings drive the agent's
    /// [`RunawayGuard`](crate::RunawayGuard): memory warnings listing its
    /// jobs, a final warning before a runaway job is killed, and the kill.
    ///
    /// The monitor, which may also watch the agent's volume
    /// ([`with_disk`](crate::HostLoadMonitor::with_disk)), runs until the
    /// agent is dropped. It reads the builder's
    /// [`clock`](AgentOptions::clock), so set the clock first.
    pub fn host_load<R, E>(
        mut self,
        executor: &E,
        monitor: crate::HostLoadMonitor<R>,
        memory: Option<aither_sandbox::JobMemory>,
    ) -> Self
    where
        R: crate::HostReader,
        E: executor_core::Executor,
    {
        self.wiring.host_load_events = Some(crate::host_load::spawn(
            executor,
            monitor,
            memory.map(crate::RunawayGuard::new),
            self.settings.clock.clone(),
        ));
        self
    }

    /// Connects the agent to an external background-task channel pair.
    ///
    /// [`terminal`](Self::terminal) and [`dyn_terminal`](Self::dyn_terminal)
    /// wire these for sandboxed execution; hosts driving an execution
    /// environment of their own (or simulating one) attach the pair here so
    /// the run loop waits on it the same way.
    pub fn background_channels(
        mut self,
        receiver: BackgroundTaskReceiver,
        job_registry: JobRegistry,
    ) -> Self {
        self.wiring.background_receiver = Some(receiver);
        self.wiring.job_registry = Some(job_registry);
        self
    }

    /// Returns the definitions of the tools registered on this builder so far.
    ///
    /// These are the definitions the built agent sends to the model, in the
    /// order the tool registry holds them.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        if let Some(fork) = &self.wiring.fork_tools {
            return fork.definitions.clone();
        }
        let mut definitions = self.wiring.tools.active_definitions();
        if let Some(signal) = &self.wiring.compact_signal {
            definitions.push(ToolDefinition::new(&crate::compact_tool::CompactTool::new(
                std::sync::Arc::clone(signal),
                self.settings.notices.clone(),
            )));
            definitions.sort_by(|a, b| a.name().cmp(b.name()));
        }
        definitions
    }

    /// Builds the agent.
    pub fn build(self) -> Agent<Advanced, Balanced, Fast, H, Compactor, Schema> {
        let AgentSettings {
            parameters,
            tier,
            config,
            context,
            prompt_cache,
            clock,
            context_sources,
            document,
            stop_hook,
            fork_source,
            identity,
            notices,
            #[cfg(feature = "skills")]
            skill_registry,
        } = self.settings;
        let Wiring {
            tools,
            connectivity,
            background_receiver,
            permission_receiver,
            kept_receiver,
            commands,
            host_load_events,
            job_registry,
            sandbox_dir,
            restore_dir,
            compact_signal,
            inherited,
            fork_tools,
        } = self.wiring;
        let tools = assemble_tools(tools, fork_tools, compact_signal.as_ref(), &notices);
        // A fork starts initialized, from what the main agent carried over.
        let initialized = inherited.is_some();
        let Inherited {
            budget,
            profile,
            effort,
        } = inherited.unwrap_or_default();
        Agent {
            warmth: crate::CacheWarmth::new(prompt_cache.lifetime(), clock),
            advanced: std::sync::Arc::new(self.advanced),
            balanced: std::sync::Arc::new(self.balanced),
            fast: std::sync::Arc::new(self.fast),
            compaction_model: self.compaction_model.map(|compactor| WithParameters {
                model: std::sync::Arc::new(compactor.model),
                parameters: compactor.parameters,
            }),
            parameters,
            idle_compacted: false,
            session: crate::agent::SessionPhase::NotStarted,
            tier,
            tools,
            hooks: self.hooks,
            budget,
            prompt_cache,
            context_sources,
            reassembled: false,
            prefix_rewrites: crate::context::PrefixRewrites::default(),
            requests_sent: 0,
            compaction: crate::compaction_state::CompactionState::default(),
            compaction_failure: None,
            tool_results: std::collections::VecDeque::new(),
            config,
            context,
            initialized,
            effort,
            profile,
            compaction_profile: None,
            background_receiver,
            permission_receiver,
            kept_receiver,
            commands,
            host_load_events,
            job_registry,
            detached_task_ids: std::collections::HashSet::new(),
            // Every agent with a sandbox, a fork included, keeps its own
            // working document there.
            working_document: document.document(sandbox_dir.as_deref(), notices.clone()),
            sandbox_dir,
            work_events: document.events,
            stop_hook,
            restore_dir,
            fork_source,
            compact_signal,
            compaction_warning: crate::agent::CompactionWarning::Armed,
            kept_exchanges: std::collections::HashSet::new(),
            identity,
            notices,
            connectivity,
            last_request: None,
            cache_keep_alive: crate::agent::CacheKeepAliveState::default(),
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry,
            #[cfg(feature = "skills")]
            active_skills: crate::context::ActiveSkills::new(),
            #[cfg(feature = "skills")]
            active_allowed_tools: None,
            handoff_schema: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[cfg(feature = "skills")]
    use aither_skills::{Skill, SkillRegistry};
    use schemars::JsonSchema;
    use serde::Deserialize;

    use futures_core::Stream;

    // Mock error type for testing
    #[derive(Debug)]
    struct MockError;

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock error")
        }
    }

    impl std::error::Error for MockError {}

    impl aither_core::llm::Classify for MockError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    // Mock LLM for testing
    #[derive(Clone)]
    struct MockLlm;

    impl LanguageModel for MockLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: aither_core::llm::LLMRequest,
        ) -> impl Stream<Item = Result<aither_core::llm::Event, Self::Error>> + Send {
            futures_lite::stream::empty()
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<
            Output = Result<aither_core::llm::model::Profile, Self::Error>,
        > + Send {
            std::future::ready(Ok(aither_core::llm::model::Profile::new(
                "mock",
                "test",
                "mock-model",
                "A mock model for testing",
                100_000,
            )))
        }
    }

    // Mock tool
    struct MockTool;

    /// Does nothing, for tests that only care that a tool was registered.
    #[derive(Debug, JsonSchema, Deserialize)]
    struct MockArgs;

    impl Tool for MockTool {
        fn name(&self) -> Cow<'static, str> {
            "mock_tool".into()
        }

        type Arguments = MockArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::text("ok")))
        }
    }

    // Mock hook
    struct MockHook;

    impl Hook for MockHook {}

    #[test]
    fn test_builder_basic() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert_eq!(agent.tools.definitions(), []);
    }

    #[test]
    fn public_model_tier_configures_agent_builder() {
        let agent = AgentBuilder::new(MockLlm)
            .tier(crate::ModelTier::Balanced)
            .build();
        assert_eq!(agent.tier, crate::ModelTier::Balanced);
    }

    #[test]
    fn test_builder_with_tool() {
        let agent = AgentBuilder::new(MockLlm).tool(MockTool).build();
        assert_eq!(agent.tools.definitions().len(), 1);
    }

    #[test]
    fn test_builder_with_system_prompt() {
        let agent = AgentBuilder::new(MockLlm)
            .system_prompt("You are helpful.")
            .build();
        assert_eq!(
            agent.config.system_prompt,
            Some("You are helpful.".to_string())
        );
    }

    #[test]
    fn test_builder_with_hook() {
        let _agent = AgentBuilder::new(MockLlm).hook(MockHook).build();
        // Type check: agent has HCons<MockHook, ()> as hook type
    }

    #[test]
    fn test_builder_with_multiple_hooks() {
        let _agent = AgentBuilder::new(MockLlm)
            .hook(MockHook)
            .hook(MockHook)
            .build();
        // Type check: agent has HCons<MockHook, HCons<MockHook, ()>>
    }

    /// Options set before a step that changes the builder's type survive it.
    #[test]
    fn options_survive_the_steps_that_change_the_builders_type() {
        let policy = ContextPolicy {
            safe_buffer: 1_234,
            ..ContextPolicy::default()
        };
        let agent = AgentBuilder::new(MockLlm)
            .context_policy(policy)
            .max_iterations(7)
            .hook(MockHook)
            .balanced_model(MockLlm, Parameters::default())
            .fast_model(MockLlm, Parameters::default())
            .compaction_model(MockLlm, Parameters::default())
            .build();
        assert_eq!(agent.config.context_policy, policy);
        assert_eq!(agent.config.max_iterations, 7);
    }

    #[test]
    fn test_builder_max_iterations() {
        let agent = AgentBuilder::new(MockLlm).max_iterations(100).build();
        assert_eq!(agent.config.max_iterations, 100);
    }

    #[test]
    fn test_builder_default_config() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert_eq!(
            agent.config.max_iterations,
            AgentConfig::default().max_iterations
        );
    }

    #[cfg(feature = "skills")]
    #[test]
    fn test_builder_preserves_skill_registry() {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: std::collections::HashMap::new(),
        });

        let agent = AgentBuilder::new(MockLlm)
            .skill_registry(Arc::new(registry))
            .build();
        assert!(agent.skill_registry.is_some());
    }

    #[cfg(feature = "skills")]
    #[test]
    fn test_runtime_skill_activation_is_disabled() {
        let mut registry = SkillRegistry::new();
        registry.register(Skill {
            name: "code-review".to_string(),
            description: "Review code carefully".to_string(),
            instructions: "Use a review checklist.".to_string(),
            allowed_tools: Some(vec!["mock_tool".to_string()]),
            resources: std::collections::HashMap::new(),
        });

        let mut agent = AgentBuilder::new(MockLlm)
            .skill_registry(Arc::new(registry))
            .build();
        let events = agent.activate_skills_for_prompt("please review this patch");

        assert!(events.is_empty());
        assert!(agent.active_skills.is_empty());
        assert!(agent.active_allowed_tools.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_keeps_its_working_documents_and_files_in_its_own_space() {
        use aither_sandbox::permission::NoopPermissionHandler;
        use aither_sandbox::{SubagentGrant, TerminalTool, ToolRegistryBuilder};
        use executor_core::tokio::TokioGlobal;

        let source = tempfile::tempdir().unwrap();
        let cache = crate::test_support::TempDir::new();
        let session = tempfile::tempdir().unwrap();
        let binary = crate::test_support::compile_baseline_fixture(source.path()).await;
        let distribution = crate::test_support::fixture_distribution(&binary, "1.0.0").await;
        let terminal = TerminalTool::new_exact(session.path(), NoopPermissionHandler, TokioGlobal)
            .await
            .unwrap()
            .initialize_tools(&distribution, cache.path())
            .await
            .unwrap();
        let registry =
            std::sync::Arc::new(ToolRegistryBuilder::new().build(terminal.outputs_dir()));
        let terminal = terminal.with_registry(registry);
        let subagent = terminal
            .binding()
            .launcher()
            .create(SubagentGrant::new(), None, None, None)
            .await
            .unwrap();
        let space = subagent.working_dir().clone();

        let main_dir = terminal.working_dir().clone();
        let main = AgentBuilder::new(MockLlm).dyn_terminal(terminal.clone().to_dyn());
        assert_eq!(
            main.wiring.sandbox_dir.as_ref(),
            Some(&main_dir),
            "the main agent keeps its working documents in its sandbox"
        );
        let main = AgentBuilder::new(MockLlm).terminal(terminal);
        assert_eq!(main.wiring.sandbox_dir.as_ref(), Some(&main_dir));
        let child = AgentBuilder::new(MockLlm).dyn_terminal(subagent);
        assert_ne!(space, main_dir);
        assert_eq!(
            child.wiring.sandbox_dir.as_ref(),
            Some(&space),
            "a subagent's working document is in its own space, not its parent's"
        );
        assert_eq!(
            child.wiring.restore_dir,
            Some(space),
            "a subagent's files are restored after compaction"
        );
    }
}
