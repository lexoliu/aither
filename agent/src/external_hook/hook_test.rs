//! The hook-test command: runs a configured hook against sample inputs and
//! reports what it decided, and whether it crashed or timed out.

use std::borrow::Cow;
use std::sync::Arc;

use aither_core::llm::ToolContext;
use aither_core::llm::tool::{Tool, ToolResult};
use aither_sandbox::TerminalMode;
use aither_sandbox::permission::PermissionRequest;
use askama::Template;
use futures::StreamExt as _;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::command::{CommandHookEvent, CommandHookInput, CommandHookOutput};
use super::lifecycle::{
    AfterCompaction, AskUser, BeforeCompaction, CommandCalled, LifecycleKind, LifecycleOutput,
    SessionStart, Stop, SubagentStart, SubagentStop, UserMessage,
};
use super::live::{Entry, HookConfig, HookKind, HookSet, LiveHooks};
use super::reviewer::{ReviewAnswer, ReviewRequest};
use super::runner::{HookCommand, HookRole, HookRunError, HookRunner};

/// How many samples run at once.
const SAMPLE_CONCURRENCY: usize = 4;

/// Arguments of `hook_test`.
///
/// Name the hook and give samples of the kind it takes: `--command` for a
/// command hook, `--review-command` and `--review-domain` for a reviewer,
/// `--event` for a lifecycle hook.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct HookTestArgs {
    /// The name of the configured hook to test. It runs in its current
    /// version, exactly as a session would run it.
    pub hook: String,
    /// The agent the samples come from: `main` (the default) or `subagent`.
    #[serde(default)]
    pub role: Option<HookRole>,
    /// Sample shell commands, for a command hook.
    #[serde(default)]
    pub command: Vec<String>,
    /// Sample scripts that need the unsafe mode, for a reviewer.
    #[serde(default)]
    pub review_command: Vec<String>,
    /// Sample network domains as `host:port`, for a reviewer.
    #[serde(default)]
    pub review_domain: Vec<String>,
    /// Sample events as JSON objects such as
    /// `{"event":"user_message","message":"hi","attachments":0}`, for a
    /// lifecycle hook; the role is added.
    #[serde(default)]
    pub event: Vec<EventSample>,
}

/// A sample lifecycle event: its name and the fields that event carries.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventSample {
    /// See [`LifecycleKind::SessionStart`].
    SessionStart(SessionStart),
    /// See [`LifecycleKind::BeforeCompaction`].
    BeforeCompaction(BeforeCompaction),
    /// See [`LifecycleKind::AfterCompaction`].
    AfterCompaction(AfterCompaction),
    /// See [`LifecycleKind::Stop`].
    Stop(Stop),
    /// See [`LifecycleKind::UserMessage`].
    UserMessage(UserMessage),
    /// See [`LifecycleKind::SubagentStart`].
    SubagentStart(SubagentStart),
    /// See [`LifecycleKind::SubagentStop`].
    SubagentStop(SubagentStop),
    /// See [`LifecycleKind::AskUser`].
    AskUser(AskUser),
    /// See [`LifecycleKind::CommandCall`].
    CommandCall(CommandCalled),
}

impl EventSample {
    const fn kind(&self) -> LifecycleKind {
        match self {
            Self::SessionStart(_) => LifecycleKind::SessionStart,
            Self::BeforeCompaction(_) => LifecycleKind::BeforeCompaction,
            Self::AfterCompaction(_) => LifecycleKind::AfterCompaction,
            Self::Stop(_) => LifecycleKind::Stop,
            Self::UserMessage(_) => LifecycleKind::UserMessage,
            Self::SubagentStart(_) => LifecycleKind::SubagentStart,
            Self::SubagentStop(_) => LifecycleKind::SubagentStop,
            Self::AskUser(_) => LifecycleKind::AskUser,
            Self::CommandCall(_) => LifecycleKind::CommandCall,
        }
    }
}

/// Why a hook cannot be tested.
#[derive(Debug, thiserror::Error)]
pub enum HookTestError {
    /// No sample was given.
    #[error("give at least one sample: --command, --review-command, --review-domain or --event")]
    NoSamples,
    /// Samples of different kinds of hook were given.
    #[error("the samples are for more than one kind of hook")]
    MixedKinds,
    /// No hook of the kind the samples are for has the name.
    #[error("there is no {kind} hook named `{name}`")]
    UnknownHook {
        /// The kind the samples are for.
        kind: HookKind,
        /// The name asked for.
        name: String,
    },
    /// A domain is not `host:port`.
    #[error("`{0}` is not a domain as host:port")]
    Domain(String),
    /// The hook exists but has no version that passed its check, so there
    /// is nothing to test.
    #[error("the {kind} hook `{name}` has no version in use: {reason}")]
    NoVersion {
        /// The kind of the hook.
        kind: HookKind,
        /// Its name.
        name: String,
        /// Why its script was refused.
        reason: String,
    },
}

/// One input a hook is run with.
#[derive(Debug)]
enum Sample {
    Command(String),
    Script(String),
    Domain { domain: String, port: u16 },
    Event(EventSample),
}

/// A sample event as a lifecycle hook reads it: the event with its fields,
/// and the role.
#[derive(Serialize)]
struct SampleEvent<'a> {
    #[serde(flatten)]
    sample: &'a EventSample,
    role: HookRole,
}

/// What came of running a hook on a sample.
#[derive(Debug)]
enum Observation {
    Allowed,
    Blocked(String),
    Approved,
    Denied(String),
    Passed,
    Context(String),
    NoContext,
    NotSubscribed,
    /// The hook crashed, or printed something that is not an answer.
    Crashed(String),
    TimedOut {
        deadline_ms: u64,
    },
}

struct Tested {
    sample: Sample,
    observation: Observation,
}

#[derive(Template)]
#[template(path = "hook_test_report.txt", escape = "none")]
struct ReportTemplate<'a> {
    hook: &'a str,
    kind: HookKind,
    /// Why the newest text of the hook is not what ran, when it is not.
    stale: Option<&'a str>,
    results: &'a [Tested],
    crashed: usize,
    timed_out: usize,
}

fn observe<T>(
    outcome: Result<Option<T>, HookRunError>,
    deadline: &HookCommand,
    decided: impl FnOnce(Option<T>) -> Observation,
) -> Observation {
    match outcome {
        Ok(answer) => decided(answer),
        Err(HookRunError::Timeout { .. }) => Observation::TimedOut {
            deadline_ms: deadline.deadline_ms,
        },
        Err(error) => Observation::Crashed(error.to_string()),
    }
}

/// Tests a configured hook from a session, as the `hook_test` command.
///
/// Register it on the registry of a terminal with
/// `ToolRegistryBuilder::configure_tool`, so the model and the user can run
/// `hook_test` from the terminal. The hook runs on its own, without the
/// fallback to a previous version or the pause that a failing hook meets in a
/// session, so that its crashes are reported.
#[derive(Debug, Clone)]
pub struct HookTestTool {
    hooks: LiveHooks,
}

impl HookTestTool {
    /// The command testing the hooks of `hooks`.
    #[must_use]
    pub const fn new(hooks: LiveHooks) -> Self {
        Self { hooks }
    }

    fn samples(args: &HookTestArgs) -> Result<(HookKind, Vec<Sample>), HookTestError> {
        let mut samples = Vec::new();
        let mut kinds = Vec::new();
        for command in &args.command {
            kinds.push(HookKind::Command);
            samples.push(Sample::Command(command.clone()));
        }
        for script in &args.review_command {
            kinds.push(HookKind::Reviewer);
            samples.push(Sample::Script(script.clone()));
        }
        for domain in &args.review_domain {
            kinds.push(HookKind::Reviewer);
            let (host, port) = domain
                .rsplit_once(':')
                .and_then(|(host, port)| Some((host, port.parse().ok()?)))
                .ok_or_else(|| HookTestError::Domain(domain.clone()))?;
            samples.push(Sample::Domain {
                domain: host.to_string(),
                port,
            });
        }
        for event in &args.event {
            kinds.push(HookKind::Lifecycle);
            samples.push(Sample::Event(event.clone()));
        }
        let kind = *kinds.first().ok_or(HookTestError::NoSamples)?;
        if kinds.iter().any(|other| *other != kind) {
            return Err(HookTestError::MixedKinds);
        }
        Ok((kind, samples))
    }
}

impl Tool for HookTestTool {
    fn name(&self) -> Cow<'static, str> {
        "hook_test".into()
    }

    type Arguments = HookTestArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let (kind, samples) = Self::samples(&args)?;
        let role = args.role.unwrap_or(HookRole::Main);
        let target = Target::find(&self.hooks.set(), kind, &args.hook)?;
        let (runner, target) = (self.hooks.runner(), &target);
        let results: Vec<Tested> = futures::stream::iter(samples)
            .map(|sample| run_sample(runner, target, role, sample))
            .buffered(SAMPLE_CONCURRENCY)
            .collect()
            .await;
        let crashed = results
            .iter()
            .filter(|tested| matches!(tested.observation, Observation::Crashed(_)))
            .count();
        let timed_out = results
            .iter()
            .filter(|tested| matches!(tested.observation, Observation::TimedOut { .. }))
            .count();
        Ok(ToolResult::text(
            ReportTemplate {
                hook: &args.hook,
                kind,
                stale: target.stale.as_deref(),
                results: &results,
                crashed,
                timed_out,
            }
            .render()
            .expect("hook test template must render"),
        ))
    }
}

/// The hook a test runs: its current version's command and, for a lifecycle
/// hook, the events it is subscribed to.
struct Target {
    command: HookCommand,
    events: Vec<LifecycleKind>,
    /// Why the newest text of the hook was refused, when the version that
    /// runs is an older one.
    stale: Option<String>,
}

impl Target {
    fn find(set: &HookSet, kind: HookKind, name: &str) -> Result<Self, HookTestError> {
        fn of<C: HookConfig>(
            entries: &[Arc<Entry<C>>],
            name: &str,
            events: impl Fn(&C) -> Vec<LifecycleKind>,
        ) -> Option<Result<Target, HookTestError>> {
            let entry = entries.iter().find(|entry| entry.config.name() == name)?;
            let Some(current) = &entry.current else {
                return Some(Err(HookTestError::NoVersion {
                    kind: C::KIND,
                    name: name.to_string(),
                    reason: entry.unusable(),
                }));
            };
            Some(Ok(Target {
                command: current.command.clone(),
                events: events(&entry.config),
                stale: entry.rejected.clone(),
            }))
        }
        match kind {
            HookKind::Command => of(&set.command, name, |_| Vec::new()),
            HookKind::Reviewer => of(&set.reviewers, name, |_| Vec::new()),
            HookKind::Lifecycle => of(&set.lifecycle, name, |config| config.events.clone()),
        }
        .unwrap_or_else(|| {
            Err(HookTestError::UnknownHook {
                kind,
                name: name.to_string(),
            })
        })
    }
}

async fn run_sample(
    runner: &HookRunner,
    target: &Target,
    role: HookRole,
    sample: Sample,
) -> Tested {
    let command = &target.command;
    let observation = match &sample {
        Sample::Command(text) => observe(
            runner
                .run::<_, CommandHookOutput>(
                    command,
                    &CommandHookInput {
                        event: CommandHookEvent::BeforeCommand,
                        role,
                        command: text,
                    },
                )
                .await,
            command,
            |answer| match answer {
                Some(CommandHookOutput::Allow {}) => Observation::Allowed,
                Some(CommandHookOutput::Block { reason }) => Observation::Blocked(reason),
                None => Observation::Crashed("it printed no answer".to_string()),
            },
        ),
        Sample::Script(script) => {
            review(
                runner,
                command,
                role,
                PermissionRequest::Command {
                    mode: TerminalMode::Unsafe,
                    script,
                },
            )
            .await
        }
        Sample::Domain { domain, port } => {
            review(
                runner,
                command,
                role,
                PermissionRequest::Domain {
                    domain,
                    port: *port,
                },
            )
            .await
        }
        Sample::Event(event) if target.events.contains(&event.kind()) => {
            let input = SampleEvent {
                sample: event,
                role,
            };
            observe(
                runner.run::<_, LifecycleOutput>(command, &input).await,
                command,
                |answer| match answer.and_then(|output| output.context) {
                    Some(context) if !context.trim().is_empty() => Observation::Context(context),
                    _ => Observation::NoContext,
                },
            )
        }
        Sample::Event(_) => Observation::NotSubscribed,
    };
    Tested {
        sample,
        observation,
    }
}

async fn review(
    runner: &HookRunner,
    command: &HookCommand,
    role: HookRole,
    request: PermissionRequest<'_>,
) -> Observation {
    observe(
        runner
            .run::<_, ReviewAnswer>(command, &ReviewRequest { role, request })
            .await,
        command,
        |answer| match answer {
            Some(ReviewAnswer::Approve {}) => Observation::Approved,
            Some(ReviewAnswer::Deny { reason }) => Observation::Denied(reason),
            Some(ReviewAnswer::Pass {}) | None => Observation::Passed,
        },
    )
}
