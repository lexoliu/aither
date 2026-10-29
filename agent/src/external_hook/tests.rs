//! Lifecycle hooks driven by real hook scripts through a scripted model.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aither_core::llm::{
    Event, LLMRequest, Tool, ToolContext, ToolResult, model::Profile as ModelProfile,
};
use aither_sandbox::ToolRegistryBuilder;
use futures_core::Stream;
use schemars::JsonSchema;
use serde::Deserialize;

use super::*;
use crate::ask_user::{self, AnswerValue, AskUserArgs, AskUserTool, QuestionAnswer};
use crate::{Agent, HCons};

const HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

/// Longest any wait in these tests lasts.
const BOUND: Duration = Duration::from_secs(60);

#[derive(Debug)]
struct MockError;

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mock error")
    }
}

impl std::error::Error for MockError {}

/// Answers each run with the next queued text and each compaction with a
/// handoff document.
#[derive(Clone)]
struct ScriptedLlm {
    answers: Arc<Mutex<VecDeque<String>>>,
}

impl ScriptedLlm {
    fn new(answers: &[&str]) -> Self {
        Self {
            answers: Arc::new(Mutex::new(
                answers.iter().map(ToString::to_string).collect(),
            )),
        }
    }
}

impl aither_core::LanguageModel for ScriptedLlm {
    type Error = MockError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let text = if request.parameters().structured_outputs {
            HANDOFF_JSON.to_string()
        } else {
            self.answers
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "done".to_string())
        };
        futures_lite::stream::iter([Ok(Event::Text(text))])
    }

    fn profile(
        &self,
    ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
        std::future::ready(Ok(ModelProfile::new(
            "mock",
            "test",
            "mock-model",
            "mock model",
            1_000_000,
        )))
    }
}

/// Writes an executable hook script.
pub fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// Appends the input to the log named by the first argument and injects
/// `saw <event>` as context.
pub const RECORDER: &str = r#"#!/bin/sh
input=$(cat)
printf '%s\n' "$input" >> "$1"
event=$(printf '%s' "$input" | sed -n 's/^{"event":"\([a-z_]*\)".*/\1/p')
printf '{"context":"saw %s"}' "$event"
"#;

pub fn recorder(dir: &Path, log: &Path, events: &[LifecycleKind]) -> LifecycleHookConfig {
    LifecycleHookConfig {
        name: "recorder".into(),
        command: HookCommand {
            command: vec![
                "sh".into(),
                script(dir, "recorder.sh", RECORDER)
                    .to_string_lossy()
                    .into_owned(),
                log.to_string_lossy().into_owned(),
            ],
            deadline_ms: 20_000,
        },
        events: events.to_vec(),
    }
}

/// The inputs the recorder logged, in order.
pub fn logged(log: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str(line).expect("the input is JSON"))
        .collect()
}

fn events_of(log: &Path) -> Vec<String> {
    logged(log)
        .iter()
        .map(|input| input["event"].as_str().unwrap().to_string())
        .collect()
}

/// How many conversation messages carry `needle`.
fn mentions<H: crate::Hook>(
    agent: &Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, H>,
    needle: &str,
) -> usize {
    agent
        .history()
        .iter()
        .filter(|message| message.content().contains(needle))
        .count()
}

fn agent_with(
    llm: ScriptedLlm,
    hooks: &ExternalHooks,
) -> Agent<ScriptedLlm, ScriptedLlm, ScriptedLlm, HCons<ExternalHooks, ()>> {
    Agent::builder(llm).hook(hooks.clone()).build()
}

fn executor() -> executor_core::AnyExecutor {
    executor_core::AnyExecutor::new(executor_core::tokio::TokioGlobal)
}

async fn bounded<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(BOUND, work)
        .await
        .expect("the work finishes within its bound")
}

#[tokio::test]
async fn session_start_user_message_and_stop_fire_with_typed_input_and_inject_once() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let hooks = ExternalHooks::new(
        vec![recorder(
            dir.path(),
            &log,
            &[
                LifecycleKind::SessionStart,
                LifecycleKind::UserMessage,
                LifecycleKind::Stop,
            ],
        )],
        executor(),
    );
    let mut agent = agent_with(ScriptedLlm::new(&["first answer", "second answer"]), &hooks);

    bounded(agent.query("hello")).await.unwrap();
    assert_eq!(events_of(&log), ["session_start", "user_message", "stop"]);
    let inputs = logged(&log);
    assert_eq!(inputs[0]["role"], "main");
    assert_eq!(
        inputs[1],
        serde_json::json!({
            "event": "user_message", "role": "main", "message": "hello", "attachments": 0
        })
    );
    assert_eq!(
        inputs[2],
        serde_json::json!({
            "event": "stop", "role": "main", "final_text": "first answer",
            "turns": 1, "reason": "no_tool_calls"
        })
    );
    // Context injected before the request reached it; the stop's waits for
    // the next request.
    assert_eq!(mentions(&agent, "saw session_start"), 1);
    assert_eq!(mentions(&agent, "saw user_message"), 1);
    assert_eq!(mentions(&agent, "saw stop"), 0);

    bounded(agent.query("again")).await.unwrap();
    assert_eq!(
        events_of(&log),
        [
            "session_start",
            "user_message",
            "stop",
            "user_message",
            "stop"
        ],
        "the session starts once"
    );
    assert_eq!(mentions(&agent, "saw session_start"), 1);
    assert_eq!(mentions(&agent, "saw user_message"), 2);
    assert_eq!(
        mentions(&agent, "saw stop"),
        1,
        "each stop is delivered once"
    );
    let alert = agent
        .history()
        .iter()
        .find(|message| message.content().contains("saw session_start"))
        .unwrap()
        .content()
        .to_string();
    assert!(
        alert.contains("The recorder hook added context when the session started:"),
        "{alert}"
    );
}

#[tokio::test]
async fn compaction_fires_before_and_after() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let hooks = ExternalHooks::new(
        vec![recorder(
            dir.path(),
            &log,
            &[
                LifecycleKind::BeforeCompaction,
                LifecycleKind::AfterCompaction,
            ],
        )],
        executor(),
    );
    let mut agent = agent_with(ScriptedLlm::new(&["answer", "next"]), &hooks);

    bounded(agent.query("hello")).await.unwrap();
    assert_eq!(events_of(&log), Vec::<String>::new());
    let compacted = bounded(agent.compact(Some("the parser")))
        .await
        .unwrap()
        .expect("there was something to compact");

    assert_eq!(events_of(&log), ["before_compaction", "after_compaction"]);
    let inputs = logged(&log);
    assert_eq!(inputs[0]["messages"], compacted.messages_compacted);
    assert_eq!(inputs[0]["focus"], "the parser");
    assert_eq!(
        inputs[1]["messages_remaining"],
        compacted.messages_remaining
    );

    bounded(agent.query("go on")).await.unwrap();
    assert_eq!(mentions(&agent, "saw after_compaction"), 1);
    bounded(agent.query("and on")).await.unwrap();
    assert_eq!(
        mentions(&agent, "saw after_compaction"),
        1,
        "delivered once"
    );
}

/// A command that echoes its text argument.
#[derive(Debug, Deserialize, JsonSchema)]
struct EchoArgs {
    text: String,
}

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "echo".into()
    }
    type Arguments = EchoArgs;
    type Res = ToolResult;

    fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
        std::future::ready(Ok(ToolResult::text(format!("echoed {}", args.text))))
    }
}

#[tokio::test]
async fn an_ipc_call_event_carries_the_resolved_name_arguments_and_result() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let hooks = ExternalHooks::new(
        vec![recorder(dir.path(), &log, &[LifecycleKind::CommandCall])],
        executor(),
    );
    let mut builder = ToolRegistryBuilder::new();
    builder.configure_tool(EchoTool);
    builder.observe_commands(hooks.command_observer());
    let registry = builder.build(dir.path().join("outputs"));

    let args = ["--text".to_string(), "hi there".to_string()];
    let envelope = bounded(registry.query_tool_handler("echo", &args)).await;
    assert!(envelope.ok, "{envelope:?}");
    let refused = bounded(registry.query_tool_handler("no-such-command", &[])).await;
    assert!(!refused.ok);

    // The hook runs in the background; wait for it to log its input.
    let deadline = Instant::now() + Duration::from_secs(20);
    while logged(&log).is_empty() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let inputs = logged(&log);
    assert_eq!(inputs.len(), 1, "only a resolved command is reported");
    assert_eq!(
        inputs[0],
        serde_json::json!({
            "event": "command_call", "role": "main", "name": "echo",
            "args": ["--text", "hi there"], "ok": true,
            "output": "echoed hi there", "error": null
        })
    );

    wait_for_context(&hooks, 1).await;
    let mut agent = agent_with(ScriptedLlm::new(&["a", "b"]), &hooks);
    bounded(agent.query("go")).await.unwrap();
    assert_eq!(mentions(&agent, "saw command_call"), 1);
    bounded(agent.query("go again")).await.unwrap();
    assert_eq!(mentions(&agent, "saw command_call"), 1, "delivered once");
}

/// Waits until `count` injected pieces are queued in `hooks`' inbox.
async fn wait_for_context(hooks: &ExternalHooks, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while hooks.queued() < count && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(hooks.queued(), count);
}

#[tokio::test]
async fn an_ipc_call_answers_before_a_slow_hook_finishes_and_its_context_arrives_once() {
    let dir = tempfile::tempdir().unwrap();
    let slow = script(
        dir.path(),
        "slow.sh",
        "#!/bin/sh\ncat >/dev/null\nsleep 3\nprintf '{\"context\":\"slow finished\"}'\n",
    );
    let hooks = ExternalHooks::new(
        vec![LifecycleHookConfig {
            name: "slow".into(),
            command: HookCommand {
                command: vec!["sh".into(), slow.to_string_lossy().into_owned()],
                deadline_ms: 20_000,
            },
            events: vec![LifecycleKind::CommandCall],
        }],
        executor(),
    );
    let mut builder = ToolRegistryBuilder::new();
    builder.configure_tool(EchoTool);
    builder.observe_commands(hooks.command_observer());
    let registry = builder.build(dir.path().join("outputs"));

    let started = Instant::now();
    let args = ["--text".to_string(), "hi".to_string()];
    let envelope = bounded(registry.query_tool_handler("echo", &args)).await;

    assert!(envelope.ok, "{envelope:?}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the call waited for its hook: {:?}",
        started.elapsed()
    );
    assert_eq!(hooks.queued(), 0, "the hook is still running");

    wait_for_context(&hooks, 1).await;
    let mut agent = agent_with(ScriptedLlm::new(&["a", "b"]), &hooks);
    bounded(agent.query("go")).await.unwrap();
    assert_eq!(mentions(&agent, "slow finished"), 1);
    bounded(agent.query("go again")).await.unwrap();
    assert_eq!(mentions(&agent, "slow finished"), 1, "delivered once");
}

#[tokio::test]
async fn an_ask_user_event_carries_the_questions_and_the_answers() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let hooks = ExternalHooks::new(
        vec![recorder(dir.path(), &log, &[LifecycleKind::AskUser])],
        executor(),
    );
    let (broker, queue) = ask_user::channel();
    let tool = AskUserTool::new("session", broker).with_hooks(hooks.clone());
    let user = tokio::spawn(async move {
        queue
            .next()
            .await
            .expect("a question arrives")
            .respond(vec![QuestionAnswer {
                section: String::new(),
                answer: AnswerValue::Selected(vec!["postgres".into()]),
            }])
            .unwrap();
    });
    let args = AskUserArgs {
        question: Some("Which database?".into()),
        option: vec!["postgres".into(), "sqlite".into()],
        multi_select: false,
        questions: Vec::new(),
    };

    bounded(Tool::call(&tool, args, ToolContext::new()))
        .await
        .unwrap();
    bounded(user).await.unwrap();

    let inputs = logged(&log);
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0]["event"], "ask_user");
    assert_eq!(inputs[0]["questions"][0]["question"], "Which database?");
    assert_eq!(inputs[0]["questions"][0]["option"][1], "sqlite");
    assert_eq!(inputs[0]["answers"][0]["answer"][0], "postgres");
    assert_eq!(crate::Hook::take_injected_context(&hooks).len(), 1);
}

#[tokio::test]
async fn a_lifecycle_hook_cannot_block() {
    let dir = tempfile::tempdir().unwrap();
    let hook = |name: &str, body: &str| LifecycleHookConfig {
        name: name.into(),
        command: HookCommand {
            command: vec![
                "sh".into(),
                script(dir.path(), name, body)
                    .to_string_lossy()
                    .into_owned(),
            ],
            deadline_ms: 20_000,
        },
        events: vec![LifecycleKind::UserMessage, LifecycleKind::Stop],
    };
    let hooks = ExternalHooks::new(
        vec![
            // Tries to block, with context attached.
            hook(
                "decider",
                "#!/bin/sh\ncat >/dev/null\nprintf '{\"decision\":\"block\",\"context\":\"smuggled\"}'\n",
            ),
            // Fails outright.
            hook(
                "crasher",
                "#!/bin/sh\ncat >/dev/null\necho broken >&2\nexit 3\n",
            ),
            // Prints something that is not an answer.
            hook("rambler", "#!/bin/sh\ncat >/dev/null\necho not json\n"),
            // Says nothing.
            hook("silent", "#!/bin/sh\ncat >/dev/null\n"),
        ],
        executor(),
    );
    let mut agent = agent_with(ScriptedLlm::new(&["the run went through"]), &hooks);

    let answer = bounded(agent.query("hello")).await.unwrap();

    assert_eq!(answer, "the run went through");
    assert_eq!(mentions(&agent, "smuggled"), 0);
    assert_eq!(crate::Hook::take_injected_context(&hooks).len(), 0);
    assert!(serde_json::from_str::<LifecycleOutput>(r#"{"approve":true}"#).is_err());
    assert_eq!(
        serde_json::from_str::<LifecycleOutput>("{}").unwrap(),
        LifecycleOutput::default()
    );
}

fn pid_is_alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid])
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap()
        .success()
}

#[tokio::test]
async fn a_hook_past_its_deadline_is_killed_with_its_process_group() {
    let dir = tempfile::tempdir().unwrap();
    let pids = dir.path().join("pids");
    let hung = script(
        dir.path(),
        "hung.sh",
        "#!/bin/sh\ncat >/dev/null\nsleep 300 &\necho $$ $! > \"$1\"\nwait\n",
    );
    let command = HookCommand {
        command: vec![
            "sh".into(),
            hung.to_string_lossy().into_owned(),
            pids.to_string_lossy().into_owned(),
        ],
        deadline_ms: 1_000,
    };

    let started = Instant::now();
    let error = bounded(
        HookRunner.run::<_, LifecycleOutput>(&command, &serde_json::json!({"event": "stop"})),
    )
    .await
    .expect_err("the hook outlives its deadline");

    assert!(matches!(error, HookRunError::Timeout { .. }), "{error}");
    assert!(started.elapsed() < Duration::from_secs(30));
    let recorded = std::fs::read_to_string(&pids).unwrap();
    let pids: Vec<&str> = recorded.split_whitespace().collect();
    assert_eq!(
        pids.len(),
        2,
        "the hook and its background child: {recorded}"
    );
    for pid in pids {
        let deadline = Instant::now() + Duration::from_secs(20);
        while pid_is_alive(pid) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            !pid_is_alive(pid),
            "process {pid} outlived its group's kill"
        );
    }
}

#[tokio::test]
async fn a_hook_that_fails_in_every_way_is_reported_and_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let run = |command: Vec<String>| {
        let command = HookCommand {
            command,
            deadline_ms: 20_000,
        };
        async move {
            bounded(HookRunner.run::<_, LifecycleOutput>(&command, &serde_json::json!({}))).await
        }
    };
    assert!(matches!(run(vec![]).await, Err(HookRunError::EmptyCommand)));
    assert!(matches!(
        run(vec![
            dir.path().join("missing").to_string_lossy().into_owned()
        ])
        .await,
        Err(HookRunError::Spawn { .. })
    ));
    assert!(matches!(
        run(vec!["sh".into(), "-c".into(), "exit 2".into()]).await,
        Err(HookRunError::Exit { .. })
    ));
    assert!(matches!(
        run(vec![
            "sh".into(),
            "-c".into(),
            "echo '{\"context\":1}'".into()
        ])
        .await,
        Err(HookRunError::Output(_))
    ));
    assert_eq!(
        run(vec!["sh".into(), "-c".into(), "true".into()])
            .await
            .unwrap(),
        None
    );
}

pub mod reviewers {
    use aither_sandbox::permission::{
        PermissionError, PermissionHandler, PermissionRequest, Review, TerminalMode,
    };

    use super::*;

    /// Logs the input to the file named by the first argument and prints the
    /// answer given as the second.
    pub const REVIEWER: &str = r#"#!/bin/sh
cat >> "$1"
printf '\n' >> "$1"
printf '%s' "$2"
"#;

    /// The existing route: records the requests that reach it and answers
    /// `allow`.
    #[derive(Clone, Default)]
    pub struct Route {
        pub asked: Arc<Mutex<Vec<(TerminalMode, String)>>>,
        pub allow: bool,
        /// The domains `check_domain` was asked about.
        pub domains_asked: Arc<Mutex<Vec<String>>>,
        /// Whether `check_domain` approves the domains it is asked about.
        pub allow_domain: bool,
        /// Domains that need no approval: the default allow-list and the
        /// domains already approved, like a user-facing handler keeps.
        pub allowed_domains: Arc<Mutex<Vec<String>>>,
    }

    pub const COMMAND: PermissionRequest<'static> = PermissionRequest::Command {
        mode: TerminalMode::Unsafe,
        script: "make install",
    };

    pub const DOMAIN: PermissionRequest<'static> = PermissionRequest::Domain {
        domain: "registry.example",
        port: 443,
    };

    impl PermissionHandler for Route {
        fn check(
            &self,
            mode: TerminalMode,
            script: &str,
        ) -> impl Future<Output = Result<bool, PermissionError>> + Send {
            self.asked.lock().unwrap().push((mode, script.to_string()));
            std::future::ready(Ok(self.allow))
        }

        fn check_domain(&self, domain: &str, _port: u16) -> impl Future<Output = bool> + Send {
            self.domains_asked.lock().unwrap().push(domain.to_string());
            let approved = {
                let mut allowed = self.allowed_domains.lock().unwrap();
                let approved = allowed.iter().any(|d| d == domain);
                if self.allow_domain && !approved {
                    allowed.push(domain.to_string());
                }
                approved
            };
            std::future::ready(self.allow_domain || approved)
        }

        fn domain_needs_approval(
            &self,
            domain: &str,
            _port: u16,
        ) -> impl Future<Output = bool> + Send {
            std::future::ready(
                !self
                    .allowed_domains
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|d| d == domain),
            )
        }

        fn remember_domain(&self, domain: &str, _port: u16) -> impl Future<Output = ()> + Send {
            self.allowed_domains
                .lock()
                .unwrap()
                .push(domain.to_string());
            std::future::ready(())
        }
    }

    pub fn reviewer(dir: &Path, name: &str, log: &Path, answer: &str) -> ReviewerConfig {
        ReviewerConfig {
            name: name.into(),
            command: HookCommand {
                command: vec![
                    "sh".into(),
                    script(dir, "reviewer.sh", REVIEWER)
                        .to_string_lossy()
                        .into_owned(),
                    log.to_string_lossy().into_owned(),
                    answer.into(),
                ],
                deadline_ms: 20_000,
            },
        }
    }

    fn handler(chain: Vec<ReviewerConfig>) -> (ReviewedPermission<Route>, Route) {
        let route = Route::default();
        (
            ReviewedPermission::new(route.clone(), ReviewerChain::new(chain)),
            route,
        )
    }

    #[tokio::test]
    async fn approve_grants_without_reaching_the_route_and_ends_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let (first, second) = (dir.path().join("first"), dir.path().join("second"));
        let (handler, route) = handler(vec![
            reviewer(dir.path(), "first", &first, r#"{"decision":"approve"}"#),
            reviewer(
                dir.path(),
                "second",
                &second,
                r#"{"decision":"deny","reason":"no"}"#,
            ),
        ]);
        assert_eq!(bounded(handler.review(&COMMAND)).await, Review::Approved);
        assert!(route.asked.lock().unwrap().is_empty());
        assert_eq!(logged(&first).len(), 1);
        assert!(!second.exists(), "the first decision ends the chain");
        assert_eq!(
            logged(&first)[0],
            serde_json::json!({"role": "main", "kind": "command", "mode": "unsafe", "script": "make install"})
        );
    }

    #[tokio::test]
    async fn deny_refuses_with_a_reason_the_model_reads() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let (handler, route) = handler(vec![
            reviewer(dir.path(), "policy", &log, r#"{"decision":"pass"}"#),
            reviewer(
                dir.path(),
                "guard",
                &log,
                r#"{"decision":"deny","reason":"touches /etc"}"#,
            ),
        ]);
        let Review::Denied(reason) = bounded(handler.review(&COMMAND)).await else {
            panic!("the guard denies");
        };
        assert_eq!(
            reason.trim(),
            "The permission reviewer guard denied this request: touches /etc"
        );
        assert!(route.asked.lock().unwrap().is_empty());
        assert_eq!(logged(&log).len(), 2, "the passing reviewer ran first");
    }

    #[tokio::test]
    async fn a_request_every_reviewer_passes_reaches_the_existing_route() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let route = Route {
            allow: true,
            ..Route::default()
        };
        let handler = ReviewedPermission::new(
            route.clone(),
            ReviewerChain::new(vec![
                reviewer(dir.path(), "a", &log, r#"{"decision":"pass"}"#),
                reviewer(dir.path(), "b", &log, ""),
            ]),
        );
        assert_eq!(bounded(handler.review(&COMMAND)).await, Review::Passed);
        assert!(
            bounded(handler.check(TerminalMode::Unsafe, "make install"))
                .await
                .unwrap()
        );
        assert_eq!(
            *route.asked.lock().unwrap(),
            [(TerminalMode::Unsafe, "make install".to_string())]
        );
        assert_eq!(logged(&log).len(), 2);
    }

    #[tokio::test]
    async fn a_subagents_request_is_labelled_and_reaches_the_same_route() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let (handler, route) = handler(vec![reviewer(
            dir.path(),
            "a",
            &log,
            r#"{"decision":"pass"}"#,
        )]);
        let subagent = handler
            .for_subagent()
            .expect("a reviewed handler labels subagents");
        assert_eq!(bounded(subagent.review(&COMMAND)).await, Review::Passed);
        bounded(subagent.check(TerminalMode::Unsafe, "make install"))
            .await
            .unwrap();
        assert_eq!(logged(&log)[0]["role"], "subagent");
        assert_eq!(route.asked.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_reviewer_that_fails_counts_as_a_pass_and_grants_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let failing = |body: &str, name: &str| {
            let path = script(dir.path(), name, body);
            ReviewerConfig {
                name: name.into(),
                command: HookCommand {
                    command: vec![path.to_string_lossy().into_owned()],
                    deadline_ms: 500,
                },
            }
        };
        let (handler, route) = handler(vec![
            failing("#!/bin/sh\nexit 3\n", "crashes"),
            failing("#!/bin/sh\nsleep 30\n", "hangs"),
            failing("#!/bin/sh\nprintf 'approve please'\n", "garbles"),
            failing(
                "#!/bin/sh\nprintf '{\"decision\":\"approve\",\"extra\":1}'\n",
                "extra_field",
            ),
            ReviewerConfig {
                name: "missing".into(),
                command: HookCommand {
                    command: vec![dir.path().join("absent").to_string_lossy().into_owned()],
                    deadline_ms: 500,
                },
            },
        ]);
        let started = Instant::now();
        assert_eq!(
            bounded(handler.review(&COMMAND)).await,
            Review::Passed,
            "a failure grants nothing"
        );
        assert!(route.asked.lock().unwrap().is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the hang was killed"
        );
        assert!(!log.exists());
    }

    #[tokio::test]
    async fn a_reviewer_approves_a_domain_and_it_is_not_reviewed_again() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let (handler, route) = handler(vec![reviewer(
            dir.path(),
            "net",
            &log,
            r#"{"decision":"approve"}"#,
        )]);
        assert!(bounded(handler.authorize_domain("registry.example", 443)).await);
        assert!(
            route.domains_asked.lock().unwrap().is_empty(),
            "the user's route was not asked"
        );
        assert_eq!(
            logged(&log),
            [serde_json::json!({
                "role": "main", "kind": "domain", "domain": "registry.example", "port": 443
            })]
        );
        // The next connection to it is remembered: no reviewer runs.
        assert!(bounded(handler.authorize_domain("registry.example", 443)).await);
        assert_eq!(logged(&log).len(), 1);
        // Another domain is reviewed.
        assert!(bounded(handler.authorize_domain("other.example", 443)).await);
        assert_eq!(logged(&log).len(), 2);
    }

    #[tokio::test]
    async fn a_reviewer_denies_a_domain_and_it_is_reviewed_again() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let (handler, route) = handler(vec![reviewer(
            dir.path(),
            "net",
            &log,
            r#"{"decision":"deny","reason":"unknown host"}"#,
        )]);
        let Review::Denied(reason) = bounded(handler.review(&DOMAIN)).await else {
            panic!("the reviewer denies");
        };
        assert!(reason.contains("unknown host"));
        assert!(!bounded(handler.authorize_domain("registry.example", 443)).await);
        assert!(!bounded(handler.authorize_domain("registry.example", 443)).await);
        assert!(route.domains_asked.lock().unwrap().is_empty());
        assert_eq!(logged(&log).len(), 3, "a denial is not remembered");
    }

    #[tokio::test]
    async fn an_allowed_domain_and_a_passed_domain_follow_the_existing_route() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let route = Route {
            allow_domain: true,
            allowed_domains: Arc::new(Mutex::new(vec!["allowed.example".into()])),
            ..Route::default()
        };
        let handler = ReviewedPermission::new(
            route.clone(),
            ReviewerChain::new(vec![reviewer(
                dir.path(),
                "net",
                &log,
                r#"{"decision":"pass"}"#,
            )]),
        );
        // Allowed without asking anyone: no reviewer, and the route decides.
        assert!(bounded(handler.authorize_domain("allowed.example", 443)).await);
        assert!(!log.exists());
        // Passed by every reviewer: the existing route is asked, once.
        assert!(bounded(handler.authorize_domain("registry.example", 443)).await);
        assert_eq!(logged(&log).len(), 1);
        assert_eq!(
            *route.domains_asked.lock().unwrap(),
            ["allowed.example", "registry.example"]
        );
        assert!(bounded(handler.authorize_domain("registry.example", 443)).await);
        assert_eq!(logged(&log).len(), 1, "the user's approval is remembered");
    }
}
