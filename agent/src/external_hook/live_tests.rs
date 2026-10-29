//! Live hooks driven by real hook scripts in temporary directories: changes
//! reach running scopes, broken changes keep the previous version and raise
//! warnings, hooks nobody can run pause and ask the user, and the hook-test
//! command reports what a hook does.

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use aither_core::llm::{Tool, ToolContext};
use aither_sandbox::GateDecision;
use aither_sandbox::permission::{PermissionHandler, Review};

use super::live::Watching;
use super::tests::{executor, live, logged, recorder, reviewers, script};
use super::*;
use crate::StopReason;
use crate::ask_user::{self, AnswerValue, AskUserQueue, QuestionAnswer};

/// Longest any wait in these tests lasts.
const BOUND: Duration = Duration::from_secs(30);

const ALLOW: &str = "#!/bin/sh\ncat >/dev/null\nprintf '{\"decision\":\"allow\"}'\n";
const BLOCK: &str =
    "#!/bin/sh\ncat >/dev/null\nprintf '{\"decision\":\"block\",\"reason\":\"second version\"}'\n";
const APPROVE: &str = "#!/bin/sh\ncat >/dev/null\nprintf '{\"decision\":\"approve\"}'\n";
const DENY: &str =
    "#!/bin/sh\ncat >/dev/null\nprintf '{\"decision\":\"deny\",\"reason\":\"second version\"}'\n";
const CRASH: &str = "#!/bin/sh\ncat >/dev/null\necho broken >&2\nexit 3\n";
const BROKEN_SYNTAX: &str = "#!/bin/sh\nif true; then\necho unfinished\n";
/// Fails while the file named by the first argument exists.
const GATE_UNLESS_FLAG: &str = "#!/bin/sh\ncat >/dev/null\nif [ -e \"$1\" ]; then echo flagged >&2; exit 1; fi\nprintf '{\"decision\":\"allow\"}'\n";
const REVIEWER_UNLESS_FLAG: &str = "#!/bin/sh\ncat >/dev/null\nif [ -e \"$1\" ]; then echo flagged >&2; exit 1; fi\nprintf '{\"decision\":\"approve\"}'\n";

fn command(dir: &Path, name: &str, file: &str, body: &str, args: &[&Path]) -> CommandHookConfig {
    CommandHookConfig {
        name: name.into(),
        command: hook_command(dir, file, body, args),
    }
}

fn reviewer(dir: &Path, name: &str, file: &str, body: &str, args: &[&Path]) -> ReviewerConfig {
    ReviewerConfig {
        name: name.into(),
        command: hook_command(dir, file, body, args),
    }
}

fn hook_command(dir: &Path, file: &str, body: &str, args: &[&Path]) -> HookCommand {
    let mut command = vec![
        "sh".to_string(),
        script(dir, file, body).to_string_lossy().into_owned(),
    ];
    command.extend(args.iter().map(|arg| arg.to_string_lossy().into_owned()));
    HookCommand {
        command,
        deadline_ms: 20_000,
    }
}

fn rewrite(dir: &Path, file: &str, body: &str) {
    script(dir, file, body);
}

fn with_commands(hooks: Vec<CommandHookConfig>) -> HooksConfig {
    HooksConfig {
        command: hooks,
        ..HooksConfig::default()
    }
}

fn with_reviewers(hooks: Vec<ReviewerConfig>) -> HooksConfig {
    HooksConfig {
        reviewers: hooks,
        ..HooksConfig::default()
    }
}

async fn bounded<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(BOUND, work)
        .await
        .expect("the work finishes within its bound")
}

/// Polls `check` until it holds.
async fn eventually<F: std::future::Future<Output = bool>>(mut check: impl FnMut() -> F) {
    bounded(async {
        while !check().await {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
}

/// Takes warnings until one satisfies `wanted`.
async fn warning_where(
    warnings: &async_channel::Receiver<HookWarning>,
    wanted: impl Fn(&HookWarning) -> bool + Sync,
) -> HookWarning {
    bounded(async {
        loop {
            let warning = warnings.recv().await.expect("the hooks are alive");
            if wanted(&warning) {
                return warning;
            }
        }
    })
    .await
}

fn blocked_with(decision: &GateDecision, text: &str) -> bool {
    matches!(decision, GateDecision::Block { reason } if reason.contains(text))
}

#[tokio::test]
async fn a_script_change_reaches_running_sessions_without_a_restart() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(
        with_commands(vec![command(dir.path(), "gate", "gate.sh", ALLOW, &[])]),
        &tools,
    )
    .await;
    let session = ExternalHooks::new(&hooks);
    let subagent = session.for_subagent();
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );

    rewrite(dir.path(), "gate.sh", BLOCK);

    eventually(|| async { blocked_with(&session.check_command("ls").await, "second version") })
        .await;
    assert!(blocked_with(
        &subagent.check_command("ls").await,
        "second version"
    ));
}

#[tokio::test]
async fn a_reviewer_script_change_reaches_running_sessions_without_a_restart() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(
        with_reviewers(vec![reviewer(
            dir.path(),
            "guard",
            "guard.sh",
            APPROVE,
            &[],
        )]),
        &tools,
    )
    .await;
    let handler = ReviewedPermission::new(reviewers::Route::default(), ExternalHooks::new(&hooks));
    assert_eq!(
        bounded(handler.review(&reviewers::COMMAND)).await,
        Review::Approved
    );

    rewrite(dir.path(), "guard.sh", DENY);

    eventually(|| async {
        matches!(handler.review(&reviewers::COMMAND).await, Review::Denied(reason) if reason.contains("second version"))
    })
    .await;
}

#[tokio::test]
async fn a_configuration_change_reaches_running_sessions_without_a_restart() {
    let tools = crate::test_support::prepared_fixture().await;
    let scripts = tempfile::tempdir().unwrap();
    let configs = tempfile::tempdir().unwrap();
    let allow = command(scripts.path(), "gate", "allow.sh", ALLOW, &[]);
    let block = command(scripts.path(), "gate", "block.sh", BLOCK, &[]);
    let file = configs.path().join("hooks.json");
    let write = |config: &HooksConfig| {
        std::fs::write(&file, serde_json::to_vec(config).unwrap()).unwrap();
    };
    write(&with_commands(vec![allow.clone()]));
    let hooks = bounded(LiveHooks::start_from_file(
        file.clone(),
        &tools.tools,
        executor(),
    ))
    .await
    .expect("the hooks start");
    let session = ExternalHooks::new(&hooks);
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );

    // The file is followed: replacing the hook in it applies at once.
    write(&with_commands(vec![block.clone()]));
    eventually(|| async { blocked_with(&session.check_command("ls").await, "second version") })
        .await;

    // So is a configuration the host pushes.
    bounded(hooks.apply(with_commands(vec![allow])))
        .await
        .unwrap();
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );

    // A configuration file that is not valid keeps the one in use.
    let warnings = hooks.warnings();
    std::fs::write(&file, "{not json").unwrap();
    let warning = warning_where(&warnings, |w| {
        matches!(w, HookWarning::ConfigInvalid { .. })
    })
    .await;
    assert!(
        warning.to_string().contains("previous configuration"),
        "{warning}"
    );
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );
}

#[tokio::test]
async fn startup_with_a_syntax_error_is_refused() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let broken = |name: &str| hook_command(dir.path(), name, BROKEN_SYNTAX, &[]);
    let cases = [
        (
            HookKind::Lifecycle,
            HooksConfig {
                lifecycle: vec![LifecycleHookConfig {
                    name: "watcher".into(),
                    command: broken("lifecycle.sh"),
                    events: vec![LifecycleKind::Stop],
                }],
                ..HooksConfig::default()
            },
            "watcher",
        ),
        (
            HookKind::Command,
            with_commands(vec![CommandHookConfig {
                name: "gate".into(),
                command: broken("command.sh"),
            }]),
            "gate",
        ),
        (
            HookKind::Reviewer,
            with_reviewers(vec![ReviewerConfig {
                name: "guard".into(),
                command: broken("reviewer.sh"),
            }]),
            "guard",
        ),
    ];
    for (expected_kind, config, expected_hook) in cases {
        let error = bounded(LiveHooks::start(config, &tools.tools, executor()))
            .await
            .expect_err("a hook with a syntax error does not start");
        match error {
            LiveHooksError::Hook {
                kind,
                hook,
                error: HookCheckError::Syntax { .. },
            } => {
                assert_eq!(kind, expected_kind);
                assert_eq!(hook, expected_hook);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[tokio::test]
async fn a_changed_hook_with_a_syntax_error_warns_and_the_previous_version_keeps_running() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let config = HooksConfig {
        lifecycle: vec![recorder(dir.path(), &log, &[LifecycleKind::Stop])],
        command: vec![command(dir.path(), "gate", "gate.sh", ALLOW, &[])],
        reviewers: vec![reviewer(dir.path(), "guard", "guard.sh", APPROVE, &[])],
    };
    let hooks = live(config, &tools).await;
    let warnings = hooks.warnings();
    let session = ExternalHooks::new(&hooks);
    let handler = ReviewedPermission::new(reviewers::Route::default(), session.clone());

    let rejected = |kind: HookKind, name: &'static str| move |warning: &HookWarning| matches!(warning, HookWarning::Rejected { kind: k, hook, kept_previous: true, .. } if *k == kind && hook == name);

    rewrite(dir.path(), "gate.sh", BROKEN_SYNTAX);
    let warning = warning_where(&warnings, rejected(HookKind::Command, "gate")).await;
    assert!(
        warning
            .to_string()
            .contains("previous version stays in use"),
        "{warning}"
    );
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );

    rewrite(dir.path(), "guard.sh", BROKEN_SYNTAX);
    warning_where(&warnings, rejected(HookKind::Reviewer, "guard")).await;
    assert_eq!(
        bounded(handler.review(&reviewers::COMMAND)).await,
        Review::Approved
    );

    rewrite(dir.path(), "recorder.sh", BROKEN_SYNTAX);
    warning_where(&warnings, rejected(HookKind::Lifecycle, "recorder")).await;
    bounded(session.fire(&Stop {
        final_text: "done".into(),
        turns: 1,
        reason: StopReason::Complete,
    }))
    .await;
    assert_eq!(logged(&log).len(), 1, "the previous recorder still runs");
    assert_eq!(crate::Hook::take_injected_context(&session).len(), 1);
}

#[tokio::test]
async fn a_changed_command_hook_that_fails_to_run_warns_and_the_previous_version_keeps_running() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let config = with_commands(vec![command(dir.path(), "gate", "gate.sh", ALLOW, &[])]);
    let hooks = live(config.clone(), &tools).await;
    let warnings = hooks.warnings();
    let session = ExternalHooks::new(&hooks);

    // Valid syntax, so the change is taken up; running it fails.
    rewrite(dir.path(), "gate.sh", CRASH);
    let warning = bounded(async {
        loop {
            // The previous version answers before and after the change lands.
            assert_eq!(session.check_command("ls").await, GateDecision::Allow);
            if let Ok(warning) = warnings.try_recv() {
                return warning;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    assert!(
        matches!(&warning, HookWarning::RunFailed { kind: HookKind::Command, hook, fell_back: true, .. } if hook == "gate"),
        "{warning:?}"
    );
    assert!(
        warning.to_string().contains("previous version took over"),
        "{warning}"
    );

    // The previous version stays in use: no further failure, even after a
    // reload that finds the failed version again.
    bounded(hooks.apply(config)).await.unwrap();
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );
    assert!(
        warnings.try_recv().is_err(),
        "the failed version was not tried again"
    );
}

#[tokio::test]
async fn a_changed_reviewer_that_fails_to_run_warns_and_the_previous_version_keeps_running() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(
        with_reviewers(vec![reviewer(
            dir.path(),
            "guard",
            "guard.sh",
            APPROVE,
            &[],
        )]),
        &tools,
    )
    .await;
    let warnings = hooks.warnings();
    let handler = ReviewedPermission::new(reviewers::Route::default(), ExternalHooks::new(&hooks));

    rewrite(dir.path(), "guard.sh", CRASH);
    let warning = bounded(async {
        loop {
            assert_eq!(handler.review(&reviewers::COMMAND).await, Review::Approved);
            if let Ok(warning) = warnings.try_recv() {
                return warning;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    assert!(
        matches!(&warning, HookWarning::RunFailed { kind: HookKind::Reviewer, hook, fell_back: true, .. } if hook == "guard"),
        "{warning:?}"
    );
}

#[tokio::test]
async fn a_failing_command_hook_with_no_previous_version_pauses_and_asks_the_user() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let flag = dir.path().join("flag");
    std::fs::write(&flag, "").unwrap();
    let hooks = live(
        with_commands(vec![command(
            dir.path(),
            "gate",
            "gate.sh",
            GATE_UNLESS_FLAG,
            &[&flag],
        )]),
        &tools,
    )
    .await;
    let warnings = hooks.warnings();
    let (broker, queue) = ask_user::channel();
    let session = ExternalHooks::new(&hooks).with_user_pause(broker, "session-1");
    let running = tokio::spawn({
        let session = session.clone();
        async move { session.check_command("ls").await }
    });

    let question = bounded(queue.next()).await.expect("the user is asked");
    let asked = question.args.clone();
    assert_eq!(asked.session_id, "session-1");
    let text = asked.request.question.clone().unwrap();
    assert!(text.contains("gate") && text.contains("flagged"), "{text}");
    assert_eq!(asked.request.option, ["Retry"]);
    let warning = warning_where(&warnings, |w| {
        matches!(
            w,
            HookWarning::RunFailed {
                fell_back: false,
                ..
            }
        )
    })
    .await;
    assert!(
        warning.to_string().contains("asks you to fix it"),
        "{warning}"
    );
    assert!(!running.is_finished(), "the session waits for the user");

    // Answering without a fix retries, fails and asks again, once.
    question
        .respond(vec![QuestionAnswer {
            section: String::new(),
            answer: AnswerValue::Selected(vec!["Retry".into()]),
        }])
        .unwrap();
    let again = answer_after_fix(&queue, &flag).await;
    assert!(again.request.question.unwrap().contains("flagged"));

    assert_eq!(bounded(running).await.unwrap(), GateDecision::Allow);
}

/// Waits for a question that arrives after the previous answer failed again,
/// shows that nothing else is asked meanwhile, fixes the hook's environment
/// and answers.
async fn answer_after_fix(queue: &AskUserQueue, flag: &Path) -> ask_user::AskUserRequestPayload {
    let request = bounded(queue.next())
        .await
        .expect("the user is asked again");
    let asked = request.args.clone();
    // Nothing spins: with the question open, no further one arrives.
    assert!(
        tokio::time::timeout(Duration::from_millis(400), queue.next())
            .await
            .is_err()
    );
    std::fs::remove_file(flag).unwrap();
    request
        .respond(vec![QuestionAnswer {
            section: String::new(),
            answer: AnswerValue::Selected(vec!["Retry".into()]),
        }])
        .unwrap();
    asked
}

#[tokio::test]
async fn a_reviewer_whose_previous_version_fails_too_pauses_and_asks_the_user() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let flag = dir.path().join("flag");
    let config = |body: &str| {
        with_reviewers(vec![reviewer(
            dir.path(),
            "guard",
            "guard.sh",
            body,
            &[&flag],
        )])
    };
    let hooks = live(config(REVIEWER_UNLESS_FLAG), &tools).await;
    let (broker, queue) = ask_user::channel();
    let handler = ReviewedPermission::new(
        reviewers::Route::default(),
        ExternalHooks::new(&hooks).with_user_pause(broker, "session-2"),
    );
    assert_eq!(
        bounded(handler.review(&reviewers::COMMAND)).await,
        Review::Approved
    );

    // A second version, applied at once; then both fail.
    let changed = format!("{REVIEWER_UNLESS_FLAG}# second version\n");
    bounded(hooks.apply(config(&changed))).await.unwrap();
    std::fs::write(&flag, "").unwrap();
    let running = tokio::spawn(async move { handler.review(&reviewers::COMMAND).await });

    let question = bounded(queue.next()).await.expect("the user is asked");
    assert_eq!(question.args.session_id, "session-2");
    assert!(
        question
            .args
            .request
            .question
            .clone()
            .unwrap()
            .contains("guard")
    );
    std::fs::remove_file(&flag).unwrap();
    question
        .respond(vec![QuestionAnswer {
            section: String::new(),
            answer: AnswerValue::Selected(vec!["Retry".into()]),
        }])
        .unwrap();
    assert_eq!(bounded(running).await.unwrap(), Review::Approved);
}

#[tokio::test]
async fn without_a_user_to_ask_a_failing_command_hook_blocks_and_a_failing_reviewer_passes() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(
        HooksConfig {
            command: vec![command(dir.path(), "gate", "gate.sh", CRASH, &[])],
            reviewers: vec![reviewer(dir.path(), "guard", "guard.sh", CRASH, &[])],
            ..HooksConfig::default()
        },
        &tools,
    )
    .await;
    let session = ExternalHooks::new(&hooks);
    assert!(blocked_with(
        &bounded(session.check_command("ls")).await,
        "gate"
    ));
    let handler = ReviewedPermission::new(reviewers::Route::default(), session);
    assert_eq!(
        bounded(handler.review(&reviewers::COMMAND)).await,
        Review::Passed
    );
}

async fn test_report(hooks: &LiveHooks, args: HookTestArgs) -> String {
    let result = bounded(Tool::call(
        &HookTestTool::new(hooks.clone()),
        args,
        ToolContext::new(),
    ))
    .await
    .expect("the hook is tested");
    result.as_text().expect("the report is text").to_string()
}

#[tokio::test]
async fn the_hook_test_command_reports_decisions_crashes_and_timeouts() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let gate = "#!/bin/sh\ninput=$(cat)\ncase \"$input\" in\n  *crash*) echo boom >&2; exit 1;;\n  *slow*) sleep 300;;\n  *'rm -rf'*) printf '{\"decision\":\"block\",\"reason\":\"no rm\"}';;\n  *) printf '{\"decision\":\"allow\"}';;\nesac\n";
    let mut hook = command(dir.path(), "gate", "gate.sh", gate, &[]);
    hook.command.deadline_ms = 1_000;
    let hooks = live(with_commands(vec![hook]), &tools).await;

    let report = test_report(
        &hooks,
        HookTestArgs {
            hook: "gate".into(),
            command: vec![
                "ls".into(),
                "rm -rf /".into(),
                "crash now".into(),
                "slow one".into(),
            ],
            ..HookTestArgs::default()
        },
    )
    .await;

    assert!(report.contains("command `ls`: allowed"), "{report}");
    assert!(
        report.contains("command `rm -rf /`: blocked: no rm"),
        "{report}"
    );
    assert!(report.contains("command `crash now`: crashed:"), "{report}");
    assert!(report.contains("boom"), "{report}");
    assert!(
        report.contains("command `slow one`: timed out after 1000 ms"),
        "{report}"
    );
    assert!(report.contains("1 crashed, 1 timed out"), "{report}");

    let error = bounded(Tool::call(
        &HookTestTool::new(hooks),
        HookTestArgs {
            hook: "missing".into(),
            command: vec!["ls".into()],
            ..HookTestArgs::default()
        },
        ToolContext::new(),
    ))
    .await
    .expect_err("an unknown hook cannot be tested");
    assert!(
        error
            .to_string()
            .contains("no command hook named `missing`"),
        "{error}"
    );
}

#[tokio::test]
async fn the_hook_test_command_runs_reviewers_and_lifecycle_hooks_on_samples() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let hooks = live(
        HooksConfig {
            lifecycle: vec![recorder(dir.path(), &log, &[LifecycleKind::UserMessage])],
            reviewers: vec![reviewer(dir.path(), "guard", "guard.sh", DENY, &[])],
            ..HooksConfig::default()
        },
        &tools,
    )
    .await;

    let report = test_report(
        &hooks,
        HookTestArgs {
            hook: "guard".into(),
            role: Some(HookRole::Subagent),
            review_command: vec!["make install".into()],
            review_domain: vec!["registry.example:443".into()],
            ..HookTestArgs::default()
        },
    )
    .await;
    assert!(
        report.contains("unsafe-mode request `make install`: denied: second version"),
        "{report}"
    );
    assert!(
        report.contains("domain registry.example:443: denied"),
        "{report}"
    );

    let report = test_report(
        &hooks,
        HookTestArgs {
            hook: "recorder".into(),
            event: vec![
                EventSample::UserMessage(UserMessage {
                    message: "hi".into(),
                    attachments: 0,
                }),
                EventSample::Stop(Stop {
                    final_text: String::new(),
                    turns: 1,
                    reason: StopReason::Complete,
                }),
            ],
            ..HookTestArgs::default()
        },
    )
    .await;
    assert!(
        report.contains("added context: saw user_message"),
        "{report}"
    );
    assert!(report.contains("not subscribed"), "{report}");
    let inputs = logged(&log);
    assert_eq!(
        inputs.len(),
        1,
        "an event the hook is not subscribed to is not run"
    );
    assert_eq!(inputs[0]["role"], "main");
}

/// Calls `hook_test` the way a terminal does: as a command registered with
/// `configure_tool`, parsed from the command line.
async fn through_the_bridge(hooks: &LiveHooks, argv: &[&str]) -> aither_sandbox::CommandEnvelope {
    let mut builder = aither_sandbox::ToolRegistryBuilder::new();
    builder.configure_tool(HookTestTool::new(hooks.clone()));
    let outputs = tempfile::tempdir().unwrap();
    let registry = builder.build(outputs.path());
    let argv: Vec<String> = argv.iter().map(ToString::to_string).collect();
    bounded(registry.query_tool_handler("hook_test", &argv)).await
}

fn text_of(envelope: &aither_sandbox::CommandEnvelope) -> &str {
    match envelope.payload.as_ref() {
        Some(aither_sandbox::CommandPayload::Text { content }) => content,
        None => envelope.error.as_deref().unwrap_or_default(),
    }
}

#[tokio::test]
async fn the_hook_test_command_takes_events_and_samples_from_the_command_line() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let hooks = live(
        HooksConfig {
            lifecycle: vec![recorder(dir.path(), &log, &[LifecycleKind::UserMessage])],
            command: vec![command(dir.path(), "gate", "gate.sh", BLOCK, &[])],
            ..HooksConfig::default()
        },
        &tools,
    )
    .await;

    let envelope = through_the_bridge(
        &hooks,
        &[
            "--hook",
            "recorder",
            "--role",
            "subagent",
            "--event",
            r#"{"event":"user_message","message":"hi","attachments":0}"#,
            "--event",
            r#"{"event":"stop","final_text":"","turns":1,"reason":"complete"}"#,
        ],
    )
    .await;
    assert!(envelope.ok, "{envelope:?}");
    let report = text_of(&envelope);
    assert!(
        report.contains("added context: saw user_message"),
        "{report}"
    );
    assert!(report.contains("not run"), "{report}");
    assert_eq!(logged(&log)[0]["role"], "subagent");
    assert_eq!(logged(&log)[0]["message"], "hi");

    // An event whose fields are wrong is refused with the schema's reason.
    let envelope = through_the_bridge(
        &hooks,
        &[
            "--hook",
            "recorder",
            "--event",
            r#"{"event":"user_message","message":"hi"}"#,
        ],
    )
    .await;
    assert!(!envelope.ok, "{envelope:?}");
    assert!(text_of(&envelope).contains("attachments"), "{envelope:?}");

    let envelope = through_the_bridge(
        &hooks,
        &["--hook", "gate", "--command", "ls", "--command", "pwd"],
    )
    .await;
    assert!(envelope.ok, "{envelope:?}");
    assert!(
        text_of(&envelope).contains("command `pwd`: blocked: second version"),
        "{envelope:?}"
    );
}

#[tokio::test]
async fn the_hook_test_command_says_when_a_refused_change_left_an_older_version_running() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(
        with_commands(vec![command(dir.path(), "gate", "gate.sh", ALLOW, &[])]),
        &tools,
    )
    .await;
    let warnings = hooks.warnings();
    rewrite(dir.path(), "gate.sh", BROKEN_SYNTAX);
    warning_where(&warnings, |w| matches!(w, HookWarning::Rejected { .. })).await;

    let report = test_report(
        &hooks,
        HookTestArgs {
            hook: "gate".into(),
            command: vec!["ls".into()],
            ..HookTestArgs::default()
        },
    )
    .await;
    assert!(report.contains("command `ls`: allowed"), "{report}");
    assert!(
        report.contains("newest text of the hook was refused")
            && report.contains("ran the previous version"),
        "{report}"
    );
}

#[tokio::test]
async fn a_new_command_hook_with_a_syntax_error_pauses_and_asks_the_user() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let gate = command(dir.path(), "gate", "gate.sh", ALLOW, &[]);
    let hooks = live(with_commands(vec![gate.clone()]), &tools).await;
    let (broker, queue) = ask_user::channel();
    let session = ExternalHooks::new(&hooks).with_user_pause(broker, "session-3");

    // The new gate has a typo: it must not let every command through.
    let typo = command(dir.path(), "typo", "typo.sh", BROKEN_SYNTAX, &[]);
    bounded(hooks.apply(with_commands(vec![gate, typo])))
        .await
        .unwrap();
    let running = tokio::spawn({
        let session = session.clone();
        async move { session.check_command("rm -rf /").await }
    });
    let question = bounded(queue.next()).await.expect("the user is asked");
    assert_eq!(question.args.session_id, "session-3");
    let text = question.args.request.question.clone().unwrap();
    assert!(
        text.contains("typo") && text.contains("syntax error"),
        "{text}"
    );
    assert!(!running.is_finished(), "the command waits for the user");

    // Fixing the script is picked up; the answer then retries the hook.
    rewrite(dir.path(), "typo.sh", ALLOW);
    eventually(|| async {
        hooks
            .set()
            .command
            .iter()
            .any(|entry| entry.config.name == "typo" && entry.current.is_some())
    })
    .await;
    question
        .respond(vec![QuestionAnswer {
            section: String::new(),
            answer: AnswerValue::Selected(vec!["Retry".into()]),
        }])
        .unwrap();
    assert_eq!(bounded(running).await.unwrap(), GateDecision::Allow);
}

#[tokio::test]
async fn without_a_user_a_new_hook_with_a_syntax_error_blocks_commands_and_passes_reviews() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(HooksConfig::default(), &tools).await;
    let warnings = hooks.warnings();
    bounded(hooks.apply(HooksConfig {
        command: vec![command(dir.path(), "typo", "typo.sh", BROKEN_SYNTAX, &[])],
        reviewers: vec![reviewer(dir.path(), "slip", "slip.sh", BROKEN_SYNTAX, &[])],
        ..HooksConfig::default()
    }))
    .await
    .unwrap();
    let session = ExternalHooks::new(&hooks);

    assert!(blocked_with(
        &bounded(session.check_command("ls")).await,
        "typo"
    ));
    let warning = warning_where(&warnings, |w| {
        matches!(
            w,
            HookWarning::RunFailed {
                kind: HookKind::Command,
                asked: false,
                ..
            }
        )
    })
    .await;
    assert!(
        warning.to_string().contains("nobody can be asked")
            && warning.to_string().contains("the command is blocked"),
        "{warning}"
    );
    let handler = ReviewedPermission::new(reviewers::Route::default(), session);
    assert_eq!(
        bounded(handler.review(&reviewers::COMMAND)).await,
        Review::Passed
    );
}

#[tokio::test]
async fn one_session_waiting_on_an_answer_does_not_hold_back_another_sessions_question() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let flag = dir.path().join("flag");
    std::fs::write(&flag, "").unwrap();
    let hooks = live(
        with_commands(vec![command(
            dir.path(),
            "gate",
            "gate.sh",
            GATE_UNLESS_FLAG,
            &[&flag],
        )]),
        &tools,
    )
    .await;
    let (broker_a, queue_a) = ask_user::channel();
    let (broker_b, queue_b) = ask_user::channel();
    let session_a = ExternalHooks::new(&hooks).with_user_pause(broker_a, "session-a");
    let session_b = ExternalHooks::new(&hooks).with_user_pause(broker_b, "session-b");
    let run_a = tokio::spawn(async move { session_a.check_command("ls").await });
    let run_b = tokio::spawn(async move { session_b.check_command("ls").await });

    // Session A's question stays open while session B is asked its own.
    let question_a = bounded(queue_a.next()).await.expect("session a is asked");
    let question_b = bounded(queue_b.next()).await.expect("session b is asked");
    assert_eq!(question_a.args.session_id, "session-a");
    assert_eq!(question_b.args.session_id, "session-b");

    std::fs::remove_file(&flag).unwrap();
    for question in [question_a, question_b] {
        question
            .respond(vec![QuestionAnswer {
                section: String::new(),
                answer: AnswerValue::Selected(vec!["Retry".into()]),
            }])
            .unwrap();
    }
    assert_eq!(bounded(run_a).await.unwrap(), GateDecision::Allow);
    assert_eq!(bounded(run_b).await.unwrap(), GateDecision::Allow);
}

#[tokio::test]
async fn a_script_in_a_directory_created_later_is_followed_once_it_exists() {
    let tools = crate::test_support::prepared_fixture().await;
    let base = tempfile::tempdir().unwrap();
    let gate = command(base.path(), "gate", "gate.sh", ALLOW, &[]);
    let hooks = live(with_commands(vec![gate.clone()]), &tools).await;
    let session = ExternalHooks::new(&hooks);

    // Nothing else is watched in this directory, and the hook's own is missing.
    let elsewhere = tempfile::tempdir().unwrap();
    let later = elsewhere.path().join("deeper").join("hooks");
    let late = CommandHookConfig {
        name: "late".into(),
        command: HookCommand {
            command: vec![
                "sh".into(),
                later.join("late.sh").to_string_lossy().into_owned(),
            ],
            deadline_ms: 20_000,
        },
    };
    bounded(hooks.apply(with_commands(vec![gate, late])))
        .await
        .unwrap();
    // Its script is not there yet: the gate it configures lets nothing through.
    assert!(blocked_with(
        &bounded(session.check_command("ls")).await,
        "late"
    ));

    std::fs::create_dir_all(&later).unwrap();
    script(&later, "late.sh", BLOCK);
    eventually(|| async { blocked_with(&session.check_command("ls").await, "second version") })
        .await;
}

#[tokio::test]
async fn an_edit_of_the_target_of_a_symlinked_script_is_followed() {
    let tools = crate::test_support::prepared_fixture().await;
    let base = tempfile::tempdir().unwrap();
    let real = base.path().join("real");
    let links = base.path().join("links");
    std::fs::create_dir_all(&real).unwrap();
    std::fs::create_dir_all(&links).unwrap();
    script(&real, "real.sh", ALLOW);
    let link = links.join("link.sh");
    std::os::unix::fs::symlink(real.join("real.sh"), &link).unwrap();
    let hooks = live(
        with_commands(vec![CommandHookConfig {
            name: "gate".into(),
            command: HookCommand {
                command: vec!["sh".into(), link.to_string_lossy().into_owned()],
                deadline_ms: 20_000,
            },
        }]),
        &tools,
    )
    .await;
    let session = ExternalHooks::new(&hooks);
    assert_eq!(
        bounded(session.check_command("ls")).await,
        GateDecision::Allow
    );

    rewrite(&real, "real.sh", BLOCK);

    eventually(|| async { blocked_with(&session.check_command("ls").await, "second version") })
        .await;
}

#[tokio::test]
async fn a_fault_is_warned_once_and_unread_warnings_are_bounded() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(HooksConfig::default(), &tools).await;
    let warnings = hooks.warnings();
    let typo = |name: &str| command(dir.path(), name, "typo.sh", BROKEN_SYNTAX, &[]);

    // Reloads that find the same fault do not raise it again.
    for _ in 0..3 {
        bounded(hooks.apply(with_commands(vec![typo("typo")])))
            .await
            .unwrap();
    }
    assert_eq!(warnings.len(), 1);
    warnings.try_recv().unwrap();

    // A host that never reads loses the oldest warnings, not memory.
    let bound = warnings.capacity().expect("the warnings are bounded");
    for index in 0..bound + 40 {
        bounded(hooks.apply(with_commands(vec![typo(&format!("typo-{index}"))])))
            .await
            .unwrap();
    }
    assert_eq!(warnings.len(), bound);
    let mut last = None;
    while let Ok(warning) = warnings.try_recv() {
        last = Some(warning);
    }
    assert!(
        matches!(last, Some(HookWarning::Rejected { hook, .. }) if hook == format!("typo-{}", bound + 39))
    );
}

#[test]
fn a_directory_removed_and_recreated_is_watched_exactly_once() {
    let base = tempfile::tempdir().unwrap();
    let directory = base.path().join("hooks");
    std::fs::create_dir(&directory).unwrap();
    let file = directory.join("gate.sh");
    let watcher = notify::recommended_watcher(|_: notify::Result<notify::Event>| {}).unwrap();
    let mut watching = Watching::new(watcher);
    assert!(watching.follow_paths(std::slice::from_ref(&file)));
    assert_eq!(watching.watched, HashSet::from([directory.clone()]));

    // The directory goes: it is forgotten, and queued to be unwatched.
    std::fs::remove_dir(&directory).unwrap();
    let removal = notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::Folder))
        .add_path(directory.clone());
    assert!(watching.concerns(&removal));
    assert!(watching.watched.is_empty());
    assert_eq!(watching.stale, HashSet::from([directory.clone()]));

    // It comes back: the stale watch is dropped and the directory is watched
    // once, not on top of the old watch.
    std::fs::create_dir(&directory).unwrap();
    assert!(watching.follow_paths(std::slice::from_ref(&file)));
    assert!(watching.stale.is_empty());
    assert_eq!(watching.watched, HashSet::from([directory]));

    // Nothing changed since: the next retarget does nothing.
    assert!(!watching.follow_paths(std::slice::from_ref(&file)));
}

#[tokio::test]
async fn a_hook_that_stays_broken_with_nobody_to_ask_is_warned_once_until_it_changes() {
    let tools = crate::test_support::prepared_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let hooks = live(HooksConfig::default(), &tools).await;
    let warnings = hooks.warnings();
    bounded(hooks.apply(with_commands(vec![command(
        dir.path(),
        "typo",
        "typo.sh",
        BROKEN_SYNTAX,
        &[],
    )])))
    .await
    .unwrap();
    let session = ExternalHooks::new(&hooks);
    let run_failures = || {
        let mut count = 0;
        while let Ok(warning) = warnings.try_recv() {
            count += usize::from(matches!(warning, HookWarning::RunFailed { .. }));
        }
        count
    };

    for _ in 0..3 {
        assert!(blocked_with(
            &bounded(session.check_command("ls")).await,
            "typo"
        ));
    }
    assert_eq!(run_failures(), 1);

    // New text is a new fault, and is warned of once in turn.
    rewrite(dir.path(), "typo.sh", "#!/bin/sh\necho 'unterminated\n");
    eventually(|| async {
        let _ = session.check_command("ls").await;
        run_failures() == 1
    })
    .await;
    for _ in 0..3 {
        let _ = bounded(session.check_command("ls")).await;
    }
    assert_eq!(run_failures(), 0);
}
