//! Command hooks driven by a real hook script that runs `ast-grep`, through
//! a real terminal tool.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::{Tool, ToolContext, ToolResult};
use aither_sandbox::permission::{PermissionError, PermissionHandler, TerminalMode};
use aither_sandbox::{
    DynTerminalTool, DynToolHandler, SubagentGrant, TerminalTool, ToolRegistryBuilder,
};
use executor_core::tokio::TokioGlobal;

use super::tests::{logged, script};
use super::*;

/// Longest any wait in these tests lasts.
const BOUND: Duration = Duration::from_secs(60);

/// Reads the command from the hook input, logs the input to the file named
/// by the first argument, and asks `ast-grep` whether the command contains
/// `rm -rf`, wherever the shell syntax nests it: a match blocks, a clean
/// run allows, and any other exit status makes the hook itself fail.
const AST_GREP_HOOK: &str = r#"#!/bin/sh
input=$(cat)
printf '%s\n' "$input" >> "$1"
command=$(printf '%s' "$input" | jq -r .command)
printf '%s' "$command" | ast-grep run --lang bash --pattern 'rm -rf $$$ARGS' --stdin >/dev/null
case $? in
  0) printf '{"decision":"block","reason":"rm -rf is not allowed here"}' ;;
  1) printf '{"decision":"allow"}' ;;
  *) exit 3 ;;
esac
"#;

/// A permission handler that grants the sandbox and nothing beyond it, and
/// records every request it is asked.
#[derive(Clone, Default)]
struct SandboxOnly {
    asked: Arc<Mutex<Vec<(TerminalMode, String)>>>,
}

impl PermissionHandler for SandboxOnly {
    fn check(
        &self,
        mode: TerminalMode,
        script: &str,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send {
        self.asked.lock().unwrap().push((mode, script.to_string()));
        std::future::ready(Ok(mode == TerminalMode::Sandboxed))
    }
}

fn config(name: &str, program: &[&str], deadline_ms: u64) -> CommandHookConfig {
    CommandHookConfig {
        name: name.into(),
        command: HookCommand {
            command: program.iter().map(ToString::to_string).collect(),
            deadline_ms,
        },
    }
}

/// The ast-grep hook, logging its inputs to `log`.
fn ast_grep_hook(dir: &Path, log: &Path) -> CommandHookConfig {
    let hook = script(dir, "ast-grep-hook.sh", AST_GREP_HOOK);
    config(
        "ast-grep",
        &["sh", &hook.to_string_lossy(), &log.to_string_lossy()],
        20_000,
    )
}

/// The managed tools every terminal here is prepared with.
struct Tools {
    distribution: aither_sandbox::ToolDistribution,
    cache: tempfile::TempDir,
    _source: tempfile::TempDir,
}

impl Tools {
    /// The baseline fixture the crate's other terminal tests use, which only
    /// makes sandboxed execution available.
    async fn stub() -> Self {
        let source = tempfile::tempdir().unwrap();
        let binary = crate::test_support::compile_baseline_fixture(source.path()).await;
        Self {
            distribution: crate::test_support::fixture_distribution(&binary, "1.0.0").await,
            cache: tempfile::tempdir().unwrap(),
            _source: source,
        }
    }

    /// The real `ripgrep`, `ast-grep` and `jq` of the repository's pinned
    /// distribution, downloaded and verified into the cache.
    async fn real() -> Self {
        let source = tempfile::tempdir().unwrap();
        let binary = crate::test_support::compile_baseline_fixture(source.path()).await;
        Self {
            distribution: crate::test_support::real_distribution(&binary).await,
            cache: tempfile::tempdir().unwrap(),
            _source: source,
        }
    }
}

/// The terminal of the main agent, gated by command hooks that run with the
/// terminal's own prepared tools on their `PATH`, and those hooks.
async fn terminal(
    tools: &Tools,
    dir: &Path,
    permission: SandboxOnly,
    command_hooks: Vec<CommandHookConfig>,
) -> (
    TerminalTool<SandboxOnly, TokioGlobal, aither_sandbox::Configured>,
    ExternalHooks,
) {
    let terminal = TerminalTool::new_exact(dir, permission, TokioGlobal)
        .await
        .expect("the terminal initializes")
        .initialize_tools(&tools.distribution, tools.cache.path())
        .await
        .expect("the managed tools are prepared");
    let hooks = ExternalHooks::new(
        Vec::new(),
        executor_core::AnyExecutor::new(TokioGlobal),
        &terminal
            .shell_sessions()
            .host_tools(&tools.distribution, tools.cache.path())
            .await
            .expect("the host tools are prepared"),
    )
    .with_command_hooks(command_hooks);
    let mut builder = ToolRegistryBuilder::new();
    builder.gate_commands(hooks.command_gate());
    let registry = Arc::new(builder.build(terminal.outputs_dir()));
    (terminal.with_registry(registry), hooks)
}

fn args(script: &str, mode: &str) -> serde_json::Value {
    serde_json::json!({
        "description": "run a test command",
        "script": script,
        "mode": mode,
        "raw": true,
    })
}

/// What the model reads for a foreground `terminal` call, or the refusal
/// that replaced it.
async fn run<P>(
    terminal: &TerminalTool<P, TokioGlobal, aither_sandbox::Configured>,
    script: &str,
    mode: &str,
) -> Result<String, String>
where
    P: PermissionHandler + 'static,
{
    let arguments = serde_json::from_value(args(script, mode)).unwrap();
    tokio::time::timeout(BOUND, terminal.call(arguments, ToolContext::new()))
        .await
        .expect("the call finishes within its bound")
        .map(|result| result.render_for_model())
        .map_err(|error| error.to_string())
}

/// The `terminal` handler of a subagent's terminal.
fn terminal_handler(terminal: DynTerminalTool) -> DynToolHandler {
    terminal
        .into_entries()
        .into_iter()
        .find(|entry| entry.definition.name() == "terminal")
        .expect("the subagent has a terminal")
        .handler
}

/// Calls a subagent's `terminal` handler.
async fn run_dyn(handler: &DynToolHandler, script: &str) -> String {
    let result: ToolResult = tokio::time::timeout(
        BOUND,
        handler(&args(script, "sandboxed").to_string(), ToolContext::new()),
    )
    .await
    .expect("the call finishes within its bound");
    result.render_for_model()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_ast_grep_hook_blocks_nested_matches_and_allows_a_non_match() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let work = tempfile::tempdir().unwrap();
    let tools = Tools::real().await;
    let (terminal, _) = terminal(
        &tools,
        work.path(),
        SandboxOnly::default(),
        vec![ast_grep_hook(dir.path(), &log)],
    )
    .await;

    for nested in [
        "echo go | rm -rf /nowhere/pipeline",
        "(cd /nowhere && rm -rf subshell)",
        "echo $(rm -rf /nowhere/substitution)",
        "rm -rf /nowhere/plain",
    ] {
        let refusal = run(&terminal, nested, "sandboxed")
            .await
            .expect_err(&format!("`{nested}` is blocked"));
        assert!(
            refusal.contains("ast-grep") && refusal.contains("rm -rf is not allowed here"),
            "{refusal}"
        );
    }

    let output = run(
        &terminal,
        "echo rm -rf is only text; echo ran-ok",
        "sandboxed",
    )
    .await
    .expect("a non-match runs");
    assert!(output.contains("ran-ok"), "{output}");

    // The hook saw each full command, once, as the main agent.
    let inputs = logged(&log);
    assert_eq!(inputs.len(), 5);
    assert_eq!(
        inputs[0],
        serde_json::json!({
            "event": "before_command", "role": "main",
            "command": "echo go | rm -rf /nowhere/pipeline"
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_background_command_is_gated_too() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let work = tempfile::tempdir().unwrap();
    let tools = Tools::real().await;
    let (terminal, _) = terminal(
        &tools,
        work.path(),
        SandboxOnly::default(),
        vec![ast_grep_hook(dir.path(), &log)],
    )
    .await;

    let mut background = args("echo $(rm -rf /nowhere)", "sandboxed");
    background["timeout"] = 30.into();
    let arguments = serde_json::from_value(background).unwrap();
    let refusal = tokio::time::timeout(
        BOUND,
        terminal
            .background_tool()
            .call(arguments, ToolContext::new()),
    )
    .await
    .expect("the call finishes within its bound")
    .expect_err("the background command is blocked");
    assert!(
        refusal.to_string().contains("rm -rf is not allowed here"),
        "{refusal}"
    );
    assert_eq!(logged(&log).len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn allowing_a_command_grants_no_permission_beyond_the_sandbox() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let work = tempfile::tempdir().unwrap();
    let tools = Tools::real().await;
    let permission = SandboxOnly::default();
    let (terminal, _) = terminal(
        &tools,
        work.path(),
        permission.clone(),
        vec![ast_grep_hook(dir.path(), &log)],
    )
    .await;

    // The hook allows this command, yet it still needs the permission an
    // unsandboxed command needs, which is refused.
    let refusal = run(&terminal, "echo escape-attempt", "unsafe")
        .await
        .expect_err("the hook's allow does not stand in for permission");
    assert!(!refusal.contains("ast-grep"), "{refusal}");
    assert_eq!(logged(&log).len(), 1, "the hook allowed it");
    assert_eq!(
        *permission.asked.lock().unwrap(),
        [(TerminalMode::Unsafe, "echo escape-attempt".to_string())],
        "the permission path still decided"
    );

    // An answer that tries to carry a grant is not an answer at all.
    let grant = script(
        dir.path(),
        "grant.sh",
        "#!/bin/sh\ncat >/dev/null\nprintf '{\"decision\":\"allow\",\"grant\":\"unsafe\"}'\n",
    );
    let permission = SandboxOnly::default();
    let (terminal, _) = self::terminal(
        &tools,
        work.path(),
        permission.clone(),
        vec![config(
            "granting",
            &["sh", &grant.to_string_lossy()],
            20_000,
        )],
    )
    .await;
    let refusal = run(&terminal, "echo escape-attempt", "unsafe")
        .await
        .expect_err("a grant is refused");
    assert!(refusal.contains("did not answer"), "{refusal}");
    assert!(permission.asked.lock().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_command_hook_that_fails_blocks_the_command_naming_the_failure() {
    let dir = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let tools = Tools::stub().await;
    let silent = script(dir.path(), "silent.sh", "#!/bin/sh\ncat >/dev/null\n");
    let nonsense = script(
        dir.path(),
        "nonsense.sh",
        "#!/bin/sh\ncat >/dev/null\nprintf 'allow'\n",
    );
    let crash = script(
        dir.path(),
        "crash.sh",
        "#!/bin/sh\ncat >/dev/null\necho broken >&2\nexit 7\n",
    );
    let slow = script(
        dir.path(),
        "slow.sh",
        "#!/bin/sh\ncat >/dev/null\nsleep 30\n",
    );
    let cases = [
        (
            config("missing", &["/nonexistent/hook"], 20_000),
            "cannot be started",
        ),
        (
            config("silent", &["sh", &silent.to_string_lossy()], 20_000),
            "printed no answer",
        ),
        (
            config("nonsense", &["sh", &nonsense.to_string_lossy()], 20_000),
            "not a valid answer",
        ),
        (
            config("crash", &["sh", &crash.to_string_lossy()], 20_000),
            "broken",
        ),
        (
            config("slow", &["sh", &slow.to_string_lossy()], 300),
            "deadline",
        ),
    ];
    for (hook, failure) in cases {
        let name = hook.name.clone();
        let (terminal, _) = terminal(&tools, work.path(), SandboxOnly::default(), vec![hook]).await;
        let refusal = run(&terminal, "echo should-not-run", "sandboxed")
            .await
            .expect_err(&format!("the failing hook `{name}` blocks"));
        assert!(
            refusal.contains(&name)
                && refusal.contains("did not answer")
                && refusal.contains(failure),
            "{name}: {refusal}"
        );
        assert!(!refusal.contains("should-not-run\n"), "{refusal}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_subagent_terminal_is_gated_and_its_hooks_see_the_subagent_role() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("log");
    let work = tempfile::tempdir().unwrap();
    let tools = Tools::real().await;
    let (terminal, hooks) = terminal(
        &tools,
        work.path(),
        SandboxOnly::default(),
        vec![ast_grep_hook(dir.path(), &log)],
    )
    .await;

    let child_hooks = hooks.for_subagent();
    let subagent = terminal
        .binding()
        .launcher()
        .create(
            SubagentGrant::new(),
            Some(child_hooks.command_observer()),
            Some(child_hooks.command_gate()),
        )
        .await
        .expect("the subagent terminal launches");
    let subagent = terminal_handler(subagent);

    let refusal = run_dyn(&subagent, "(rm -rf /nowhere)").await;
    assert!(refusal.contains("rm -rf is not allowed here"), "{refusal}");
    let ran = run_dyn(&subagent, "echo child-ok").await;
    assert!(ran.contains("child-ok"), "{ran}");

    let roles: Vec<_> = logged(&log)
        .iter()
        .map(|input| input["role"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(roles, ["subagent", "subagent"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hook_finds_the_terminals_prepared_tools_ahead_of_the_host_path() {
    let dir = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let tools = Tools::stub().await;
    let which = script(
        dir.path(),
        "which.sh",
        r#"#!/bin/sh
cat >/dev/null
for tool in apply_patch rg ast-grep jq; do
  printf '%s=%s\n' "$tool" "$(command -v "$tool")"
done > "$1"
printf '{"decision":"allow"}'
"#,
    );
    let found = dir.path().join("found");
    let (terminal, _) = terminal(
        &tools,
        work.path(),
        SandboxOnly::default(),
        vec![config(
            "which",
            &["sh", &which.to_string_lossy(), &found.to_string_lossy()],
            20_000,
        )],
    )
    .await;
    let prepared = terminal.shell_sessions().local_tools().unwrap().bin_dir();

    run(&terminal, "echo ok", "sandboxed")
        .await
        .expect("the hook allows");

    let found = std::fs::read_to_string(&found).unwrap();
    for tool in ["apply_patch", "rg", "ast-grep", "jq"] {
        let expected = format!("{tool}={}", prepared.join(tool).display());
        assert!(found.lines().any(|line| line == expected), "{found}");
    }
}
