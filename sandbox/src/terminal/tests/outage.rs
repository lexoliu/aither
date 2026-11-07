//! Commands state the sleep and offline intervals they ran through.

use aither_connectivity::{Connectivity, Controller};
use aither_core::llm::InterruptionCause;

use super::*;

type Terminal = TerminalTool<TestPermissionHandler, executor_core::tokio::TokioGlobal, Configured>;

/// A terminal recording `connectivity`; the cache directory must outlive it.
async fn terminal(
    dir: &Path,
    connectivity: &Connectivity,
) -> (crate::tool_runtime::tests::UnlockingTempDir, Terminal) {
    let (cache, shell_sessions) = prepared_test_registry().await;
    let tool = TerminalTool::new_exact(
        dir,
        TestPermissionHandler::default(),
        executor_core::tokio::TokioGlobal,
    )
    .await
    .expect("terminal tool should initialize")
    .adopt_prepared_tools(shell_sessions)
    .await
    .expect("prepared tools should be adopted")
    .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
        local: true,
        container: false,
        ssh: false,
    })
    .with_connectivity(connectivity);
    let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
    (cache, tool.with_registry(registry))
}

/// A script that announces itself, then spins on the test's release file (a
/// foreground `sleep` is refused).
fn held_script(tool: &Terminal) -> String {
    let started = tool.working_dir().join("started");
    let release = tool.working_dir().join("release");
    format!(
        "touch '{}'; while [ ! -f '{}' ]; do :; done; echo done",
        started.display(),
        release.display()
    )
}

/// Runs the held script, and `during` on its own task once the script runs.
async fn run_held(
    tool: &Terminal,
    background: bool,
    during: impl FnOnce() + Send + 'static,
) -> aither_core::Result<ToolResult> {
    let started = tool.working_dir().join("started");
    let release = tool.working_dir().join("release");
    tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !started.exists() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the script should start");
        during();
        std::fs::write(release, "").expect("release file");
    });
    let args = TerminalArgs {
        background,
        ..background_args("hold", &held_script(tool))
    };
    tool.call(args, ToolContext::new()).await
}

fn has_interruption_line(rendered: &str, line_start: &str) -> bool {
    rendered
        .lines()
        .any(|line| line.starts_with(line_start) && line.ends_with(" while this ran."))
}

fn sleep_and_wake(controller: &Controller) -> impl FnOnce() + Send + 'static {
    let controller = controller.clone();
    move || {
        controller.sleep();
        controller.wake();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_foreground_command_spanning_sleep_and_wake_reports_it() {
    let dir = tempfile::tempdir().unwrap();
    let (connectivity, controller) = Connectivity::manual(true);
    let (_cache, tool) = terminal(dir.path(), &connectivity).await;

    let result = run_held(&tool, false, sleep_and_wake(&controller))
        .await
        .expect("terminal call should succeed");

    let rendered = result.render_for_model();
    assert!(
        has_interruption_line(&rendered, "The machine was asleep from "),
        "{rendered}"
    );
    assert!(rendered.starts_with("done\n"), "{rendered}");
    assert!(!rendered.contains("offline"), "{rendered}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_foreground_command_spanning_offline_and_online_reports_it() {
    let dir = tempfile::tempdir().unwrap();
    let (connectivity, controller) = Connectivity::manual(true);
    let (_cache, tool) = terminal(dir.path(), &connectivity).await;

    let result = run_held(&tool, false, {
        let controller = controller.clone();
        move || {
            controller.set_online(false);
            controller.set_online(true);
        }
    })
    .await
    .expect("terminal call should succeed");

    let rendered = result.render_for_model();
    assert!(
        has_interruption_line(&rendered, "The network was offline from "),
        "{rendered}"
    );
    assert!(!rendered.contains("asleep"), "{rendered}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_outage_still_open_when_the_command_ends_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let (connectivity, controller) = Connectivity::manual(true);
    let (_cache, tool) = terminal(dir.path(), &connectivity).await;

    let result = run_held(&tool, false, {
        let controller = controller.clone();
        move || controller.set_online(false)
    })
    .await
    .expect("terminal call should succeed");

    let rendered = result.render_for_model();
    assert!(
        has_interruption_line(&rendered, "The network was offline from "),
        "{rendered}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_command_with_no_interruption_renders_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let (connectivity, controller) = Connectivity::manual(true);
    let (_cache, tool) = terminal(dir.path(), &connectivity).await;
    // An outage that is over before the command starts is not its own.
    controller.sleep();
    controller.wake();

    let result = run_held(&tool, false, || ())
        .await
        .expect("terminal call should succeed");

    assert_eq!(result.render_for_model(), "done\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_background_jobs_completion_carries_the_interval() {
    let dir = tempfile::tempdir().unwrap();
    let (connectivity, controller) = Connectivity::manual(true);
    let (_cache, tool) = terminal(dir.path(), &connectivity).await;
    let receiver = tool.background_receiver();

    run_held(&tool, true, sleep_and_wake(&controller))
        .await
        .expect("terminal call should succeed");
    let task_id = started_task_id(&receiver);
    let completed = wait_for_completed_task(&receiver, &task_id).await;

    let command = completed.result.expect("the job should complete");
    assert_eq!(command.interruptions.len(), 1);
    assert_eq!(command.interruptions[0].cause, InterruptionCause::Asleep);
}
