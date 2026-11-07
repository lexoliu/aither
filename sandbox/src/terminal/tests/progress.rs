//! A command's progress reports reach its terminal job's output, in order,
//! before the command's result.

use aither_core::llm::tool::Progress;

use super::*;

/// Reports two steps of progress, waits for the test to release it when
/// `hold` is set, then answers `RESULT`.
struct ProgressProbe {
    hold: Option<Receiver<()>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ProgressProbeArgs {}

impl Tool for ProgressProbe {
    fn name(&self) -> Cow<'static, str> {
        "progress_probe".into()
    }

    type Arguments = ProgressProbeArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        _arguments: Self::Arguments,
        mut cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        cx.report_progress(Progress::new(1.0).with_total(2.0).with_message("first"))
            .await?;
        cx.report_progress(Progress::new(2.0).with_total(2.0).with_message("second"))
            .await?;
        if let Some(hold) = &self.hold {
            hold.recv().await.unwrap();
        }
        Ok(ToolResult::text("RESULT"))
    }
}

const FIRST: &str = "progress: 1/2 - first";
const SECOND: &str = "progress: 2/2 - second";

async fn terminal(
    dir: &Path,
    probe: ProgressProbe,
) -> (
    crate::tool_runtime::tests::UnlockingTempDir,
    ConfiguredTestTool,
) {
    let (cache, tool) = unconfigured_test_terminal(
        dir,
        Duration::from_secs(20),
        Duration::from_secs(60),
        Duration::from_secs(600),
    )
    .await;
    let mut builder = ToolRegistryBuilder::new();
    builder.configure_tool(probe);
    let registry = Arc::new(builder.build(tool.output_store().clone()));
    (cache, tool.with_registry(registry))
}

/// The positions of `needles` in `text`, which must hold each of them.
fn positions(text: &str, needles: &[&str]) -> Vec<usize> {
    needles
        .iter()
        .map(|needle| {
            text.find(needle)
                .unwrap_or_else(|| panic!("`{needle}` missing from: {text}"))
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_foreground_call_shows_progress_before_the_result() {
    let dir = tempfile::tempdir().expect("temp dir should be created");
    let (_cache, tool) = terminal(dir.path(), ProgressProbe { hold: None }).await;

    let result = tool
        .call(
            terminal_args("progress probe", "progress_probe"),
            ToolContext::new(),
        )
        .await
        .expect("terminal call should succeed");
    let command = result.command().expect("a command result");
    assert_eq!(command.outcome, CommandOutcome::Exited { code: 0 });
    let stdout = output_text(&command.stdout).expect("text output");

    let at = positions(stdout, &[FIRST, SECOND, "RESULT"]);
    assert!(at.is_sorted(), "reports out of order: {stdout}");
    assert!(
        stdout.contains(&format!("{FIRST}\n{SECOND}\n")),
        "each report is its own line: {stdout:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_background_job_holds_progress_while_it_runs() {
    let dir = tempfile::tempdir().expect("temp dir should be created");
    let (release, hold) = async_channel::bounded(1);
    let (_cache, tool) = terminal(dir.path(), ProgressProbe { hold: Some(hold) }).await;
    let task_id = start_background_job(&tool, "progress_probe").await;

    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let read = loop {
        let read = query_control(tool.registry(), "terminal_read", &["--task-id", &task_id])
            .await
            .to_string();
        if read.contains(SECOND) {
            break read;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the job never showed its progress: {read}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let at = positions(&read, &[FIRST, SECOND]);
    assert!(at.is_sorted(), "reports out of order: {read}");
    assert!(
        !read.contains("RESULT"),
        "the result arrived while the command was held: {read}"
    );

    release.send(()).await.unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let read = query_control(tool.registry(), "terminal_read", &["--task-id", &task_id])
            .await
            .to_string();
        if read.contains("RESULT") {
            let at = positions(&read, &[FIRST, SECOND, "RESULT"]);
            assert!(at.is_sorted(), "result before progress: {read}");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the job never finished: {read}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
