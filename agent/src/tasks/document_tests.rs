use super::*;
use crate::tasks::WorkComment;
use crate::tasks::tracker::test_tracker::{FixedTracker, item, related};

async fn write_tasks(root: &std::path::Path, content: &str) {
    async_fs::write(root.join("tasks.md"), content)
        .await
        .expect("write tasks.md");
}

fn document(root: &std::path::Path) -> (TasksDocument, TaskEvents) {
    let events = TaskEvents::new();
    (TasksDocument::new(root, 60, events.clone()), events)
}

/// One model turn that ran tools: the request, then its end.
async fn tool_turn(source: &mut TasksDocument) -> Option<Block> {
    source.on_request(false).await;
    source.on_tool_turn_end().await
}

struct Answered;
impl crate::working_docs::TaskBlockerResolver for Answered {
    fn is_open(&self, mark: &str) -> impl std::future::Future<Output = bool> + Send {
        // Mirrors a host resolver: only issued-and-still-open marks hold;
        // unknown payloads are free text and trusted.
        std::future::ready(!matches!(mark, "req-answered" | "req-withdrawn"))
    }
}

#[tokio::test]
async fn a_session_with_nothing_to_track_never_gets_an_alert() {
    let root = tempfile::tempdir().expect("tempdir");
    let (mut source, _events) = document(root.path());
    for _ in 0..20 {
        assert!(source.on_request(false).await.is_none());
        assert!(source.on_tool_turn_end().await.is_none());
    }
    assert!(source.on_final_turn().await.is_none());
}

#[tokio::test]
async fn a_current_document_gets_no_alert_however_many_turns_run() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Roots\n- #1 search\n## Notes\nall done\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");
    for _ in 0..20 {
        assert!(tool_turn(&mut source).await.is_none());
    }
    assert!(source.on_final_turn().await.is_none());
}

#[tokio::test]
async fn the_session_start_carries_the_whole_document() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Doing\n- first\n").await;
    let (mut source, _events) = document(root.path());

    let block = source.on_request(false).await.expect("shows the document");
    assert!(block.content().contains("Contents of tasks.md"));
    assert!(block.content().contains("- first"));
    assert!(!block.content().contains("Changes to tasks.md"));
    assert!(
        source.on_request(false).await.is_none(),
        "unchanged adds nothing"
    );
}

#[tokio::test]
async fn a_change_is_shown_once_as_its_diff_and_a_reassembly_shows_it_whole() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Doing\n- first\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");

    write_tasks(root.path(), "## Doing\n- second\n").await;
    let block = source.on_request(false).await.expect("shows the diff");
    assert!(block.content().contains("Changes to tasks.md"));
    assert!(block.content().contains("+ - second"));
    assert!(!block.content().contains("Contents of tasks.md"));
    assert!(source.on_request(false).await.is_none());

    let block = source.on_request(true).await.expect("shows the document");
    assert!(block.content().contains("Contents of tasks.md"));
}

#[tokio::test]
async fn an_absent_document_is_never_shown() {
    let root = tempfile::tempdir().expect("tempdir");
    let (mut source, _events) = document(root.path());
    assert!(source.on_request(false).await.is_none());
    assert!(source.on_request(true).await.is_none());
    assert!(source.render().await.is_none());
}

#[tokio::test]
async fn a_merged_pr_still_listed_alerts_once_naming_it_and_its_row() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Doing\n- #129 ship it, PR #12 open\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    // The merge happens during a turn's tools; the model sees it at the
    // next request and has that turn to record it.
    events.record(TaskEvent::PullRequestMerged(WorkRef::pull_request(12)));
    let alert = tool_turn(&mut source).await.expect("one alert");
    assert_eq!(
        alert.content(),
        "<tasks_alert>\nPR #12 was merged, and tasks.md still says: \"#129 ship it, PR #12 open\"\n</tasks_alert>"
    );
    assert!(
        tool_turn(&mut source).await.is_none(),
        "the alert is one-shot"
    );
}

#[tokio::test]
async fn an_event_the_turn_records_raises_nothing() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Doing\n- #129 ship it, PR #12 open\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(TaskEvent::PullRequestMerged(WorkRef::pull_request(12)));
    source.on_request(false).await;
    write_tasks(root.path(), "## Doing\n- #129 follow-up review\n").await;
    assert!(source.on_tool_turn_end().await.is_none());
}

#[tokio::test]
async fn each_tracked_event_left_unrecorded_yields_exactly_one_alert_naming_it() {
    let cases = [
        (
            TaskEvent::IssueCreated(WorkRef::issue(5)),
            "Issue #5 was created",
        ),
        (
            TaskEvent::IssueClosed(WorkRef::issue(5)),
            "Issue #5 was closed",
        ),
        (
            TaskEvent::PullRequestOpened(WorkRef::pull_request(7)),
            "PR #7 was opened",
        ),
        (
            TaskEvent::PullRequestMerged(WorkRef::pull_request(7)),
            "PR #7 was merged",
        ),
        (
            TaskEvent::PullRequestRejected(WorkRef::pull_request(7)),
            "PR #7 was rejected",
        ),
        (
            TaskEvent::PullRequestClosed(WorkRef::pull_request(7)),
            "PR #7 was closed",
        ),
        (
            TaskEvent::QuestionAsked("req-1".into()),
            "Question req-1 was asked",
        ),
        (
            TaskEvent::QuestionAnswered("req-1".into()),
            "Question req-1 was answered",
        ),
        (
            TaskEvent::QuestionWithdrawn("req-1".into()),
            "Question req-1 was withdrawn",
        ),
        (
            TaskEvent::SubagentStarted("reviewer".into()),
            "Subagent reviewer started",
        ),
        (
            TaskEvent::SubagentFinished("reviewer".into()),
            "Subagent reviewer finished",
        ),
        (
            TaskEvent::JobFinished("bg-3".into()),
            "Background job bg-3 finished",
        ),
    ];
    let row = "#5 land PR #7 after the reviewer subagent, bg-3 builds BLOCKING req-1";
    for (event, phrase) in cases {
        let root = tempfile::tempdir().expect("tempdir");
        write_tasks(root.path(), &format!("## Doing\n- {row}\n")).await;
        let (mut source, events) = document(root.path());
        source.on_request(false).await.expect("session start");

        events.record(event);
        let alert = tool_turn(&mut source).await.expect("alert");
        assert_eq!(
            alert.content().matches(phrase).count(),
            1,
            "{}",
            alert.content()
        );
        assert!(alert.content().contains(row), "{}", alert.content());
        assert!(
            tool_turn(&mut source).await.is_none(),
            "{phrase} alerts once"
        );
    }
}

#[tokio::test]
async fn an_event_no_row_names_alerts_only_when_it_opens_work() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Doing\n- #1 search\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(TaskEvent::JobFinished("bg-9".into()));
    events.record(TaskEvent::SubagentFinished("explore".into()));
    events.record(TaskEvent::PullRequestMerged(WorkRef::pull_request(4)));
    assert!(tool_turn(&mut source).await.is_none());

    events.record(TaskEvent::IssueCreated(WorkRef::issue(5)));
    let alert = tool_turn(&mut source).await.expect("new work alerts");
    assert!(
        alert
            .content()
            .contains("Issue #5 was created; no row of tasks.md names it."),
        "{}",
        alert.content()
    );
}

#[tokio::test]
async fn an_event_with_no_document_alerts_only_when_it_opens_work() {
    let root = tempfile::tempdir().expect("tempdir");
    let (mut source, events) = document(root.path());
    source.on_request(false).await;
    events.record(TaskEvent::JobFinished("bg-1".into()));
    assert!(tool_turn(&mut source).await.is_none());
    events.record(TaskEvent::PullRequestOpened(WorkRef::pull_request(2)));
    assert!(tool_turn(&mut source).await.is_some());
}

#[tokio::test]
async fn an_unrecorded_event_at_the_end_of_the_turn_keeps_the_run_going() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Roots\n- #1 search, PR #12 open\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(TaskEvent::PullRequestMerged(WorkRef::pull_request(12)));
    source.on_request(false).await;
    let alert = source.on_final_turn().await.expect("the gate holds");
    assert!(alert.content().contains("PR #12 was merged"));
    assert!(alert.content().contains("before ending the turn"));
    assert!(
        source.on_final_turn().await.is_none(),
        "the event alerts once"
    );
}

#[tokio::test]
async fn a_malformed_document_holds_the_end_of_the_turn() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "plan\n## Doing\n- [ ] old style\n## Plans\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");

    let alert = source.on_final_turn().await.expect("the gate holds");
    assert_eq!(
        alert.content(),
        "<tasks_alert>\nLine 1 of tasks.md is outside every section.\nLine 3 of tasks.md is a checkbox; rows are not ticked, a finished row is deleted.\nLine 4 of tasks.md starts `## Plans`, which is no section.\nBring tasks.md up to date before ending the turn.\n</tasks_alert>"
    );
    assert!(
        source.on_final_turn().await.is_some(),
        "every attempt to end the turn meets the gate"
    );

    write_tasks(root.path(), "## Notes\nfixed\n").await;
    assert!(source.on_final_turn().await.is_none());
}

#[tokio::test]
async fn an_edit_that_breaks_the_format_alerts_once() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Doing\n- fine\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");

    write_tasks(root.path(), "## Doing\n- fine\nloose line\n").await;
    let alert = tool_turn(&mut source).await.expect("the edit alerts");
    assert!(
        alert
            .content()
            .contains("Line 3 of tasks.md is not a `- ` row.")
    );
    assert!(!alert.content().contains("before ending the turn"));
    assert!(
        tool_turn(&mut source).await.is_none(),
        "unchanged, no repeat"
    );
}

#[tokio::test]
async fn a_row_naming_a_finished_item_holds_the_end_of_the_turn() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Roots\n- #1 search, PR #12 open\n").await;
    let tracker = FixedTracker::default()
        .with(item(WorkRef::issue(1), WorkState::Open))
        .with(item(WorkRef::pull_request(12), WorkState::Open));
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker.clone()));
    source.on_request(false).await.expect("session start");
    assert!(source.on_final_turn().await.is_none());

    tracker.set_state(&WorkRef::pull_request(12), WorkState::Merged);
    let alert = source.on_final_turn().await.expect("the gate holds");
    assert!(
        alert
            .content()
            .contains("The row \"#1 search, PR #12 open\" names PR #12, which is merged."),
        "{}",
        alert.content()
    );

    write_tasks(root.path(), "## Roots\n- #1 search\n").await;
    assert!(source.on_final_turn().await.is_none());
}

#[tokio::test]
async fn an_event_and_the_stale_row_it_explains_are_named_once() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(root.path(), "## Roots\n- #1 search, PR #12 open\n").await;
    let tracker = FixedTracker::default()
        .with(item(WorkRef::issue(1), WorkState::Open))
        .with(item(WorkRef::pull_request(12), WorkState::Merged));
    let (source, events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker));
    source.on_request(false).await.expect("session start");

    events.record(TaskEvent::PullRequestMerged(WorkRef::pull_request(12)));
    source.on_request(false).await;
    let alert = source.on_final_turn().await.expect("the gate holds");
    assert_eq!(
        alert.content().matches("PR #12").count(),
        2,
        "{}",
        alert.content()
    );
    assert!(!alert.content().contains("which is merged"));
}

#[tokio::test]
async fn a_mark_that_no_longer_waits_is_a_stale_row() {
    let root = tempfile::tempdir().expect("tempdir");
    write_tasks(
        root.path(),
        "## Doing\n- pick a db BLOCKING req-answered\n- write poem BLOCKING waiting on the user\n",
    )
    .await;
    let (source, _events) = document(root.path());
    let mut source = source.with_blocker_resolver(AnyTaskBlockerResolver::new(Answered));
    source.on_request(false).await.expect("session start");
    let alert = source.on_final_turn().await.expect("the gate holds");
    assert!(
        alert.content().contains(
            "The row \"pick a db BLOCKING req-answered\" is marked BLOCKING req-answered, which no longer waits."
        ),
        "{}",
        alert.content()
    );
    assert!(!alert.content().contains("write poem"));
}

#[tokio::test]
async fn past_the_line_limit_one_alert_asks_for_a_rewrite() {
    let root = tempfile::tempdir().expect("tempdir");
    let long = |lines: usize| {
        let mut content = String::from("## Notes\n");
        for _ in 1..lines {
            content.push_str("note\n");
        }
        content
    };
    write_tasks(root.path(), &long(5)).await;
    let events = TaskEvents::new();
    let mut source = TasksDocument::new(root.path(), 8, events);
    source.on_request(false).await.expect("session start");
    assert!(tool_turn(&mut source).await.is_none());

    write_tasks(root.path(), &long(9)).await;
    let alert = tool_turn(&mut source).await.expect("the limit alerts");
    assert!(
        alert
            .content()
            .contains("tasks.md has 9 lines, more than its limit of 8. Rewrite it"),
        "{}",
        alert.content()
    );
    write_tasks(root.path(), &long(10)).await;
    assert!(tool_turn(&mut source).await.is_none(), "once per crossing");

    write_tasks(root.path(), &long(4)).await;
    assert!(tool_turn(&mut source).await.is_none());
    write_tasks(root.path(), &long(12)).await;
    assert!(
        tool_turn(&mut source).await.is_some(),
        "a new crossing alerts"
    );
}

#[tokio::test]
async fn an_actionable_row_keeps_the_turn_and_blocked_rows_do_not() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut blocked = item(WorkRef::issue(3), WorkState::Open);
    blocked.blocked_by = vec![related(WorkRef::issue(4), WorkState::Open)];
    let tracker = FixedTracker::default()
        .with(blocked)
        .with(item(WorkRef::issue(4), WorkState::Open))
        .with(item(WorkRef::issue(5), WorkState::Open));
    let (source, _events) = document(root.path());
    let mut source = source
        .with_work_tracker(AnyWorkTracker::new(tracker))
        .with_blocker_resolver(AnyTaskBlockerResolver::new(Answered));

    write_tasks(
        root.path(),
        "## Doing\n- #3 waits on #4\n## Queued\n- pick a db BLOCKING req-open\n",
    )
    .await;
    source.on_request(false).await.expect("session start");
    assert!(
        source.on_final_turn().await.is_none(),
        "every open row is blocked"
    );

    write_tasks(
        root.path(),
        "## Doing\n- #3 waits on #4\n- #5 ready\n## Queued\n- pick a db BLOCKING req-open\n",
    )
    .await;
    let alert = source.on_final_turn().await.expect("one row is actionable");
    assert_eq!(
        alert.content(),
        "<system_reminder>tasks.md still has rows under Doing or Queued that nothing blocks; continue with them. A row waiting on an open ask_user answer — or on anything else the run cannot resolve — is marked BLOCKING <request_id or reason>, and a row whose issue is blocked by an open issue waits for it: blocked rows wait for what they name instead of holding the turn open. A mark naming a request that is no longer open does not hold.</system_reminder>"
    );
}

#[tokio::test]
async fn roots_bring_their_trees_in_full_once() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut epic = item(WorkRef::issue(1), WorkState::Open);
    epic.sub_issues = vec![related(WorkRef::issue(2), WorkState::Open)];
    epic.blocked_by = vec![related(WorkRef::issue(3), WorkState::Closed)];
    let mut child = item(WorkRef::issue(2), WorkState::Open);
    child.comments = vec![WorkComment {
        author: Some("lexoliu".to_string()),
        body: "decided: sqlite".to_string(),
    }];
    let tracker = FixedTracker::default()
        .with(epic)
        .with(child)
        .with(item(WorkRef::issue(3), WorkState::Closed))
        .with(item(WorkRef::issue(8), WorkState::Open));
    write_tasks(root.path(), "## Roots\n- #1 search\n").await;
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker));

    let block = source.on_request(false).await.expect("session start");
    let text = block.content();
    for needle in [
        "<external-content",
        "Issue #1 (open): title of #1",
        "Sub-issues: #2 (open)",
        "Blocked by: #3 (closed)",
        "body of #1",
        "Issue #2 (open): title of #2",
        "Comment by lexoliu:\ndecided: sqlite",
        "Issue #3 (closed): title of #3",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in {text}");
    }
    assert!(
        source.on_request(false).await.is_none(),
        "trees are sent once"
    );

    write_tasks(root.path(), "## Roots\n- #1 search\n- #8 docs, #9 gone\n").await;
    let block = source.on_request(false).await.expect("a new root");
    let text = block.content();
    assert!(text.contains("Issue #8 (open): title of #8"), "{text}");
    assert!(text.contains("Not loaded: #9 does not exist"), "{text}");
    assert!(
        !text.contains("title of #1"),
        "a known root is not resent: {text}"
    );

    let block = source.on_request(true).await.expect("a reassembly");
    assert!(block.content().contains("title of #1"));
    assert!(block.content().contains("title of #8"));
}
