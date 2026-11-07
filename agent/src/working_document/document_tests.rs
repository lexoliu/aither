use super::*;
use crate::working_document::blocker::BlockerResolver;
use crate::working_document::test_schema::ListSchema;
use crate::working_document::tracker::test_tracker::{FixedTracker, item, related};
use crate::working_document::{BlockerError, WorkComment, WorkState};

async fn write_plan(root: &std::path::Path, content: &str) {
    async_fs::write(root.join("plan.md"), content)
        .await
        .expect("write plan.md");
}

fn document(root: &std::path::Path) -> (WorkingDocument, WorkEvents) {
    let events = WorkEvents::new();
    let spec = DocumentSpec::new("plan.md", ListSchema);
    (WorkingDocument::new(&spec, root, events.clone()), events)
}

/// One model turn that ran tools: the request, then its end.
async fn tool_turn(source: &mut WorkingDocument) -> Option<Block> {
    source.on_request(false).await;
    source.on_tool_turn_end().await
}

/// What the review as the turn ends finds, and the alert it raises when
/// the run is held.
async fn turn_end(source: &mut WorkingDocument) -> (DocumentReview, Option<String>) {
    let review = source.review().await;
    let alert = source
        .alert(&review, true)
        .map(|block| block.content().to_string());
    (review, alert)
}

fn actionable(review: &DocumentReview) -> Vec<&str> {
    review.actionable_rows().map(DocumentRow::summary).collect()
}

struct Answered;
impl BlockerResolver for Answered {
    fn is_open(
        &self,
        blocker: &str,
    ) -> impl std::future::Future<Output = Result<bool, BlockerError>> + Send {
        // Mirrors a host resolver: only issued-and-still-open blockers hold;
        // unknown payloads are free text and trusted; a blocker its store
        // cannot read fails.
        std::future::ready(match blocker {
            "req-answered" | "req-withdrawn" => Ok(false),
            "req-unreadable" => Err(BlockerError::Unavailable {
                blocker: blocker.to_string(),
                reason: "database is locked".to_string(),
            }),
            _ => Ok(true),
        })
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
    let (review, alert) = turn_end(&mut source).await;
    assert!(alert.is_none());
    assert_eq!(review.document(), &ParsedDocument::new());
}

#[tokio::test]
async fn a_current_document_gets_no_alert_however_many_turns_run() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "root #1 search\n~ all done\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");
    for _ in 0..20 {
        assert!(tool_turn(&mut source).await.is_none());
    }
    let (review, alert) = turn_end(&mut source).await;
    assert!(alert.is_none());
    assert_eq!(actionable(&review), Vec::<&str>::new());
}

#[tokio::test]
async fn the_session_start_carries_the_whole_document_under_its_name() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* first\n").await;
    let (mut source, _events) = document(root.path());

    let block = source.on_request(false).await.expect("shows the document");
    assert_eq!(
        block.content(),
        "<working_document>\nContents of plan.md:\n* first\n\n</working_document>"
    );
    assert!(
        source.on_request(false).await.is_none(),
        "unchanged adds nothing"
    );
}

#[tokio::test]
async fn a_change_is_shown_once_as_its_diff_and_a_reassembly_shows_it_whole() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* first\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");

    write_plan(root.path(), "* second\n").await;
    let block = source.on_request(false).await.expect("shows the diff");
    assert!(block.content().contains("Changes to plan.md"));
    assert!(block.content().contains("+ * second"));
    assert!(!block.content().contains("Contents of plan.md"));
    assert!(source.on_request(false).await.is_none());

    write_plan(root.path(), "* second\n\n").await;
    let block = source.on_request(false).await.expect("a blank-line change");
    assert!(
        block
            .content()
            .contains("plan.md changed in blank lines only."),
        "{}",
        block.content()
    );

    let block = source.on_request(true).await.expect("shows the document");
    assert!(block.content().contains("Contents of plan.md"));
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
    write_plan(root.path(), "* #129 ship it, PR #12 open\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    // The merge happens during a turn's tools; the model sees it at the
    // next request and has that turn to record it.
    events.record(WorkEvent::PullRequestMerged(WorkRef::pull_request(12)));
    let alert = tool_turn(&mut source).await.expect("one alert");
    assert_eq!(
        alert.content(),
        "<document_alert>\nPR #12 was merged, and plan.md still says: \"#129 ship it, PR #12 open\"\n</document_alert>"
    );
    assert!(
        tool_turn(&mut source).await.is_none(),
        "the alert is one-shot"
    );
}

#[tokio::test]
async fn an_event_the_turn_records_raises_nothing() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* #129 ship it, PR #12 open\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::PullRequestMerged(WorkRef::pull_request(12)));
    source.on_request(false).await;
    write_plan(root.path(), "* #129 follow-up review\n").await;
    assert!(source.on_tool_turn_end().await.is_none());
}

#[tokio::test]
async fn each_tracked_event_left_unrecorded_yields_exactly_one_alert_naming_it() {
    let cases = [
        (
            WorkEvent::IssueClosed(WorkRef::issue(5)),
            "Issue #5 was closed",
        ),
        (
            WorkEvent::PullRequestMerged(WorkRef::pull_request(7)),
            "PR #7 was merged",
        ),
        (
            WorkEvent::PullRequestRejected(WorkRef::pull_request(7)),
            "PR #7 was rejected",
        ),
        (
            WorkEvent::PullRequestClosed(WorkRef::pull_request(7)),
            "PR #7 was closed",
        ),
        (
            WorkEvent::QuestionAnswered("req-1".into()),
            "Question req-1 was answered",
        ),
        (
            WorkEvent::QuestionWithdrawn("req-1".into()),
            "Question req-1 was withdrawn",
        ),
        (
            WorkEvent::SubagentFinished("reviewer".into()),
            "Subagent reviewer finished",
        ),
        (
            WorkEvent::JobFinished("bg-3".into()),
            "Background job bg-3 finished",
        ),
    ];
    let row = "#5 land PR #7 after the reviewer subagent, bg-3 builds waits req-1";
    for (event, phrase) in cases {
        let root = tempfile::tempdir().expect("tempdir");
        write_plan(root.path(), &format!("* {row}\n")).await;
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

/// An event that opens work is recorded by a row naming it, and alerts
/// once, naming no row, when none does.
#[tokio::test]
async fn each_opening_event_left_without_a_row_yields_exactly_one_alert() {
    let cases = [
        (
            WorkEvent::IssueCreated(WorkRef::issue(5)),
            "Issue #5 was created; no row of plan.md names it.",
        ),
        (
            WorkEvent::PullRequestOpened(WorkRef::pull_request(7)),
            "PR #7 was opened; no row of plan.md names it.",
        ),
        (
            WorkEvent::QuestionAsked("req-1".into()),
            "Question req-1 was asked; no row of plan.md names it.",
        ),
        (
            WorkEvent::SubagentStarted("reviewer".into()),
            "Subagent reviewer started; no row of plan.md names it.",
        ),
    ];
    let row = "#5 land PR #7 after the reviewer subagent waits req-1";
    for (event, phrase) in cases {
        let root = tempfile::tempdir().expect("tempdir");
        write_plan(root.path(), "* #1 search\n").await;
        let (mut source, events) = document(root.path());
        source.on_request(false).await.expect("session start");
        events.record(event.clone());
        let alert = tool_turn(&mut source).await.expect("alert");
        assert_eq!(
            alert.content().matches(phrase).count(),
            1,
            "{}",
            alert.content()
        );
        assert!(
            tool_turn(&mut source).await.is_none(),
            "{phrase} alerts once"
        );

        let root = tempfile::tempdir().expect("tempdir");
        write_plan(root.path(), &format!("* {row}\n")).await;
        let (mut source, events) = document(root.path());
        source.on_request(false).await.expect("session start");
        events.record(event);
        assert!(
            tool_turn(&mut source).await.is_none(),
            "a row naming it records {phrase}"
        );
    }
}

/// A question asked as the turn ends is a finding until a row carries it,
/// and the carrying row then waits on it instead of being actionable.
#[tokio::test]
async fn an_unrecorded_question_is_found_once_and_its_blocker_then_holds() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* #1 pick a db\n").await;
    let (source, events) = document(root.path());
    let mut source = source.with_blocker_resolver(AnyBlockerResolver::new(Answered));
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::QuestionAsked("req-open".into()));
    source.on_request(false).await;
    let (review, alert) = turn_end(&mut source).await;
    let alert = alert.expect("the review finds it");
    assert!(alert.contains("Question req-open was asked"), "{alert}");
    assert_eq!(actionable(&review), vec!["#1 pick a db"]);

    write_plan(root.path(), "* #1 pick a db waits req-open\n").await;
    source.on_request(false).await;
    let (review, alert) = turn_end(&mut source).await;
    assert!(alert.is_none());
    assert!(
        actionable(&review).is_empty(),
        "the blocked row waits for the answer instead of being pushed on"
    );
}

/// A subagent that finished before the model saw it start, or a question
/// withdrawn in the same batch, leaves nothing to track.
#[tokio::test]
async fn an_opening_event_ended_in_the_same_batch_raises_nothing() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* #1 search\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");
    events.record(WorkEvent::SubagentStarted("explore-1".into()));
    events.record(WorkEvent::SubagentFinished("explore-1".into()));
    events.record(WorkEvent::QuestionAsked("req-2".into()));
    events.record(WorkEvent::QuestionWithdrawn("req-2".into()));
    assert!(tool_turn(&mut source).await.is_none());
}

/// An event whose own tool turn already wrote it down needs no alert: the
/// edit made alongside it counts.
#[tokio::test]
async fn an_event_recorded_in_its_own_tool_turn_raises_nothing() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* #5 fix login\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::IssueClosed(WorkRef::issue(5)));
    write_plan(root.path(), "* #5 closed, follow-up review\n").await;
    assert!(tool_turn(&mut source).await.is_none());
    assert!(tool_turn(&mut source).await.is_none());
}

/// An issue created under a root — a sub-issue, or a blocker of an issue in
/// the tree — needs no row of its own; one outside every tree does.
#[tokio::test]
async fn an_issue_created_inside_a_root_tree_needs_no_row() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut epic = item(WorkRef::issue(1), WorkState::Open);
    epic.sub_issues = vec![related(WorkRef::issue(2), WorkState::Open)];
    let mut child = item(WorkRef::issue(2), WorkState::Open);
    child.blocked_by = vec![related(
        WorkRef::issue(3).in_repository("Owner/Repo"),
        WorkState::Open,
    )];
    let tracker = FixedTracker::default().with(epic).with(child).with(item(
        WorkRef::issue(3).in_repository("Owner/Repo"),
        WorkState::Open,
    ));
    write_plan(root.path(), "root #1 search\n").await;
    let (source, events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker));
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::IssueCreated(WorkRef::issue(2)));
    events.record(WorkEvent::IssueCreated(WorkRef::issue(3)));
    assert!(tool_turn(&mut source).await.is_none());

    events.record(WorkEvent::IssueCreated(WorkRef::issue(9)));
    let alert = tool_turn(&mut source).await.expect("outside every tree");
    assert!(
        alert.content().contains("Issue #9 was created"),
        "{}",
        alert.content()
    );
}

/// An answered question whose row still waits on it is named once, as the
/// event, not again as a dead blocker.
#[tokio::test]
async fn an_answer_and_the_dead_blocker_it_explains_are_named_once() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* pick a db waits req-answered\n").await;
    let (source, events) = document(root.path());
    let mut source = source.with_blocker_resolver(AnyBlockerResolver::new(Answered));
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::QuestionAnswered("req-answered".into()));
    source.on_request(false).await;
    let (_, alert) = turn_end(&mut source).await;
    let alert = alert.expect("the review finds it");
    assert!(alert.contains("Question req-answered was answered"));
    assert!(!alert.contains("no longer waits"), "{alert}");
}

/// A blocker the host cannot resolve keeps holding its row: it is neither
/// stale nor taken for answered.
#[tokio::test]
async fn an_unresolvable_blocker_keeps_holding_its_row() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* pick a db waits req-unreadable\n").await;
    let (source, _events) = document(root.path());
    let mut source = source.with_blocker_resolver(AnyBlockerResolver::new(Answered));
    source.on_request(false).await.expect("session start");
    let (review, alert) = turn_end(&mut source).await;
    assert!(alert.is_none());
    assert_eq!(actionable(&review), Vec::<&str>::new());
    assert!(review.blocker_holds("req-unreadable"));
}

/// An issue that cannot be read is not taken for unblocked, nor is its row
/// called stale.
#[tokio::test]
async fn an_unreadable_issue_is_not_taken_for_unblocked() {
    let root = tempfile::tempdir().expect("tempdir");
    let tracker = FixedTracker::default().with(item(WorkRef::issue(3), WorkState::Open));
    tracker.fail(&WorkRef::issue(3));
    write_plan(root.path(), "* #3 index\n").await;
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker.clone()));
    source.on_request(false).await.expect("session start");
    let (review, alert) = turn_end(&mut source).await;
    assert!(alert.is_none());
    assert_eq!(actionable(&review), Vec::<&str>::new());

    tracker.heal(&WorkRef::issue(3));
    let (review, _) = turn_end(&mut source).await;
    assert_eq!(
        actionable(&review),
        vec!["#3 index"],
        "readable again, the row is actionable"
    );
}

/// A root whose tree the backend failed to read is said once and fetched
/// again at the next request, which brings its tree once it reads.
#[tokio::test]
async fn a_root_that_failed_to_read_is_fetched_again() {
    let root = tempfile::tempdir().expect("tempdir");
    let tracker = FixedTracker::default().with(item(WorkRef::issue(1), WorkState::Open));
    tracker.fail(&WorkRef::issue(1));
    write_plan(root.path(), "root #1 search\n").await;
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker.clone()));

    let block = source.on_request(false).await.expect("session start");
    assert!(
        block.content().contains("Not loaded: #1 cannot be read"),
        "{}",
        block.content()
    );
    assert!(
        source.on_request(false).await.is_none(),
        "a failure still failing is not said again"
    );

    tracker.heal(&WorkRef::issue(1));
    let block = source.on_request(false).await.expect("the retry reads");
    assert!(
        block.content().contains("Issue #1 (open): title of #1"),
        "{}",
        block.content()
    );
    assert!(source.on_request(false).await.is_none(), "then it is shown");
}

/// An agent rebuilt on a restored conversation takes what that
/// conversation holds of the document: it shows only what changed since,
/// and no root tree again.
#[tokio::test]
async fn a_restored_checkpoint_shows_only_what_changed_since() {
    let root = tempfile::tempdir().expect("tempdir");
    let tracker = FixedTracker::default().with(item(WorkRef::issue(1), WorkState::Open));
    write_plan(root.path(), "root #1 search\n").await;
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker.clone()));
    source.on_request(false).await.expect("session start");
    let checkpoint = source.checkpoint();
    assert!(checkpoint.is_some());

    let (rebuilt, _events) = document(root.path());
    let mut rebuilt = rebuilt.with_work_tracker(AnyWorkTracker::new(tracker.clone()));
    rebuilt.restore(checkpoint.clone());
    assert!(
        rebuilt.on_request(false).await.is_none(),
        "the restored conversation already holds it all"
    );

    let (rebuilt, _events) = document(root.path());
    let mut rebuilt = rebuilt.with_work_tracker(AnyWorkTracker::new(tracker));
    rebuilt.restore(checkpoint);
    write_plan(root.path(), "root #1 search\n~ new\n").await;
    let block = rebuilt.on_request(false).await.expect("the change");
    assert!(block.content().contains("Changes to plan.md"));
    assert!(!block.content().contains("Contents of plan.md"));
    assert!(!block.content().contains("title of #1"));
}

/// The checkpoint keeps the shape stored checkpoints were written in.
#[test]
fn a_checkpoint_round_trips() {
    let checkpoint = DocumentCheckpoint {
        content: Some("* first\n".to_string()),
        roots: vec![WorkRef::issue(1)],
    };
    let json = serde_json::to_value(&checkpoint).expect("serializes");
    assert_eq!(json["content"], "* first\n");
    let read: DocumentCheckpoint = serde_json::from_value(json).expect("reads back");
    assert_eq!(read, checkpoint);
}

#[tokio::test]
async fn an_event_no_row_names_alerts_only_when_it_opens_work() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* #1 search\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::JobFinished("bg-9".into()));
    events.record(WorkEvent::SubagentFinished("explore".into()));
    events.record(WorkEvent::PullRequestMerged(WorkRef::pull_request(4)));
    assert!(tool_turn(&mut source).await.is_none());

    events.record(WorkEvent::IssueCreated(WorkRef::issue(5)));
    let alert = tool_turn(&mut source).await.expect("new work alerts");
    assert!(
        alert
            .content()
            .contains("Issue #5 was created; no row of plan.md names it."),
        "{}",
        alert.content()
    );
}

#[tokio::test]
async fn an_event_with_no_document_alerts_only_when_it_opens_work() {
    let root = tempfile::tempdir().expect("tempdir");
    let (mut source, events) = document(root.path());
    source.on_request(false).await;
    events.record(WorkEvent::JobFinished("bg-1".into()));
    assert!(tool_turn(&mut source).await.is_none());
    events.record(WorkEvent::PullRequestOpened(WorkRef::pull_request(2)));
    assert!(tool_turn(&mut source).await.is_some());
}

#[tokio::test]
async fn an_unrecorded_event_at_the_end_of_the_turn_is_found_once() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "root #1 search, PR #12 open\n").await;
    let (mut source, events) = document(root.path());
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::PullRequestMerged(WorkRef::pull_request(12)));
    source.on_request(false).await;
    let (review, alert) = turn_end(&mut source).await;
    assert_eq!(review.findings().len(), 1);
    let alert = alert.expect("the review finds it");
    assert!(alert.contains("PR #12 was merged"));
    assert!(alert.contains("Bring plan.md up to date before ending the turn."));
    assert!(
        !source
            .alert(&review, false)
            .expect("the same findings")
            .content()
            .contains("before ending the turn"),
        "a run that ends is not told to keep going"
    );
    let (review, _) = turn_end(&mut source).await;
    assert!(review.findings().is_empty(), "the event is found once");
}

#[tokio::test]
async fn the_schemas_problems_alert_once_per_edit_and_at_every_turn_end() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "* fine\n").await;
    let (mut source, _events) = document(root.path());
    source.on_request(false).await.expect("session start");

    write_plan(root.path(), "* fine\n[ ] old style\n").await;
    let alert = tool_turn(&mut source).await.expect("the edit alerts");
    assert_eq!(
        alert.content(),
        "<document_alert>\nLine 2 of plan.md is a checkbox.\n</document_alert>"
    );
    assert!(
        tool_turn(&mut source).await.is_none(),
        "unchanged, no repeat"
    );
    for _ in 0..2 {
        let (review, _) = turn_end(&mut source).await;
        assert_eq!(
            review.findings(),
            &[DocumentAlertItem::Problem(
                "Line 2 of plan.md is a checkbox.".to_string()
            )]
        );
    }
    write_plan(root.path(), "* fine\n").await;
    assert_eq!(turn_end(&mut source).await.0.findings(), &[]);
}

#[tokio::test]
async fn a_row_naming_a_finished_item_is_stale() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "root #1 search, PR #12 open\n").await;
    let tracker = FixedTracker::default()
        .with(item(WorkRef::issue(1), WorkState::Open))
        .with(item(WorkRef::pull_request(12), WorkState::Open));
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker.clone()));
    source.on_request(false).await.expect("session start");
    assert!(turn_end(&mut source).await.1.is_none());

    tracker.set_state(&WorkRef::pull_request(12), WorkState::Merged);
    let (review, alert) = turn_end(&mut source).await;
    let alert = alert.expect("stale");
    assert!(
        alert.contains("The row \"#1 search, PR #12 open\" names PR #12, which is merged."),
        "{alert}"
    );
    assert!(matches!(
        review.status(&WorkRef::pull_request(12)),
        Some(Ok(status)) if status.state == WorkState::Merged
    ));

    write_plan(root.path(), "root #1 search\n").await;
    assert!(turn_end(&mut source).await.1.is_none());
}

#[tokio::test]
async fn an_event_and_the_stale_row_it_explains_are_named_once() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(root.path(), "root #1 search, PR #12 open\n").await;
    let tracker = FixedTracker::default()
        .with(item(WorkRef::issue(1), WorkState::Open))
        .with(item(WorkRef::pull_request(12), WorkState::Merged));
    let (source, events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker));
    source.on_request(false).await.expect("session start");

    events.record(WorkEvent::PullRequestMerged(WorkRef::pull_request(12)));
    source.on_request(false).await;
    let alert = turn_end(&mut source).await.1.expect("found");
    assert_eq!(alert.matches("PR #12").count(), 2, "{alert}");
    assert!(!alert.contains("which is merged"));
}

#[tokio::test]
async fn a_blocker_that_no_longer_waits_is_a_stale_row() {
    let root = tempfile::tempdir().expect("tempdir");
    write_plan(
        root.path(),
        "* pick a db waits req-answered\n* write poem waits on the user\n",
    )
    .await;
    let (source, _events) = document(root.path());
    let mut source = source.with_blocker_resolver(AnyBlockerResolver::new(Answered));
    source.on_request(false).await.expect("session start");
    let (review, alert) = turn_end(&mut source).await;
    let alert = alert.expect("stale");
    assert!(
        alert.contains(
            "The row \"pick a db waits req-answered\" waits on req-answered, which no longer waits."
        ),
        "{alert}"
    );
    assert!(!alert.contains("write poem"));
    assert_eq!(
        actionable(&review),
        vec!["pick a db waits req-answered"],
        "a dead blocker holds nothing"
    );
}

#[tokio::test]
async fn open_rows_that_nothing_blocks_are_actionable_and_others_are_not() {
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
        .with_blocker_resolver(AnyBlockerResolver::new(Answered));

    write_plan(
        root.path(),
        "* #3 after #4\n* pick a db waits req-open\n~ #5 a note, no work\n",
    )
    .await;
    source.on_request(false).await.expect("session start");
    assert!(
        actionable(&turn_end(&mut source).await.0).is_empty(),
        "every open row is blocked"
    );

    write_plan(
        root.path(),
        "* #3 after #4\n* #5 ready\n* pick a db waits req-open\n",
    )
    .await;
    assert_eq!(actionable(&turn_end(&mut source).await.0), vec!["#5 ready"]);
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
    write_plan(root.path(), "root #1 search\n").await;
    let (source, _events) = document(root.path());
    let mut source = source.with_work_tracker(AnyWorkTracker::new(tracker));

    let block = source.on_request(false).await.expect("session start");
    let text = block.content();
    for needle in [
        "The issues under the roots of plan.md",
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

    write_plan(root.path(), "root #1 search\nroot #8 docs, #9 gone\n").await;
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
