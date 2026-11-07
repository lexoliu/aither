//! A turn whose every call is refused for re-reading unchanged output parks
//! until background work reports, instead of sending the model another
//! request; with nothing running, the refusal goes back to the model.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::{Event, LLMRequest, LanguageModel, ToolCall, model::Profile};
use aither_sandbox::builtin::ReadTerminalDeltaTool;
use aither_sandbox::{
    BackgroundTaskSender, CompletedTask, JobRegistry, TerminalError, TerminalMode,
    background_task_channel, job_registry_channel,
};
use futures_core::Stream;

use crate::{Agent, AgentBuilder, AgentOptions, BackgroundJobPolicy};

const TASK: &str = "amber-forest-thunder-pearl";
/// What the background build reports when it finishes.
const FINISHED: &str = "the build finished";
/// The start of the text an unchanged `terminal_read` is refused with.
const REFUSED: &str = "produced no new output since the last read";

#[derive(Debug)]
struct NeverError;

impl std::fmt::Display for NeverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never")
    }
}

impl std::error::Error for NeverError {}

impl aither_core::llm::Classify for NeverError {
    fn failure(&self) -> aither_core::llm::Failure {
        aither_core::llm::Failure::permanent()
    }
}

/// A model that answers every request with the same `terminal_read` of the
/// task, saying it will wait, until a request carries `until`; then it
/// ends its turn.
#[derive(Clone)]
struct PollingLlm {
    until: &'static str,
    requests: Arc<Mutex<Vec<LLMRequest>>>,
}

impl PollingLlm {
    fn new(until: &'static str) -> Self {
        Self {
            until,
            requests: Arc::default(),
        }
    }

    fn requests(&self) -> Vec<LLMRequest> {
        self.requests.lock().unwrap().clone()
    }
}

fn mentions(request: &LLMRequest, text: &str) -> bool {
    request
        .messages()
        .iter()
        .any(|message| message.content().contains(text))
}

impl LanguageModel for PollingLlm {
    type Error = NeverError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let done = mentions(&request, self.until);
        let seen = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len()
        };
        let events = if done {
            vec![Event::Text("The build is done.".to_string())]
        } else {
            vec![
                Event::Text("I'll wait for the background build.".to_string()),
                Event::ToolCall(ToolCall::new(
                    format!("read_{seen}"),
                    "terminal_read",
                    serde_json::json!({ "task_id": TASK }),
                )),
            ]
        };
        futures_lite::stream::iter(events.into_iter().map(Ok))
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "polling",
            "test",
            "polling",
            "polling model",
            128_000,
        )))
    }
}

/// A registry holding the running task with some output, and an agent whose
/// `terminal_read` reads it and whose background receiver knows the task.
async fn polling_agent(
    llm: PollingLlm,
    policy: BackgroundJobPolicy,
) -> (Agent<PollingLlm>, BackgroundTaskSender, JobRegistry) {
    let (registry, service) = job_registry_channel();
    tokio::spawn(service.serve());
    registry
        .register(
            7,
            TASK,
            "exec-build",
            "cargo check",
            TerminalMode::Sandboxed,
            None,
        )
        .await;
    registry.append_stdout(7, b"Compiling\n".to_vec()).await;

    let (sender, receiver) = background_task_channel();
    let mut agent = AgentBuilder::new(llm)
        .agent_kind(crate::AgentKind::Chatbot)
        .background_jobs(policy)
        .tool(ReadTerminalDeltaTool::new(
            registry.clone(),
            aither_sandbox::AnySandboxNotices::default(),
        ))
        .build();
    agent.background_receiver = Some(receiver);
    // `terminal` detached the task earlier, so the model knows it.
    agent.detached_task_ids.insert(TASK.to_string());
    (agent, sender, registry)
}

async fn finish_build(sender: &BackgroundTaskSender) {
    sender
        .send_completed(CompletedTask {
            task_id: TASK.to_string(),
            script: "cargo check".to_string(),
            result: Err(TerminalError::Execution(FINISHED.to_string())),
            kill: None,
            timeout_kill: None,
            pending: None,
        })
        .await;
    sender.task_finished();
}

/// Waits until `llm` has seen `count` requests.
async fn requests_reach(llm: &PollingLlm, count: usize) {
    tokio::time::timeout(Duration::from_secs(20), async {
        while llm.requests().len() < count {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the model must be asked");
}

#[tokio::test]
async fn a_refused_reread_parks_the_turn_until_the_job_completes() {
    let llm = PollingLlm::new(FINISHED);
    let (mut agent, sender, _registry) =
        polling_agent(llm.clone(), BackgroundJobPolicy::Await).await;
    sender.task_started();

    let build = async {
        // The first read returns output; the second, unchanged, is refused.
        requests_reach(&llm, 2).await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            llm.requests().len(),
            2,
            "a refused re-read with the build running must not reach the model again"
        );
        finish_build(&sender).await;
    };
    let (answer, ()) = tokio::time::timeout(
        Duration::from_secs(30),
        futures::future::join(agent.query("Check the build"), build),
    )
    .await
    .expect("the run must end once the build completes");
    assert_eq!(answer.expect("the run succeeds"), "The build is done.");

    let requests = llm.requests();
    assert_eq!(
        requests.len(),
        3,
        "the job's completion resumes the run with one request"
    );
    let resumed = &requests[2];
    assert!(
        mentions(resumed, REFUSED),
        "the refused call keeps its result in the transcript"
    );
    assert!(
        mentions(resumed, FINISHED),
        "the resumed request carries the completion"
    );
}

#[tokio::test]
async fn a_detached_run_ends_on_a_refused_reread_and_the_completion_resumes_it() {
    let llm = PollingLlm::new(FINISHED);
    let (mut agent, sender, _registry) =
        polling_agent(llm.clone(), BackgroundJobPolicy::Detach).await;
    sender.task_started();

    tokio::time::timeout(Duration::from_secs(20), agent.query("Check the build"))
        .await
        .expect("a parked detached run ends")
        .expect("the run succeeds");
    assert_eq!(
        llm.requests().len(),
        2,
        "the run ends at the refused re-read"
    );

    finish_build(&sender).await;
    let wake = tokio::time::timeout(Duration::from_secs(20), agent.wait_background())
        .await
        .expect("the completion wakes the idle agent");
    assert!(wake.needs_model, "the completion is news for the model");
    drop(wake);
    tokio::time::timeout(
        Duration::from_secs(20),
        crate::test_support::finish_run(agent.resume()),
    )
    .await
    .expect("the resumed run ends")
    .expect("the resumed run succeeds");

    let requests = llm.requests();
    assert_eq!(requests.len(), 3);
    assert!(mentions(&requests[2], REFUSED) && mentions(&requests[2], FINISHED));
}

#[tokio::test]
async fn a_refused_reread_with_nothing_running_returns_the_refusal() {
    let llm = PollingLlm::new(REFUSED);
    let (mut agent, _sender, _registry) =
        polling_agent(llm.clone(), BackgroundJobPolicy::Await).await;

    let answer = tokio::time::timeout(Duration::from_secs(20), agent.query("Check the build"))
        .await
        .expect("the run ends")
        .expect("the run succeeds");
    assert_eq!(answer, "The build is done.");

    let requests = llm.requests();
    assert_eq!(
        requests.len(),
        3,
        "with nothing running, the refusal goes straight back to the model"
    );
    assert!(mentions(&requests[2], REFUSED));
}
