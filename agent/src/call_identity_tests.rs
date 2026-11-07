//! The agent loop gives every tool call its id and the identity of the agent
//! making it, for the main agent and for a subagent alike.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use aither_core::llm::{
    AgentIdentity, CallIdentity, Event, LLMRequest, LanguageModel, Tool, ToolCall, ToolCallId,
    ToolContext, ToolResult, model::Profile,
};
use futures_core::Stream;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::specialized::{SubagentArgs, SubagentTool};
use crate::{AgentBuilder, AgentOptions};

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

/// Answers each request with the next scripted response, shared by every
/// clone, so a main agent and the subagents it starts read one script.
#[derive(Clone)]
struct ScriptedLlm(Arc<Mutex<VecDeque<Vec<Event>>>>);

impl ScriptedLlm {
    fn new(script: impl IntoIterator<Item = Vec<Event>>) -> Self {
        Self(Arc::new(Mutex::new(script.into_iter().collect())))
    }
}

impl LanguageModel for ScriptedLlm {
    type Error = NeverError;

    fn respond(
        &self,
        _request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let events = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .expect("the script has a response for every request");
        futures_lite::stream::iter(events.into_iter().map(Ok))
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "scripted",
            "test",
            "scripted",
            "scripted model",
            128_000,
        )))
    }
}

/// Records which call reached it.
#[derive(Debug, Deserialize, JsonSchema)]
struct ProbeArgs {}

/// Records the identity every call to it carries.
#[derive(Clone, Default)]
struct IdentityRecorder(Arc<Mutex<Vec<Option<CallIdentity>>>>);

impl Tool for IdentityRecorder {
    fn name(&self) -> Cow<'static, str> {
        "probe".into()
    }

    type Arguments = ProbeArgs;
    type Res = ToolResult;

    fn call(
        &self,
        _arguments: Self::Arguments,
        cx: ToolContext,
    ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
        self.0.lock().unwrap().push(cx.identity().cloned());
        std::future::ready(Ok(ToolResult::text("probed")))
    }
}

fn probe_call(id: &str) -> Event {
    Event::ToolCall(ToolCall::new(id, "probe", serde_json::json!({})))
}

#[tokio::test]
async fn the_main_agent_dispatches_each_call_with_its_own_id() {
    let llm = ScriptedLlm::new([
        vec![probe_call("call_a"), probe_call("call_b")],
        vec![Event::Text("done".to_string())],
    ]);
    let recorder = IdentityRecorder::default();
    let mut agent = AgentBuilder::new(llm)
        .agent_kind(crate::AgentKind::Chatbot)
        .tool(recorder.clone())
        .build();

    agent.query("probe twice").await.expect("main run");

    let mut seen = recorder.0.lock().unwrap().clone();
    seen.sort_by_key(|identity| {
        identity
            .as_ref()
            .map(|identity| identity.call().to_string())
    });
    assert_eq!(
        seen,
        [
            Some(CallIdentity::new(
                ToolCallId::new("call_a"),
                AgentIdentity::Main
            )),
            Some(CallIdentity::new(
                ToolCallId::new("call_b"),
                AgentIdentity::Main
            )),
        ]
    );
}

#[tokio::test]
async fn a_subagent_dispatches_its_calls_as_that_subagent_started_by_the_calling_call() {
    let llm = ScriptedLlm::new([
        vec![probe_call("child_call")],
        vec![Event::Text("done".to_string())],
    ]);
    let recorder = IdentityRecorder::default();
    let child_recorder = recorder.clone();
    let dir = tempfile::tempdir().expect("temp dir");
    let worker = crate::SubagentDefinition::parse(
        "name = \"worker\"\ndescription = \"worker for identity tests\"\ntier = \"fast\"\naccess = \"read_only\"\nprompt = '''\nProbe.\n'''\n",
    )
    .expect("parse definition");
    let tool = SubagentTool::new(llm)
        .with_definitions([worker])
        .with_base_dir(dir.path())
        .with_agent_setup("worker", move |builder| {
            builder.tool(child_recorder.clone())
        });
    let prompt_file = dir.path().join("prompt.txt");
    std::fs::write(&prompt_file, "probe once").expect("write prompt file");
    let parent = CallIdentity::new(ToolCallId::new("parent_call"), AgentIdentity::Main);

    Tool::call(
        &tool,
        SubagentArgs {
            name: "worker".to_string(),
            prompt_file: prompt_file.to_string_lossy().into_owned(),
            id: None,
            agent: None,
            worktrees: Vec::new(),
            commit_worktrees: Vec::new(),
        },
        ToolContext::new().with_identity(parent.clone()),
    )
    .await
    .expect("subagent run");

    let seen = recorder.0.lock().unwrap().clone();
    let [Some(identity)] = seen.as_slice() else {
        panic!("the subagent made one identified call, got {seen:?}");
    };
    assert_eq!(identity.call().as_str(), "child_call");
    let AgentIdentity::Subagent(subagent) = identity.agent() else {
        panic!("the call is made by a subagent, got {identity:?}");
    };
    assert_eq!(subagent.name(), "worker");
    assert_eq!(subagent.started_by(), Some(&parent));
}
