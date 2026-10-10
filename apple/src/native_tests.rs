//! OS 27 framework-session tests. No system-model inference is involved.
use super::*;
use aither_core::llm::tool::{ToolContext, ToolDefinition};
use aither_core::llm::{Tool, ToolCall};
use futures_lite::StreamExt;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::time::Duration;

fn request() -> Value {
    json!({"instructions": [], "history": [], "prompt": {"text": "test"},
        "mode": "text", "options": {}, "tools_are_external": true})
}

fn planned(request: &Value, turns: &Value) -> PlannedRequest {
    PlannedRequest {
        json: serde_json::to_vec(request).unwrap(),
        images: Vec::new(),
        script: Some(serde_json::to_vec(&json!({"turns": turns})).unwrap()),
    }
}

async fn events(driver: Driver<'_>) -> Vec<Result<Event, AppleError>> {
    tokio::time::timeout(Duration::from_secs(10), driver.collect())
        .await
        .expect("native stream failed to terminate within 10 seconds")
}

fn text(events: &[Result<Event, AppleError>]) -> String {
    events
        .iter()
        .map(
            |event| match event.as_ref().expect("successful native event") {
                Event::Text(text) => text.as_str(),
                _ => "",
            },
        )
        .collect()
}

#[tokio::test]
#[ignore = "requires macOS/iOS 27 and test; deterministic executor, not inference"]
async fn native_text_usage_errors_and_isolation() {
    let first = Driver::external(async {
        Ok(planned(
            &request(),
            &json!([{"steps":[
                {"text":"日本"}, {"text":"語🦀"}, {"usage":{"input":3,"output":4}}
            ]}]),
        ))
    });
    let second = Driver::external(async {
        Ok(planned(
            &request(),
            &json!([{"steps":[{"text":"independent"}]}]),
        ))
    });
    let (a, b) = tokio::join!(events(first), events(second));
    assert_eq!(text(&a), "日本語🦀");
    assert_eq!(text(&b), "independent");
    assert!(a.iter().any(|e| matches!(e, Ok(Event::Usage(u)) if u.prompt_tokens == Some(3) && u.completion_tokens == Some(4))));
    let failed = events(Driver::external(async {
        Ok(planned(
            &request(),
            &json!([{"steps":[{"fail":"guardrail_violation"}]}]),
        ))
    }))
    .await;
    assert!(
        failed
            .iter()
            .any(|e| matches!(e, Err(AppleError::GuardrailViolation(_)))),
        "{failed:?}"
    );
}

#[derive(schemars::JsonSchema, serde::Deserialize)]
struct EchoArgs {
    value: String,
}
struct Echo;
impl Tool for Echo {
    type Arguments = EchoArgs;
    type Res = ToolResult;
    fn name(&self) -> Cow<'static, str> {
        "echo".into()
    }
    fn description(&self) -> Cow<'static, str> {
        "Returns the supplied value.".into()
    }
    fn call(
        &self,
        args: EchoArgs,
        _: ToolContext,
    ) -> impl Future<Output = aither_core::Result<ToolResult>> + Send {
        core::future::ready(Ok(if args.value == "fail" {
            ToolResult::Error {
                message: "deliberate failure".into(),
            }
        } else {
            ToolResult::text(args.value)
        }))
    }
}

fn tool_request(external: bool) -> Value {
    let definition = ToolDefinition::new(&Echo);
    let schema =
        crate::schema::value_to_wire_schema(&definition.arguments_openai_schema(), "echo").unwrap();
    let mut wire = request();
    wire["tools"] =
        json!([{"name":"echo","description":"Returns supplied value.","schema":schema}]);
    wire["tools_are_external"] = json!(external);
    wire
}

fn calls() -> Value {
    json!([{"id":"native-a","name":"echo","arguments":"{\"value\":\"same\"}"},
        {"id":"native-b","name":"echo","arguments":"{\"value\":\"same\"}"}])
}

#[tokio::test]
#[ignore = "requires macOS/iOS 27 and test; deterministic executor, not inference"]
async fn native_external_batch_and_transcript_replay() {
    let output = events(Driver::external(async {
        Ok(planned(
            &tool_request(true),
            &json!([{"steps":[{"toolCalls":calls()}]}]),
        ))
    }))
    .await;
    let captured: Vec<&ToolCall> = output
        .iter()
        .map(|e| e.as_ref().expect("external tool boundary"))
        .filter_map(|e| {
            if let Event::ToolCall(call) = e {
                Some(call)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(captured.len(), 2, "{output:?}");
    assert_eq!(
        captured.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        ["native-a", "native-b"]
    );
    let mut wire = tool_request(true);
    wire["history"] = json!([
        {"kind":"prompt","text":"original"},
        {"kind":"tool_calls","calls":calls()},
        {"kind":"tool_output","id":"native-a","name":"echo","text":"first"},
        {"kind":"tool_output","id":"native-b","name":"echo","text":"second"}
    ]);
    wire["prompt"]["text"] = json!("");
    let output = events(Driver::external(async move {
        Ok(planned(
            &wire,
            &json!([{"steps":[]},{"steps":[{"probe":true}]}]),
        ))
    }))
    .await;
    let probe: Value = serde_json::from_str(&text(&output)).unwrap();
    assert_eq!(probe["toolDefinitions"], json!(["echo"]));
    let entries = probe["entries"].as_array().unwrap();
    assert!(
        entries
            .iter()
            .any(|e| e.as_str().unwrap().starts_with("toolCalls:native-a:echo:"))
    );
    assert!(
        entries
            .iter()
            .any(|e| e == "toolOutput:native-a:echo:first")
    );
    assert!(
        entries
            .iter()
            .any(|e| e == "toolOutput:native-b:echo:second")
    );
}

#[tokio::test]
#[ignore = "requires macOS/iOS 27 and test; deterministic executor, not inference"]
async fn native_internal_identical_calls_and_tool_failure() {
    let mut tools = Tools::new();
    tools.register(Echo).unwrap();
    let output = events(Driver::internal(
        async {
            Ok(planned(
                &tool_request(false),
                &json!([
                    {"steps":[{"toolCalls":calls()}]}, {"steps":[{"text":"finished"}]}
                ]),
            ))
        },
        &tools,
    ))
    .await;
    assert_eq!(text(&output), "finished");
    assert_eq!(
        output
            .iter()
            .filter(
                |e| matches!(e, Ok(Event::BuiltInToolResult { result, .. }) if result == "same")
            )
            .count(),
        2
    );
    assert!(!output.iter().any(|e| matches!(e, Ok(Event::ToolCall(_)))));
    let failure = events(Driver::internal(
        async {
            Ok(planned(
                &tool_request(false),
                &json!([{"steps":[{"toolCalls":[
                    {"id":"failure","name":"echo","arguments":"{\"value\":\"fail\"}"}
                ]}]}]),
            ))
        },
        &tools,
    ))
    .await;
    assert!(
        failure
            .iter()
            .any(|e| matches!(e, Err(AppleError::Tool { tool, .. }) if tool == "echo")),
        "{failure:?}"
    );
}

#[tokio::test]
#[ignore = "requires macOS/iOS 27 and test; deterministic executor, not inference"]
async fn native_guided_refs_inline_objects_and_unions() {
    for schema in [
        json!({"type":"integer","minimum":1.5,"maximum":5.9}),
        json!({"type":"integer","minimum":i64::MIN,"maximum":i64::MAX}),
        json!({"$defs":{"A":{"type":"integer"},"B":{"type":"array","items":{"type":"string"}}},
            "type":"object","properties":{"n":{"$ref":"#/$defs/A"},"list":{"$ref":"#/$defs/B"}},"required":["n","list"]}),
        json!({"type":"object","properties":{
            "left":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]},
            "right":{"type":"object","properties":{"y":{"type":"integer"}},"required":["y"]}
        },"required":["left","right"]}),
        json!({"anyOf":[
            {"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},
            {"type":"object","properties":{"b":{"type":"integer"}},"required":["b"]}
        ]}),
        json!({"type":"object","properties":{"nullable":{"type":["string","null"]}},"required":["nullable"]}),
    ] {
        let schema = crate::schema::value_to_wire_schema(&schema, "Root").unwrap();
        let bytes = serde_json::to_vec(&schema).unwrap();
        // SAFETY: native code only borrows the byte slice for this call.
        assert_eq!(
            unsafe { validate_schema(bytes.as_ptr(), bytes.len()) },
            0,
            "{schema:?}"
        );
    }
    let mut wire = request();
    wire["mode"] = json!("guided");
    wire["schema"] = serde_json::to_value(crate::schema::value_to_wire_schema(
        &json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}), "Answer").unwrap()).unwrap();
    let output = events(Driver::external(async move {
        Ok(planned(
            &wire,
            &json!([{"steps":[{"text":"{\"answer\":"},{"text":"\"yes\"}"}]}]),
        ))
    }))
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&text(&output)).unwrap(),
        json!({"answer":"yes"})
    );
}

unsafe extern "C" {
    #[link_name = "aither_apple_test_schema"]
    fn validate_schema(data: *const u8, len: usize) -> i32;
}

#[tokio::test]
#[ignore = "requires macOS/iOS 27 and test; deterministic executor, not inference"]
async fn native_cancel_after_start_acknowledges_release() {
    let (released, release_rx) = async_channel::bounded(1);
    let active = start_observed(
        &planned(
            &request(),
            &json!([{"steps":[{"text":"started"},{"hang":true}]}]),
        ),
        Some(released),
    )
    .unwrap();
    let mut driver = Driver {
        starting: None,
        active: Some(active),
        prelude: VecDeque::new(),
        tools: None,
        pending_tools: Vec::new(),
        done: false,
    };
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if matches!(driver.next().await.unwrap().unwrap(), Event::Text(ref text) if text == "started") { break; }
        }
    }).await.unwrap();
    drop(driver);
    tokio::time::timeout(Duration::from_secs(10), release_rx.recv())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
#[ignore = "requires macOS/iOS 27 and test; deterministic executor, not inference"]
async fn native_actor_latches_and_backpressure() {
    let output = events(Driver::external(async {
        Ok(planned(
            &request(),
            &json!([{"steps":[{"bridgeChecks":true},{"text":"passed"}]}]),
        ))
    }))
    .await;
    assert_eq!(text(&output), "passed");
    let steps: Vec<Value> = (0..256).map(|_| json!({"text":"x"})).collect();
    let output = events(Driver::external(async move {
        Ok(planned(&request(), &json!([{"steps":steps}])))
    }))
    .await;
    assert_eq!(text(&output), "x".repeat(256));
}
