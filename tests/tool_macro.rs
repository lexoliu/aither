//! The `#[tool]` macro hands a `ToolContext` parameter the call's context and
//! keeps it out of the argument schema the model sees.
#![cfg(feature = "derive")]

use std::sync::mpsc;

use aither::llm::tool::{Progress, ProgressSink, Tools};
use aither::llm::{Tool, ToolContext};
use aither_derive::tool;

/// Counts up to `steps`, reporting each step as progress.
#[tool]
async fn count(steps: u32, mut cx: ToolContext, label: String) -> aither::Result<String> {
    for step in 1..=steps {
        cx.report_progress(Progress::new(f64::from(step)).with_total(f64::from(steps)))
            .await?;
    }
    Ok(format!("{label}: {steps}"))
}

/// Forwards every report to a channel the test drains.
struct Collect(mpsc::Sender<Progress>);

impl ProgressSink for Collect {
    async fn report(&mut self, progress: Progress) {
        self.0.send(progress).expect("test receiver alive");
    }
}

#[test]
fn context_parameter_is_not_an_argument() {
    let schema = aither::llm::tool::ToolDefinition::new(&Count).arguments_openai_schema();
    let properties = schema["properties"].as_object().expect("object schema");
    let mut names: Vec<&str> = properties.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(names, ["label", "steps"]);
}

#[tokio::test]
async fn context_parameter_receives_the_call_context() {
    let (tx, rx) = mpsc::channel();
    let output = Count
        .call(
            CountArgs {
                steps: 2,
                label: "counted".into(),
            },
            ToolContext::with_progress(Collect(tx)),
        )
        .await
        .expect("tool runs");
    assert_eq!(output.as_text(), Some("counted: 2"));
    assert_eq!(
        rx.try_iter().collect::<Vec<_>>(),
        [
            Progress::new(1.0).with_total(2.0),
            Progress::new(2.0).with_total(2.0),
        ]
    );

    // Through the registry the context travels the same way.
    let mut tools = Tools::new();
    tools.register(Count).expect("registers");
    let (tx, rx) = mpsc::channel();
    tools
        .call(
            "count",
            r#"{"steps": 1, "label": "once"}"#,
            ToolContext::with_progress(Collect(tx)),
        )
        .await
        .expect("tool runs");
    assert_eq!(
        rx.try_iter().collect::<Vec<_>>(),
        [Progress::new(1.0).with_total(1.0)]
    );
}
