//! Regression tests for the tool registry's type recovery and description handling.

use aither_core::llm::tool::{RegisterError, Tool, ToolContext, ToolResult, Tools};
use schemars::JsonSchema;
use serde::Deserialize;
use std::borrow::Cow;
use std::future::{Future, ready};

/// Adds two numbers together.
#[derive(JsonSchema, Deserialize)]
struct AddArgs {
    a: u32,
    b: u32,
}

struct Calc;

impl Tool for Calc {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("calc")
    }
    type Arguments = AddArgs;
    type Res = ToolResult;
    fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Self::Res>> + Send {
        ready(Ok(ToolResult::text((args.a + args.b).to_string())))
    }
}

#[derive(JsonSchema, Deserialize)]
struct NoDocArgs {
    #[allow(dead_code)]
    a: u32,
}

struct NoDoc;

impl Tool for NoDoc {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("nodoc")
    }
    type Arguments = NoDocArgs;
    type Res = ToolResult;
    fn call(
        &self,
        _args: Self::Arguments,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Self::Res>> + Send {
        ready(Ok(ToolResult::Done))
    }
}

struct Described;

impl Tool for Described {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("described")
    }
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Explicitly described tool.")
    }
    type Arguments = NoDocArgs;
    type Res = ToolResult;
    fn call(
        &self,
        _args: Self::Arguments,
        _cx: ToolContext,
    ) -> impl Future<Output = aither_core::Result<Self::Res>> + Send {
        ready(Ok(ToolResult::Done))
    }
}

#[test]
fn get_recovers_the_concrete_tool() {
    let mut tools = Tools::new();
    tools.register(Calc).unwrap();
    assert!(
        tools.get::<Calc>().is_some(),
        "get() must find a tool that was registered"
    );
}

#[test]
fn get_mut_recovers_the_concrete_tool() {
    let mut tools = Tools::new();
    tools.register(Calc).unwrap();
    assert!(tools.get_mut::<Calc>().is_some());
}

#[test]
fn get_returns_none_for_unregistered_type() {
    let mut tools = Tools::new();
    tools.register(Calc).unwrap();
    assert!(tools.get::<NoDoc>().is_none());
}

#[test]
fn description_comes_from_arguments_rustdoc() {
    let mut tools = Tools::new();
    tools.register(Calc).unwrap();
    let def = &tools.definitions()[0];
    assert_eq!(def.description(), "Adds two numbers together.");
}

#[test]
fn tool_without_any_description_is_rejected() {
    let mut tools = Tools::new();
    assert_eq!(
        tools.register(NoDoc),
        Err(RegisterError::EmptyDescription(Cow::Borrowed("nodoc"))),
        "a tool with no description must not reach the model"
    );
}

#[test]
fn explicit_description_overrides_rustdoc() {
    let mut tools = Tools::new();
    tools.register(Described).unwrap();
    assert_eq!(
        tools.definitions()[0].description(),
        "Explicitly described tool."
    );
}

#[test]
fn duplicate_names_are_rejected_without_panicking() {
    let mut tools = Tools::new();
    tools.register(Calc).unwrap();
    assert_eq!(
        tools.register(Calc),
        Err(RegisterError::DuplicateName(Cow::Borrowed("calc")))
    );
}

#[tokio::test]
async fn registered_tool_executes() {
    let mut tools = Tools::new();
    tools.register(Calc).unwrap();
    let out = tools
        .call("calc", r#"{"a":2,"b":3}"#, ToolContext::new())
        .await
        .unwrap();
    assert_eq!(
        out,
        ToolResult::Text {
            text: "5".to_string()
        }
    );
}
