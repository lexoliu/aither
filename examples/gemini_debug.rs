//! Debug helper to inspect Gemini structured outputs and tool calling.
//!
//! Run with: `GEMINI_API_KEY=... cargo run --example gemini_debug`

use std::borrow::Cow;

use aither_core::{
    LanguageModel,
    llm::{LLMRequest, Message, Tool, model::Parameters, oneshot, tool::Tools},
};
use aither_gemini::GeminiBackend;
use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const MODEL: &str = "gemini-2.0-flash";

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SimpleStruct {
    title: String,
    value: i32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key =
        std::env::var("GEMINI_API_KEY").context("set GEMINI_API_KEY in your environment")?;
    let gemini = GeminiBackend::new(api_key).with_text_model(MODEL);

    println!("== Structured output test ==");
    structured_output(&gemini).await?;

    println!("\n== Tool call test ==");
    tool_call(&gemini).await?;

    Ok(())
}

async fn structured_output(gemini: &GeminiBackend) -> Result<()> {
    let mut params = Parameters::default();
    params.structured_outputs = true;
    params.response_format = Some(schemars::schema_for!(SimpleStruct));

    let req = oneshot(
        "Return a JSON object matching the provided schema.",
        "Title should be 'debug', value should be 42.",
    )
    .with_parameters(params);

    let raw = gemini.respond(req).into_future().await?;
    println!("Raw structured response:\n{raw}\n");

    match serde_json::from_str::<SimpleStruct>(&raw) {
        Ok(parsed) => println!("Parsed struct: {:?}", parsed),
        Err(err) => println!("Failed to parse into SimpleStruct: {err}"),
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EchoArgs {
    text: String,
}

#[derive(Debug)]
struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> Cow<'static, str> {
        "echo_tool".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Echo back the provided text".into()
    }

    type Arguments = EchoArgs;

    fn call(
        &mut self,
        arguments: Self::Arguments,
    ) -> impl core::future::Future<Output = aither_core::Result<String>> + Send {
        let text = arguments.text;
        async move { Ok(format!("echo: {text}")) }
    }
}

async fn tool_call(gemini: &GeminiBackend) -> Result<()> {
    let mut tools = Tools::new();
    tools.register(EchoTool);

    let mut params = Parameters::default();
    params.tool_choice = Some(vec!["echo_tool".to_string()]);

    let request = LLMRequest::new([
        Message::system("You are a tool tester."),
        Message::user(
            "Call the echo_tool with text 'hello tools' and return only the tool result.",
        ),
    ])
    .with_parameters(params)
    .with_tools(&mut tools);

    let final_text = gemini.respond(request).into_future().await?;
    println!("Tool call response:\n{final_text}");
    Ok(())
}
