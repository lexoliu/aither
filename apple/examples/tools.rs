//! Native tool dispatch to borrowed Rust tools.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aither_apple::AppleIntelligence;
    use aither_core::llm::tool::Tools;
    use aither_core::llm::{LLMRequest, LanguageModel, Message, Tool, ToolContext, ToolResult};
    use futures_lite::StreamExt;
    use std::borrow::Cow;
    #[derive(schemars::JsonSchema, serde::Deserialize)]
    struct Arguments {
        left: i32,
        right: i32,
    }
    struct Add;
    impl Tool for Add {
        type Arguments = Arguments;
        type Res = ToolResult;
        fn name(&self) -> Cow<'static, str> {
            "add".into()
        }
        fn description(&self) -> Cow<'static, str> {
            "Add two signed integers exactly.".into()
        }
        fn call(
            &self,
            args: Arguments,
            _: ToolContext,
        ) -> impl Future<Output = aither_core::Result<ToolResult>> + Send {
            std::future::ready(Ok(ToolResult::text(
                (i64::from(args.left) + i64::from(args.right)).to_string(),
            )))
        }
    }
    let model = AppleIntelligence::new()?;
    let mut tools = Tools::new();
    tools.register(Add)?;
    let request = LLMRequest::new([Message::user("Use add to calculate 193 plus 248.")]);
    let mut stream = model.respond_with_tools(request.with_tools(&mut tools));
    while let Some(event) = stream.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {
    eprintln!("This example requires macOS or iOS 26 and an eligible device.");
}
