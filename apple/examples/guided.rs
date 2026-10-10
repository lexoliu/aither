//! Generate a typed value using the native generation schema.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aither_apple::AppleIntelligence;
    use aither_core::llm::{LLMRequest, LanguageModel, Message};
    #[derive(Debug, schemars::JsonSchema, serde::Deserialize)]
    struct Description {
        subject: String,
        colors: Vec<String>,
    }
    let description: Description = AppleIntelligence::new()?
        .generate(LLMRequest::new([Message::user(
            "Describe an autumn maple leaf.",
        )]))
        .await?;
    println!("{}: {}", description.subject, description.colors.join(", "));
    Ok(())
}
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {
    eprintln!("This example requires macOS or iOS 26 and an eligible device.");
}
