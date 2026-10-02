//! Stream text from the on-device model.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aither_apple::AppleIntelligence;
    use aither_core::llm::{LLMRequest, LanguageModel, Message};
    use futures_lite::StreamExt;
    let model = AppleIntelligence::new()?;
    let mut stream = model.respond(LLMRequest::new([Message::user(
        "Explain why leaves change color.",
    )]));
    while let Some(event) = stream.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {
    eprintln!("This example requires macOS or iOS 26 and an eligible device.");
}
