//! Integration tests for the Gemini backend.

use aither_core::{
    AudioGenerator, AudioTranscriber, EmbeddingModel, ImageGenerator, LanguageModel,
    image::{Prompt, Size},
    llm::{LanguageModelProvider, Message, collect_text},
};
use aither_gemini::{Gemini, GeminiProvider};
use futures_lite::{StreamExt, pin};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

fn install_rustls_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

fn api_key() -> String {
    install_rustls_provider();
    env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set")
}

#[tokio::test]
async fn test_provider_list_models() {
    let provider = GeminiProvider::new(api_key());
    let models = provider.list_models().await.expect("Failed to list models");
    assert!(!models.is_empty());

    println!("Available models:");
    for model in &models {
        println!("- {}", model.name);
    }
    // Check for a known model
    let has_flash = models.iter().any(|m| m.name.contains("flash"));
    assert!(has_flash, "Should contain flash model");
}

#[tokio::test]
#[ignore = "Requires external Gemini API quota and network access."]
async fn test_chat_respond() {
    let backend = Gemini::new(api_key());
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is 2+2? Answer with just the number."),
    ];
    let request = aither_core::llm::LLMRequest::new(messages);
    let stream = backend.respond(request);
    let text = collect_text(stream)
        .await
        .expect("Failed to collect response");
    assert!(text.contains('4'));
}

#[derive(JsonSchema, Deserialize, Serialize, Debug, PartialEq)]
struct MathResult {
    answer: i32,
    explanation: String,
}

#[tokio::test]
#[ignore = "Requires external Gemini API quota and network access."]
async fn test_structured_generate() {
    let backend = Gemini::new(api_key());
    let request = aither_core::llm::oneshot("You are a math tutor.", "What is 5 * 5?");

    let result: MathResult = LanguageModel::generate(&backend, request)
        .await
        .expect("Failed to generate structured output");

    assert_eq!(result.answer, 25);
    assert!(!result.explanation.is_empty());
}

#[tokio::test]
#[ignore = "Requires external Gemini API quota and network access."]
async fn test_embedding() {
    let backend = Gemini::new(api_key());
    let vec = backend.embed("Hello world").await.expect("Failed to embed");
    assert_eq!(vec.len(), backend.dim());
    assert!(!vec.is_empty());
}

#[tokio::test]
#[ignore = "Audio generation not supported by available Gemini models/API access."]
async fn test_audio_cycle() {
    // This test generates audio then transcribes it back
    let backend = Gemini::new(api_key());

    // 1. Generate Audio
    let text = "Hello, this is a test of the emergency broadcast system.";
    let stream = AudioGenerator::generate(&backend, text);
    pin!(stream);

    let mut audio_data = Vec::new();
    while let Some(chunk) = stream.next().await {
        audio_data.extend_from_slice(&chunk);
    }
    assert!(!audio_data.is_empty(), "Should generate audio data");

    // 2. Transcribe Audio
    let stream = backend.transcribe(&audio_data);
    pin!(stream);

    let mut transcribed_text = String::new();
    while let Some(chunk) = stream.next().await {
        transcribed_text.push_str(&chunk);
    }

    println!("Original: {text}");
    println!("Transcribed: {transcribed_text}");

    // Fuzzy match or just check length/content
    assert!(!transcribed_text.is_empty(), "Should transcribe text");
}

#[tokio::test]
#[ignore = "Image generation not supported by available Gemini models/API access."]
async fn test_image_generate() {
    // Verify image generation (might be slow or costly, maybe skip if not needed?)
    // Gemini flash image is usually cheap/free in some tiers.
    let backend = Gemini::new(api_key());

    let prompt = Prompt::new("A simple red circle on a white background");
    let size = Size::square(512);

    let stream = backend.create(prompt, size);
    pin!(stream);

    let mut image_data = Vec::new();
    while let Some(result) = stream.next().await {
        image_data = result.expect("Failed to generate image");
    }

    assert!(!image_data.is_empty());
    // Basic check for PNG/JPEG header?
    // Gemini usually returns JPEG or PNG.
}

#[tokio::test]
#[ignore = "Requires external Gemini API quota and network access."]
async fn test_moderation() {
    use aither_core::moderation::Moderation;
    let backend = Gemini::new(api_key());
    let result = backend
        .moderate("I love puppies")
        .await
        .expect("Moderation failed");
    assert!(!result.is_flagged());
}

#[tokio::test]
#[ignore = "Requires external Gemini API quota and network access."]
async fn test_google_search() {
    use aither_core::llm::model::Parameters;
    let backend = Gemini::new(api_key());
    let messages = vec![aither_core::llm::Message::user(
        "Who won the men's Wimbledon singles title in 2023?",
    )];
    let request = aither_core::llm::LLMRequest::new(messages)
        .with_parameters(Parameters::default().websearch(true));

    let stream = backend.respond(request);
    let text = collect_text(stream)
        .await
        .expect("Failed to collect response with Google Search");

    // As of today (Nov 2025), Carlos Alcaraz won Wimbledon 2023
    assert!(
        text.contains("Carlos Alcaraz") || text.contains("Alcaraz"),
        "Response should mention Carlos Alcaraz"
    );
    println!("Google Search response: {text}");
}

#[tokio::test]
#[ignore = "Requires external Gemini API quota and network access."]
async fn test_code_execution() {
    use aither_core::llm::model::Parameters;
    let backend = Gemini::new(api_key());
    let messages = vec![aither_core::llm::Message::user(
        "What is the sum of the first 50 prime numbers? Generate and run code for the calculation.",
    )];
    let request = aither_core::llm::LLMRequest::new(messages)
        .with_parameters(Parameters::default().code_execution(true));

    let stream = backend.respond(request);
    let text = collect_text(stream)
        .await
        .expect("Failed to collect response with Code Execution");

    // The sum of the first 50 primes is 5117. The model should output this result after running code.
    assert!(
        text.contains("5117"),
        "Response should contain the correct sum (5117)"
    );
    assert!(
        text.contains("```python") || text.contains("```output"),
        "Response should contain code execution blocks"
    );
    println!("Code Execution response: {text}");
}
