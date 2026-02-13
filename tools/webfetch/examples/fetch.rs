//! Simple fetch example demonstrating the webfetch crate.

use std::time::Duration;

use aither_webfetch::{FetchRequest, fetch_with_request};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let url = args
        .iter()
        .skip(1)
        .find(|arg| !arg.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "https://example.com".to_string());
    let timeout_ms = args
        .iter()
        .position(|arg| arg == "--timeout-ms")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|raw| raw.parse::<u64>().ok());
    let explicit_jina_key = args
        .iter()
        .position(|arg| arg == "--jina-key")
        .and_then(|idx| args.get(idx + 1))
        .cloned();

    println!("Fetching: {url}");
    println!("---\n");

    let mut request = FetchRequest::new(url.clone());
    if let Some(timeout_ms) = timeout_ms {
        request = request.with_deadline(Duration::from_millis(timeout_ms));
    }
    if let Some(key) = explicit_jina_key {
        request = request.with_jina_api_key(key);
    }

    let result = fetch_with_request(request).await?;

    println!("Title: {:?}", result.title);
    println!("Extractor: {:?}", result.extractor);
    println!("Content length: {} chars", result.content.len());
    println!("\n--- Content ---\n");
    println!("{}", &result.content[..result.content.len().min(2000)]);

    Ok(())
}
