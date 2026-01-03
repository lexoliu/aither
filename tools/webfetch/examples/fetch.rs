//! Simple fetch example demonstrating the webfetch crate.

use aither_webfetch::fetch;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "https://example.com".to_string());

    println!("Fetching: {url}");
    println!("---\n");

    let result = fetch(&url).await?;

    println!("Title: {:?}", result.title);
    println!("Content length: {} chars", result.content.len());
    println!("\n--- Content ---\n");
    println!("{}", &result.content[..result.content.len().min(2000)]);

    Ok(())
}
