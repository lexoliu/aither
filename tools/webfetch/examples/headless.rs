//! Headless browser fetch example.
//!
//! Run with: cargo run -p aither-webfetch --features headless --example headless [URL]
//! Add --raw to see raw HTML instead of processed content.

use aither_webfetch::{fetch_html_raw, fetch_with_browser};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let raw_mode = args.iter().any(|a| a == "--raw");
    let url = args.iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .unwrap_or("https://example.com");

    println!("Fetching: {url}\n");

    if raw_mode {
        // Debug mode - show raw HTML
        let html = fetch_html_raw(url).await?;
        println!("--- Raw HTML ({} chars) ---\n", html.len());
        // Search for post content indicators
        if html.contains("好想") {
            println!(">>> Found post content in HTML!");
        }
        if html.contains("Something went wrong") {
            println!(">>> X.com showing error page");
        }
        if html.contains("Don't miss what's happening") {
            println!(">>> X.com showing login prompt");
        }
        println!("\n--- First 3000 chars ---\n");
        println!("{}", &html[..html.len().min(3000)]);
    } else {
        let result = fetch_with_browser(url).await?;
        println!("Title: {:?}\n", result.title);
        println!("--- Content ---\n");
        println!("{}", result.content);
    }

    Ok(())
}
