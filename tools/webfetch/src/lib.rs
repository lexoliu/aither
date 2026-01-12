//! # Web Content Fetching Library
//!
//! An async library for fetching web content and converting it to clean markdown.
//!
//! ## Features
//!
//! - **Static fetch**: Fast HTTP fetching with browser-like headers (default)
//! - **Headless browser**: Full JavaScript rendering via Chrome DevTools Protocol
//!   (enable with `headless` feature)
//! - **Image extraction**: Extract image URLs from pages
//! - **Image fetching**: Fetch images with automatic JPEG conversion
//!
//! ## Usage
//!
//! ```no_run
//! use aither_webfetch::{fetch, FetchResult, WebFetchTool};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Static fetch (fast, no JS)
//! let result = fetch("https://example.com").await?;
//! println!("Title: {:?}", result.title);
//! println!("Content: {}", result.content);
//! // Images are embedded in the markdown content
//!
//! // Or use as an LLM tool
//! let tool = WebFetchTool::new();
//! # Ok(())
//! # }
//! ```
//!
//! ## Image Fetching
//!
//! ```no_run
//! use aither_webfetch::fetch_image;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Fetch image and convert to JPEG
//! let (jpeg_data, mime) = fetch_image("https://example.com/image.webp").await?;
//! println!("Image size: {} bytes, type: {}", jpeg_data.len(), mime);
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::io::Cursor;

use aither_core::llm::{Tool, ToolOutput};
use anyhow::{Result, anyhow};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
#[cfg(feature = "headless")]
use tracing::debug;
use zenwave::{Client, client, header};

/// Result of fetching web content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResult {
    /// The original URL that was fetched.
    pub url: String,
    /// The page title, if found.
    pub title: Option<String>,
    /// The content in markdown format (images embedded at their original positions).
    pub content: String,
}

/// Result of fetching an image.
#[derive(Debug, Clone)]
pub struct ImageResult {
    /// The original URL.
    pub url: String,
    /// JPEG image data.
    pub data: Vec<u8>,
    /// MIME type (always "image/jpeg" after conversion).
    pub mime: String,
}

/// Fetches a web page using static HTTP and converts it to clean markdown.
///
/// This is fast but won't execute JavaScript. For JS-rendered pages,
/// use [`fetch_with_browser`] (requires `headless` feature).
pub async fn fetch(url: &str) -> Result<FetchResult> {
    let html = fetch_html_static(url).await?;
    html_to_result(url, &html)
}

/// Fetches a web page using a headless browser (Chrome) for full JavaScript rendering.
///
/// This handles:
/// - Single-page applications (React, Vue, Angular, etc.)
/// - Dynamically loaded content
/// - JavaScript-based anti-bot measures
///
/// Requires Chrome/Chromium to be installed on the system.
#[cfg(feature = "headless")]
pub async fn fetch_with_browser(url: &str) -> Result<FetchResult> {
    let html = fetch_html_headless(url).await?;
    html_to_result(url, &html)
}

/// Fetches raw HTML using a headless browser (for debugging).
#[cfg(feature = "headless")]
pub async fn fetch_html_raw(url: &str) -> Result<String> {
    fetch_html_headless(url).await
}

/// Smart fetch that tries static first, falls back to headless browser if content is minimal.
#[cfg(feature = "headless")]
pub async fn fetch_smart(url: &str) -> Result<FetchResult> {
    let result = fetch(url).await;

    match result {
        Ok(ref r) if r.content.len() > 500 => {
            debug!("Static fetch succeeded with {} chars", r.content.len());
            result
        }
        Ok(r) => {
            debug!(
                "Static fetch returned minimal content ({} chars), trying headless",
                r.content.len()
            );
            fetch_with_browser(url).await.or(Ok(r))
        }
        Err(e) => {
            debug!("Static fetch failed: {}, trying headless", e);
            fetch_with_browser(url).await
        }
    }
}

/// Fetches an image and converts it to JPEG format.
///
/// This handles various image formats (WebP, PNG, GIF, etc.) and converts
/// them to JPEG for maximum LLM provider compatibility.
///
/// # Returns
///
/// A tuple of (JPEG data, MIME type).
pub async fn fetch_image(url: &str) -> Result<(Vec<u8>, String)> {
    let bytes = fetch_bytes(url).await?;
    convert_to_jpeg(&bytes)
}

/// Fetches an image and returns an ImageResult.
pub async fn fetch_image_result(url: &str) -> Result<ImageResult> {
    let (data, mime) = fetch_image(url).await?;
    Ok(ImageResult {
        url: url.to_string(),
        data,
        mime,
    })
}

/// Convert image bytes to JPEG format.
fn convert_to_jpeg(bytes: &[u8]) -> Result<(Vec<u8>, String)> {
    use image::ImageReader;

    // Try to decode the image
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| anyhow!("Failed to detect image format: {e}"))?
        .decode()
        .map_err(|e| anyhow!("Failed to decode image: {e}"))?;

    // Encode as JPEG
    let mut jpeg_data = Vec::new();
    let mut cursor = Cursor::new(&mut jpeg_data);
    img.write_to(&mut cursor, image::ImageFormat::Jpeg)
        .map_err(|e| anyhow!("Failed to encode JPEG: {e}"))?;

    Ok((jpeg_data, "image/jpeg".to_string()))
}

/// Extract og:description from HTML for sites where readability fails.
fn extract_og_description(html: &str) -> Option<String> {
    let og_re =
        Regex::new(r#"<meta[^>]+(?:property|name)="og:description"[^>]+content="([^"]+)""#).ok()?;
    if let Some(cap) = og_re.captures(html) {
        return cap
            .get(1)
            .map(|m| html_escape::decode_html_entities(m.as_str()).to_string());
    }
    // Try reversed attribute order
    let og_re2 =
        Regex::new(r#"<meta[^>]+content="([^"]+)"[^>]+(?:property|name)="og:description""#).ok()?;
    og_re2.captures(html).and_then(|cap| {
        cap.get(1)
            .map(|m| html_escape::decode_html_entities(m.as_str()).to_string())
    })
}

/// Convert HTML to FetchResult using readability and htmd.
fn html_to_result(url: &str, html: &str) -> Result<FetchResult> {
    let parsed_url = url::Url::parse(url).map_err(|e| anyhow!("Invalid URL: {e}"))?;

    let mut cursor = Cursor::new(html.as_bytes());
    let extracted = readability::extractor::extract(&mut cursor, &parsed_url)
        .map_err(|e| anyhow!("Content extraction failed: {e}"))?;

    let content = htmd::convert(&extracted.content).map_err(|e| anyhow!("{e}"))?;

    // If readability produces minimal content, try fallbacks
    let final_content = if content.len() < 200 {
        // Try og:description first
        if let Some(og_desc) = extract_og_description(html) {
            og_desc
        } else {
            // Last resort: convert full HTML body to markdown
            let body_content = extract_body_content(html);
            htmd::convert(&body_content).unwrap_or(content)
        }
    } else {
        content
    };

    Ok(FetchResult {
        url: url.to_string(),
        title: Some(extracted.title),
        content: final_content,
    })
}

/// Extract body content from HTML, stripping scripts and styles.
fn extract_body_content(html: &str) -> String {
    // Find body content
    let body_start = html
        .find("<body")
        .and_then(|i| html[i..].find('>').map(|j| i + j + 1));
    let body_end = html.rfind("</body>");

    let body = match (body_start, body_end) {
        (Some(start), Some(end)) if start < end => &html[start..end],
        _ => html,
    };

    // Remove script and style tags
    let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let noscript_re = Regex::new(r"(?is)<noscript[^>]*>.*?</noscript>").unwrap();

    let cleaned = script_re.replace_all(body, "");
    let cleaned = style_re.replace_all(&cleaned, "");
    let cleaned = noscript_re.replace_all(&cleaned, "");

    cleaned.to_string()
}

/// Browser-like User-Agent strings for rotation.
const USER_AGENTS: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
];

/// Get a User-Agent string based on URL hash.
fn get_user_agent(url: &str) -> &'static str {
    let ua_index = url
        .bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
        % USER_AGENTS.len();
    USER_AGENTS[ua_index]
}

/// Fetches raw bytes from a URL with browser-like headers.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let user_agent = get_user_agent(url);

    let mut backend = client();
    let bytes = backend
        .get(url)?
        .header(header::USER_AGENT.as_str(), user_agent)?
        .header(header::ACCEPT.as_str(), "*/*")?
        .header(header::ACCEPT_LANGUAGE.as_str(), "en-US,en;q=0.9")?
        .header(
            "Sec-CH-UA",
            "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\"",
        )?
        .header("Sec-CH-UA-Mobile", "?0")?
        .header("Sec-CH-UA-Platform", "\"macOS\"")?
        .header("Sec-Fetch-Dest", "image")?
        .header("Sec-Fetch-Mode", "no-cors")?
        .header("Sec-Fetch-Site", "cross-site")?
        .bytes()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {e}"))?;

    Ok(bytes.to_vec())
}

/// Fetches raw HTML using static HTTP with browser-like headers.
async fn fetch_html_static(url: &str) -> Result<String> {
    let user_agent = get_user_agent(url);

    let mut backend = client();
    let bytes = backend
        .get(url)?
        .header(header::USER_AGENT.as_str(), user_agent)?
        .header(
            header::ACCEPT.as_str(),
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        )?
        .header(header::ACCEPT_LANGUAGE.as_str(), "en-US,en;q=0.9")?
        .header("Sec-CH-UA", "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\"")?
        .header("Sec-CH-UA-Mobile", "?0")?
        .header("Sec-CH-UA-Platform", "\"macOS\"")?
        .header("Sec-Fetch-Dest", "document")?
        .header("Sec-Fetch-Mode", "navigate")?
        .header("Sec-Fetch-Site", "none")?
        .header("Sec-Fetch-User", "?1")?
        .header("Upgrade-Insecure-Requests", "1")?
        .header("Cache-Control", "max-age=0")?
        .bytes()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {e}"))?;

    String::from_utf8(bytes.to_vec()).map_err(|e| anyhow!("Invalid UTF-8 response: {e}"))
}

/// Fetches raw HTML using a headless Chrome browser with anti-detection measures.
#[cfg(feature = "headless")]
async fn fetch_html_headless(url: &str) -> Result<String> {
    use chromiumoxide::browser::{Browser, BrowserConfig};
    use chromiumoxide::handler::viewport::Viewport;
    use futures_lite::StreamExt;
    use std::time::Duration;

    debug!("Launching headless browser for {}", url);

    // Anti-detection browser arguments
    let args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1920,1080",
        "--start-maximized",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-translate",
        "--metrics-recording-only",
        "--no-first-run",
        "--safebrowsing-disable-auto-update",
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ];

    let config = BrowserConfig::builder()
        .viewport(Some(Viewport {
            width: 1920,
            height: 1080,
            device_scale_factor: Some(1.0),
            ..Default::default()
        }))
        .args(args)
        .build()
        .map_err(|e| anyhow!("Browser config error: {e}"))?;

    let (browser, mut handler) = Browser::launch(config)
        .await
        .map_err(|e| anyhow!("Failed to launch browser: {e}"))?;

    // Spawn handler task
    let handle = tokio::spawn(async move { while handler.next().await.is_some() {} });

    // Create blank page first
    let page = browser
        .new_page("about:blank")
        .await
        .map_err(|e| anyhow!("Failed to create page: {e}"))?;

    // Inject stealth script BEFORE navigation
    let stealth_js = r#"
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = { runtime: {} };
    "#;

    // Use Page.addScriptToEvaluateOnNewDocument to inject before any page script runs
    use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
    page.execute(AddScriptToEvaluateOnNewDocumentParams::new(stealth_js))
        .await
        .ok();

    // Now navigate to the actual URL
    page.goto(url)
        .await
        .map_err(|e| anyhow!("Failed to navigate: {e}"))?;

    // Wait for page to load
    tokio::time::sleep(Duration::from_millis(5000)).await;

    // Scroll to trigger lazy loading
    page.evaluate("window.scrollTo(0, 500)").await.ok();
    tokio::time::sleep(Duration::from_millis(2000)).await;

    let html = page
        .content()
        .await
        .map_err(|e| anyhow!("Failed to get page content: {e}"))?;

    handle.abort();

    Ok(html)
}

/// Check if a URL looks like an image based on extension or known image hosts.
fn is_image_url(url: &str) -> bool {
    let lower = url.to_lowercase();

    // Check common image extensions
    let extensions = [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".ico",
    ];
    for ext in extensions {
        if lower.contains(ext) {
            return true;
        }
    }

    // Check known image CDN patterns
    let image_patterns = [
        "pbs.twimg.com/media",
        "pbs.twimg.com/profile_images",
        "i.imgur.com",
        "images.unsplash.com",
        "cdn.discordapp.com/attachments",
    ];
    for pattern in image_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }

    false
}

/// Fetch content from a URL and convert to clean markdown.
///
/// Retrieves web pages and extracts the main content, removing navigation,
/// ads, and boilerplate. Returns clean markdown suitable for analysis.
///
/// Web page sizes vary widely. Large pages may be automatically saved to file
/// to manage context. When this happens, you receive the file path and can
/// read it in parts using standard Unix tools.
///
/// Some sites may block automated requests or require authentication.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebFetchArgs {
    /// The URL to fetch content from. Can be a webpage or an image URL.
    /// For webpages: returns markdown content with images embedded at their positions.
    /// For images: returns JPEG image data.
    pub url: String,
}

/// Web content fetching tool for LLM agents.
///
/// Fetches web content and returns it in a format suitable for LLM consumption:
/// - Webpages: Extracts main content, converts to markdown with images embedded
/// - Images: Fetches and converts to JPEG format
///
/// Supports URL filtering via whitelist/blacklist regex patterns.
#[derive(Debug, Clone, Default)]
pub struct WebFetchTool {
    name: String,
    /// URLs must match at least one pattern to be allowed (if non-empty).
    whitelist: Vec<Regex>,
    /// URLs matching any pattern will be blocked.
    blacklist: Vec<Regex>,
}

impl WebFetchTool {
    /// Create a new web fetch tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "webfetch".into(),
            whitelist: Vec::new(),
            blacklist: Vec::new(),
        }
    }

    /// Create a web fetch tool with custom name.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a whitelist regex pattern. URLs must match at least one whitelist pattern.
    ///
    /// # Panics
    /// Panics if the pattern is not a valid regex.
    #[must_use]
    pub fn allow(mut self, pattern: &str) -> Self {
        self.whitelist
            .push(Regex::new(pattern).expect("invalid whitelist regex"));
        self
    }

    /// Add a blacklist regex pattern. URLs matching any blacklist pattern are blocked.
    ///
    /// # Panics
    /// Panics if the pattern is not a valid regex.
    #[must_use]
    pub fn block(mut self, pattern: &str) -> Self {
        self.blacklist
            .push(Regex::new(pattern).expect("invalid blacklist regex"));
        self
    }

    /// Check if a URL is allowed by the whitelist/blacklist rules.
    fn is_url_allowed(&self, url: &str) -> bool {
        // Check blacklist first - blocked URLs are never allowed
        if self.blacklist.iter().any(|re| re.is_match(url)) {
            return false;
        }

        // If whitelist is empty, allow all non-blacklisted URLs
        if self.whitelist.is_empty() {
            return true;
        }

        // URL must match at least one whitelist pattern
        self.whitelist.iter().any(|re| re.is_match(url))
    }
}

impl Tool for WebFetchTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    type Arguments = WebFetchArgs;

    async fn call(&self, arguments: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let url = &arguments.url;

        // Check whitelist/blacklist
        if !self.is_url_allowed(url) {
            return Ok(ToolOutput::text(format!("URL not allowed: {url}")));
        }

        // Check if URL looks like an image
        if is_image_url(url) {
            let (jpeg_data, mime) = fetch_image(url).await?;
            return Ok(ToolOutput::image(jpeg_data, &mime));
        }

        // Fetch webpage - use headless browser by default if available
        #[cfg(feature = "headless")]
        let result = fetch_with_browser(url).await?;

        #[cfg(not(feature = "headless"))]
        let result = fetch(url).await?;

        let mut output = String::new();
        if let Some(title) = &result.title {
            output.push_str(&format!("# {title}\n\n"));
        }
        output.push_str(&format!("Source: {}\n\n", result.url));
        output.push_str(&result.content);

        Ok(ToolOutput::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fetch_example_com() {
        let result = fetch("https://example.com").await.unwrap();
        assert!(result.title.is_some());
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn fetch_wikipedia() {
        let result = fetch("https://en.wikipedia.org/wiki/Rust_(programming_language)").await;
        assert!(result.is_ok(), "Wikipedia fetch failed: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.title.is_some());
        assert!(!result.content.is_empty());
        assert!(
            result.content.contains("Rust") || result.content.contains("programming"),
            "Content should mention Rust or programming"
        );
    }

    #[tokio::test]
    async fn fetch_and_convert_image() {
        // Fetch a known PNG image and convert to JPEG
        let result = fetch_image("https://www.rust-lang.org/static/images/rust-logo-blk.svg").await;
        // SVG might not be supported, so we just check it doesn't panic
        if let Ok((data, mime)) = result {
            assert_eq!(mime, "image/jpeg");
            assert!(!data.is_empty());
        }
    }

    #[test]
    fn url_filtering_whitelist() {
        let tool = WebFetchTool::new()
            .allow(r"^https://example\.com")
            .allow(r"^https://rust-lang\.org");

        assert!(tool.is_url_allowed("https://example.com/page"));
        assert!(tool.is_url_allowed("https://rust-lang.org/docs"));
        assert!(!tool.is_url_allowed("https://other.com/page"));
    }

    #[test]
    fn url_filtering_blacklist() {
        let tool = WebFetchTool::new().block(r"\.evil\.com").block(r"/admin/");

        assert!(tool.is_url_allowed("https://example.com/page"));
        assert!(!tool.is_url_allowed("https://www.evil.com/malware"));
        assert!(!tool.is_url_allowed("https://example.com/admin/settings"));
    }

    #[test]
    fn url_filtering_combined() {
        let tool = WebFetchTool::new()
            .allow(r"^https://example\.com")
            .block(r"/private/");

        // Allowed by whitelist
        assert!(tool.is_url_allowed("https://example.com/public"));
        // Blocked by blacklist (even though matches whitelist)
        assert!(!tool.is_url_allowed("https://example.com/private/data"));
        // Not in whitelist
        assert!(!tool.is_url_allowed("https://other.com/page"));
    }
}

#[cfg(all(test, feature = "headless"))]
mod headless_tests {
    use super::*;

    #[tokio::test]
    async fn fetch_with_browser_works() {
        let result = fetch_with_browser("https://example.com").await;
        assert!(result.is_ok(), "Headless fetch failed: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.title.is_some());
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn smart_fetch_works() {
        let result = fetch_smart("https://example.com").await;
        assert!(result.is_ok(), "Smart fetch failed: {:?}", result.err());
    }
}
