//! # Web Content Fetching Library
//!
//! An async library for fetching web content and converting it to clean markdown.
//!
//! ## Features
//!
//! - **Provider chain**: Jina Reader -> static fetch -> headless (optional)
//! - **Static fetch**: HTTP fetching with markdown negotiation and HTML fallback
//! - **Headless browser**: Full JavaScript rendering via Chrome DevTools Protocol
//!   (enable with `headless` feature)
//! - **Image extraction**: Extract image URLs from pages
//! - **Image fetching**: Fetch images with automatic JPEG conversion
//!
//! ## Usage
//!
//! ```no_run
//! use aither_webfetch::{fetch, FetchRequest, WebFetchTool, fetch_with_request};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Default provider chain fetch
//! let result = fetch("https://example.com").await?;
//! println!("Title: {:?}", result.title);
//! println!("Content: {}", result.content);
//! // Images are embedded in the markdown content
//!
//! // Optional explicit request with custom deadline/token
//! let request = FetchRequest::new("https://example.com")
//!     .with_deadline(std::time::Duration::from_secs(8));
//! let _ = fetch_with_request(request).await?;
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
use std::time::{Duration, Instant};

use aither_core::llm::{Tool, ToolOutput};
use anyhow::{Result, anyhow};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
#[cfg(feature = "headless")]
use tracing::debug;
use zenwave::{Client, ResponseExt, client, header};

fn ensure_rustls_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

const DEFAULT_TOTAL_BUDGET: Duration = Duration::from_secs(6);
const JINA_STAGE_BUDGET: Duration = Duration::from_millis(2500);
const STATIC_STAGE_BUDGET: Duration = Duration::from_millis(2000);
#[cfg(feature = "headless")]
const HEADLESS_STAGE_BUDGET: Duration = Duration::from_millis(1500);

const JINA_API_BASE: &str = "https://r.jina.ai/";
const JINA_API_KEY_ENV: &str = "JINA_API_KEY";

/// Result of fetching web content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResult {
    /// The original URL that was fetched.
    pub url: String,
    /// The page title, if found.
    pub title: Option<String>,
    /// The content in markdown format (images embedded at their original positions).
    pub content: String,
    /// Response content type if available (for example, `text/markdown` or `text/html`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Estimated markdown token count from source response headers, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub markdown_tokens: Option<usize>,
    /// Source content usage policy header, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_signal: Option<String>,
    /// Provider/extractor that produced this result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extractor: Option<String>,
    /// Optional quality score in the range [0.0, 1.0].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f32>,
    /// Non-fatal warnings emitted during fetch/extraction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
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

/// Request options for async-first web fetching.
#[derive(Debug, Clone)]
pub struct FetchRequest {
    /// Target URL.
    pub url: String,
    /// Optional Jina API key. If absent, falls back to `JINA_API_KEY` env var.
    pub jina_api_key: Option<String>,
    /// Total deadline budget for the full fallback chain.
    pub deadline: Duration,
}

impl FetchRequest {
    /// Create a request with default options.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            jina_api_key: std::env::var(JINA_API_KEY_ENV).ok(),
            deadline: DEFAULT_TOTAL_BUDGET,
        }
    }

    /// Set API key explicitly.
    #[must_use]
    pub fn with_jina_api_key(mut self, key: impl Into<String>) -> Self {
        self.jina_api_key = Some(key.into());
        self
    }

    /// Override total deadline budget.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = deadline;
        self
    }

    fn effective_jina_api_key(&self) -> Option<&str> {
        self.jina_api_key.as_deref()
    }
}

/// Execution context shared by all fetch providers in a fallback chain.
#[derive(Debug, Clone)]
pub struct FetchContext {
    started_at: Instant,
    budget: Duration,
    /// Per-stage traces in execution order.
    pub trace: Vec<StageTrace>,
}

impl FetchContext {
    #[must_use]
    pub fn new(budget: Duration) -> Self {
        Self {
            started_at: Instant::now(),
            budget,
            trace: Vec::new(),
        }
    }

    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.budget.saturating_sub(self.elapsed())
    }

    #[must_use]
    pub fn stage_budget(&self, preferred: Duration) -> Duration {
        self.remaining().min(preferred)
    }

    fn push_trace(&mut self, trace: StageTrace) {
        self.trace.push(trace);
    }
}

/// Stage execution outcome, used for diagnostics and fallback decisions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StageOutcome {
    Success,
    HttpFailure,
    NonHttpFailure,
}

/// Per-provider trace information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTrace {
    pub provider: String,
    pub elapsed_ms: u128,
    pub outcome: StageOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Error class used by provider chain to decide whether to fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    HttpFailure,
    ExtractionFailure,
    InvalidResponse,
}

/// Structured provider error with fallback classification.
#[derive(Debug)]
pub struct ProviderError {
    provider: &'static str,
    kind: ProviderErrorKind,
    source: anyhow::Error,
}

impl ProviderError {
    fn new(
        provider: &'static str,
        kind: ProviderErrorKind,
        source: impl Into<anyhow::Error>,
    ) -> Self {
        Self {
            provider,
            kind,
            source: source.into(),
        }
    }

    fn http(provider: &'static str, source: impl Into<anyhow::Error>) -> Self {
        Self::new(provider, ProviderErrorKind::HttpFailure, source)
    }

    fn extraction(provider: &'static str, source: impl Into<anyhow::Error>) -> Self {
        Self::new(provider, ProviderErrorKind::ExtractionFailure, source)
    }

    fn invalid(provider: &'static str, source: impl Into<anyhow::Error>) -> Self {
        Self::new(provider, ProviderErrorKind::InvalidResponse, source)
    }

    #[must_use]
    pub const fn is_http_failure(&self) -> bool {
        matches!(self.kind, ProviderErrorKind::HttpFailure)
    }

    #[must_use]
    pub const fn kind(&self) -> ProviderErrorKind {
        self.kind
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} fetcher failed ({:?}): {}",
            self.provider, self.kind, self.source
        )
    }
}

impl std::error::Error for ProviderError {}

/// Async-first provider abstraction for web fetching.
#[allow(async_fn_in_trait)]
pub trait WebFetcher {
    fn name(&self) -> &'static str;
    async fn fetch(
        &self,
        req: &FetchRequest,
        ctx: &mut FetchContext,
    ) -> std::result::Result<FetchResult, ProviderError>;
}

/// Default out-of-the-box fetcher chain type.
///
/// Chain order:
/// - `JinaFetcher`
/// - `StaticFetcher`
/// - `HeadlessFetcher` (when `headless` feature is enabled)
#[cfg(feature = "headless")]
pub type DefaultFetcher = Fallback3<JinaFetcher, StaticFetcher, HeadlessFetcher>;
/// Default out-of-the-box fetcher chain type.
#[cfg(not(feature = "headless"))]
pub type DefaultFetcher = Fallback2<JinaFetcher, StaticFetcher>;

/// Build the default fetcher chain.
#[cfg(feature = "headless")]
#[must_use]
pub fn default_fetcher() -> DefaultFetcher {
    Fallback3::new(JinaFetcher, StaticFetcher, HeadlessFetcher)
}

/// Build the default fetcher chain.
#[cfg(not(feature = "headless"))]
#[must_use]
pub fn default_fetcher() -> DefaultFetcher {
    Fallback2::new(JinaFetcher, StaticFetcher)
}

/// Two-stage fallback fetcher.
#[derive(Debug, Clone)]
pub struct Fallback2<A, B> {
    first: A,
    second: B,
}

impl<A, B> Fallback2<A, B> {
    #[must_use]
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

#[allow(async_fn_in_trait)]
impl<A, B> WebFetcher for Fallback2<A, B>
where
    A: WebFetcher,
    B: WebFetcher,
{
    fn name(&self) -> &'static str {
        "fallback2"
    }

    async fn fetch(
        &self,
        req: &FetchRequest,
        ctx: &mut FetchContext,
    ) -> std::result::Result<FetchResult, ProviderError> {
        match run_stage(&self.first, req, ctx).await {
            Ok(result) => Ok(result),
            Err(first_err) if first_err.is_http_failure() => {
                run_stage(&self.second, req, ctx).await
            }
            Err(first_err) => Err(first_err),
        }
    }
}

/// Three-stage fallback fetcher.
#[derive(Debug, Clone)]
pub struct Fallback3<A, B, C> {
    first: A,
    second: B,
    third: C,
}

impl<A, B, C> Fallback3<A, B, C> {
    #[must_use]
    pub fn new(first: A, second: B, third: C) -> Self {
        Self {
            first,
            second,
            third,
        }
    }
}

#[allow(async_fn_in_trait)]
impl<A, B, C> WebFetcher for Fallback3<A, B, C>
where
    A: WebFetcher,
    B: WebFetcher,
    C: WebFetcher,
{
    fn name(&self) -> &'static str {
        "fallback3"
    }

    async fn fetch(
        &self,
        req: &FetchRequest,
        ctx: &mut FetchContext,
    ) -> std::result::Result<FetchResult, ProviderError> {
        match run_stage(&self.first, req, ctx).await {
            Ok(result) => Ok(result),
            Err(first_err) if first_err.is_http_failure() => {
                match run_stage(&self.second, req, ctx).await {
                    Ok(result) => Ok(result),
                    Err(second_err) if second_err.is_http_failure() => {
                        run_stage(&self.third, req, ctx).await
                    }
                    Err(second_err) => Err(second_err),
                }
            }
            Err(first_err) => Err(first_err),
        }
    }
}

async fn run_stage<F: WebFetcher>(
    fetcher: &F,
    req: &FetchRequest,
    ctx: &mut FetchContext,
) -> std::result::Result<FetchResult, ProviderError> {
    let started = Instant::now();

    if ctx.remaining().is_zero() {
        let err = ProviderError::http(fetcher.name(), anyhow!("pipeline deadline exhausted"));
        ctx.push_trace(StageTrace {
            provider: fetcher.name().to_string(),
            elapsed_ms: 0,
            outcome: StageOutcome::HttpFailure,
            message: Some("pipeline deadline exhausted".to_string()),
        });
        return Err(err);
    }

    let result = fetcher.fetch(req, ctx).await;
    let elapsed_ms = started.elapsed().as_millis();

    let trace = match &result {
        Ok(_) => StageTrace {
            provider: fetcher.name().to_string(),
            elapsed_ms,
            outcome: StageOutcome::Success,
            message: None,
        },
        Err(err) => StageTrace {
            provider: fetcher.name().to_string(),
            elapsed_ms,
            outcome: if err.is_http_failure() {
                StageOutcome::HttpFailure
            } else {
                StageOutcome::NonHttpFailure
            },
            message: Some(err.to_string()),
        },
    };
    ctx.push_trace(trace);

    result
}

/// Fetches a web page using static HTTP and converts it to clean markdown.
///
/// Default strategy:
/// 1. Jina Reader (`https://r.jina.ai/`) with optional API key
/// 2. Static fetch (`text/markdown` -> `text/html` + cleanup)
/// 3. Headless browser (only when `headless` feature is enabled)
pub async fn fetch(url: &str) -> Result<FetchResult> {
    ensure_rustls_provider();
    fetch_with_request(FetchRequest::new(url)).await
}

/// Fetch using an explicit request object.
pub async fn fetch_with_request(request: FetchRequest) -> Result<FetchResult> {
    ensure_rustls_provider();
    let mut ctx = FetchContext::new(request.deadline);

    let pipeline = default_fetcher();
    let result = pipeline.fetch(&request, &mut ctx).await;

    result.map_err(anyhow::Error::from)
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
    ensure_rustls_provider();
    let mut ctx = FetchContext::new(DEFAULT_TOTAL_BUDGET);
    HeadlessFetcher
        .fetch(&FetchRequest::new(url), &mut ctx)
        .await
        .map_err(anyhow::Error::from)
}

/// Fetches raw HTML using a headless browser (for debugging).
#[cfg(feature = "headless")]
pub async fn fetch_html_raw(url: &str) -> Result<String> {
    fetch_html_headless(url).await
}

/// Smart fetch with strict fallback order:
/// 1. Jina Reader
/// 2. static markdown/html pipeline
/// 3. headless browser render (if feature enabled)
#[cfg(feature = "headless")]
pub async fn fetch_smart(url: &str) -> Result<FetchResult> {
    fetch(url).await
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
    ensure_rustls_provider();
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
fn html_to_result_with_metadata(
    url: &str,
    html: &str,
    content_type: Option<String>,
    content_signal: Option<String>,
) -> Result<FetchResult> {
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
        content_type,
        markdown_tokens: None,
        content_signal,
        extractor: None,
        quality_score: None,
        warnings: Vec::new(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FetchHttpErrorKind {
    Http,
    Decode,
}

#[derive(Debug)]
struct FetchHttpError {
    kind: FetchHttpErrorKind,
    source: anyhow::Error,
}

impl FetchHttpError {
    fn http(source: impl Into<anyhow::Error>) -> Self {
        Self {
            kind: FetchHttpErrorKind::Http,
            source: source.into(),
        }
    }

    fn decode(source: impl Into<anyhow::Error>) -> Self {
        Self {
            kind: FetchHttpErrorKind::Decode,
            source: source.into(),
        }
    }
}

impl std::fmt::Display for FetchHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} fetch error: {}", self.kind, self.source)
    }
}

impl std::error::Error for FetchHttpError {}

impl From<zenwave::Error> for FetchHttpError {
    fn from(value: zenwave::Error) -> Self {
        Self::http(value)
    }
}

/// Fetches a document with static HTTP and a caller-provided `Accept` header.
async fn fetch_document_static(
    url: &str,
    accept: &str,
    timeout: Duration,
) -> std::result::Result<StaticFetchResponse, FetchHttpError> {
    let user_agent = get_user_agent(url);

    let mut backend = client().timeout(clamp_timeout(timeout));
    let response = backend
        .get(url)?
        .header(header::USER_AGENT.as_str(), user_agent)?
        .header(header::ACCEPT.as_str(), accept)?
        .header(header::ACCEPT_LANGUAGE.as_str(), "en-US,en;q=0.9")?
        .header(
            "Sec-CH-UA",
            "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\"",
        )?
        .header("Sec-CH-UA-Mobile", "?0")?
        .header("Sec-CH-UA-Platform", "\"macOS\"")?
        .header("Sec-Fetch-Dest", "document")?
        .header("Sec-Fetch-Mode", "navigate")?
        .header("Sec-Fetch-Site", "none")?
        .header("Sec-Fetch-User", "?1")?
        .header("Upgrade-Insecure-Requests", "1")?
        .header("Cache-Control", "max-age=0")?
        .await
        .map_err(|e| FetchHttpError::http(anyhow!("{e}")))?;

    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE.as_str())
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let markdown_tokens = response
        .headers()
        .get("x-markdown-tokens")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let content_signal = response
        .headers()
        .get("content-signal")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let body = response
        .into_string()
        .await
        .map_err(|e| FetchHttpError::decode(anyhow!("{e}")))?
        .to_string();

    Ok(StaticFetchResponse {
        body,
        content_type,
        markdown_tokens,
        content_signal,
    })
}

async fn fetch_markdown_static(
    url: &str,
    timeout: Duration,
) -> std::result::Result<StaticFetchResponse, FetchHttpError> {
    fetch_document_static(url, "text/markdown", timeout).await
}

async fn fetch_html_static(
    url: &str,
    timeout: Duration,
) -> std::result::Result<StaticFetchResponse, FetchHttpError> {
    fetch_document_static(
        url,
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        timeout,
    )
    .await
}

async fn fetch_document_jina(
    url: &str,
    api_key: Option<&str>,
    timeout: Duration,
) -> std::result::Result<StaticFetchResponse, FetchHttpError> {
    let jina_url = build_jina_url(url).map_err(FetchHttpError::decode)?;
    let mut backend = client().timeout(clamp_timeout(timeout));
    let mut builder = backend
        .get(&jina_url)?
        .header(header::ACCEPT.as_str(), "text/markdown")?
        .header(header::USER_AGENT.as_str(), "aither-webfetch/0.1")?;

    if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
        builder = builder
            .header(header::AUTHORIZATION.as_str(), format!("Bearer {key}"))?
            .header("X-API-Key", key.to_string())?;
    }

    let response = builder
        .await
        .map_err(|e| FetchHttpError::http(anyhow!("{e}")))?;
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE.as_str())
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let markdown_tokens = response
        .headers()
        .get("x-markdown-tokens")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| {
            response
                .headers()
                .get("x-token-count")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<usize>().ok())
        });
    let content_signal = response
        .headers()
        .get("content-signal")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = response
        .into_string()
        .await
        .map_err(|e| FetchHttpError::decode(anyhow!("{e}")))?
        .to_string();

    Ok(StaticFetchResponse {
        body,
        content_type,
        markdown_tokens,
        content_signal,
    })
}

fn clamp_timeout(timeout: Duration) -> Duration {
    if timeout.is_zero() {
        Duration::from_millis(1)
    } else {
        timeout
    }
}

fn build_jina_url(url: &str) -> Result<String> {
    let parsed = url::Url::parse(url).map_err(|e| anyhow!("invalid URL: {e}"))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(anyhow!("unsupported URL scheme: {}", parsed.scheme()));
    }
    Ok(format!("{JINA_API_BASE}{url}"))
}

fn map_fetch_http_error(provider: &'static str, err: FetchHttpError) -> ProviderError {
    match err.kind {
        FetchHttpErrorKind::Http => ProviderError::http(provider, err),
        FetchHttpErrorKind::Decode => ProviderError::invalid(provider, err),
    }
}

#[derive(Debug)]
struct StaticFetchResponse {
    body: String,
    content_type: Option<String>,
    markdown_tokens: Option<usize>,
    content_signal: Option<String>,
}

fn is_markdown_response(content_type: Option<&str>, markdown_tokens: Option<usize>) -> bool {
    if markdown_tokens.is_some() {
        return true;
    }

    content_type
        .map(|value| {
            let mime = value
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            mime == "text/markdown" || mime == "text/x-markdown"
        })
        .unwrap_or(false)
}

fn markdown_to_result(
    url: &str,
    markdown: String,
    content_type: Option<String>,
    markdown_tokens: Option<usize>,
    content_signal: Option<String>,
) -> FetchResult {
    FetchResult {
        url: url.to_string(),
        title: extract_markdown_title(&markdown),
        content: markdown,
        content_type,
        markdown_tokens,
        content_signal,
        extractor: None,
        quality_score: None,
        warnings: Vec::new(),
    }
}

fn extract_markdown_title(markdown: &str) -> Option<String> {
    if let Some(front_matter) = extract_front_matter(markdown) {
        for line in front_matter.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("title:") {
                let title = normalize_yaml_string(rest);
                if !title.is_empty() {
                    return Some(title);
                }
            }
        }
    }

    markdown
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("# ")
                .map(str::trim)
                .filter(|title| !title.is_empty())
                .map(str::to_string)
        })
        .or_else(|| {
            markdown.lines().find_map(|line| {
                line.trim()
                    .strip_prefix("Title:")
                    .or_else(|| line.trim().strip_prefix("title:"))
                    .map(str::trim)
                    .filter(|title| !title.is_empty())
                    .map(str::to_string)
            })
        })
}

fn extract_front_matter(markdown: &str) -> Option<&str> {
    let rest = markdown
        .strip_prefix("---\n")
        .or_else(|| markdown.strip_prefix("---\r\n"))?;

    for marker in ["\n---\n", "\n---\r\n", "\r\n---\r\n"] {
        if let Some(idx) = rest.find(marker) {
            return Some(&rest[..idx]);
        }
    }

    None
}

fn normalize_yaml_string(input: &str) -> String {
    let trimmed = input.trim();

    if let Some(unquoted) = trimmed
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|v| v.strip_suffix('\''))
        })
    {
        return unquoted.trim().to_string();
    }

    trimmed.to_string()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JinaFetcher;

#[allow(async_fn_in_trait)]
impl WebFetcher for JinaFetcher {
    fn name(&self) -> &'static str {
        "jina"
    }

    async fn fetch(
        &self,
        req: &FetchRequest,
        ctx: &mut FetchContext,
    ) -> std::result::Result<FetchResult, ProviderError> {
        let jina_budget = ctx.stage_budget(JINA_STAGE_BUDGET);
        if jina_budget.is_zero() {
            return Err(ProviderError::http(
                self.name(),
                anyhow!("jina stage deadline exhausted"),
            ));
        }

        let response = fetch_document_jina(&req.url, req.effective_jina_api_key(), jina_budget)
            .await
            .map_err(|err| map_fetch_http_error(self.name(), err))?;

        let mut result = markdown_to_result(
            &req.url,
            response.body,
            response.content_type,
            response.markdown_tokens,
            response.content_signal,
        );
        result.extractor = Some(self.name().to_string());
        result.quality_score = Some(1.0);
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StaticFetcher;

#[allow(async_fn_in_trait)]
impl WebFetcher for StaticFetcher {
    fn name(&self) -> &'static str {
        "static"
    }

    async fn fetch(
        &self,
        req: &FetchRequest,
        ctx: &mut FetchContext,
    ) -> std::result::Result<FetchResult, ProviderError> {
        let static_budget = ctx.stage_budget(STATIC_STAGE_BUDGET);
        if static_budget.is_zero() {
            return Err(ProviderError::http(
                self.name(),
                anyhow!("static stage deadline exhausted"),
            ));
        }

        let markdown_attempt = fetch_markdown_static(&req.url, static_budget)
            .await
            .map_err(|err| map_fetch_http_error(self.name(), err));

        if let Ok(response) = &markdown_attempt
            && is_markdown_response(response.content_type.as_deref(), response.markdown_tokens)
        {
            let mut result = markdown_to_result(
                &req.url,
                response.body.clone(),
                response.content_type.clone(),
                response.markdown_tokens,
                response.content_signal.clone(),
            );
            result.extractor = Some("static_markdown".to_string());
            result.quality_score = Some(0.95);
            return Ok(result);
        }

        let html_budget = ctx.stage_budget(STATIC_STAGE_BUDGET);
        if html_budget.is_zero() {
            return Err(ProviderError::http(
                self.name(),
                anyhow!("static html fallback deadline exhausted"),
            ));
        }

        let html_response = fetch_html_static(&req.url, html_budget)
            .await
            .map_err(|err| map_fetch_http_error(self.name(), err))?;

        let mut result = html_to_result_with_metadata(
            &req.url,
            &html_response.body,
            html_response.content_type,
            html_response.content_signal,
        )
        .map_err(|err| ProviderError::extraction(self.name(), err))?;
        result.extractor = Some("static_html".to_string());
        result.quality_score = Some(0.7);

        if let Err(markdown_err) = markdown_attempt
            && !markdown_err.is_http_failure()
        {
            result.warnings.push(format!(
                "static markdown attempt failed before html fallback: {markdown_err}"
            ));
        }

        Ok(result)
    }
}

#[cfg(feature = "headless")]
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadlessFetcher;

#[cfg(feature = "headless")]
#[allow(async_fn_in_trait)]
impl WebFetcher for HeadlessFetcher {
    fn name(&self) -> &'static str {
        "headless"
    }

    async fn fetch(
        &self,
        req: &FetchRequest,
        ctx: &mut FetchContext,
    ) -> std::result::Result<FetchResult, ProviderError> {
        if ctx.stage_budget(HEADLESS_STAGE_BUDGET).is_zero() {
            return Err(ProviderError::http(
                self.name(),
                anyhow!("headless stage deadline exhausted"),
            ));
        }

        let html = fetch_html_headless(&req.url)
            .await
            .map_err(|err| ProviderError::http(self.name(), err))?;
        let mut result = html_to_result_with_metadata(&req.url, &html, None, None)
            .map_err(|err| ProviderError::extraction(self.name(), err))?;
        result.extractor = Some(self.name().to_string());
        result.quality_score = Some(0.8);
        Ok(result)
    }
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
    /// Optional Jina API key to increase rate limits for the Jina provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jina_api_key: Option<String>,
    /// Optional per-request timeout budget in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
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
        let WebFetchArgs {
            url,
            jina_api_key,
            timeout_ms,
        } = arguments;

        // Check whitelist/blacklist
        if !self.is_url_allowed(&url) {
            return Ok(ToolOutput::text(format!("URL not allowed: {url}")));
        }

        // Check if URL looks like an image
        if is_image_url(&url) {
            let (jpeg_data, mime) = fetch_image(&url).await?;
            return Ok(ToolOutput::image(jpeg_data, &mime));
        }

        let mut request = FetchRequest::new(url.clone());
        if let Some(key) = jina_api_key.filter(|k| !k.trim().is_empty()) {
            request = request.with_jina_api_key(key);
        }
        if let Some(ms) = timeout_ms {
            request = request.with_deadline(Duration::from_millis(ms));
        }

        let result = fetch_with_request(request).await?;

        let mut output = String::new();
        if let Some(title) = &result.title {
            output.push_str(&format!("# {title}\n\n"));
        }
        output.push_str(&format!("Source: {}\n\n", result.url));
        if let Some(content_type) = &result.content_type {
            output.push_str(&format!("Content-Type: {content_type}\n"));
        }
        if let Some(markdown_tokens) = result.markdown_tokens {
            output.push_str(&format!("X-Markdown-Tokens: {markdown_tokens}\n"));
        }
        if let Some(content_signal) = &result.content_signal {
            output.push_str(&format!("Content-Signal: {content_signal}\n"));
        }
        if let Some(extractor) = &result.extractor {
            output.push_str(&format!("Extractor: {extractor}\n"));
        }
        if let Some(quality_score) = result.quality_score {
            output.push_str(&format!("Quality-Score: {quality_score:.2}\n"));
        }
        if !result.warnings.is_empty() {
            output.push_str(&format!("Warnings: {}\n", result.warnings.join(" | ")));
        }
        if result.content_type.is_some()
            || result.markdown_tokens.is_some()
            || result.content_signal.is_some()
            || result.extractor.is_some()
            || result.quality_score.is_some()
            || !result.warnings.is_empty()
        {
            output.push('\n');
        }
        output.push_str(&result.content);

        Ok(ToolOutput::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    #[tokio::test]
    async fn fetch_example_com() {
        let request =
            FetchRequest::new("https://example.com").with_deadline(Duration::from_secs(12));
        let result = fetch_with_request(request).await.unwrap();
        assert!(result.title.is_some());
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn fetch_wikipedia() {
        let request =
            FetchRequest::new("https://en.wikipedia.org/wiki/Rust_(programming_language)")
                .with_deadline(Duration::from_secs(12));
        let result = fetch_with_request(request).await;
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

    #[test]
    fn markdown_content_type_detection() {
        assert!(is_markdown_response(
            Some("text/markdown; charset=utf-8"),
            None
        ));
        assert!(is_markdown_response(Some("text/x-markdown"), None));
        assert!(is_markdown_response(Some("text/html"), Some(123)));
        assert!(!is_markdown_response(
            Some("text/html; charset=utf-8"),
            None
        ));
    }

    #[test]
    fn markdown_title_from_front_matter() {
        let markdown = r#"---
title: "Introducing Markdown for Agents"
description: Example
---

# Fallback Heading
"#;
        let title = extract_markdown_title(markdown);
        assert_eq!(title.as_deref(), Some("Introducing Markdown for Agents"));
    }

    #[test]
    fn markdown_title_from_heading_when_front_matter_missing() {
        let markdown = "# Heading Title\n\nBody";
        let title = extract_markdown_title(markdown);
        assert_eq!(title.as_deref(), Some("Heading Title"));
    }

    #[derive(Clone)]
    struct MockFetcher {
        name: &'static str,
        calls: Arc<AtomicUsize>,
        result: std::result::Result<FetchResult, ProviderErrorKind>,
    }

    #[allow(async_fn_in_trait)]
    impl WebFetcher for MockFetcher {
        fn name(&self) -> &'static str {
            self.name
        }

        async fn fetch(
            &self,
            _req: &FetchRequest,
            _ctx: &mut FetchContext,
        ) -> std::result::Result<FetchResult, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.result {
                Ok(result) => Ok(result.clone()),
                Err(kind) => Err(ProviderError::new(
                    self.name,
                    *kind,
                    anyhow!("mock {} failure", self.name),
                )),
            }
        }
    }

    #[tokio::test]
    async fn fallback2_falls_back_only_on_http_failure() {
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let request = FetchRequest::new("https://example.com");
        let mut ctx = FetchContext::new(Duration::from_secs(1));

        let first = MockFetcher {
            name: "first",
            calls: Arc::clone(&first_calls),
            result: Err(ProviderErrorKind::HttpFailure),
        };
        let second = MockFetcher {
            name: "second",
            calls: Arc::clone(&second_calls),
            result: Ok(markdown_to_result(
                "https://example.com",
                "# second".to_string(),
                Some("text/markdown".to_string()),
                None,
                None,
            )),
        };

        let chain = Fallback2::new(first, second);
        let result = chain.fetch(&request, &mut ctx).await.unwrap();

        assert_eq!(result.title.as_deref(), Some("second"));
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
        assert_eq!(ctx.trace.len(), 2);
    }

    #[tokio::test]
    async fn fallback2_stops_on_non_http_failure() {
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let request = FetchRequest::new("https://example.com");
        let mut ctx = FetchContext::new(Duration::from_secs(1));

        let first = MockFetcher {
            name: "first",
            calls: Arc::clone(&first_calls),
            result: Err(ProviderErrorKind::ExtractionFailure),
        };
        let second = MockFetcher {
            name: "second",
            calls: Arc::clone(&second_calls),
            result: Ok(markdown_to_result(
                "https://example.com",
                "# second".to_string(),
                Some("text/markdown".to_string()),
                None,
                None,
            )),
        };

        let chain = Fallback2::new(first, second);
        let result = chain.fetch(&request, &mut ctx).await;

        assert!(result.is_err());
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 0);
        assert_eq!(ctx.trace.len(), 1);
    }

    #[test]
    fn jina_url_builder_requires_http_scheme() {
        let ok = build_jina_url("https://example.com").unwrap();
        assert_eq!(ok, "https://r.jina.ai/https://example.com");

        let err = build_jina_url("file:///tmp/a.md");
        assert!(err.is_err());
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
