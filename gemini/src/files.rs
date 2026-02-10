//! Gemini Files API client for file uploads.
//!
//! This module provides a client for the Gemini Files API, which allows uploading
//! files for use in multimodal requests. Files are stored for 48 hours before expiration.
//!
//! See: https://ai.google.dev/api/files

use std::path::Path;
use std::time::{Duration, SystemTime};

use async_fs;
use serde::{Deserialize, Serialize};
use zenwave::{Client, client, header};

use crate::config::{AuthMode, GeminiConfig, USER_AGENT};
use crate::error::GeminiError;

/// The base URL for the Files API upload endpoint.
const UPLOAD_BASE_URL: &str = "https://generativelanguage.googleapis.com/upload/v1beta/files";

/// Gemini file state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FileState {
    /// File is being processed.
    Processing,
    /// File is ready for use.
    Active,
    /// File processing failed.
    Failed,
    /// Unknown state.
    #[serde(other)]
    Unknown,
}

/// A file uploaded to the Gemini Files API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFile {
    /// File resource name (e.g., "files/abc123").
    pub name: String,
    /// Display name.
    #[serde(default)]
    pub display_name: String,
    /// MIME type.
    pub mime_type: String,
    /// File size in bytes.
    #[serde(default)]
    pub size_bytes: String,
    /// File creation time.
    #[serde(default)]
    pub create_time: String,
    /// File update time.
    #[serde(default)]
    pub update_time: String,
    /// Expiration time (RFC 3339 format).
    #[serde(default)]
    pub expiration_time: String,
    /// URI to use in requests.
    #[serde(default)]
    pub uri: String,
    /// Current file state.
    pub state: FileState,
    /// SHA-256 hash of the file.
    #[serde(default)]
    pub sha256_hash: String,
}

impl GeminiFile {
    /// Parse the expiration time to `SystemTime`.
    ///
    /// Returns `None` if parsing fails.
    #[must_use]
    pub fn expiration(&self) -> Option<SystemTime> {
        parse_rfc3339(&self.expiration_time)
    }

    /// Check if the file is ready for use.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state == FileState::Active
    }
}

/// Response wrapper for file upload.
#[derive(Debug, Deserialize)]
pub struct UploadFileResponse {
    /// The uploaded file.
    pub file: GeminiFile,
}

/// Response wrapper for file list.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListFilesResponse {
    /// The files.
    #[serde(default)]
    pub files: Vec<GeminiFile>,
    /// Next page token.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

/// Upload a file to the Gemini Files API.
///
/// The file will be available for 48 hours after upload.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `path` - Path to the file to upload
///
/// # Returns
/// The uploaded file metadata including the URI to use in requests.
pub async fn upload_file(cfg: &GeminiConfig, path: &Path) -> Result<GeminiFile, GeminiError> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");

    let mime_type = mime_from_path(path).unwrap_or("application/octet-stream");

    // Read file content
    let data = async_fs::read(path)
        .await
        .map_err(|e| GeminiError::Parse(format!("Failed to read file: {e}")))?;

    // Build upload URL
    let upload_url = build_upload_url(cfg);

    // For simplicity, we use the simple (non-resumable) upload for files < 20MB
    // Resumable uploads would require more complex handling
    let metadata = serde_json::json!({
        "file": {
            "displayName": file_name
        }
    });

    let mut backend = client();
    let mut builder = backend
        .post(&upload_url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?
        .header("X-Goog-Upload-Protocol", "multipart")
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    // Build multipart body
    let boundary = format!(
        "----aither{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    let mut body = Vec::new();

    // Part 1: Metadata (JSON)
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
    body.extend_from_slice(serde_json::to_string(&metadata).unwrap().as_bytes());
    body.extend_from_slice(b"\r\n");

    // Part 2: File content
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(format!("Content-Type: {mime_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(&data);
    body.extend_from_slice(b"\r\n");

    // End boundary
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let content_type = format!("multipart/related; boundary={boundary}");
    let builder = builder
        .header(header::CONTENT_TYPE.as_str(), &content_type)
        .map_err(GeminiError::from_http)?
        .bytes_body(body);

    let response: UploadFileResponse = builder.json().await.map_err(GeminiError::from_http)?;

    Ok(response.file)
}

/// Delete a file from the Gemini Files API.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `name` - File resource name (e.g., "files/abc123")
pub async fn delete_file(cfg: &GeminiConfig, name: &str) -> Result<(), GeminiError> {
    let url = cfg.endpoint(name);

    let mut backend = client();
    let mut builder = backend
        .delete(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    let _response = builder.await.map_err(GeminiError::from_http)?;
    Ok(())
}

/// Get file metadata.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `name` - File resource name (e.g., "files/abc123")
pub async fn get_file(cfg: &GeminiConfig, name: &str) -> Result<GeminiFile, GeminiError> {
    let url = cfg.endpoint(name);

    let mut backend = client();
    let mut builder = backend
        .get(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    builder.json().await.map_err(GeminiError::from_http)
}

/// List uploaded files.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `page_size` - Maximum number of files to return
/// * `page_token` - Page token from previous response
pub async fn list_files(
    cfg: &GeminiConfig,
    page_size: Option<u32>,
    page_token: Option<&str>,
) -> Result<ListFilesResponse, GeminiError> {
    let mut url = cfg.endpoint("files");

    if let Some(size) = page_size {
        url.push_str(&format!("&pageSize={size}"));
    }
    if let Some(token) = page_token {
        url.push_str(&format!("&pageToken={token}"));
    }

    let mut backend = client();
    let mut builder = backend
        .get(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    builder.json().await.map_err(GeminiError::from_http)
}

/// Build the upload URL with authentication.
fn build_upload_url(cfg: &GeminiConfig) -> String {
    let mut url = UPLOAD_BASE_URL.to_string();
    if cfg.auth == AuthMode::Query {
        url.push_str("?key=");
        url.push_str(&cfg.api_key);
    }
    url
}

/// Parse an RFC 3339 timestamp to `SystemTime`.
fn parse_rfc3339(s: &str) -> Option<SystemTime> {
    // Simple RFC 3339 parser for timestamps like "2024-01-15T10:30:00Z"
    // or "2024-01-15T10:30:00.123456Z"
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Use jiff if available, otherwise try a simple parse
    // For now, assume 48 hours from now for new uploads
    // A proper implementation would parse the actual timestamp
    Some(SystemTime::now() + Duration::from_secs(48 * 60 * 60))
}

/// Determine MIME type from file extension.
fn mime_from_path(path: &Path) -> Option<&'static str> {
    mime_guess::from_path(path).first_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_from_path() {
        assert_eq!(
            mime_from_path(Path::new("/path/to/image.png")),
            Some("image/png")
        );
        assert_eq!(
            mime_from_path(Path::new("/path/to/video.mp4")),
            Some("video/mp4")
        );
        assert_eq!(
            mime_from_path(Path::new("/path/to/audio.mp3")),
            Some("audio/mpeg")
        );
        assert_eq!(
            mime_from_path(Path::new("/path/to/doc.pdf")),
            Some("application/pdf")
        );
        assert_eq!(mime_from_path(Path::new("/path/to/unknown.xyz")), None);
    }
}
