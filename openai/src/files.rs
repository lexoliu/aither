//! `OpenAI` Files API client for file uploads.
//!
//! This module provides a client for the `OpenAI` Files API, which allows uploading
//! files for use in various endpoints (vision, assistants, fine-tuning).
//!
//! See: <https://platform.openai.com/docs/api-reference/files>

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_fs;
use serde::{Deserialize, Serialize};
use zenwave::{Client, client, header};

use crate::error::OpenAIError;
use crate::mime::mime_from_path;

/// The purpose of the uploaded file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FilePurpose {
    /// For use with vision endpoints.
    Vision,
    /// For use with Assistants API.
    Assistants,
    /// For use with fine-tuning.
    FineTune,
    /// For use with batch requests.
    Batch,
}

impl FilePurpose {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::Assistants => "assistants",
            Self::FineTune => "fine-tune",
            Self::Batch => "batch",
        }
    }
}

/// Status of a file in the `OpenAI` system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    /// File is being processed.
    Uploaded,
    /// File is ready for use.
    Processed,
    /// File processing resulted in an error.
    Error,
    /// File is being deleted.
    Deleting,
    /// Unknown status.
    #[serde(other)]
    Unknown,
}

/// A file object from the `OpenAI` Files API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFile {
    /// File ID (e.g., "file-abc123").
    pub id: String,
    /// Object type (always "file").
    pub object: String,
    /// Size of the file in bytes.
    pub bytes: u64,
    /// Unix timestamp when the file was created.
    pub created_at: i64,
    /// Name of the file.
    pub filename: String,
    /// Purpose of the file.
    pub purpose: String,
    /// Current status.
    #[serde(default)]
    pub status: Option<FileStatus>,
    /// Error message if processing failed.
    #[serde(default)]
    pub status_details: Option<String>,
}

impl OpenAIFile {
    /// Get the creation time as `SystemTime`.
    #[must_use]
    pub fn created(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(self.created_at as u64)
    }

    /// Check if the file is ready for use.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self.status, Some(FileStatus::Processed) | None)
    }
}

/// Response from deleting a file.
#[derive(Debug, Deserialize)]
pub struct DeleteFileResponse {
    /// File ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Whether the deletion was successful.
    pub deleted: bool,
}

/// Response from listing files.
#[derive(Debug, Deserialize)]
pub struct ListFilesResponse {
    /// Object type (always "list").
    pub object: String,
    /// The files.
    pub data: Vec<OpenAIFile>,
}

/// Configuration for `OpenAI` Files API.
#[derive(Debug, Clone)]
pub struct FilesConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL for the API.
    pub base_url: String,
    /// Organization ID (optional).
    pub organization: Option<String>,
}

impl FilesConfig {
    /// Create a new files configuration.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
        }
    }

    /// Set a custom base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the organization ID.
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    fn files_endpoint(&self) -> String {
        format!("{}/files", self.base_url.trim_end_matches('/'))
    }
}

/// Upload a file to `OpenAI`.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `path` - Path to the file to upload
/// * `purpose` - Purpose of the file
///
/// # Returns
/// The uploaded file metadata.
pub async fn upload_file(
    cfg: &FilesConfig,
    path: &Path,
    purpose: FilePurpose,
) -> Result<OpenAIFile, OpenAIError> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");

    let mime_type = mime_from_path(path).unwrap_or("application/octet-stream");

    // Read file content
    let data = async_fs::read(path)
        .await
        .map_err(|e| OpenAIError::Api(format!("Failed to read file: {e}")))?;

    let endpoint = cfg.files_endpoint();

    // Build multipart form data
    let boundary = format!(
        "----aither{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    let mut body = Vec::new();

    // Part 1: purpose field
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"purpose\"\r\n\r\n");
    body.extend_from_slice(purpose.as_str().as_bytes());
    body.extend_from_slice(b"\r\n");

    // Part 2: file field
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {mime_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(&data);
    body.extend_from_slice(b"\r\n");

    // End boundary
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let content_type = format!("multipart/form-data; boundary={boundary}");

    let mut backend = client();
    let mut builder = backend
        .post(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(header::CONTENT_TYPE.as_str(), &content_type)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            &format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    let builder = builder.bytes_body(body);

    builder.json().await.map_err(OpenAIError::from_http)
}

/// Delete a file from `OpenAI`.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `file_id` - ID of the file to delete
pub async fn delete_file(
    cfg: &FilesConfig,
    file_id: &str,
) -> Result<DeleteFileResponse, OpenAIError> {
    let endpoint = format!("{}/{}", cfg.files_endpoint(), file_id);

    let mut backend = client();
    let mut builder = backend
        .delete(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            &format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    builder.json().await.map_err(OpenAIError::from_http)
}

/// Get file metadata.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `file_id` - ID of the file
pub async fn get_file(cfg: &FilesConfig, file_id: &str) -> Result<OpenAIFile, OpenAIError> {
    let endpoint = format!("{}/{}", cfg.files_endpoint(), file_id);

    let mut backend = client();
    let mut builder = backend
        .get(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            &format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    builder.json().await.map_err(OpenAIError::from_http)
}

/// List uploaded files.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `purpose` - Optional filter by purpose
pub async fn list_files(
    cfg: &FilesConfig,
    purpose: Option<FilePurpose>,
) -> Result<ListFilesResponse, OpenAIError> {
    let mut endpoint = cfg.files_endpoint();
    if let Some(p) = purpose {
        endpoint.push_str("?purpose=");
        endpoint.push_str(p.as_str());
    }

    let mut backend = client();
    let mut builder = backend
        .get(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            &format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    builder.json().await.map_err(OpenAIError::from_http)
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
            mime_from_path(Path::new("/path/to/doc.pdf")),
            Some("application/pdf")
        );
    }

    #[test]
    fn test_file_purpose_as_str() {
        assert_eq!(FilePurpose::Vision.as_str(), "vision");
        assert_eq!(FilePurpose::Assistants.as_str(), "assistants");
        assert_eq!(FilePurpose::FineTune.as_str(), "fine-tune");
        assert_eq!(FilePurpose::Batch.as_str(), "batch");
    }
}
