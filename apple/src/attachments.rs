//! Resolving `Attachment` URLs into image bytes for native prompting.
//!
//! Everything stays on device: `file:` URLs are read locally, `data:` URLs are
//! decoded in place, and `http(s)` URLs are fetched once into memory — no
//! upload, no provider file API.

use aither_core::llm::Attachment;
use mime::Mime;
use zenwave::Client;

use crate::error::{AppleError, AttachmentError};

/// An image resolved to in-memory bytes plus its declared MIME type.
#[derive(Debug)]
pub struct ResolvedImage {
    pub bytes: Vec<u8>,
    pub media_type: String,
}

fn unsupported_media(media_type: &Mime) -> AppleError {
    AttachmentError::UnsupportedMediaType(media_type.essence_str().to_string()).into()
}

/// Resolves an attachment into image bytes.
///
/// The declared media type must be an image subtype the platform decodes;
/// the bridge validates it against `CGImageSource` before attaching.
pub async fn resolve_image(attachment: &Attachment) -> Result<ResolvedImage, AppleError> {
    let media_type = attachment.media_type();
    if media_type.type_() != mime::IMAGE {
        return Err(unsupported_media(media_type));
    }
    let bytes = match attachment.url().scheme() {
        "file" => {
            let path = attachment
                .url()
                .to_file_path()
                .map_err(|()| AttachmentError::UnsupportedScheme("file (non-local)".into()))?;
            async_fs::read(&path).await.map_err(AttachmentError::Io)?
        }
        "http" | "https" => {
            let mut backend = zenwave::client();
            let bytes = backend
                .get(attachment.url().as_str())
                .map_err(|e| AttachmentError::Http(e.to_string()))?
                .bytes()
                .await
                .map_err(|e| AttachmentError::Http(e.to_string()))?;
            bytes.to_vec()
        }
        "data" => decode_data_url(attachment.url())?,
        other => {
            return Err(AttachmentError::UnsupportedScheme(other.to_string()).into());
        }
    };
    Ok(ResolvedImage {
        bytes,
        media_type: media_type.essence_str().to_string(),
    })
}

fn decode_data_url(url: &url::Url) -> Result<Vec<u8>, AppleError> {
    let data = data_url::DataUrl::process(url.as_str())
        .map_err(|e| AttachmentError::InvalidDataUrl(e.to_string()))?;
    let (bytes, _) = data
        .decode_to_vec()
        .map_err(|e| AttachmentError::InvalidDataUrl(e.to_string()))?;
    Ok(bytes)
}
