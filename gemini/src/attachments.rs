use std::path::Path;

use aither_attachments::{FileCache, default_cache_dir};
use aither_core::llm::Message;
use url::Url;

use crate::config::GeminiConfig;
use crate::error::GeminiError;
use crate::files::upload_file;

pub(crate) async fn resolve_messages(
    cfg: &GeminiConfig,
    messages: Vec<Message>,
) -> Result<Vec<Message>, GeminiError> {
    if messages.iter().all(|msg| msg.attachments().is_empty()) {
        return Ok(messages);
    }

    let mut cache = FileCache::open(default_cache_dir()).await.map_err(GeminiError::from)?;
    let mut cache_dirty = cache.prune_expired();

    let mut resolved = Vec::with_capacity(messages.len());
    for message in messages {
        let Message::User { content, attachments } = message else {
            resolved.push(message);
            continue;
        };

        let mut next_attachments = Vec::with_capacity(attachments.len());
        for attachment in attachments {
            let resolved_url = resolve_attachment(cfg, &mut cache, &attachment).await?;
            cache_dirty |= resolved_url.cache_updated;
            next_attachments.push(resolved_url.url);
        }

        resolved.push(Message::User {
            content,
            attachments: next_attachments,
        });
    }

    if cache_dirty {
        cache.save().await.map_err(GeminiError::from)?;
    }

    Ok(resolved)
}

struct ResolvedUrl {
    url: Url,
    cache_updated: bool,
}

async fn resolve_attachment(
    cfg: &GeminiConfig,
    cache: &mut FileCache,
    attachment: &Url,
) -> Result<ResolvedUrl, GeminiError> {
    match attachment.scheme() {
        "file" => {
            let path = attachment.to_file_path().map_err(|_| {
                GeminiError::Api("Attachment file URL could not be converted to path".to_string())
            })?;
            resolve_file_attachment(cfg, cache, &path).await
        }
        "http" | "https" => {
            if is_gemini_file_uri(attachment) {
                Ok(ResolvedUrl {
                    url: attachment.clone(),
                    cache_updated: false,
                })
            } else {
                Err(GeminiError::Api(
                    "HTTP attachments must be uploaded via Gemini Files API".to_string(),
                ))
            }
        }
        "data" => Err(GeminiError::Api(
            "data: attachments must be uploaded via Gemini Files API".to_string(),
        )),
        other => Err(GeminiError::Api(format!(
            "Unsupported attachment URL scheme: {other}"
        ))),
    }
}

async fn resolve_file_attachment(
    cfg: &GeminiConfig,
    cache: &mut FileCache,
    path: &Path,
) -> Result<ResolvedUrl, GeminiError> {
    let provider = "gemini";

    if let Some(entry) = cache.get(path, provider).await.map_err(GeminiError::from)? {
        let url = Url::parse(&entry.reference)
            .map_err(|e| GeminiError::Api(format!("Invalid cached Gemini file URL: {e}")))?;
        return Ok(ResolvedUrl {
            url,
            cache_updated: false,
        });
    }

    let file = upload_file(cfg, path).await?;
    if !file.is_ready() {
        return Err(GeminiError::Api(format!(
            "Gemini file upload not ready (state: {:?})",
            file.state
        )));
    }
    if file.uri.is_empty() {
        return Err(GeminiError::Api("Gemini file upload missing URI".to_string()));
    }

    let expires_at = file.expiration();
    cache
        .insert(path, provider, file.uri.clone(), expires_at)
        .await
        .map_err(GeminiError::from)?;

    let url = Url::parse(&file.uri)
        .map_err(|e| GeminiError::Api(format!("Invalid Gemini file URI: {e}")))?;
    Ok(ResolvedUrl {
        url,
        cache_updated: true,
    })
}

fn is_gemini_file_uri(url: &Url) -> bool {
    url.host_str()
        .map(|h| h == "generativelanguage.googleapis.com")
        .unwrap_or(false)
}
