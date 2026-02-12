use url::Url;

use crate::error::OpenAIError;

#[cfg(not(target_arch = "wasm32"))]
use aither_core::llm::Message;
#[cfg(not(target_arch = "wasm32"))]
use crate::client::Config;

#[cfg(not(target_arch = "wasm32"))]
use aither_attachments::{FileCache, default_cache_dir};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use crate::files::{FilePurpose, FilesConfig, upload_file};
#[cfg(not(target_arch = "wasm32"))]
use crate::mime::mime_from_path;

#[derive(Clone, Copy)]
pub struct OpenAIFileKind(&'static str);

impl OpenAIFileKind {
    pub(crate) const IMAGE: Self = Self("image");
    pub(crate) const FILE: Self = Self("file");

    pub(crate) fn is_image(self) -> bool {
        self.0 == Self::IMAGE.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn resolve_messages(
    cfg: &Config,
    messages: Vec<Message>,
) -> Result<Vec<Message>, OpenAIError> {
    if messages.iter().all(|msg| msg.attachments().is_empty()) {
        return Ok(messages);
    }

    let mut cache = FileCache::open(default_cache_dir())
        .await
        .map_err(OpenAIError::from)?;
    let mut cache_dirty = cache.prune_expired();

    let mut resolved = Vec::with_capacity(messages.len());
    for message in messages {
        let Message::User {
            content,
            attachments,
        } = message
        else {
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
        cache.save().await.map_err(OpenAIError::from)?;
    }

    Ok(resolved)
}

#[cfg(not(target_arch = "wasm32"))]
struct ResolvedUrl {
    url: Url,
    cache_updated: bool,
}

#[cfg(not(target_arch = "wasm32"))]
async fn resolve_attachment(
    cfg: &Config,
    cache: &mut FileCache,
    attachment: &Url,
) -> Result<ResolvedUrl, OpenAIError> {
    match attachment.scheme() {
        "file" => {
            let path = attachment.to_file_path().map_err(|()| {
                OpenAIError::Api("Attachment file URL could not be converted to path".to_string())
            })?;
            resolve_file_attachment(cfg, cache, &path).await
        }
        "http" | "https" | "data" => Ok(ResolvedUrl {
            url: attachment.clone(),
            cache_updated: false,
        }),
        other => Err(OpenAIError::Api(format!(
            "Unsupported attachment URL scheme: {other}"
        ))),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn resolve_file_attachment(
    cfg: &Config,
    cache: &mut FileCache,
    path: &Path,
) -> Result<ResolvedUrl, OpenAIError> {
    let kind = file_kind_for_path(path)?;
    let provider = "openai";

    if let Some(entry) = cache.get(path, provider).await.map_err(OpenAIError::from)? {
        let url = build_openai_file_url(kind, &entry.reference)?;
        return Ok(ResolvedUrl {
            url,
            cache_updated: false,
        });
    }

    let files_cfg = build_files_config(cfg);
    let purpose = purpose_for_kind(kind);
    let file = upload_file(&files_cfg, path, purpose).await?;
    if !file.is_ready() {
        return Err(OpenAIError::Api(format!(
            "OpenAI file upload not ready (status: {:?})",
            file.status
        )));
    }

    cache
        .insert(path, provider, file.id.clone(), None)
        .await
        .map_err(OpenAIError::from)?;

    let url = build_openai_file_url(kind, &file.id)?;
    Ok(ResolvedUrl {
        url,
        cache_updated: true,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn build_files_config(cfg: &Config) -> FilesConfig {
    let mut files_cfg = FilesConfig::new(cfg.api_key.clone()).with_base_url(cfg.base_url.clone());
    if let Some(org) = &cfg.organization {
        files_cfg = files_cfg.with_organization(org.clone());
    }
    files_cfg
}

#[cfg(not(target_arch = "wasm32"))]
fn file_kind_for_path(path: &Path) -> Result<OpenAIFileKind, OpenAIError> {
    let mime = mime_from_path(path).ok_or_else(|| {
        OpenAIError::Api(format!(
            "Unable to infer MIME type for attachment: {}",
            path.display()
        ))
    })?;
    Ok(if mime.starts_with("image/") {
        OpenAIFileKind::IMAGE
    } else {
        OpenAIFileKind::FILE
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn purpose_for_kind(kind: OpenAIFileKind) -> FilePurpose {
    if kind.is_image() {
        FilePurpose::Vision
    } else {
        FilePurpose::Assistants
    }
}

fn build_openai_file_url(kind: OpenAIFileKind, id: &str) -> Result<Url, OpenAIError> {
    let raw = format!("openai://{}/{}", kind.0, id);
    Url::parse(&raw).map_err(|e| OpenAIError::Api(format!("Invalid OpenAI file URL: {e}")))
}

pub fn parse_openai_file_url(url: &Url) -> Option<(OpenAIFileKind, String)> {
    if url.scheme() != "openai" {
        return None;
    }
    let kind = match url.host_str()? {
        "image" => OpenAIFileKind::IMAGE,
        "file" => OpenAIFileKind::FILE,
        _ => return None,
    };
    let id = url.path_segments()?.next()?.to_string();
    Some((kind, id))
}
