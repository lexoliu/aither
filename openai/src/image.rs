use crate::{
    client::{Config, OpenAI},
    error::OpenAIError,
};
use aither_core::image::{Data, ImageGenerator, Prompt, Size};
use async_stream::try_stream;
use base64::{Engine as _, engine::general_purpose};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zenwave::{
    Client, client, header,
    multipart::{MultipartPart, encode as encode_multipart},
};

impl ImageGenerator for OpenAI {
    type Error = OpenAIError;

    fn create(
        &self,
        prompt: Prompt,
        size: Size,
    ) -> impl Stream<Item = Result<Data, Self::Error>> + Send {
        let cfg = self.config();
        let prompt_text = prompt.text().to_owned();
        let size_token = format_size(size);
        try_stream! {
            let images = generate_images(cfg, prompt_text, size_token).await?;
            for image in images {
                yield image;
            }
        }
    }

    fn edit(
        &self,
        prompt: Prompt,
        mask: &[u8],
    ) -> impl Stream<Item = Result<Data, Self::Error>> + Send {
        let cfg = self.config();
        let prompt_text = prompt.text().to_owned();
        let size_token = format_size(Size::square(1024));
        let base = prompt.images().first().cloned().ok_or_else(|| {
            OpenAIError::Api("image editing requires a base image via Prompt::with_image".into())
        });
        let mask_bytes = mask.to_vec();
        try_stream! {
            let base_image = base?;
            let images = edit_image(cfg, prompt_text, size_token, base_image, mask_bytes).await?;
            for image in images {
                yield image;
            }
        }
    }
}

async fn generate_images(
    cfg: Arc<Config>,
    prompt: String,
    size: String,
) -> Result<Vec<Data>, OpenAIError> {
    let endpoint = cfg.request_url("/images/generations");
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
    builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
    if let Some(org) = &cfg.organization {
        builder = builder.header("OpenAI-Organization", org.clone());
    }
    let request = ImageGenerationRequest {
        model: &cfg.image_model,
        prompt: &prompt,
        size: &size,
        response_format: "b64_json",
        n: 1,
    };
    builder = builder.json_body(&request).map_err(OpenAIError::from)?;
    let response: ImageResponse = builder.json().await.map_err(OpenAIError::from)?;
    response.into_images()
}

async fn edit_image(
    cfg: Arc<Config>,
    prompt: String,
    size: String,
    image: Vec<u8>,
    mask: Vec<u8>,
) -> Result<Vec<Data>, OpenAIError> {
    let endpoint = cfg.request_url("/images/edits");
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
    builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
    if let Some(org) = &cfg.organization {
        builder = builder.header("OpenAI-Organization", org.clone());
    }
    let mut parts = vec![
        MultipartPart::text("model", cfg.image_model.clone()),
        MultipartPart::text("prompt", prompt),
        MultipartPart::text("size", size),
        MultipartPart::text("response_format", "b64_json"),
        MultipartPart::binary("image", "image.png", "application/octet-stream", image),
    ];
    if !mask.is_empty() {
        parts.push(MultipartPart::binary(
            "mask",
            "mask.png",
            "application/octet-stream",
            mask,
        ));
    }
    let (boundary, body) = encode_multipart(parts);
    builder = builder.header(
        header::CONTENT_TYPE.as_str(),
        format!("multipart/form-data; boundary={boundary}"),
    );
    builder = builder.bytes_body(body);
    let response: ImageResponse = builder.json().await.map_err(OpenAIError::from)?;
    response.into_images()
}

fn format_size(size: Size) -> String {
    format!("{}x{}", size.width(), size.height())
}

#[derive(Debug, Serialize)]
struct ImageGenerationRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    size: &'a str,
    #[serde(rename = "response_format")]
    response_format: &'static str,
    n: u8,
}

#[derive(Debug, Deserialize)]
struct ImageResponse {
    data: Vec<ImagePayload>,
}

impl ImageResponse {
    fn into_images(self) -> Result<Vec<Data>, OpenAIError> {
        self.data
            .into_iter()
            .map(ImagePayload::into_bytes)
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct ImagePayload {
    #[serde(default)]
    b64_json: Option<String>,
}

impl ImagePayload {
    fn into_bytes(self) -> Result<Data, OpenAIError> {
        let encoded = self
            .b64_json
            .ok_or_else(|| OpenAIError::Api("image response missing `b64_json` field".into()))?;
        general_purpose::STANDARD
            .decode(encoded)
            .map_err(OpenAIError::from)
    }
}
