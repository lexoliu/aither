use crate::{
    client::{Config, OpenAI},
    error::OpenAIError,
};
use aither_core::audio::{AudioGenerator, AudioTranscriber, Data};
use async_stream::stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zenwave::{
    Client, client, header,
    multipart::{MultipartPart, encode as encode_multipart},
};

impl AudioGenerator for OpenAI {
    fn generate(&self, prompt: &str) -> impl futures_core::Stream<Item = Data> + Send {
        let cfg = self.config();
        let text = prompt.to_owned();
        stream! {
            let result = synthesize(cfg, text).await;
            yield handle_audio_result(result, "synthesis");
        }
    }
}

impl AudioTranscriber for OpenAI {
    fn transcribe(&self, audio: &[u8]) -> impl futures_core::Stream<Item = String> + Send {
        let cfg = self.config();
        let payload = audio.to_vec();
        stream! {
            let result = transcribe_once(cfg, payload).await;
            yield handle_transcription_result(result);
        }
    }
}

async fn synthesize(cfg: Arc<Config>, text: String) -> Result<Vec<u8>, OpenAIError> {
    let endpoint = cfg.request_url("/audio/speech");
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
    builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
    if let Some(org) = &cfg.organization {
        builder = builder.header("OpenAI-Organization", org.clone());
    }

    let request = SpeechRequest {
        model: &cfg.audio_model,
        input: &text,
        voice: &cfg.audio_voice,
        format: &cfg.audio_format,
    };
    builder = builder.json_body(&request).map_err(OpenAIError::from)?;
    let bytes = builder.bytes().await.map_err(OpenAIError::from)?;
    Ok(bytes.to_vec())
}

async fn transcribe_once(cfg: Arc<Config>, audio: Vec<u8>) -> Result<String, OpenAIError> {
    let endpoint = cfg.request_url("/audio/transcriptions");
    let mut backend = client();
    let mut builder = backend.post(endpoint);
    builder = builder.header(header::AUTHORIZATION.as_str(), cfg.request_auth());
    builder = builder.header(header::USER_AGENT.as_str(), "aither-openai/0.1");
    if let Some(org) = &cfg.organization {
        builder = builder.header("OpenAI-Organization", org.clone());
    }

    let parts = vec![
        MultipartPart::text("model", cfg.transcription_model.clone()),
        MultipartPart::text("response_format", "json"),
        MultipartPart::binary("file", "audio.wav", "application/octet-stream", audio),
    ];

    let (boundary, body) = encode_multipart(parts);
    builder = builder.header(
        header::CONTENT_TYPE.as_str(),
        format!("multipart/form-data; boundary={boundary}"),
    );
    builder = builder.bytes_body(body);

    let response: TranscriptionResponse = builder.json().await.map_err(OpenAIError::from)?;
    Ok(response.text)
}

#[derive(Debug, Serialize)]
struct SpeechRequest<'a> {
    model: &'a str,
    input: &'a str,
    voice: &'a str,
    #[serde(rename = "response_format")]
    format: &'a str,
}

#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    text: String,
}

fn handle_audio_result(result: Result<Vec<u8>, OpenAIError>, context: &'static str) -> Vec<u8> {
    match result {
        Ok(bytes) => bytes,
        Err(err) => {
            assert!(
                !cfg!(debug_assertions),
                "OpenAI audio {context} failed: {err}"
            );
            Vec::new()
        }
    }
}

fn handle_transcription_result(result: Result<String, OpenAIError>) -> String {
    match result {
        Ok(text) => text,
        Err(err) => {
            assert!(
                !cfg!(debug_assertions),
                "OpenAI audio transcription failed: {err}"
            );
            String::new()
        }
    }
}
