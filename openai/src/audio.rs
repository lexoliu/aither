use crate::{
    client::{Config, OpenAI},
    error::OpenAIError,
};
use aither_core::audio::{AudioGenerator, AudioTranscriber, Data};
use futures_lite::StreamExt;
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
        futures_lite::stream::iter(vec![synthesize(cfg, text)])
            .then(|fut| fut)
            .map(|result| handle_audio_result(result, "synthesis"))
    }
}

impl AudioTranscriber for OpenAI {
    fn transcribe(&self, audio: &[u8]) -> impl futures_core::Stream<Item = String> + Send {
        let cfg = self.config();
        let payload = audio.to_vec();
        futures_lite::stream::iter(vec![transcribe_once(cfg, payload)])
            .then(|fut| fut)
            .map(handle_transcription_result)
    }
}

async fn synthesize(cfg: Arc<Config>, text: String) -> Result<Vec<u8>, OpenAIError> {
    let endpoint = cfg.request_url("/audio/speech");
    let mut backend = client();
    let mut builder = backend
        .post(endpoint)
        .map_err(|e| OpenAIError::Http(e))?
        .header(header::AUTHORIZATION.as_str(), cfg.request_auth())
        .map_err(|e| OpenAIError::Http(e))?
        .header(header::USER_AGENT.as_str(), "aither-openai/0.1")
        .map_err(|e| OpenAIError::Http(e))?;
    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org.clone())
            .map_err(|e| OpenAIError::Http(e))?;
    }

    let request = SpeechRequest {
        model: &cfg.audio_model,
        input: &text,
        voice: &cfg.audio_voice,
        format: &cfg.audio_format,
    };
    let bytes = builder
        .json_body(&request)
        .map_err(|e| OpenAIError::Http(e))?
        .bytes()
        .await
        .map_err(|e| OpenAIError::Http(e))?;

    Ok(bytes.to_vec())
}

async fn transcribe_once(cfg: Arc<Config>, audio: Vec<u8>) -> Result<String, OpenAIError> {
    let endpoint = cfg.request_url("/audio/transcriptions");
    let mut backend = client();
    let mut builder = backend
        .post(endpoint)
        .map_err(|e| OpenAIError::Http(e))?
        .header(header::AUTHORIZATION.as_str(), cfg.request_auth())
        .map_err(|e| OpenAIError::Http(e))?
        .header(header::USER_AGENT.as_str(), "aither-openai/0.1")
        .map_err(|e| OpenAIError::Http(e))?;
    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org.clone())
            .map_err(|e| OpenAIError::Http(e))?;
    }

    let parts = vec![
        MultipartPart::text("model", cfg.transcription_model.clone()),
        MultipartPart::text("response_format", "json"),
        MultipartPart::binary("file", "audio.wav", "application/octet-stream", audio),
    ];

    let (boundary, body) = encode_multipart(parts);
    let response: TranscriptionResponse = builder
        .header(
            header::CONTENT_TYPE.as_str(),
            format!("multipart/form-data; boundary={boundary}"),
        )
        .map_err(|e| OpenAIError::Http(e))?
        .bytes_body(body)
        .json()
        .await
        .map_err(|e| OpenAIError::Http(e))?;

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
