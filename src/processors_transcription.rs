// Direct transcription processor (current behavior)
use crate::processors::{AudioProcessor, ProcessContext, VoiceMode};
use anyhow::Result;
use std::any::Any;
use std::path::Path;

/// Backend type for transcription
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionBackend {
    WhisperCpp,
    OpenAICompat,
    OpenAI,
}

/// Configuration for transcription
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub backend: TranscriptionBackend,
    pub openai_compat_url: String,
    pub openai_compat_model: String,
    pub whisper_url: String,
    pub openai_url: String,
    pub openai_api_key: String,
    /// Model name sent to the OpenAI-compatible transcription endpoint.
    /// "whisper-1" for OpenAI cloud; overridable for local servers (e.g. "Whisper-Base").
    pub openai_model: String,
}

/// Direct transcription processor
///
/// This is the current voxtty behavior - just transcribe audio to text
pub struct TranscriptionProcessor {
    config: TranscriptionConfig,
}

impl TranscriptionProcessor {
    pub fn new(config: TranscriptionConfig) -> Self {
        Self { config }
    }

    fn transcribe_openai_compat(&self, audio_path: &Path) -> Result<String> {
        let client = reqwest::blocking::Client::new();
        let file = std::fs::read(audio_path)?;

        let form = reqwest::blocking::multipart::Form::new()
            .text("model", self.config.openai_compat_model.clone())
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(file).file_name(
                    audio_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                ),
            );

        #[derive(serde::Deserialize)]
        struct Response {
            text: String,
        }

        let response = client
            .post(&self.config.openai_compat_url)
            .multipart(form)
            .send()?
            .json::<Response>()?;

        Ok(response.text)
    }

    fn transcribe_whisper_cpp(&self, audio_path: &Path) -> Result<String> {
        let client = reqwest::blocking::Client::new();
        let file = std::fs::read(audio_path)?;

        let form = reqwest::blocking::multipart::Form::new()
            .text("temperature", "0.2")
            // OpenAI-style underscore param; the hyphenated form is ignored by
            // whisper.cpp (returns JSON instead of plain text).
            .text("response_format", "text")
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(file).file_name(
                    audio_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                ),
            );

        let response = client
            .post(&self.config.whisper_url)
            .multipart(form)
            .send()?
            .text()?;

        // Accept either plain text or a JSON {"text": ...} envelope.
        let body = response.trim();
        let text = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
            .unwrap_or_else(|| body.to_string());
        Ok(text.trim().to_string())
    }

    fn transcribe_openai(&self, audio_path: &Path) -> Result<String> {
        let client = reqwest::blocking::Client::new();
        let file = std::fs::read(audio_path)?;

        let form = reqwest::blocking::multipart::Form::new()
            .text("model", self.config.openai_model.clone())
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(file)
                    .file_name(
                        audio_path
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .to_string(),
                    )
                    .mime_str("audio/wav")?,
            );

        #[derive(serde::Deserialize)]
        struct Response {
            text: String,
        }

        let mut request = client.post(&self.config.openai_url).multipart(form);

        // Local OpenAI-compatible servers (e.g. Lemonade) usually need no auth.
        if !self.config.openai_api_key.is_empty() {
            request = request.header(
                "Authorization",
                format!("Bearer {}", self.config.openai_api_key),
            );
        }

        let response = request.send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI API error: {} - {}", status, error_text);
        }

        let result: Response = response.json()?;
        Ok(result.text)
    }
}

impl AudioProcessor for TranscriptionProcessor {
    fn process(&self, audio_path: &Path, context: &ProcessContext) -> Result<String> {
        if context.debug {
            println!(
                "[DEBUG] TranscriptionProcessor: Processing audio with {:?}",
                self.config.backend
            );
        }

        match self.config.backend {
            TranscriptionBackend::OpenAICompat => self.transcribe_openai_compat(audio_path),
            TranscriptionBackend::WhisperCpp => self.transcribe_whisper_cpp(audio_path),
            TranscriptionBackend::OpenAI => self.transcribe_openai(audio_path),
        }
    }

    fn name(&self) -> &str {
        "TranscriptionProcessor"
    }

    fn supports_mode(&self, mode: &VoiceMode) -> bool {
        matches!(mode, VoiceMode::Dictation)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
