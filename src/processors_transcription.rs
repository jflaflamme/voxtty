// Direct transcription processor (current behavior)
use crate::processors::{AudioProcessor, ProcessContext, VoiceMode};
use anyhow::Result;
use std::any::Any;
use std::path::Path;

/// Backend type for transcription
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionBackend {
    WhisperCpp,
    Speaches,
    OpenAI,
}

/// Configuration for transcription
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub backend: TranscriptionBackend,
    pub speaches_url: String,
    pub speaches_model: String,
    pub whisper_url: String,
    pub openai_url: String,
    pub openai_api_key: String,
}

/// Direct transcription processor
///
/// This is the current VoiceTypr behavior - just transcribe audio to text
pub struct TranscriptionProcessor {
    config: TranscriptionConfig,
}

impl TranscriptionProcessor {
    pub fn new(config: TranscriptionConfig) -> Self {
        Self { config }
    }

    fn transcribe_speaches(&self, audio_path: &Path) -> Result<String> {
        let client = reqwest::blocking::Client::new();
        let file = std::fs::read(audio_path)?;

        let form = reqwest::blocking::multipart::Form::new()
            .text("model", self.config.speaches_model.clone())
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
            .post(&self.config.speaches_url)
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
            .text("response-format", "text")
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

        Ok(response.trim().to_string())
    }

    fn transcribe_openai(&self, audio_path: &Path) -> Result<String> {
        let client = reqwest::blocking::Client::new();
        let file = std::fs::read(audio_path)?;

        let form = reqwest::blocking::multipart::Form::new()
            .text("model", "whisper-1")
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

        let response = client
            .post(&self.config.openai_url)
            .header(
                "Authorization",
                format!("Bearer {}", self.config.openai_api_key),
            )
            .multipart(form)
            .send()?;

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
            TranscriptionBackend::Speaches => self.transcribe_speaches(audio_path),
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
