// OpenAI-compatible TTS client
// Uses the OpenAI-compatible POST /v1/audio/speech endpoint.
// Works with any server implementing it (Lemonade/Kokoro, openedai-speech, LocalAI, etc.)
//
// API spec:
//   POST /v1/audio/speech
//   Body: { "input": "text", "model": "kokoro-v1", "voice": "shimmer", "speed": 1.0, "response_format": "mp3" }
//   Response: raw audio bytes

use anyhow::{Context, Result};
use rodio::{OutputStream, Sink};
use std::sync::atomic::{AtomicBool, Ordering};

/// OpenAI-compatible TTS client (e.g. Lemonade/Kokoro)
pub struct OpenAiTts {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub voice: String,
    pub speed: f32,
}

impl OpenAiTts {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            base_url,
            api_key,
            model: "kokoro-v1".to_string(),
            voice: "shimmer".to_string(),
            speed: 1.0,
        }
    }

    /// Set the model identifier (e.g. "kokoro-v1")
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Set the voice identifier (e.g. "shimmer", "bella", "josh", etc.)
    pub fn with_voice(mut self, voice: String) -> Self {
        self.voice = voice;
        self
    }

    /// Set the playback speed multiplier (e.g. 0.5 for half speed, 2.0 for double)
    #[allow(dead_code)]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Speak text with interrupt support.
    /// Returns immediately after starting playback; caller controls interruption via flag.
    pub fn speak_interruptible(
        &self,
        text: &str,
        interrupt_flag: Option<std::sync::Arc<AtomicBool>>,
    ) -> Result<()> {
        eprintln!("🔊 TTS (OpenAI-compatible): {}", text);

        // Generate audio via HTTP
        let audio_bytes = self.generate_audio(text)?;

        // Decode and play (takes ownership)
        self.play_audio_bytes(audio_bytes, interrupt_flag)
    }

    /// Generate audio bytes from text (non-blocking, returns bytes)
    pub fn generate_audio(&self, text: &str) -> Result<Vec<u8>> {
        let url = format!("{}/v1/audio/speech", self.base_url);

        let payload = serde_json::json!({
            "input": text,
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": "mp3"
        });

        let mut request = reqwest::blocking::Client::new()
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload);

        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI-compatible TTS error: {} - {}", status, error_text);
        }

        Ok(response.bytes()?.to_vec())
    }

    /// Play audio bytes (mp3 or wav) with rodio, with optional interrupt support
    pub fn play_audio_bytes(
        &self,
        bytes: Vec<u8>,
        interrupt_flag: Option<std::sync::Arc<AtomicBool>>,
    ) -> Result<()> {
        let (_stream, stream_handle) =
            OutputStream::try_default().context("Failed to create audio output stream")?;

        let sink = Sink::try_new(&stream_handle).context("Failed to create audio sink")?;

        // Try mp3 first, fall back to wav. Cursor owns the bytes so the decoder
        // gets a 'static source.
        if let Ok(source) = rodio::Decoder::new_mp3(std::io::Cursor::new(bytes.clone())) {
            sink.append(source);
        } else {
            // Fallback: try generic decoder (works for wav)
            let source = rodio::Decoder::new(std::io::Cursor::new(bytes))
                .context("Failed to decode audio (not mp3 or wav)")?;
            sink.append(source);
        }

        // Wait for playback to complete, checking for interrupts
        if let Some(interrupt) = interrupt_flag {
            while !sink.empty() {
                if interrupt.load(Ordering::SeqCst) {
                    eprintln!("🛑 TTS interrupted during playback");
                    sink.clear();
                    sink.stop();
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        } else {
            sink.sleep_until_end();
        }

        Ok(())
    }
}
