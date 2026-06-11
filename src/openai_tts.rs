// OpenAI-compatible TTS client
// Uses the OpenAI-compatible POST /v1/audio/speech endpoint.
// Works with any server implementing it (Lemonade/Kokoro, openedai-speech, LocalAI, etc.)
//
// API spec:
//   POST /v1/audio/speech
//   Body: { "input": "text", "model": "kokoro-v1", "voice": "shimmer", "speed": 1.0, "response_format": "mp3" }
//   Response: raw audio bytes

use anyhow::{Context, Result};
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, Sink};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sample rate of raw PCM streamed by the server (qwen3-tts emits 24 kHz mono s16le).
const STREAM_SAMPLE_RATE: u32 = 24000;

/// OpenAI-compatible TTS client (e.g. Lemonade/Kokoro)
pub struct OpenAiTts {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub voice: String,
    pub speed: f32,
    /// Optional style/tone instruction (e.g. "Speak in a calm, neutral tone").
    /// Sent as the `instruct` field; ignored by servers that don't support it.
    pub instruct: Option<String>,
    /// Stream PCM audio and play as it arrives (low latency). Falls back to
    /// buffered playback if the server doesn't support streaming.
    pub stream: bool,
    /// Sampling temperature (e.g. Qwen3-TTS, default 0.9 server-side).
    /// Lower = steadier, more uniform delivery. None = omit the field.
    pub temperature: Option<f32>,
    /// Random seed for reproducible delivery. None = omit the field.
    pub seed: Option<i64>,
}

impl OpenAiTts {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            base_url,
            api_key,
            model: "kokoro-v1".to_string(),
            voice: "shimmer".to_string(),
            speed: 1.0,
            instruct: None,
            stream: true,
            temperature: None,
            seed: None,
        }
    }

    /// Enable/disable streaming playback (default: enabled).
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set an optional sampling temperature (sent as `temperature`).
    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set an optional generation seed (sent as `seed`).
    pub fn with_seed(mut self, seed: Option<i64>) -> Self {
        self.seed = seed;
        self
    }

    /// Add optional generation fields shared by both request paths.
    fn extend_payload(&self, payload: &mut serde_json::Value) {
        if let Some(ref instruct) = self.instruct {
            payload["instruct"] = serde_json::Value::String(instruct.clone());
        }
        if let Some(temperature) = self.temperature {
            payload["temperature"] = serde_json::json!(temperature);
        }
        if let Some(seed) = self.seed {
            payload["seed"] = serde_json::json!(seed);
        }
    }

    /// Set the model identifier (e.g. "kokoro-v1")
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Set an optional style/tone instruction (sent as `instruct`).
    pub fn with_instruct(mut self, instruct: Option<String>) -> Self {
        self.instruct = instruct.filter(|s| !s.is_empty());
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

        if self.stream {
            match self.speak_streaming(text, interrupt_flag.clone()) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    eprintln!("TTS streaming failed ({e}); falling back to buffered playback");
                }
            }
        }

        // Generate audio via HTTP
        let audio_bytes = self.generate_audio(text)?;

        // Decode and play (takes ownership)
        self.play_audio_bytes(audio_bytes, interrupt_flag)
    }

    /// Stream raw PCM from the server and play chunks as they arrive.
    /// First audio is audible ~0.4s after the request instead of after the
    /// whole utterance is generated.
    fn speak_streaming(
        &self,
        text: &str,
        interrupt_flag: Option<std::sync::Arc<AtomicBool>>,
    ) -> Result<()> {
        let url = format!("{}/v1/audio/speech", self.base_url);

        let mut payload = serde_json::json!({
            "input": text,
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": "pcm",
            "stream": true
        });
        self.extend_payload(&mut payload);

        // No timeout: slow TTS servers (VoxCPM2 on iGPU generates ~2x slower than
        // realtime) legitimately stream for minutes; reqwest's 30s default would
        // cut playback mid-sentence.
        let mut request = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .context("Failed to build HTTP client")?
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload);
        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }

        let mut response = request.send()?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI-compatible TTS error: {} - {}", status, error_text);
        }

        // If the server ignored `stream`/`pcm` and sent an encoded format,
        // buffer and decode it instead of misreading it as raw samples.
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !(content_type.contains("pcm") || content_type.contains("octet-stream")) {
            let mut bytes = Vec::new();
            response.read_to_end(&mut bytes)?;
            return self.play_audio_bytes(bytes, interrupt_flag);
        }

        let (_stream, stream_handle) =
            OutputStream::try_default().context("Failed to create audio output stream")?;
        let sink = Sink::try_new(&stream_handle).context("Failed to create audio sink")?;

        // Pre-buffer before starting playback so servers that generate slower
        // than realtime (e.g. VoxCPM2-Khmer on the iGPU) don't stutter. Fast
        // servers fill the buffer instantly, so this adds no latency for them.
        // If the sink still drains mid-stream, pause and re-buffer the same
        // amount before resuming. TTS_PREBUFFER_SECS=0 restores play-at-once.
        let prebuffer_secs: f32 = std::env::var("TTS_PREBUFFER_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1.5);
        let prebuffer_samples = (prebuffer_secs * STREAM_SAMPLE_RATE as f32) as usize;
        let mut playing = prebuffer_samples == 0;
        if !playing {
            sink.pause();
        }
        let mut queued_samples: usize = 0;

        // s16le mono samples; carry holds the byte of a sample split across reads.
        let mut carry: Vec<u8> = Vec::new();
        let mut buf = [0u8; 32 * 1024];
        loop {
            if let Some(ref interrupt) = interrupt_flag {
                if interrupt.load(Ordering::SeqCst) {
                    eprintln!("🛑 TTS interrupted during streaming");
                    sink.clear();
                    sink.stop();
                    return Ok(());
                }
            }
            let n = response.read(&mut buf)?;
            if n == 0 {
                break;
            }
            carry.extend_from_slice(&buf[..n]);
            let usable = carry.len() - (carry.len() % 2);
            if usable > 0 {
                let samples: Vec<i16> = carry[..usable]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]))
                    .collect();
                carry.drain(..usable);
                if playing && prebuffer_samples > 0 && sink.empty() {
                    // Underrun: playback outran generation; re-buffer before resuming.
                    sink.pause();
                    playing = false;
                    queued_samples = 0;
                }
                queued_samples += samples.len();
                sink.append(SamplesBuffer::new(1, STREAM_SAMPLE_RATE, samples));
                if !playing && queued_samples >= prebuffer_samples {
                    sink.play();
                    playing = true;
                }
            }
        }
        if !playing {
            // Stream ended before the buffer filled (short utterance): play it out.
            sink.play();
        }

        // Drain remaining playback, honoring interrupts.
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

    /// Generate audio bytes from text (non-blocking, returns bytes)
    pub fn generate_audio(&self, text: &str) -> Result<Vec<u8>> {
        let url = format!("{}/v1/audio/speech", self.base_url);

        let mut payload = serde_json::json!({
            "input": text,
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": "mp3"
        });
        // Optional tone/style/sampling control (Qwen3-TTS etc.). Harmless on servers that ignore them.
        self.extend_payload(&mut payload);

        // No timeout: slow TTS servers (VoxCPM2 on iGPU generates ~2x slower than
        // realtime) legitimately stream for minutes; reqwest's 30s default would
        // cut playback mid-sentence.
        let mut request = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .context("Failed to build HTTP client")?
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
