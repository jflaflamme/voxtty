// ElevenLabs TTS WebSocket client
use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tungstenite::client::IntoClientRequest;

/// Pronunciation dictionary locator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationDictionaryLocator {
    pub pronunciation_dictionary_id: String,
    pub version_id: String,
}

/// ElevenLabs TTS client using WebSocket
pub struct ElevenLabsTts {
    api_key: String,
    voice_id: String,
    base_url: String,
    pronunciation_dict: Option<PronunciationDictionaryLocator>,
    context_counter: Arc<AtomicU64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TtsMessage {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    try_trigger_generation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    flush: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    close_context: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dictionary_locators: Option<Vec<PronunciationDictionaryLocator>>,
}

#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct TtsResponse {
    audio: Option<String>,
    #[serde(default)]
    contextId: Option<String>,
    #[serde(default)]
    is_final: Option<bool>,
    #[serde(default)]
    isFinal: Option<bool>,
    #[serde(default)]
    normalizedAlignment: Option<serde_json::Value>,
    #[serde(default)]
    alignment: Option<serde_json::Value>,
}

impl ElevenLabsTts {
    pub fn new(api_key: String, voice_id: String) -> Self {
        Self {
            api_key,
            voice_id,
            base_url: "wss://api.elevenlabs.io".to_string(),
            pronunciation_dict: None,
            context_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Generate a unique context ID
    fn next_context_id(&self) -> String {
        let id = self.context_counter.fetch_add(1, Ordering::SeqCst);
        format!("ctx_{}", id)
    }

    /// Set pronunciation dictionary locator
    pub fn with_pronunciation_dict(mut self, dict: PronunciationDictionaryLocator) -> Self {
        self.pronunciation_dict = Some(dict);
        self
    }

    /// Speak text and play with interrupt support
    /// Creates a fresh audio sink each call (old audio stopped by caller via interrupt flag)
    pub async fn speak_and_play_interruptible(
        &self,
        text: &str,
        interrupt_flag: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        use http::header::HeaderValue;

        eprintln!("🔊 TTS: {}", text);

        let url = format!(
            "{}/v1/text-to-speech/{}/multi-stream-input?model_id=eleven_turbo_v2_5&output_format=mp3_44100_128",
            self.base_url, self.voice_id
        );

        let context_id = self.next_context_id();

        // Create fresh audio output for this TTS call
        let (_stream, stream_handle) =
            rodio::OutputStream::try_default().context("Failed to create audio output stream")?;
        let sink = rodio::Sink::try_new(&stream_handle).context("Failed to create audio sink")?;

        // Connect WebSocket
        let mut request = url.into_client_request()?;
        request.headers_mut().insert(
            "xi-api-key",
            HeaderValue::from_str(&self.api_key).context("Invalid API key format")?,
        );

        let (ws_stream, _response) = connect_async(request)
            .await
            .context("Failed to connect to ElevenLabs WebSocket")?;

        let (mut write, mut read) = ws_stream.split();

        // Initialize with pronunciation dictionary
        let pronunciation_dicts = self
            .pronunciation_dict
            .as_ref()
            .map(|dict| vec![dict.clone()]);

        let init_msg = TtsMessage {
            text: " ".to_string(),
            context_id: Some(context_id.clone()),
            try_trigger_generation: None,
            flush: None,
            close_context: None,
            pronunciation_dictionary_locators: pronunciation_dicts,
        };
        write
            .send(Message::Text(serde_json::to_string(&init_msg)?))
            .await?;

        // Send text
        let text_msg = TtsMessage {
            text: text.to_string(),
            context_id: Some(context_id.clone()),
            try_trigger_generation: None,
            flush: None,
            close_context: None,
            pronunciation_dictionary_locators: None,
        };
        write
            .send(Message::Text(serde_json::to_string(&text_msg)?))
            .await?;

        // Flush to trigger generation
        let flush_msg = TtsMessage {
            text: "".to_string(),
            context_id: Some(context_id.clone()),
            try_trigger_generation: None,
            flush: Some(true),
            close_context: None,
            pronunciation_dictionary_locators: None,
        };
        write
            .send(Message::Text(serde_json::to_string(&flush_msg)?))
            .await?;

        // Stream audio chunks to sink
        let mut chunk_count = 0;
        let mut total_bytes = 0;

        loop {
            // Check interrupt before waiting for data
            if let Some(ref interrupt) = interrupt_flag {
                if interrupt.load(Ordering::SeqCst) {
                    eprintln!("🛑 Interrupted during streaming");
                    sink.clear();
                    sink.stop();
                    return Ok(());
                }
            }

            let msg_result = timeout(Duration::from_secs(5), read.next()).await;

            match msg_result {
                Ok(Some(msg_res)) => {
                    let msg = msg_res?;
                    match msg {
                        Message::Text(text) => {
                            let response: TtsResponse = serde_json::from_str(&text)
                                .context(format!("Failed to parse response: {}", text))?;

                            if let Some(audio_base64) = response.audio {
                                chunk_count += 1;
                                let chunk = general_purpose::STANDARD
                                    .decode(&audio_base64)
                                    .context("Failed to decode base64 audio")?;
                                total_bytes += chunk.len();

                                let cursor = Cursor::new(chunk);
                                if let Ok(source) = rodio::Decoder::new(cursor) {
                                    sink.append(source);
                                }
                            }

                            if response.is_final.unwrap_or(false)
                                || response.isFinal.unwrap_or(false)
                            {
                                break;
                            }
                        }
                        Message::Close(_) => break,
                        _ => {}
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    if total_bytes > 0 {
                        break;
                    }
                    anyhow::bail!("Timeout waiting for audio from ElevenLabs");
                }
            }
        }

        eprintln!(
            "✅ TTS streamed {} chunks ({} bytes)",
            chunk_count, total_bytes
        );

        // Wait for playback to complete, checking for interrupts
        if let Some(interrupt) = interrupt_flag {
            while !sink.empty() {
                if interrupt.load(Ordering::SeqCst) {
                    eprintln!("🛑 Interrupted during playback");
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
