// ElevenLabs TTS WebSocket client
use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tungstenite::client::IntoClientRequest;

/// Pronunciation dictionary locator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationDictionaryLocator {
    pub pronunciation_dictionary_id: String,
    pub version_id: String,
}

/// Pronunciation dictionary rule (alias type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasRule {
    pub string_to_replace: String,
    #[serde(rename = "type")]
    pub rule_type: String, // "alias"
    pub alias: String,
}

/// Pronunciation dictionary rule (phoneme type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeRule {
    pub string_to_replace: String,
    #[serde(rename = "type")]
    pub rule_type: String, // "phoneme"
    pub phoneme: String,
    pub alphabet: String, // "ipa" or "cmu"
}

/// ElevenLabs TTS client using WebSocket
pub struct ElevenLabsTts {
    api_key: String,
    voice_id: String,
    base_url: String,
    pronunciation_dict: Option<PronunciationDictionaryLocator>,
    context_counter: std::sync::Arc<std::sync::atomic::AtomicU64>, // For generating unique context IDs
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
            context_counter: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Generate a unique context ID
    fn next_context_id(&self) -> String {
        let id = self
            .context_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("ctx_{}", id)
    }

    /// Set pronunciation dictionary locator
    pub fn with_pronunciation_dict(mut self, dict: PronunciationDictionaryLocator) -> Self {
        self.pronunciation_dict = Some(dict);
        self
    }

    /// Generate speech from text and return audio data (MP3)
    #[allow(dead_code)]
    pub async fn speak(&self, text: &str) -> Result<Vec<u8>> {
        use http::header::HeaderValue;

        // Use multi-stream-input endpoint
        let url = format!(
            "{}/v1/text-to-speech/{}/multi-stream-input?model_id=eleven_turbo_v2_5&output_format=mp3_44100_128",
            self.base_url, self.voice_id
        );

        // Generate context ID
        let context_id = self.next_context_id();

        eprintln!("🔌 Connecting to: {}", url);

        // Create WebSocket request with API key header
        let mut request = url.into_client_request()?;
        request.headers_mut().insert(
            "xi-api-key",
            HeaderValue::from_str(&self.api_key).context("Invalid API key format")?,
        );

        eprintln!(
            "🔑 API Key: {}***",
            &self.api_key[..8.min(self.api_key.len())]
        );

        let (ws_stream, response) = connect_async(request)
            .await
            .context("Failed to connect to ElevenLabs WebSocket")?;

        eprintln!("✅ WebSocket connected! Status: {:?}", response.status());

        let (mut write, mut read) = ws_stream.split();

        // Send authentication header (if needed as first message)
        // Note: Some implementations need xi-api-key in the connection header
        // This might need adjustment based on actual API behavior

        // 1. Initialize connection with a space
        // Note: pronunciation_dictionary_locators MUST be sent in the FIRST message only
        eprintln!("📤 Sending init message...");
        let pronunciation_dicts = self
            .pronunciation_dict
            .as_ref()
            .map(|dict| vec![dict.clone()]);

        if let Some(ref dicts) = pronunciation_dicts {
            eprintln!("📖 Using pronunciation dictionary: {:?}", dicts);
        }

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
            .await
            .context("Failed to send init message")?;

        // 2. Send the actual text (no pronunciation_dictionary_locators after first message)
        eprintln!("📤 Sending text: {}", text);
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
            .await
            .context("Failed to send text message")?;

        // 3. Flush to trigger generation
        eprintln!("📤 Sending flush message...");
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
            .await
            .context("Failed to send flush message")?;

        eprintln!("⏳ Waiting for audio chunks...");

        // Collect audio chunks
        let mut audio_data = Vec::new();
        let mut chunk_count = 0;

        loop {
            // Wait for next message with 5s timeout to prevent hanging if isFinal is missing
            let msg_result = timeout(Duration::from_secs(5), read.next()).await;

            match msg_result {
                Ok(Some(msg_res)) => {
                    let msg = msg_res?;
                    match msg {
                        Message::Text(text) => {
                            eprintln!("📥 Received message: {}", &text[..100.min(text.len())]);

                            let response: TtsResponse = serde_json::from_str(&text)
                                .context(format!("Failed to parse response: {}", text))?;

                            // Verify context ID matches what we sent
                            if let Some(ref ctx_id) = response.contextId {
                                if ctx_id != &context_id {
                                    eprintln!(
                                        "[WARNING] Context ID mismatch: expected '{}', got '{}'",
                                        context_id, ctx_id
                                    );
                                }
                            }

                            if let Some(audio_base64) = response.audio {
                                chunk_count += 1;
                                // Decode base64 audio and append
                                let chunk = general_purpose::STANDARD
                                    .decode(&audio_base64)
                                    .context("Failed to decode base64 audio")?;
                                eprintln!("🎵 Chunk {}: {} bytes", chunk_count, chunk.len());
                                audio_data.extend_from_slice(&chunk);
                            }

                            // Check termination signals:
                            // Only break on explicit is_final/isFinal flag
                            // Don't break on alignment alone as more audio chunks may follow
                            if response.is_final.unwrap_or(false)
                                || response.isFinal.unwrap_or(false)
                            {
                                eprintln!("🏁 Received final signal (is_final=true)");
                                break;
                            }

                            // Log alignment data for debugging but don't break
                            if response.normalizedAlignment.is_some()
                                || response.alignment.is_some()
                            {
                                eprintln!("📊 Received alignment data (continuing to wait for more chunks)");
                            }
                        }
                        Message::Close(frame) => {
                            eprintln!("🔌 WebSocket closed: {:?}", frame);
                            break;
                        }
                        other => {
                            eprintln!("📨 Other message type: {:?}", other);
                        }
                    }
                }
                Ok(None) => {
                    eprintln!("🔌 Stream ended");
                    break;
                }
                Err(_) => {
                    eprintln!("⚠️ Timeout waiting for data (5s)");
                    if !audio_data.is_empty() {
                        eprintln!(
                            "✅ Assuming stream finished due to timeout with {} bytes",
                            audio_data.len()
                        );
                        break;
                    }
                    anyhow::bail!("Timeout waiting for audio from ElevenLabs");
                }
            }
        }

        eprintln!(
            "✅ Collected {} chunks, total {} bytes",
            chunk_count,
            audio_data.len()
        );

        Ok(audio_data)
    }

    /// Speak text and play it immediately with real-time streaming
    /// Returns early if interrupted (returns Ok with no error)
    pub async fn speak_and_play(&self, text: &str) -> Result<()> {
        self.speak_and_play_interruptible(text, None).await
    }

    /// Speak text and play with interrupt support
    /// interrupt_flag: When set to true, playback will stop immediately
    pub async fn speak_and_play_interruptible(
        &self,
        text: &str,
        interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<()> {
        use http::header::HeaderValue;

        eprintln!(
            "🔊 ElevenLabs: Real-time streaming with multi-context for: {}",
            text
        );

        // Use multi-stream-input endpoint for context support
        let url = format!(
            "{}/v1/text-to-speech/{}/multi-stream-input?model_id=eleven_turbo_v2_5&output_format=mp3_44100_128",
            self.base_url, self.voice_id
        );

        // Generate unique context ID for this speech
        let context_id = self.next_context_id();
        eprintln!("[MULTI-CONTEXT] Using context ID: {}", context_id);

        eprintln!("🔌 Connecting to: {}", url);

        // Create WebSocket request with API key header
        let mut request = url.into_client_request()?;
        request.headers_mut().insert(
            "xi-api-key",
            HeaderValue::from_str(&self.api_key).context("Invalid API key format")?,
        );

        let (ws_stream, response) = connect_async(request)
            .await
            .context("Failed to connect to ElevenLabs WebSocket")?;

        eprintln!("✅ WebSocket connected! Status: {:?}", response.status());

        let (mut write, mut read) = ws_stream.split();

        // Initialize connection with pronunciation dictionary and context_id
        eprintln!("📤 Sending init message...");
        let pronunciation_dicts = self
            .pronunciation_dict
            .as_ref()
            .map(|dict| vec![dict.clone()]);

        if let Some(ref dicts) = pronunciation_dicts {
            eprintln!("📖 Using pronunciation dictionary: {:?}", dicts);
        }

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
            .await
            .context("Failed to send init message")?;

        // Send the actual text with context_id
        eprintln!("📤 Sending text: {}", text);
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
            .await
            .context("Failed to send text message")?;

        // Flush to trigger generation with context_id
        eprintln!("📤 Sending flush message...");
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
            .await
            .context("Failed to send flush message")?;

        // Create audio output stream and sink for real-time playback
        eprintln!("🔊 Initializing audio output...");
        let (_stream, stream_handle) =
            rodio::OutputStream::try_default().context("Failed to create audio output stream")?;
        let sink = rodio::Sink::try_new(&stream_handle).context("Failed to create audio sink")?;

        eprintln!("⏳ Streaming audio chunks in real-time...");

        let mut chunk_count = 0;
        let mut total_bytes = 0;

        loop {
            // Wait for next message with 5s timeout
            let msg_result = timeout(Duration::from_secs(5), read.next()).await;

            match msg_result {
                Ok(Some(msg_res)) => {
                    let msg = msg_res?;
                    match msg {
                        Message::Text(text) => {
                            let response: TtsResponse = serde_json::from_str(&text)
                                .context(format!("Failed to parse response: {}", text))?;

                            // Verify context ID matches what we sent
                            if let Some(ref ctx_id) = response.contextId {
                                if ctx_id != &context_id {
                                    eprintln!("[MULTI-CONTEXT WARNING] Context ID mismatch: expected '{}', got '{}'",
                                        context_id, ctx_id);
                                }
                            }

                            if let Some(audio_base64) = response.audio {
                                chunk_count += 1;
                                // Decode base64 audio
                                let chunk = general_purpose::STANDARD
                                    .decode(&audio_base64)
                                    .context("Failed to decode base64 audio")?;
                                total_bytes += chunk.len();

                                eprintln!(
                                    "🎵 Playing chunk {}: {} bytes",
                                    chunk_count,
                                    chunk.len()
                                );

                                // Stream chunk to audio sink immediately
                                let cursor = Cursor::new(chunk);
                                match rodio::Decoder::new(cursor) {
                                    Ok(source) => {
                                        sink.append(source);
                                    }
                                    Err(e) => {
                                        eprintln!("⚠️ Failed to decode audio chunk: {}", e);
                                        // Continue to next chunk instead of failing
                                    }
                                }
                            }

                            // Check for completion
                            if response.is_final.unwrap_or(false)
                                || response.isFinal.unwrap_or(false)
                            {
                                eprintln!("🏁 Received final signal");
                                break;
                            }

                            if response.normalizedAlignment.is_some()
                                || response.alignment.is_some()
                            {
                                eprintln!("📊 Alignment data received");
                            }
                        }
                        Message::Close(frame) => {
                            eprintln!("🔌 WebSocket closed: {:?}", frame);
                            break;
                        }
                        other => {
                            eprintln!("📨 Other message type: {:?}", other);
                        }
                    }
                }
                Ok(None) => {
                    eprintln!("🔌 Stream ended");
                    break;
                }
                Err(_) => {
                    eprintln!("⚠️ Timeout waiting for data (5s)");
                    if total_bytes > 0 {
                        eprintln!(
                            "✅ Stream finished with {} chunks ({} bytes total)",
                            chunk_count, total_bytes
                        );
                        break;
                    }
                    anyhow::bail!("Timeout waiting for audio from ElevenLabs");
                }
            }
        }

        // Wait for all queued audio to finish playing, but check for interrupts
        eprintln!("⏳ Waiting for playback to complete...");

        if let Some(interrupt) = interrupt_flag {
            // Poll for interrupt while playing
            use std::sync::atomic::Ordering;
            eprintln!(
                "[MULTI-CONTEXT] Starting interrupt polling for context '{}'...",
                context_id
            );
            let mut check_count = 0;
            let context_id_clone = context_id.clone();

            while !sink.empty() {
                check_count += 1;
                let interrupt_value = interrupt.load(Ordering::SeqCst);
                if check_count % 10 == 0 {
                    // Log every 10 checks (500ms)
                    eprintln!(
                        "[MULTI-CONTEXT] Check #{}: interrupt={}, sink.empty()={}",
                        check_count,
                        interrupt_value,
                        sink.empty()
                    );
                }
                if interrupt_value {
                    eprintln!(
                        "🛑 User interrupted - stopping context '{}' immediately!",
                        context_id_clone
                    );

                    // Immediately clear and stop local playback
                    // Note: We don't send close_context to server here because write is borrowed by the receive loop
                    // The context will timeout automatically on the server after 20 seconds
                    sink.clear(); // Clear buffered audio immediately
                    sink.stop(); // Stop playback
                    eprintln!(
                        "[MULTI-CONTEXT] Local audio cleared and stopped for context '{}'",
                        context_id_clone
                    );
                    return Ok(()); // Return success but stop playing
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            eprintln!(
                "[MULTI-CONTEXT] Context '{}' playback loop ended naturally",
                context_id
            );
            // Playback finished naturally
            sink.sleep_until_end();
        } else {
            eprintln!(
                "[MULTI-CONTEXT] No interrupt support for context '{}', just waiting...",
                context_id
            );
            // No interrupt support, just wait
            sink.sleep_until_end();
        }

        eprintln!(
            "✅ Real-time playback completed ({} chunks, {} bytes)",
            chunk_count, total_bytes
        );

        Ok(())
    }
}

/// Play audio file using system audio player
#[allow(dead_code)]
fn play_audio(file_path: &std::path::Path) -> Result<()> {
    // Try different audio players in order of preference
    let players = vec![
        ("mpv", vec!["--no-video", "--really-quiet"]),
        ("ffplay", vec!["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("aplay", vec![]),
        ("paplay", vec![]),
    ];

    for (player, args) in players {
        if let Ok(mut child) = std::process::Command::new(player)
            .args(&args)
            .arg(file_path)
            .spawn()
        {
            match child.wait() {
                Ok(status) => {
                    if status.success() {
                        eprintln!("✅ Audio player '{}' completed successfully", player);
                        return Ok(());
                    } else {
                        eprintln!("⚠️ Audio player '{}' exited with: {}", player, status);
                        // Try next player
                        continue;
                    }
                }
                Err(e) => {
                    eprintln!("⚠️ Audio player '{}' wait error: {}", player, e);
                    // Try next player
                    continue;
                }
            }
        }
    }

    anyhow::bail!(
        "No audio player found or all players failed. Install mpv, ffplay, aplay, or paplay"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_elevenlabs_connection() {
        // This test requires ELEVENLABS_API_KEY environment variable
        if let Ok(api_key) = std::env::var("ELEVENLABS_API_KEY") {
            let voice_id = "21m00Tcm4TlvDq8ikWAM"; // Rachel voice (default)
            let tts = ElevenLabsTts::new(api_key, voice_id.to_string());

            let result = tts.speak("Hello, this is a test.").await;
            assert!(result.is_ok());
        }
    }
}
