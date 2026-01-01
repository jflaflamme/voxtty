// Realtime WebSocket transcription support
// Supports OpenAI, Speaches, and ElevenLabs realtime APIs

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use tungstenite::{connect, Message};
use url::Url;

/// Realtime transcription provider
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealtimeProvider {
    /// OpenAI Realtime API (wss://api.openai.com/v1/realtime)
    OpenAI,
    /// Speaches (OpenAI-compatible, self-hosted)
    Speaches,
    /// ElevenLabs ScribeRealtime v2
    ElevenLabs,
}

/// Configuration for realtime transcription
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    pub provider: RealtimeProvider,
    pub api_key: String,
    /// Base URL (for Speaches self-hosted)
    pub base_url: Option<String>,
    /// Model to use (e.g., "gpt-4o-transcribe", "deepdml/faster-whisper-large-v3-turbo-ct2")
    pub model: Option<String>,
    /// Language hint (e.g., "en")
    pub language: Option<String>,
    /// Sample rate of audio being sent
    pub sample_rate: u32,
    /// Enable debug logging
    pub debug: bool,
    /// Suppress normal output (for TUI mode)
    pub quiet: bool,
}

/// Transcription result from realtime API
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TranscriptionResult {
    pub text: String,
    pub is_final: bool,
}

/// Events sent to the realtime transcription client
#[allow(dead_code)]
pub enum RealtimeEvent {
    /// Audio chunk to transcribe (PCM i16 samples)
    AudioChunk(Vec<i16>),
    /// Commit the current audio buffer
    Commit,
    /// Stop transcription
    Stop,
}

/// Events received from the realtime transcription
pub enum TranscriptionEvent {
    /// Partial transcription (may change)
    Partial(String),
    /// Final committed transcription
    Final(String),
    /// Speech started
    SpeechStarted,
    /// Speech stopped
    SpeechStopped,
    /// Error occurred
    Error(String),
    /// Connection is being established
    Connecting,
    /// Connection established
    Connected,
    /// Connection closed
    Closed,
}

// ============================================================================
// OpenAI/Speaches Realtime Protocol Messages
// ============================================================================

// For OpenAI transcription-only mode (intent=transcription)
#[derive(Serialize)]
struct OpenAITranscriptionSessionUpdate {
    #[serde(rename = "type")]
    msg_type: String,
    session: OpenAITranscriptionSession,
}

#[derive(Serialize)]
struct OpenAITranscriptionSession {
    input_audio_format: String,
    input_audio_transcription: OpenAIInputAudioTranscription,
    turn_detection: OpenAITurnDetection,
}

#[derive(Serialize)]
struct OpenAIInputAudioTranscription {
    model: String,
}

#[derive(Serialize)]
struct OpenAITurnDetection {
    #[serde(rename = "type")]
    detection_type: String,
    threshold: f32,
    silence_duration_ms: u32,
}

// For Speaches (uses older session.update format)
#[derive(Serialize)]
struct SpeachesSessionUpdate {
    #[serde(rename = "type")]
    msg_type: String,
    session: SpeachesSessionConfig,
}

#[derive(Serialize)]
struct SpeachesSessionConfig {
    input_audio_transcription: Option<SpeachesTranscriptionConfig>,
    turn_detection: Option<OpenAITurnDetection>,
}

#[derive(Serialize)]
struct SpeachesTranscriptionConfig {
    model: String,
}

#[derive(Serialize)]
struct OpenAIAudioAppend {
    #[serde(rename = "type")]
    msg_type: String,
    audio: String, // base64 encoded PCM
}

#[derive(Serialize)]
struct OpenAICommit {
    #[serde(rename = "type")]
    msg_type: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponse {
    #[serde(rename = "type")]
    msg_type: String,
    transcript: Option<String>,
    error: Option<OpenAIError>,
}

#[derive(Deserialize, Debug)]
struct OpenAIError {
    message: String,
}

// ============================================================================
// ElevenLabs Realtime Protocol Messages
// ============================================================================

#[derive(Serialize)]
struct ElevenLabsAudioChunk {
    message_type: String,
    audio_base_64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
}

#[derive(Serialize)]
struct ElevenLabsCommit {
    message_type: String,
}

#[derive(Deserialize, Debug)]
struct ElevenLabsResponse {
    message_type: String,
    text: Option<String>,
    error: Option<ElevenLabsError>,
}

#[derive(Deserialize, Debug)]
struct ElevenLabsError {
    message: String,
}

// ============================================================================
// Realtime Transcription Client
// ============================================================================

pub struct RealtimeTranscriber {
    config: RealtimeConfig,
    event_tx: Option<Sender<RealtimeEvent>>,
    transcription_rx: Option<Receiver<TranscriptionEvent>>,
    running: Arc<Mutex<bool>>,
}

impl RealtimeTranscriber {
    pub fn new(config: RealtimeConfig) -> Self {
        Self {
            config,
            event_tx: None,
            transcription_rx: None,
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the realtime transcription connection
    pub fn start(&mut self) -> Result<()> {
        let (event_tx, event_rx) = mpsc::channel::<RealtimeEvent>();
        let (transcription_tx, transcription_rx) = mpsc::channel::<TranscriptionEvent>();

        self.event_tx = Some(event_tx);
        self.transcription_rx = Some(transcription_rx);

        let config = self.config.clone();
        let running = self.running.clone();
        *running.lock() = true;

        thread::spawn(move || {
            if let Err(e) = run_websocket_loop(config.clone(), event_rx, transcription_tx, running)
            {
                if !config.quiet {
                    eprintln!("[Realtime] WebSocket error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Send audio chunk for transcription
    pub fn send_audio(&self, samples: Vec<i16>) -> Result<()> {
        if let Some(tx) = &self.event_tx {
            tx.send(RealtimeEvent::AudioChunk(samples))
                .context("Failed to send audio chunk")?;
        }
        Ok(())
    }

    /// Commit the current audio buffer
    #[allow(dead_code)]
    pub fn commit(&self) -> Result<()> {
        if let Some(tx) = &self.event_tx {
            tx.send(RealtimeEvent::Commit)
                .context("Failed to send commit")?;
        }
        Ok(())
    }

    /// Try to receive a transcription event (non-blocking)
    pub fn try_recv(&self) -> Option<TranscriptionEvent> {
        self.transcription_rx.as_ref()?.try_recv().ok()
    }

    /// Stop the transcription
    pub fn stop(&mut self) {
        *self.running.lock() = false;
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(RealtimeEvent::Stop);
        }
    }

    #[allow(dead_code)]
    pub fn is_running(&self) -> bool {
        *self.running.lock()
    }
}

fn run_websocket_loop(
    config: RealtimeConfig,
    event_rx: Receiver<RealtimeEvent>,
    transcription_tx: Sender<TranscriptionEvent>,
    running: Arc<Mutex<bool>>,
) -> Result<()> {
    let url = build_websocket_url(&config)?;
    let request = build_websocket_request(&config, &url)?;

    if !config.quiet {
        println!("[Realtime] Connecting to {}...", url);
    }
    let _ = transcription_tx.send(TranscriptionEvent::Connecting);

    let (mut socket, _response) = match connect(request) {
        Ok(result) => result,
        Err(e) => {
            if !config.quiet {
                eprintln!("[Realtime] Connection error details: {:?}", e);
            }
            return Err(anyhow::anyhow!("Failed to connect to WebSocket: {}", e));
        }
    };
    if !config.quiet {
        println!("[Realtime] Connected!");
    }
    let _ = transcription_tx.send(TranscriptionEvent::Connected);

    // Send initial configuration
    send_initial_config(&mut socket, &config)?;

    // Set read timeout for non-blocking behavior
    match socket.get_ref() {
        tungstenite::stream::MaybeTlsStream::Plain(ref stream) => {
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(10)));
        }
        tungstenite::stream::MaybeTlsStream::NativeTls(ref tls_stream) => {
            let _ = tls_stream
                .get_ref()
                .set_read_timeout(Some(std::time::Duration::from_millis(10)));
        }
        _ => {
            // Other TLS backends might not support non-blocking
            if config.debug {
                eprintln!("[DEBUG] Warning: Could not set read timeout on TLS stream");
            }
        }
    }

    while *running.lock() {
        // Check for incoming events from audio capture
        match event_rx.try_recv() {
            Ok(RealtimeEvent::AudioChunk(samples)) => {
                if let Err(e) = send_audio_chunk(&mut socket, &config, &samples, config.debug) {
                    if !config.quiet {
                        eprintln!("[Realtime] Failed to send audio: {}", e);
                    }
                }
            }
            Ok(RealtimeEvent::Commit) => {
                if let Err(e) = send_commit(&mut socket, &config) {
                    if !config.quiet {
                        eprintln!("[Realtime] Failed to send commit: {}", e);
                    }
                }
            }
            Ok(RealtimeEvent::Stop) => {
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                break;
            }
        }

        // Check for incoming WebSocket messages (with timeout)
        match socket.read() {
            Ok(Message::Text(text)) => {
                if config.debug {
                    eprintln!("[DEBUG] Received: {}", text);
                }
                if let Some(event) = parse_transcription_message(&config, &text) {
                    // Log important events even in non-debug mode
                    match &event {
                        TranscriptionEvent::SpeechStarted => {
                            if !config.quiet {
                                eprintln!("[Realtime] Speech detected!");
                            }
                        }
                        TranscriptionEvent::Error(msg) => {
                            eprintln!("[Realtime ERROR] {}", msg);
                        }
                        _ => {}
                    }
                    let _ = transcription_tx.send(event);
                } else if config.debug {
                    eprintln!("[DEBUG] Unhandled message type");
                }
            }
            Ok(Message::Close(frame)) => {
                if config.debug {
                    if let Some(f) = &frame {
                        eprintln!(
                            "[DEBUG] WebSocket closed: code={}, reason={}",
                            f.code, f.reason
                        );
                    } else {
                        eprintln!("[DEBUG] WebSocket closed without frame");
                    }
                }
                let _ = transcription_tx.send(TranscriptionEvent::Closed);
                break;
            }
            Ok(_) => {} // Ignore binary, ping, pong
            Err(tungstenite::Error::Io(ref e))
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => {
                let _ = transcription_tx.send(TranscriptionEvent::Error(e.to_string()));
                break;
            }
        }

        // Small sleep to prevent busy loop
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let _ = socket.close(None);
    let _ = transcription_tx.send(TranscriptionEvent::Closed);
    Ok(())
}

fn build_websocket_url(config: &RealtimeConfig) -> Result<Url> {
    match config.provider {
        RealtimeProvider::OpenAI => {
            // For transcription-only mode, don't specify model in URL
            let mut url = Url::parse("wss://api.openai.com/v1/realtime")?;
            url.query_pairs_mut().append_pair("intent", "transcription");
            Ok(url)
        }
        RealtimeProvider::Speaches => {
            let base = config.base_url.as_deref().unwrap_or("ws://localhost:8000");
            let base = base
                .replace("http://", "ws://")
                .replace("https://", "wss://");
            let mut url = Url::parse(&format!("{}/v1/realtime", base))?;
            url.query_pairs_mut().append_pair("intent", "transcription");
            if let Some(model) = &config.model {
                url.query_pairs_mut().append_pair("model", model);
            } else {
                url.query_pairs_mut()
                    .append_pair("model", "Systran/faster-distil-whisper-small.en");
            }
            if !config.api_key.is_empty() {
                url.query_pairs_mut()
                    .append_pair("api_key", &config.api_key);
            }
            Ok(url)
        }
        RealtimeProvider::ElevenLabs => {
            let mut url = Url::parse("wss://api.elevenlabs.io/v1/speech-to-text/realtime")?;
            // Use VAD for automatic speech detection
            // Increased thresholds to allow for natural pauses while speaking
            url.query_pairs_mut()
                .append_pair("commit_strategy", "vad")
                .append_pair("vad_threshold", "0.3") // Lowered from 0.5 for better sensitivity
                .append_pair("vad_silence_threshold_secs", "1.2") // Reduced from 1.8
                .append_pair("min_silence_duration_ms", "1200"); // Reduced from 1800
                                                                 // Audio format
            let format = format!("pcm_{}", config.sample_rate);
            url.query_pairs_mut().append_pair("audio_format", &format);
            // Language
            if let Some(lang) = &config.language {
                url.query_pairs_mut().append_pair("language_code", lang);
            }
            Ok(url)
        }
    }
}

fn build_websocket_request(
    config: &RealtimeConfig,
    url: &Url,
) -> Result<tungstenite::http::Request<()>> {
    // Generate WebSocket key
    let key = tungstenite::handshake::client::generate_key();

    let mut request = tungstenite::http::Request::builder()
        .method("GET")
        .uri(url.as_str())
        .header("Host", url.host_str().unwrap_or(""))
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", key);

    match config.provider {
        RealtimeProvider::OpenAI => {
            request = request
                .header("Authorization", format!("Bearer {}", config.api_key))
                .header("OpenAI-Beta", "realtime=v1");
        }
        RealtimeProvider::Speaches => {
            if !config.api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", config.api_key));
            }
        }
        RealtimeProvider::ElevenLabs => {
            request = request.header("xi-api-key", &config.api_key);
        }
    }

    Ok(request.body(())?)
}

fn send_initial_config(
    socket: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
    config: &RealtimeConfig,
) -> Result<()> {
    match config.provider {
        RealtimeProvider::OpenAI => {
            // OpenAI transcription-only mode uses transcription_session.update
            let session_update = OpenAITranscriptionSessionUpdate {
                msg_type: "transcription_session.update".to_string(),
                session: OpenAITranscriptionSession {
                    input_audio_format: "pcm16".to_string(),
                    input_audio_transcription: OpenAIInputAudioTranscription {
                        model: config
                            .model
                            .clone()
                            .unwrap_or_else(|| "whisper-1".to_string()),
                    },
                    turn_detection: OpenAITurnDetection {
                        detection_type: "server_vad".to_string(),
                        threshold: 0.3, // Lowered from 0.5 for better sensitivity
                        silence_duration_ms: 1200, // Reduced from 1800ms for faster response
                    },
                },
            };
            let msg = serde_json::to_string(&session_update)?;
            socket.send(Message::Text(msg))?;
        }
        RealtimeProvider::Speaches => {
            // Speaches uses the older session.update format (OpenAI-compatible)
            let session_update = SpeachesSessionUpdate {
                msg_type: "session.update".to_string(),
                session: SpeachesSessionConfig {
                    input_audio_transcription: Some(SpeachesTranscriptionConfig {
                        model: config
                            .model
                            .clone()
                            .unwrap_or_else(|| "whisper-1".to_string()),
                    }),
                    turn_detection: Some(OpenAITurnDetection {
                        detection_type: "server_vad".to_string(),
                        threshold: 0.3, // Lowered from 0.5 for better sensitivity
                        silence_duration_ms: 800, // Reduced from 1000ms for faster response
                    }),
                },
            };
            let msg = serde_json::to_string(&session_update)?;
            socket.send(Message::Text(msg))?;
        }
        RealtimeProvider::ElevenLabs => {
            // ElevenLabs doesn't need initial config - it's in the URL params
        }
    }
    Ok(())
}

fn send_audio_chunk(
    socket: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
    config: &RealtimeConfig,
    samples: &[i16],
    debug: bool,
) -> Result<()> {
    // Convert i16 samples to bytes (little-endian)
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let audio_b64 = BASE64.encode(&bytes);

    if debug {
        // Calculate audio level for debug
        let max_sample = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        let level_db = if max_sample > 0 {
            20.0 * (max_sample as f32 / 32768.0).log10()
        } else {
            -100.0
        };
        eprintln!(
            "[DEBUG] Sending {} samples ({} bytes b64), level: {:.1} dB, max: {}",
            samples.len(),
            audio_b64.len(),
            level_db,
            max_sample
        );
    }

    let msg = match config.provider {
        RealtimeProvider::OpenAI | RealtimeProvider::Speaches => {
            let append = OpenAIAudioAppend {
                msg_type: "input_audio_buffer.append".to_string(),
                audio: audio_b64,
            };
            serde_json::to_string(&append)?
        }
        RealtimeProvider::ElevenLabs => {
            let chunk = ElevenLabsAudioChunk {
                message_type: "input_audio_chunk".to_string(),
                audio_base_64: audio_b64,
                sample_rate: Some(config.sample_rate),
            };
            serde_json::to_string(&chunk)?
        }
    };

    socket.send(Message::Text(msg))?;
    Ok(())
}

fn send_commit(
    socket: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
    config: &RealtimeConfig,
) -> Result<()> {
    let msg = match config.provider {
        RealtimeProvider::OpenAI | RealtimeProvider::Speaches => {
            let commit = OpenAICommit {
                msg_type: "input_audio_buffer.commit".to_string(),
            };
            serde_json::to_string(&commit)?
        }
        RealtimeProvider::ElevenLabs => {
            let commit = ElevenLabsCommit {
                message_type: "commit".to_string(),
            };
            serde_json::to_string(&commit)?
        }
    };

    socket.send(Message::Text(msg))?;
    Ok(())
}

fn parse_transcription_message(config: &RealtimeConfig, text: &str) -> Option<TranscriptionEvent> {
    match config.provider {
        RealtimeProvider::OpenAI | RealtimeProvider::Speaches => {
            let response: OpenAIResponse = serde_json::from_str(text).ok()?;

            if config.debug {
                eprintln!("[Realtime] Received event type: {}", response.msg_type);
            }

            match response.msg_type.as_str() {
                // OpenAI transcription-only API events
                "transcription.delta" => {
                    // Partial/incremental transcription in transcription-only mode
                    response
                        .transcript
                        .filter(|t| !t.trim().is_empty())
                        .map(TranscriptionEvent::Partial)
                }
                "transcription.done" => response.transcript.map(TranscriptionEvent::Final),
                "conversation.item.input_audio_transcription.delta" => {
                    // Partial/incremental transcription (real-time updates as you speak)
                    response
                        .transcript
                        .filter(|t| !t.trim().is_empty())
                        .map(TranscriptionEvent::Partial)
                }
                "conversation.item.input_audio_transcription.completed" => {
                    // Legacy conversation API format (for Speaches compatibility)
                    response.transcript.map(TranscriptionEvent::Final)
                }
                "input_audio_buffer.speech_started" => Some(TranscriptionEvent::SpeechStarted),
                "input_audio_buffer.speech_stopped" => Some(TranscriptionEvent::SpeechStopped),
                "error" => response.error.map(|e| TranscriptionEvent::Error(e.message)),
                _ => None,
            }
        }
        RealtimeProvider::ElevenLabs => {
            let response: ElevenLabsResponse = serde_json::from_str(text).ok()?;

            if config.debug {
                eprintln!("[Realtime] Received event type: {}", response.message_type);
            }

            match response.message_type.as_str() {
                "partial_transcript" => response
                    .text
                    .filter(|t| !t.trim().is_empty() && t.trim() != "(silence)")
                    .map(TranscriptionEvent::Partial),
                "committed_transcript" | "committed_transcript_with_timestamps" => response
                    .text
                    .filter(|t| !t.trim().is_empty() && t.trim() != "(silence)")
                    .map(TranscriptionEvent::Final),
                "auth_error" | "quota_exceeded" | "quota_exceeded_error" | "rate_limited_error" => {
                    response
                        .error
                        .map(|e| TranscriptionEvent::Error(e.message))
                        .or(Some(TranscriptionEvent::Error(response.message_type)))
                }
                _ => None,
            }
        }
    }
}
