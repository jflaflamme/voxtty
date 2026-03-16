// voxtty - Voice assistant that listens on Linux
// Copyright (C) 2025 Jean-Francois Laflamme
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// Repository: https://github.com/jflaflamme/voxtty

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use dialoguer::Select;
use hound::WavWriter;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use webrtc_vad::{Vad, VadMode};

// Platform-specific tray imports
#[cfg(target_os = "linux")]
use ksni::{Tray, TrayService};

mod app_state;
mod controls;
mod conversation;
mod elevenlabs_tts;
mod mcp_tools;
mod model_selector;
mod models_cache;
mod modes;
mod processors;
mod processors_assistant;
mod processors_conversation;
mod processors_transcription;
mod realtime;
mod sounds;
mod tui;

use app_state::{print_mode_change, sync_state};
use model_selector::*;
use modes::*;
use processors::*;
use processors_assistant::*;
use processors_transcription::*;
use realtime::{RealtimeConfig, RealtimeProvider, RealtimeTranscriber, TranscriptionEvent};

use tui::ConnectionStatus;

const WHISPER_URL: &str = "http://127.0.0.1:7777/inference";
const SPEACHES_DEFAULT_URL: &str = "http://localhost:8000/v1/audio/transcriptions";
const SPEACHES_DEFAULT_MODEL: &str = "Systran/faster-distil-whisper-small.en";
const YDOTOOL_FALLBACK_SOCKET: &str = "/tmp/.ydotool_socket";
const VAD_FRAME_MS: usize = 30;
const SILENCE_DURATION_MS: u64 = 1000;
const MIN_SPEECH_DURATION_MS: u64 = 200;
const AMPLITUDE_THRESHOLD: i16 = 1000;

/// Helper to get the preferred audio host
fn get_audio_host() -> cpal::Host {
    cpal::default_host()
}

/// Backend type for transcription
#[derive(Clone, Copy, PartialEq, Debug)]
enum Backend {
    WhisperCpp,
    Speaches,
    OpenAI,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Config {
    #[serde(default = "default_ydotool_socket")]
    ydotool_socket: String,

    #[serde(default = "default_audio_device")]
    audio_device: String,

    #[serde(default = "default_backend")]
    backend: String,

    #[serde(default = "default_speaches_url")]
    speaches_base_url: String,

    #[serde(default = "default_speaches_model")]
    transcription_model_id: String,

    #[serde(default = "default_whisper_url")]
    whisper_url: String,

    // OpenAI Whisper configuration
    #[serde(default = "default_openai_api_key")]
    openai_api_key: String,

    #[serde(default = "default_transcription_url")]
    transcription_url: String,

    // ElevenLabs configuration
    #[serde(default = "default_elevenlabs_api_key")]
    elevenlabs_api_key: String,

    #[serde(default = "default_elevenlabs_voice_id")]
    elevenlabs_voice_id: String,

    #[serde(default)]
    elevenlabs_pronunciation_dict_id: Option<String>,

    #[serde(default)]
    elevenlabs_pronunciation_dict_version: Option<String>,

    // Assistant mode configuration
    #[serde(default = "default_chat_completion_base_url")]
    chat_completion_base_url: String,

    #[serde(default = "default_chat_completion_api_key")]
    chat_completion_api_key: String,

    #[serde(default = "default_llm_model")]
    llm_model: String,

    #[serde(default = "default_system_prompt")]
    system_prompt: String,

    #[serde(default = "default_code_system_prompt")]
    code_system_prompt: String,
}

fn default_ydotool_socket() -> String {
    // Try user runtime directory first, fallback to /tmp
    let uid = unsafe { libc::getuid() };
    let user_socket = format!("/run/user/{}/.ydotool_socket", uid);
    if std::path::Path::new(&user_socket).exists() {
        user_socket
    } else if std::path::Path::new(YDOTOOL_FALLBACK_SOCKET).exists() {
        YDOTOOL_FALLBACK_SOCKET.to_string()
    } else {
        user_socket // Return user socket as default even if it doesn't exist yet
    }
}

fn default_audio_device() -> String {
    "default".to_string()
}

fn default_backend() -> String {
    "whisper.cpp".to_string()
}

fn default_speaches_url() -> String {
    SPEACHES_DEFAULT_URL.to_string()
}

fn default_speaches_model() -> String {
    SPEACHES_DEFAULT_MODEL.to_string()
}

fn default_whisper_url() -> String {
    WHISPER_URL.to_string()
}

fn default_openai_api_key() -> String {
    String::new() // Empty by default, must be set via OPENAI_API_KEY env var
}

fn default_elevenlabs_api_key() -> String {
    String::new() // Empty by default, must be set via ELEVENLABS_API_KEY env var
}

fn default_elevenlabs_voice_id() -> String {
    "21m00Tcm4TlvDq8ikWAM".to_string() // Rachel (public ElevenLabs voice)
}

fn default_chat_completion_base_url() -> String {
    // Default to Ollama (local) - no API key needed
    "http://localhost:11434/v1".to_string()
}

fn default_chat_completion_api_key() -> String {
    String::new() // Empty by default, set via ANTHROPIC_API_KEY or OPENAI_API_KEY for cloud
}

fn default_transcription_url() -> String {
    "https://api.openai.com/v1/audio/transcriptions".to_string()
}

fn default_llm_model() -> String {
    // Default to a common Ollama model
    "llama3.2".to_string()
}

fn default_system_prompt() -> String {
    include_str!("../prompts/assistant.md").to_string()
}

fn default_code_system_prompt() -> String {
    include_str!("../prompts/code.md").to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ydotool_socket: default_ydotool_socket(),
            audio_device: default_audio_device(),
            backend: default_backend(),
            speaches_base_url: default_speaches_url(),
            transcription_model_id: default_speaches_model(),
            whisper_url: default_whisper_url(),
            openai_api_key: default_openai_api_key(),
            transcription_url: default_transcription_url(),
            elevenlabs_api_key: default_elevenlabs_api_key(),
            elevenlabs_voice_id: default_elevenlabs_voice_id(),
            elevenlabs_pronunciation_dict_id: None,
            elevenlabs_pronunciation_dict_version: None,
            chat_completion_base_url: default_chat_completion_base_url(),
            chat_completion_api_key: default_chat_completion_api_key(),
            llm_model: default_llm_model(),
            system_prompt: default_system_prompt(),
            code_system_prompt: default_code_system_prompt(),
        }
    }
}

impl Config {
    fn load() -> Result<Self> {
        // Priority: CLI args > env vars > config file > defaults
        let mut config = match Self::load_from_file() {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("⚠️  Failed to load config file: {}", e);
                eprintln!("   Using default configuration");
                if let Ok(path) = Self::config_path() {
                    eprintln!("   Config file: {}", path.display());
                }
                Self::default()
            }
        };

        // Override with environment variables
        if let Ok(socket) = std::env::var("YDOTOOL_SOCKET") {
            config.ydotool_socket = socket;
        }
        if let Ok(device) = std::env::var("VOICETYPR_AUDIO_DEVICE") {
            config.audio_device = device;
        }
        if let Ok(backend) = std::env::var("VOICETYPR_BACKEND") {
            config.backend = backend;
        }
        if let Ok(url) = std::env::var("SPEACHES_BASE_URL") {
            config.speaches_base_url = url;
        }
        if let Ok(model) = std::env::var("TRANSCRIPTION_MODEL_ID") {
            config.transcription_model_id = model;
        }
        if let Ok(url) = std::env::var("WHISPER_URL") {
            config.whisper_url = url;
        }

        // OpenAI Whisper API key (for --openai transcription backend)
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config.openai_api_key = key;
        }
        if let Ok(url) = std::env::var("TRANSCRIPTION_URL") {
            config.transcription_url = url;
        }

        // ElevenLabs API key (for --elevenlabs transcription backend)
        if let Ok(key) = std::env::var("ELEVENLABS_API_KEY") {
            config.elevenlabs_api_key = key;
        }

        // Assistant mode environment variables
        if let Ok(url) = std::env::var("CHAT_COMPLETION_BASE_URL") {
            config.chat_completion_base_url = url;
        }
        // CHAT_COMPLETION_API_KEY always wins as explicit override.
        // ANTHROPIC_API_KEY only applies when --llm anthropic is set
        // (avoids overwriting Groq/OpenAI-compatible keys from config file).
        if let Ok(key) = std::env::var("CHAT_COMPLETION_API_KEY") {
            config.chat_completion_api_key = key;
        }
        if let Ok(model) = std::env::var("LLM_MODEL") {
            config.llm_model = model;
        }
        if let Ok(prompt) = std::env::var("SYSTEM_PROMPT") {
            config.system_prompt = prompt;
        }
        if let Ok(prompt) = std::env::var("CODE_SYSTEM_PROMPT") {
            config.code_system_prompt = prompt;
        }

        Ok(config)
    }

    fn load_from_file() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_path).context("Failed to read config file")?;

        // Preprocess to handle duplicate keys by keeping only the last occurrence
        let deduplicated = Self::deduplicate_toml_keys(&contents);

        let config: Config =
            toml::from_str(&deduplicated).context("Failed to parse config file")?;

        Ok(config)
    }

    /// Remove duplicate keys from TOML, keeping only the last occurrence
    fn deduplicate_toml_keys(content: &str) -> String {
        use std::collections::HashMap;

        let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

        // Identify all key occurrences
        let mut keys_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(eq_pos) = trimmed.find('=') {
                let key = trimmed[..eq_pos].trim().to_string();
                if !key.is_empty() {
                    keys_to_indices.entry(key).or_default().push(idx);
                }
            }
        }

        // Mark duplicates (all but last occurrence)
        let mut skip_indices = std::collections::HashSet::new();
        for (_, indices) in keys_to_indices.iter() {
            if indices.len() > 1 {
                // Skip all but the last occurrence
                for &idx in &indices[..indices.len() - 1] {
                    skip_indices.insert(idx);
                }
            }
        }

        // Build result, skipping duplicate lines
        lines
            .iter()
            .enumerate()
            .filter(|(idx, _)| !skip_indices.contains(idx))
            .map(|(_, line)| line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Could not determine config directory")?
            .join("voxtty");

        fs::create_dir_all(&config_dir)?;
        Ok(config_dir.join("config.toml"))
    }

    #[allow(dead_code)]
    fn save_example() -> Result<()> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            return Ok(()); // Don't overwrite existing config
        }

        let example = Self::default();
        let toml_string = toml::to_string_pretty(&example)?;

        let with_comments = format!(
            "# VoiceTypr Configuration File\n\
             # Location: {}\n\
             #\n\
             # Priority: CLI flags > Environment variables > This file > Built-in defaults\n\
             #\n\
             # ydotool socket path\n\
             # Default: /run/user/1000/.ydotool_socket (or /tmp/.ydotool_socket as fallback)\n\
             {}\n\
             #\n\
             # Audio Input Device\n\
             # Default: \"default\" (uses system default)\n\
             # Can be overridden with --device flag or VOICETYPR_AUDIO_DEVICE env var\n\
             # audio_device = \"default\"\n\
             #\n\
             # Backend selection: \"whisper.cpp\" or \"speaches\"\n\
             # Default: whisper.cpp\n\
             # Can be overridden with --speaches flag or VOICETYPR_BACKEND env var\n\
             # backend = \"whisper.cpp\"\n\
             #\n\
             # Speaches backend configuration (used when backend = \"speaches\" or --speaches flag)\n\
             # speaches_base_url = \"http://localhost:8000/v1/audio/transcriptions\"\n\
             # transcription_model_id = \"Systran/faster-distil-whisper-small.en\"\n\
             #\n\
             # whisper.cpp backend configuration (used when backend = \"whisper.cpp\")\n\
             # whisper_url = \"http://127.0.0.1:7777/inference\"\n\
             #\n\
             # Assistant mode configuration (used with --assistant flag)\n\
             # Default: OpenAI API (https://api.openai.com/v1)\n\
             # Override with CHAT_COMPLETION_BASE_URL env var for Ollama or other providers\n\
             # chat_completion_base_url = \"https://api.openai.com/v1\"\n\
             #\n\
             # API Key for chat completions (required for OpenAI, not needed for Ollama)\n\
             # Set via CHAT_COMPLETION_API_KEY or OPENAI_API_KEY env var\n\
             # chat_completion_api_key = \"\"\n\
             #\n\
             # LLM model for assistant mode\n\
             # Default: gpt-4o-mini (for OpenAI)\n\
             # For Ollama: llama3.2, mistral, etc.\n\
             # llm_model = \"gpt-4o-mini\"\n\
             #\n\
             # Transcription URL for assistant mode audio transcription\n\
             # Default: https://api.openai.com/v1/audio/transcriptions (OpenAI Whisper)\n\
             # Auto-uses Speaches URL when --speaches flag is set\n\
             # transcription_url = \"https://api.openai.com/v1/audio/transcriptions\"\n\
             #\n\
             # System prompts for assistant modes\n\
             # IMPORTANT: Responses are typed directly as keyboard input!\n\
             # The LLM should output only the text to be typed, without markdown or explanations.\n\
             #\n\
             # Default assistant prompt:\n\
             # system_prompt = \"You are a helpful writing assistant. Your responses will be typed directly as keyboard input into the user's active application. Be concise, clear, and output only the text that should be typed. Do not include explanations, markdown formatting, or meta-commentary unless specifically requested.\"\n\
             #\n\
             # Default code mode prompt:\n\
             # code_system_prompt = \"You are a code generation assistant. Your responses will be typed directly as keyboard input into the user's code editor. Generate clean, working code with appropriate comments. Output only the code that should be typed, without markdown code blocks, explanations, or meta-commentary unless specifically requested.\"\n",
            config_path.display(),
            toml_string
        );

        fs::write(&config_path, with_comments)?;
        println!("Created example config at: {}", config_path.display());

        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(name = "voxtty")]
#[command(about = "Voice assistant that listens on Linux — say 'code mode' to switch, run local or cloud, type system-wide", long_about = None)]
struct Args {
    #[arg(long, help = "Run echo test to verify audio input")]
    echo_test: bool,

    #[arg(long, help = "List available audio input devices and exit")]
    list_devices: bool,

    #[arg(long, help = "Interactively select audio input device")]
    select_device: bool,

    #[arg(long, help = "Select audio input device by name")]
    device: Option<String>,

    #[arg(long, help = "Enable debug output")]
    debug: bool,

    #[arg(long, help = "Use Speaches API instead of whisper.cpp")]
    speaches: bool,

    #[arg(long, help = "Use OpenAI Whisper API (cloud, requires OPENAI_API_KEY)")]
    openai: bool,

    #[arg(long, help = "Use ElevenLabs API (cloud, requires ELEVENLABS_API_KEY)")]
    elevenlabs: bool,

    #[arg(
        long,
        help = "Enable bidirectional conversation mode with clarification questions (requires ElevenLabs TTS)"
    )]
    bidirectional: bool,

    #[arg(long, help = "Enable realtime WebSocket streaming (lower latency)")]
    realtime: bool,

    #[arg(long, help = "Enable system tray icon")]
    tray: bool,

    #[arg(long, help = "Enable terminal UI (TUI) mode")]
    tui: bool,

    #[arg(
        long,
        help = "Enable text output (typing) in TUI mode. WARNING: Keep focus on target app, not TUI terminal!"
    )]
    tui_output: bool,

    #[arg(long, help = "Enable assistant modes (wake word activated)")]
    assistant: bool,

    #[arg(
        long,
        help = "Enable voice command mode switching without full assistant mode"
    )]
    auto: bool,

    #[arg(
        long,
        help = "Start in paused state (say 'resume' or click tray to activate)"
    )]
    start_paused: bool,

    #[arg(long, help = "Interactively select AI model from models.dev")]
    select_model: bool,

    #[arg(
        long,
        value_name = "PROVIDER",
        help = "LLM provider for assistant mode: ollama, anthropic, openai"
    )]
    llm: Option<String>,

    #[arg(
        long,
        help = "Enable MCP tool support (loads ~/.config/voxtty/mcp_servers.toml or .mcp.json)",
        conflicts_with = "mock_mcp"
    )]
    mcp: bool,

    #[arg(
        long,
        help = "Use a built-in mock MCP server for testing",
        conflicts_with = "mcp"
    )]
    mock_mcp: bool,
}

// Linux-specific tray implementation using ksni (DBus StatusNotifierItem)
#[cfg(target_os = "linux")]
struct VoiceTypingTray {
    enabled: Arc<Mutex<bool>>,
    paused: Arc<Mutex<bool>>, // Voice command paused state
    current_mode: Arc<Mutex<VoiceMode>>,
    assistant_enabled: bool,
    realtime_status: Arc<Mutex<ConnectionStatus>>, // WebSocket connection status
    base_url: String,
    update_counter: Arc<Mutex<u32>>,  // Force menu refresh
    output_enabled: Arc<Mutex<bool>>, // Text output toggle (like TUI 'o' key)
    tui_state: Option<Arc<Mutex<crate::tui::AppState>>>, // TUI state for graceful shutdown
}

#[cfg(target_os = "linux")]
impl Tray for VoiceTypingTray {
    fn id(&self) -> String {
        env!("CARGO_PKG_NAME").into()
    }

    fn title(&self) -> String {
        let enabled = self.enabled.lock().unwrap();
        let paused = self.paused.lock().unwrap();
        let mode = self.current_mode.lock().unwrap();
        let status = self.realtime_status.lock().unwrap();
        let conn_status = match *status {
            ConnectionStatus::Connected => "",
            ConnectionStatus::Connecting => " [Connecting]",
            ConnectionStatus::Disconnected => " [Disconnected]",
        };
        if *paused {
            format!("Voice Typing: PAUSED{}", conn_status)
        } else if *enabled {
            format!("Voice Typing: ON ({:?}){}", mode, conn_status)
        } else {
            format!("Voice Typing: OFF{}", conn_status)
        }
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        let enabled = *self.enabled.lock().unwrap();
        let paused = *self.paused.lock().unwrap();
        let mode = self.current_mode.lock().unwrap().clone();

        // Get letter and base color for mode
        let (letter, mode_r, mode_g, mode_b) = match &mode {
            VoiceMode::Dictation => ('D', 76u8, 175, 80), // Green
            VoiceMode::Assistant { .. } => ('A', 33, 150, 243), // Blue
            VoiceMode::Code { .. } => ('C', 156, 39, 176), // Purple
            VoiceMode::Command => ('$', 255, 193, 7),     // Yellow/Gold
        };

        // Override color based on state
        let (r, g, b) = if !enabled {
            (128u8, 128, 128) // Gray for disabled
        } else if paused {
            (255, 165, 0) // Orange for paused
        } else {
            (mode_r, mode_g, mode_b) // Mode color when active
        };

        // Create 22x22 icon with letter
        let size = 22i32;
        let mut data = Vec::with_capacity((size * size * 4) as usize);

        // Simple 5x7 pixel font for D, A, C, $
        let letter_pixels: [[u8; 5]; 7] = match letter {
            'D' => [
                [1, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 1, 1, 0, 0],
            ],
            'A' => [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
            ],
            'C' => [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            '$' => [
                [0, 1, 1, 1, 0],
                [1, 0, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            _ => [[0; 5]; 7],
        };

        for y in 0..size {
            for x in 0..size {
                let cx = x - size / 2;
                let cy = y - size / 2;
                let dist = ((cx * cx + cy * cy) as f32).sqrt();

                // Check if inside circle
                let in_circle = dist < 10.0;

                // Check if pixel is part of the letter (centered in circle)
                let letter_x = x - 8; // Offset to center 5-wide letter
                let letter_y = y - 7; // Offset to center 7-tall letter
                let in_letter = if (0..5).contains(&letter_x) && (0..7).contains(&letter_y) {
                    letter_pixels[letter_y as usize][letter_x as usize] == 1
                } else {
                    false
                };

                let (ar, ag, ab, aa) = if in_circle {
                    if in_letter {
                        (255u8, 255, 255, 255) // White letter
                    } else {
                        (r, g, b, 255u8) // Colored background
                    }
                } else {
                    (0, 0, 0, 0) // Transparent outside
                };

                // ARGB32 network byte order
                data.push(aa);
                data.push(ar);
                data.push(ag);
                data.push(ab);
            }
        }

        vec![ksni::Icon {
            width: size,
            height: size,
            data,
        }]
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;
        let enabled = *self.enabled.lock().unwrap();
        let paused = *self.paused.lock().unwrap();
        let output_enabled = *self.output_enabled.lock().unwrap();
        let current_mode = self.current_mode.lock().unwrap().clone();

        let mut items = vec![StandardItem {
            label: if enabled {
                "Disable Voice Typing".into()
            } else {
                "Enable Voice Typing".into()
            },
            activate: Box::new(|this: &mut VoiceTypingTray| {
                let mut enabled = this.enabled.lock().unwrap();
                *enabled = !*enabled;

                // Play sound feedback
                if *enabled {
                    sounds::play_resume();
                } else {
                    sounds::play_pause();
                }

                // Increment counter to force menu rebuild
                drop(enabled);
                let mut count = this.update_counter.lock().unwrap();
                *count = count.wrapping_add(1);
            }),
            ..Default::default()
        }
        .into()];

        // Add pause/resume toggle
        items.push(
            StandardItem {
                label: if paused {
                    "▶ Resume".into()
                } else {
                    "⏸ Pause".into()
                },
                activate: Box::new(|this: &mut VoiceTypingTray| {
                    let mut paused = this.paused.lock().unwrap();
                    *paused = !*paused;
                    if *paused {
                        sounds::play_pause();
                    } else {
                        sounds::play_resume();
                    }
                    // Increment counter to force menu rebuild
                    let mut count = this.update_counter.lock().unwrap();
                    *count = count.wrapping_add(1);
                }),
                ..Default::default()
            }
            .into(),
        );

        // Add output toggle
        items.push(
            StandardItem {
                label: if output_enabled {
                    "✓ Text Output Enabled".into()
                } else {
                    "✗ Text Output Disabled".into()
                },
                activate: Box::new(|this: &mut VoiceTypingTray| {
                    let mut output = this.output_enabled.lock().unwrap();
                    *output = !*output;
                    // Increment counter to force menu rebuild
                    let mut count = this.update_counter.lock().unwrap();
                    *count = count.wrapping_add(1);
                }),
                ..Default::default()
            }
            .into(),
        );

        // Add mode selection if assistant is enabled
        if self.assistant_enabled {
            items.push(MenuItem::Separator);

            // Dictation mode
            items.push(
                StandardItem {
                    label: if matches!(current_mode, VoiceMode::Dictation) {
                        "● Dictation Mode".into()
                    } else {
                        "○ Dictation Mode".into()
                    },
                    activate: Box::new(|this: &mut VoiceTypingTray| {
                        let mut mode = this.current_mode.lock().unwrap();
                        *mode = VoiceMode::Dictation;
                        drop(mode);

                        // Increment counter to force menu rebuild
                        let mut count = this.update_counter.lock().unwrap();
                        *count = count.wrapping_add(1);
                        drop(count);

                        sounds::play_mode_change();
                        print_mode_change(&VoiceMode::Dictation, &this.base_url);
                    }),
                    ..Default::default()
                }
                .into(),
            );

            // Assistant mode
            items.push(
                StandardItem {
                    label: if matches!(current_mode, VoiceMode::Assistant { .. }) {
                        "● Assistant Mode".into()
                    } else {
                        "○ Assistant Mode".into()
                    },
                    activate: Box::new(|this: &mut VoiceTypingTray| {
                        let new_mode = VoiceMode::Assistant {
                            context: Vec::new(),
                        };
                        let mut mode = this.current_mode.lock().unwrap();
                        *mode = new_mode.clone();
                        drop(mode);

                        // Increment counter to force menu rebuild
                        let mut count = this.update_counter.lock().unwrap();
                        *count = count.wrapping_add(1);
                        drop(count);

                        sounds::play_mode_change();
                        print_mode_change(&new_mode, &this.base_url);
                    }),
                    ..Default::default()
                }
                .into(),
            );

            // Code mode
            items.push(
                StandardItem {
                    label: if matches!(current_mode, VoiceMode::Code { .. }) {
                        "● Code Mode".into()
                    } else {
                        "○ Code Mode".into()
                    },
                    activate: Box::new(|this: &mut VoiceTypingTray| {
                        let new_mode = VoiceMode::Code { language: None };
                        let mut mode = this.current_mode.lock().unwrap();
                        *mode = new_mode.clone();
                        drop(mode);

                        // Increment counter to force menu rebuild
                        let mut count = this.update_counter.lock().unwrap();
                        *count = count.wrapping_add(1);
                        drop(count);

                        sounds::play_mode_change();
                        print_mode_change(&new_mode, &this.base_url);
                    }),
                    ..Default::default()
                }
                .into(),
            );

            // Command mode
            items.push(
                StandardItem {
                    label: if matches!(current_mode, VoiceMode::Command) {
                        "● Command Mode".into()
                    } else {
                        "○ Command Mode".into()
                    },
                    activate: Box::new(|this: &mut VoiceTypingTray| {
                        let new_mode = VoiceMode::Command;
                        let mut mode = this.current_mode.lock().unwrap();
                        *mode = new_mode.clone();
                        drop(mode);

                        // Increment counter to force menu rebuild
                        let mut count = this.update_counter.lock().unwrap();
                        *count = count.wrapping_add(1);
                        drop(count);

                        sounds::play_mode_change();
                        print_mode_change(&new_mode, &this.base_url);
                    }),
                    ..Default::default()
                }
                .into(),
            );
        }

        items.push(MenuItem::Separator);
        items.push(
            StandardItem {
                label: "Quit".into(),
                activate: Box::new(|this: &mut VoiceTypingTray| {
                    // Signal TUI to exit gracefully if present
                    if let Some(ref tui_state) = this.tui_state {
                        if let Ok(mut s) = tui_state.lock() {
                            s.should_exit = true;
                            drop(s);
                            // Give TUI time to clean up
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    }
                    std::process::exit(0);
                }),
                ..Default::default()
            }
            .into(),
        );

        items
    }
}

// Removed: print_mode_change() now in app_state module

struct AudioRecorder {
    device: cpal::Device,
    config: cpal::StreamConfig,
    vad: Vad,
    vad_sample_rate: u32,
}

impl AudioRecorder {
    fn new(device: cpal::Device, quiet: bool) -> Result<Self> {
        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        let device_sample_rate = supported_config.sample_rate().0;

        let vad_sample_rate = match device_sample_rate {
            8000 | 16000 | 32000 | 48000 => device_sample_rate,
            r if r < 8000 => 8000,
            r if r < 16000 => 8000,
            r if r < 32000 => 16000,
            r if r < 48000 => 32000,
            _ => 48000,
        };

        if !quiet {
            println!(
                "Audio: {} Hz, {} channels, {:?} format (VAD: {} Hz)",
                device_sample_rate,
                supported_config.channels(),
                supported_config.sample_format(),
                vad_sample_rate
            );
        }

        let config = supported_config.into();

        let mut vad = Vad::new();
        vad.set_mode(VadMode::LowBitrate);

        Ok(Self {
            device,
            config,
            vad,
            vad_sample_rate,
        })
    }

    fn into_device(self) -> cpal::Device {
        self.device
    }

    fn set_device(&mut self, device: cpal::Device) -> Result<()> {
        let supported_config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        let device_sample_rate = supported_config.sample_rate().0;

        let vad_sample_rate = match device_sample_rate {
            8000 | 16000 | 32000 | 48000 => device_sample_rate,
            r if r < 8000 => 8000,
            r if r < 16000 => 8000,
            r if r < 32000 => 16000,
            r if r < 48000 => 32000,
            _ => 48000,
        };

        self.device = device;
        self.config = supported_config.into();
        self.vad_sample_rate = vad_sample_rate;
        Ok(())
    }

    fn record_until_silence(&mut self, debug: bool, enabled: Arc<Mutex<bool>>) -> Result<Vec<i16>> {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let samples_clone = samples.clone();

        let err_fn = |err| eprintln!("Stream error: {}", err);

        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut samples = samples_clone.lock().unwrap();
                for &sample in data {
                    let sample_i16 = (sample * i16::MAX as f32) as i16;
                    samples.push(sample_i16);
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;

        let frame_size = (self.vad_sample_rate as usize * VAD_FRAME_MS) / 1000;
        let silence_frames = (SILENCE_DURATION_MS as usize * 1000) / (VAD_FRAME_MS * 1000);
        let min_speech_frames = (MIN_SPEECH_DURATION_MS as usize * 1000) / (VAD_FRAME_MS * 1000);

        let mut silence_count = 0;
        let mut speech_count = 0;
        let mut has_speech = false;
        let mut frame_count = 0;

        loop {
            thread::sleep(Duration::from_millis(VAD_FRAME_MS as u64));

            // Check if disabled during recording
            let is_enabled = *enabled.lock().unwrap();
            if !is_enabled && has_speech {
                if debug {
                    println!("[DEBUG] Voice typing disabled during recording, stopping...");
                }
                break;
            }

            let current_samples = {
                let samples = samples.lock().unwrap();
                samples.clone()
            };

            if debug && frame_count % 10 == 0 {
                println!(
                    "[DEBUG] Total samples: {}, Frame size: {}",
                    current_samples.len(),
                    frame_size
                );
            }

            if current_samples.len() >= frame_size {
                let frame = &current_samples[current_samples.len() - frame_size..];

                let max_amplitude = frame.iter().map(|&s| s.abs()).max().unwrap_or(0);
                if debug && frame_count % 10 == 0 {
                    println!("[DEBUG] Max amplitude: {}", max_amplitude);
                }

                let is_voice = self.vad.is_voice_segment(frame).unwrap_or(false)
                    || max_amplitude > AMPLITUDE_THRESHOLD;

                if is_voice {
                    if debug && silence_count == 0 {
                        println!(
                            "[DEBUG] 🎤 Speech detected! Amplitude: {} (frame {})",
                            max_amplitude, frame_count
                        );
                    }
                    silence_count = 0;
                    speech_count += 1;
                    if speech_count >= min_speech_frames {
                        has_speech = true;
                    }
                } else if has_speech {
                    silence_count += 1;
                    if debug {
                        println!("[DEBUG] 🔇 Silence {}/{}", silence_count, silence_frames);
                    }
                    if silence_count >= silence_frames {
                        break;
                    }
                }
            }
            frame_count += 1;
        }

        drop(stream);

        let final_samples = samples.lock().unwrap().clone();
        Ok(final_samples)
    }
}

fn list_input_devices() -> Result<()> {
    let host = get_audio_host();
    let devices = host.input_devices()?;
    println!("Available audio input devices ({:?}):", host.id());
    for (index, device) in devices.enumerate() {
        if let Ok(name) = device.name() {
            println!("  {}: {}", index, name);
        }
    }
    Ok(())
}

fn select_input_device(
    interactive: bool,
    quiet: bool,
    device_name: Option<String>,
) -> Result<cpal::Device> {
    let host = get_audio_host();
    let mut devices: Vec<_> = host.input_devices()?.collect();

    if devices.is_empty() {
        anyhow::bail!("No input devices found");
    }

    // Try to match by name if provided
    if let Some(name) = device_name {
        if name != "default" {
            // First try exact match
            if let Some(pos) = devices
                .iter()
                .position(|d| d.name().unwrap_or_default() == name)
            {
                let device = devices.remove(pos);
                if !quiet {
                    println!("Using audio device: {}", device.name()?);
                }
                return Ok(device);
            }

            // Try partial match
            if let Some(pos) = devices
                .iter()
                .position(|d| d.name().unwrap_or_default().contains(&name))
            {
                let device = devices.remove(pos);
                if !quiet {
                    println!("Using audio device: {}", device.name()?);
                }
                return Ok(device);
            }

            if !quiet {
                eprintln!(
                    "Warning: Device '{}' not found, falling back to default/selection",
                    name
                );
            }
        }
    }

    if !interactive {
        let device = host
            .default_input_device()
            .context("No default input device found")?;
        if !quiet {
            println!("🎤 Using default audio device: {}", device.name()?);
        }
        eprintln!(
            "[AUDIO DEVICE] Selected: {} | Time: {}",
            device.name().unwrap_or_else(|_| "Unknown".to_string()),
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        );
        return Ok(device);
    }

    if devices.len() == 1 {
        let device = devices.remove(0);
        if !quiet {
            println!("Using audio device: {}", device.name()?);
        }
        return Ok(device);
    }

    let device_names: Vec<String> = devices
        .iter()
        .map(|d| d.name().unwrap_or_else(|_| "Unknown".to_string()))
        .collect();

    let default_idx = device_names
        .iter()
        .position(|name| name.contains("default"))
        .unwrap_or(0);

    let selection = Select::new()
        .with_prompt("Select audio input device")
        .items(&device_names)
        .default(default_idx)
        .interact()?;

    Ok(devices.remove(selection))
}

fn save_wav(samples: &[i16], path: &PathBuf, sample_rate: u32, channels: u16) -> Result<()> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[derive(Deserialize)]
struct SpeachesResponse {
    text: String,
}

fn transcribe_audio(audio_path: &PathBuf, backend: Backend, config: &Config) -> Result<String> {
    let client = reqwest::blocking::Client::new();
    let file = std::fs::read(audio_path)?;

    match backend {
        Backend::Speaches => {
            let url = &config.speaches_base_url;
            let model = &config.transcription_model_id;

            let form = reqwest::blocking::multipart::Form::new()
                .text("model", model.to_string())
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
                .post(url)
                .multipart(form)
                .send()?
                .json::<SpeachesResponse>()?;

            Ok(response.text)
        }
        Backend::OpenAI => {
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

            let response = client
                .post(&config.transcription_url)
                .header("Authorization", format!("Bearer {}", config.openai_api_key))
                .multipart(form)
                .send()?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response
                    .text()
                    .unwrap_or_else(|_| "Unknown error".to_string());
                anyhow::bail!("OpenAI API error: {} - {}", status, error_text);
            }

            let result: SpeachesResponse = response.json()?;
            Ok(result.text)
        }
        Backend::WhisperCpp => {
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
                .post(&config.whisper_url)
                .multipart(form)
                .send()?
                .text()?;

            Ok(response.trim().to_string())
        }
    }
}

fn type_text(text: &str, config: &Config, debug: bool) -> Result<()> {
    // Filter out common false transcriptions
    if text.len() <= 15 && text == "Thanks for watching!" {
        return Ok(());
    }

    if text.contains('[') {
        return Ok(());
    }

    if debug {
        println!("[DEBUG] Typing text via ydotool: {:?}", text);
    }

    Command::new("ydotool")
        .env("YDOTOOL_SOCKET", &config.ydotool_socket)
        .arg("type")
        .arg(text)
        .spawn()?;

    Ok(())
}

/// Parse command JSON response and extract the shell command
fn parse_command_json(
    response: &str,
    fallback: &str,
    tui_state: Option<&Arc<Mutex<tui::AppState>>>,
    debug: bool,
) -> String {
    use serde_json::Value;

    // Try to parse JSON
    if let Ok(json) = serde_json::from_str::<Value>(response) {
        if let Some(command) = json.get("command").and_then(|c| c.as_str()) {
            // Support both old keys (heard/interpretation) and new tool keys (hearing/understanding)
            let interpretation = json
                .get("interpretation")
                .or_else(|| json.get("understanding"))
                .and_then(|i| i.as_str());

            let heard = json
                .get("heard")
                .or_else(|| json.get("hearing"))
                .and_then(|h| h.as_str());

            // Check for audio field (response to speak)
            if let Some(audio_text) = json.get("audio").and_then(|a| a.as_str()) {
                if !audio_text.is_empty() && debug {
                    println!("[DEBUG] Audio response: {}", audio_text);
                    // TODO: Trigger TTS if available
                }
            }

            if debug {
                if let (Some(h), Some(interp)) = (heard, interpretation) {
                    println!("[DEBUG] Heard: {}", h);
                    println!("[DEBUG] Interpretation: {}", interp);
                    println!("[DEBUG] Command: {}", command);

                    // Display additional metadata if available
                    if let Some(confidence) = json.get("confidence").and_then(|c| c.as_str()) {
                        println!("[DEBUG] Confidence: {}", confidence);
                    }
                    if let Some(risk) = json.get("risk").and_then(|r| r.as_str()) {
                        println!("[DEBUG] Risk: {}", risk);
                    }
                    if let Some(category) = json.get("category").and_then(|c| c.as_str()) {
                        println!("[DEBUG] Category: {}", category);
                    }
                }
            }

            // Check if command is empty (rejected)
            if command.is_empty() {
                if let Some(interp) = interpretation {
                    // Display rejection message in TUI
                    if let Some(state) = tui_state {
                        if let Ok(mut s) = state.lock() {
                            if let Some(h) = heard {
                                s.last_input = format!("🎤 {}", h);
                            }
                            s.last_transcription = format!("🚫 {}", interp);
                            s.last_transcription_time = Some(std::time::Instant::now());
                        }
                    } else {
                        // Non-TUI mode: print rejection to stderr
                        eprintln!("🚫 {}", interp);
                    }
                }
                return String::new(); // Return empty to prevent typing
            }

            // Update TUI with interpretation and risk warnings
            if let Some(state) = tui_state {
                if let Ok(mut s) = state.lock() {
                    if let Some(h) = heard {
                        s.last_input = format!("🎤 {}", h);
                    }

                    // Display risk warnings for medium/high risk commands
                    if let Some(risk) = json.get("risk").and_then(|r| r.as_str()) {
                        match risk {
                            "high" => {
                                s.last_transcription = format!("⚠️  HIGH RISK: {}", command);
                                s.last_transcription_time = Some(std::time::Instant::now());
                            }
                            "medium" => {
                                s.last_transcription = format!("⚡ MEDIUM RISK: {}", command);
                                s.last_transcription_time = Some(std::time::Instant::now());
                            }
                            _ => {}
                        }
                    }
                }
            }

            return command.to_string();
        }
    }

    // Fallback: if JSON parsing fails, use original transcription
    if debug {
        eprintln!(
            "[DEBUG] Failed to parse command JSON, using fallback: {}",
            fallback
        );
    }
    fallback.to_string()
}

fn type_command(text: &str, config: &Config, debug: bool) -> Result<()> {
    if debug {
        println!("[DEBUG] Typing command via ydotool: {:?}", text);
    }

    // Type the command text
    Command::new("ydotool")
        .env("YDOTOOL_SOCKET", &config.ydotool_socket)
        .arg("type")
        .arg(text)
        .spawn()?
        .wait()?;

    // Press Enter key (key code 28 for Return/Enter)
    Command::new("ydotool")
        .env("YDOTOOL_SOCKET", &config.ydotool_socket)
        .arg("key")
        .arg("28:1") // Press
        .arg("28:0") // Release
        .spawn()?;

    Ok(())
}

// Removed: playback_audio() now in controls module

/// Check if the transcription backend is available
fn check_backend_health(url: &str, is_speaches: bool) -> Result<()> {
    use std::time::Duration;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    // Extract base URL (remove path for health check)
    let base_url = if is_speaches {
        // Speaches: http://localhost:8000/v1/audio/transcriptions -> http://localhost:8000
        url.replace("/v1/audio/transcriptions", "")
    } else {
        // whisper.cpp: http://127.0.0.1:7777/inference -> http://127.0.0.1:7777
        url.replace("/inference", "")
    };

    // Try to connect to the server
    let health_url = if is_speaches {
        format!("{}/health", base_url)
    } else {
        // whisper.cpp doesn't have a health endpoint, just try the base
        base_url.clone()
    };

    let response = client.get(&health_url).send();

    match response {
        Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 404 => {
            // 404 is OK - server is running but endpoint may not exist
            Ok(())
        }
        Ok(resp) => {
            anyhow::bail!("Server returned status {}", resp.status())
        }
        Err(e) => {
            if e.is_connect() {
                anyhow::bail!("Connection refused - is the server running?")
            } else if e.is_timeout() {
                anyhow::bail!("Connection timeout - server not responding")
            } else {
                anyhow::bail!("{}", e)
            }
        }
    }
}

fn echo_test_mode(mut recorder: AudioRecorder, debug: bool) -> Result<()> {
    println!("\n=== Echo Test Mode ===");
    println!("Listening... Speak and pause to hear playback.\n");

    let sample_rate = recorder.config.sample_rate.0;
    let channels = recorder.config.channels;

    // Echo test is always enabled
    let enabled = Arc::new(Mutex::new(true));

    loop {
        if debug {
            println!("[ECHO] Waiting for speech...");
        }
        match recorder.record_until_silence(debug, enabled.clone()) {
            Ok(samples) => {
                if debug {
                    println!("[ECHO] Recorded {} samples", samples.len());
                }
                if samples.is_empty() {
                    continue;
                }

                // Use controls module playback
                controls::playback_audio(&samples, sample_rate, channels, false)?;
                if debug {
                    println!("[ECHO] Ready for next recording...\n");
                }
            }
            Err(e) => {
                eprintln!("Recording failed: {}", e);
            }
        }
    }
}

fn redirect_stderr_to_file() -> Result<()> {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    // Create a log file in the temporary directory
    let log_path = std::env::temp_dir().join("voxtty_error.log");

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .context("Failed to open error log file")?;

    let fd = file.as_raw_fd();
    unsafe {
        libc::dup2(fd, libc::STDERR_FILENO);
    }

    Ok(())
}

fn get_available_devices() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(host) = get_audio_host().input_devices() {
        for device in host {
            if let Ok(name) = device.name() {
                names.push(name);
            }
        }
    }
    names
}

/// Create ElevenLabs TTS client with pronunciation dictionary from config
fn create_elevenlabs_tts(config: &Config) -> elevenlabs_tts::ElevenLabsTts {
    use elevenlabs_tts::{ElevenLabsTts, PronunciationDictionaryLocator};

    let mut tts = ElevenLabsTts::new(
        config.elevenlabs_api_key.clone(),
        config.elevenlabs_voice_id.clone(),
    );

    // Add pronunciation dictionary if configured
    if let (Some(dict_id), Some(version_id)) = (
        config.elevenlabs_pronunciation_dict_id.clone(),
        config.elevenlabs_pronunciation_dict_version.clone(),
    ) {
        tts = tts.with_pronunciation_dict(PronunciationDictionaryLocator {
            pronunciation_dictionary_id: dict_id,
            version_id,
        });
    }

    tts
}

fn main() -> Result<()> {
    let mut args = Args::parse();

    if args.list_devices {
        return list_input_devices();
    }

    // TUI mode automatically enables assistant mode for voice commands
    if args.tui {
        args.assistant = true;
    }

    // Bidirectional mode requires assistant mode for LLM
    if args.bidirectional {
        args.assistant = true;
    }

    // Load config first to validate API keys BEFORE TUI initialization
    let mut config = Config::load()?;

    // Initialize core shared state
    let wake_word_detector = WakeWordDetector::new();
    // Start in Assistant mode if --assistant flag is provided, otherwise Dictation
    let initial_mode = if args.assistant {
        VoiceMode::Assistant { context: vec![] }
    } else {
        VoiceMode::Dictation
    };
    let current_mode = Arc::new(Mutex::new(initial_mode.clone()));
    let enabled = Arc::new(Mutex::new(true));
    let paused = Arc::new(Mutex::new(args.start_paused)); // Voice command pause state
    let is_tts_speaking = Arc::new(Mutex::new(false)); // Flag to pause audio capture during TTS playback
    let tts_interrupt = Arc::new(std::sync::atomic::AtomicBool::new(false)); // Flag to interrupt TTS when user speaks
    let realtime_status = Arc::new(Mutex::new(ConnectionStatus::Disconnected));
    let output_enabled = Arc::new(Mutex::new(true)); // Text output enabled by default

    // Validate API keys BEFORE starting TUI to show errors properly
    // This prevents terminal corruption when exiting on error

    // Check if ElevenLabs backend is being used (flag or config)
    let using_elevenlabs = args.elevenlabs
        || (!args.openai && !args.speaches && config.backend.to_lowercase() == "elevenlabs");

    if using_elevenlabs && config.elevenlabs_api_key.is_empty() {
        eprintln!("❌ ElevenLabs backend requires an API key");
        eprintln!("   Set ELEVENLABS_API_KEY environment variable or add to config file");
        eprintln!();
        eprintln!("Debug info:");
        eprintln!("  - Config backend: {}", config.backend);
        eprintln!(
            "  - API key in config: {}",
            if config.elevenlabs_api_key.is_empty() {
                "empty"
            } else {
                "present"
            }
        );
        eprintln!("  - Config file path: {:?}", Config::config_path().ok());
        eprintln!();
        std::process::exit(1);
    }

    // Check if OpenAI backend is being used (flag or config)
    let using_openai = args.openai
        || (!args.elevenlabs && !args.speaches && config.backend.to_lowercase() == "openai");

    if using_openai && config.openai_api_key.is_empty() {
        eprintln!("❌ OpenAI Whisper backend requires an API key");
        eprintln!("   Set OPENAI_API_KEY environment variable or add to config file");
        eprintln!();
        std::process::exit(1);
    }

    // Validate realtime API keys if --realtime flag is used
    if args.realtime {
        if using_elevenlabs && config.elevenlabs_api_key.is_empty() {
            eprintln!("❌ ElevenLabs realtime requires an API key");
            eprintln!("   Set ELEVENLABS_API_KEY environment variable or add to config file");
            eprintln!();
            std::process::exit(1);
        }
        if using_openai && config.openai_api_key.is_empty() {
            eprintln!("❌ OpenAI realtime requires an API key");
            eprintln!("   Set OPENAI_API_KEY environment variable or add to config file");
            eprintln!();
            std::process::exit(1);
        }
    }

    // Validate bidirectional mode requirements
    if args.bidirectional && config.elevenlabs_api_key.is_empty() {
        eprintln!("❌ Bidirectional mode requires ELEVENLABS_API_KEY environment variable");
        eprintln!("   Set it with: export ELEVENLABS_API_KEY=your_key");
        eprintln!();
        std::process::exit(1);
    }

    // Redirect stderr if TUI is enabled to prevent screen corruption
    // Do this AFTER validation so errors are visible
    if args.tui {
        if let Err(e) = redirect_stderr_to_file() {
            eprintln!("Warning: Failed to redirect stderr: {}", e);
        }
    }

    // Initialize TUI state if enabled
    let tui_state = if args.tui {
        // Initialize other state fields from config/args
        let backend_base = if args.elevenlabs {
            "ElevenLabs"
        } else if args.openai {
            "OpenAI"
        } else if args.speaches {
            "Speaches"
        } else {
            match config.backend.to_lowercase().as_str() {
                "speaches" => "Speaches",
                "openai" => "OpenAI",
                "elevenlabs" => "ElevenLabs",
                _ => "whisper.cpp",
            }
        };

        let state = tui::AppState {
            available_devices: get_available_devices(),
            backend: if args.realtime {
                format!("{} (Realtime)", backend_base)
            } else {
                backend_base.to_string()
            },
            mode: initial_mode.clone(),
            output_enabled: args.tui_output,
            bidirectional_enabled: args.bidirectional,
            is_enabled: true,
            ..Default::default()
        };

        Some(Arc::new(Mutex::new(state)))
    } else {
        None
    };

    // Launch TUI in background thread if requested
    if let Some(state) = tui_state.clone() {
        // Set up signal handler for graceful shutdown
        let tui_state_for_handler = state.clone();
        ctrlc::set_handler(move || {
            eprintln!("Received exit signal, shutting down...");
            if let Ok(mut s) = tui_state_for_handler.lock() {
                s.should_exit = true;
            }
            // Give TUI a moment to clean up, then exit forcefully
            std::thread::sleep(Duration::from_millis(250));
            std::process::exit(0);
        })
        .expect("Error setting Ctrl-C handler");

        use tui::TuiApp;

        thread::spawn(move || {
            let mut app = TuiApp::new(state);
            let _ = app.run(); // Run TUI, ignore errors on exit
        });

        // Give TUI time to start
        thread::sleep(Duration::from_millis(100));
    }

    // Interactive model selection if requested
    if args.select_model {
        let selector = ModelSelector::new();
        let model_config = selector.interactive_select()?;

        println!("\n✓ Model configuration updated");
        println!("  Provider: {}", model_config.provider_id);
        println!("  Model: {}", model_config.model_id);
        println!("  Base URL: {}", model_config.base_url);

        // Update config with selected model
        config.chat_completion_base_url = model_config.base_url;
        config.chat_completion_api_key = model_config.api_key;
        config.llm_model = model_config.model_id;

        // Save to config file
        let config_path = Config::config_path()?;
        let toml_string = toml::to_string_pretty(&config)?;
        fs::write(&config_path, toml_string)?;
        println!("  Saved to: {}", config_path.display());
        println!("\nYou can now use --assistant flag to enable assistant mode with this model.\n");
        if tui_state.is_none() {
            return Ok(());
        }
    }

    // Determine which backend to use (CLI flag overrides config)
    // Priority: --elevenlabs > --openai > --speaches > config file > whisper.cpp (default)
    let backend = if args.elevenlabs || args.openai {
        Backend::OpenAI // We'll use realtime provider enum for actual routing
    } else if args.speaches {
        Backend::Speaches
    } else {
        match config.backend.to_lowercase().as_str() {
            "speaches" => Backend::Speaches,
            "openai" => Backend::OpenAI,
            "elevenlabs" => Backend::OpenAI,
            _ => Backend::WhisperCpp,
        }
    };

    // Determine realtime provider if --realtime is enabled
    let realtime_provider = if args.realtime {
        if args.elevenlabs {
            Some(RealtimeProvider::ElevenLabs)
        } else if args.openai {
            Some(RealtimeProvider::OpenAI)
        } else if args.speaches {
            Some(RealtimeProvider::Speaches)
        } else {
            // Default to Speaches for realtime if no cloud provider specified
            Some(RealtimeProvider::Speaches)
        }
    } else {
        None
    };

    // API key validation already done before TUI initialization

    // Only show console output if TUI is not active
    if tui_state.is_none() {
        println!("VoiceTypr - Privacy-focused Voice Typing");
        println!("=========================================\n");

        // Show configuration
        println!("Configuration:");
        println!("  ydotool socket: {}", config.ydotool_socket);

        // Show backend info based on realtime or standard mode
        if let Some(provider) = &realtime_provider {
            match provider {
                RealtimeProvider::ElevenLabs => {
                    println!("  Backend: ElevenLabs ScribeRealtime v2 (cloud)");
                    println!("  Mode: ⚡ Realtime WebSocket (~150ms latency)");
                    println!("  ⚠️  Audio streamed to ElevenLabs servers");
                }
                RealtimeProvider::OpenAI => {
                    println!("  Backend: OpenAI Realtime (cloud)");
                    println!("  Mode: ⚡ Realtime WebSocket (~300ms latency)");
                    println!("  ⚠️  Audio streamed to OpenAI servers");
                }
                RealtimeProvider::Speaches => {
                    println!("  Backend: Speaches Realtime (local)");
                    println!("  URL: {}", config.speaches_base_url);
                    println!("  Mode: ⚡ Realtime WebSocket (local)");
                }
            }
        } else {
            match backend {
                Backend::Speaches => {
                    println!("  Backend: Speaches (local)");
                    println!("  URL: {}", config.speaches_base_url);
                    println!("  Model: {}", config.transcription_model_id);
                }
                Backend::OpenAI => {
                    if args.elevenlabs {
                        println!("  Backend: ElevenLabs (cloud)");
                        println!("  ⚠️  Audio sent to ElevenLabs servers");
                    } else {
                        println!("  Backend: OpenAI Whisper (cloud)");
                        println!("  URL: {}", config.transcription_url);
                        println!("  ⚠️  Audio sent to OpenAI servers");
                    }
                }
                Backend::WhisperCpp => {
                    println!("  Backend: whisper.cpp (local)");
                    println!("  URL: {}", config.whisper_url);
                }
            }
        }
        if args.assistant {
            println!("  Assistant: Enabled (wake word activated)");
        }
        println!();
    }

    // Check backend connectivity
    let (backend_url, is_speaches_style) = match backend {
        Backend::Speaches => (config.speaches_base_url.as_str(), true),
        Backend::OpenAI => (config.transcription_url.as_str(), true),
        Backend::WhisperCpp => (config.whisper_url.as_str(), false),
    };

    if tui_state.is_none() {
        print!("Checking backend... ");
    }
    match check_backend_health(backend_url, is_speaches_style) {
        Ok(()) => {
            if tui_state.is_none() {
                println!("✓ Backend is ready");
            }
        }
        Err(e) => {
            if tui_state.is_none() {
                println!("✗ Backend not available");
                eprintln!("\n❌ Cannot connect to transcription backend:");
                eprintln!("   URL: {}", backend_url);
                eprintln!("   Error: {}", e);
                eprintln!();
                match backend {
                    Backend::Speaches => {
                        eprintln!("💡 To start Speaches backend:");
                        eprintln!(
                            "   docker run -d -p 8000:8000 ghcr.io/speaches-ai/speaches:latest"
                        );
                    }
                    Backend::OpenAI => {
                        eprintln!("💡 Check your internet connection and API key");
                        eprintln!("   export OPENAI_API_KEY=sk-your-key");
                    }
                    Backend::WhisperCpp => {
                        eprintln!("💡 To start whisper.cpp server:");
                        eprintln!("   ./server -m models/ggml-small.en.bin --port 7777");
                    }
                }
                eprintln!();
            }
            std::process::exit(1);
        }
    }
    if tui_state.is_none() {
        println!();
        // Print initial mode
        print_mode_change(&initial_mode, &config.chat_completion_base_url);
    }

    // Initialize processor registry
    let mut registry = ProcessorRegistry::new();

    // Register transcription processor (always available)
    let transcription_backend = match backend {
        Backend::Speaches => TranscriptionBackend::Speaches,
        Backend::OpenAI => TranscriptionBackend::OpenAI,
        Backend::WhisperCpp => TranscriptionBackend::WhisperCpp,
    };

    let transcription_config = TranscriptionConfig {
        backend: transcription_backend,
        speaches_url: config.speaches_base_url.clone(),
        speaches_model: config.transcription_model_id.clone(),
        whisper_url: config.whisper_url.clone(),
        openai_url: config.transcription_url.clone(),
        openai_api_key: config.openai_api_key.clone(),
    };
    registry.register(Box::new(TranscriptionProcessor::new(transcription_config)));

    // Load MCP tools in a background thread to avoid blocking realtime connection
    let mcp_manager: Option<Arc<Mutex<mcp_tools::McpManager>>> = if args.mcp || args.mock_mcp {
        let mcp_config = if args.mock_mcp {
            // Built-in mock MCP server for testing
            let mock_script = include_str!("../test_mcp_server.py");
            let mock_path = std::env::temp_dir().join("voxtty_mock_mcp_server.py");
            std::fs::write(&mock_path, mock_script).ok();

            mcp_tools::McpConfig {
                servers: vec![mcp_tools::McpServerConfig {
                    name: "mock".to_string(),
                    command: "python3".to_string(),
                    args: vec![mock_path.to_string_lossy().to_string()],
                    env: std::collections::HashMap::new(),
                }],
            }
        } else {
            match mcp_tools::McpManager::load_config() {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("Warning: Failed to load MCP config: {}", e);
                    eprintln!(
                            "   Create ~/.config/voxtty/mcp_servers.toml or .mcp.json to configure MCP servers"
                        );
                    mcp_tools::McpConfig {
                        servers: Vec::new(),
                    }
                }
            }
        };

        if !mcp_config.servers.is_empty() {
            // Create empty manager now, populate in background thread
            let mgr = Arc::new(Mutex::new(mcp_tools::McpManager::empty()));
            let mgr_bg = Arc::clone(&mgr);
            let tui_state_bg = tui_state.clone();
            let is_tui = tui_state.is_some();

            // Show loading state in TUI immediately
            if let Some(ref state) = tui_state {
                if let Ok(mut s) = state.lock() {
                    s.mcp_info = Some((0, 0));
                }
            }

            std::thread::spawn(move || {
                if !is_tui {
                    eprintln!("🔧 Connecting to MCP servers...");
                }
                let manager = mcp_tools::McpManager::from_config(&mcp_config);
                let server_count = manager.server_count();
                let tool_count = manager.tool_count();
                if !is_tui {
                    eprintln!(
                        "   {} server(s) connected, {} tool(s) available",
                        server_count, tool_count
                    );
                }
                // Replace empty manager with the initialized one
                mgr_bg.lock().unwrap().replace_with(manager);
                // Update TUI with MCP info
                if let Some(ref state) = tui_state_bg {
                    if let Ok(mut s) = state.lock() {
                        s.mcp_info = Some((server_count, tool_count));
                    }
                }
            });

            Some(mgr)
        } else {
            None
        }
    } else {
        None
    };

    // Register assistant processor if enabled (--assistant or --auto)
    if args.assistant || args.auto {
        // Override LLM provider based on --llm flag
        let (llm_base_url, llm_api_key, llm_model) = match args.llm.as_deref() {
            Some("anthropic") => {
                let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
                if key.is_empty() {
                    if tui_state.is_none() {
                        eprintln!(
                            "❌ --llm anthropic requires ANTHROPIC_API_KEY environment variable"
                        );
                    }
                    std::process::exit(1);
                }
                (
                    "https://api.anthropic.com/v1".to_string(),
                    key,
                    "claude-sonnet-4-5-20250929".to_string(),
                )
            }
            Some("openai") => {
                // For OpenAI LLM, use CHAT_COMPLETION_API_KEY or fall back to OPENAI_API_KEY
                let key = std::env::var("CHAT_COMPLETION_API_KEY")
                    .or_else(|_| std::env::var("OPENAI_API_KEY"))
                    .unwrap_or_default();
                if key.is_empty() {
                    if tui_state.is_none() {
                        eprintln!("❌ --llm openai requires CHAT_COMPLETION_API_KEY or OPENAI_API_KEY environment variable");
                    }
                    std::process::exit(1);
                }
                (
                    "https://api.openai.com/v1".to_string(),
                    key,
                    "gpt-4o".to_string(),
                )
            }
            Some("ollama") | None => {
                // Use config values, but auto-detect API key from env if URL is OpenAI/Anthropic
                let mut api_key = config.chat_completion_api_key.clone();

                // If API key is empty and URL is OpenAI, try env var
                if api_key.is_empty() && config.chat_completion_base_url.contains("openai.com") {
                    api_key = std::env::var("CHAT_COMPLETION_API_KEY")
                        .or_else(|_| std::env::var("OPENAI_API_KEY"))
                        .unwrap_or_default();
                }

                // If API key is empty and URL is Anthropic, try env var
                if api_key.is_empty() && config.chat_completion_base_url.contains("anthropic.com") {
                    api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
                }

                (
                    config.chat_completion_base_url.clone(),
                    api_key,
                    config.llm_model.clone(),
                )
            }
            Some(other) => {
                if tui_state.is_none() {
                    eprintln!("❌ Unknown LLM provider: {}", other);
                    eprintln!("   Valid options: ollama, anthropic, openai");
                }
                std::process::exit(1);
            }
        };

        // Validate API key for cloud LLM providers (when not using --llm flag)
        if args.llm.is_none() {
            let needs_api_key =
                llm_base_url.contains("anthropic.com") || llm_base_url.contains("openai.com");

            if needs_api_key && llm_api_key.is_empty() {
                if tui_state.is_none() {
                    if llm_base_url.contains("anthropic.com") {
                        eprintln!(
                            "❌ Anthropic API requires ANTHROPIC_API_KEY environment variable"
                        );
                    } else {
                        eprintln!(
                            "❌ OpenAI API requires CHAT_COMPLETION_API_KEY environment variable"
                        );
                    }
                    eprintln!("   Or use --llm ollama for local inference");
                    eprintln!();
                }
                std::process::exit(1);
            }
        }

        // Determine transcription URL, model, and API key based on selected backend
        let (transcription_url, transcription_model, transcription_api_key) = match backend {
            Backend::Speaches => (
                config.speaches_base_url.clone(),
                config.transcription_model_id.clone(),
                String::new(),
            ),
            Backend::OpenAI => (
                config.transcription_url.clone(),
                "whisper-1".to_string(),
                config.openai_api_key.clone(),
            ),
            Backend::WhisperCpp => (config.whisper_url.clone(), "".to_string(), String::new()),
        };

        // Only register AssistantProcessor if NOT in bidirectional mode
        // (bidirectional mode uses ConversationProcessor instead)
        if !args.bidirectional {
            let assistant_config = SpeachesAssistantConfig {
                base_url: llm_base_url.clone(),
                api_key: llm_api_key,
                transcription_url,
                transcription_api_key,
                transcription_model,
                llm_model: llm_model.clone(),
                system_prompt: config.system_prompt.clone(),
                code_system_prompt: config.code_system_prompt.clone(),
            };
            let mut backend = SpeachesAssistantBackend::new(assistant_config);
            if let Some(ref mgr) = mcp_manager {
                backend = backend.with_mcp_manager(Arc::clone(mgr));
            }
            let assistant_backend = Box::new(backend);
            registry.register(Box::new(AssistantProcessor::new(assistant_backend)));
        }

        if tui_state.is_none() {
            println!("🤖 Assistant modes available:");
            println!("   • Say 'hey assistant' for writing help");
            println!("   • Say 'code mode' for code generation");
            println!("   • Say 'dictation mode' to return to normal");
            println!("   • LLM: {} ({})", llm_model, llm_base_url);

            // Privacy check: Detect local vs cloud LLM backends
            // Local (privacy-preserving): localhost, 127.0.0.1, 0.0.0.0 (e.g., Ollama)
            // Cloud (sends data): Any other URL (OpenAI, Anthropic, Google, etc.)
            let is_local = llm_base_url.contains("localhost")
                || llm_base_url.contains("127.0.0.1")
                || llm_base_url.contains("0.0.0.0");

            if !is_local {
                println!();
                println!("⚠️  PRIVACY NOTICE: Using cloud-based AI model");
                println!(
                    "   Your voice transcriptions will be sent to: {}",
                    llm_base_url
                );
                println!("   For complete privacy, use --llm ollama");
            }

            println!();
        }
    }

    // Store ElevenLabs configuration for bidirectional mode startup message
    let elevenlabs_key_for_startup: Option<String>;

    // Register conversation processor if bidirectional mode enabled
    if args.bidirectional {
        // API key validation already done before TUI initialization
        if config.elevenlabs_api_key.is_empty() {
            eprintln!("❌ This should never happen - API key validation was done earlier");
            std::process::exit(1);
        }

        // Get LLM configuration (reuse from assistant mode if available, or use defaults)
        let (llm_base_url, llm_api_key, llm_model) = if args.assistant || args.auto {
            // Reuse the LLM settings from assistant mode
            match args.llm.as_deref() {
                Some("anthropic") => {
                    let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
                    (
                        "https://api.anthropic.com/v1".to_string(),
                        key,
                        "claude-sonnet-4-5-20250929".to_string(),
                    )
                }
                Some("openai") => {
                    let key = std::env::var("CHAT_COMPLETION_API_KEY")
                        .or_else(|_| std::env::var("OPENAI_API_KEY"))
                        .unwrap_or_default();
                    (
                        "https://api.openai.com/v1".to_string(),
                        key,
                        "gpt-4o".to_string(),
                    )
                }
                _ => (
                    config.chat_completion_base_url.clone(),
                    config.chat_completion_api_key.clone(),
                    config.llm_model.clone(),
                ),
            }
        } else {
            // Default LLM settings
            (
                config.chat_completion_base_url.clone(),
                config.chat_completion_api_key.clone(),
                config.llm_model.clone(),
            )
        };

        use crate::processors_conversation::ConversationProcessor;

        // Create TTS client with pronunciation dictionary support
        let tts_client = Arc::new(create_elevenlabs_tts(&config));

        // Store API key for startup message (outside this block scope)
        elevenlabs_key_for_startup = Some(config.elevenlabs_api_key.clone());

        let mut conversation_processor = ConversationProcessor::with_tts_client(
            llm_base_url.clone(),
            llm_api_key.clone(),
            llm_model.clone(),
            tts_client,
            is_tts_speaking.clone(),
            tts_interrupt.clone(),
        );

        // Inject transcription function
        let config_clone = config.clone();
        let backend_clone = backend;
        conversation_processor.set_transcription_fn(move |path| {
            transcribe_audio(&path.to_path_buf(), backend_clone, &config_clone)
        });

        // Inject MCP tools if available
        if let Some(ref mgr) = mcp_manager {
            conversation_processor.set_mcp(Arc::clone(mgr));
        }

        registry.register(Box::new(conversation_processor));
        eprintln!("[DEBUG MAIN] ConversationProcessor registered");

        if tui_state.is_none() {
            println!("💬 Bidirectional conversation mode enabled");
            println!("   • Voice: {}", config.elevenlabs_voice_id);
            println!("   • LLM: {} ({})", llm_model, llm_base_url);
            println!("   • Assistant will ask clarifying questions via voice");
            println!("   • Startup message will play after realtime connection established");
            if config.elevenlabs_pronunciation_dict_id.is_some() {
                println!("   • Using pronunciation dictionary");
            }
            println!();
        }
    } else {
        // Initialize with None when bidirectional mode is not enabled
        elevenlabs_key_for_startup = None;
    }

    let device_name = if let Some(d) = args.device.clone() {
        Some(d)
    } else if config.audio_device != "default" {
        Some(config.audio_device.clone())
    } else {
        None
    };

    let device = select_input_device(args.select_device, args.tui, device_name)?;

    // Update TUI with device name
    if let Some(ref state) = tui_state {
        if let Ok(mut s) = state.lock() {
            s.selected_device = Some(device.name().unwrap_or_else(|_| "Unknown".to_string()));
        }
    }

    let mut recorder = AudioRecorder::new(device, args.tui)?;

    if args.echo_test {
        return echo_test_mode(recorder, args.debug);
    }

    // Setup controls
    let tray_handle = if args.tray {
        let enabled_clone = enabled.clone();
        let paused_clone = paused.clone();
        let mode_clone = current_mode.clone();
        let assistant_enabled = args.assistant;
        let realtime_status_clone = realtime_status.clone();
        let base_url_clone = config.chat_completion_base_url.clone();
        let update_counter = Arc::new(Mutex::new(0u32));
        let update_counter_clone = update_counter.clone();
        let output_enabled_clone = output_enabled.clone();

        let service = TrayService::new(VoiceTypingTray {
            enabled: enabled_clone,
            paused: paused_clone,
            current_mode: mode_clone,
            assistant_enabled,
            realtime_status: realtime_status_clone,
            base_url: base_url_clone,
            update_counter: update_counter_clone,
            output_enabled: output_enabled_clone,
            tui_state: tui_state.clone(),
        });

        let handle = service.handle();

        thread::spawn(move || {
            let _ = service.run();
        });

        if tui_state.is_none() {
            if args.assistant {
                println!("Controls: System tray - Click icon to toggle, select mode from menu");
            } else {
                println!("Controls: System tray - Click icon to toggle");
            }
            println!();
        }

        Some((handle, update_counter))
    } else {
        None
    };

    // Use realtime streaming mode if enabled
    if let Some(mut provider) = realtime_provider {
        if tui_state.is_none() {
            if args.start_paused {
                println!("Listening... (realtime streaming mode) - ⏸️  PAUSED\n");
                println!("Say 'resume' or 'wake up' to start, or click the tray icon.\n");
            } else {
                println!("Listening... (realtime streaming mode)\n");
            }
        }

        // Create realtime config
        let realtime_config = RealtimeConfig {
            provider,
            api_key: match provider {
                RealtimeProvider::ElevenLabs => config.elevenlabs_api_key.clone(),
                RealtimeProvider::OpenAI => config.openai_api_key.clone(),
                RealtimeProvider::Speaches => String::new(),
            },
            base_url: if provider == RealtimeProvider::Speaches {
                Some(config.speaches_base_url.clone())
            } else {
                None
            },
            model: if provider == RealtimeProvider::Speaches {
                Some(config.transcription_model_id.clone())
            } else {
                None
            },
            language: Some("en".to_string()),
            sample_rate: 16000, // We'll resample to 16kHz for realtime
            debug: args.debug,
            quiet: args.tui,
        };

        // Start realtime transcriber (will be restarted on disconnect)
        let mut transcriber = RealtimeTranscriber::new(realtime_config.clone());

        // Only start if not in paused mode
        if !args.start_paused {
            if let Err(e) = transcriber.start() {
                if tui_state.is_none() {
                    eprintln!("❌ Failed to start realtime transcription: {}", e);
                }
                std::process::exit(1);
            }
            // Status will be set via TranscriptionEvent::Connecting -> Connected
        }

        // Use the shared paused state for voice commands
        let paused_clone = paused.clone();

        // Audio capture for realtime streaming (reuse selected device)
        let device = recorder.into_device();

        let supported_config = device.default_input_config()?;
        let sample_rate = supported_config.sample_rate().0;
        let channels = supported_config.channels() as usize;

        // Buffer for audio samples
        let audio_buffer: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
        let audio_buffer_clone = audio_buffer.clone();

        // Clone TUI state for audio callback
        let tui_state_realtime = tui_state.clone();

        // Clone paused and realtime_status states for audio callback
        let _paused_for_callback = paused_clone.clone();
        let realtime_status_for_callback = realtime_status.clone();
        // is_tts_speaking is used for barge-in detection in the event loop

        // Build input stream
        let stream = device.build_input_stream(
            &supported_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert f32 to i16 and downsample to mono if needed
                let samples: Vec<i16> = data
                    .iter()
                    .step_by(channels) // Take only first channel (mono)
                    .map(|&s| (s * i16::MAX as f32) as i16)
                    .collect();

                // Update TUI audio level immediately for real-time visualization
                if let Some(ref state) = tui_state_realtime {
                    let sum: f32 = data.iter().step_by(channels).map(|&s| s.abs()).sum();
                    let avg_level = sum / (data.len() / channels) as f32;

                    if let Ok(mut s) = state.lock() {
                        s.audio_level = avg_level;

                        // Update audio history for Sparkline (scale to 0-100)
                        let level_scaled = (avg_level * 100.0) as u64;
                        s.audio_history.push_back(level_scaled);

                        // Keep only last 100 samples
                        if s.audio_history.len() > 100 {
                            s.audio_history.pop_front();
                        }
                    }
                }

                // Only buffer audio if:
                // 1. Realtime connection is established
                // 2. TTS is not speaking (prevent self-transcription)
                // Note: We buffer even when paused to enable wake word detection
                let is_connected =
                    *realtime_status_for_callback.lock().unwrap() == ConnectionStatus::Connected;

                // Keep sending audio even during TTS playback to enable barge-in
                // (speech detection triggers interrupt of TTS)
                if is_connected {
                    let mut buffer = audio_buffer_clone.lock().unwrap();
                    buffer.extend(samples);
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;

        stream.play()?;

        // Chunk size for streaming (100ms of audio at input sample rate)
        let target_samples = 1600_usize; // 100ms at 16kHz
        let ratio = sample_rate as f32 / 16000.0;
        let chunk_samples = (target_samples as f32 * ratio) as usize;

        if args.debug {
            println!(
                "[DEBUG] Realtime audio: {}Hz input -> 16kHz output, chunk size: {} samples",
                sample_rate, chunk_samples
            );
        }

        // Track if we were previously disabled (for reconnection on re-enable)
        let mut was_disabled = false;

        // Track if startup message has been played (for bidirectional mode)
        let startup_message_played = Arc::new(Mutex::new(false));
        let startup_message_played_clone = startup_message_played.clone();

        // Clone TUI state for updates
        let tui_state_clone = tui_state.clone();
        let output_enabled_clone = output_enabled.clone();
        let mut last_paused = *paused.lock().unwrap();
        let mut last_enabled = *enabled.lock().unwrap();
        let mut last_output_enabled = *output_enabled.lock().unwrap();
        let mut last_mode = current_mode.lock().unwrap().clone();
        let mut last_tts_active = false;

        // Main realtime loop
        loop {
            let is_enabled = *enabled.lock().unwrap();

            // Reset interrupt flag when TTS finishes (works with and without TUI)
            {
                let tts_active = *is_tts_speaking.lock().unwrap();
                if !tts_active && tts_interrupt.load(std::sync::atomic::Ordering::Relaxed) {
                    tts_interrupt.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Update TUI listening state and check exit
            if let Some(ref state) = tui_state_clone {
                if let Ok(mut s) = state.lock() {
                    if s.should_exit {
                        std::process::exit(0);
                    }

                    // Update processing status based on TTS activity
                    let tts_active = *is_tts_speaking.lock().unwrap();

                    // Debug: Log status checks (only when processing or TTS active)
                    if s.is_processing || tts_active {
                        use std::fs::OpenOptions;
                        use std::io::Write;
                        let _ = OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open("/tmp/voxtty_status.log")
                            .and_then(|mut file| {
                                writeln!(file, "[TUI] is_processing={}, tts_active={}, last_tts_active={}, current_status={:?}",
                                    s.is_processing, tts_active, last_tts_active, s.processing_status)
                            });
                    }

                    // Update status when TTS becomes active
                    if tts_active && !last_tts_active {
                        // TTS just started - reset interrupt flag
                        tts_interrupt.store(false, std::sync::atomic::Ordering::Relaxed);
                        s.is_processing = true;
                        s.processing_status = crate::tui::ProcessingStatus::PlayingAudio;
                    } else if !tts_active && last_tts_active {
                        // TTS just finished - conversation turn is complete, reset interrupt flag
                        tts_interrupt.store(false, std::sync::atomic::Ordering::Relaxed);
                        s.is_processing = false;
                        s.processing_status = crate::tui::ProcessingStatus::Idle;
                    } else if tts_active {
                        // TTS is still active - keep status as PlayingAudio
                        s.is_processing = true;
                        s.processing_status = crate::tui::ProcessingStatus::PlayingAudio;
                    }

                    last_tts_active = tts_active;

                    // Handle backend switch request from TUI
                    if s.backend_switch_requested {
                        s.backend_switch_requested = false;

                        // Toggle between OpenAI and Speaches
                        let new_provider = if matches!(provider, RealtimeProvider::OpenAI) {
                            RealtimeProvider::Speaches
                        } else {
                            RealtimeProvider::OpenAI
                        };

                        // Update display
                        s.backend = format!(
                            "{} (Realtime)",
                            match new_provider {
                                RealtimeProvider::OpenAI => "OpenAI",
                                RealtimeProvider::Speaches => "Speaches",
                                RealtimeProvider::ElevenLabs => "ElevenLabs",
                            }
                        );

                        drop(s);

                        // Stop current transcriber
                        transcriber.stop();
                        *realtime_status.lock().unwrap() = ConnectionStatus::Disconnected;
                        if let Ok(mut st) = state.lock() {
                            st.realtime_status = ConnectionStatus::Disconnected;
                        }

                        // Wait a bit for cleanup
                        thread::sleep(Duration::from_millis(500));

                        // Create new transcriber with new provider
                        let new_config = RealtimeConfig {
                            provider: new_provider,
                            api_key: match new_provider {
                                RealtimeProvider::ElevenLabs => config.elevenlabs_api_key.clone(),
                                RealtimeProvider::OpenAI => config.openai_api_key.clone(),
                                RealtimeProvider::Speaches => String::new(),
                            },
                            base_url: if new_provider == RealtimeProvider::Speaches {
                                Some(config.speaches_base_url.clone())
                            } else {
                                None
                            },
                            model: if new_provider == RealtimeProvider::Speaches {
                                Some(config.transcription_model_id.clone())
                            } else {
                                None
                            },
                            language: Some("en".to_string()),
                            sample_rate: 16000,
                            debug: args.debug,
                            quiet: args.tui,
                        };

                        transcriber = RealtimeTranscriber::new(new_config);
                        if let Err(e) = transcriber.start() {
                            if let Ok(mut s) = state.lock() {
                                s.echo_test_status = format!("Backend switch failed: {}", e);
                            }
                        }
                        // Status will be set via TranscriptionEvent::Connecting -> Connected

                        // Update the provider variable for future switches
                        provider = new_provider;

                        continue;
                    }

                    // Handle echo test request from TUI
                    if s.echo_test_requested {
                        s.echo_test_requested = false;
                        s.echo_test_status = "Recording... Speak now!".to_string();
                        drop(s);

                        // Collect audio for echo test (3 seconds worth)
                        let echo_samples = Arc::new(Mutex::new(Vec::new()));
                        let echo_samples_clone = echo_samples.clone();
                        let buffer_clone = audio_buffer.clone();

                        // Capture audio for 3 seconds
                        let capture_duration = Duration::from_secs(3);
                        let capture_start = Instant::now();

                        while capture_start.elapsed() < capture_duration {
                            let mut buffer = buffer_clone.lock().unwrap();
                            let mut echo_buf = echo_samples_clone.lock().unwrap();
                            echo_buf.extend(buffer.drain(..));
                            drop(buffer);
                            drop(echo_buf);
                            thread::sleep(Duration::from_millis(50));
                        }

                        let samples = echo_samples.lock().unwrap().clone();

                        if !samples.is_empty() {
                            // Update status
                            if let Ok(mut s) = state.lock() {
                                s.echo_test_status = "Playing back...".to_string();
                            }

                            // Play back at device sample rate as mono (samples are mono due to step_by(channels))
                            if let Err(e) =
                                controls::playback_audio(&samples, sample_rate, 1, false)
                            {
                                if let Ok(mut s) = state.lock() {
                                    s.echo_test_status = format!("Playback failed: {}", e);
                                }
                            } else {
                                if let Ok(mut s) = state.lock() {
                                    s.echo_test_status = "Echo test complete!".to_string();
                                }
                            }
                        } else if let Ok(mut s) = state.lock() {
                            s.echo_test_status = "No audio detected".to_string();
                        }

                        // Clear status after 3 seconds
                        let state_clone = state.clone();
                        thread::spawn(move || {
                            thread::sleep(Duration::from_secs(3));
                            if let Ok(mut s) = state_clone.lock() {
                                s.echo_test_status = String::new();
                            }
                        });

                        // Continue to next iteration
                        continue;
                    }

                    s.is_listening = is_enabled;
                }

                // Remember old pause state before sync
                let old_paused = last_paused;

                // Sync all state bidirectionally
                let tray_counter = tray_handle.as_ref().map(|(_, counter)| counter.clone());
                sync_state(
                    &enabled,
                    &paused,
                    &output_enabled_clone,
                    &current_mode,
                    &tui_state_clone,
                    &tray_counter,
                    &mut last_enabled,
                    &mut last_paused,
                    &mut last_output_enabled,
                    &mut last_mode,
                );

                // Update tray if state changed
                if let Some((ref handle, _)) = tray_handle {
                    handle.update(|_| {});
                }

                // Handle pause state changes (from tray or TUI)
                let current_paused = *paused.lock().unwrap();
                let pause_changed = old_paused != current_paused;

                if pause_changed {
                    if current_paused {
                        // Pausing - stop connection and clear buffer
                        transcriber.stop();
                        *realtime_status.lock().unwrap() = ConnectionStatus::Disconnected;
                        if let Some(ref state) = tui_state_clone {
                            if let Ok(mut s) = state.lock() {
                                s.realtime_status = ConnectionStatus::Disconnected;
                            }
                        }
                        audio_buffer.lock().unwrap().clear();
                    } else {
                        // Resuming - start connection
                        let status = *realtime_status.lock().unwrap();
                        if status != ConnectionStatus::Connected {
                            if let Err(e) = transcriber.start() {
                                if let Some(ref state) = tui_state_clone {
                                    if let Ok(mut s) = state.lock() {
                                        s.error_message = Some(format!("Failed to connect: {}", e));
                                        s.error_timestamp = Some(Instant::now());
                                    }
                                }
                            }
                            // Status will be set via TranscriptionEvent::Connecting -> Connected
                        }
                    }
                }
            }

            // Handle enable/disable transitions
            if !is_enabled && !was_disabled {
                // Just got disabled - do nothing, keep connection alive for wake words
                if tui_state_clone.is_none() {
                    println!("🔇 Voice detection disabled");
                }
                was_disabled = true;
            } else if is_enabled && was_disabled {
                // Just got re-enabled - no need to reconnect, already connected
                if tui_state_clone.is_none() {
                    println!("🎤 Voice detection enabled");
                }
                was_disabled = false;
            }

            // Send audio chunks to transcriber (only when enabled)
            if is_enabled {
                let mut buffer = audio_buffer.lock().unwrap();
                if buffer.len() >= chunk_samples {
                    // Calculate audio level for debugging
                    let max_sample = buffer.iter().map(|&s| s.abs()).max().unwrap_or(0);
                    let avg_level =
                        buffer.iter().map(|&s| s.abs() as f32).sum::<f32>() / buffer.len() as f32;

                    if args.debug && tui_state_clone.is_none() {
                        eprintln!(
                            "[DEBUG AUDIO] Sending {} samples | max={} avg={:.0} | {}",
                            buffer.len(),
                            max_sample,
                            avg_level,
                            chrono::Local::now().format("%H:%M:%S%.3f")
                        );
                    }

                    // Audio level is now calculated in the audio callback for real-time updates

                    // Resample to 16kHz if needed (simple decimation)
                    let resampled: Vec<i16> = if ratio > 1.0 {
                        buffer
                            .iter()
                            .step_by(ratio.round() as usize)
                            .copied()
                            .collect()
                    } else {
                        buffer.clone()
                    };

                    if let Err(e) = transcriber.send_audio(resampled) {
                        if tui_state_clone.is_none() {
                            eprintln!("[ERROR] Failed to send audio to transcriber: {}", e);
                        }
                    }
                    buffer.clear();
                }
            } else {
                // Clear audio buffer while disabled to avoid stale data
                audio_buffer.lock().unwrap().clear();
            }

            // Check for transcription results
            while let Some(event) = transcriber.try_recv() {
                match event {
                    TranscriptionEvent::Final(text) => {
                        if !text.is_empty() {
                            // Interrupt TTS if user speaks (barge-in)
                            // This handles providers like ElevenLabs that don't emit SpeechStarted
                            let was_tts_active = *is_tts_speaking.lock().unwrap();
                            if was_tts_active {
                                eprintln!("🛑 User spoke during TTS - interrupting playback!");
                                tts_interrupt.store(true, std::sync::atomic::Ordering::SeqCst);
                            }

                            if args.debug && tui_state_clone.is_none() {
                                eprintln!(
                                    "[DEBUG TRANSCRIPTION] Received: '{}' | {} chars | {}",
                                    text,
                                    text.len(),
                                    chrono::Local::now().format("%H:%M:%S%.3f")
                                );
                            }
                            // Always check for pause/resume commands (works in all modes)
                            let (command, should_type) = wake_word_detector.detect_command(&text);

                            match command {
                                VoiceCommand::Pause => {
                                    *paused_clone.lock().unwrap() = true;

                                    // Keep transcriber running for wake word detection
                                    // Just clear the audio buffer to prevent stale data
                                    audio_buffer.lock().unwrap().clear();

                                    sounds::play_pause();
                                    if tui_state_clone.is_none() {
                                        println!(
                                            "⏸️  Paused - say 'resume' or 'wake up' to continue"
                                        );
                                    }
                                    // Update TUI state
                                    if let Some(ref state) = tui_state_clone {
                                        if let Ok(mut s) = state.lock() {
                                            s.is_paused = true;
                                        }
                                    }
                                    // Update tray icon to show paused state
                                    if let Some((ref handle, ref counter)) = tray_handle {
                                        let mut count = counter.lock().unwrap();
                                        *count = count.wrapping_add(1);
                                        drop(count);
                                        handle.update(|_| {});
                                    }
                                    continue;
                                }
                                VoiceCommand::Resume => {
                                    *paused_clone.lock().unwrap() = false;

                                    sounds::play_resume();
                                    if tui_state_clone.is_none() {
                                        println!("▶️  Resumed");
                                    }
                                    // Update TUI state
                                    if let Some(ref state) = tui_state_clone {
                                        if let Ok(mut s) = state.lock() {
                                            s.is_paused = false;
                                        }
                                    }
                                    // Update tray icon to show active state
                                    if let Some((ref handle, ref counter)) = tray_handle {
                                        let mut count = counter.lock().unwrap();
                                        *count = count.wrapping_add(1);
                                        drop(count);
                                        handle.update(|_| {});
                                    }
                                    continue;
                                }
                                VoiceCommand::SwitchMode(mode) => {
                                    // Mode switching only works with --assistant or --auto
                                    if args.auto || args.assistant {
                                        let mut current = current_mode.lock().unwrap();
                                        *current = mode.clone();
                                        drop(current);
                                        sounds::play_mode_change();
                                        if tui_state_clone.is_none() {
                                            print_mode_change(
                                                &mode,
                                                &config.chat_completion_base_url,
                                            );
                                        }

                                        // Update TUI state
                                        if let Some(ref state) = tui_state_clone {
                                            if let Ok(mut s) = state.lock() {
                                                s.mode = mode.clone();
                                                // Clear previous transcription when switching modes
                                                s.last_input.clear();
                                                s.last_transcription.clear();
                                            }
                                        }

                                        // Update tray menu if tray is enabled
                                        if let Some((ref handle, ref counter)) = tray_handle {
                                            let mut count = counter.lock().unwrap();
                                            *count = count.wrapping_add(1);
                                            drop(count);
                                            handle.update(|tray| {
                                                let _counter = tray.update_counter.lock().unwrap();
                                            });
                                        }
                                        continue; // Don't type the wake word
                                    }
                                }
                                VoiceCommand::None => {}
                            }

                            if !should_type {
                                continue;
                            }

                            // Check if paused
                            let is_paused = *paused_clone.lock().unwrap();
                            if is_paused {
                                // In TUI mode, still show transcription but don't process/type
                                if let Some(ref state) = tui_state_clone {
                                    if let Ok(mut s) = state.lock() {
                                        s.last_input = text.clone();
                                        s.last_transcription = format!("[PAUSED] {}", text);
                                        s.last_transcription_time = Some(Instant::now());
                                    }
                                } else if args.debug {
                                    println!("[DEBUG] Paused, ignoring: {}", text);
                                }
                                continue;
                            }

                            // Get current mode and process accordingly
                            let mode_snapshot = current_mode.lock().unwrap().clone();

                            // Store raw input in TUI state (for assistant/code modes)
                            if let Some(ref state) = tui_state_clone {
                                if let Ok(mut s) = state.lock() {
                                    // If last_input was empty and now we have new input,
                                    // this is a new conversation - increment the ID
                                    if s.last_input.is_empty() && !text.trim().is_empty() {
                                        s.current_conversation_id += 1;
                                    }
                                    s.last_input = text.clone();
                                }
                            }

                            let output_text = match mode_snapshot {
                                VoiceMode::Assistant { .. }
                                | VoiceMode::Code { .. }
                                | VoiceMode::Command => {
                                    // Set processing flag for TUI
                                    if let Some(ref state) = tui_state_clone {
                                        if let Ok(mut s) = state.lock() {
                                            s.is_processing = true;
                                            s.processing_status =
                                                crate::tui::ProcessingStatus::Thinking;
                                        }
                                    }

                                    // Process through LLM for assistant/code/command modes
                                    if tui_state_clone.is_none() {
                                        println!("🤖 Processing: {}", text);
                                    }
                                    match registry.find_processor(&mode_snapshot) {
                                        Some(processor) => {
                                            // Use process_text for realtime (we already have transcription)
                                            // Check for ConversationProcessor first (bidirectional mode)
                                            use crate::processors_conversation::ConversationProcessor;
                                            if let Some(conversation) = processor
                                                .as_any()
                                                .downcast_ref::<ConversationProcessor>(
                                            ) {
                                                eprintln!(
                                                    "[DEBUG MAIN] Using ConversationProcessor"
                                                );
                                                match conversation.process_text(
                                                    &text,
                                                    &mode_snapshot,
                                                    args.debug,
                                                ) {
                                                    Ok(response) => response,
                                                    Err(e) => {
                                                        let error_msg =
                                                            format!("❌ LLM Error: {}", e);
                                                        if tui_state_clone.is_none() {
                                                            eprintln!("{}", error_msg);
                                                        } else {
                                                            // Reset processing status on error
                                                            if let Some(ref state) = tui_state_clone
                                                            {
                                                                if let Ok(mut s) = state.lock() {
                                                                    s.is_processing = false;
                                                                    s.processing_status = crate::tui::ProcessingStatus::Idle;
                                                                    s.last_transcription =
                                                                        error_msg.clone();
                                                                    s.last_transcription_time =
                                                                        Some(Instant::now());
                                                                }
                                                            }
                                                        }
                                                        continue;
                                                    }
                                                }
                                            } else if let Some(assistant) = processor
                                                .as_any()
                                                .downcast_ref::<AssistantProcessor>(
                                            ) {
                                                eprintln!("[DEBUG MAIN] Using AssistantProcessor (fallback)");
                                                match assistant.process_text(
                                                    &text,
                                                    &mode_snapshot,
                                                    args.debug,
                                                ) {
                                                    Ok(response) => {
                                                        // For Command mode, parse JSON and extract command
                                                        if matches!(
                                                            mode_snapshot,
                                                            VoiceMode::Command
                                                        ) {
                                                            parse_command_json(
                                                                &response,
                                                                &text,
                                                                tui_state_clone.as_ref(),
                                                                args.debug,
                                                            )
                                                        } else {
                                                            response
                                                        }
                                                    }
                                                    Err(e) => {
                                                        let error_msg =
                                                            format!("❌ LLM Error: {}", e);
                                                        if tui_state_clone.is_none() {
                                                            eprintln!("{}", error_msg);
                                                        } else {
                                                            // Update TUI with error and reset processing status
                                                            if let Some(ref state) = tui_state_clone
                                                            {
                                                                if let Ok(mut s) = state.lock() {
                                                                    s.is_processing = false;
                                                                    s.processing_status = crate::tui::ProcessingStatus::Idle;
                                                                    s.last_transcription =
                                                                        error_msg;
                                                                    s.last_transcription_time =
                                                                        Some(Instant::now());
                                                                }
                                                            }
                                                        }
                                                        continue;
                                                    }
                                                }
                                            } else {
                                                // Fallback to just transcription
                                                text.clone()
                                            }
                                        }
                                        None => text.clone(),
                                    }
                                }
                                VoiceMode::Dictation => text.clone(),
                            };

                            if tui_state_clone.is_none() {
                                if matches!(mode_snapshot, VoiceMode::Command) {
                                    println!("💻 $ {}", output_text);
                                } else {
                                    println!("📝 {}", output_text);
                                }
                            }

                            // Update TUI with transcription
                            if let Some(ref state) = tui_state_clone {
                                if let Ok(mut s) = state.lock() {
                                    // Log status reset to file
                                    use std::fs::OpenOptions;
                                    use std::io::Write;
                                    let _ = OpenOptions::new()
                                        .create(true)
                                        .append(true)
                                        .open("/tmp/voxtty_status.log")
                                        .and_then(|mut file| {
                                            writeln!(
                                                file,
                                                "[RESET] Output starts with: {}",
                                                &output_text[..50.min(output_text.len())]
                                            )
                                        });

                                    // Update status based on response type
                                    if output_text.starts_with("🔊") {
                                        // TTS response - immediately set status to PlayingAudio
                                        s.is_processing = true;
                                        s.processing_status =
                                            crate::tui::ProcessingStatus::PlayingAudio;
                                    } else {
                                        // Non-TTS response - reset to Idle
                                        s.is_processing = false;
                                        s.processing_status = crate::tui::ProcessingStatus::Idle;
                                    }

                                    // Add to conversation history
                                    use crate::tui::ConversationEntry;
                                    // Strip 🔊 prefix from TTS responses before storing in history
                                    let display_output = if output_text.starts_with("🔊 ") {
                                        output_text.trim_start_matches("🔊 ").to_string()
                                    } else {
                                        output_text.clone()
                                    };
                                    let entry = ConversationEntry {
                                        input: s.last_input.clone(),
                                        output: display_output,
                                        conversation_id: s.current_conversation_id,
                                    };
                                    s.conversation_history.push_back(entry);

                                    // Keep only last 50 entries to prevent memory bloat
                                    while s.conversation_history.len() > 50 {
                                        s.conversation_history.pop_front();
                                    }

                                    // Clear last_input after adding to history to prevent ghosting
                                    // (it's now in conversation_history, no need to keep it in last_input)
                                    s.last_input.clear();

                                    s.last_transcription = output_text.clone();
                                    s.last_transcription_time = Some(Instant::now());
                                    // Clear partial transcription when final transcription is received
                                    s.partial_transcription = None;
                                }
                            }

                            // Type text if not in TUI, or if in TUI with output enabled
                            // (avoid typing into TUI terminal when it has focus)
                            let should_type = if let Some(ref state) = tui_state_clone {
                                state.lock().map(|s| s.output_enabled).unwrap_or(false)
                            } else {
                                // In tray mode (non-TUI), check the tray output_enabled flag
                                *output_enabled_clone.lock().unwrap()
                            };

                            // Don't type if output starts with 🔊 (spoken response)
                            if should_type
                                && !output_text.is_empty()
                                && !output_text.starts_with("🔊")
                            {
                                // Set Writing status for TUI
                                if let Some(ref state) = tui_state_clone {
                                    if let Ok(mut s) = state.lock() {
                                        s.processing_status = crate::tui::ProcessingStatus::Writing;
                                    }
                                }

                                // Use type_command for Command mode (includes Enter press)
                                let type_result = if matches!(mode_snapshot, VoiceMode::Command) {
                                    type_command(&output_text, &config, args.debug)
                                } else {
                                    type_text(&output_text, &config, args.debug)
                                };

                                if let Err(e) = type_result {
                                    if tui_state_clone.is_none() {
                                        eprintln!("❌ Failed to type: {}", e);
                                    }
                                }

                                // Reset processing status after typing completes
                                if let Some(ref state) = tui_state_clone {
                                    if let Ok(mut s) = state.lock() {
                                        s.is_processing = false;
                                        s.processing_status = crate::tui::ProcessingStatus::Idle;
                                    }
                                }
                            }
                        }
                    }
                    TranscriptionEvent::Partial(text) => {
                        // Interrupt TTS on partial transcript (fastest barge-in)
                        if !text.is_empty() {
                            let is_tts_active = *is_tts_speaking.lock().unwrap();
                            if is_tts_active {
                                eprintln!("🛑 Partial speech detected - interrupting TTS!");
                                tts_interrupt.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        }

                        // Update TUI with partial transcription
                        if let Some(ref state) = tui_state_clone {
                            if let Ok(mut s) = state.lock() {
                                s.partial_transcription = if text.is_empty() {
                                    None
                                } else {
                                    Some(text.clone())
                                };
                            }
                        } else if args.debug && !text.is_empty() {
                            // In non-TUI mode, print to stdout in debug mode
                            print!("\r⏳ {}...", text);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                        }
                    }
                    TranscriptionEvent::SpeechStarted => {
                        // Interrupt TTS if user starts speaking (barge-in)
                        let is_tts_active = *is_tts_speaking.lock().unwrap();
                        if is_tts_active {
                            eprintln!("🛑 User started speaking - interrupting AI playback!");
                            tts_interrupt.store(true, std::sync::atomic::Ordering::SeqCst);
                        }

                        // Update TUI VAD state
                        if let Some(ref state) = tui_state_clone {
                            if let Ok(mut s) = state.lock() {
                                s.vad_active = true;

                                if is_tts_active {
                                    // Add interruption indicator to conversation history
                                    let interrupt_entry = crate::tui::ConversationEntry {
                                        input: String::new(),
                                        output: "⚠️  [Interrupted]".to_string(),
                                        conversation_id: s.current_conversation_id,
                                    };
                                    s.conversation_history.push_back(interrupt_entry);

                                    s.is_processing = false;
                                    s.processing_status = crate::tui::ProcessingStatus::Idle;
                                }
                            }
                        }
                        if args.debug {
                            println!("[DEBUG] Speech started");
                        }
                    }
                    TranscriptionEvent::SpeechStopped => {
                        // Update TUI VAD state
                        if let Some(ref state) = tui_state_clone {
                            if let Ok(mut s) = state.lock() {
                                s.vad_active = false;
                            }
                        }
                        if args.debug {
                            println!("[DEBUG] Speech stopped");
                        }
                    }
                    TranscriptionEvent::Error(e) => {
                        if let Some(ref tui_state) = tui_state_clone {
                            let mut state = tui_state.lock().unwrap();
                            state.error_message = Some(format!("Realtime error: {}", e));
                            state.error_timestamp = Some(Instant::now());
                        } else {
                            eprintln!("❌ Realtime error: {}", e);
                        }
                    }
                    TranscriptionEvent::Connecting => {
                        *realtime_status.lock().unwrap() = ConnectionStatus::Connecting;
                        if args.debug {
                            println!("[DEBUG] Realtime connection starting...");
                        }
                        // Update TUI state
                        if let Some(ref tui_state) = tui_state_clone {
                            let mut state = tui_state.lock().unwrap();
                            state.realtime_status = ConnectionStatus::Connecting;
                        }
                        // Update tray to show connecting status
                        if let Some((ref handle, ref counter)) = tray_handle {
                            let mut count = counter.lock().unwrap();
                            *count = count.wrapping_add(1);
                            drop(count);
                            handle.update(|_| {});
                        }
                    }
                    TranscriptionEvent::Connected => {
                        *realtime_status.lock().unwrap() = ConnectionStatus::Connected;
                        if args.debug {
                            println!("[DEBUG] Realtime connection established");
                        }
                        // Update TUI state
                        if let Some(ref tui_state) = tui_state_clone {
                            let mut state = tui_state.lock().unwrap();
                            state.realtime_status = ConnectionStatus::Connected;
                        }
                        // Update tray to show connected status
                        if let Some((ref handle, ref counter)) = tray_handle {
                            let mut count = counter.lock().unwrap();
                            *count = count.wrapping_add(1);
                            drop(count);
                            handle.update(|_| {});
                        }

                        // Play a sound immediately for quick feedback in bidirectional mode
                        if args.bidirectional {
                            sounds::play_resume();
                        }

                        // Speak startup message on first connection (bidirectional mode only)
                        if args.bidirectional {
                            let mut played = startup_message_played_clone.lock().unwrap();
                            if !*played {
                                *played = true;
                                drop(played);

                                // Speak startup confirmation
                                if let Some(ref _key) = elevenlabs_key_for_startup {
                                    let tts = create_elevenlabs_tts(&config);
                                    let startup_message =
                                        "Bidirectional mode activated. I'm ready to assist you.";

                                    if tui_state_clone.is_none() {
                                        println!("🔊 Speaking startup message...");
                                    }

                                    // Run TTS in background thread to not block event loop
                                    let interrupt_for_startup = tts_interrupt.clone();
                                    let is_tts_speaking_startup = is_tts_speaking.clone();
                                    std::thread::spawn(move || {
                                        *is_tts_speaking_startup.lock().unwrap() = true;
                                        let rt = tokio::runtime::Runtime::new().unwrap();
                                        match rt.block_on(tts.speak_and_play_interruptible(
                                            startup_message,
                                            Some(interrupt_for_startup),
                                        )) {
                                            Ok(_) => {
                                                eprintln!("✅ ElevenLabs TTS: Startup message spoken successfully");
                                            }
                                            Err(e) => {
                                                eprintln!("❌ ElevenLabs TTS Error: {}", e);
                                                eprintln!(
                                                    "   Check: 1) ELEVENLABS_API_KEY is valid"
                                                );
                                                eprintln!("          2) Voice ID is correct");
                                                eprintln!("          3) mpv/ffplay/aplay/paplay is installed");
                                            }
                                        }
                                        *is_tts_speaking_startup.lock().unwrap() = false;
                                    });
                                }
                            }
                        }
                    }
                    TranscriptionEvent::Closed => {
                        *realtime_status.lock().unwrap() = ConnectionStatus::Disconnected;
                        // Update TUI state
                        if let Some(ref tui_state) = tui_state_clone {
                            let mut state = tui_state.lock().unwrap();
                            state.realtime_status = ConnectionStatus::Disconnected;
                        }
                        // Update tray to show disconnected status
                        if let Some((ref handle, ref counter)) = tray_handle {
                            let mut count = counter.lock().unwrap();
                            *count = count.wrapping_add(1);
                            drop(count);
                            handle.update(|_| {});
                        }

                        // Only reconnect if not paused
                        let is_paused = *paused_clone.lock().unwrap();
                        if !is_paused {
                            if tui_state_clone.is_none() {
                                println!("🔄 Connection closed, reconnecting...");
                            }
                            // Stop old transcriber and create a new one
                            transcriber.stop();
                            thread::sleep(Duration::from_secs(1)); // Brief delay before reconnect
                            transcriber = RealtimeTranscriber::new(realtime_config.clone());
                            if let Err(e) = transcriber.start() {
                                if tui_state_clone.is_none() {
                                    eprintln!("❌ Failed to reconnect: {}", e);
                                }
                                // Wait longer before retry
                                thread::sleep(Duration::from_secs(5));
                            } else {
                                // Note: Don't set to Connected here - wait for Connected event
                                if tui_state_clone.is_none() {
                                    println!("✅ Reconnected to realtime transcription");
                                }
                            }
                        } else {
                            if tui_state_clone.is_none() {
                                println!("⏸️  Connection closed (paused, not reconnecting)");
                            }
                        }
                    }
                }
            }

            thread::sleep(Duration::from_millis(50));
        }
    }

    if tui_state.is_none() {
        if args.start_paused {
            println!("Listening... Speak and pause for transcription. - ⏸️  PAUSED\n");
            println!("Say 'resume' or 'wake up' to start, or click the tray icon.\n");
        } else {
            println!("Listening... Speak and pause for transcription.\n");
        }
    }

    let mut last_paused = *paused.lock().unwrap();
    let mut last_mode = current_mode.lock().unwrap().clone();

    loop {
        // Check for TUI exit request and sync state
        if let Some(ref state) = tui_state {
            if let Ok(mut s) = state.lock() {
                if s.should_exit {
                    std::process::exit(0);
                }

                // Handle echo test request from TUI
                if s.echo_test_requested {
                    s.echo_test_requested = false;
                    s.echo_test_status = "Recording... Speak now!".to_string();
                    drop(s);

                    // Run echo test
                    let sample_rate = recorder.config.sample_rate.0;
                    let channels = recorder.config.channels;
                    let test_enabled = Arc::new(Mutex::new(true));

                    match recorder.record_until_silence(args.debug, test_enabled.clone()) {
                        Ok(samples) => {
                            if !samples.is_empty() {
                                // Update status
                                if let Ok(mut s) = state.lock() {
                                    s.echo_test_status = "Playing back...".to_string();
                                }

                                // Play back
                                if let Err(e) =
                                    controls::playback_audio(&samples, sample_rate, channels, false)
                                {
                                    if let Ok(mut s) = state.lock() {
                                        s.echo_test_status = format!("Playback failed: {}", e);
                                    }
                                } else {
                                    if let Ok(mut s) = state.lock() {
                                        s.echo_test_status = "Echo test complete!".to_string();
                                    }
                                }

                                // Clear status after 3 seconds
                                let state_clone = state.clone();
                                thread::spawn(move || {
                                    thread::sleep(Duration::from_secs(3));
                                    if let Ok(mut s) = state_clone.lock() {
                                        s.echo_test_status = String::new();
                                    }
                                });
                            } else {
                                if let Ok(mut s) = state.lock() {
                                    s.echo_test_status = "No audio detected".to_string();
                                }
                                // Clear status after 3 seconds
                                let state_clone = state.clone();
                                thread::spawn(move || {
                                    thread::sleep(Duration::from_secs(3));
                                    if let Ok(mut s) = state_clone.lock() {
                                        s.echo_test_status = String::new();
                                    }
                                });
                            }
                        }
                        Err(e) => {
                            if let Ok(mut s) = state.lock() {
                                s.echo_test_status = format!("Echo test failed: {}", e);
                            }
                            // Clear status after 3 seconds
                            let state_clone = state.clone();
                            thread::spawn(move || {
                                thread::sleep(Duration::from_secs(3));
                                if let Ok(mut s) = state_clone.lock() {
                                    s.echo_test_status = String::new();
                                }
                            });
                        }
                    }

                    // Re-acquire lock for rest of the loop
                    continue;
                }

                // Handle device switch request from TUI
                if let Some(new_device_name) = s.device_switch_requested.take() {
                    drop(s); // Drop lock while selecting device

                    if let Ok(new_device) =
                        select_input_device(false, true, Some(new_device_name.clone()))
                    {
                        if let Err(e) = recorder.set_device(new_device) {
                            if args.debug {
                                eprintln!("[DEBUG] Failed to switch device: {}", e);
                            }
                        } else if args.debug {
                            println!("[DEBUG] Switched to device: {}", new_device_name);
                        }
                    }

                    // Re-acquire lock
                    continue;
                }

                drop(s);

                // Sync all state bidirectionally using shared function
                let tray_counter = tray_handle.as_ref().map(|(_, counter)| counter.clone());
                let mut local_enabled = *enabled.lock().unwrap();
                let mut local_output = *output_enabled.lock().unwrap();
                sync_state(
                    &enabled,
                    &paused,
                    &output_enabled,
                    &current_mode,
                    &tui_state,
                    &tray_counter,
                    &mut local_enabled,
                    &mut last_paused,
                    &mut local_output,
                    &mut last_mode,
                );

                // Update tray if state changed
                if let Some((ref handle, _)) = tray_handle {
                    handle.update(|_| {});
                }
            }
        }

        let is_enabled = *enabled.lock().unwrap();

        if !is_enabled {
            thread::sleep(Duration::from_millis(100));
            continue;
        }

        if args.debug {
            println!("[DEBUG] Waiting for speech...");
        }
        match recorder.record_until_silence(args.debug, enabled.clone()) {
            Ok(samples) => {
                if args.debug {
                    println!("[DEBUG] Recorded {} samples", samples.len());
                }
                if samples.is_empty() {
                    continue;
                }

                // Check if still enabled after recording
                let is_enabled = *enabled.lock().unwrap();
                if !is_enabled {
                    if args.debug {
                        println!("[DEBUG] Discarding recording - voice typing was disabled");
                    }
                    continue;
                }

                let tmp_path = PathBuf::from(format!(
                    "/tmp/voice_{}.wav",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_millis()
                ));

                let sample_rate = recorder.config.sample_rate.0;
                let channels = recorder.config.channels;

                if let Err(e) = save_wav(&samples, &tmp_path, sample_rate, channels) {
                    eprintln!("Failed to save audio: {}", e);
                    continue;
                }
                let mode_snapshot = current_mode.lock().unwrap().clone();
                if args.debug {
                    println!("[DEBUG] Saved audio to {:?}", tmp_path);
                    println!("[DEBUG] Current mode: {:?}", mode_snapshot);
                }

                // Set Transcribing status for TUI
                if let Some(ref state) = tui_state {
                    if let Ok(mut s) = state.lock() {
                        s.is_processing = true;
                        s.processing_status = crate::tui::ProcessingStatus::Transcribing;
                    }
                }

                // First, always transcribe the audio
                let transcription_result = transcribe_audio(&tmp_path, backend, &config);

                // Store transcription for later use
                let transcription_text = match &transcription_result {
                    Ok(t) => t.clone(),
                    Err(_) => String::new(),
                };

                // Process audio based on current mode
                let result = match transcription_result {
                    Ok(transcription) => {
                        if args.debug {
                            println!("[DEBUG] Transcription: {}", transcription);
                        }

                        // Always check for pause/resume commands (works in all modes)
                        let (command, should_type) =
                            wake_word_detector.detect_command(&transcription);

                        match command {
                            VoiceCommand::Pause => {
                                *paused.lock().unwrap() = true;
                                sounds::play_pause();
                                if tui_state.is_none() {
                                    println!("⏸️  Paused - say 'resume' or 'wake up' to continue");
                                }
                                // Update TUI state
                                if let Some(ref state) = tui_state {
                                    if let Ok(mut s) = state.lock() {
                                        s.is_paused = true;
                                    }
                                }
                                // Update tray icon to show paused state
                                if let Some((ref handle, ref counter)) = tray_handle {
                                    let mut count = counter.lock().unwrap();
                                    *count = count.wrapping_add(1);
                                    drop(count);
                                    handle.update(|_| {});
                                }
                                continue;
                            }
                            VoiceCommand::Resume => {
                                *paused.lock().unwrap() = false;
                                sounds::play_resume();
                                if tui_state.is_none() {
                                    println!("▶️  Resumed");
                                }
                                // Update TUI state
                                if let Some(ref state) = tui_state {
                                    if let Ok(mut s) = state.lock() {
                                        s.is_paused = false;
                                    }
                                }
                                // Update tray icon to show active state
                                if let Some((ref handle, ref counter)) = tray_handle {
                                    let mut count = counter.lock().unwrap();
                                    *count = count.wrapping_add(1);
                                    drop(count);
                                    handle.update(|_| {});
                                }
                                continue;
                            }
                            VoiceCommand::SwitchMode(mode) => {
                                // Mode switching only works with --assistant or --auto
                                if args.assistant || args.auto {
                                    let mut current = current_mode.lock().unwrap();
                                    *current = mode.clone();
                                    drop(current);
                                    sounds::play_mode_change();

                                    // Update TUI or print to console
                                    if let Some(ref state) = tui_state {
                                        if let Ok(mut s) = state.lock() {
                                            s.mode = mode.clone();
                                            // Clear previous transcription when switching modes
                                            s.last_input.clear();
                                            s.last_transcription.clear();
                                        }
                                    } else {
                                        print_mode_change(&mode, &config.chat_completion_base_url);
                                    }

                                    // Update tray menu if tray is enabled
                                    if let Some((ref handle, ref counter)) = tray_handle {
                                        // Increment counter to force menu rebuild
                                        let mut count = counter.lock().unwrap();
                                        *count = count.wrapping_add(1);
                                        drop(count);

                                        // Trigger tray update
                                        handle.update(|tray| {
                                            // Access the counter to ensure state change is detected
                                            let _counter = tray.update_counter.lock().unwrap();
                                        });
                                    }

                                    continue; // Don't type the wake word
                                }
                            }
                            VoiceCommand::None => {}
                        }

                        // If we get here, no command was detected or mode switch was ignored
                        if !should_type {
                            continue;
                        }

                        // Store raw transcription in TUI state (for assistant/code modes)
                        if let Some(ref state) = tui_state {
                            if let Ok(mut s) = state.lock() {
                                s.last_input = transcription.clone();
                            }
                        }

                        // Process with appropriate processor based on mode
                        let processor =
                            registry.find_processor(&mode_snapshot).ok_or_else(|| {
                                anyhow::anyhow!("No processor for mode {:?}", mode_snapshot)
                            })?;

                        let context = ProcessContext {
                            mode: mode_snapshot.clone(),
                            debug: args.debug,
                        };

                        // Set Thinking status for LLM modes before processing
                        if matches!(
                            mode_snapshot,
                            VoiceMode::Assistant { .. }
                                | VoiceMode::Code { .. }
                                | VoiceMode::Command
                        ) {
                            if let Some(ref state) = tui_state {
                                if let Ok(mut s) = state.lock() {
                                    s.processing_status = crate::tui::ProcessingStatus::Thinking;
                                }
                            }
                        }

                        Ok(processor.process(&tmp_path, &context)?)
                    }
                    Err(e) => Err(e),
                };

                match result {
                    Ok(text) => {
                        // For Command mode, unwrap tool call first, then parse
                        let output_text = if matches!(mode_snapshot, VoiceMode::Command) {
                            // Check if it's a tool call wrapper
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some(tool_name) =
                                    json.get("_voxtty_tool").and_then(|s| s.as_str())
                                {
                                    if tool_name == "process_command" {
                                        // Unwrap args and pass to parse_command_json
                                        if let Some(tool_args) = json.get("args") {
                                            let args_str = tool_args.to_string();
                                            parse_command_json(
                                                &args_str,
                                                &transcription_text,
                                                tui_state.as_ref(),
                                                args.debug,
                                            )
                                        } else {
                                            text.clone()
                                        }
                                    } else if tool_name == "speak" {
                                        // Handle speak tool
                                        if let Some(tool_args) = json.get("args") {
                                            if let Some(speak_text) =
                                                tool_args.get("text").and_then(|t| t.as_str())
                                            {
                                                if !speak_text.is_empty() {
                                                    if args.debug {
                                                        println!(
                                                            "[DEBUG] Speaking response: {}",
                                                            speak_text
                                                        );
                                                    }
                                                    if !config.elevenlabs_api_key.is_empty() {
                                                        let tts = create_elevenlabs_tts(&config);
                                                        let speak_text_owned =
                                                            speak_text.to_string();
                                                        let interrupt_clone = tts_interrupt.clone();
                                                        let is_speaking_clone =
                                                            is_tts_speaking.clone();
                                                        std::thread::spawn(move || {
                                                            *is_speaking_clone.lock().unwrap() =
                                                                true;
                                                            let rt = tokio::runtime::Runtime::new()
                                                                .unwrap();
                                                            if let Err(e) = rt.block_on(
                                                                tts.speak_and_play_interruptible(
                                                                    &speak_text_owned,
                                                                    Some(interrupt_clone),
                                                                ),
                                                            ) {
                                                                eprintln!("TTS Error: {}", e);
                                                            }
                                                            *is_speaking_clone.lock().unwrap() =
                                                                false;
                                                        });
                                                    }
                                                    format!("🔊 {}", speak_text)
                                                } else {
                                                    String::new()
                                                }
                                            } else {
                                                String::new()
                                            }
                                        } else {
                                            String::new()
                                        }
                                    } else {
                                        // Unknown tool
                                        text.clone()
                                    }
                                } else {
                                    // Not a tool wrapper - legacy format or error
                                    parse_command_json(
                                        &text,
                                        &transcription_text,
                                        tui_state.as_ref(),
                                        args.debug,
                                    )
                                }
                            } else {
                                // Not JSON - return as-is
                                text.clone()
                            }
                        } else {
                            // Check if it's a tool call (JSON wrapper)
                            if args.debug {
                                println!("[DEBUG] LLM response: {}", &text[..text.len().min(200)]);
                            }
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some(tool_name) =
                                    json.get("_voxtty_tool").and_then(|s| s.as_str())
                                {
                                    if tool_name == "switch_mode" {
                                        if let Some(tool_args) = json.get("args") {
                                            if let Some(mode_str) =
                                                tool_args.get("mode").and_then(|m| m.as_str())
                                            {
                                                let new_mode = match mode_str {
                                                    "dictation" => VoiceMode::Dictation,
                                                    "assistant" => {
                                                        VoiceMode::Assistant { context: vec![] }
                                                    }
                                                    "code" => VoiceMode::Code { language: None },
                                                    "command" => VoiceMode::Command,
                                                    _ => VoiceMode::Dictation,
                                                };

                                                // Update mode
                                                let mut m = current_mode.lock().unwrap();
                                                *m = new_mode.clone();
                                                drop(m);

                                                // Speak confirmation if provided
                                                if let Some(confirmation) = tool_args
                                                    .get("confirmation")
                                                    .and_then(|c| c.as_str())
                                                {
                                                    if !config.elevenlabs_api_key.is_empty() {
                                                        let tts = create_elevenlabs_tts(&config);
                                                        let conf_text = confirmation.to_string();
                                                        let interrupt_clone = tts_interrupt.clone();
                                                        let is_speaking_clone =
                                                            is_tts_speaking.clone();
                                                        std::thread::spawn(move || {
                                                            *is_speaking_clone.lock().unwrap() =
                                                                true;
                                                            let rt = tokio::runtime::Runtime::new()
                                                                .unwrap();
                                                            if let Err(e) = rt.block_on(
                                                                tts.speak_and_play_interruptible(
                                                                    &conf_text,
                                                                    Some(interrupt_clone),
                                                                ),
                                                            ) {
                                                                eprintln!("TTS Error: {}", e);
                                                            }
                                                            *is_speaking_clone.lock().unwrap() =
                                                                false;
                                                        });
                                                    }
                                                    format!("🔊 {}", confirmation)
                                                } else {
                                                    format!("🔊 Switched to {} mode", mode_str)
                                                }
                                            } else {
                                                String::new()
                                            }
                                        } else {
                                            String::new()
                                        }
                                    } else if tool_name == "speak" {
                                        if let Some(tool_args) = json.get("args") {
                                            if let Some(speak_text) =
                                                tool_args.get("text").and_then(|t| t.as_str())
                                            {
                                                if !speak_text.is_empty() {
                                                    if args.debug {
                                                        println!(
                                                            "[DEBUG] Speaking response: {}",
                                                            speak_text
                                                        );
                                                    }

                                                    // Speak the response using ElevenLabsTts
                                                    if !config.elevenlabs_api_key.is_empty() {
                                                        let tts = create_elevenlabs_tts(&config);
                                                        let speak_text_owned =
                                                            speak_text.to_string();
                                                        let is_tts_speaking_clone =
                                                            is_tts_speaking.clone();
                                                        let interrupt_clone = tts_interrupt.clone();

                                                        std::thread::spawn(move || {
                                                            // Set flag to true before speaking
                                                            *is_tts_speaking_clone
                                                                .lock()
                                                                .unwrap() = true;

                                                            let rt = tokio::runtime::Runtime::new()
                                                                .unwrap();
                                                            if let Err(e) = rt.block_on(
                                                                tts.speak_and_play_interruptible(
                                                                    &speak_text_owned,
                                                                    Some(interrupt_clone),
                                                                ),
                                                            ) {
                                                                eprintln!("TTS Error: {}", e);
                                                            }

                                                            // Set flag back to false after speaking
                                                            *is_tts_speaking_clone
                                                                .lock()
                                                                .unwrap() = false;
                                                        });
                                                    } else {
                                                        eprintln!("TTS request ignored: ELEVENLABS_API_KEY not set");
                                                    }
                                                }
                                            }
                                        }
                                        // Don't output text for typing, but return text for TUI display
                                        if let Some(tool_args) = json.get("args") {
                                            if let Some(speak_text) =
                                                tool_args.get("text").and_then(|t| t.as_str())
                                            {
                                                format!("🔊 {}", speak_text)
                                            } else {
                                                String::new()
                                            }
                                        } else {
                                            String::new()
                                        }
                                    } else {
                                        // Unknown tool, treat as dictation (not JSON output)
                                        text.clone()
                                    }
                                } else {
                                    // Not a tool wrapper - this is regular dictation text
                                    // In Assistant mode, if the LLM returns plain text without tool wrapper,
                                    // it's the corrected dictation that should be typed
                                    text.clone()
                                }
                            } else {
                                // Not JSON at all - plain dictation text
                                text.clone()
                            }
                        };

                        // Update TUI or print to console
                        if let Some(ref state) = tui_state {
                            if let Ok(mut s) = state.lock() {
                                // Add to conversation history
                                use crate::tui::ConversationEntry;
                                // Strip 🔊 prefix from TTS responses before storing in history
                                let display_output = if output_text.starts_with("🔊 ") {
                                    output_text.trim_start_matches("🔊 ").to_string()
                                } else {
                                    output_text.clone()
                                };
                                let entry = ConversationEntry {
                                    input: s.last_input.clone(),
                                    output: display_output,
                                    conversation_id: s.current_conversation_id,
                                };
                                s.conversation_history.push_back(entry);

                                // Keep only last 50 entries to prevent memory bloat
                                while s.conversation_history.len() > 50 {
                                    s.conversation_history.pop_front();
                                }

                                // Update status based on response type
                                if output_text.starts_with("🔊") {
                                    // TTS response - immediately set status to PlayingAudio
                                    s.is_processing = true;
                                    s.processing_status =
                                        crate::tui::ProcessingStatus::PlayingAudio;
                                } else {
                                    // Non-TTS response - reset to Idle
                                    s.is_processing = false;
                                    s.processing_status = crate::tui::ProcessingStatus::Idle;
                                }

                                // In dictation/command mode, clear last_input since input = output
                                if matches!(
                                    mode_snapshot,
                                    VoiceMode::Dictation | VoiceMode::Command
                                ) {
                                    s.last_input.clear();
                                }
                                // Only update if not already set by parse_command_json (for rejections)
                                if !matches!(mode_snapshot, VoiceMode::Command)
                                    || !output_text.is_empty()
                                {
                                    s.last_transcription = output_text.clone();
                                    s.last_transcription_time = Some(Instant::now());
                                }
                            }
                        } else if matches!(mode_snapshot, VoiceMode::Command) {
                            if !output_text.is_empty() {
                                println!("💻 $ {}", output_text);
                            }
                        } else {
                            println!("{}", output_text);
                        }

                        // Type text if not in TUI, or if in TUI with output enabled
                        // (avoid typing into TUI terminal when it has focus)
                        let should_type = if let Some(ref state) = tui_state {
                            state.lock().map(|s| s.output_enabled).unwrap_or(false)
                        } else {
                            // In tray mode (non-TUI), check the tray output_enabled flag
                            *output_enabled.lock().unwrap()
                        };

                        if should_type && !output_text.is_empty() && !output_text.starts_with("🔊")
                        {
                            // Set Writing status for TUI
                            if let Some(ref state) = tui_state {
                                if let Ok(mut s) = state.lock() {
                                    s.processing_status = crate::tui::ProcessingStatus::Writing;
                                }
                            }

                            let type_result = if matches!(mode_snapshot, VoiceMode::Command) {
                                type_command(&output_text, &config, args.debug)
                            } else {
                                type_text(&output_text, &config, args.debug)
                            };

                            if let Err(e) = type_result {
                                if tui_state.is_none() {
                                    eprintln!("Failed to type text: {}", e);
                                }
                            }

                            // Reset processing status after typing completes
                            if let Some(ref state) = tui_state {
                                if let Ok(mut s) = state.lock() {
                                    s.is_processing = false;
                                    s.processing_status = crate::tui::ProcessingStatus::Idle;
                                }
                            }
                        } else {
                            // No typing needed - reset status immediately
                            if let Some(ref state) = tui_state {
                                if let Ok(mut s) = state.lock() {
                                    s.is_processing = false;
                                    s.processing_status = crate::tui::ProcessingStatus::Idle;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("❌ Error: {}", e);

                        // Update TUI with error or print to console
                        if let Some(ref state) = tui_state {
                            if let Ok(mut s) = state.lock() {
                                s.last_transcription = error_msg.clone();
                                s.last_transcription_time = Some(Instant::now());
                                // Reset processing status on error
                                s.is_processing = false;
                                s.processing_status = crate::tui::ProcessingStatus::Idle;
                            }
                        } else {
                            eprintln!("\n{}", error_msg);
                        }

                        // Only show troubleshooting in non-TUI mode
                        if tui_state.is_none() {
                            match backend {
                                Backend::Speaches => {
                                    eprintln!(
                                        "   Backend: Speaches ({})",
                                        config.speaches_base_url
                                    );
                                    eprintln!("   Troubleshooting:");
                                    eprintln!(
                                    "   • Check if Speaches is running: docker ps | grep speaches"
                                );
                                    eprintln!(
                                        "   • Test connection: curl {}/health",
                                        config
                                            .speaches_base_url
                                            .trim_end_matches("/v1/audio/transcriptions")
                                    );
                                    eprintln!("   • View logs: docker logs speaches");
                                }
                                Backend::OpenAI => {
                                    eprintln!(
                                        "   Backend: OpenAI Whisper ({})",
                                        config.transcription_url
                                    );
                                    eprintln!("   Troubleshooting:");
                                    eprintln!("   • Check your API key: echo $OPENAI_API_KEY");
                                    eprintln!("   • Check internet connection");
                                    eprintln!("   • Verify API key has access to Whisper API");
                                }
                                Backend::WhisperCpp => {
                                    eprintln!("   Backend: whisper.cpp ({})", config.whisper_url);
                                    eprintln!("   Troubleshooting:");
                                    eprintln!("   • Check if whisper.cpp server is running");
                                    eprintln!("   • Test connection: curl {}", config.whisper_url);
                                    eprintln!("   • Restart server: ./server -l en -m models/ggml-tiny.en.bin --port 7777 --convert");
                                }
                            }
                            eprintln!();
                        } // end if tui_state.is_none()
                    }
                }

                let _ = std::fs::remove_file(&tmp_path);
            }
            Err(e) => {
                eprintln!("Recording failed: {}", e);
            }
        }
    }
}
