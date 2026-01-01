# Changelog

All notable changes to voxtty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Notification sounds for pause/resume/mode changes (cross-platform using rodio)
- Auto-reconnection for WebSocket when connection drops
- Connection status in tray tooltip (`[Disconnected]` indicator)
- Reconnection on re-enable after disable via tray

### Changed
- Reduced notification sound volume from 30% to 10%
- Updated README with realtime mode documentation and marketing copy

## [0.1.0] - 2024-12-09

### Added
- **Realtime WebSocket streaming** (~150ms latency)
  - Support for Speaches, ElevenLabs, and OpenAI realtime APIs
  - `--realtime` flag to enable streaming mode
  - `--elevenlabs` and `--openai` provider flags
- **Voice command system**
  - Pause/resume: "pause", "go to sleep", "resume", "wake up"
  - Mode switching: "dictation mode", "assistant mode", "code mode"
  - `--auto` flag for voice commands without full assistant mode
- **Mode-indicating tray icon**
  - Letter indicator: D (Dictation), A (Assistant), C (Code)
  - Color coding: Green (Dictation), Blue (Assistant), Purple (Code), Orange (Paused), Gray (Disabled)
- **Assistant mode** with LLM integration
  - Wake word activation ("hey assistant")
  - Code generation mode
  - Support for Ollama, OpenAI, Anthropic, Google, DeepSeek, OpenRouter
  - Interactive model selection (`--select-model`)
  - Privacy warnings for cloud AI services
- **Cross-platform foundations**
  - tray-icon and enigo for macOS/Windows support
  - ksni fallback for Linux
- Privacy-focused offline voice-to-text with Whisper AI
- Real-time speech-to-text conversion with WebRTC VAD
- System tray icon with toggle control
- System-wide typing via ydotool
- Support for whisper.cpp and Speaches API backends
- Echo test mode for audio verification (`--echo-test`)
- Interactive device selection (`--select-device`)
- Debug mode (`--debug`)
- Systemd user service support

### Backend Support
- **whisper.cpp** - Local C++ server
- **Speaches** - Self-hosted Docker API (recommended)
- **OpenAI** - Cloud transcription and realtime
- **ElevenLabs** - Cloud realtime transcription

## Links

- [Repository](https://github.com/jflaflamme/voxtty)
- [Issues](https://github.com/jflaflamme/voxtty/issues)
- [Releases](https://github.com/jflaflamme/voxtty/releases)
