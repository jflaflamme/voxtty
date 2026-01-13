# voxtty

**Voice assistant that listens on Linux — say 'code mode' to switch, run local or cloud, type system-wide**

voxtty is more than dictation — it's a voice-controlled assistant for Linux. Switch modes hands-free with wake words ("hey assistant", "code mode"), choose your backend (local Whisper or cloud AI), and type anywhere. System tray control, realtime streaming, and complete privacy when offline.

Built in Rust for reliability. Designed for developers who value control.

## 🚀 Quick Start

### Installation Method 1: Debian Package (Recommended)

```bash
# 1. Install voxtty
sudo dpkg -i voxtty_0.1.0-1_amd64.deb

# 2. Configure environment (add to ~/.bashrc for persistence)
export YDOTOOL_SOCKET=/tmp/.ydotool_socket
export SPEACHES_BASE_URL=http://localhost:8000/v1/audio/transcriptions

# 3. Setup ydotool
sudo systemctl enable --now ydotool.service

# 4. Start Speaches backend (Docker)
docker run -d --name speaches -p 8000:8000 \
  ghcr.io/speaches-ai/speaches:latest

# 5. Test your microphone (IMPORTANT!)
# After installing the .deb package, run voxtty from the command line:
voxtty --echo-test

# 6. Start voice typing
voxtty --speaches --tray
```

### Installation Method 2: Standalone Binary (Development)

```bash
# 1. Build voxtty
cargo build --release

# 2. Set environment (per-session)
export YDOTOOL_SOCKET=/tmp/.ydotool_socket

# 3. Setup backend and ydotool (same as above)

# 4. Run from build directory
./target/release/voxtty --echo-test
./target/release/voxtty --speaches --tray
```

**That's it!** Click the tray icon to toggle voice typing on/off.

### Optional: AI Assistant Mode

```bash
# 1. Select an AI model (interactive)
voxtty --select-model
# Choose from: OpenAI, Anthropic, Google, DeepSeek, Ollama (local/free), OpenRouter

# 2. Use assistant mode with wake words OR tray menu
voxtty --assistant --tray
# Voice commands: Say "hey assistant" for writing help, "code mode" for code
# GUI: Click tray icon → Select mode (Dictation/Assistant/Code)
# Note: Privacy warnings shown automatically when using cloud AI (OpenAI, etc.)
```

📖 **For detailed configuration, model selection, and troubleshooting**, see the sections below in this README.

## 🎯 Inspiration & Evolution

voxtty was inspired by [themanyone/voice_typing](https://github.com/themanyone/voice_typing), a brilliant bash-based voice typing solution. While the original project demonstrated the power of offline voice typing, voxtty takes it further by:

- **Rewritten in Rust** - Memory-safe, fast, and reliable
- **Speaches Backend** - Uses [Speaches AI](https://github.com/speaches-ai/speaches) for a more extensible and maintainable transcription backend
- **Better Performance** - ~2 seconds latency on i7 CPU (no GPU required) with basic model
- **Enhanced UX** - System tray integration for seamless control
- **Production Ready** - Proper error handling, device selection, and configuration options

### Why Speaches?

[Speaches](https://github.com/speaches-ai/speaches) provides a superior backend option compared to direct whisper.cpp integration:

- **OpenAI-Compatible API** - Standard REST interface for easy integration
- **Docker Support** - Run in containers with consistent environments
- **CPU Optimized** - Excellent performance even without GPU (~2s latency on i7)
- **Model Flexibility** - Easy switching between different Whisper model sizes
- **Network Ready** - Can run locally or on a dedicated transcription server
- **Better Extensibility** - Clean API makes it easy to add features and improvements

### Backend Comparison

| Feature | whisper.cpp | Speaches | Realtime (WebSocket) |
|---------|-------------|----------|----------------------|
| **Setup** | Manual build | Docker one-liner | API key or self-hosted |
| **Latency** | ~3-4s | ~2s | **~150ms** |
| **Privacy** | 100% offline | 100% offline | Depends on provider |
| **Providers** | Local only | Local only | Speaches, ElevenLabs, OpenAI |
| **Best For** | Minimal setup | Production use | **Lowest latency** |

**Realtime Providers:**
- **Speaches** - Self-hosted, free, ~150ms latency, 🔒 **100% LOCAL** (privacy-preserving)
- **ElevenLabs** - Cloud, excellent accuracy, requires API key, ☁️ **CLOUD** (sends audio to third-party)
- **OpenAI** - Cloud, GPT-4o transcription, requires API key, ☁️ **CLOUD** (sends audio to third-party)

## ✨ Features

### 🔒 Privacy First
- **100% Offline Processing** - All transcription happens locally using Whisper AI
- **No Cloud Services** - Your voice never leaves your machine (for dictation mode)
- **Privacy Warnings** - Automatic alerts when using cloud AI services (Assistant/Code modes)
- **No Data Collection** - Zero telemetry, zero tracking
- **Self-Hosted** - Run Speaches backend in Docker on your own hardware
- **Network Isolated** - Works completely offline, no internet required
- **Local AI Option** - Use Ollama for 100% offline Assistant/Code modes

### ⚡ Realtime Streaming Mode
- **~150ms Latency** - WebSocket-based streaming for near-instant transcription
- **Multiple Providers** - ElevenLabs, OpenAI Realtime, or Speaches WebSocket
- **Auto-Reconnection** - Automatically reconnects if connection drops
- **Connection Status** - Tray tooltip shows `[Disconnected]` when offline

### 🎤 Smart Voice Detection
- **Voice Activity Detection (VAD)** - Automatically detects when you start and stop speaking
- **WebRTC VAD Engine** - Industry-standard voice detection with low false positives
- **Amplitude Threshold** - Dual detection system for reliable speech capture
- **Configurable Silence Detection** - Customizable pause duration before transcription

### ⌨️ System-Wide Integration
- **Universal Typing** - Works in any application via ydotool
- **No GUI Required** - Runs in TTY, X11, Wayland, or any Linux environment
- **Instant Text Insertion** - Transcribed text appears directly where you're typing

### 🎛️ Flexible Control
- **System Tray Icon** - Quick toggle on/off with visual status indicator (click to enable/disable)
- **GUI Mode Switching** - Switch between Dictation/Assistant/Code modes from tray menu (when `--assistant` enabled)
- **Voice Commands** - Wake words for hands-free mode switching ("hey assistant", "code mode", "dictation mode")
- **Audio Feedback** - Notification sounds for pause/resume/mode changes
- **Multiple Backends** - Support for whisper.cpp, Speaches, OpenAI, or ElevenLabs
- **Interactive Device Selection** - Choose your preferred microphone
- **Always Available** - Runs in background, ready when you need it

### 🔧 Developer Friendly
- **Echo Test Mode** - Built-in `--echo-test` CLI flag to verify audio input with instant playback
- **Debug Mode** - Detailed logging for troubleshooting with `--debug` flag
- **Flexible Configuration** - Environment variables and CLI flags for easy customization
- **Backend Agnostic** - Switch backends with a single flag, no code changes
- **Clean Rust Codebase** - Modern, safe, and maintainable

## 📋 Requirements

### Core Dependencies
- **Whisper AI Backend** (choose one):
  - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Lightweight C++ server
  - [Speaches](https://github.com/speaches-ai/speaches) - Modern API server (recommended)
- **Audio**: ALSA (libasound2-dev)
- **System Integration**: ydotool (for typing), DBus (for system tray)
- **Build Tools**: Rust 1.70+, cargo, pkg-config

### Optional: AI Assistant Mode
- **LLM Provider** (choose one):
  - [Ollama](https://ollama.com/) - Free, local AI models (recommended for privacy)
  - [OpenAI](https://openai.com/) - GPT-4o, GPT-4o-mini (API key required)
  - [Anthropic](https://anthropic.com/) - Claude models (API key required)
  - [Google](https://ai.google.dev/) - Gemini models (API key required)
  - [DeepSeek](https://deepseek.com/) - Affordable models (API key required)
  - [OpenRouter](https://openrouter.ai/) - Access multiple providers (API key required)

### Runtime Dependencies
- `ydotool` - System-wide input simulation
- `alsa-utils` - Audio utilities
- `pulseaudio` or `pipewire-pulse` - Audio server (recommended)

## 🚀 Installation

### Choose Your Installation Method

| Method | Best For | Binary Location | Configuration |
|--------|----------|-----------------|---------------|
| **Debian Package** | End users, production | `/usr/bin/voxtty` | Add to `~/.bashrc` (persistent) |
| **Standalone Binary** | Development, testing | `./target/release/voxtty` | Export per-session |

### Option 1: Debian Package (Recommended for End Users)

**Pros**: System-wide installation, managed dependencies, easy updates  
**Cons**: Requires sudo, system-wide changes

```bash
# Download the latest release
wget https://github.com/jflaflamme/voxtty/releases/latest/download/voxtty_0.1.0-1_amd64.deb

# Install
sudo dpkg -i voxtty_0.1.0-1_amd64.deb

# Install dependencies if needed
sudo apt-get install -f

# Binary is now at /usr/bin/voxtty
which voxtty

# IMPORTANT: After installation, you must run voxtty from the command line
# It does not create a desktop launcher - it's a CLI tool
voxtty --help
```

**Configuration**: Add environment variables to `~/.bashrc` for persistence:
```bash
echo 'export YDOTOOL_SOCKET=/tmp/.ydotool_socket' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Standalone Binary (Recommended for Development)

**Pros**: No sudo needed, easy to update, isolated from system
**Cons**: Manual dependency management, per-session config

```bash
# Clone the repository
git clone https://github.com/jflaflamme/voxtty.git
cd voxtty

# Build release binary
cargo build --release

# Install to /usr/local/bin
sudo cp target/release/voxtty /usr/local/bin/

# Verify installation
voxtty --version
```

**Configuration**: Export environment variables per-session:
```bash
export YDOTOOL_SOCKET=/tmp/.ydotool_socket
voxtty --echo-test
```

### Option 3: Systemd User Service (Auto-start on Login)

For automatic startup with realtime transcription:

```bash
# 1. Install binary
sudo cp target/release/voxtty /usr/local/bin/

# 2. Create environment file with API keys
mkdir -p ~/.config/voxtty
cat > ~/.config/voxtty/env << 'EOF'
ELEVENLABS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
EOF
chmod 600 ~/.config/voxtty/env

# 3. Create systemd service
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/voxtty.service << 'EOF'
[Unit]
Description=voxtty voice typing
After=graphical-session.target

[Service]
EnvironmentFile=%h/.config/voxtty/env
ExecStart=/usr/local/bin/voxtty --tray --auto --realtime --elevenlabs
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# 4. Enable and start
systemctl --user daemon-reload
systemctl --user enable --now voxtty

# 5. Check status
systemctl --user status voxtty
```

**Service management:**
```bash
systemctl --user status voxtty    # Check status
systemctl --user stop voxtty      # Stop
systemctl --user restart voxtty   # Restart
journalctl --user -u voxtty -f    # Watch logs
```

### Optional: Desktop Entry (App Menu)

```bash
cat > ~/.local/share/applications/voxtty.desktop << 'EOF'
[Desktop Entry]
Name=voxtty
Comment=Voice to text typing
Exec=voxtty --tray --auto
Icon=audio-input-microphone
Terminal=false
Type=Application
Categories=Utility;Accessibility;
EOF
```

### Option 4: Build Your Own Debian Package

**For**: Creating custom packages or contributing

#### Prerequisites

```bash
# Install build dependencies
sudo apt install debhelper-compat cargo rustc pkg-config libasound2-dev libdbus-1-dev
```

#### Build Process

The Debian package is built using standard Debian packaging tools:

```bash
# Method 1: Use the build script (recommended)
./packaging/scripts/build-deb.sh

# Method 2: Manual build (same as the script)
cargo build --release
dpkg-buildpackage -rfakeroot -us -uc

# The package will be created in the parent directory
ls -lh ../voxtty_*.deb
```

#### What Gets Built

The build process creates:
- **Binary package**: `voxtty_0.1.0-1_amd64.deb` - Main installable package
- **Debug symbols**: `voxtty-dbgsym_0.1.0-1_amd64.deb` - Debug symbols (optional)
- **Build artifacts**: `.buildinfo`, `.changes` files - Build metadata

#### Package Contents

```
/usr/bin/voxtty              # Main executable
/usr/share/doc/voxtty/       # Documentation
/usr/share/man/man1/            # Man pages (if included)
```

#### Install Your Custom Package

```bash
# Install the package
sudo dpkg -i ../voxtty_*.deb

# If dependencies are missing, fix them
sudo apt-get install -f

# Verify installation
which voxtty
voxtty --version
```

#### Debian Packaging Files

The package is configured via files in the `debian/` directory:

- **`debian/control`** - Package metadata, dependencies, and description
- **`debian/rules`** - Build instructions (uses dh with Cargo)
- **`debian/changelog`** - Version history and release notes
- **`debian/install`** - Files to include in the package
- **`debian/compat`** - Debhelper compatibility level

#### Customizing the Package

To modify the package:

1. **Change version**: Edit `debian/changelog`
   ```bash
   dch -v 0.2.0-1 "New release with feature X"
   ```

2. **Add dependencies**: Edit `debian/control`
   ```
   Depends: ${shlibs:Depends}, ${misc:Depends}, your-new-dependency
   ```

3. **Modify build**: Edit `debian/rules`
   ```makefile
   override_dh_auto_build:
       cargo build --release --features your-feature
   ```

4. **Rebuild package**:
   ```bash
   ./packaging/scripts/build-deb.sh
   ```

#### Troubleshooting Package Build

**Build fails with missing dependencies?**
```bash
sudo apt-get build-dep .
```

**Clean build artifacts?**
```bash
cargo clean
debian/rules clean
rm -f ../voxtty_*
```

**Test package without installing?**
```bash
dpkg-deb --contents ../voxtty_*.deb
dpkg-deb --info ../voxtty_*.deb
```

## ⚙️ Setup

### 1. Install Whisper Backend

#### Option A: whisper.cpp (Recommended for local use)

```bash
# Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download a model (tiny.en is fastest, small.en is more accurate)
bash ./models/download-ggml-model.sh tiny.en

# Start the server
./server -l en -m models/ggml-tiny.en.bin --port 7777 --convert
```

#### Option B: Speaches API (Recommended - Better Performance & Extensibility)

Speaches provides superior performance and flexibility compared to whisper.cpp. On an i7 CPU (no GPU), expect ~2 second latency with the basic model.

```bash
# Quick start with Docker (CPU-only)
docker run -d \
  --name speaches \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/speaches-ai/speaches:latest

# Or with docker-compose
cat > docker-compose.yml <<EOF
services:
  speaches:
    image: ghcr.io/speaches-ai/speaches:latest
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - TRANSCRIPTION_MODEL_ID=Systran/faster-distil-whisper-small.en
EOF

docker-compose up -d

# IMPORTANT: Initial Speaches Configuration
# After starting Speaches for the first time, you must configure it:

# 1. Set the base URL for Speaches API
export SPEACHES_BASE_URL="http://localhost:8000"

# 2. Check available models in the registry
curl "$SPEACHES_BASE_URL/v1/registry"

# 3. Download and activate your chosen model (first-time setup)
curl "$SPEACHES_BASE_URL/v1/models/Systran/faster-distil-whisper-small.en" -X POST

# 4. Configure voxtty to use Speaches
export SPEACHES_BASE_URL="http://localhost:8000/v1/audio/transcriptions"
export TRANSCRIPTION_MODEL_ID="Systran/faster-distil-whisper-small.en"

# 5. Test the connection (requires a test audio file)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "model=Systran/faster-distil-whisper-small.en"
```

**Initial Setup Notes:**
- The `/v1/registry` endpoint shows all available Whisper models
- The `/v1/models/{model_name}` POST endpoint downloads and activates a model
- Model download happens once; subsequent starts use the cached model
- Choose model size based on your needs: `tiny.en` (fastest) → `small.en` (balanced) → `medium.en` (accurate)

**Performance Notes:**
- **CPU-only (i7)**: ~2 seconds latency with basic model
- **GPU-enabled**: Sub-second latency possible
- **Model sizes**: tiny.en (fastest) → small.en (balanced) → medium.en (accurate)
- **100% Offline**: No internet required after initial model download

### 2. Configure ydotool

```bash
# Add to ~/.bashrc or ~/.zshrc
export YDOTOOL_SOCKET=/tmp/.ydotool_socket

# Enable and start ydotool service
sudo systemctl enable ydotool.service
sudo systemctl start ydotool.service

# Verify it's running
sudo systemctl status ydotool.service

# Test typing
ydotool type "Hello, World!"
```

### 3. Verify Audio Input (Important!)

Before using voxtty, verify your microphone works correctly with the built-in echo test:

```bash
# Run echo test - speak and hear your voice played back
voxtty --echo-test

# Select specific device interactively, then test
voxtty --select-device --echo-test

# Test with debug output to see audio levels
voxtty --echo-test --debug
```

**Echo Test Mode**: Speak into your microphone, pause, and you'll hear your recording played back. This confirms:
- ✅ Microphone is working
- ✅ Audio levels are correct
- ✅ VAD (Voice Activity Detection) is triggering properly
- ✅ No audio driver issues

## 🎯 Usage

### Basic Usage

```bash
# Start with default settings (whisper.cpp backend)
voxtty

# Start with system tray icon
voxtty --tray

# Use Speaches API backend
voxtty --speaches

# Enable debug output
voxtty --debug
```

### Advanced Usage

```bash
# Interactive device selection with debug output
voxtty --select-device --debug

# Echo test with specific device
voxtty --select-device --echo-test

# Speaches API with tray control
voxtty --speaches --tray
```

### Realtime Streaming Mode

For the lowest latency (~150ms), use realtime WebSocket streaming:

```bash
# Realtime with Speaches (self-hosted, free)
voxtty --realtime --speaches --tray

# Realtime with ElevenLabs (cloud, requires API key)
export ELEVENLABS_API_KEY=your_key_here
voxtty --realtime --elevenlabs --tray

# Realtime with OpenAI (cloud, requires API key)
export OPENAI_API_KEY=your_key_here
voxtty --realtime --openai --tray

# Realtime with voice commands (pause/resume/mode switching)
voxtty --realtime --speaches --auto --tray
```

**Realtime Features:**
- Audio feedback sounds for pause/resume/mode changes
- Auto-reconnects if WebSocket connection drops
- Tray tooltip shows connection status
- Voice commands work continuously (no need to wait for silence)

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--echo-test` | **Run audio echo test** - Speak and hear playback to verify microphone |
| `--select-device` | Interactively choose audio input device |
| `--debug` | Enable detailed debug logging (shows VAD triggers, audio levels) |
| `--speaches` | **Use Speaches backend** instead of whisper.cpp (default) |
| `--tray` | Enable system tray icon with click-to-toggle control |
| `--tui` | **Enable Terminal UI (TUI) mode** - Full-screen terminal interface |
| `--realtime` | **Enable realtime WebSocket streaming** (~150ms latency) |
| `--elevenlabs` | Use ElevenLabs for realtime transcription (requires API key) |
| `--openai` | Use OpenAI for transcription |
| `--assistant` | Enable assistant modes with wake word activation |
| `--auto` | Enable voice commands without full assistant mode |

**Configuration Priority**: CLI flags → Environment variables → Defaults

### Terminal UI (TUI) Mode

Launch voxtty with a consolidated single-screen dashboard:

```bash
# Launch TUI in demo mode
voxtty --tui

# TUI with specific backend
voxtty --tui --speaches
voxtty --tui --realtime --elevenlabs
```

**Single Dashboard View:**
```
┌─────────────────────────────────────────────────┐
│ voxtty | Dictation | Speaches | LISTENING       │  ← Status bar
├──────────────────┬──────────────────────────────┤
│ Live Audio       │ Configuration                │
│ ████████         │ Model: GPT-4o-mini           │
│ VAD: ● ACTIVE    │ [m] Select Model             │
│ Device: Default  │ [d] Select Device            │
├──────────────────┴──────────────────────────────┤
│ Last Transcription (5s ago)                     │
│ "testing one two three..."                      │
├──────────────────┬──────────────────────────────┤
│ Mode Selection   │ Actions                      │
│ [1] ▶ Dictation  │ [p] Pause                    │
│ [2]   Assistant  │ [e] Echo Test                │
│ [3]   Code       │                              │
└──────────────────┴──────────────────────────────┘
[q]Quit  [?]Help  [1-3]Mode  [p]Pause  [e]Echo
```

**Everything At a Glance:**
- **Live audio visualization** - Real-time voice level bar graph
- **VAD indicator** - Voice Activity Detection status (● ACTIVE / ○ Inactive)
- **Last transcription** - Most recent text with timestamp
- **Mode switcher** - Quick [1-3] keys to switch modes
- **Quick actions** - One-key access to echo test, pause, device selection
- **Model info** - Current AI model configuration

**Keyboard Shortcuts:**
- `1-3` - Switch mode (Dictation/Assistant/Code)
- `p` or `Space` - Pause/Resume listening
- `e` - Run echo test
- `m` - Select AI model
- `d` - Select audio device
- `?` or `h` - Toggle help screen
- `q` or `Esc` - Quit

**No Navigation Needed** - All controls visible on one screen!

## 🎮 Controls

### System Tray Icon

The tray icon shows a colored circle with a letter indicating the current mode:

| Icon | Color | Meaning |
|------|-------|---------|
| **D** | 🟢 Green | Dictation mode (active) |
| **A** | 🔵 Blue | Assistant mode (active) |
| **C** | 🟣 Purple | Code mode (active) |
| **D/A/C** | 🟠 Orange | Paused (listening for "resume") |
| **D/A/C** | ⚫ Gray | Disabled (click to enable) |

- **Left Click** - Toggle voice typing on/off
- **Right Click** - Menu to switch modes (when `--assistant` or `--auto` enabled)

### Voice Commands

Wake words for hands-free control (requires `--auto` or `--assistant` flag):

| Command | Wake Words |
|---------|------------|
| **Dictation Mode** | "dictation mode", "normal mode", "typing mode", "type mode" |
| **Assistant Mode** | "hey assistant", "assistant mode" |
| **Code Mode** | "code mode", "coding mode", "write code" |
| **Pause** | "pause", "stop listening", "go to sleep" |
| **Resume** | "resume", "start listening", "wake up" |

## 🔧 Configuration

voxtty uses a **layered configuration system** for maximum flexibility:

### Configuration Layers (Priority Order)

```
1. CLI Flags (highest)
2. Environment Variables
3. Config File (~/.config/voxtty/config.toml)
4. Auto-Detection (ydotool socket)
5. Built-in Defaults (lowest)
```

### Config File

voxtty automatically creates `~/.config/voxtty/config.toml` on first run:

```toml
# ydotool socket path (auto-detected if not specified)
ydotool_socket = "/run/user/1000/.ydotool_socket"

# Speaches backend
speaches_base_url = "http://localhost:8000/v1/audio/transcriptions"
transcription_model_id = "Systran/faster-distil-whisper-small.en"

# whisper.cpp backend
whisper_url = "http://127.0.0.1:7777/inference"
```

### Environment Variables (Override Config File)

```bash
export YDOTOOL_SOCKET=/run/user/$(id -u)/.ydotool_socket
export SPEACHES_BASE_URL=http://localhost:8000/v1/audio/transcriptions
export TRANSCRIPTION_MODEL_ID=Systran/faster-distil-whisper-small.en
```

### Backend Selection

| Backend | CLI Flag | Default URL | Configuration |
|---------|----------|-------------|---------------|
| whisper.cpp | (default) | `http://127.0.0.1:7777/inference` | Config file or env var |
| Speaches | `--speaches` | `http://localhost:8000/v1/audio/transcriptions` | Config file or env var |

### Privacy Summary by Component

Quick reference for privacy-conscious users:

| Component | Backend | Privacy | Internet Required | CLI Flag |
|-----------|---------|---------|-------------------|----------|
| **Transcription** | whisper.cpp | 🔒 100% Local | No | (default) |
| | Speaches | 🔒 100% Local | No | `--speaches` |
| | Speaches Realtime | 🔒 100% Local | No | `--realtime --speaches` |
| | OpenAI Realtime | ☁️ Cloud | Yes | `--realtime --openai` |
| | ElevenLabs | ☁️ Cloud | Yes | `--realtime --elevenlabs` |
| **LLM (Assistant/Code)** | Ollama | 🔒 100% Local | No | `--llm ollama` |
| | Anthropic Claude | ☁️ Cloud | Yes | `--llm anthropic` |
| | OpenAI GPT | ☁️ Cloud | Yes | `--llm openai` |
| | Google Gemini | ☁️ Cloud | Yes | `--llm google` |
| | DeepSeek | ☁️ Cloud | Yes | `--llm deepseek` |

**Privacy Tip**: For complete privacy, use:
```bash
# 100% offline voice typing
voxtty --speaches --tray

# 100% offline with AI assistance
voxtty --speaches --assistant --llm ollama --tray
```

### ⚠️ Important: ydotool Setup

**Ubuntu's ydotool package is BROKEN**. You MUST build from source:

```bash
git clone https://github.com/ReimuNotMoe/ydotool.git
cd ydotool && mkdir build && cd build
cmake -DSYSTEMD_USER_SERVICE=ON ..
make -j $(nproc) && sudo make install
systemctl --user enable --now ydotoold.service
```

📖 **See the relevant sections in this README for setup and configuration details.**

### Audio Tuning

If you experience issues with voice detection:

1. **Recording never stops** - Microphone volume too high
   - Lower mic volume in system settings
   - Increase silence threshold in code

2. **Recording doesn't start** - Microphone volume too low
   - Increase mic volume in system settings
   - Decrease amplitude threshold in code

3. **Background noise triggers recording** - Environment too noisy
   - Use push-to-talk via hotkey toggle
   - Increase VAD sensitivity

## 🏗️ Architecture

### What voxtty IS (and what it's NOT)

voxtty is a **voice-to-text application** that listens to your microphone and types text system-wide. It's designed for direct user interaction, not as a protocol server.

**voxtty is NOT an MCP server**, and here's why:

- **MCP (Model Context Protocol)** is Anthropic's protocol for connecting AI models to external tools and data sources
- **voxtty** is a standalone desktop application for voice typing, not a tool server
- **Different use case**: MCP servers provide context to AI models; voxtty provides voice input to users
- **Different architecture**: voxtty uses a processor pattern with direct LLM API calls, not the MCP protocol

**Why we considered MCP support (but didn't implement it)**:

We explored adding MCP as an **Assistant backend** (like we have for OpenAI, Anthropic, Ollama) to allow the Assistant mode to call external tools. The architecture already supports it via the `AssistantBackend` trait:

```rust
// src/processors_assistant.rs
pub trait AssistantBackend {
    fn process_with_llm(&self, audio_path: &Path, mode: &VoiceMode) -> Result<String>;
}

// Implemented: SpeachesAssistantBackend (uses direct LLM APIs)
// Not implemented: MCPAssistantBackend (would use MCP protocol)
```

**Why it's not implemented yet**:

1. **Core functionality works without it** - Direct LLM API calls (OpenAI, Anthropic, Ollama) cover most use cases
2. **Additional complexity** - MCP adds protocol overhead and server dependencies
3. **Can be added later** - The architecture is ready; it's a backend swap, not a redesign
4. **Community input needed** - Uncertain if users want MCP integration for this use case

**If you need MCP functionality**: Consider using voxtty for voice input and a separate MCP-enabled tool for tool calling. The Unix philosophy of composing single-purpose tools often works better than one tool doing everything.

That said, if there's demand, we can implement the `MCPAssistantBackend` - the plumbing is already there!

### Core Components

- **Audio Capture** - CPAL for cross-platform audio input
- **Voice Detection** - WebRTC VAD + amplitude threshold
- **Transcription** - Whisper.cpp or Speaches API
- **Text Input** - ydotool for system-wide typing
- **UI Controls** - ksni (system tray)

### Audio Pipeline

```
Microphone → CPAL → VAD → WAV Buffer → Whisper AI → ydotool → Text Output
```

### Detection Algorithm

1. Capture audio in 30ms frames at 16kHz
2. Run WebRTC VAD on each frame
3. Check amplitude threshold (>1000)
4. Require 200ms of speech to start
5. Wait 1000ms of silence to stop
6. Transcribe and type result

## 🐛 Troubleshooting

### Quick Fixes

**Audio not working?**
```bash
voxtty --echo-test
```

**Transcription failing?**
```bash
# Check backend is running
docker ps | grep speaches          # For Speaches
curl http://127.0.0.1:7777/        # For whisper.cpp
```

**Text not typing?**
```bash
# Check ydotool
systemctl --user status ydotoold.service
ydotool type "test"
```

**Need more help?**
```bash
# Run with debug output
voxtty --debug --speaches
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone https://github.com/jflaflamme/voxtty.git
cd voxtty

# Build in debug mode
cargo build

# Run with debug output
cargo run -- --debug --echo-test

# Run tests
cargo test

# Check code quality
cargo clippy
cargo fmt --check
```

## 📚 Documentation

All documentation is contained in this README. Additional detailed guides coming soon!

## 📝 License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - State-of-the-art speech recognition
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Efficient C++ implementation
- [ydotool](https://github.com/ReimuNotMoe/ydotool) - Generic command-line automation tool
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) - Voice activity detection

## 🔗 Links

- **Repository**: https://github.com/jflaflamme/voxtty
- **Issues**: https://github.com/jflaflamme/voxtty/issues
- **Releases**: https://github.com/jflaflamme/voxtty/releases

---

**Made with ❤️ by Jean-Francois Laflamme**

*Privacy-focused voice typing for everyone*
