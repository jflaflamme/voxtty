# Contributing to voxtty

Thanks for your interest in contributing to voxtty! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.70+ and Cargo
- Linux system (Ubuntu/Debian recommended for development)
- Development dependencies:
  ```bash
  sudo apt install libasound2-dev libdbus-1-dev pkg-config
  ```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/jflaflamme/voxtty.git
cd voxtty

# Build in debug mode
cargo build

# Run tests
cargo test

# Run with debug output
cargo run -- --debug --echo-test
```

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check existing [issues](https://github.com/jflaflamme/voxtty/issues) to avoid duplicates
2. Use the bug report template
3. Include:
   - voxtty version (`voxtty --version`)
   - OS and version
   - Steps to reproduce
   - Expected vs actual behavior
   - Debug output (`--debug` flag)

### Suggesting Features

1. Check existing issues and discussions first
2. Use the feature request template
3. Explain the use case and why it would benefit users

### Code Contributions

#### Workflow

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Test** your changes:
   ```bash
   cargo test
   cargo clippy
   cargo fmt --check
   ```
5. **Commit** with clear messages (see commit guidelines below)
6. **Push** to your fork
7. **Open a Pull Request** against `main`

#### Code Style

- Follow Rust conventions and idioms
- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes without warnings
- Write descriptive variable and function names
- Add comments for complex logic

#### Commit Messages

Write clear, concise commit messages:

```
Add realtime WebSocket reconnection

- Auto-reconnect when connection drops
- Show connection status in tray tooltip
- Add 1 second delay before reconnect attempt
```

- First line: imperative mood, max 50 chars
- Body: explain what and why (not how)
- Reference issues: `Fixes #123` or `Closes #456`

#### Pull Request Guidelines

- Fill out the PR template completely
- Keep PRs focused on a single change
- Update documentation if needed
- Add tests for new functionality
- Ensure CI passes

## Architecture Overview

```
src/
├── main.rs                    # Entry point, CLI, main loop
├── processors.rs              # AudioProcessor trait
├── processors_transcription.rs # STT processors
├── processors_assistant.rs    # LLM processors
├── modes.rs                   # Voice modes and wake words
├── model_selector.rs          # Interactive model selection
├── realtime.rs                # WebSocket streaming
└── sounds.rs                  # Notification sounds
```

### Key Concepts

- **Processors**: Implement `AudioProcessor` trait for different transcription/processing modes
- **Voice Modes**: Dictation, Assistant, Code - each with different processing pipelines
- **Backends**: whisper.cpp, Speaches, OpenAI, ElevenLabs - all via standard APIs

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Test specific module
cargo test realtime::

# Manual testing
cargo run -- --echo-test           # Verify microphone
cargo run -- --debug --speaches    # Test with debug output
```

## Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for development workflow changes
- Add doc comments for public APIs
- Update relevant docs/ files

## Release Process

1. Update version in `Cargo.toml`
2. Update `debian/changelog`
3. Update `CHANGELOG.md`
4. Create git tag: `git tag -a v0.x.x -m "Release v0.x.x"`
5. Build Debian package: `./packaging/scripts/build-deb.sh`

## Getting Help

- Open a [Discussion](https://github.com/jflaflamme/voxtty/discussions) for questions
- Check existing issues and docs
- Join the community chat (if available)

## License

By contributing, you agree that your contributions will be licensed under the GPL-2.0 License.

---

Thank you for contributing to voxtty!
