# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in voxtty, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainer directly at: jflaflamme@github (or open a private security advisory on GitHub)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity, typically 1-4 weeks

## Security Considerations

### Privacy by Design

voxtty is designed with privacy as a core principle:

- **Local processing**: Dictation mode uses only local backends (whisper.cpp, Speaches)
- **No telemetry**: Zero data collection or tracking
- **User control**: You choose which backends to use

### When Using Cloud Services

If you enable cloud backends (OpenAI, ElevenLabs, Anthropic):

- Audio data is sent to third-party servers
- Review their privacy policies
- voxtty displays privacy warnings when cloud services are active

### API Keys

- Store API keys in environment variables or config files
- Never commit API keys to version control
- Config file location: `~/.config/voxtty/config.toml`
- Recommended permissions: `chmod 600 ~/.config/voxtty/config.toml`

### System Integration

- voxtty uses `ydotool` for text input (requires socket access)
- System tray uses DBus (standard Linux IPC)
- No elevated privileges required for normal operation

## Known Limitations

- Audio is temporarily stored in `/tmp/` during processing (deleted after use)
- WebSocket connections use TLS when connecting to cloud providers
- Local backends (whisper.cpp, Speaches) may use unencrypted HTTP on localhost

## Security Updates

Security fixes are released as patch versions and announced in:
- GitHub Releases
- CHANGELOG.md
