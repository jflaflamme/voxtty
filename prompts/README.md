# VoiceTypr System Prompts

This directory contains the system prompts used for different voice modes in VoiceTypr. These prompts are compiled into the binary using Rust's `include_str!` macro, so any changes require a rebuild.

## Available Prompts

### `assistant.md`
Used for **Assistant Mode** (key `2` or say "assistant mode")
- Purpose: Improve dictated text with grammar correction and clarity
- Default behavior: Clean up spoken text while maintaining meaning
- Customize: Adjust tone, formality, or add domain-specific knowledge

### `code.md`
Used for **Code Mode** (key `3` or say "code mode")
- Purpose: Generate code from natural language descriptions
- Default behavior: Return clean code without markdown formatting
- Customize: Add language preferences, coding standards, or frameworks

### `command.md`
Used for **Command Mode** (key `4` or say "command mode")
- Purpose: Convert voice commands to shell commands
- Default behavior: Returns JSON with heard text, interpretation, and command
- Customize: Add more voice patterns, aliases, or custom commands

## How to Customize

1. **Edit the markdown files** in this directory
2. **Rebuild the project**: `cargo build --release`
3. **Restart voxtty** to use the new prompts

## Format

- Use plain markdown for readability
- The entire file content becomes the system prompt
- Be clear and specific about output format requirements
- Test prompts thoroughly after changes

## Tips

- **Assistant Mode**: Focus on output formatting and tone
- **Code Mode**: Specify language preferences and comment style
- **Command Mode**: Add common voice-to-command mappings you use
- Keep prompts concise - longer prompts may increase latency

## Examples

To add a custom command pattern to `command.md`:
```markdown
- "docker pee ess" → "docker ps"
- "docker run" → "docker run"
- "git push origin master" → "git push origin master"
```

To change assistant tone in `assistant.md`:
```markdown
You are a professional business writing assistant. Use formal language,
proper grammar, and corporate tone. Expand abbreviations and ensure
all communication is clear and professional.
```
