# ElevenLabs Pronunciation Dictionary Setup

This guide explains how to set up custom pronunciation for "Voxtty" (and other words) using ElevenLabs pronunciation dictionaries.

## Overview

ElevenLabs pronunciation dictionaries allow you to customize how specific words are pronounced in text-to-speech output. Voxtty now supports pronunciation dictionaries in bidirectional conversation mode and anywhere ElevenLabs TTS is used.

## Quick Setup for "Voxtty" Pronunciation

### Step 1: Create the Dictionary

Run the provided script to create a pronunciation dictionary that pronounces "Voxtty" as "vox-t-t-y":

```bash
export ELEVENLABS_API_KEY=your_api_key_here
./create_voxtty_dict.sh
```

The script will output something like:

```
✅ Successfully created pronunciation dictionary!

Dictionary ID: dict_abc123xyz
Version ID: ver_def456uvw
Name: Voxtty Pronunciation

To use this dictionary, add the following to your ~/.config/voxtty/config.toml:

elevenlabs_pronunciation_dict_id = "dict_abc123xyz"
elevenlabs_pronunciation_dict_version = "ver_def456uvw"
```

### Step 2: Update Config File

Add the dictionary IDs to your `~/.config/voxtty/config.toml`:

```toml
# ElevenLabs configuration
elevenlabs_api_key = "your_api_key"
elevenlabs_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (default)

# Pronunciation dictionary (optional)
elevenlabs_pronunciation_dict_id = "dict_abc123xyz"
elevenlabs_pronunciation_dict_version = "ver_def456uvw"
```

### Step 3: Test

Run voxtty with bidirectional mode and ask it to say "Voxtty":

```bash
./target/release/voxtty --bidirectional --assistant --speaches --tray
```

Then speak: "Say the word Voxtty"

The TTS should now pronounce it as "vox-t-t-y" instead of trying to pronounce it as a single word.

## Creating Custom Pronunciation Dictionaries

### Using the API Directly

You can create custom pronunciation dictionaries with multiple rules:

```bash
curl -X POST "https://api.elevenlabs.io/v1/pronunciation-dictionaries/add-from-rules" \
  -H "xi-api-key: $ELEVENLABS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Custom Dictionary",
    "description": "Custom pronunciations",
    "rules": [
      {
        "string_to_replace": "Voxtty",
        "type": "alias",
        "alias": "vox-t-t-y"
      },
      {
        "string_to_replace": "API",
        "type": "alias",
        "alias": "A-P-I"
      }
    ]
  }'
```

### Rule Types

**Alias Rules** (recommended for simple replacements):
```json
{
  "string_to_replace": "Voxtty",
  "type": "alias",
  "alias": "vox-t-t-y"
}
```

**Phoneme Rules** (advanced, uses IPA or CMU Arpabet):
```json
{
  "string_to_replace": "Voxtty",
  "type": "phoneme",
  "phoneme": "V AA K S T IY T IY",
  "alphabet": "cmu"
}
```

Note: CMU Arpabet is recommended over IPA for more consistent results.

## How It Works

1. **WebSocket Initialization**: The pronunciation dictionary locators are sent in the **first WebSocket message** when establishing the TTS connection.

2. **Text Processing**: When text is sent for speech synthesis, ElevenLabs checks the dictionary from start to end and applies the **first matching rule** for each word.

3. **Case Sensitivity**: Dictionary searches are **case-sensitive**, so "Voxtty" and "voxtty" are treated differently.

## Technical Implementation

### Code Structure

- **`src/elevenlabs_tts.rs`**:
  - `PronunciationDictionaryLocator` struct
  - `create_pronunciation_dictionary()` function for API calls
  - WebSocket message format with `pronunciation_dictionary_locators` field

- **`src/main.rs`**:
  - Config fields: `elevenlabs_pronunciation_dict_id`, `elevenlabs_pronunciation_dict_version`
  - Helper function: `create_elevenlabs_tts()` to build TTS client with dictionary support

### Example Usage in Code

```rust
use voxtty::elevenlabs_tts::{ElevenLabsTts, PronunciationDictionaryLocator};

let mut tts = ElevenLabsTts::new(api_key, voice_id);

// Add pronunciation dictionary
tts = tts.with_pronunciation_dict(PronunciationDictionaryLocator {
    pronunciation_dictionary_id: "dict_abc123xyz".to_string(),
    version_id: "ver_def456uvw".to_string(),
});

// Use TTS normally
tts.speak_and_play("Welcome to Voxtty!").await?;
```

## Troubleshooting

### Dictionary Not Working

1. **Verify IDs**: Check that the dictionary ID and version ID in your config match the output from the creation script.

2. **Check Logs**: Run voxtty with debug logging to see if the dictionary is being loaded:
   ```bash
   ./target/release/voxtty --debug --bidirectional --assistant --speaches
   ```
   Look for: `📖 Using pronunciation dictionary: ...`

3. **Test Simple Text**: Try speaking just the word "Voxtty" to isolate the issue.

### Case Sensitivity Issues

If "voxtty" (lowercase) isn't being pronounced correctly, add additional rules:

```json
{
  "string_to_replace": "voxtty",
  "type": "alias",
  "alias": "vox-t-t-y"
}
```

## References

- [ElevenLabs Pronunciation Dictionary Best Practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices#pronunciation-dictionaries)
- [Create Pronunciation Dictionary API](https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-rules)
- [WebSocket TTS Documentation](https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input)
- [GitHub Issue #295 - WebSocket Dictionary Support](https://github.com/elevenlabs/elevenlabs-python/issues/295)
