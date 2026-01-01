#!/bin/bash
# Update ElevenLabs pronunciation dictionary for voxtty

set -e

# Check if API key is set
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "Error: ELEVENLABS_API_KEY environment variable not set"
    echo "Usage: export ELEVENLABS_API_KEY=your_key_here && ./update_pronunciation.sh"
    exit 1
fi

DICT_ID="VVaDZcFhaoX4PgMNmhHe"

echo "📖 Adding pronunciation rules for 'voxtty'..."

# Add pronunciation rules
response=$(curl -s -X POST "https://api.elevenlabs.io/v1/pronunciation-dictionaries/${DICT_ID}/add-rules" \
  -H "xi-api-key: ${ELEVENLABS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "rules": [
      {
        "type": "alias",
        "string_to_replace": "voxtty",
        "alias": "vocks T T Y"
      },
      {
        "type": "alias",
        "string_to_replace": "Voxtty",
        "alias": "Vocks T T Y"
      }
    ]
  }')

echo "$response" | jq .

# Check if successful
if echo "$response" | jq -e '.version_id' > /dev/null 2>&1; then
    echo ""
    echo "✅ Pronunciation rules added successfully!"

    # Get new version ID
    new_version=$(echo "$response" | jq -r '.version_id')
    echo "📝 New version ID: $new_version"

    echo ""
    echo "To use this new version, update your config:"
    echo "  elevenlabs_pronunciation_dict_version = \"$new_version\""
    echo ""
    echo "Or the script will update it automatically..."

    # Update config file
    config_file="$HOME/.config/voxtty/config.toml"
    if [ -f "$config_file" ]; then
        # Backup first
        cp "$config_file" "$config_file.bak"

        # Update version in config
        sed -i "s/elevenlabs_pronunciation_dict_version = .*/elevenlabs_pronunciation_dict_version = \"$new_version\"/" "$config_file"

        echo "✅ Updated $config_file"
        echo "📋 Backup saved to $config_file.bak"
    else
        echo "⚠️  Config file not found at $config_file"
        echo "Please update manually."
    fi
else
    echo ""
    echo "❌ Failed to add pronunciation rules"
    echo "Error details above"
    exit 1
fi
