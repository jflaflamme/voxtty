#!/bin/bash
# Create Voxtty pronunciation dictionary in ElevenLabs
# Usage: ./create_voxtty_dict.sh

set -e

# Check for API key
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "Error: ELEVENLABS_API_KEY environment variable not set"
    echo "Please set it with: export ELEVENLABS_API_KEY=your_api_key"
    exit 1
fi

echo "Creating Voxtty pronunciation dictionary..."

# Create JSON request body
REQUEST_BODY=$(cat <<EOF
{
  "name": "Voxtty Pronunciation",
  "description": "Pronunciation dictionary for Voxtty voice assistant",
  "rules": [
    {
      "string_to_replace": "Voxtty",
      "type": "alias",
      "alias": "vox-t-t-y"
    }
  ]
}
EOF
)

# Call ElevenLabs API
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "https://api.elevenlabs.io/v1/pronunciation-dictionaries/add-from-rules" \
    -H "xi-api-key: $ELEVENLABS_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY")

# Split response and HTTP code
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "Failed to create pronunciation dictionary (HTTP $HTTP_CODE):"
    echo "$BODY" | jq -r '.' 2>/dev/null || echo "$BODY"
    exit 1
fi

# Parse response
DICT_ID=$(echo "$BODY" | jq -r '.id')
VERSION_ID=$(echo "$BODY" | jq -r '.version_id')
NAME=$(echo "$BODY" | jq -r '.name')
DESCRIPTION=$(echo "$BODY" | jq -r '.description // empty')

echo "✅ Successfully created pronunciation dictionary!"
echo ""
echo "Dictionary ID: $DICT_ID"
echo "Version ID: $VERSION_ID"
echo "Name: $NAME"
[ -n "$DESCRIPTION" ] && echo "Description: $DESCRIPTION"
echo ""
echo "To use this dictionary, add the following to your ~/.config/voxtty/config.toml:"
echo ""
echo "elevenlabs_pronunciation_dict_id = \"$DICT_ID\""
echo "elevenlabs_pronunciation_dict_version = \"$VERSION_ID\""
echo ""
echo "This will make ElevenLabs pronounce 'Voxtty' as 'vox-t-t-y'"
