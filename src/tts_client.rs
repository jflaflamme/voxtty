// Unified TTS client: dispatches to ElevenLabs (cloud, async API) or an
// OpenAI-compatible server (e.g. Lemonade/Kokoro, blocking API).
use crate::elevenlabs_tts::ElevenLabsTts;
use crate::openai_tts::OpenAiTts;
use anyhow::Result;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// A TTS backend the conversation processor can speak through.
pub enum TtsClient {
    ElevenLabs(ElevenLabsTts),
    OpenAi(OpenAiTts),
}

impl TtsClient {
    /// Speak `text` and block until playback finishes (or `interrupt_flag` is set).
    /// Hides the async/sync difference between the two backends from callers.
    pub fn speak_blocking(
        &self,
        text: &str,
        interrupt_flag: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        match self {
            TtsClient::OpenAi(client) => client.speak_interruptible(text, interrupt_flag),
            TtsClient::ElevenLabs(client) => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(client.speak_and_play_interruptible(text, interrupt_flag))
            }
        }
    }
}
