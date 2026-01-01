// Audio processing trait and implementations
use anyhow::Result;
use std::any::Any;
use std::path::Path;

/// Context passed to processors
#[derive(Debug, Clone)]
pub struct ProcessContext {
    pub mode: VoiceMode,
    pub debug: bool,
}

/// Voice mode determines how audio is processed
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceMode {
    /// Direct transcription (current behavior)
    Dictation,
    /// LLM-assisted writing
    Assistant { context: Vec<String> },
    /// Code generation
    Code { language: Option<String> },
    /// Command mode - execute shell commands
    Command,
}

/// Trait for audio processors
///
/// Processors take audio and return text to be typed.
/// Different processors can handle different modes.
pub trait AudioProcessor: Send + Sync {
    /// Process audio file and return text to type
    fn process(&self, audio_path: &Path, context: &ProcessContext) -> Result<String>;

    /// Processor name for logging
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Check if this processor supports the given mode
    fn supports_mode(&self, mode: &VoiceMode) -> bool;

    /// Allow downcasting to concrete type
    fn as_any(&self) -> &dyn Any;
}

/// Registry of available processors
pub struct ProcessorRegistry {
    processors: Vec<Box<dyn AudioProcessor>>,
}

impl ProcessorRegistry {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Register a processor
    pub fn register(&mut self, processor: Box<dyn AudioProcessor>) {
        self.processors.push(processor);
    }

    /// Find processor that supports the given mode
    pub fn find_processor(&self, mode: &VoiceMode) -> Option<&dyn AudioProcessor> {
        self.processors
            .iter()
            .find(|p| p.supports_mode(mode))
            .map(|p| p.as_ref())
    }
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
