// Mode detection and management
use crate::processors::VoiceMode;

/// Voice command action
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceCommand {
    /// Switch to a new mode
    SwitchMode(VoiceMode),
    /// Pause transcription (stop typing, keep listening for resume)
    Pause,
    /// Resume transcription
    Resume,
    /// No command detected
    None,
}

/// Wake word detector
pub struct WakeWordDetector {
    assistant_words: Vec<String>,
    code_words: Vec<String>,
    dictation_words: Vec<String>,
    command_words: Vec<String>,
    pause_words: Vec<String>,
    resume_words: Vec<String>,
}

impl WakeWordDetector {
    pub fn new() -> Self {
        Self {
            assistant_words: vec!["hey assistant".to_string(), "assistant mode".to_string()],
            code_words: vec![
                "code mode".to_string(),
                "coding mode".to_string(),
                "write code".to_string(),
            ],
            dictation_words: vec![
                "dictation mode".to_string(),
                "normal mode".to_string(),
                "typing mode".to_string(),
                "type mode".to_string(),
            ],
            command_words: vec![
                "command mode".to_string(),
                "terminal mode".to_string(),
                "shell mode".to_string(),
                "sysadmin".to_string(),
                "system mode".to_string(),
                "console mode".to_string(),
            ],
            pause_words: vec![
                "pause".to_string(),
                "stop listening".to_string(),
                "go to sleep".to_string(),
            ],
            resume_words: vec![
                "resume".to_string(),
                "start listening".to_string(),
                "wake up".to_string(),
            ],
        }
    }

    /// Detect voice command from transcribed text
    /// Returns (command, should_type_text)
    pub fn detect_command(&self, text: &str) -> (VoiceCommand, bool) {
        let lower = text.to_lowercase();

        // Check for pause/resume first (higher priority)
        for word in &self.pause_words {
            if lower.contains(word) {
                return (VoiceCommand::Pause, false);
            }
        }

        for word in &self.resume_words {
            if lower.contains(word) {
                return (VoiceCommand::Resume, false);
            }
        }

        // Check for mode switches
        for word in &self.assistant_words {
            if lower.contains(word) {
                return (
                    VoiceCommand::SwitchMode(VoiceMode::Assistant { context: vec![] }),
                    false,
                );
            }
        }

        for word in &self.code_words {
            if lower.contains(word) {
                return (
                    VoiceCommand::SwitchMode(VoiceMode::Code { language: None }),
                    false,
                );
            }
        }

        for word in &self.dictation_words {
            if lower.contains(word) {
                return (VoiceCommand::SwitchMode(VoiceMode::Dictation), false);
            }
        }

        for word in &self.command_words {
            if lower.contains(word) {
                return (VoiceCommand::SwitchMode(VoiceMode::Command), false);
            }
        }

        // No command detected, type the text
        (VoiceCommand::None, true)
    }

    /// Legacy method for backward compatibility
    /// Returns (new_mode, should_type_text)
    #[allow(dead_code)]
    pub fn detect(&self, text: &str) -> (Option<VoiceMode>, bool) {
        let (command, should_type) = self.detect_command(text);
        match command {
            VoiceCommand::SwitchMode(mode) => (Some(mode), should_type),
            _ => (None, should_type),
        }
    }
}

impl Default for WakeWordDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wake_word_detection() {
        let detector = WakeWordDetector::new();

        // Test assistant mode
        let (mode, should_type) = detector.detect("Hey assistant, help me");
        assert!(matches!(mode, Some(VoiceMode::Assistant { .. })));
        assert!(!should_type);

        // Test code mode
        let (mode, should_type) = detector.detect("Code mode, create a function");
        assert!(matches!(mode, Some(VoiceMode::Code { .. })));
        assert!(!should_type);

        // Test dictation mode
        let (mode, should_type) = detector.detect("Dictation mode");
        assert!(matches!(mode, Some(VoiceMode::Dictation)));
        assert!(!should_type);

        // Test normal text
        let (mode, should_type) = detector.detect("This is normal text");
        assert!(mode.is_none());
        assert!(should_type);
    }

    #[test]
    fn test_pause_resume_detection() {
        let detector = WakeWordDetector::new();

        // Test pause
        let (cmd, should_type) = detector.detect_command("pause");
        assert_eq!(cmd, VoiceCommand::Pause);
        assert!(!should_type);

        let (cmd, _) = detector.detect_command("Stop listening please");
        assert_eq!(cmd, VoiceCommand::Pause);

        let (cmd, _) = detector.detect_command("go to sleep");
        assert_eq!(cmd, VoiceCommand::Pause);

        // Test resume
        let (cmd, should_type) = detector.detect_command("resume");
        assert_eq!(cmd, VoiceCommand::Resume);
        assert!(!should_type);

        let (cmd, _) = detector.detect_command("Start listening");
        assert_eq!(cmd, VoiceCommand::Resume);

        let (cmd, _) = detector.detect_command("wake up");
        assert_eq!(cmd, VoiceCommand::Resume);
    }
}
