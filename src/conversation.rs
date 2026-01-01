// Bidirectional conversation handling with ElevenLabs TTS
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Represents a single turn in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// The speaker (User or Assistant)
    pub speaker: Speaker,
    /// The message content
    pub message: String,
    /// When this turn occurred (not serialized, always set to now on deserialization)
    #[serde(skip, default = "Instant::now")]
    #[allow(dead_code)]
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Speaker {
    User,
    Assistant,
}

/// Conversation state for multi-turn interactions
#[derive(Debug, Clone)]
pub struct ConversationContext {
    /// History of all turns in this conversation
    pub history: Vec<ConversationTurn>,
    /// The original user intent (first message)
    pub original_intent: Option<String>,
    /// Current state of the conversation
    pub state: ConversationState,
    /// Number of clarification questions asked
    pub clarification_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConversationState {
    /// Ready for new input
    Idle,
    /// Processing user input
    Processing,
    /// Waiting for user to answer a clarification question
    WaitingForClarification,
    /// Ready to execute the final action
    ReadyToExecute,
    /// Conversation completed
    Completed,
}

impl ConversationContext {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            original_intent: None,
            state: ConversationState::Idle,
            clarification_count: 0,
        }
    }

    /// Add a user message to the conversation
    pub fn add_user_message(&mut self, message: String) {
        if self.original_intent.is_none() {
            self.original_intent = Some(message.clone());
        }

        self.history.push(ConversationTurn {
            speaker: Speaker::User,
            message,
            timestamp: Instant::now(),
        });
    }

    /// Add an assistant message to the conversation
    pub fn add_assistant_message(&mut self, message: String) {
        self.history.push(ConversationTurn {
            speaker: Speaker::Assistant,
            message,
            timestamp: Instant::now(),
        });
    }

    /// Get the conversation history formatted for LLM context
    pub fn get_context_for_llm(&self) -> Vec<String> {
        self.history
            .iter()
            .map(|turn| {
                let prefix = match turn.speaker {
                    Speaker::User => "User:",
                    Speaker::Assistant => "Assistant:",
                };
                format!("{} {}", prefix, turn.message)
            })
            .collect()
    }

    /// Reset the conversation
    pub fn reset(&mut self) {
        self.history.clear();
        self.original_intent = None;
        self.state = ConversationState::Idle;
        self.clarification_count = 0;
    }

    /// Check if we should limit clarification questions
    pub fn can_ask_clarification(&self) -> bool {
        // Limit to 2 clarification questions to avoid infinite loops
        self.clarification_count < 2
    }
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Response from LLM analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct LlmAnalysisResponse {
    /// Whether clarification is needed
    pub needs_clarification: bool,
    /// The clarification question to ask (if needed)
    pub clarification_question: Option<String>,
    /// The final response/action (if no clarification needed)
    pub response: Option<String>,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
}
