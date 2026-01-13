// Bidirectional conversation processor with clarification support
use crate::conversation::{ConversationContext, ConversationState, LlmAnalysisResponse};
use crate::elevenlabs_tts::ElevenLabsTts;
use crate::processors::{AudioProcessor, ProcessContext, VoiceMode};
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::any::Any;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

type TranscriptionFn = Arc<dyn Fn(&Path) -> Result<String> + Send + Sync>;

/// Processor that handles bidirectional conversations with clarification
pub struct ConversationProcessor {
    http_client: Client,
    api_base_url: String,
    api_key: String,
    model: String,
    context: Arc<Mutex<ConversationContext>>,
    tts_client: Option<Arc<ElevenLabsTts>>,
    transcription_fn: Option<TranscriptionFn>,
    is_tts_speaking: Arc<Mutex<bool>>,
    tts_interrupt: Arc<AtomicBool>,
}

impl ConversationProcessor {
    /// Create a new ConversationProcessor with a pre-configured TTS client
    pub fn with_tts_client(
        api_base_url: String,
        api_key: String,
        model: String,
        tts_client: Arc<ElevenLabsTts>,
        is_tts_speaking: Arc<Mutex<bool>>,
        tts_interrupt: Arc<AtomicBool>,
    ) -> Self {
        Self {
            http_client: Client::new(),
            api_base_url,
            api_key,
            model,
            context: Arc::new(Mutex::new(ConversationContext::new())),
            tts_client: Some(tts_client),
            transcription_fn: None,
            is_tts_speaking,
            tts_interrupt,
        }
    }

    /// Analyze transcription and determine if clarification is needed
    async fn analyze_transcription(
        &self,
        transcription: &str,
        context: &ConversationContext,
        mode: &VoiceMode,
    ) -> Result<LlmAnalysisResponse> {
        let system_prompt = match mode {
            VoiceMode::Command => include_str!("../prompts/command.md").to_string(),
            VoiceMode::Code { .. } => include_str!("../prompts/code.md").to_string(),
            _ => include_str!("../prompts/assistant.md").to_string(),
        };

        let mut messages = vec![json!({
            "role": "system",
            "content": system_prompt
        })];

        // Add conversation history
        for msg in context.get_context_for_llm() {
            messages.push(json!({
                "role": "user",
                "content": msg
            }));
        }

        // Add current transcription
        messages.push(json!({
            "role": "user",
            "content": format!("Analyze this request: {}", transcription)
        }));

        // Add tool support - tools depend on mode
        let mut tools = serde_json::json!([]);

        // process_command tool (Command mode only)
        if matches!(mode, VoiceMode::Command) {
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "process_command",
                    "description": "Process a voice command and convert it to a shell command with safety analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hearing": {
                                "type": "string",
                                "description": "The exact text that was heard/transcribed"
                            },
                            "understanding": {
                                "type": "string",
                                "description": "Explanation of the user's intent"
                            },
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute (or empty string if unsafe/rejected)"
                            },
                            "risk": {
                                "type": "string",
                                "enum": ["safe", "low", "medium", "high", "destructive"],
                                "description": "Safety risk level of the command"
                            }
                        },
                        "required": ["hearing", "understanding", "command", "risk"]
                    }
                }
            }));
        }

        // speak tool (Assistant, Command, and Code modes)
        if matches!(
            mode,
            VoiceMode::Assistant { .. } | VoiceMode::Command | VoiceMode::Code { .. }
        ) {
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "speak",
                    "description": "Speak a response back to the user via TTS. Use this for clarifications, rejections, answers to questions, or confirmations. Does NOT type text to keyboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to speak"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }));

            // type_text tool
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text to the keyboard (simulates typing). Use this for dictation, writing emails, code, etc. Does NOT speak via TTS.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to type to the keyboard"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }));

            // switch_mode tool
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "switch_mode",
                    "description": "Switch voice input mode when user requests (e.g., 'switch to dictation', 'code mode').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["dictation", "assistant", "code", "command"],
                                "description": "The mode to switch to"
                            },
                            "confirmation": {
                                "type": "string",
                                "description": "Brief confirmation message to speak"
                            }
                        },
                        "required": ["mode", "confirmation"]
                    }
                }
            }));
        }

        // In Command, Assistant, and Code modes, force tool usage to prevent unwanted typing
        let tool_choice = if matches!(
            mode,
            VoiceMode::Command | VoiceMode::Assistant { .. } | VoiceMode::Code { .. }
        ) {
            "required"
        } else {
            "auto"
        };

        let request_body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "tools": tools,
            "tool_choice": tool_choice
        });

        let response = self
            .http_client
            .post(format!("{}/chat/completions", self.api_base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .context("Failed to send analysis request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API request failed: {} - {}", status, error_text);
        }

        let response_json: serde_json::Value = response.json().await?;

        eprintln!(
            "[DEBUG] LLM Response: {}",
            serde_json::to_string_pretty(&response_json).unwrap_or_default()
        );

        // Check for tool calls (same pattern as AssistantProcessor)
        if let Some(tool_calls) = response_json["choices"][0]["message"]["tool_calls"].as_array() {
            eprintln!("[DEBUG] Found {} tool calls", tool_calls.len());
            if let Some(tool_call) = tool_calls.first() {
                if let Some(tool_name) = tool_call["function"]["name"].as_str() {
                    eprintln!("[DEBUG] Tool name: {}", tool_name);

                    if tool_name == "process_command" {
                        if let Some(args_str) = tool_call["function"]["arguments"].as_str() {
                            eprintln!("[DEBUG] Tool args: {}", args_str);
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                                if let Some(command) = args["command"].as_str() {
                                    eprintln!("[DEBUG] Extracted command: {}", command);
                                    // Return the command for execution
                                    return Ok(LlmAnalysisResponse {
                                        needs_clarification: false,
                                        clarification_question: None,
                                        response: Some(command.to_string()),
                                        confidence: 1.0,
                                    });
                                }
                            }
                        }
                    } else if tool_name == "speak" {
                        if let Some(args_str) = tool_call["function"]["arguments"].as_str() {
                            eprintln!("[DEBUG] Tool args: {}", args_str);
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                                if let Some(speak_text) = args["text"].as_str() {
                                    eprintln!("[DEBUG] Extracted speak text: {}", speak_text);
                                    // Speak tool is for FINAL ANSWERS via TTS (not clarifications)
                                    // We'll speak it in the handler and mark as complete
                                    return Ok(LlmAnalysisResponse {
                                        needs_clarification: false,
                                        clarification_question: None,
                                        response: Some(format!("🔊 {}", speak_text)),
                                        confidence: 1.0,
                                    });
                                }
                            }
                        }
                    } else if tool_name == "type_text" {
                        if let Some(args_str) = tool_call["function"]["arguments"].as_str() {
                            eprintln!("[DEBUG] Tool args: {}", args_str);
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                                if let Some(type_text) = args["text"].as_str() {
                                    eprintln!("[DEBUG] Extracted type text: {}", type_text);
                                    // Return the text to be typed (not spoken)
                                    return Ok(LlmAnalysisResponse {
                                        needs_clarification: false,
                                        clarification_question: None,
                                        response: Some(type_text.to_string()),
                                        confidence: 1.0,
                                    });
                                }
                            }
                        }
                    } else if tool_name == "switch_mode" {
                        if let Some(args_str) = tool_call["function"]["arguments"].as_str() {
                            eprintln!("[DEBUG] Tool args: {}", args_str);
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                                if let (Some(mode_str), Some(confirmation)) =
                                    (args["mode"].as_str(), args["confirmation"].as_str())
                                {
                                    eprintln!(
                                        "[DEBUG] Mode switch request: {} with confirmation: {}",
                                        mode_str, confirmation
                                    );
                                    // Note: Mode switching is handled by the main loop's wake word detector
                                    // We just need to speak the confirmation and return empty
                                    return Ok(LlmAnalysisResponse {
                                        needs_clarification: true,
                                        clarification_question: Some(confirmation.to_string()),
                                        response: None,
                                        confidence: 1.0,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        } else {
            eprintln!("[DEBUG] No tool calls found in response");
        }

        // Fallback to content
        if let Some(content) = response_json["choices"][0]["message"]["content"].as_str() {
            eprintln!("[DEBUG] Using content: {}", content);

            // In bidirectional mode (Assistant/Command/Code), the LLM should ALWAYS use tools
            // If it returns content directly, it's likely a mistake - return error
            if matches!(
                mode,
                VoiceMode::Assistant { .. } | VoiceMode::Command | VoiceMode::Code { .. }
            ) {
                eprintln!(
                    "[WARNING] LLM returned content instead of using a tool in bidirectional mode"
                );
                eprintln!("[WARNING] Content: {}", content);
                return Ok(LlmAnalysisResponse {
                    needs_clarification: false,
                    clarification_question: None,
                    response: Some("🔊 I had trouble understanding your request. Please try again with more specific details, or say 'type as is' if you want me to type exactly what you said.".to_string()),
                    confidence: 0.0,
                });
            }

            // In Dictation/Code modes, return content for typing (no tools available)
            return Ok(LlmAnalysisResponse {
                needs_clarification: false,
                clarification_question: None,
                response: Some(content.to_string()),
                confidence: 1.0,
            });
        }

        eprintln!("[DEBUG] No content or tool call found!");
        anyhow::bail!("No content or tool call in LLM response")
    }

    /// Handle clarification by speaking the question and waiting for response
    async fn handle_clarification(&self, question: &str) -> Result<String> {
        eprintln!("🤔 Clarification needed: {}", question);

        if let Some(tts) = &self.tts_client {
            eprintln!("🔊 Speaking question via ElevenLabs...");
            let tts_clone = Arc::clone(tts);
            let question_owned = question.to_string();
            let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

            // Spawn TTS in background thread to prevent blocking
            *is_speaking_clone.lock().unwrap() = true;
            eprintln!("[DEBUG] Set is_tts_speaking = true (clarification)");
            let interrupt_clone = Arc::clone(&self.tts_interrupt);
            std::thread::spawn(move || {
                eprintln!("[DEBUG TTS Thread] Starting clarification TTS playback");
                let rt = tokio::runtime::Runtime::new().unwrap();
                if let Err(e) = rt.block_on(
                    tts_clone.speak_and_play_interruptible(&question_owned, Some(interrupt_clone)),
                ) {
                    eprintln!("❌ TTS Error: {}", e);
                }
                *is_speaking_clone.lock().unwrap() = false;
                eprintln!("[DEBUG TTS Thread] Set is_tts_speaking = false (clarification)");
                eprintln!("✅ Question spoken, waiting for your response...");
            });
        } else {
            eprintln!("⚠️  No TTS client configured - cannot speak question");
        }

        // Wait for user response (this will come from the next VAD trigger)
        // For now, return a placeholder - the actual implementation will
        // integrate with the main audio loop
        Ok(String::new())
    }
}

impl AudioProcessor for ConversationProcessor {
    fn process(&self, audio_path: &Path, proc_context: &ProcessContext) -> Result<String> {
        // First, transcribe the audio
        let transcription = if let Some(transcribe_fn) = &self.transcription_fn {
            transcribe_fn(audio_path)?
        } else {
            anyhow::bail!("No transcription function configured");
        };

        eprintln!(
            "[DEBUG ConversationProcessor] Transcription: {}",
            transcription
        );

        if transcription.trim().is_empty() {
            eprintln!("[DEBUG ConversationProcessor] Empty transcription, returning");
            return Ok(String::new());
        }

        let mode = &proc_context.mode;

        // Use tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()?;

        rt.block_on(async {
            // Get current state (lock scope minimized)
            let current_state = {
                let ctx = self.context.lock().unwrap();
                ctx.state.clone()
            };

            match current_state {
                ConversationState::Idle | ConversationState::Processing => {
                    // New conversation or continuation - update state
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_user_message(transcription.clone());
                        ctx.state = ConversationState::Processing;
                    }

                    // Get context snapshot for analysis (separate lock scope)
                    let ctx_snapshot = self.context.lock().unwrap().clone();

                    // Analyze if we need clarification (no lock held during await)
                    eprintln!("[DEBUG ConversationProcessor] Calling analyze_transcription");
                    let analysis = self
                        .analyze_transcription(&transcription, &ctx_snapshot, mode)
                        .await?;
                    eprintln!(
                        "[DEBUG ConversationProcessor] Analysis complete: needs_clarification={}",
                        analysis.needs_clarification
                    );

                    // Check if clarification is needed
                    let (needs_clarification, can_clarify) = {
                        let ctx = self.context.lock().unwrap();
                        (analysis.needs_clarification, ctx.can_ask_clarification())
                    };

                    if needs_clarification && can_clarify {
                        if let Some(question) = analysis.clarification_question {
                            // Update context state
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(question.clone());
                                ctx.state = ConversationState::WaitingForClarification;
                                ctx.clarification_count += 1;
                            }

                            // Speak the question (no lock held)
                            self.handle_clarification(&question).await?;

                            // Return formatted output for display (not typing)
                            Ok(format!("🔊 {}", question))
                        } else {
                            Ok(String::new())
                        }
                    } else {
                        // No clarification needed or limit reached - execute
                        {
                            let mut ctx = self.context.lock().unwrap();
                            ctx.state = ConversationState::ReadyToExecute;
                        }

                        if let Some(response) = analysis.response {
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(response.clone());
                            }

                            // Check if this is a TTS response (starts with 🔊)
                            if response.starts_with("🔊 ") {
                                let speak_text = response.trim_start_matches("🔊 ").trim();

                                // Speak via TTS in background
                                if let Some(tts) = &self.tts_client {
                                    eprintln!("🔊 Speaking final answer via ElevenLabs...");
                                    let tts_clone = Arc::clone(tts);
                                    let speak_text_owned = speak_text.to_string();
                                    let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

                                    // Spawn TTS in background thread
                                    *is_speaking_clone.lock().unwrap() = true;
                                    eprintln!("[DEBUG] Set is_tts_speaking = true");
                                    let interrupt_clone = Arc::clone(&self.tts_interrupt);
                                    std::thread::spawn(move || {
                                        eprintln!("[DEBUG TTS Thread] Starting TTS playback");
                                        let rt = tokio::runtime::Runtime::new().unwrap();
                                        if let Err(e) = rt.block_on(tts_clone.speak_and_play_interruptible(&speak_text_owned, Some(interrupt_clone))) {
                                            eprintln!("❌ TTS Error: {}", e);
                                        }
                                        *is_speaking_clone.lock().unwrap() = false;
                                        eprintln!("[DEBUG TTS Thread] Set is_tts_speaking = false");
                                        eprintln!("✅ Answer spoken");
                                    });
                                }

                                // Return the formatted text for display (don't type it)
                                Ok(response)
                            } else {
                                // Regular text response - return for typing
                                Ok(response)
                            }
                        } else {
                            // LLM didn't use tools properly - return error message
                            eprintln!("[WARNING] No response from LLM, returning error");
                            Ok("🔊 I'm not sure what you want me to do. Please be more specific, or say 'type as is' to type your words exactly as you said them.".to_string())
                        }
                    }
                }

                ConversationState::WaitingForClarification => {
                    // User is answering a clarification question - update state
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_user_message(transcription.clone());
                        ctx.state = ConversationState::Processing;
                    }

                    // Get context snapshot for re-analysis
                    let ctx_snapshot = self.context.lock().unwrap().clone();

                    // Re-analyze with the new context (no lock held during await)
                    let analysis = self
                        .analyze_transcription(&transcription, &ctx_snapshot, mode)
                        .await?;

                    // Check if more clarification is needed
                    let (needs_clarification, can_clarify) = {
                        let ctx = self.context.lock().unwrap();
                        (analysis.needs_clarification, ctx.can_ask_clarification())
                    };

                    if needs_clarification && can_clarify {
                        if let Some(question) = analysis.clarification_question {
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(question.clone());
                                ctx.clarification_count += 1;
                            }

                            self.handle_clarification(&question).await?;
                            Ok(format!("🔊 {}", question))
                        } else {
                            Ok(String::new())
                        }
                    } else {
                        {
                            let mut ctx = self.context.lock().unwrap();
                            ctx.state = ConversationState::Completed;
                        }

                        if let Some(response) = analysis.response {
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(response.clone());
                            }

                            // Check if this is a TTS response (starts with 🔊)
                            if response.starts_with("🔊 ") {
                                let speak_text = response.trim_start_matches("🔊 ").trim();

                                // Speak via TTS in background
                                if let Some(tts) = &self.tts_client {
                                    eprintln!("🔊 Speaking final answer via ElevenLabs...");
                                    let tts_clone = Arc::clone(tts);
                                    let speak_text_owned = speak_text.to_string();
                                    let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

                                    // Spawn TTS in background thread
                                    *is_speaking_clone.lock().unwrap() = true;
                                    eprintln!("[DEBUG] Set is_tts_speaking = true");
                                    let interrupt_clone = Arc::clone(&self.tts_interrupt);
                                    std::thread::spawn(move || {
                                        eprintln!("[DEBUG TTS Thread] Starting TTS playback");
                                        let rt = tokio::runtime::Runtime::new().unwrap();
                                        if let Err(e) = rt.block_on(tts_clone.speak_and_play_interruptible(&speak_text_owned, Some(interrupt_clone))) {
                                            eprintln!("❌ TTS Error: {}", e);
                                        }
                                        *is_speaking_clone.lock().unwrap() = false;
                                        eprintln!("[DEBUG TTS Thread] Set is_tts_speaking = false");
                                        eprintln!("✅ Answer spoken");
                                    });
                                }

                                // Return the formatted text for display (don't type it)
                                Ok(response)
                            } else {
                                // Regular text response - return for typing
                                Ok(response)
                            }
                        } else {
                            // LLM didn't use tools properly - return error message
                            eprintln!("[WARNING] No response from LLM, returning error");
                            Ok("🔊 I'm not sure what you want me to do. Please be more specific, or say 'type as is' to type your words exactly as you said them.".to_string())
                        }
                    }
                }

                ConversationState::ReadyToExecute | ConversationState::Completed => {
                    // Reset for new conversation
                    let mut ctx = self.context.lock().unwrap();
                    ctx.reset();
                    Ok(String::new())
                }
            }
        })
    }

    fn name(&self) -> &str {
        "conversation"
    }

    fn supports_mode(&self, mode: &VoiceMode) -> bool {
        matches!(
            mode,
            VoiceMode::Assistant { .. } | VoiceMode::Command | VoiceMode::Code { .. }
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Set the transcription function for the processor
impl ConversationProcessor {
    #[allow(dead_code)]
    pub fn set_transcription_fn<F>(&mut self, f: F)
    where
        F: Fn(&Path) -> Result<String> + Send + Sync + 'static,
    {
        self.transcription_fn = Some(Arc::new(f));
    }

    /// Get the conversation context (for debugging/monitoring)
    #[allow(dead_code)]
    pub fn get_context(&self) -> Arc<Mutex<ConversationContext>> {
        Arc::clone(&self.context)
    }

    /// Reset the conversation
    #[allow(dead_code)]
    pub fn reset_conversation(&self) {
        let mut ctx = self.context.lock().unwrap();
        ctx.reset();
    }

    /// Process text directly (for realtime mode)
    pub fn process_text(&self, text: &str, mode: &VoiceMode, _debug: bool) -> Result<String> {
        eprintln!(
            "[DEBUG ConversationProcessor::process_text] Input: {}",
            text
        );

        if text.trim().is_empty() {
            return Ok(String::new());
        }

        // Use tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()?;

        rt.block_on(async {
            // Update context state (minimized lock scope)
            {
                let mut ctx = self.context.lock().unwrap();
                ctx.add_user_message(text.to_string());
                ctx.state = ConversationState::Processing;
            }

            // Get context snapshot for analysis (separate lock scope)
            let ctx_snapshot = self.context.lock().unwrap().clone();

            // Analyze the text (no lock held during await)
            let analysis = self
                .analyze_transcription(text, &ctx_snapshot, mode)
                .await?;

            // Check if clarification is needed
            let (needs_clarification, can_clarify) = {
                let ctx = self.context.lock().unwrap();
                (analysis.needs_clarification, ctx.can_ask_clarification())
            };

            if needs_clarification && can_clarify {
                if let Some(question) = analysis.clarification_question {
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_assistant_message(question.clone());
                        ctx.state = ConversationState::WaitingForClarification;
                        ctx.clarification_count += 1;
                    }

                    // Speak the question (no lock held)
                    self.handle_clarification(&question).await?;

                    // Return formatted output for display
                    Ok(format!("🔊 {}", question))
                } else {
                    Ok(String::new())
                }
            } else {
                {
                    let mut ctx = self.context.lock().unwrap();
                    ctx.state = ConversationState::ReadyToExecute;
                }

                if let Some(response) = analysis.response {
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_assistant_message(response.clone());
                    }

                    // Check if this is a TTS response (starts with 🔊)
                    if response.starts_with("🔊 ") {
                        let speak_text = response.trim_start_matches("🔊 ").trim();

                        // Speak via TTS in background
                        if let Some(tts) = &self.tts_client {
                            eprintln!("🔊 Speaking final answer via ElevenLabs...");
                            let tts_clone = Arc::clone(tts);
                            let speak_text_owned = speak_text.to_string();
                            let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

                            // Spawn TTS in background thread
                            *is_speaking_clone.lock().unwrap() = true;
                            eprintln!("[DEBUG process_text] Set is_tts_speaking = true");
                            let interrupt_clone = Arc::clone(&self.tts_interrupt);
                            std::thread::spawn(move || {
                                eprintln!("[DEBUG process_text Thread] Starting TTS playback");
                                let rt = tokio::runtime::Runtime::new().unwrap();
                                if let Err(e) = rt.block_on(tts_clone.speak_and_play_interruptible(
                                    &speak_text_owned,
                                    Some(interrupt_clone),
                                )) {
                                    eprintln!("❌ TTS Error: {}", e);
                                }
                                *is_speaking_clone.lock().unwrap() = false;
                                eprintln!(
                                    "[DEBUG process_text Thread] Set is_tts_speaking = false"
                                );
                                eprintln!("✅ Answer spoken");
                            });
                        }

                        // Return the formatted text for display (don't type it)
                        Ok(response)
                    } else {
                        // Regular text response - return for typing
                        Ok(response)
                    }
                } else {
                    // LLM didn't use tools properly - return error message
                    eprintln!("[WARNING] No response from LLM in process_text, returning error");
                    Ok("🔊 I didn't understand that request. Could you rephrase it?".to_string())
                }
            }
        })
    }
}
