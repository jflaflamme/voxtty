// Assistant processor with pluggable backends
use crate::processors::{AudioProcessor, ProcessContext, VoiceMode};
use anyhow::Result;
use std::any::Any;
use std::path::Path;

/// Assistant backend trait
///
/// Different backends (Speaches, MCP) implement this trait
pub trait AssistantBackend: Send + Sync {
    /// Process audio with LLM assistance
    fn process_with_llm(&self, audio_path: &Path, mode: &VoiceMode, debug: bool) -> Result<String>;

    /// Process text directly with LLM (for realtime mode where transcription is already done)
    fn process_text_with_llm(&self, text: &str, mode: &VoiceMode, debug: bool) -> Result<String>;

    /// Backend name
    fn name(&self) -> &str;
}

/// Assistant processor that delegates to backends
pub struct AssistantProcessor {
    backend: Box<dyn AssistantBackend>,
}

impl AssistantProcessor {
    pub fn new(backend: Box<dyn AssistantBackend>) -> Self {
        Self { backend }
    }

    /// Process text directly with LLM (for realtime mode)
    pub fn process_text(&self, text: &str, mode: &VoiceMode, debug: bool) -> Result<String> {
        self.backend.process_text_with_llm(text, mode, debug)
    }
}

impl AudioProcessor for AssistantProcessor {
    fn process(&self, audio_path: &Path, context: &ProcessContext) -> Result<String> {
        if context.debug {
            println!(
                "[DEBUG] AssistantProcessor: Using backend '{}'",
                self.backend.name()
            );
        }

        self.backend
            .process_with_llm(audio_path, &context.mode, context.debug)
    }

    fn name(&self) -> &str {
        "AssistantProcessor"
    }

    fn supports_mode(&self, mode: &VoiceMode) -> bool {
        matches!(
            mode,
            VoiceMode::Assistant { .. } | VoiceMode::Code { .. } | VoiceMode::Command
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// Speaches Voice Chat Backend
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpeachesAssistantConfig {
    pub base_url: String,
    pub api_key: String,
    pub transcription_url: String,
    pub transcription_api_key: String, // Separate API key for transcription (OpenAI Whisper)
    pub transcription_model: String,
    pub llm_model: String,
    pub system_prompt: String,
    pub code_system_prompt: String,
}

pub struct SpeachesAssistantBackend {
    config: SpeachesAssistantConfig,
}

impl SpeachesAssistantBackend {
    pub fn new(config: SpeachesAssistantConfig) -> Self {
        Self { config }
    }

    fn get_system_prompt(&self, mode: &VoiceMode) -> String {
        match mode {
            VoiceMode::Code { .. } => self.config.code_system_prompt.clone(),
            VoiceMode::Command => include_str!("../prompts/command.md").to_string(),
            _ => self.config.system_prompt.clone(),
        }
    }

    fn transcribe_audio(&self, audio_path: &Path, debug: bool) -> Result<String> {
        if debug {
            println!(
                "[DEBUG] Transcribing audio with model: {}",
                self.config.transcription_model
            );
            println!(
                "[DEBUG] Transcription URL: {}",
                self.config.transcription_url
            );
        }

        let client = reqwest::blocking::Client::new();
        let file = std::fs::read(audio_path)?;

        // Add a prompt to help Whisper recognize technical terms correctly
        // This helps disambiguate similar-sounding words like "command" vs "comment"
        let technical_prompt = "Voice commands for system administration and terminal operations. \
            Technical terms: command mode, terminal mode, console mode, sysadmin, dictation mode, \
            assistant mode, code mode, shell commands.";

        let form = reqwest::blocking::multipart::Form::new()
            .text("model", self.config.transcription_model.clone())
            .text("prompt", technical_prompt)
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(file).file_name(
                    audio_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                ),
            );

        let mut request = client.post(&self.config.transcription_url).multipart(form);

        // Use transcription_api_key if set, otherwise fall back to main api_key
        let transcription_key = if !self.config.transcription_api_key.is_empty() {
            &self.config.transcription_api_key
        } else {
            &self.config.api_key
        };

        if !transcription_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", transcription_key));
        }

        let response = request.send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unable to read error response".to_string());
            anyhow::bail!("Transcription API error: {} - {}", status, error_text);
        }

        #[derive(serde::Deserialize)]
        struct TranscriptionResponse {
            text: String,
        }

        let result: TranscriptionResponse = response.json()?;
        Ok(result.text)
    }
}

impl SpeachesAssistantBackend {
    /// Send text to LLM and get response
    fn call_llm(&self, text: &str, mode: &VoiceMode, debug: bool) -> Result<String> {
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/chat/completions", self.config.base_url);

        let system_prompt = self.get_system_prompt(mode);

        let mut body = serde_json::json!({
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "temperature": 0.3
        });

        // Add tool definitions
        // 1. "process_command" - always available in Command mode
        // 2. "speak" - available in Assistant and Command mode
        let mut tools = serde_json::json!([]);
        let mut has_tools = false;

        if matches!(mode, VoiceMode::Command) {
            let command_tool = serde_json::json!({
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
            });
            tools.as_array_mut().unwrap().push(command_tool);
            has_tools = true;
        }

        if matches!(mode, VoiceMode::Assistant { .. } | VoiceMode::Command) {
            let speak_tool = serde_json::json!({
                "type": "function",
                "function": {
                    "name": "speak",
                    "description": "Speak a response back to the user via TTS. Use this for clarifications, rejections, answers to questions, or confirmations.",
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
            });
            tools.as_array_mut().unwrap().push(speak_tool);
            has_tools = true;

            // Add switch_mode tool
            let switch_mode_tool = serde_json::json!({
                "type": "function",
                "function": {
                    "name": "switch_mode",
                    "description": "Switch voice input mode. Use when user requests to change mode (e.g., 'switch to dictation', 'code mode', 'assistant mode').",
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
                                "description": "Brief confirmation message to speak (e.g., 'Switching to code mode')"
                            }
                        },
                        "required": ["mode", "confirmation"]
                    }
                }
            });
            tools.as_array_mut().unwrap().push(switch_mode_tool);
        }

        if has_tools {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("tools".to_string(), tools);
                // In Command mode, force tool usage to prevent plain text responses
                // In Assistant mode, allow LLM to decide (for dictation vs speak)
                let tool_choice = if matches!(mode, VoiceMode::Command) {
                    serde_json::json!("required")
                } else {
                    serde_json::json!("auto")
                };
                obj.insert("tool_choice".to_string(), tool_choice);
            }
        }

        if debug {
            println!("[DEBUG] Sending request to Chat Completion API");
            println!("[DEBUG] URL: {}", url);
            println!("[DEBUG] Model: {}", self.config.llm_model);
            println!("[DEBUG] API Key set: {}", !self.config.api_key.is_empty());
        }

        // Send request with Authorization header if API key is set
        let mut request = client.post(&url).json(&body);

        if !self.config.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response = request.send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unable to read error response".to_string());
            anyhow::bail!("Chat Completion API error: {} - {}", status, error_text);
        }

        #[derive(serde::Deserialize, Debug)]
        struct FunctionCall {
            name: String,
            arguments: String,
        }

        #[derive(serde::Deserialize, Debug)]
        struct ToolCall {
            function: FunctionCall,
        }

        #[derive(serde::Deserialize, Debug)]
        struct Message {
            content: Option<String>,
            tool_calls: Option<Vec<ToolCall>>,
        }

        #[derive(serde::Deserialize, Debug)]
        struct Choice {
            message: Message,
        }

        #[derive(serde::Deserialize, Debug)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        let result: ChatResponse = response.json()?;

        if result.choices.is_empty() {
            anyhow::bail!("No response from LLM");
        }

        let message = &result.choices[0].message;

        // Check for tool calls first
        if let Some(tool_calls) = &message.tool_calls {
            if let Some(call) = tool_calls.first() {
                if debug {
                    println!(
                        "[DEBUG] Received tool call: {} args: {}",
                        call.function.name, call.function.arguments
                    );
                }

                // Wrap the tool call in our internal format so main.rs knows what to do
                // Format: {"_voxtty_tool": "name", "args": {...}}
                let args_value: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or(serde_json::Value::Null);

                let wrapper = serde_json::json!({
                    "_voxtty_tool": call.function.name,
                    "args": args_value
                });

                return Ok(wrapper.to_string());
            }
        }

        // Fallback to content
        Ok(message.content.clone().unwrap_or_default())
    }
}

impl AssistantBackend for SpeachesAssistantBackend {
    fn process_with_llm(&self, audio_path: &Path, mode: &VoiceMode, debug: bool) -> Result<String> {
        if debug {
            println!(
                "[DEBUG] SpeachesAssistantBackend: Processing with mode {:?}",
                mode
            );
        }

        // Step 1: Transcribe the audio first
        let transcription = self.transcribe_audio(audio_path, debug)?;

        if debug {
            println!("[DEBUG] Transcription: {}", transcription);
        }

        // Step 2: Send transcription to LLM
        self.call_llm(&transcription, mode, debug)
    }

    fn process_text_with_llm(&self, text: &str, mode: &VoiceMode, debug: bool) -> Result<String> {
        if debug {
            println!(
                "[DEBUG] SpeachesAssistantBackend: Processing text with mode {:?}",
                mode
            );
        }

        // Send text directly to LLM (transcription already done by realtime)
        self.call_llm(text, mode, debug)
    }

    fn name(&self) -> &str {
        "ChatCompletionAPI"
    }
}

// ============================================================================
// MCP Backend (Future Implementation)
// ============================================================================

#[derive(Debug, Clone)]
pub struct MCPAssistantConfig {
    #[allow(dead_code)]
    pub server_url: String,
    // Add MCP-specific config
}

pub struct MCPAssistantBackend {
    #[allow(dead_code)]
    config: MCPAssistantConfig,
}

impl MCPAssistantBackend {
    #[allow(dead_code)]
    pub fn new(config: MCPAssistantConfig) -> Self {
        Self { config }
    }
}

impl AssistantBackend for MCPAssistantBackend {
    fn process_with_llm(
        &self,
        _audio_path: &Path,
        _mode: &VoiceMode,
        _debug: bool,
    ) -> Result<String> {
        // TODO: Implement MCP integration
        anyhow::bail!("MCP backend not yet implemented")
    }

    fn process_text_with_llm(
        &self,
        _text: &str,
        _mode: &VoiceMode,
        _debug: bool,
    ) -> Result<String> {
        // TODO: Implement MCP integration
        anyhow::bail!("MCP backend not yet implemented")
    }

    fn name(&self) -> &str {
        "MCP"
    }
}
