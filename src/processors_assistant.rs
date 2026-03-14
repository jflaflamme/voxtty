// Assistant processor with pluggable backends
use crate::mcp_tools::McpManager;
use crate::processors::{AudioProcessor, ProcessContext, VoiceMode};
use anyhow::Result;
use std::any::Any;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Built-in tool names that are handled by main.rs, not by MCP
const BUILTIN_TOOLS: &[&str] = &["process_command", "speak", "switch_mode"];

/// Maximum number of MCP tool call iterations per conversation turn
const MAX_TOOL_ITERATIONS: usize = 5;

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
    pub mcp_tools: Vec<serde_json::Value>,
}

pub struct SpeachesAssistantBackend {
    config: SpeachesAssistantConfig,
    mcp_manager: Option<Arc<Mutex<McpManager>>>,
}

impl SpeachesAssistantBackend {
    pub fn new(config: SpeachesAssistantConfig) -> Self {
        Self {
            config,
            mcp_manager: None,
        }
    }

    pub fn with_mcp_manager(mut self, manager: Arc<Mutex<McpManager>>) -> Self {
        self.mcp_manager = Some(manager);
        self
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

#[derive(serde::Deserialize, Debug)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(serde::Deserialize, Debug)]
struct ToolCall {
    id: Option<String>,
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

impl SpeachesAssistantBackend {
    /// Build the tools array for the LLM request
    fn build_tools(&self, mode: &VoiceMode) -> (serde_json::Value, bool) {
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

        // Append MCP tools
        for mcp_tool in &self.config.mcp_tools {
            tools.as_array_mut().unwrap().push(mcp_tool.clone());
            has_tools = true;
        }

        (tools, has_tools)
    }

    /// Send a chat completion request and parse the response
    fn send_chat_request(
        &self,
        body: &serde_json::Value,
        debug: bool,
    ) -> Result<ChatResponse> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let url = format!("{}/chat/completions", self.config.base_url);

        if debug {
            println!("[DEBUG] Sending request to Chat Completion API");
            println!("[DEBUG] URL: {}", url);
            println!("[DEBUG] Model: {}", self.config.llm_model);
            println!("[DEBUG] API Key set: {}", !self.config.api_key.is_empty());
        }

        let mut request = client.post(&url).json(body);

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

        let result: ChatResponse = response.json()?;

        if result.choices.is_empty() {
            anyhow::bail!("No response from LLM");
        }

        Ok(result)
    }

    /// Send text to LLM and get response, with MCP tool call loop
    fn call_llm(&self, text: &str, mode: &VoiceMode, debug: bool) -> Result<String> {
        let mut system_prompt = self.get_system_prompt(mode);

        // Inject current date/time so the LLM always knows
        let now = chrono::Local::now();
        system_prompt.push_str(&format!(
            "\n\n## CURRENT CONTEXT\n\n- **Date**: {}\n- **Time**: {}\n",
            now.format("%A, %B %d, %Y"),
            now.format("%I:%M %p")
        ));

        // Append MCP tool descriptions so the LLM knows what external tools are available
        if !self.config.mcp_tools.is_empty() {
            system_prompt.push_str("\n\n## EXTERNAL TOOLS (MCP)\n\n");
            system_prompt.push_str(
                "You also have access to external tools provided by MCP servers. \
                 Use these tools when the user's request matches their capabilities. \
                 After calling an external tool, use `speak` to tell the user the result.\n\n",
            );
            for tool in &self.config.mcp_tools {
                if let (Some(name), Some(desc)) = (
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str()),
                    tool.get("function")
                        .and_then(|f| f.get("description"))
                        .and_then(|d| d.as_str()),
                ) {
                    system_prompt.push_str(&format!("- **`{}`**: {}\n", name, desc));
                }
            }
        }

        let mut messages = vec![
            serde_json::json!({
                "role": "system",
                "content": system_prompt
            }),
            serde_json::json!({
                "role": "user",
                "content": text
            }),
        ];

        let (tools, has_tools) = self.build_tools(mode);

        for iteration in 0..=MAX_TOOL_ITERATIONS {
            let mut body = serde_json::json!({
                "model": self.config.llm_model,
                "messages": messages,
                "temperature": 0.3
            });

            if has_tools {
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tools".to_string(), tools.clone());
                    let tool_choice = if matches!(mode, VoiceMode::Command) && iteration == 0 {
                        serde_json::json!("required")
                    } else {
                        serde_json::json!("auto")
                    };
                    obj.insert("tool_choice".to_string(), tool_choice);
                }
            }

            let result = self.send_chat_request(&body, debug)?;
            let message = &result.choices[0].message;

            // Check for tool calls
            if let Some(tool_calls) = &message.tool_calls {
                if let Some(call) = tool_calls.first() {
                    let tool_name = &call.function.name;
                    let tool_id = call.id.as_deref().unwrap_or("call_0");

                    if debug {
                        println!(
                            "[DEBUG] Received tool call: {} args: {} (iteration {})",
                            tool_name, call.function.arguments, iteration
                        );
                    }

                    let args_value: serde_json::Value =
                        serde_json::from_str(&call.function.arguments)
                            .unwrap_or(serde_json::Value::Null);

                    // If it's a built-in tool, return it for main.rs to handle
                    if BUILTIN_TOOLS.contains(&tool_name.as_str()) {
                        let wrapper = serde_json::json!({
                            "_voxtty_tool": tool_name,
                            "args": args_value
                        });
                        return Ok(wrapper.to_string());
                    }

                    // It's an MCP tool — execute it and loop back
                    if let Some(ref mcp_mgr) = self.mcp_manager {
                        if iteration >= MAX_TOOL_ITERATIONS {
                            if debug {
                                println!(
                                    "[DEBUG] Max MCP tool iterations ({}) reached, returning last content",
                                    MAX_TOOL_ITERATIONS
                                );
                            }
                            return Ok(message.content.clone().unwrap_or_default());
                        }

                        let tool_result = match mcp_mgr.lock().unwrap().call_tool(tool_name, args_value.clone()) {
                            Ok(result) => result,
                            Err(e) => format!("Error calling tool '{}': {}", tool_name, e),
                        };

                        if debug {
                            println!(
                                "[DEBUG] MCP tool '{}' result: {}",
                                tool_name,
                                &tool_result[..tool_result.len().min(200)]
                            );
                        }

                        // Add assistant message with tool call
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": serde_json::Value::Null,
                            "tool_calls": [{
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": call.function.arguments
                                }
                            }]
                        }));

                        // Add tool result message
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result
                        }));

                        continue; // Loop back to LLM with tool result
                    }

                    // MCP manager not available — return tool call as-is
                    let wrapper = serde_json::json!({
                        "_voxtty_tool": tool_name,
                        "args": args_value
                    });
                    return Ok(wrapper.to_string());
                }
            }

            // No tool call — return content
            return Ok(message.content.clone().unwrap_or_default());
        }

        anyhow::bail!("Exceeded maximum tool call iterations")
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
